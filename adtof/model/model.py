from copy import copy
from dataclasses import dataclass
import datetime
import gc
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import sklearn.model_selection
from tensorflow.compat.v1.profiler import Profiler

from adtof import config
from adtof.io.midiProxy import PrettyMidiWrapper
from adtof.io import mir
from adtof.io.textReader import TextReader
from adtof.model.dataLoader import DataLoader
from adtof.model.peakPicking import PeakPicking, TatumPeakPicking
from adtof.model import eval
from adtof.model.layers.tatumPooling import TatumPooling
from adtof.model.layers.context import _add_context
from adtof.model.layers.positionalEncoding import positional_encoding
from adtof.model.track import Track
from adtof.config import update
from adtof.model.hyperparameters import models


class Model(object):
    @staticmethod
    def modelFactory(modelName="crnn-ADTOF", scenario="all", fold=0, pre_trained_path=None, searchSpaceIteration=None, **kwargs):
        """
        Instantiate and return a model with its hyperparameters.
        """
        # Get parameters from the model name
        hparams = update(kwargs, models[modelName])

        # Get parameters of the grid search iteration
        if searchSpaceIteration is not None:
            parameterGrid = sklearn.model_selection.ParameterGrid(hparams["searchSpace"])
            hparams.update(parameterGrid[searchSpaceIteration])
            print(f"searchSpaceIteration: {searchSpaceIteration}/{len(parameterGrid)-1}")

        # Build the model
        return (
            Model(
                "_".join([modelName, scenario, str(fold) + ("." + str(searchSpaceIteration) if searchSpaceIteration is not None else "")]),
                pre_trained_path=pre_trained_path,
                **hparams,
            ),
            hparams,
        )

    def __init__(self, name, pre_trained_path=None, **kwargs):
        """
        Encapsulate a TF model which knows how to create itself and ease the workflow.
        Should be created with the static function Model.modelFactory()
        """
        n_bins = mir.getDim(**kwargs)
        self.model = self._createModel(n_bins=n_bins, output=len(kwargs["labels"]), **kwargs)
        self.name = name
        self.path = os.path.join(config.CHECKPOINT_DIR, name)
        # if model is already trained, load the weights and flag to not retrain
        load_path = os.path.join(config.CHECKPOINT_DIR, pre_trained_path) if pre_trained_path else self.path
        if os.path.exists(load_path + ".index"):  # TODO: This check is not case sensitive, but macOS/Linux are
            logging.info("Loading model weights %s", load_path)
            self.model.load_weights(load_path)
            self.weightLoadedFlag = True
        else:
            self.weightLoadedFlag = False

    def getModelStatistics(self, trainingSequence=400, n_bins=168, n_channels=1, **kwargs):
        """
        Return a dictionnary of the model statistics to identify scaling wrt to:
        number of parameters, number of trainable parameters, number of layers, flops.
        """
        forward_pass = tf.function(self.model.call, input_signature=[tf.TensorSpec(shape=(1, trainingSequence, n_bins, n_channels))])
        profiler = Profiler(graph=forward_pass.get_concrete_function().graph)
        pro = profiler.profile_graph(options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
        return {
            "Model Parameters": self.model.count_params(),
            "Trainable Parameters": np.sum([np.prod(v.get_shape().as_list()) for v in self.model.trainable_variables]),
            "Flops": pro.total_float_ops // 2,
        }

    def _getCRNNFunctional(
        self,
        context=25,
        n_bins=168,
        output=5,
        trainingSequence=400,
        n_channels=1,
        conv_Filter=[32, 64],
        GRU_units=[60, 60, 60],
        tatumSubdivision=None,
        samePadding=False,
        lambdaRegularizer=None,
        output_layers=False,
        **kwargs,
    ):
        """
        Get the convolutional Recurrent Neural Network.
        """
        # ----- CNN part
        # Note: It is not possible to use only the sequential API because some layers have two inputs, which is only supported by the functional API
        # Input: (batchSize, trainingSequence, n_bins, n_channels)
        inputTensor = tf.keras.Input(shape=(None, n_bins, n_channels), name="x")
        padding = "same" if samePadding else "valid"
        l = lambdaRegularizer
        CNNLayers = []
        for i, filter in enumerate(conv_Filter):
            CNNLayers += [
                tf.keras.layers.Conv2D(filter, (3, 3), activation="relu", padding=padding, kernel_regularizer=tf.keras.regularizers.L2(l) if l else None, name="conv1" + str(i)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filter, (3, 3), activation="relu", padding=padding, kernel_regularizer=tf.keras.regularizers.L2(l) if l else None, name="conv2" + str(i)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(1, 3), strides=(1, 3), padding=padding),
                tf.keras.layers.Dropout(0.3),
            ]
        CNN = tf.keras.models.Sequential(CNNLayers)

        if n_channels > 1:
            x = [CNN(tf.keras.layers.Reshape((-1, n_bins, 1))(channel)) for channel in tf.unstack(inputTensor, axis=-1)]  # Split the input tensor into the different channels
            x = [tf.keras.layers.Reshape((-1, channel.shape[2] * channel.shape[3]))(channel) for channel in x]  # Flatten the output of the CNN to each time step
            x = tf.concat(x, -1)  # Concatenate the output of the CNN for each channel
        else:
            x = CNN(inputTensor)  # Apply the CNN to the mono signal
            x = tf.keras.layers.Reshape((-1, x.shape[2] * x.shape[3]))(x)  # Flatten the output of the CNN to each time step
        latentFeatures = x

        # ----- context part
        # TODO how to handle context and tatum synchronicity? -> synchronicity doesn't seem to improve the results anyway
        if context:
            cnnTimeReceptionField = len(conv_Filter) * (2 * 2) + 1  # Reception field of each output of the CNN
            contextFrames = context - cnnTimeReceptionField + 1  # How many frames have to be appened to increase the receptive field for the RNN
            # timeDimOutput = (trainingSequence - 2 * 2) - 2 * 2
            # featureDimOutput = ((n_bins - 2 * 2) // 3 - 2 * 2) // 3 * conv_Filter[-1]
            if contextFrames > 1:
                x = tf.keras.layers.Lambda(_add_context, arguments={"context_frames": contextFrames})(x)
            elif contextFrames <= 0:
                raise ValueError("The context is smaller than the receptive field of the CNN")

        # ----- Tatum synchronicity
        if tatumSubdivision is not None:
            tatumsBoundariesFrame = tf.keras.Input(shape=(None, 2), dtype=tf.int32, name="tatumsBoundariesFrame")  # shape=(trainingSequence, 2)
            inputTensor = (inputTensor, tatumsBoundariesFrame)
            x = TatumPooling()(x, tatumsBoundariesFrame)

        # ----- RNN part
        for units in GRU_units:
            x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units, stateful=False, return_sequences=True, kernel_regularizer=tf.keras.regularizers.L2(l) if l else None))(x)

        # TODO add dropout before the dense layer?
        # x = tf.keras.layers.Dropout(0.3)(x)
        outputTensor = tf.keras.layers.Dense(output, activation="sigmoid", name="denseOutput")(x)

        if output_layers:  # Save output of the CNN for debugging
            tfModel = tf.keras.Model(inputTensor, [outputTensor, latentFeatures])
        else:
            tfModel = tf.keras.Model(inputTensor, outputTensor)

        # tfModel.summary()
        return tfModel

    def _getTransformerFunctional(
        self,
        context=25,
        n_bins=168,
        output=5,
        trainingSequence=400,
        n_channels=1,
        conv_Filter=[64, 64, 32, 32],
        tatumSubdivision=None,
        num_attention_heads=5,
        num_attention_layers=5,
        pos_encoding="std",
        samePadding=False,
        output_layers=False,
        positional_encoding_weight=None,
        **kwargs,
    ):
        """
        Get the convolutional Transformer Neural Network.
        It is not possible to use the sequential API because some layers have two inputs, which is only supported by the functional API
        """

        # Input: (batchSize, trainingSequence, n_bins, n_channels)
        # batch size: is ommited
        # trainingSequence: number of time steps, set to None because tracks have different length
        # n_bins: number of frequency band in the input vector, for each time-step
        # n_channel: set to 1 if mono, or 2 if stereo
        inputTensor = tf.keras.Input(shape=(None, n_bins, n_channels), name="x")

        # ----- Encoder: CNN part
        padding = "same" if samePadding else "valid"
        x = tf.keras.layers.Conv2D(conv_Filter[0], (3, 3), activation="relu", strides=(1, 1), padding=padding, name="conv11")(inputTensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(conv_Filter[1], (3, 3), activation="relu", strides=(1, 1), padding=padding, name="conv12")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(1, 3), strides=(1, 3), padding=padding)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Conv2D(conv_Filter[2], (3, 3), activation="relu", strides=(1, 1), padding=padding, name="conv21")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(conv_Filter[3], (3, 3), activation="relu", strides=(1, 1), padding=padding, name="conv22")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(1, 3), strides=(1, 3), padding=padding)(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        # ----- context part: Stack time steps to increase the receptive field
        cnnTimeReceptionField = 2 * 2 + 2 * 2 + 1  # Reception field of each output of the CNN
        contextFrames = context - cnnTimeReceptionField + 1  # How many frames have to be appened to increase the receptive field for the RNN
        # timeDimOutput = (trainingSequence - 2 * 2) - 2 * 2
        # featureDimOutput = ((n_bins - 2 * 2) // 3 - 2 * 2) // 3 * conv_Filter[3]
        featureDimOutput = x.shape[2] * x.shape[3]
        x = tf.keras.layers.Reshape((-1, featureDimOutput))(x)  # Flatten the output of the CNN to each time step
        if contextFrames > 1:
            x = tf.keras.layers.Lambda(_add_context, arguments={"context_frames": contextFrames})(x)
        latentFeatures = x

        # ----- Tatum synchronicity
        # TODO how to handle context and tatum synchronicity? -> synchronicity doesn't seem to improve the results anyway
        if tatumSubdivision is not None:
            tatumsBoundariesFrame = tf.keras.Input(shape=(None, 2), dtype=tf.int32, name="tatumsBoundariesFrame")  # shape=(trainingSequence, 2)
            inputTensor = (inputTensor, tatumsBoundariesFrame)
            x = TatumPooling()(x, tatumsBoundariesFrame)

        # ----- Append positional encoding
        # TODO: How can we parametrize the number of latent features in the positional encoding?
        seq_len = tf.shape(x)[1]
        pos_encoding = positional_encoding(trainingSequence, featureDimOutput, encoding=pos_encoding)
        # TODO: likely not needed because the encoder is not normalized and the latent space is warm
        if positional_encoding_weight == "reduced":  # Apply reduction of the PE weight from TF tranformer demo.
            x *= tf.math.sqrt(tf.cast(featureDimOutput, tf.float32))
        if positional_encoding_weight == "increased":
            x /= 2
        x += pos_encoding[:, :seq_len, :]
        pooledFeatures = x

        # ----- Decoder: self attention part
        # I = 2-8 heads
        # L = 8-7 stack of self-attention layers
        # N = 256 tatums in the training sequence
        # DF = 96-120 latent features for the positional encoding
        # DFF = 4 * DF dimension of the feed forward network
        # Dk = DF / I
        attention_scores = {}
        for i in range(num_attention_layers):  # Stack of self-attention layers
            normed = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
                x
            )  # TODO: Is this x the one before the loop, or the one at the end of the current loop? TODO the batch normalization has to be in between each head?
            out, attention_score = tf.keras.layers.MultiHeadAttention(num_heads=num_attention_heads, key_dim=featureDimOutput // num_attention_heads, dropout=0.1)(
                query=normed, value=normed, key=normed, return_attention_scores=True
            )  # Self attention = same input as query, key and value
            attention_scores[i] = attention_score
            x = x + out  # Residual connection
            # point wise feed forward network (a simple Dense layer applied to a 2D input implicitly behaves like a TimeDistributed layer)
            x = tf.keras.layers.Dense(featureDimOutput * 4, activation="relu", name="pointwiseFeedforwardRELU" + str(i))(x)  # Shape `(batch_size, seq_len, dff)`.
            x = tf.keras.layers.Dense(featureDimOutput, name="pointwiseFeedforwardNone" + str(i))(x)  # Shape `(batch_size, seq_len, d_model)`.

        # ----- Output part
        outputTensor = tf.keras.layers.Dense(output, activation="sigmoid", name="denseOutput")(x)
        if output_layers:
            tfModel = tf.keras.Model(inputTensor, [outputTensor, attention_scores, latentFeatures, pooledFeatures])
        else:
            tfModel = tf.keras.Model(inputTensor, outputTensor)

        # tfModel.summary()
        return tfModel

    def _createModel(self, architecture="CRNN", learningRate=0.001 / 2, **kwargs):
        """
        Return a tf model with the given architecture
        """

        # TODO: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#optional_set_the_correct_initial_bias
        # TODO How to handle the bidirectional aggregation ? by default in tf.keras it's sum
        # Between each miniBatch the recurent units lose their state by default,
        # Prevent that if we feed the same track across multiple mini-batches
        if architecture == "CRNN":
            tfModel = self._getCRNNFunctional(**kwargs)
        elif architecture == "CNN-SelfAtt":
            tfModel = self._getTransformerFunctional(**kwargs)

        else:
            raise ValueError("%s not known", architecture)

        # Very interesting read on loss functions: https://gombru.github.io/2018/05/23/cross_entropy_loss/
        # How softmax cross entropy can be used in multilabel classification,
        # and how binary cross entropy work for multi label
        # learning_rate = CustomSchedule(320)
        tfModel.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learningRate),  # "adam",  #tf.keras.optimizers.RMSprop(learning_rate=0.001),
            loss=tf.keras.backend.binary_crossentropy,  # tf.nn.sigmoid_cross_entropy_with_logits,  #tf.keras.backend.binary_crossentropy,
            # metrics=[tf.keras.metrics.Precision(thresholds=0.25), tf.keras.metrics.Recall(thresholds=0.25)],  # PeakPicking(hitDistance=0.05)
            weighted_metrics=[],
        )
        return tfModel

    def fit(
        self,
        dataset_train,
        dataset_val,
        log_dir,
        steps_per_epoch,
        validation_steps,
        reduce_patience=3,
        stopping_patience=6,
        min_steps_per_epoch=1,
        min_validation_steps=0,
        classWeight=None,
        **kwargs,
    ):
        """
        Fits the model to the train dataset and validate on the val dataset to reduce LR on plateau and do an earlystopping.
        """
        logging.info("Training model %s", self.name)
        logging.info("class weight %s", str(classWeight))

        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=log_dir + self.name + datetime.datetime.now().strftime("%d-%m-%H:%M"), histogram_freq=1),  # , profile_batch="15, 30"
            tf.keras.callbacks.ModelCheckpoint(self.path, save_weights_only=True, save_best_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, verbose=1, patience=reduce_patience),
            tf.keras.callbacks.EarlyStopping(min_delta=0.0001, patience=stopping_patience, verbose=1, restore_best_weights=True),
            # tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: log_layer_activation(epoch, viz_example, model, activation_model, file_writer))
        ]
        self.model.fit(
            dataset_train,
            epochs=1000,  # Very high number of epoch to stop only with ealy stopping
            steps_per_epoch=max(steps_per_epoch, min_steps_per_epoch),
            callbacks=callbacks,
            validation_data=dataset_val,
            validation_steps=max(validation_steps, min_validation_steps),
            class_weight=classWeight,
        )

    def predict(self, track: Track, trainingSequence=400, limitInputSize=60000, context=9, samePadding=False, tatumSubdivision=None, architecture="CNN-SelfAtt", **kwargs):
        """
        Call model.predict if possible
        If the model is a CRNN and can't utilize the batch parallelisation, call directly the model

        limitInputSize:
            Specifies the maximum window size before a segmentation Fault might be raised. If the input is larger than this limit, it is split into overlapping windows of this size.
        TODO identify the error Computed output size would be negative: -2 [input_size: 0, effective_filter_size: 3, stride: 1]
        """
        # print("call predict with shape", str(x.shape))
        if trainingSequence == 1:
            raise DeprecationWarning()
            # return self.model.predict(x, **kwargs)

        # Define the window size for the computation.
        if architecture == "CNN-SelfAtt":  # If self-attention, the window size is the training sequence, with half overlap
            window = trainingSequence
            warmup = trainingSequence // 4
        else:  # If RNN, the window size is a large chunck (10min), with training sequence warmup
            window = limitInputSize
            warmup = trainingSequence
        step = window - 2 * warmup

        predictions = []
        # if x does not fit, call multiple times predict with smaller overlapping window
        windows = range(0, track.samplesCardinality - warmup - (0 if samePadding else context), step)
        for sampleIdx in windows:
            input = track.getSlice(sampleIdx, window, tatumSubdivision=tatumSubdivision, **kwargs)[0]
            prediction = self.model.predict_on_batch([v.reshape([1] + list(v.shape)) for v in input.values()])[0]
            if len(windows) == 1:  # If there is only one window, there is no warmup
                predictions.append(prediction)
            elif sampleIdx == 0:  # If it's the first window, there is no warm-up at the beginning, only at the end
                predictions.append(prediction[: window - warmup])
            elif sampleIdx == windows[-1]:  # If it's the last window, there is no warm-up at the end, only at the beginning
                predictions.append(prediction[warmup:])
            else:  # If it's not the first window, there is a warm-up at the beginning and at the end
                predictions.append(prediction[warmup:][:step])

        return np.concatenate(predictions)

    @staticmethod
    def predictEnsemble(models, x, aggregation=np.mean):
        """
        Aggregate predictions from an ensemble of models
        """
        return aggregation([model.predict(x) for model in models], axis=0)

    def predictFolder(self, inputFolder, outputFolder, writeMidi=True, **kwargs):
        """
        Run the model prediction followed by the peak picking procedure on all the tracks from the folder.
        Write a text file with the prediction in the output folder.
        If writeMidi=True, write also a midi file containing the predictions

        Parameters
        ----------
        inputFolder :
            Path to a folder containing music or one specific music
        outputFolder :
            Path to the location where to store the output
        writeMidi : bool, optional
           if a midi file for the prediction should be written as well, by default True
        """
        pp = PeakPicking()
        pp.setParameters(**kwargs)
        dl = DataLoader(inputFolder, crossValidation=False)
        # Create a generator yielding full tracks one by one
        tracks = dl.getGen(training=False, **kwargs)

        # Predict the file and write the output
        for track in tracks():
            try:
                if not os.path.exists(outputFolder):
                    os.makedirs(outputFolder)
                outputTrackPath = os.path.join(outputFolder, track.title + ".txt")
                if os.path.exists(outputTrackPath):
                    continue

                Y = self.predict(track, **kwargs)
                sparseResultIdx = pp.predict([Y], **kwargs)[0]  # peakPicking.peakPicking(Y,ppProcess=ppp, timeOffset=kwargs["labelOffset"] / kwargs["sampleRate"], **kwargs)

                # write text
                formatedOutput = [(time, pitch) for pitch, times in sparseResultIdx.items() for time in times]
                formatedOutput.sort(key=lambda x: x[0])
                TextReader().writteBeats(outputTrackPath, formatedOutput)

                #  write midi
                if writeMidi:
                    midi = PrettyMidiWrapper.fromDict(sparseResultIdx)
                    midi.write(os.path.join(outputFolder, track.title + ".mid"))

            except Exception as e:
                logging.error(str(e))

    def evaluate(self, dataset, peakThreshold=None, earlyStopping=None, datasetName="", modelName="", multiplePPThreshold=False, debug=False, tatumSubdivision=None, **kwargs):
        """
        Run model.predict on the dataset followed by madmom.peakpicking. Find the best threshold for the peak
        The dataset needs to return full tracks and not independant samples
        """
        # Infer/Estimate/Predict
        predictions = []
        Y = []
        LOSS = []
        NAME = []
        TATUMSTIME = []
        for i, track in enumerate(dataset):
            try:
                startTime = time.time()
                predictions.append(self.predict(track, tatumSubdivision=tatumSubdivision, **kwargs))
                Y.append(track.y)
                NAME.append(track.title)
                LOSS.append(self.model.compute_loss(y=tf.constant(track.yDense[:-1]), y_pred=tf.constant(predictions[-1])).numpy())  # TODO: should add sample Weight?
                if tatumSubdivision is not None:
                    TATUMSTIME.append(track.tatumsTime)
                logging.debug("track %s predicted in %s", i, time.time() - startTime)
                if earlyStopping is not None and i >= earlyStopping:
                    break
            except Exception as e:
                logging.error("track %s not predicted! Skipped because of %s", i, str(e))

        if len(predictions) == 0:
            raise Exception("No track predicted")

        # Peak picking for F-measure
        # The annotations are offset, so we don't correct for the estimation offset during evaluation. Only during infering
        newkwargs = copy(kwargs)
        newkwargs["labelOffset"] = 0
        pp = PeakPicking() if tatumSubdivision is None else TatumPeakPicking()
        if peakThreshold == None:
            if multiplePPThreshold:
                pp.fitIndependentLabel(predictions, Y, tatumsTime=TATUMSTIME, **newkwargs)
            else:
                pp.fit(predictions, Y, **newkwargs)
        else:
            pp.setParameters(peakThreshold, **newkwargs)
        score = pp.score(predictions, Y, tatumsTime=TATUMSTIME, **newkwargs)

        score["mean loss"] = np.mean(LOSS)
        score["sum loss"] = np.average(LOSS, weights=[len(y) for y in predictions])

        if debug:
            # Plot confusion matrices
            sparsePredictions = pp.predict(predictions, tatumsTime=TATUMSTIME, **newkwargs)
            path = "logs/confusion/" + modelName + "/" + datasetName
            config.checkPathExists(path)
            eval.plotPseudoConfusionMatrices(Y, sparsePredictions, saveFigure=path)

            # Plot results for different tolerance distance
            toleranceThresholds = np.arange(0.01, 0.1, 0.01)
            metric = "sum F all"
            results = [eval.runEvaluation(Y, sparsePredictions, distanceThreshold=t)[metric] for t in toleranceThresholds]
            score["sum F all varyingThreshold"] = results
            plt.figure()
            plt.plot(toleranceThresholds, results)
            plt.ylim(0, 1)
            plt.ylabel(metric)
            plt.xlabel("Threshold")
            plt.title(datasetName)
            path = "logs/varyingThreshold/" + modelName + "/" + datasetName + "-perf.pdf"
            config.checkPathExists(path)
            plt.savefig(path)

            # plot activation
            eval.plotActivation(predictions, Y, sparsePredictions, trackI=0, **newkwargs)
            plt.savefig("test.pdf")

            # Save midi file for debug listening
            tracksResult = {}
            for i, sparsePrediction in enumerate(sparsePredictions):
                path = os.path.join("logs/debug/", modelName, datasetName, NAME[i] + ".midi")
                config.checkPathExists(path)
                PrettyMidiWrapper.fromDict(sparsePrediction).write(path)
                tracksResult[NAME[i]] = eval.runEvaluation([Y[i]], [sparsePrediction])

            path = os.path.join("logs/debug/", modelName, datasetName, "results.csv")
            config.checkPathExists(path)
            pd.DataFrame.from_dict(tracksResult, orient="index").to_csv(path)

        # Clean (some?) memory leak from tf
        K.clear_session()
        gc.collect()

        return score, pp.getParameters()
