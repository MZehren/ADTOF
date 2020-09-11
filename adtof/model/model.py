import logging
import time
from collections import defaultdict
from os import stat
import os
import datetime

import madmom
from markdown.test_tools import Kwargs
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy.core.defchararray import mod
from tensorflow.keras import layers
from tensorflow.python.eager.monitoring import Sampler

from adtof import config
from adtof.model import eval
from adtof.model import peakPicking


class Model(object):
    """
    Richard Vogl model
    http://ifs.tuwien.ac.at/~vogl/
    """

    @staticmethod
    def modelFactory(fold=0):
        models = {
            "cnn-stride(1,3)-shuffledinput": {
                "labels": config.LABELS_5,
                "classWeights": config.WEIGHTS_5 / 2,
                "sampleRate": 100,
                "diff": True,
                "samplePerTrack": 20,
                "batchSize": 100,
                "context": 25,
                "labelOffset": 1,
                "labelRadiation": 1,
                "learningRate": 0.0001,
                "normalize": False,
                "model": "CNN",
                "fmin": 20,
                "fmax": 20000,
                "pad": False,
                "beat_targ": False,
                "tracksLimit": None,
            },
            # (
            #     "crnn-stride(1,3)",
            #     {
            #         "labels": config.LABELS_5,
            #         "classWeights": config.WEIGHTS_5 / 2,
            #         "sampleRate": 100,
            #         "diff": True,
            #         "samplePerTrack": 20,
            #         "batchSize": 100,
            #         "context": 25,  # in RNN, The context is used as time serie, using a bigger one is not increasing the total params
            #         "labelOffset": 1,
            #         "labelRadiation": 1,
            #         "learningRate": 0.0001,
            #         "normalize": False,
            #         "model": "CRNN",
            #         "fmin": 20,
            #         "fmax": 20000,
            #         "pad": False,
            #         "beat_targ": False,
            #     },
            # ),
        }

        for modelName, hparams in models.items():
            modelName += "_Fold" + str(fold)
            yield (Model(modelName, **hparams), hparams)

    def __init__(self, name, **kwargs):
        self.model = self._createModel(n_bins=168 if kwargs["diff"] else 84, output=len(kwargs["labels"]), **kwargs)
        self.name = name
        self.path = os.path.join(config.CHECKPOINT_DIR, name)
        # if model is already trained, load the weights
        if os.path.exists(self.path + ".index"):
            logging.info("Loading model weights %s", self.path)
            self.model.load_weights(self.path)
            self.weightLoadedFlag = True
        else:
            self.weightLoadedFlag = False

    def _getCNN(self, context, n_bins, output):
        """
        Parameters from Vogl

        import pickle
        file="/madmom-0.16.dev0/madmom/models/drums/2018/drums_cnn0_O8_S0.pkl"
        # file = "/vendors/madmom-0.16.dev0/madmom/models/drums/2018/drums_crnn1_O8_S0.pkl"
        with open(file, "rb") as f:
            u = pickle._Unpickler(f)
            u.encoding = "latin1"
            p = u.load()
            print(p)

        00:<madmom.ml.nn.layers.ConvolutionalLayer object at 0x7fd5b2f38ac8>        valid padding, stride 1
        01:<madmom.ml.nn.layers.BatchNormLayer object at 0x7fd5722e5d30>
        02:<madmom.ml.nn.layers.ConvolutionalLayer object at 0x7fd5722e5cf8>        valid padding, stride 1
        03:<madmom.ml.nn.layers.BatchNormLayer object at 0x7fd5722e5f28>
        04:<madmom.ml.nn.layers.MaxPoolLayer object at 0x7fd5b3eca978>              size (1,3), stride(1,3)
        05:<madmom.ml.nn.layers.ConvolutionalLayer object at 0x7fd5b3eca6d8>        valid padding, stride 1
        06:<madmom.ml.nn.layers.BatchNormLayer object at 0x7fd5b3eca438>
        07:<madmom.ml.nn.layers.ConvolutionalLayer object at 0x7fd5b3eca048>        valid padding, stride 1
        08:<madmom.ml.nn.layers.BatchNormLayer object at 0x7fd5b3ec3da0>
        09:<madmom.ml.nn.layers.MaxPoolLayer object at 0x7fd57228d208>              size (1,3), stride(1,3)
                                                                                    Shape is [batch, context, features, kernel] = [None, 17, 16, 64]
        10:<madmom.ml.nn.layers.StrideLayer object at 0x7fd57228d278> Re-arrange    (stride) the data in blocks of given size (17).
                                                                                    Shape is [batch, ?, features] = [None, 1, 17*16*64]
        11:<madmom.ml.nn.layers.FeedForwardLayer object at 0x7fd57228d2b0>
        12:<madmom.ml.nn.layers.BatchNormLayer object at 0x7fd57228d390>
        13:<madmom.ml.nn.layers.FeedForwardLayer object at 0x7fd57228d588>
        14:<madmom.ml.nn.layers.BatchNormLayer object at 0x7fd57228d6a0>
        15:<madmom.ml.nn.layers.FeedForwardLayer object at 0x7fd57228d860>

        TODO: Can we handle the left and right channel as input shape (context, n_bins, 2)?
        """
        tfModel = tf.keras.Sequential()
        tfModel.add(
            tf.keras.layers.Conv2D(
                32, (3, 3), input_shape=(context, n_bins, 1), activation="relu", strides=(1, 1), padding="valid", name="conv11"
            )
        )
        tfModel.add(tf.keras.layers.BatchNormalization())
        tfModel.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", strides=(1, 1), padding="valid", name="conv12"))
        tfModel.add(tf.keras.layers.BatchNormalization())
        tfModel.add(tf.keras.layers.MaxPool2D(pool_size=(1, 3), strides=(1, 3), padding="valid"))
        tfModel.add(tf.keras.layers.Dropout(0.3))
        tfModel.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", strides=(1, 1), padding="valid", name="conv21"))
        tfModel.add(tf.keras.layers.BatchNormalization())
        tfModel.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", strides=(1, 1), padding="valid", name="conv22"))
        tfModel.add(tf.keras.layers.BatchNormalization())
        tfModel.add(tf.keras.layers.MaxPool2D(pool_size=(1, 3), strides=(1, 3), padding="valid"))
        tfModel.add(tf.keras.layers.Dropout(0.3))
        tfModel.add(tf.keras.layers.Flatten())
        tfModel.add(tf.keras.layers.Dense(256, activation="relu", name="dense1"))
        tfModel.add(tf.keras.layers.BatchNormalization())
        tfModel.add(tf.keras.layers.Dense(256, activation="relu", name="dense2"))
        tfModel.add(tf.keras.layers.BatchNormalization())
        tfModel.add(tf.keras.layers.Dense(output, activation="sigmoid", name="denseOutput"))
        # tfModel.build()
        # tfModel.summary()
        return tfModel

    def _getCRNN(self, context, n_bins, output, batchSize):
        """
        00:<madmom.ml.nn.layers.ConvolutionalLayer object at 0x1031c1190>
        01:<madmom.ml.nn.layers.BatchNormLayer object at 0x14f6ff4f0>
        02:<madmom.ml.nn.layers.ConvolutionalLayer object at 0x14f6ff730>
        03:<madmom.ml.nn.layers.BatchNormLayer object at 0x14f6ff820>
        04:<madmom.ml.nn.layers.MaxPoolLayer object at 0x14f6ff9a0>
        05:<madmom.ml.nn.layers.ConvolutionalLayer object at 0x14f6ffa30>
        06:<madmom.ml.nn.layers.BatchNormLayer object at 0x14f6ffb50>
        07:<madmom.ml.nn.layers.ConvolutionalLayer object at 0x14f6ffd30>
        08:<madmom.ml.nn.layers.BatchNormLayer object at 0x14f6ffe20>
        09:<madmom.ml.nn.layers.MaxPoolLayer object at 0x14f707040>
        10:<madmom.ml.nn.layers.StrideLayer object at 0x14f7070d0> Re-arrange (stride) the data in blocks of given size (5).
        11:<madmom.ml.nn.layers.BidirectionalLayer object at 0x14f707100>
        12:<madmom.ml.nn.layers.BidirectionalLayer object at 0x14f707280>
        13:<madmom.ml.nn.layers.BidirectionalLayer object at 0x14f7104c0>
        14:<madmom.ml.nn.layers.FeedForwardLayer object at 0x14f710640>

        TODO: How to handle recurence granularity?
        https://www.tensorflow.org/guide/keras/rnn#cross-batch_statefulness
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN#masking_2
        https://adgefficiency.com/tf2-lstm-hidden/
        https://www.tensorflow.org/tutorials/structured_data/time_series
        TODO: How to handle bidirectionnality?
        """
        tfModel = tf.keras.Sequential()
        tfModel.add(
            tf.keras.layers.Conv2D(
                32,
                (3, 3),
                input_shape=(context, n_bins, 1),
                batch_input_shape=(batchSize, context, n_bins, 1),
                activation="relu",
                strides=(1, 1),
                padding="valid",
                name="conv11",
            )
        )
        tfModel.add(tf.keras.layers.BatchNormalization())
        tfModel.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", strides=(1, 1), padding="valid", name="conv12"))
        tfModel.add(tf.keras.layers.BatchNormalization())
        tfModel.add(tf.keras.layers.MaxPool2D(pool_size=(1, 3), strides=(1, 3), padding="valid"))
        tfModel.add(tf.keras.layers.Dropout(0.3))
        tfModel.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", strides=(1, 1), padding="valid", name="conv21"))
        tfModel.add(tf.keras.layers.BatchNormalization())
        tfModel.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", strides=(1, 1), padding="valid", name="conv22"))
        tfModel.add(tf.keras.layers.BatchNormalization())
        tfModel.add(tf.keras.layers.MaxPool2D(pool_size=(1, 3), strides=(1, 3), padding="valid"))
        tfModel.add(tf.keras.layers.Dropout(0.3))

        # we set the model as a sequential model, the recurrency is done inside the batch and not outside
        # tfModel.add(tf.keras.layers.Flatten())  replace the flatten by a reshape to [batchSize, timeSerieDim, featureDim]
        timeSerieDim = context - 4 * 2
        featureDim = ((n_bins - 2 * 2) // 3 - 2 * 2) // 3 * 64
        tfModel.add(tf.keras.layers.Reshape((1, -1)))  # timeSerieDim might change if the full track is provided
        tfModel.add(
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(60, stateful=True, return_sequences=True))
        )  # return the whole sequence for the next layers
        tfModel.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(60, stateful=True, return_sequences=True)))
        tfModel.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(60, stateful=True, return_sequences=False)))
        tfModel.add(tf.keras.layers.Dense(output, activation="sigmoid", name="denseOutput"))
        tfModel.build()
        tfModel.summary()
        return tfModel

    def _createModel(self, model="cnn", context=25, n_bins=168, output=5, learningRate=0.001 / 2, batchSize=100, **kwargs):
        """Return a tf model based 
        
        Keyword Arguments:
            context {int} -- [description] (default: {25})
            n_bins {int} -- [description] (default: {84})
            output {int} -- number of classes in the output (should be the events: 36, 40, 41, 46, 49) (default: {5})
            outputWeight {list} --  (default: {[]}) 
        
        Returns:
            [type] -- [description]
        """

        # TODO: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#optional_set_the_correct_initial_bias
        # TODO How to handle the bidirectional aggregation ? by default in tf.keras it's sum
        # Between each miniBatch the recurent units lose their state by default,
        # Prevent that if we feed the same track across multiple mini-batches
        if model == "CNN":
            tfModel = self._getCNN(context, n_bins, output)

        elif model == "CRNN":
            tfModel = self._getCRNN(context, n_bins, output, batchSize)

        else:
            raise ValueError("%s not known", model)

        # Very interesting read on loss functions: https://gombru.github.io/2018/05/23/cross_entropy_loss/
        # How softmax cross entropy can be used in multilabel classification,
        # and how binary cross entropy work for multi label
        tfModel.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),  # "adam",  #tf.keras.optimizers.RMSprop(learning_rate=0.001),
            loss=tf.keras.backend.binary_crossentropy,  # tf.nn.sigmoid_cross_entropy_with_logits,  #tf.keras.backend.binary_crossentropy,
            metrics=["Precision", "Recall"],  # PeakPicking(hitDistance=0.05)
        )
        return tfModel

    def fit(self, dataset_train, dataset_val, log_dir, steps_per_epoch, validation_steps, **kwargs):
        logging.info("Training model %s", self.name)

        callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir + self.name + datetime.datetime.now().strftime("%d-%m-%H:%M"),
                histogram_freq=0,
                write_graph=False,
                write_images=False,
            ),
            tf.keras.callbacks.ModelCheckpoint(self.path, save_weights_only=True,),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=30, verbose=1, restore_best_weights=True),
            # tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: log_layer_activation(epoch, viz_example, model, activation_model, file_writer))
        ]
        self.model.fit(
            dataset_train,
            epochs=1000,  # Very high number of epoch to stop only with ealy stopping
            initial_epoch=0,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            validation_data=dataset_val,
            validation_steps=validation_steps
            # class_weight=classWeight
        )

    def predict(self, dataset):
        """
        TODO
        """
        predictions = []

        gen = dataset()
        for i, x in enumerate(gen):
            startTime = time.time()
            predictions.append(self.model.predict(x))
            logging.debug("track %s predicted in %s", i, time.time() - startTime)
        return predictions

    def evaluate(self, dataset, peakThreshold=None, **kwargs):
        """
        Run model.predict on the dataset followed by madmom.peakpicking. Find the best threshold for the peak 
        """
        predictions = []
        Y = []
        gen = dataset()
        for i, (x, y) in enumerate(gen):
            startTime = time.time()
            predictions.append(self.model.predict(x))
            Y.append(y)
            logging.debug("track %s predicted in %s", i, time.time() - startTime)

        if peakThreshold == None:
            return peakPicking.fitPeakPicking(predictions, Y, **kwargs)
        else:
            return peakPicking.fitPeakPicking(predictions, Y, peakPickingSteps=[peakThreshold])

    def vizPredictions(self, dataset, labels=[35], batchSize=100, **kwargs):
        """
        Plot the input, output and target of the model
        """

        for x, y, _ in dataset:
            predictions = self.model.predict(x)
            import matplotlib.pyplot as plt

            f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.plot(predictions, alpha=0.8)
            ax1.set_ylabel("Prediction + Truth")

            for classIdx in range(len(labels)):
                idx = [i for i, v in enumerate(y[:, classIdx]) if v == 1]
                ax1.scatter(idx, [classIdx / 20 + 1 for _ in range(len(idx))])
            ax2.set_ylabel("Input")
            ax2.set_xlabel("Time step")
            ax2.matshow(tf.transpose(tf.reshape(x[:, 0], (batchSize, -1))), aspect="auto")
            ax1.legend(labels)
            plt.show()


# def log_layers(epoch, input, model, activation_model, file_writer):
#     log_layer_activation(epoch, input, model, activation_model, file_writer)
#     log_layer_weights(epoch, input, model, activation_model, file_writer)


# def log_layer_activation(epoch, input, model, activation_model, file_writer):
#     # Create a figure to contain the plot.
#     layer_names = ["activation 0 input"] + ["activation " + str(i + 1) + " " + layer.name for i, layer in enumerate(model.layers)]
#     layer_activations = [input] + activation_model.predict(input)
#     for iLayer, (layer_name, layer_activation) in enumerate(zip(layer_names, layer_activations)):  # Displays the feature maps
#         # Start next subplot.
#         figure = plt.figure(figsize=(20, 10))

#         if len(layer_activation.shape) == 4:
#             _, sizeW, sizeH, nKernels = layer_activation.shape
#             for iKernel in range(nKernels):
#                 plt.subplot(nKernels // 5 + 1, 5, iKernel + 1, title=str(iKernel))
#                 plt.xticks([])
#                 plt.yticks([])
#                 plt.grid(False)
#                 plt.imshow(layer_activation[0, :, :, iKernel], cmap="viridis")

#         # Save the plot to a PNG in memory.
#         buf = io.BytesIO()
#         plt.savefig(buf, format="png")
#         # Closing the figure prevents it from being displayed directly inside
#         # the notebook.
#         plt.close(figure)
#         buf.seek(0)
#         # Convert PNG buffer to TF image
#         image = tf.image.decode_png(buf.getvalue(), channels=4)
#         # Add the batch dimension
#         image = tf.expand_dims(image, 0)

#         with file_writer.as_default():
#             tf.summary.image(layer_name, image, step=epoch)


# def log_layer_weights(epoch, input, model, activation_model, file_writer):
#     # Create a figure to contain the plot.
#     layer_names = ["weights " + str(i + 1) + " " + layer.name for i, layer in enumerate(model.layers) if len(layer.get_weights())]
#     layer_weigths = [layer.get_weights()[0] for layer in model.layers if len(layer.get_weights())]
#     for iLayer, (layer_name, layer_weigth) in enumerate(zip(layer_names, layer_weigths)):  # Displays the feature maps
#         # Start next subplot.
#         figure = plt.figure(figsize=(20, 10))

#         if len(layer_weigth.shape) == 4:
#             sizeW, sizeH, nChannel, nKernels = layer_weigth.shape
#             for iChannel, iKernel in itertools.product(range(1), range(nKernels)):
#                 plt.subplot(1, nKernels, iChannel * nKernels + iKernel + 1, title=str(iKernel) + "-" + str(iChannel))
#                 plt.xticks([])
#                 plt.yticks([])
#                 plt.grid(False)
#                 plt.imshow(layer_weigth[:, :, iChannel, iKernel], cmap="viridis")

#         # Save the plot to a PNG in memory.
#         buf = io.BytesIO()
#         plt.savefig(buf, format="png")
#         # Closing the figure prevents it from being displayed directly inside
#         # the notebook.
#         plt.close(figure)
#         buf.seek(0)
#         # Convert PNG buffer to TF image
#         image = tf.image.decode_png(buf.getvalue(), channels=4)
#         # Add the batch dimension
#         image = tf.expand_dims(image, 0)

#         with file_writer.as_default():
#             tf.summary.image(layer_name, image, step=epoch)

