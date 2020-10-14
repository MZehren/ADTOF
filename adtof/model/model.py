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
from adtof.io.mir import MIR


class Model(object):
    """
    Richard Vogl model
    http://ifs.tuwien.ac.at/~vogl/
    """

    @staticmethod
    def modelFactory(fold=0):
        models = {
            # "cnn-stride(1,3)-shuffledinput": {
            #     "labels": config.LABELS_5,
            #     "classWeights": config.WEIGHTS_5 / 2,
            #     "sampleRate": 100,
            #     "diff": True,
            #     "samplePerTrack": 20,
            #     "batchSize": 100,
            #     "context": 25,
            #     "labelOffset": 1,
            #     "labelRadiation": 1,
            #     "learningRate": 0.0001,
            #     "normalize": False,
            #     "model": "CNN",
            #     "fmin": 20,
            #     "fmax": 20000,
            #     "pad": False,
            #     "beat_targ": False,
            #     "tracksLimit": None,
            # },
            # "TCN": {
            #     "labels": config.LABELS_5,
            #     "classWeights": config.WEIGHTS_5 / 2,
            #     "sampleRate": 100,
            #     "diff": False,
            #     "samplePerTrack": 20,
            #     "batchSize": 100,
            #     "context": 8193,
            #     "labelOffset": 8193 // 2,
            #     "labelRadiation": 1,
            #     "learningRate": 0.0001,
            #     "normalize": False,
            #     "model": "TCN",
            #     "fmin": 30,
            #     "fmax": 17000,
            #     "pad": False,
            #     "beat_targ": False,
            #     "tracksLimit": None,
            # },
            "crnn": {
                "labels": config.LABELS_5,
                "classWeights": config.WEIGHTS_5 / 2,
                "sampleRate": 100,
                "diff": True,
                "samplePerTrack": 400,
                "trainingSequence": 400,
                "batchSize": 8,
                "context": 13,  # in RNN, The context is used as time serie, using a bigger one is not increasing the total params
                "labelOffset": 1,
                "labelRadiation": 1,
                "learningRate": 0.0001,
                "normalize": False,
                "model": "CRNN",
                "fmin": 20,
                "fmax": 20000,
                "pad": False,
                "beat_targ": False,
            }
        }

        for modelName, hparams in models.items():
            modelName += "_Fold" + str(fold)
            yield (Model(modelName, **hparams), hparams)

    def __init__(self, name, **kwargs):
        n_bins = MIR(**kwargs).getDim()
        self.model = self._createModel(n_bins=n_bins, output=len(kwargs["labels"]), **kwargs)
        self.name = name
        self.path = os.path.join(config.CHECKPOINT_DIR, name)
        # if model is already trained, load the weights
        if os.path.exists(self.path + ".index"):  # TODO: This check is not case sensitive, but macOS is
            logging.info("Loading model weights %s", self.path)
            self.model.load_weights(self.path)
            self.weightLoadedFlag = True
        else:
            self.weightLoadedFlag = False

    def _getCNN(self, context, n_bins, output):
        """
        Parameters from Vogl

        import pickle
        # file="/madmom-0.16.dev0/madmom/models/drums/2018/drums_cnn0_O8_S0.pkl"
        file = "vendors/madmom-0.16.dev0/madmom/models/drums/2018/drums_crnn1_O8_S0.pkl"
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

        Total params: 4,591,589
        Trainable params: 4,590,181
        Non-trainable params: 1,408

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

    def _getTCNSequential(self, context, n_bins, output):
        """
        Model from MatthewDavies and Bock - 2019 - Temporal convolutional networks for musical audio
        
        Total params: 237,061
        Trainable params: 235,461
        Non-trainable params: 1,600
        """
        tfModel = tf.keras.Sequential()
        # Conv blocks
        tfModel.add(tf.keras.layers.Conv2D(16, (3, 3), input_shape=(context, n_bins, 1), activation="elu"))
        tfModel.add(tf.keras.layers.BatchNormalization())
        tfModel.add(tf.keras.layers.MaxPool2D(pool_size=(1, 3), strides=(1, 3)))
        tfModel.add(tf.keras.layers.Dropout(0.1))

        tfModel.add(tf.keras.layers.Conv2D(16, (3, 3), activation="elu"))
        tfModel.add(tf.keras.layers.BatchNormalization())
        tfModel.add(tf.keras.layers.MaxPool2D(pool_size=(1, 3), strides=(1, 3)))
        tfModel.add(tf.keras.layers.Dropout(0.1))

        tfModel.add(tf.keras.layers.Conv2D(16, (1, 8), activation="elu"))
        tfModel.add(tf.keras.layers.BatchNormalization())
        tfModel.add(tf.keras.layers.Dropout(0.1))

        # TCN
        tfModel.add(tf.keras.layers.Reshape((-1, 16)))
        for i in range(11):
            tfModel.add(tf.keras.layers.Conv1D(16, 5, activation="elu", strides=1, dilation_rate=2 ** i))
            tfModel.add(tf.keras.layers.BatchNormalization())
            tfModel.add(tf.keras.layers.Dropout(0.1))

        tfModel.add(tf.keras.layers.Flatten())
        tfModel.add(tf.keras.layers.Dense(output, activation="sigmoid", name="denseOutput"))
        # tfModel.build()
        # tfModel.summary()
        return tfModel

    def _getTCNFunctional(self, contex, n_bins, output):
        """[summary]

        Parameters
        ----------
        contex : [type]
            [description]
        n_bins : [type]
            [description]
        output : [type]
            [description]
        """
        cnnInput = tf.keras.Input(shape=(5, n_bins, output))

        x = tf.keras.layers.Conv2D(16, (3, 3), activation="elu")(cnnInput)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(1, 3), strides=(1, 3))(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        x = tf.keras.layers.Conv2D(16, (3, 3), activation="elu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(1, 3), strides=(1, 3))(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        x = tf.keras.layers.Conv2D(16, (1, 8), activation="elu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        cnnOutput = tf.keras.layers.Dropout(0.1)(x)

        cnnModel = tf.keras.Model(cnnInput, cnnOutput)
        cnnModel.summary()

        tcnInput = tf.keras.Input(shape=())

    def _getCRNN(self, context, n_bins, output, batchSize, trainingSequence):
        """

        00:<madmom.ml.nn.layers.ConvolutionalLayer object at 0x1031c1190>   (1, 64, 3, 3)
        01:<madmom.ml.nn.layers.BatchNormLayer object at 0x14f6ff4f0>       (64)
        02:<madmom.ml.nn.layers.ConvolutionalLayer object at 0x14f6ff730>   (64, 64, 3, 3)
        03:<madmom.ml.nn.layers.BatchNormLayer object at 0x14f6ff820>       (64)
        04:<madmom.ml.nn.layers.MaxPoolLayer object at 0x14f6ff9a0>
        05:<madmom.ml.nn.layers.ConvolutionalLayer object at 0x14f6ffa30>   (64, 32, 3, 3)
        06:<madmom.ml.nn.layers.BatchNormLayer object at 0x14f6ffb50>       (32)
        07:<madmom.ml.nn.layers.ConvolutionalLayer object at 0x14f6ffd30>   (32, 32, 3, 3)
        08:<madmom.ml.nn.layers.BatchNormLayer object at 0x14f6ffe20>       (32)
        09:<madmom.ml.nn.layers.MaxPoolLayer object at 0x14f707040>
        10:<madmom.ml.nn.layers.StrideLayer object at 0x14f7070d0> Re-arrange (stride) the data in blocks of given size (5).
        11:<madmom.ml.nn.layers.BidirectionalLayer object at 0x14f707100>   (1120, 60)
        12:<madmom.ml.nn.layers.BidirectionalLayer object at 0x14f707280>   (120, 60)
        13:<madmom.ml.nn.layers.BidirectionalLayer object at 0x14f7104c0>   (120, 60)
        14:<madmom.ml.nn.layers.FeedForwardLayer object at 0x14f710640>     (120, 8)

        Shape of one track being analyzed
        ('0', '<madmom.ml.nn.layers.ConvolutionalLayer object at 0x129dcd9d0>', 'in ',  (31511, 79),        'out', (31509, 77, 64))
        ('1', '<madmom.ml.nn.layers.BatchNormLayer object at 0x129dcdf10>', 'in ',      (31509, 77, 64),    'out', (31509, 77, 64))
        ('2', '<madmom.ml.nn.layers.ConvolutionalLayer object at 0x129dcd190>', 'in ',  (31509, 77, 64),    'out', (31507, 75, 64))
        ('3', '<madmom.ml.nn.layers.BatchNormLayer object at 0x129dcdf90>', 'in ',      (31507, 75, 64),    'out', (31507, 75, 64))
        ('4', '<madmom.ml.nn.layers.MaxPoolLayer object at 0x129d4ff50>', 'in ',        (31507, 75, 64),    'out', (31507, 25, 64))
        ('5', '<madmom.ml.nn.layers.ConvolutionalLayer object at 0x129d4f250>', 'in ',  (31507, 25, 64),    'out', (31505, 23, 32))
        ('6', '<madmom.ml.nn.layers.BatchNormLayer object at 0x129d4f750>', 'in ',      (31505, 23, 32),    'out', (31505, 23, 32))
        ('7', '<madmom.ml.nn.layers.ConvolutionalLayer object at 0x129ddb090>', 'in ',  (31505, 23, 32),    'out', (31503, 21, 32))
        ('8', '<madmom.ml.nn.layers.BatchNormLayer object at 0x129ddb2d0>', 'in ',      (31503, 21, 32),    'out', (31503, 21, 32))
        ('9', '<madmom.ml.nn.layers.MaxPoolLayer object at 0x129ddb510>', 'in ',        (31503, 21, 32),    'out', (31503, 7, 32))
        ('10', '<madmom.ml.nn.layers.StrideLayer object at 0x129ddb550>', 'in ',        (31503, 7, 32),     'out', (31499, 1120))
        ('11', '<madmom.ml.nn.layers.BidirectionalLayer object at 0x129ddb610>', 'in ', (31499, 1120),      'out', (31499, 120))
        ('12', '<madmom.ml.nn.layers.BidirectionalLayer object at 0x129dfc210>', 'in ', (31499, 120),       'out', (31499, 120))
        ('13', '<madmom.ml.nn.layers.BidirectionalLayer object at 0x129dfce50>', 'in ', (31499, 120),       'out', (31499, 120))
        ('14', '<madmom.ml.nn.layers.FeedForwardLayer object at 0x129e04990>', 'in ',   (31499, 120),       'out', (31499, 8))


        TODO: How to handle recurence granularity?
        https://www.tensorflow.org/guide/keras/rnn#cross-batch_statefulness
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN#masking_2
        https://adgefficiency.com/tf2-lstm-hidden/
        https://www.tensorflow.org/tutorials/structured_data/time_series
        TODO: How to handle bidirectionnality?
        TODO: How to handle mini_batch of size 8 with training sequence of 400 instances and context of 13
        """
        tfModel = tf.keras.Sequential()
        tfModel.add(
            tf.keras.layers.Conv2D(
                32,
                (3, 3),
                input_shape=(trainingSequence, n_bins, 1),
                batch_input_shape=(batchSize, trainingSequence, n_bins, 1),
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
        timeSerieDim = trainingSequence - (context - 1)
        featureDim = ((n_bins - 2 * 2) // 3 - 2 * 2) // 3 * 64
        tfModel.add(tf.keras.layers.Reshape((timeSerieDim, -1)))  # timeSerieDim might change if the full track is provided
        # return the whole sequence for the next layers with return_sequence
        tfModel.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(50, stateful=False, return_sequences=True)))
        tfModel.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(50, stateful=False, return_sequences=True)))
        tfModel.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(50, stateful=False, return_sequences=True)))
        tfModel.add(tf.keras.layers.Dense(output, activation="sigmoid", name="denseOutput"))
        tfModel.build()
        tfModel.summary()
        return tfModel

    def _createModel(
        self, model="cnn", context=25, n_bins=168, output=5, learningRate=0.001 / 2, batchSize=100, trainingSequence=100, **kwargs
    ):
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
            tfModel = self._getCRNN(context, n_bins, output, batchSize, trainingSequence)
        elif model == "TCN":
            tfModel = self._getTCNSequential(context, n_bins, output)
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

    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)

    @staticmethod
    def predictEnsemble(models, x, aggregation=np.mean):
        return aggregation([model.predict(x) for model in models], axis=0)

    def evaluate(self, dataset, peakThreshold=None, context=20, batchSize=32, **kwargs):
        """
        Run model.predict on the dataset followed by madmom.peakpicking. Find the best threshold for the peak 
        """
        gen = dataset()
        predictions = []
        Y = []
        self.model.reset_states()
        for i, (x, y) in enumerate(gen):
            startTime = time.time()

            def localGenerator():
                totalSamples2 = len(x) - context
                for i in range(0, totalSamples2 - batchSize, batchSize):
                    yield np.array([x[i + j : i + j + context] for j in range(batchSize)])

            predictions.append(self.predict(localGenerator()))
            self.model.reset_states()  # TODO is this mandatory
            Y.append(y)
            logging.debug("track %s predicted in %s", i, time.time() - startTime)
        if peakThreshold == None:
            return peakPicking.fitPeakPicking(predictions, Y, **kwargs)
        else:
            return peakPicking.fitPeakPicking(predictions, Y, peakPickingSteps=[peakThreshold], **kwargs)

    def vizPredictions(self, dataset, labels=[35], batchSize=100, **kwargs):
        """
        Plot the input, output and target of the model
        """

        for x, y, _ in dataset:
            predictions = self.predict(x, batch_size=batchSize)
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
