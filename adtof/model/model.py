import datetime
import logging
import os
import time

import numpy as np
import pretty_midi
import tensorflow as tf
from adtof import config
from adtof.io.mir import MIR
from adtof.io.textReader import TextReader
from adtof.model import peakPicking
from adtof.model.dataLoader import DataLoader


class Model(object):
    @staticmethod
    def modelFactory(modelName="crnn-ADTOF", fold=0):
        """
        Instantiate and return a model with its hyperparameters.
        """
        models = {
            "crnn-all": {
                "labels": config.LABELS_5,  # Classes to predict
                "classWeights": config.WEIGHTS_5,  # Weights applied to the classes during training
                "emptyWeight": 1,  # weight of a sample without any onset
                "sampleRate": 100,  # sample rate of the predictions
                "diff": True,  # stacking the spectrogram input with its first order difference
                "samplePerTrack": 1,  # Number of training sequence per track extracted before going to the next track during training
                "trainingSequence": 400,  # How many Fourier transforms constitue a training sequence
                "batchSize": 8,  # How many training sequences per minibatch.
                "context": 9,  # Number of samples given to the Convolutional layer (only 9 supported atm)
                "labelOffset": 1,  # How many samples to offset the ground truth labels to make sure the attack is not missed
                "labelRadiation": 1,  # How many samples from the target have a non-null value
                "learningRate": 0.001,  # Learning rate
                "normalize": False,  # Normalizing the network input per track
                "architecture": "CRNN",  # What model architecture is used
                "fmin": 20,  # Min frequency limit to the Fourier transform
                "fmax": 20000,  # Max frequency limit to the Fourier transform
                "validation_epoch": 10,  # how many training sequence per track of the validation set has to be seen to consider an epoch
                "training_epoch": 10,  # how many training sequence per track of the training set has to be seen to consider an epoch
                "reduce_patience": 10,  # how many epoch without improvement on validation before reducing the lr
                "stopping_patience": 25,  # how many epoch without improvement on validation before stopping the training
                "peakThreshold": 0.1,  # peakThreshold computed on the validation set
            },
            "crnn-ptTMIDT": {
                "labels": config.LABELS_5,
                "classWeights": config.WEIGHTS_5,
                "emptyWeight": 1,
                "sampleRate": 100,
                "diff": True,
                "samplePerTrack": 1,
                "trainingSequence": 400,
                "batchSize": 8,
                "context": 9,
                "labelOffset": 1,
                "labelRadiation": 1,
                "learningRate": 0.001,
                "normalize": False,
                "architecture": "CRNN",
                "fmin": 20,
                "fmax": 20000,
                "validation_epoch": 10,
                "training_epoch": 10,
                "reduce_patience": 10,
                "stopping_patience": 25,
                "peakThreshold": 0.16999999999999998,
            },
            "crnn-TMIDT": {
                "labels": config.LABELS_5,
                "classWeights": config.WEIGHTS_5,
                "emptyWeight": 1,
                "sampleRate": 100,
                "diff": True,
                "samplePerTrack": 1,
                "trainingSequence": 400,
                "batchSize": 8,
                "context": 9,
                "labelOffset": 1,
                "labelRadiation": 1,
                "learningRate": 0.001,
                "normalize": False,
                "architecture": "CRNN",
                "fmin": 20,
                "fmax": 20000,
                "validation_epoch": 0.5,
                "training_epoch": 0.5,
                "reduce_patience": 5,
                "stopping_patience": 10,
            },
            "crnn-ADTOF": {
                "labels": config.LABELS_5,
                "classWeights": config.WEIGHTS_5,
                "emptyWeight": 1,
                "sampleRate": 100,
                "diff": True,
                "samplePerTrack": 1,
                "trainingSequence": 400,
                "batchSize": 8,
                "context": 9,
                "labelOffset": 1,
                "labelRadiation": 1,
                "learningRate": 0.0005,
                "normalize": False,
                "architecture": "CRNN",
                "fmin": 20,
                "fmax": 20000,
                "validation_epoch": 1,
                "training_epoch": 1,
                "reduce_patience": 10,
                "stopping_patience": 25,
                "peakThreshold": 0.22999999999999995,
            },
        }

        hparams = models[modelName]
        return (Model(modelName + "_Fold" + str(fold), **hparams), hparams)

    def __init__(self, name, **kwargs):
        """
        Encapsulate a TF model which knows how to create itself and ease the workflow.
        Should be created with the static function Model.modelFactory()
        """
        n_bins = MIR(**kwargs).getDim()
        self.model = self._createModel(n_bins=n_bins, output=len(kwargs["labels"]), **kwargs)
        self.name = name
        self.path = os.path.join(config.CHECKPOINT_DIR, name)
        # if model is already trained, load the weights and flag to not retrain
        if os.path.exists(self.path + ".index"):  # TODO: This check is not case sensitive, but macOS/Linux are
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
        Richard Vogl model
        http://ifs.tuwien.ac.at/~vogl/

        00:<madmom.ml.nn.layers.ConvolutionalLayer object at 0x1031c1190>   (1, 64, 3, 3)   linear
        01:<madmom.ml.nn.layers.BatchNormLayer object at 0x14f6ff4f0>       (64)            relu
        02:<madmom.ml.nn.layers.ConvolutionalLayer object at 0x14f6ff730>   (64, 64, 3, 3)  linear
        03:<madmom.ml.nn.layers.BatchNormLayer object at 0x14f6ff820>       (64)            relu
        04:<madmom.ml.nn.layers.MaxPoolLayer object at 0x14f6ff9a0>
        05:<madmom.ml.nn.layers.ConvolutionalLayer object at 0x14f6ffa30>   (64, 32, 3, 3)  linear
        06:<madmom.ml.nn.layers.BatchNormLayer object at 0x14f6ffb50>       (32)            relu
        07:<madmom.ml.nn.layers.ConvolutionalLayer object at 0x14f6ffd30>   (32, 32, 3, 3)  linear
        08:<madmom.ml.nn.layers.BatchNormLayer object at 0x14f6ffe20>       (32)            relu
        09:<madmom.ml.nn.layers.MaxPoolLayer object at 0x14f707040>
        10:<madmom.ml.nn.layers.StrideLayer object at 0x14f7070d0> Re-arrange (stride) the data in blocks of given size (5).
        11:<madmom.ml.nn.layers.BidirectionalLayer object at 0x14f707100>   (1120, 60)      tanh (sigmoid for gate)
        12:<madmom.ml.nn.layers.BidirectionalLayer object at 0x14f707280>   (120, 60)       tanh (sigmoid for gate)
        13:<madmom.ml.nn.layers.BidirectionalLayer object at 0x14f7104c0>   (120, 60)       tanh (sigmoid for gate)
        14:<madmom.ml.nn.layers.FeedForwardLayer object at 0x14f710640>     (120, 8)        sigmoid

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
        TODO: Why is the conv blocks inversed. Vogl's article explains 32 filters then 64 filters, the code has 64 filters, then 32
        """
        # Compte the input size
        xWindowSize = context + (trainingSequence - 1)

        tfModel = tf.keras.Sequential()

        tfModel.add(
            tf.keras.layers.Conv2D(
                64,
                (3, 3),
                # input shape is optional for the first layer, if ommited the model is built with the first call to training.
                # Specifying it anyway for safe measure, the time axis is set to None because tracks have different length
                # in the input_shape, the batch size is ommited
                input_shape=(None, n_bins, 1,),
                activation="relu",
                strides=(1, 1),
                padding="valid",
                name="conv11",
            )
        )
        tfModel.add(tf.keras.layers.BatchNormalization())
        tfModel.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", strides=(1, 1), padding="valid", name="conv12"))
        tfModel.add(tf.keras.layers.BatchNormalization())
        tfModel.add(tf.keras.layers.MaxPool2D(pool_size=(1, 3), strides=(1, 3), padding="valid"))
        tfModel.add(tf.keras.layers.Dropout(0.3))
        tfModel.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", strides=(1, 1), padding="valid", name="conv21"))
        tfModel.add(tf.keras.layers.BatchNormalization())
        tfModel.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", strides=(1, 1), padding="valid", name="conv22"))
        tfModel.add(tf.keras.layers.BatchNormalization())
        tfModel.add(tf.keras.layers.MaxPool2D(pool_size=(1, 3), strides=(1, 3), padding="valid"))
        tfModel.add(tf.keras.layers.Dropout(0.3))

        # we set the model as a sequential model, the recurrency is done inside the batch and not between batches
        # tfModel.add(tf.keras.layers.Flatten())  replace the flatten by a reshape to [batchSize, timeSerieDim, featureDim]
        timeSerieDim = xWindowSize - (context - 1)
        featureDim = ((n_bins - 2 * 2) // 3 - 2 * 2) // 3 * 32
        # TODO change to a stride layer to actually collapse the context into one value, even if the convolution doesn't reduce the size to one.s
        tfModel.add(tf.keras.layers.Reshape((-1, featureDim)))
        tfModel.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(50, stateful=False, return_sequences=True)))
        tfModel.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(50, stateful=False, return_sequences=True)))
        tfModel.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(50, stateful=False, return_sequences=True)))
        tfModel.add(tf.keras.layers.Dense(output, activation="sigmoid", name="denseOutput"))
        # tfModel.build((batchSize, None, n_bins, 1))
        # tfModel.summary()
        return tfModel

    def _createModel(
        self, architecture="CRNN", context=25, n_bins=168, output=5, learningRate=0.001 / 2, batchSize=100, trainingSequence=100, **kwargs
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
        if architecture == "CNN":
            tfModel = self._getCNN(context, n_bins, output)
        elif architecture == "CRNN":
            tfModel = self._getCRNN(context, n_bins, output, batchSize, trainingSequence)
        elif architecture == "TCN":
            tfModel = self._getTCNSequential(context, n_bins, output)
        else:
            raise ValueError("%s not known", architecture)

        # Very interesting read on loss functions: https://gombru.github.io/2018/05/23/cross_entropy_loss/
        # How softmax cross entropy can be used in multilabel classification,
        # and how binary cross entropy work for multi label
        tfModel.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),  # "adam",  #tf.keras.optimizers.RMSprop(learning_rate=0.001),
            loss=tf.keras.backend.binary_crossentropy,  # tf.nn.sigmoid_cross_entropy_with_logits,  #tf.keras.backend.binary_crossentropy,
            metrics=["Precision", "Recall"],  # PeakPicking(hitDistance=0.05)
        )
        return tfModel

    def fit(self, dataset_train, dataset_val, log_dir, steps_per_epoch, validation_steps, reduce_patience=3, stopping_patience=6, **kwargs):
        """
        Fits the model to the train dataset and validate on the val dataset to reduce LR on plateau and do an earlystopping. 
        """
        logging.info("Training model %s", self.name)

        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=log_dir + self.name + datetime.datetime.now().strftime("%d-%m-%H:%M")),
            tf.keras.callbacks.ModelCheckpoint(self.path, save_weights_only=True, save_best_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, verbose=1, patience=reduce_patience),
            tf.keras.callbacks.EarlyStopping(min_delta=0.0001, patience=stopping_patience, verbose=1, restore_best_weights=True),
            # tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: log_layer_activation(epoch, viz_example, model, activation_model, file_writer))
        ]
        self.model.fit(
            dataset_train,
            epochs=1000,  # Very high number of epoch to stop only with ealy stopping
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            validation_data=dataset_val,
            validation_steps=validation_steps,
        )

    def predict(self, x, trainingSequence=1, limitInputSize=60000, **kwargs):
        """
        Call model.predict if possible
        If the model is a CRNN and can't utilize the batch parallelisation, call directly the model 

        limitInputSize:
            Set a limit to 10 minutes otherwise a segmentation Fault might be raised! 
            TODO: could split the input in two to prevent that
        """
        if trainingSequence == 1:
            return self.model.predict(x, **kwargs)
        else:
            if len(x) > limitInputSize:
                raise ValueError("The input array is too big")
            # Put everyting in the first batch
            return np.array(self.model(np.array([x]), training=False)[0])

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
        logging.info("prediction folder " + str(inputFolder))
        ppp = peakPicking.getPPProcess(**kwargs)
        dl = DataLoader(inputFolder, crossValidation=False, lazyLoading=True)

        # Create a generator yielding full tracks one by one
        predictParam = {k: v for k, v in kwargs.items()}
        predictParam["repeat"] = False
        predictParam["samplePerTrack"] = None
        tracks = dl.getGen(**predictParam)

        # Predict the file and write the output
        for (x, _), track in zip(tracks(), dl.audioPaths):
            try:
                if not os.path.exists(outputFolder):
                    os.makedirs(outputFolder)
                outputTrackPath = os.path.join(outputFolder, config.getFileBasename(track) + ".txt")
                if os.path.exists(outputTrackPath):
                    continue

                Y = self.predict(x, **kwargs)
                sparseResultIdx = peakPicking.peakPicking(
                    Y, ppProcess=ppp, timeOffset=kwargs["labelOffset"] / kwargs["sampleRate"], **kwargs
                )

                # write text
                formatedOutput = [(time, pitch) for pitch, times in sparseResultIdx.items() for time in times]
                formatedOutput.sort(key=lambda x: x[0])
                TextReader().writteBeats(outputTrackPath, formatedOutput)

                #  write midi
                if writeMidi:
                    midi = pretty_midi.PrettyMIDI()
                    instrument = pretty_midi.Instrument(program=1, is_drum=True)
                    midi.instruments.append(instrument)
                    for pitch, notes in sparseResultIdx.items():
                        for i in notes:
                            note = pretty_midi.Note(velocity=100, pitch=pitch, start=i, end=i)
                            instrument.notes.append(note)
                    midi.write(os.path.join(outputFolder, config.getFileBasename(track) + ".mid"))

            except Exception as e:
                logging.error(str(e))

    def evaluate(self, dataset, peakThreshold=None, context=20, trainingSequence=1, batchSize=32, **kwargs):
        """
        Run model.predict on the dataset followed by madmom.peakpicking. Find the best threshold for the peak 
        The dataset needs to return full tracks and not independant samples  
        """
        predictions = []
        Y = []

        def localGenerator(seq, sequence, batch):
            """
            iterate over a sequence by building batches of length sequence
            """
            totalSamples = len(seq) - sequence
            for i in range(0, totalSamples - batch, batch):
                yield np.array([seq[i + j : i + j + sequence] for j in range(batch)])

        for i, (x, y) in enumerate(dataset):
            try:
                startTime = time.time()
                if trainingSequence == 1:  # TODO put that into the predict method?
                    predictions.append(self.predict(localGenerator(x, context, batchSize)))
                else:
                    predictions.append(self.predict(x, trainingSequence=trainingSequence))
                logging.debug("track %s predicted in %s", i, time.time() - startTime)
                Y.append(y)
            except Exception as e:
                logging.error("track %s not predicted! Skipped because of %s", i, str(e))

        if peakThreshold == None:
            return peakPicking.fitPeakPicking(predictions, Y, **kwargs)
        else:
            return peakPicking.fitPeakPicking(predictions, Y, peakPickingSteps=[peakThreshold], **kwargs)

    def vizPredictions(self, dataset, labels=[35], batchSize=100, trainingSequence=1, **kwargs):
        """
        Plot the input, output and target of the model
        """

        for x, y, _ in dataset:
            predictions = self.predict(x, batch_size=batchSize)

            # if the data is not a batch of samples, but a batch of sequence of samples (i.e: model crnn)
            if trainingSequence > 1:
                predictions = predictions[0]
                y = y[0]
                x = x[0]

            import matplotlib.pyplot as plt

            f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.plot(predictions, alpha=0.8)

            ax1.set_ylabel("Prediction + Truth")

            for classIdx in range(len(labels)):
                idx = [i for i, v in enumerate(y[:, classIdx]) if v == 1]
                ax1.scatter(idx, [classIdx / 20 + 1 for _ in range(len(idx))])
            ax2.set_ylabel("Input")
            ax2.set_xlabel("Time step")
            if trainingSequence == 1:
                ax2.matshow(tf.transpose(tf.reshape(x[:, 0], (batchSize, -1))), aspect="auto")
            else:
                ax2.matshow(tf.transpose(tf.reshape(x, x.shape[:2])), aspect="auto")
            ax1.legend(labels)
            plt.show()

            # Paper viz
            plt.plot(np.array([c + i for i, c in enumerate(predictions.T)]).T)
            pp = peakPicking.peakPicking(predictions, **kwargs)

            for cli, cl in enumerate(labels):
                idx = [int(v * kwargs["sampleRate"]) for v in pp[cl]]
                plt.scatter(idx, [predictions[sample][cli] + float(cli) for sample in idx], facecolors="none", edgecolors="r")
            plt.show()

