import logging
import time
from collections import defaultdict
from os import stat

import madmom
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy.core.defchararray import mod
from tensorflow.keras import layers
from tensorflow.python.eager.monitoring import Sampler

from adtof.deepModels.peakPicking import PeakPicking
from adtof.io import eval


class RV1TF(object):
    """
    Richard Vogl model
    http://ifs.tuwien.ac.at/~vogl/
    """

    # def __init__(self, peakThreshold=0.15, sampleRate=100, **kwargs):
    #     self.pp = PeakPicking()
    #     self.ppp = madmom.features.notes.NotePeakPickingProcessor(
    #         threshold=peakThreshold, smooth=0, pre_avg=0.1, post_avg=0.01, pre_max=0.02, post_max=0.01, combine=0.02, fps=sampleRate
    #     )

    def _getCNN(self, context, n_bins, output):
        """
        Parameters from Vogl

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
        tfModel.add(tf.keras.layers.Reshape((-1, featureDim)))  # timeSerieDim might change if the full track is provided
        tfModel.add(
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(60, stateful=True, return_sequences=True))
        )  # return the whole sequence for the next layers
        tfModel.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(60, stateful=True, return_sequences=True)))
        tfModel.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(60, stateful=True, return_sequences=False)))
        tfModel.add(tf.keras.layers.Dense(output, activation="sigmoid", name="denseOutput"))
        return tfModel

    def createModel(
        self, model="cnn", context=25, n_bins=168, output=5, learningRate=0.001 / 2, batchSize=100, samplePerTrack=100, **kwargs
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

    def predict(self, _model, dataset, peakPickingTarget="sum F all", peakPickingStep=0.05, sampleRate=100, labelOffset=0, **kwargs):
        """
        Run model.predict on the dataset followed by madmom.peakpicking. Find the best threshold for the peak 
        """
        predictions = []
        Y = []
        gen = dataset()
        for i, (x, y) in enumerate([next(gen), next(gen)]):
            startTime = time.time()
            predictions.append(_model.predict(x))
            Y.append(y)
            logging.debug("track %s predicted in %s", i, time.time() - startTime)

        timeOffset = 0  # labelOffset / sampleRate
        results = []
        for peakThreshold in np.arange(peakPickingStep, 0.5, peakPickingStep):
            ppp = madmom.features.notes.NotePeakPickingProcessor(
                threshold=peakThreshold, smooth=0, pre_avg=0.1, post_avg=0.01, pre_max=0.02, post_max=0.01, combine=0.02, fps=sampleRate
            )
            YHat = [self.processPP(ppp, prediction, timeOffset, **kwargs) for prediction in predictions]
            score = eval.runEvaluation(Y, YHat)
            score["peakThreshold"] = peakThreshold
            results.append(score)

        return max(results, key=lambda x: x[peakPickingTarget])

    def processPP(self, ppp, prediction, timeOffset, labels=[36], **kwargs):
        """
        Call Madmom's peak picking processor on one track's prediction and transform the output in the correct format 
        """
        sparseResultIdx = ppp.process(prediction)
        result = defaultdict(list)
        for time, pitch in sparseResultIdx:
            result[labels[int(pitch - 21)]].append(time + timeOffset)

        return result


def log_layers(epoch, input, model, activation_model, file_writer):
    log_layer_activation(epoch, input, model, activation_model, file_writer)
    log_layer_weights(epoch, input, model, activation_model, file_writer)


def log_layer_activation(epoch, input, model, activation_model, file_writer):
    # Create a figure to contain the plot.
    layer_names = ["activation 0 input"] + ["activation " + str(i + 1) + " " + layer.name for i, layer in enumerate(model.layers)]
    layer_activations = [input] + activation_model.predict(input)
    for iLayer, (layer_name, layer_activation) in enumerate(zip(layer_names, layer_activations)):  # Displays the feature maps
        # Start next subplot.
        figure = plt.figure(figsize=(20, 10))

        if len(layer_activation.shape) == 4:
            _, sizeW, sizeH, nKernels = layer_activation.shape
            for iKernel in range(nKernels):
                plt.subplot(nKernels // 5 + 1, 5, iKernel + 1, title=str(iKernel))
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(layer_activation[0, :, :, iKernel], cmap="viridis")

        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)

        with file_writer.as_default():
            tf.summary.image(layer_name, image, step=epoch)


def log_layer_weights(epoch, input, model, activation_model, file_writer):
    # Create a figure to contain the plot.
    layer_names = ["weights " + str(i + 1) + " " + layer.name for i, layer in enumerate(model.layers) if len(layer.get_weights())]
    layer_weigths = [layer.get_weights()[0] for layer in model.layers if len(layer.get_weights())]
    for iLayer, (layer_name, layer_weigth) in enumerate(zip(layer_names, layer_weigths)):  # Displays the feature maps
        # Start next subplot.
        figure = plt.figure(figsize=(20, 10))

        if len(layer_weigth.shape) == 4:
            sizeW, sizeH, nChannel, nKernels = layer_weigth.shape
            for iChannel, iKernel in itertools.product(range(1), range(nKernels)):
                plt.subplot(1, nKernels, iChannel * nKernels + iKernel + 1, title=str(iKernel) + "-" + str(iChannel))
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(layer_weigth[:, :, iChannel, iKernel], cmap="viridis")

        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)

        with file_writer.as_default():
            tf.summary.image(layer_name, image, step=epoch)


# rv1 = RV1()

# """
# load
# """
# import pickle

# file="/home/mickael/Documents/programming/madmom-0.16.dev0/madmom/models/drums/2018/drums_cnn0_O8_S0.pkl"

# with open(file, "rb") as f:
#     u = pickle._Unpickler(f)
#     u.encoding = "latin1"
#     p = u.load()
#     print(p)


# file = "/Users/mzehren/Programming/ADTOF/vendors/madmom-0.16.dev0/madmom/models/drums/2018/drums_crnn1_O8_S0.pkl"
