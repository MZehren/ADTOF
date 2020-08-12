import matplotlib.pyplot as plt
import tensorflow as tf

from adtof.deepModels.peakPicking import PeakPicking


class RV1TF(object):
    """
    Richard Vogl model
    http://ifs.tuwien.ac.at/~vogl/
    """

    # def __init__(self):
    #     # self.model = self.createModel()
    #     pass

    def createModel(self, context=25, n_bins=168, output=5, learningRate=0.001 / 2, **kwargs):
        """Return a ts model based 
        
        Keyword Arguments:
            context {int} -- [description] (default: {25})
            n_bins {int} -- [description] (default: {84})
            output {int} -- number of classes in the output (should be the events: 36, 40, 41, 46, 49) (default: {5})
            outputWeight {list} --  (default: {[]}) 
        
        Returns:
            [type] -- [description]
        """
        # How to handle the bidirectional aggregation ? Sum, or nothing ?
        # TODO: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#optional_set_the_correct_initial_bias
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(32, (3, 3), input_shape=(context, n_bins, 1), activation="relu", name="conv11"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(32, (3, 3), activation="relu", name="conv12"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu", name="conv21"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu", name="conv22"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
                tf.keras.layers.Dropout(0.3),
                # tf.keras.layers.Bidirectional(tf.keras.layers.GRU(60)),
                # tf.keras.layers.Bidirectional(tf.keras.layers.GRU(60)),
                # tf.keras.layers.Bidirectional(tf.keras.layers.GRU(60)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation="relu", name="dense1"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(256, activation="relu", name="dense2"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(output, activation="sigmoid", name="denseOutput"),
            ]
        )

        # Very interesting read on loss functions: https://gombru.github.io/2018/05/23/cross_entropy_loss/
        # How softmax cross entropy can be used in multilabel classification,
        # and how binary cross entropy work for multi label
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),  # "adam",  #tf.keras.optimizers.RMSprop(learning_rate=0.001),
            loss=tf.keras.backend.binary_crossentropy,  # tf.nn.sigmoid_cross_entropy_with_logits,  #tf.keras.backend.binary_crossentropy,
            metrics=["Precision", "Recall"],  # PeakPicking(hitDistance=0.05)
        )
        return model


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
#     u.encoding = 'latin1'
#     p = u.load()
#     print(p)

# 00:<madmom.ml.nn.layers.ConvolutionalLayer object at 0x7fd5b2f38ac8>
# 01:<madmom.ml.nn.layers.BatchNormLayer object at 0x7fd5722e5d30>
# 02:<madmom.ml.nn.layers.ConvolutionalLayer object at 0x7fd5722e5cf8>
# 03:<madmom.ml.nn.layers.BatchNormLayer object at 0x7fd5722e5f28>
# 04:<madmom.ml.nn.layers.MaxPoolLayer object at 0x7fd5b3eca978>
# 05:<madmom.ml.nn.layers.ConvolutionalLayer object at 0x7fd5b3eca6d8>
# 06:<madmom.ml.nn.layers.BatchNormLayer object at 0x7fd5b3eca438>
# 07:<madmom.ml.nn.layers.ConvolutionalLayer object at 0x7fd5b3eca048>
# 08:<madmom.ml.nn.layers.BatchNormLayer object at 0x7fd5b3ec3da0>
# 09:<madmom.ml.nn.layers.MaxPoolLayer object at 0x7fd57228d208>
# 10:<madmom.ml.nn.layers.StrideLayer object at 0x7fd57228d278>
# 11:<madmom.ml.nn.layers.FeedForwardLayer object at 0x7fd57228d2b0>
# 12:<madmom.ml.nn.layers.BatchNormLayer object at 0x7fd57228d390>
# 13:<madmom.ml.nn.layers.FeedForwardLayer object at 0x7fd57228d588>
# 14:<madmom.ml.nn.layers.BatchNormLayer object at 0x7fd57228d6a0>
# 15:<madmom.ml.nn.layers.FeedForwardLayer object at 0x7fd57228d860>
