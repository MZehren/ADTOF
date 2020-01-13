import tensorflow as tf


class RV1TF(object):
    """
    Richard Vogl model
    http://ifs.tuwien.ac.at/~vogl/
    """

    # def __init__(self):
    #     # self.model = self.createModel()
    #     pass

    def createModel(self, context=25, n_bins=168, output=5):
        """Return a ts model based 
        
        Keyword Arguments:
            context {int} -- [description] (default: {25})
            n_bins {int} -- [description] (default: {84})
            output {int} -- number of classes in the output (should be the events: 36, 40, 41, 46, 49) (default: {5})
            outputWeight {list} --  (default: {[]}) 
        
        Returns:
            [type] -- [description]
        """
        # When to apply the dropout?
        # How to handle the bidirectional aggregation ? Sum, or nothing ?
        # How to handle the context for the learning 400 samples before learning?

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), input_shape=(context, n_bins, 1), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
            tf.keras.layers.Dropout(0.3),
            # tf.keras.layers.Bidirectional(tf.keras.layers.GRU(60)),
            # tf.keras.layers.Bidirectional(tf.keras.layers.GRU(60)),
            # tf.keras.layers.Bidirectional(tf.keras.layers.GRU(60)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(256, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(output, activation=tf.keras.activations.sigmoid)
        ])

        model.compile(
            optimizer="adam",  #tf.keras.optimizers.RMSprop(learning_rate=0.001),
            loss=tf.nn.sigmoid_cross_entropy_with_logits,  #tf.keras.backend.binary_crossentropy, 
            metrics=["accuracy", tf.keras.metrics.BinaryAccuracy()]  # PeakPicking()
        )
        return model


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
