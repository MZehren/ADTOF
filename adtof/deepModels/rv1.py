import tensorflow as tf


class RV1(object):
    """
    Richard Vogl model
    http://ifs.tuwien.ac.at/~vogl/
    """

    # def __init__(self):
    #     # self.model = self.createModel()
    #     pass

    def createModel(self):
        """
        TODO
        """
        # When to apply the dropout?
        # How to handle the bidirectional aggregation ? Sum, or nothing ?
        # How to handle the context for the learning 400 samples before learning?
        #  TODO: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
        context = 25
        n_bins = 84
        output = 5
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(n_bins, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(output)
        ])
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Conv2D(32, (3, 3), input_shape=(context, n_bins)),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Conv2D(32, (3, 3)),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
        #     tf.keras.layers.Dropout(0.3),
        #     tf.keras.layers.Conv2D(64, (3, 3)),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Conv2D(64, (3, 3)),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
        #     tf.keras.layers.Dropout(0.3),
        #     tf.keras.layers.Bidirectional(tf.keras.layers.GRU(60)),
        #     tf.keras.layers.Bidirectional(tf.keras.layers.GRU(60)),
        #     tf.keras.layers.Bidirectional(tf.keras.layers.GRU(60)),
        #     tf.keras.layers.Dense(3)
        # ])

        model.compile(optimizer="adam", loss="sigmoid_cross_entropy_with_logits")
        return model

    def train(self, X, Y):
        pass


# rv1 = RV1()