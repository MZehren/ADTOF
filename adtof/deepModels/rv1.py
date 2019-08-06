import tensorflow as tf


class RV1(object):
    """
    Richard Vogl model
    http://ifs.tuwien.ac.at/~vogl/
    """

    # def __init__(self):
    #     # self.model = self.createModel()
    #     pass

    def createModel(self, context = 25, n_bins=84, output=5):
        """
        TODO
        """
        # When to apply the dropout?
        # How to handle the bidirectional aggregation ? Sum, or nothing ?
        # How to handle the context for the learning 400 samples before learning?
        

        # model = tf.keras.Sequential([
        #     # Adds a densely-connected layer with 64 units to the model:
        #     tf.keras.layers.Dense(128, activation='relu', input_shape=(n_bins,)),
        #     # Add another:
        #     tf.keras.layers.Dense(64, activation='relu'),
        #     # Add a softmax layer with 10 output units:
        #     tf.keras.layers.Dense(output, activation='softmax')
        # ])
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), input_shape=(context, n_bins, 1), ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(64, (3, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
            tf.keras.layers.Dropout(0.3),
            # tf.keras.layers.Bidirectional(tf.keras.layers.GRU(60)),
            # tf.keras.layers.Bidirectional(tf.keras.layers.GRU(60)),
            # tf.keras.layers.Bidirectional(tf.keras.layers.GRU(60)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(output)
        ])

        model.compile(optimizer="adam", loss=tf.compat.v2.nn.sigmoid_cross_entropy_with_logits)
        return model

    def train(self, X, Y):
        pass


# rv1 = RV1()