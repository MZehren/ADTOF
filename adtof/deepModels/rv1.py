import tensorflow as tf


class RV1(object):
    """
    Richard Vogl model
    http://ifs.tuwien.ac.at/~vogl/
    """

    # def __init__(self):
    #     # self.model = self.createModel()
    #     pass

    def createModel(self, context=25, n_bins=256, output=5):
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

        # model = tf.keras.Sequential([
        #     # Adds a densely-connected layer with 64 units to the model:
        #     tf.keras.layers.Dense(128, activation='relu', input_shape=(n_bins,)),
        #     # Add another:
        #     tf.keras.layers.Dense(64, activation='relu'),
        #     # Add a softmax layer with 10 output units:
        #     tf.keras.layers.Dense(output, activation='softmax')
        # ])

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(context, n_bins, 1)))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(output, activation='softmax'))

        # model = tf.keras.Sequential([
        #     tf.keras.layers.Conv2D(
        #         32,
        #         (3, 3),
        #         input_shape=(context, n_bins, 1),
        #     ),
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
        #     # tf.keras.layers.Bidirectional(tf.keras.layers.GRU(60)),
        #     # tf.keras.layers.Bidirectional(tf.keras.layers.GRU(60)),
        #     # tf.keras.layers.Bidirectional(tf.keras.layers.GRU(60)),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(256, activation=tf.keras.activations.relu),
        #     tf.keras.layers.Dense(output, activation=tf.keras.activations.sigmoid)
        # ])

        model.compile(
            optimizer="adam",  #tf.keras.optimizers.RMSprop(learning_rate=0.001),
            loss=tf.compat.v2.nn.sigmoid_cross_entropy_with_logits, metrics=['accuracy'])
        return model


# rv1 = RV1()