import tensorflow as tf
import numpy as np


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
            loss=tf.compat.v2.nn.sigmoid_cross_entropy_with_logits,
            metrics=[PeakPicking()]
        )
        return model


class PeakPicking(tf.keras.metrics.Accuracy):

    def __init__(self, sampleRate=100, hitDistance=0.5):
        self.sampleRate = sampleRate
        self.hitDistance = hitDistance
        self.batch_y_true = []
        self.batch_y_pred = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.batch_y_true.append(y_true)
        self.batch_y_pred.append(y_pred)

    def result(self):
        peaks = self._statick_threshold_peak_picking(self.batch_y_pred)
        score = self._get_f_measure(peaks, self.batch_y_true, self.hitDistance * self.sampleRate)
        return score

    def _statick_threshold_peak_picking(self, values: List, threshold: float = 0.2, windowSize: int = 2):
        """
        return the peak positions in index
        """
        result = []
        for classValues in values:  # For all the classes
            peaksPosition = []
            mySortedList = sorted([(i, value) for i, value in enumerate(classValues)], key=lambda x: x[1], reverse=True)
            for i, value in mySortedList:  #For all the values by decreasing order
                if value >= threshold:
                    isMaximum = value == np.max(classValues[max(i - windowSize // 2, 0):i + windowSize // 2 + 1])
                    isAboveMean = value >= np.mean(classValues[max(i - windowSize // 2, 0):i + windowSize // 2 + 1]) + threshold
                    if isMaximum and isAboveMean:
                        peaksPosition.append(i)
                else:
                    break
            result.append(peaksPosition)
        return result

    def _get_f_measure(self, peaksIndexes: List, yTrue: List, distance):
        """
        peaksIndexes: Sparse list of the peaks in each class
        yTrue: Dense list of the truth values in each class
        """
        precision = []
        recall = []
        fMeasure = []
        for i, categoriePeaks in peaksIndexes:  #TODO: add sum metric as well?
            hits = len([1 for index in categoriePeaks if any([v for v in yTrue[i][max(index - distance, 0):index + distance]])])
            precision.append(hits / len(categoriePeaks))  #TODO: change that when the labels are weighted
            recall.append(hits / np.sum(yTrue[i]))
            fMeasure.append(2 * (precision * recall) / (precision + recall))

        return np.mean(fMeasure)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.batch_y_true = []
        self.batch_y_pred = []


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