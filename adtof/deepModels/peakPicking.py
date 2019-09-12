import tensorflow as tf
import numpy as np


class PeakPicking(tf.keras.metrics.Metric):

    def __init__(self, name="peak_picking", sampleRate=100, hitDistance=0.5, **kwargs):
        super(tf.keras.metrics.Metric, self).__init__(name=name, **kwargs)
        self.sampleRate = sampleRate
        self.hitDistance = hitDistance
        self.batch_y_true = self.add_weight(name='true', aggregation=tf.compat.v2.VariableAggregation.NONE)
        self.batch_y_pred = self.add_weight(name='pred', aggregation=tf.compat.v2.VariableAggregation.NONE)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.batch_y_true.append(y_true)
        self.batch_y_pred.append(y_pred)

    def result(self):
        peaks = self._statick_threshold_peak_picking(self.batch_y_pred)
        score = self._get_f_measure(peaks, self.batch_y_true, self.hitDistance * self.sampleRate)
        return score

    def _statick_threshold_peak_picking(self, values, threshold: float = 0.2, windowSize: int = 2):
        """
        return the peak positions in index
        TODO: only works in eager execution, or needs to be a tf function or something which means that the basic python functionalities can't be used 
        """
        result = []
        for classValues in values:  # For all the classes
            peaksPosition = []
            # mySortedList = sorted([(i, value) for i, value in enumerate(classValues)], key=lambda x: x[1], reverse=True)
            for i, value in enumerate(classValues):  #For all the values by decreasing order
                isMaximum = value == np.max(classValues[max(i - windowSize // 2, 0):i + windowSize // 2 + 1])
                isAboveMean = value >= np.mean(
                    classValues[max(i - windowSize // 2, 0):i + windowSize // 2 + 1]) + threshold
                if isMaximum and isAboveMean:
                    peaksPosition.append(i)
            result.append(peaksPosition)
        return result

    def _get_f_measure(self, peaksIndexes, yTrue, distance):
        """
        peaksIndexes: Sparse list of the peaks in each class
        yTrue: Dense list of the truth values in each class
        """
        precision = []
        recall = []
        fMeasure = []
        for i, categoriePeaks in enumerate(peaksIndexes):  #TODO: add sum metric as well?
            hits = len([
                1 for index in categoriePeaks if any([v for v in yTrue[i][max(index - distance, 0):index + distance]])
            ])
            precision.append(hits / len(categoriePeaks))  #TODO: change that when the labels are weighted
            recall.append(hits / np.sum(yTrue[i]))
            fMeasure.append(2 * (precision * recall) / (precision + recall))

        return np.mean(fMeasure)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.batch_y_true = []
        self.batch_y_pred = []
