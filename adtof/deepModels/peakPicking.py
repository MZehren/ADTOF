import tensorflow as tf
import numpy as np
import mir_eval


class PeakPicking(tf.keras.metrics.Metric):

    def __init__(self, name="peak_picking", sampleRate=50, hitDistance=0.5, **kwargs):
        super(tf.keras.metrics.Metric, self).__init__(name=name, **kwargs)
        self.sampleRate = sampleRate
        self.hitDistance = hitDistance
        self.batch_y_true = None  #self.add_weight(name='true', aggregation=tf.compat.v2.VariableAggregation.NONE)
        self.batch_y_pred = None  #self.add_weight(name='pred', aggregation=tf.compat.v2.VariableAggregation.NONE)
        self.batch_index = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Called when an iteration is done. Y_true is a mini_batch

        Compute the peaks of this batch in the format: TODO
        """
        dense_peaks_pred = self._dense_peak_picking(y_pred)

        offset = self.batch_index * len(y_true)
        self.batch_index += 1
        sparse_pred = [self.sparsify(dense_peaks_pred[:, drumClass], offset) for drumClass in range(y_pred.shape[1])]
        sparse_true = [self.sparsify(y_true[:, drumClass], offset) for drumClass in range(y_true.shape[1])]

        if self.batch_y_true is None:
            self.batch_y_pred = sparse_pred
            self.batch_y_true = sparse_true
        else:
            self.batch_y_pred = [self.batch_y_pred[i] + sparse_pred[i] for i in range(len(sparse_pred))]
            self.batch_y_true = [self.batch_y_true[i] + sparse_true[i] for i in range(len(sparse_true))]

    def result(self):
        score = self._get_f_measure(self.batch_y_pred, self.batch_y_true, self.hitDistance * self.sampleRate)
        return score

    def _dense_peak_picking(self, values, threshold: float = 0.2, windowSize: int = 2):
        """
        Return a dense representation of the peaks
        from [0,10,1,0]
        return [False, True, False, False]


        a peak must be the maximum value within a window of size m + 1 (i.e.: fa(n) = max(fa(n − m), · · · , fa(n))), 
        and exceeding the mean value plus a threshold δ within a window of size a + 1 (i.e.: fa(n) ≥ mean(fa(n − a), · · · , fa(n)) +δ). 
        Additionally, a peak must have at least a distance of w + 1 to the last detected peak nlp (i.e.: n − nlp > w,).
        The parameters for peak picking are the same as used in [1]: m = a = w = 2. 

        appropriately trained DNNs produce spiky activation functions, therefore, low thresholds (0.1 − 0.2) give best results.
        """
        # https://www.tensorflow.org/api_docs/python/tf/math/unsorted_segment_max ?
        maximised = tf.constant(True, shape=values.shape)
        for i in range(windowSize):
            segments = tf.constant([(j + i) // windowSize for j in range(len(values))])
            # TODO: is there a better way to get the peaks of a overlaping segments?
            segmentsMax = tf.math.segment_max(values, segments)
            argSegmentsMax = [[segmentsMax[r // windowSize][c] == values[r][c] for c, _ in enumerate(values[r])] for r, _ in enumerate(values)]
            maximised = tf.math.logical_and(argSegmentsMax, maximised)

        thresholded = tf.math.greater(values, threshold)

        # TODO is min distance peaks usefull ?
        return tf.math.logical_and(thresholded, maximised)

    def sparsify(self, denseVector, offset):
        return [i + offset for i, v in enumerate(denseVector) if v]

    def _get_f_measure(self, peaksIndexesPred, peaksIndexesTrue, distance):
        """
        peaksIndexes: Sparse list of the peaks in each class
        yTrue: Dense list of the truth values in each class
        """
        F = 0
        for drumClass in range(len(peaksIndexesPred)):
            f, p, r = mir_eval.onset.f_measure(np.array(peaksIndexesPred[drumClass]), np.array(peaksIndexesTrue[drumClass]), window=distance)
            F += f

        return F / len(peaksIndexesPred)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.batch_y_true = None
        self.batch_y_pred = None
        self.batch_index = 0
