import tensorflow as tf
import numpy as np
import mir_eval


class PeakPicking(tf.keras.metrics.Metric):

    def __init__(self, name="peak_picking", sampleRate=50, hitDistance=0.5, **kwargs):
        super(tf.keras.metrics.Metric, self).__init__(name=name, **kwargs)
        self.sampleRate = sampleRate
        self.hitDistance = hitDistance
        self.batch_y_true = None #self.add_weight(name='true', aggregation=tf.compat.v2.VariableAggregation.NONE)
        self.batch_y_pred = None #self.add_weight(name='pred', aggregation=tf.compat.v2.VariableAggregation.NONE)

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.batch_y_true is None:
            self.batch_y_true = y_true
            self.batch_y_pred = y_pred
        else:
            self.batch_y_true = np.concatenate((self.batch_y_true, y_true))
            self.batch_y_pred = np.concatenate((self.batch_y_pred, y_pred))

    def result(self):
        peaks = self._dense_peak_picking(self.batch_y_pred)
        score = self._get_f_measure(peaks, self.batch_y_true, self.hitDistance * self.sampleRate)
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

    def sparsify(self, denseVector):
        return [i for i, v in enumerate(denseVector) if v]

    def _get_f_measure(self, peaksIndexes, yTrue, distance):
        """
        peaksIndexes: Sparse list of the peaks in each class
        yTrue: Dense list of the truth values in each class
        """
        F = 0
        for drumClass in range(peaksIndexes.shape[1]):
            estSparse = self.sparsify(peaksIndexes[:,drumClass])
            annotSparse = self.sparsify(yTrue[:, drumClass])
            f, p, r = mir_eval.onset.f_measure(np.array(estSparse), np.array(annotSparse), window=distance)
            F += f

        return F/peaksIndexes.shape[1]

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.batch_y_true = None
        self.batch_y_pred = None
