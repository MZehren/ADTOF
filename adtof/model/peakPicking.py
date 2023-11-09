from collections import defaultdict

import madmom
import mir_eval
import numpy as np
import tensorflow as tf
from adtof.model import eval
from adtof import config


class PeakPicking(object):
    def getParameters(self):
        """
        Get the peak picking parameters
        """
        if isinstance(self.processors, list):
            return [p.threshold for p in self.processors]
        else:
            return self.processors.threshold

    def setParameters(self, peakThreshold, sampleRate=100, **kwargs):
        """
        Set the peak picking parameters as a scalar (same peak picking threshold is used for all labels))
        or a vector (For each label, the value of the corresponding index is used as peak picking threshold).
        """
        if isinstance(peakThreshold, list):
            self.processors = [
                madmom.features.notes.NotePeakPickingProcessor(threshold=t, smooth=0, pre_avg=0.1, post_avg=0.01, pre_max=0.02, post_max=0.01, combine=0.02, fps=sampleRate)
                for t in peakThreshold
            ]
        else:
            self.processors = madmom.features.notes.NotePeakPickingProcessor(
                threshold=peakThreshold, smooth=0, pre_avg=0.1, post_avg=0.01, pre_max=0.02, post_max=0.01, combine=0.02, fps=sampleRate
            )

    def fitIndependentLabel(self, X, Y, labels=config.LABELS_5, **kwargs) -> (float):
        """
        Fit the peak picking parameters for each label independently (i.e. a different threshold is used for BD, SD, HH, TT, and CY)
        Calls fit() for each label under the hood.

        Returns the best score achieved
        TODO: it happened that fitIndependent returned a peakPickingTarget score lower than fit. This means there is a bug
        TODO: Fit should not be inside the peak picking class. Separation of concerns
        """
        bests = []
        for i, label in enumerate(labels):  # Get the best threshold per label
            labelX = np.array([[[time[i]] for time in track] for track in X])  # Fit only on the class data
            labelY = [{k: v for k, v in track.items() if k == label} for track in Y]
            labelScore = self.fit(labelX, labelY, labels=[label], **kwargs)
            bests.append(self.getParameters())
        # Get the overall score for the set of thresholds
        self.setParameters(bests)
        return self.score(X, Y, labels=labels, **kwargs)

    def fit(self, X, Y, parameterGrid=np.arange(0.1, 0.9, 0.02), peakPickingTarget="sum F all", **kwargs) -> (float):
        """
        Find the best peak picking parameter from the parameterGrid with a simple grid search
        """
        results = []
        for parameters in parameterGrid:
            self.setParameters(parameters)
            score = self.score(X, Y, octave=False, **kwargs)
            results.append(score)

        bestI = max(range(len(results)), key=lambda i: results[i][peakPickingTarget])
        self.setParameters(parameterGrid[bestI])
        return results[bestI]

    def score(self, X, Y, labels=config.LABELS_5, octave=True, **kwargs) -> float:
        """
        Predict X and eval on Y
        """
        pred = self.predict(X, labels=labels, **kwargs)
        result = eval.runEvaluation(Y, pred, classes=labels)
        if octave:
            result.update({k + " octave": v for k, v in eval.runEvaluation(Y, pred, classes=labels, octave=True).items()})
        return result

    def predict(self, X, labelOffset=0, sampleRate=100, labels=config.LABELS_5, **kwargs):# -> dict[int, list[float]]:
        """
        Predict X with the parameters set with SetParameters() or fit().
        TODO The peak picking should't be aware of the conversion to second. This is an issue of the model to convert back to the real time/
        """
        # Get the peakpicking process for each label, or use the one provided for all labels
        processList = self.processors if isinstance(self.processors, list) and len(labels) == len(self.processors) else [self.processors for i in range(len(labels))]
        timeOffset = labelOffset / sampleRate
        # Process each label estimation independently and retrieve the onsets times
        return [{labels[i]: [time + timeOffset for time, pitch in p.process(np.reshape(np.array(x)[:, i], (-1, 1)))] for i, p in enumerate(processList)} for x in X]


class TatumPeakPicking(PeakPicking):
    def predict(self, X, tatumsTime=None, labels=config.LABELS_5, peakMinDistance=1, **kwargs):# -> dict[int, list[float]]:
        results = []
        thresholds = self.getParameters() if isinstance(self.getParameters(), list) else [self.getParameters() for i in range(len(labels))]
        for x, tracktatumsTime in zip(X, tatumsTime):
            x = np.array(x).T
            result = {}
            for i, _ in enumerate(labels):
                result[labels[i]] = [
                    tracktatumsTime[tatum]
                    for tatum, prediction in enumerate(x[i])
                    if prediction > thresholds[i] and prediction == max(x[i][max(tatum - peakMinDistance, 0) : tatum + peakMinDistance + 1])
                ]
            results.append(result)
        return results
