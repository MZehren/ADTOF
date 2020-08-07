import madmom
import numpy as np

from adtof.converters.converter import Converter
from adtof.io.textReader import TextReader


class MadmomBeatConverter(Converter):
    """
    Compute the beats of a track with Madmom
    """

    def convert(self, audioInputPath, beatsEstimationOutputPath, beatActivationOutputPath, transitionLambda=100, correctToActivation=True):

        fps = 100
        act = madmom.features.RNNDownBeatProcessor()(audioInputPath)
        accuAct = act[:, 0] + act[:, 1]  # accumulate beat and down-beat activation curves
        np.save(beatActivationOutputPath, accuAct)
        proc = madmom.features.DBNDownBeatTrackingProcessor(
            beats_per_bar=[3, 4], fps=fps, transition_lambda=transitionLambda, correct=correctToActivation
        )
        beats = proc(act)
        TextReader().writteBeats(beatsEstimationOutputPath, beats)
