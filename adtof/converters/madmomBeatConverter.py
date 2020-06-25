import os
from adtof.converters.converter import Converter
from adtof.io.textReader import TextReader
import madmom


class MadmomBeatConverter(Converter):
    """
    Compute the beats of a track with Madmom
    """

    def convert(self, inputFile, outputFile, transitionLambda=100, correctToActivation=True):

        fps = 100
        act = madmom.features.RNNDownBeatProcessor()(inputFile)
        proc = madmom.features.DBNDownBeatTrackingProcessor(
            beats_per_bar=[3, 4], fps=fps, transition_lambda=transitionLambda, correct=correctToActivation
        )
        beats = proc(act)
        TextReader().writteBeats(outputFile, beats)

