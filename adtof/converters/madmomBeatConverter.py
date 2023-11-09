import madmom
import numpy as np

from adtof.converters.converter import Converter
from adtof.io.textReader import TextReader


class MadmomBeatConverter(Converter):
    """
    Compute the beats of a track with Madmom
    """

    def convert(
        self,
        audioInputPath,
        missalignedMidiInput,
        beatsEstimationOutputPath,
        beatActivationOutputPath,
        transitionLambda=100,
        correctToActivation=True,
    ):

        fps = 100
        # tempi = PrettyMidiWrapper(missalignedMidiInput).get_tempo_changes()[1]
        # maxBpm = max(tempi) + 20
        # minBpm = min(tempi) - 20
        # MIN_BPM = 55.
        # MAX_BPM = 215.

        act = madmom.features.RNNDownBeatProcessor()(audioInputPath)
        accuAct = act[:, 0] + act[:, 1]  # accumulate beat and down-beat activation curves
        if beatActivationOutputPath is not None:
            np.save(beatActivationOutputPath, accuAct)
        proc = madmom.features.DBNDownBeatTrackingProcessor(
            beats_per_bar=[3, 4],
            fps=fps,
            transition_lambda=transitionLambda,
            correct=correctToActivation,  # TODO: max_bpm=maxBpm, min_bpm=minBpm,
        )
        beats = proc(act)
        TextReader().writteBeats(beatsEstimationOutputPath, beats)
