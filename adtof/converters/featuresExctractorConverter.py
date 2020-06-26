import os
from adtof.converters.converter import Converter
from adtof.io.mir import MIR

import numpy as np


class FeaturesExtractorConverter(Converter):
    """
    Compute the beats of a track with Madmom
    """

    def convert(self, inputAudio, outputfeatures, fps=100):
        features = MIR().open(inputAudio)
        np.save(outputfeatures, features)
