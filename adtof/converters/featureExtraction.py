from . import Converter
import librosa
import numpy as np
from automix.model.classes.signal import Signal


class FeatureExtraction(Converter):
    """
    Load the track to be fed inside a NN
    """

    def __init__(self):
        self.sampleRate = 44100
        self.frameRate = 44100/512
        self.n_bins = 84
        self.fMin = 20

    def open(self, path):
        y, sr = librosa.load(path, sr=self.sampleRate)
        cqts = np.abs(librosa.cqt(y, sr=sr, hop_length=int(np.round(sr/self.frameRate)), n_bins=self.n_bins, fmin=self.fMin))
        return Signal(cqts, sampleRate=self.frameRate)