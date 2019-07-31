import librosa
import numpy as np


class CQT(object):
    """
    Load the track to be fed inside a NN
    """

    def __init__(self):
        # TODO: load the parameters externally
        self.sampleRate = 44100
        self.frameRate = 44100 / 512
        self.n_bins = 84
        self.fMin = 20

    def open(self, path: str):
        """
        Load an audio track and return an array numpy
        """
        y, sr = librosa.load(path, sr=self.sampleRate)
        # TODO: add the dB conversion
        cqts = np.abs(
            librosa.cqt(y, sr=sr, hop_length=int(np.round(sr / self.frameRate)), n_bins=self.n_bins, fmin=self.fMin))
        return cqts
