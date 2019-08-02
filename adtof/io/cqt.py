import librosa
import numpy as np


class CQT(object):
    """
    Load the track to be fed inside a NN
    """

    def __init__(self):
        # TODO: load the parameters externally
        self.sampleRate = 44100
        # requirement of k*2**(n_octaves - 1) exists so that recursive downsampling inside CQT retains frame alignment.
        # hop length = 448 or frameRate = 98.4375Hz
        self.frameRate = 98.4375
        self.n_bins = 84
        self.fMin = 20

    def open(self, path: str):
        """
        Load an audio track and return an array numpy
        """
        y, sr = librosa.load(path, sr=self.sampleRate)
        # TODO: add the dB conversion
        # TODO: add 0.25s of zero padding at the start for instant onsets 
        # TODO: add first order difference
        cqts = np.abs(librosa.cqt(y, sr=sr, hop_length=int(np.round(sr / self.frameRate)), n_bins=self.n_bins, fmin=self.fMin))
        return cqts.T
