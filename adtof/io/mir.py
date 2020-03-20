import librosa
import madmom
import matplotlib.pyplot as plt
import numpy as np


class MIR(object):
    """
    Load the track to be fed inside a NN
    """

    def __init__(self, frameRate=100, frameSize=2048):
        # TODO: load the parameters externally
        self.sampleRate = 44100
        self.frameRate = frameRate
        self.frameSize = frameSize
        self.n_bins = 84
        self.fMin = 32.70  #20

    def open(self, path: str):
        """
        Load an audio track and return an array numpy
        """
        return self.openMadmom(path)

    def openMadmom(self, path: str):
        """
        
        """
        # Log spec
        hopSize = self.sampleRate / self.frameRate
        spec = madmom.audio.FilteredSpectrogram(
            path, sample_rate=self.sampleRate, frame_size=self.frameSize, hop_size=hopSize, num_channels=1, fmax=20000
        )
        # norm
        max = np.max(spec)
        min = np.min(spec)
        spec = (spec - min) / (max - min)
        # stack diff
        # diff = np.diff(spec, axis=0)
        # spec = spec[1:]
        diff = madmom.audio.spectrogram.SpectrogramDifference(spec)
        diff = np.abs(diff)  #diff.clip(min=0) #np.abs(diff)  # (diff + 1) /2
        result = np.concatenate((spec, diff), axis=1)
        return result

    def openLibrosa(self, path: str):
        """
        
        """
        # TODO: change from librosa and use ffmpeg instead, or sox ?
        # TODO: Given a mono audio input signal, sampled at 44.1 kHz,
        # the input representation is derived from a set of log magnitude spectrograms which are grouped to have approximately
        # logarithmic frequency spacing between adjacent bins. Three
        # such spectrograms are calculated at a fixed hop size of 10 ms
        # with increasing window sizes of 23.2 ms, 46.4 ms and 92.9 ms.
        # From each, the per-bin first-order difference spectorgram is
        # calculated, where only the positive differences are retained to
        # capture the energy rise in individual frequency bands.

        y, sr = librosa.load(path, sr=self.sampleRate)
        # TODO: add 0.25s of zero padding at the start for instant onsets
        assert (sr / self.frameRate).is_integer()
        result = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=int(np.round(sr / self.frameRate)))
        result = librosa.amplitude_to_db(result).T
        diff = np.diff(result, axis=0)
        result = result[1:]
        result = np.concatenate((result, diff), axis=1)
        # self.viz(result)

        max = np.max(result)
        min = np.min(result)
        result = (result - min) / (max - min)
        return result

    def openCQT(self, path: str):
        """
        Load an audio track and return an array numpy
        """
        # requirement of k*2**(n_octaves - 1) exists so that recursive downsampling inside CQT retains frame alignment.
        # hop length = 448 or frameRate = 98.4375Hz
        y, sr = librosa.load(path, sr=self.sampleRate)
        # TODO: add 0.25s of zero padding at the start for instant onsets
        cqt = librosa.cqt(y, sr=sr, hop_length=int(np.round(sr / self.frameRate)), n_bins=self.n_bins, fmin=self.fMin)
        linear_cqt = np.abs(cqt)
        # freqs = librosa.cqt_frequencies(linear_cqt.shape[0], fmin=self.fMin)
        # result = librosa.perceptual_weighting(linear_cqt**2, freqs, ref=np.max)
        # result += np.min(result) * -1
        result = librosa.amplitude_to_db(linear_cqt)

        # max = np.max(result)
        # min = np.min(result)
        # result = (result - min) / (max - min)
        # self.viz(result.T)

        return result.T

    def viz(self, result):
        """debug viz
        
        Arguments:
            result {2D matrix} -- the stft to display
        """
        plt.matshow(result)
        plt.show()
