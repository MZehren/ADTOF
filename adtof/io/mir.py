import os

import madmom
import matplotlib.pyplot as plt
import numpy as np
from adtof import config


class MIR(object):
    """
    Load the track to be fed inside a NN
    """

    def __init__(self, frameRate=100, frameSize=2048):
        """
        Configure the parameters for the feature extraction
        """
        self.frameRate = frameRate
        self.frameSize = frameSize
        self.sampleRate = 44100
        self.hopSize = int(self.sampleRate / self.frameRate)
        self.n_bins = 12  # Per octave
        self.fmin = 20
        self.fmax = 20000
        self.logarithmicMagnitude = True
        self.diff = False

        self.proc = self.getMadmomProc()

    def open(self, path: str):
        """
        Load an audio track and return a numpy array
        """
        return self.proc(path)

    def plot(self, values):
        fig, ax = plt.subplots(len(values))
        for i, value in enumerate(values):
            ax[i].matshow(value.T)
        plt.show()

    def getMadmomProc(self):
        """Initiate the processor from the list of parameters in the class attributes
        mplementation based on Richard Vogl's http://www.ifs.tuwien.ac.at/~vogl/dafx2018/

        Returns
        -------
            callable (path) -> numpy array 
            
        """
        from madmom.audio.spectrogram import LogarithmicFilteredSpectrogramProcessor, SpectrogramDifferenceProcessor
        from madmom.audio.filters import LogarithmicFilterbank
        from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
        from madmom.audio.stft import ShortTimeFourierTransformProcessor
        from madmom.processors import SequentialProcessor, ParallelProcessor

        sig = SignalProcessor(num_channels=1, sample_rate=self.sampleRate)
        frames = FramedSignalProcessor(frame_size=self.frameSize, fps=self.frameRate)
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        spec = LogarithmicFilteredSpectrogramProcessor(
            num_channels=1,
            sample_rate=self.sampleRate,
            filterbank=LogarithmicFilterbank,
            frame_size=self.frameSize,
            fps=self.frameRate,
            num_bands=self.n_bins,
            fmin=self.fmin,
            fmax=self.fmax,
            norm_filters=True,
        )
        if self.diff:
            diff = SpectrogramDifferenceProcessor(diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
            return SequentialProcessor((sig, frames, stft, spec, diff))
        else:
            return SequentialProcessor((sig, frames, stft, spec))

    def openMadmomOld(self, path: str):
        """
        follow Vogl article
        """
        # Log spec

        spec = madmom.audio.FilteredSpectrogram(
            path,
            sample_rate=self.sampleRate,
            frame_size=self.frameSize,
            hop_size=self.hopSize,
            fmin=self.fmin,
            fmax=self.fmax,
            num_channels=1,
            # log=self.logarithmicMagnitude
            # dtype="int16",
        )

        # norm
        # max = np.max(spec)
        # min = np.min(spec)
        # spec = (spec - min) / (max - min)

        # stack diff
        diff = madmom.audio.spectrogram.SpectrogramDifference(spec, diff_frames=1, positive_diffs=True)
        result = np.concatenate((spec, diff), axis=1)
        return result

    def openLibrosa(self, path: str):
        """
        
        """
        raise DeprecationWarning()
        # TODO: change from librosa and use ffmpeg instead, or sox ?
        # TODO: Given a mono audio input signal, sampled at 44.1 kHz,
        # the input representation is derived from a set of log magnitude spectrograms which are grouped to have approximately
        # logarithmic frequency spacing between adjacent bins. Three
        # such spectrograms are calculated at a fixed hop size of 10 ms
        # with increasing window sizes of 23.2 ms, 46.4 ms and 92.9 ms.
        # From each, the per-bin first-order difference spectorgram is
        # calculated, where only the positive differences are retained to
        # capture the energy rise in individual frequency bands.
        import librosa

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
