import logging
import pickle

import librosa
import madmom
import matplotlib.pyplot as plt
import numpy as np
from adtof.converters.converter import Converter


class MIR(object):
    """
    Load the track to be fed inside a NN
    """

    def __init__(
        self, frameRate=100, frameSize=2048, diff=False, inputSampleRate=44100, n_bins=12, fmin=20, fmax=20000, normalize=False, **kwargs
    ):
        """
        Configure the parameters for the feature extraction
        """
        self.frameRate = frameRate
        self.frameSize = frameSize
        self.diff = diff
        self.sampleRate = inputSampleRate
        self.hopSize = int(self.sampleRate / self.frameRate)
        self.n_bins = n_bins  # Per octave
        self.fmin = fmin
        self.fmax = fmax
        self.normalize = normalize
        self.proc, self.diffProc = self.getMadmomProc()

    def getDim(self):
        """
        Compute the size of the vector returned by the 
        "log magnitude spectrogram followed by a logarithmic grouping of frequency bins"
        """
        fftFrequencies = madmom.audio.stft.fft_frequencies(self.frameSize // 2, self.sampleRate)
        targetFrequencies = madmom.audio.filters.log_frequencies(self.n_bins, self.fmin, self.fmax)
        # align to bins
        bins = madmom.audio.filters.frequencies2bins(targetFrequencies, fftFrequencies, unique_bins=True)
        filters = madmom.audio.filters.TriangularFilter.filters(bins, norm=True, overlap=True)

        if self.diff:
            return len(filters) * 2
        else:
            return len(filters)

    def open(self, audioPath: str, cachePath: str = None):
        """
        Load an audio track and return a numpy array
        """
        result = None
        if cachePath is not None and Converter.checkPathExists(cachePath):  # Getting the cached file
            try:
                with open(cachePath, "rb") as cacheFile:
                    result = pickle.load(cacheFile)
                # result = np.load(cachePath, allow_pickle=False)
            except Exception as e:
                logging.warn("Cache file %s failed to load\n%s", cachePath, e)

        else:  # Processing the file
            result = self.proc(audioPath)
            if cachePath is not None:
                try:
                    with open(cachePath, "wb") as cacheFile:
                        pickle.dump(result, cacheFile)
                    # np.save(cachePath, result, allow_pickle=False)
                except Exception as e:
                    logging.warning("Couldn't cache processed audio \n%s", e)

        # Normalizing the frequencies
        if self.normalize:
            max = np.max(result)
            min = np.min(result)
            result = (result - min) / (max - min)

        # Adding the diff
        if self.diff:
            result = self.diffProc(result)

        # Removing all the cached extra data from madmom
        result = np.array(result)

        # self.plot([result[:2500]])
        return result

    def plot(self, values):
        fig, ax = plt.subplots(len(values))
        for i, value in enumerate(values):
            if len(values) > 1:
                ax[i].matshow(value.T)
            else:
                ax.matshow(value.T)
        plt.show()

    def getMadmomProc(self):
        """Initiate the processor from the list of parameters in the class attributes
        mplementation based on Richard Vogl's http://www.ifs.tuwien.ac.at/~vogl/dafx2018/

        Returns
        -------
            callable (path) -> numpy array 
            
        """
        from madmom.audio.filters import LogarithmicFilterbank
        from madmom.audio.signal import FramedSignalProcessor, SignalProcessor
        from madmom.audio.spectrogram import LogarithmicFilteredSpectrogramProcessor, SpectrogramDifferenceProcessor
        from madmom.audio.stft import ShortTimeFourierTransformProcessor
        from madmom.processors import ParallelProcessor, SequentialProcessor

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
        # madmom needs a np.array with a stft field to compute the diff_frame. This field is lost when it is cached with np.save()
        # Thus I compute the thingy before hand
        diff_frames = madmom.audio.spectrogram._diff_frames(0.5, hop_size=self.hopSize, frame_size=self.frameSize, window=np.hanning)
        diff = SpectrogramDifferenceProcessor(diff_frames=diff_frames, positive_diffs=True, stack_diffs=np.hstack)
        return SequentialProcessor((sig, frames, stft, spec)), SequentialProcessor((diff,))

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
