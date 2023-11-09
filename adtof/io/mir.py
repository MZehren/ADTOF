from distutils.command.config import config
import logging
import pickle

import librosa
import madmom
import matplotlib.pyplot as plt
import numpy as np
from adtof import config

from madmom.audio.filters import LogarithmicFilterbank
from madmom.audio.signal import FramedSignalProcessor, SignalProcessor
from madmom.audio.spectrogram import LogarithmicFilteredSpectrogramProcessor, SpectrogramDifferenceProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.processors import ParallelProcessor, SequentialProcessor
import copy


def preProcess(audioPath: str, cachePath: str = None, **kwargs):
    """
    Load an audio track and return a numpy array
    """
    result = None
    if cachePath is not None and config.checkPathExists(cachePath):  # Getting the cached file
        try:
            with open(cachePath, "rb") as cacheFile:
                result = pickle.load(cacheFile)
            # result = np.load(cachePath, allow_pickle=False)
        except Exception as e:
            logging.warn("Cache file %s failed to load\n%s", cachePath, e)

    if result is None:  # Processing the file
        # result = openLibrosa(audioPath, **kwargs)
        result = openMadmom(audioPath, **kwargs)
        if cachePath is not None:
            try:
                with open(cachePath, "wb") as cacheFile:
                    pickle.dump(result, cacheFile)
                # np.save(cachePath, result, allow_pickle=False)
            except Exception as e:
                logging.warning("Couldn't cache processed audio \n%s", e)

    # Removing all the cached extra data from madmom
    result = np.array(result)

    return result


def getDim(bandsPerOctave=12, fmin=20, fmax=20000, frameSize=2048, trackSampleRate=44100, **kwargs):
    # Librosa
    # return int(bandsPerOctave * np.log2(fmax / fmin))

    # Madmom
    fftFrequencies = madmom.audio.stft.fft_frequencies(frameSize // 2, trackSampleRate)
    targetFrequencies = madmom.audio.filters.log_frequencies(bandsPerOctave, fmin, fmax)
    # align to bins
    bins = madmom.audio.filters.frequencies2bins(targetFrequencies, fftFrequencies, unique_bins=True)
    filters = madmom.audio.filters.TriangularFilter.filters(bins, norm=True, overlap=True)

    return len(filters)


def openLibrosa(
    audioPath: str, trackSampleRate=44100, sampleRate=100, bandsPerOctave=12, fmin=20, fmax=20000, window="hann", pitchScale="mel", magnitudeScale="log", n_channels=1, **kwargs
):
    # Load audio
    if n_channels == 1:
        mono = True
    elif n_channels == 2:
        mono = False
    else:
        raise ValueError("Invalid number of channels")

    Y, sr = librosa.load(audioPath, sr=trackSampleRate, mono=mono)
    # print(Y)
    if mono:
        Y = Y.reshape((1,) + Y.shape)

    # Compute the spectrogram
    hop_length = int(np.round(sr / sampleRate))
    dim = getDim(bandsPerOctave=bandsPerOctave, fmin=fmin, fmax=fmax)
    if pitchScale == "mel":
        S = np.array(
            [
                librosa.feature.melspectrogram(
                    y=y,
                    sr=sr,
                    hop_length=hop_length,
                    n_mels=dim,
                    fmin=fmin,
                    fmax=fmax,
                    window=window,
                    power=1,
                    # norm="slaney",
                    # htk=True,
                )
                for y in Y
            ]
        )
    # elif pitchScale == "linear":
    #     S = np.abs(librosa.stft(y=y, hop_length=hop_length, window=window, n_fft=dim * 2))
    # elif pitchScale == "cqt":
    #     S = np.abs(librosa.cqt(y=y, sr=sr, hop_length=512, n_bins=dim, bins_per_octave=bandsPerOctave, fmin=fmin, window=window))
    else:
        raise ValueError("Invalid pitch scale")

    # Scale the spectrogram
    if magnitudeScale == "log":
        S = np.log10(S + 1)
    elif magnitudeScale == "db":
        S = librosa.amplitude_to_db(S)
    elif magnitudeScale == None:
        pass
    else:
        raise ValueError("Invalid magnitude scale")

    # set the dimensions to (time, frequency, channel)
    S = S.T
    return S


def openMadmom(audioPath: str, sampleRate=100, frameSize=2048, inputSampleRate=44100, bandsPerOctave=12, fmin=20, fmax=20000, normalize=False, n_channels=1, **kwargs):
    """
    Implementation based on Richard Vogl's http://www.ifs.tuwien.ac.at/~vogl/dafx2018/
    """
    # Create processors
    sig = SignalProcessor(num_channels=n_channels, sample_rate=inputSampleRate)
    frames = FramedSignalProcessor(frame_size=frameSize, fps=sampleRate)
    stft = ShortTimeFourierTransformProcessor()
    spec = LogarithmicFilteredSpectrogramProcessor(
        num_channels=1,
        sample_rate=inputSampleRate,
        filterbank=LogarithmicFilterbank,
        frame_size=frameSize,
        fps=sampleRate,
        num_bands=bandsPerOctave,
        fmin=fmin,
        fmax=fmax,
        norm_filters=True,
    )

    # Call processors
    if n_channels == 1:
        S = SequentialProcessor((sig, frames, stft, spec))(audioPath)
        S = S.reshape((S.shape[0], S.shape[1], 1))
    elif n_channels == 2:
        monoProc = SequentialProcessor((frames, stft, spec))
        Y = SequentialProcessor((sig,))(audioPath)
        S = np.array([monoProc(Y[:, channel]).T for channel in range(n_channels)]).T

    return S


def viz(S, y=None, timeLim=1000):
    """
    plot a matrix of dimension (time, frequency, channel)
    """
    # Create subplots
    fig, axs = plt.subplots(S.shape[2], 1, figsize=(10, 10), squeeze=False)
    for channel in range(S.shape[2]):
        axs[channel][0].imshow(S[:, :, channel].T, origin="lower", aspect="auto", interpolation="none")
        axs[channel][0].set_title("Channel {}".format(channel))

        if y is not None:
            axs[channel][0].plot(y * 80, alpha=0.8)

    plt.show()
