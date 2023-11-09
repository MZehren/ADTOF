import random
import numpy as np
import tensorflow as tf
import librosa
from scipy.interpolate import interp1d
from adtof import config

# From Jacques and Robel
# Gaussian noise
# Remix noise
# Remix attacks
# Transposition with and without time compensation

# From Cartwright an Bello
# Varying background and pitch
# white noise


# From Callender et al.
# Mixup: randomly select pairs of examples,
# repeating the shorter of examples until it was as long as the longer one
# then mixing their audio samples and underlying MIDI together (prior to STFT)
# Shuffle mixup: split the examples in 1s chunks, splice them together in random order
# shuffles the order AND mixes examples together
def _reverseStereo(x):
    """Inverse left and right channels from a spectrogram"""
    return tf.reverse(x, axis=[-1])


def _resize(npArray, lenght):
    """
    Resize a spectrogram to a given length
    either truncate, or pad with zeros
    """
    if len(npArray) > lenght:
        return npArray[:lenght]
    elif len(npArray) < lenght:
        npad = [(0, 0)] * npArray.ndim
        npad[0] = (0, lenght - len(npArray))
        return np.pad(npArray, npad, mode="constant", constant_values=0)
    else:
        return npArray


def _stretchInTime(x, y, w, speedUpRate):
    """Stretch a spectrogram in time"""
    originalLength = len(x)
    # x
    if isinstance(x, tf.Tensor):
        x = x.numpy()
    x = np.stack([librosa.phase_vocoder(x[:, :, channel].T, rate=speedUpRate).T for channel in range(x.shape[-1])], axis=-1)
    x = _resize(x, originalLength)
    # y
    xp = np.arange(0, len(y), speedUpRate)
    nearest = interp1d(np.arange(len(y)), y, axis=0, kind="nearest", fill_value="extrapolate")
    y = nearest(xp)
    y = _resize(y, originalLength)

    # w
    nearest = interp1d(np.arange(len(w)), w, axis=0, kind="nearest", fill_value="extrapolate")
    w = nearest(xp)
    w = _resize(w, originalLength)
    return x, y, w


def _mixup(sample1, sample2, alpha):
    """Mix two samples together from Zhang et al."""
    lam = np.random.beta(alpha, alpha)
    sample = [lam * sample1[i] + (1 - lam) * sample2[i] for i in range(len(sample1))]
    return sample


def _shuffle(x, y, w, minDistance=50, instrumentToSplit=set([0, 1]), numSegments=4):
    """Shuffle the samples of one track"""
    # Find segments between onsets from instrumentToSplit with a minimum length of minDistance
    # onsets, inst = np.where(y == 1)
    # indexesWithOnset = [o for (o, i) in zip(onsets, inst) if i in instrumentToSplit]
    # indexesWithIsolatedOnset = [0] + [onset - 2 for onset, diff in zip(indexesWithOnset, np.diff([0] + list(indexesWithOnset))) if diff > minDistance] + [len(y)]
    # segments = [(a, b) for a, b in zip(indexesWithIsolatedOnset, indexesWithIsolatedOnset[1:])]

    # Segments evenly spaced
    boundaries = range(0, len(y), len(y) // numSegments)
    segments = [(a, b) for a, b in zip(boundaries, list(boundaries[1:]) + [len(y)])]
    # Shuffle segments
    random.shuffle(segments)

    # Concatenate segments
    x = np.concatenate([x[a:b] for a, b in segments], axis=0)
    y = np.concatenate([y[a:b] for a, b in segments], axis=0)
    w = np.concatenate([w[a:b] for a, b in segments], axis=0)

    return x, y, w


def dataAugmentationGen(previousGen, reverseStereoProbability=0.5, stretchSTD=0.1, mixupAlpha=0.2, shuffleProbability=0.5, **kwargs):
    """return a generator augmenting the data from previousGen"""

    def gen():
        iterator = previousGen()
        for x, y, w in iterator:
            # Mixup
            if mixupAlpha > 0:
                x2, y2, w2 = next(iterator)
                x["x"], y, w = _mixup((x["x"], y, w), (x2["x"], y2, w2), mixupAlpha)

            # Shuffle
            if random.random() < shuffleProbability:
                x["x"], y, w = _shuffle(x["x"], y, w)

            # Reverse stereo
            if random.random() < reverseStereoProbability:
                x["x"] = _reverseStereo(x["x"])

            # Stretch in time
            speedUpRate = np.random.normal(1, stretchSTD)
            if speedUpRate != 1:
                x["x"], y, w = _stretchInTime(x["x"], y, w, speedUpRate)

            yield x, y, w

    return gen
