import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from adtof import config
from adtof.io.converters.converter import Converter
from adtof.io.mir import MIR
from adtof.io.myMidi import MidiProxy


class TorchIterableDataset(torch.utils.data.IterableDataset):

    def __init__(
        self, folderPath, sampleRate=50, context=25, midiLatency=0, classWeight=[2 / 16, 8 / 16, 16 / 16, 2 / 16, 4 / 16], train=True, split=0.8
    ):
        """
        TODO: change the sampleRate to 100 Hz?
        sampleRate = if the highest speed is a note each 20ms,    
                    the sr should be 1/0.02=50 
        context = how many frames are given with each samples
        midiLatency = how many frames the onsets are offseted to make sure that the transient is not discarded
        """
        super().__init__()
        tracks = config.getFilesInFolder(folderPath, config.AUDIO)
        midis = config.getFilesInFolder(folderPath, config.MIDI_CONVERTED)
        alignments = config.getFilesInFolder(folderPath, config.MIDI_ALIGNED)

        if train:
            self.tracks = tracks[:int(len(tracks) * split)]
            self.midis = midis[:int(len(midis) * split)]
            self.alignments = alignments[:int(len(alignments) * split)]
        else:
            self.tracks = tracks[int(len(tracks) * split):]
            self.midis = midis[int(len(tracks) * split):]
            self.alignments = alignments[int(len(tracks) * split):]
        self.mir = MIR(frameRate=sampleRate)
        self.X = {}
        self.Y = {}
        self.sampleRate = sampleRate
        self.midiLatency = midiLatency
        self.context = context

    def __iter__(self):
        i = random.randrange(len(self.tracks))
        if i not in self.X:
            track = self.tracks[i]
            midi = self.midis[i]
            alignment = self.alignments[i]

            alignmentInput = pd.read_csv(alignment, escapechar=" ")
            # y = MidiProxy(midi).getDenseEncoding(sampleRate=self.sampleRate)
            # TODO apply the offset correction
            y = MidiProxy(midi).getDenseEncoding(sampleRate=sampleRate, offset=-alignmentInput.offset[0], playback=1 / alignmentInput.playback[0])
            x = self.mir.open(track)

            for rowI, row in enumerate(y):
                if max(row) == 1:
                    firstNoteIdx = rowI
                    break

            self.X[i] = x[firstNoteIdx - self.midiLatency:min(len(y) - 1, len(x) - self.context - 1) + self.context - self.midiLatency]
            self.Y[i] = y[firstNoteIdx:min(len(y) - 1, len(x) - self.context - 1)]

        j = random.randrange(len(self.Y[i]))
        # for j in range(len(self.Y[i])):
        yield self.X[i][j:j + self.context], self.Y[i][j]
        # TODO change the track for each minibatch


def readTrack(i, tracks, midis, alignments, sampleRate=50, context=25, midiLatency=0):
    """
    Read the mp3, the midi and the alignment files and generate a balanced list of samples 
    """
    # initiate vars
    print("new track read:", i)
    track = tracks[i]
    midi = midis[i]
    alignment = alignments[i]
    mir = MIR(frameRate=sampleRate)

    # read files
    alignmentInput = pd.read_csv(alignment, escapechar=" ")
    y = MidiProxy(midi).getDenseEncoding(sampleRate=sampleRate, offset=-alignmentInput.offset[0], playback=1 / alignmentInput.playback[0])
    x = mir.open(track)
    x = x.reshape(x.shape + (1, ))  # Add the channel dimension TODO: remove?

    # Trim before the first midi note and after the last uncovered part
    for rowI, row in enumerate(y):
        if max(row) == 1:
            firstNoteIdx = rowI
            break
    lastSampleIdx = min(len(y) - 1, len(x) - context - 1)
    X = x[firstNoteIdx - midiLatency:lastSampleIdx + context - midiLatency]
    Y = y[firstNoteIdx:lastSampleIdx]
    return X, Y


def balanceDistribution(X, Y):
    """ 
    balance the distribution of the labels Y by removing the labels without events such as there is only half of them empty.
    """
    nonEmptyIndexes = [i for i, row in enumerate(Y) if np.max(row) == 1]
    emptyIndexes = [(nonEmptyIndexes[i] + nonEmptyIndexes[i + 1]) // 2 for i in range(len(nonEmptyIndexes) - 1)]
    idxUsed = np.array(list(zip(nonEmptyIndexes, emptyIndexes))).flatten()
    return np.unique(idxUsed)


def getTFGenerator(folderPath, sampleRate=50, context=25, midiLatency=0, train=True, split=0.8):
    """
    TODO: change the sampleRate to 100 Hz?
    sampleRate = if the highest speed is a note each 20ms,    
                the sr should be 1/0.02=50 
    context = how many frames are given with each samples
    midiLatency = how many frames the onsets are offseted to make sure that the transient is not discarded
    """
    # ----INIT----
    tracks = config.getFilesInFolder(folderPath, config.AUDIO)
    midis = config.getFilesInFolder(folderPath, config.MIDI_CONVERTED)
    alignments = config.getFilesInFolder(folderPath, config.MIDI_ALIGNED)

    # train, test = sklearn.model_selection.train_test_split(candidateName, test_size=test_size, random_state=1)
    if train:
        tracks = tracks[:int(len(tracks) * split)]
        midis = midis[:int(len(midis) * split)]
        alignments = alignments[:int(len(alignments) * split)]
    else:
        tracks = tracks[int(len(tracks) * split):]
        midis = midis[int(len(tracks) * split):]
        alignments = alignments[int(len(tracks) * split):]

    DATA = {}

    # ----gen----
    def gen():
        trackIdx = 0
        while True:
            trackIdx = trackIdx + 1 % len(tracks)
            if trackIdx not in DATA:
                X, Y = readTrack(trackIdx, tracks, midis, alignments, sampleRate=sampleRate, context=context, midiLatency=midiLatency)
                indexes = balanceDistribution(X, Y)
                DATA[trackIdx] = {"x": X, "y": Y, "indexes": indexes, "cursor": 0}

            data = DATA[trackIdx]
            for _ in range(2):
                cursor = data["cursor"]
                sampleIdx = data["indexes"][cursor]
                data["cursor"] = (cursor + 1) % len(data["indexes"])
                yield data["x"][sampleIdx:sampleIdx + context], data["y"][sampleIdx]

    return gen


def getClassWeight(folderPath):
    # midis = config.getFilesInFolder(folderPath, config.MIDI_CONVERTED)
    # Y = [MidiProxy(midi).getDenseEncoding(sampleRate=50) for midi in midis]
    # concat = np.concatenate(Y)
    # uni = np.unique(concat, return_counts=True, axis=0)
    #          [36, 40, 41, 46, 49]
    # 00:array([0., 0., 0., 0., 0.]) / 1734488 = 91% of all samples
    # 02:array([0., 0., 0., 1., 0.]) / 64280
    # 15:array([1., 0., 0., 1., 0.]) / 21493
    # 13:array([1., 0., 0., 0., 0.]) / 19349
    # 07:array([0., 1., 0., 0., 0.]) / 18689
    # 09:array([0., 1., 0., 1., 0.]) / 14716
    # 20:array([1., 1., 0., 0., 0.]) / 11436
    # 01:array([0., 0., 0., 0., 1.]) / 5297
    # 14:array([1., 0., 0., 0., 1.]) / 4900
    # 04:array([0., 0., 1., 0., 0.]) / 4667
    # 03:array([0., 0., 0., 1., 1.]) / 1969
    # 22:array([1., 1., 0., 1., 0.]) / 4500
    # 17:array([1., 0., 1., 0., 0.]) / 1418
    # 21:array([1., 1., 0., 0., 1.]) / 1276
    # 08:array([0., 1., 0., 0., 1.]) / 1127
    # 16:array([1., 0., 0., 1., 1.]) / 519
    # 11:array([0., 1., 1., 0., 0.]) / 448
    # 05:array([0., 0., 1., 0., 1.]) / 376
    # 06:array([0., 0., 1., 1., 0.]) / 251
    # 24:array([1., 1., 1., 0., 0.]) / 225
    # 23:array([1., 1., 0., 1., 1.]) / 128
    # 18:array([1., 0., 1., 0., 1.]) / 111
    # 19:array([1., 0., 1., 1., 0.]) / 80
    # 10:array([0., 1., 0., 1., 1.]) / 43
    # 12:array([0., 1., 1., 1., 0.]) / 2

    # # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#calculate_class_weights
    # weights = {i: 1 / concat[:, i].sum() * len(concat) / len(concat[0]) for i in range(len(concat[0]))}
    weights = {0: 5.843319324520516, 1: 7.270538125118844, 2: 50.45626814462919, 3: 3.5409710967670245, 4: 24.28284008637114}
    return weights


def vizDataset(dataset, samples=1):
    X = []
    Y = []
    for i in range(samples):
        x, y = next(dataset)
        X.append(x[0].reshape((168)))
        Y.append(y)
    plt.matshow(np.array(X).T)
    print(np.sum(Y))
    for i in range(len(Y[0])):
        times = [t for t, y in enumerate(Y) if y[i]]
        plt.plot(times, np.ones(len(times)) * i * 10, "or")
    plt.show()


# def getTFGenerator(candidateName, test_size=0.1):
#     """
#     WIP: Create a generator dynamically generating converted tracks
#     """

#     def generateGenerator(data):
#         """
#         Create a generator with the tracks in data
#         TODO: this is ugly
#         """

#         def gen(context=25, midiLatency=12, classWeight=[2 / 16, 8 / 16, 16 / 16, 2 / 16, 4 / 16]):
#             """
#             [36, 40, 41, 46, 49]
#             """
#             mir = MIR()
#             for midiPath, audiPath, converter in data:
#                 try:
#                     # TODO: update: _, audio, _ = converter.getConvertibleFiles(path)
#                     # Get the y: midi in dense matrix representation
#                     y = converter.convert(midiPath).getDenseEncoding(sampleRate=100, timeShift=0, radiation=0)
#                     y = y[midiLatency:]
#                     if np.sum(y) == 0:
#                         warnings.warn("Midi doesn't have notes " + midiPath)
#                         continue

#                     # Get the x: audio with stft or cqt or whatever + overlap windows to get some context
#                     x = mir.open(audiPath)
#                     x = np.array([x[i:i + context] for i in range(len(x) - context)])
#                     x = x.reshape(x.shape + (1, ))  # Add the channel dimension

#                     for i in range(min(len(y) - 1, len(x) - 1)):
#                         # sampleWeight = 1  #max(1/16, np.sum(classWeight * y[i])) #TODO: compute the ideal weight based on the distribution of the samples
#                         yield x[i], y[i]
#                 except Exception as e:
#                     print(midiPath, e)
#             print("DEBUG: real new epoch")

#         return gen

#     train, test = sklearn.model_selection.train_test_split(candidateName, test_size=test_size, random_state=1)

#     # next(Converter.generateGenerator(train)())
