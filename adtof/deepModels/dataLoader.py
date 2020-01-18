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
            y = MidiProxy(midi).getDenseEncoding(sampleRate=sampleRate, offset=-alignmentInput.offset[0], playback= 1/alignmentInput.playback[0])
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


def getTFGenerator(
    folderPath, sampleRate=50, context=25, midiLatency=0, train=True, split=0.8
):
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
    mir = MIR(frameRate=sampleRate)
    X = {}
    Y = {}
    sampleRate = sampleRate
    midiLatency = midiLatency
    context = context

    # ----gen----
    def gen():
        while True:
            for i, _ in enumerate(tracks):
            # i = random.randrange(len(tracks))
                if i not in X:
                    print("new track read:", i)
                    track = tracks[i]
                    midi = midis[i]
                    alignment = alignments[i]

                    alignmentInput = pd.read_csv(alignment, escapechar=" ")
                    # y = MidiProxy(midi).getDenseEncoding(sampleRate=sampleRate)
                    # TODO apply the offset correction
                    y = MidiProxy(midi).getDenseEncoding(sampleRate=sampleRate, offset=-alignmentInput.offset[0], playback=1 / alignmentInput.playback[0])
                    x = mir.open(track)
                    x = x.reshape(x.shape + (1, ))  # TODO: remove Add the channel dimension

                    for rowI, row in enumerate(y):
                        if max(row) == 1:
                            firstNoteIdx = rowI
                            break

                    X[i] = x[firstNoteIdx - midiLatency:min(len(y) - 1, len(x) - context - 1) + context - midiLatency]
                    Y[i] = y[firstNoteIdx:min(len(y) - 1, len(x) - context - 1)]

                # TODO change the track only between each minibatch?
                j = random.randrange(len(Y[i]))
                for j in range(len(Y[i])):
                    # if sum(Y[i][j]) == 0:
                    #     continue
                    yield X[i][j:j + context], Y[i][j]

    return gen

def getClassWeight(folderPath):
    # midis = config.getFilesInFolder(folderPath, config.MIDI_CONVERTED)
    # Y = [MidiProxy(midi).getDenseEncoding(sampleRate=50) for midi in midis]
    # concat = np.concatenate(Y)
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
