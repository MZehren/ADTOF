import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from adtof import config
from adtof.io.converters.converter import Converter
from adtof.io.mir import MIR
from adtof.io.myMidi import MidiProxy


def readTrack(i, tracks, midis, alignments, sampleRate=50, context=25, midiLatency=0, labels=[36, 40, 41, 46, 49]):
    """
    Read the mp3, the midi and the alignment files and generate a balanced list of samples 
    """
    # initiate vars
    print("new track read:", tracks[i])
    track = tracks[i]
    midi = midis[i]
    alignment = alignments[i]
    mir = MIR(frameRate=sampleRate)

    # read files
    alignmentInput = pd.read_csv(alignment, escapechar=" ")
    y = MidiProxy(midi).getDenseEncoding(
        sampleRate=sampleRate, offset=-alignmentInput.offset[0], playback=1 / alignmentInput.playback[0], keys=labels
    )
    x = mir.open(track)
    x = x.reshape(x.shape + (1, ))  # Add the channel dimension TODO: remove?

    # Trim before the first midi note and after the last uncovered part
    firstNoteIdx = -1
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


def getTFGenerator(folderPath, sampleRate=50, context=25, midiLatency=0, train=True, split=0.8, labels=[36, 40, 41, 46, 49], minF=0.5):
    """
    TODO: change the sampleRate to 100 Hz?
    sampleRate = if the highest speed is a note each 20ms,    
                the sr should be 1/0.02=50 
    context = how many frames are given with each samples
    midiLatency = how many frames the onsets are offseted to make sure that the transient is not discarded
    """
    tracks = config.getFilesInFolder(folderPath, config.AUDIO)
    midis = config.getFilesInFolder(folderPath, config.MIDI_CONVERTED)
    alignments = config.getFilesInFolder(folderPath, config.MIDI_ALIGNED)

    # F-Measure filtering
    bools = [pd.read_csv(alignment, escapechar=" ")["F-measure"][0] > minF for alignment in alignments]
    tracks = tracks[bools].tolist() # TODO why tracks[0] returns a np._str, how to get the value?
    midis = midis[bools].tolist()
    alignments = alignments[bools].tolist()

    # train, test = sklearn.model_selection.train_test_split(candidateName, test_size=test_size, random_state=1)
    if train:
        tracks = tracks[:int(len(tracks) * split)]
        midis = midis[:int(len(midis) * split)]
        alignments = alignments[:int(len(alignments) * split)]
    else:
        tracks = tracks[int(len(tracks) * split):]
        midis = midis[int(len(midis) * split):]
        alignments = alignments[int(len(alignments) * split):]

    # Lazy cache dictionnary
    DATA = {}

    def gen():
        trackIdx = 0
        while True:
            trackIdx = (trackIdx + 1) % len(tracks)
            if trackIdx not in DATA:
                X, Y = readTrack(trackIdx, tracks, midis, alignments, sampleRate=sampleRate, context=context, midiLatency=midiLatency, labels=labels)
                indexes = balanceDistribution(X, Y)
                DATA[trackIdx] = {"x": X, "y": Y, "indexes": indexes, "cursor": len(indexes) // 2}

            data = DATA[trackIdx]
            if len(data["y"]) == 0:  # In case the tracks doesn't have notes
                continue
            n = 2 if train else 100
            for _ in range(n):
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


def vizDataset(folderPath, samples=100, labels=[36], sampleRate=50, condensed = False):
    gen = getTFGenerator(folderPath, train=False, labels=labels, sampleRate=sampleRate, midiLatency=10)()

    X = []
    Y = []
    if condensed:
        for i in range(samples):
            x, y = next(gen)
            X.append(x[0].reshape((168)))
            Y.append(y)

        plt.matshow(np.array(X).T)
        print(np.sum(Y))
        for i in range(len(Y[0])):
            times = [t for t, y in enumerate(Y) if y[i]]
            plt.plot(times, np.ones(len(times)) * i * 10, "or")
        plt.show()
    else:
        fig = plt.figure(figsize=(8, 8))
        columns = 2
        rows = 5
        for i in range(1, columns*rows+1):
            fig.add_subplot(rows, columns, i)
            x, y = next(gen)
            
            plt.imshow(np.reshape(x, (25,168)))
            if y[0]:
                plt.plot([0], [0], "or")
        plt.show()

