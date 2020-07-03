import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from adtof import config
from adtof.io.midiProxy import MidiProxy
from adtof.io.mir import MIR
from adtof.io.textReader import TextReader


def readTrack(i, tracks, drums, sampleRate=100, context=25, midiLatency=0, labels=[36, 40, 41, 46, 49]):
    """
    Read the mp3, the midi and the alignment files and generate a balanced list of samples 
    """
    # initiate vars
    print("new track read:", tracks[i])
    track = tracks[i]
    drum = drums[i]
    mir = MIR(frameRate=sampleRate)

    # read files
    # notes = MidiProxy(midi).getOnsets(separated=True)
    notes = TextReader().getOnsets(drum, convertPitches=False, separated=True)
    y = getDenseEncoding(drum, notes, sampleRate=sampleRate, keys=labels)
    x = mir.open(track)
    x = x.reshape(x.shape + (1,))  # Add the channel dimension

    # Trim before the first midi note (to remove the unannotated count in)
    # and after the last uncovered part
    # If there are no event for the pitch, skip it. Default to index 0 if there are not pitch with event.
    firstNoteIdx = round(min([notes[pitch][0] for pitch in labels if len(notes[pitch])], default=0) * sampleRate)
    lastSampleIdx = min(len(y) - 1, len(x) - context - 1)
    X = x[firstNoteIdx - midiLatency : lastSampleIdx + context - midiLatency]
    Y = y[firstNoteIdx:lastSampleIdx]
    return X, Y


def getDenseEncoding(filename, notes, sampleRate=100, offset=0, playback=1, keys=[36, 40, 41, 46, 49], radiation=1):
    """
    Encode in a dense matrix the midi onsets

    sampleRate = sampleRate
    timeShift = offset of the midi, so the event is actually annotated later
    keys = pitch of the offset in each column of the matrix
    radiation = how many rows from the event are also set to 1
    
    """
    length = np.max([values[-1] for values in notes.values()])
    result = []
    for key in keys:
        # Size of the dense matrix
        row = np.zeros(int(np.round((length / playback + offset) * sampleRate)) + 1)
        for time in notes[key]:
            # indexs at 1 in the dense matrix
            index = int(np.round((time / playback + offset) * sampleRate))
            if index <= 0:
                print(str(time), filename)

            for i in range(max(0, index - radiation), min(index + radiation + 1, len(row) - 1)):
                row[i] = 0.5
            row[index] = 1

        result.append(row)
        if len(notes[key]) == 0:
            logging.info(filename + " " + str(key) + " is not represented in this track")

    return np.array(result).T


def balanceDistribution(X, Y):
    """ 
    balance the distribution of the labels Y by removing the labels without events such as there is only half of them empty.
    """
    nonEmptyIndexes = [i for i, row in enumerate(Y) if np.max(row) == 1]
    emptyIndexes = [(nonEmptyIndexes[i] + nonEmptyIndexes[i + 1]) // 2 for i in range(len(nonEmptyIndexes) - 1)]
    idxUsed = np.array(list(zip(nonEmptyIndexes, emptyIndexes))).flatten()
    return np.unique(idxUsed)


def getTFGenerator(
    folderPath,
    sampleRate=100,
    context=25,
    midiLatency=0,
    train=True,
    split=0.85,
    labels=[36, 40, 41, 46, 49],
    samplePerTrack=100,
    balanceClasses=False,
    limitInstances=-1,
    Shuffle=True,
):
    """
    TODO: change the sampleRate to 100 Hz?
    sampleRate = 
        - If the highest speed we need to discretize is 20Hz, then we should double the speed 40Hz ->  0.025ms of window
        - Realisticly speaking the fastest tempo I witnessed is around 250 bpm at 16th notes -> 250/60*16 = 66Hz the sr should be 132Hz

    context = how many frames are given with each samples
    midiLatency = how many frames the onsets are offseted to make sure that the transient is not discarded
    """
    tracks = config.getFilesInFolder(folderPath, config.AUDIO)
    drums = config.getFilesInFolder(folderPath, config.ALIGNED_DRUM)

    # Getting the intersection of audio and annotations
    tracks, drums = config.getIntersectionOfPaths(tracks, drums)

    # Split
    # train, test = sklearn.model_selection.train_test_split(candidateName, test_size=test_size, random_state=1)
    if train:
        tracks = tracks[: int(len(tracks) * split)].tolist()
        drums = drums[: int(len(drums) * split)].tolist()
    else:
        tracks = tracks[int(len(tracks) * split) :].tolist()
        drums = drums[int(len(drums) * split) :].tolist()

    # shuffle
    if Shuffle:
        tracks, drums = sklearn.utils.shuffle(tracks, drums)

    # Lazy cache dictionnary
    DATA = {}

    def gen():
        trackIdx = 0
        maxTrackIdx = len(tracks) if limitInstances == -1 else min(len(tracks), limitInstances)
        while True:
            trackIdx = (trackIdx + 1) % maxTrackIdx
            if trackIdx not in DATA:
                X, Y = readTrack(trackIdx, tracks, drums, sampleRate=sampleRate, context=context, midiLatency=midiLatency, labels=labels)
                indexes = balanceDistribution(X, Y)
                DATA[trackIdx] = {"x": X, "y": Y, "indexes": indexes, "cursor": 0}

            data = DATA[trackIdx]
            if len(data["y"]) == 0:  # In case the tracks doesn't have notes
                continue

            for _ in range(samplePerTrack):
                cursor = data["cursor"]
                if balanceClasses:
                    data["cursor"] = (cursor + 1) % len(data["indexes"])
                    sampleIdx = data["indexes"][cursor]
                else:
                    data["cursor"] = (cursor + 1) % (len(data["x"]) - context)
                    sampleIdx = cursor
                yield data["x"][sampleIdx : sampleIdx + context], data["y"][sampleIdx]

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


def vizDataset(folderPath, samples=100, labels=[36], sampleRate=50, condensed=False):
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
        for i in range(1, columns * rows + 1):
            fig.add_subplot(rows, columns, i)
            x, y = next(gen)

            plt.imshow(np.reshape(x, (25, 168)))
            if y[0]:
                plt.plot([0], [0], "or")
        plt.show()
