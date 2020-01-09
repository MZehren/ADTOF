#!/usr/bin/env python
# encoding: utf-8
"""
TODO
"""
import argparse
import datetime
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch

from adtof import config
from adtof.deepModels.peakPicking import PeakPicking
from adtof.deepModels.rv1pt import RV1Torch
from adtof.io.converters.converter import Converter
from adtof.io.mir import MIR
from adtof.io.myMidi import MidiProxy

logging.basicConfig(filename='logs/conversion.log', level=logging.DEBUG)

class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, folderPath, sampleRate=50, context=25, midiLatency=0, classWeight=[2 / 16, 8 / 16, 16 / 16, 2 / 16, 4 / 16], train=True, split=0.8):
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
            self.tracks = tracks[: int(len(tracks) * split)]
            self.midis = midis[: int(len(midis) * split)]
            self.alignments = alignments[: int(len(alignments) * split)]
        else:
            self.tracks = tracks[int(len(tracks) * split):]
            self.midis = midis[int(len(tracks) * split):]
            self.alignments = alignments[int(len(tracks) * split) :]
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
            midi =self. midis[i]
            alignment = self.alignments[i]    
           
            alignmentInput = pd.read_csv(alignment, escapechar=" ")
            y = MidiProxy(midi).getDenseEncoding(sampleRate=self.sampleRate)
            # TODO apply the offset correction
            # y = MidiProxy(midi).getDenseEncoding(sampleRate=sampleRate, offset=-alignmentInput.offset[0], playback= 1/alignmentInput.playback[0])
            x = self.mir.open(track)

            for rowI, row in enumerate(y):
                if max(row) == 1:
                    firstNoteIdx = rowI
                    break
            
            self.X[i] = x[firstNoteIdx - self.midiLatency: min(len(y) - 1, len(x) - self.context - 1) + self.context - self.midiLatency]
            self.Y[i] = y[firstNoteIdx: min(len(y) - 1, len(x) - self.context - 1)]
            
        j = random.randrange(len(self.Y[i]))
        # for j in range(len(self.Y[i])):
        yield self.X[i][j:j+self.context], self.Y[i][j]
            # TODO change the track for each minibatch

def vizDataset(dataset, samples = 1):
    X = []
    Y = []
    for i in range(samples):
        x, y = next(dataset)
        X.append(x[0])
        Y.append(y)
    plt.matshow(np.array(X).T)
    print(np.sum(Y))
    for i in range(len(Y[0])):
        times = [t for t, y in enumerate(Y) if y[i]]
        plt.plot(times, np.ones(len(times)) * i * 10, "or")
    plt.show()

def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    parser = argparse.ArgumentParser(description='todo')
    parser.add_argument('folderPath', type=str, help="Path.")
    args = parser.parse_args()

    # Get the data
    trainLoader = torch.utils.data.DataLoader(MyIterableDataset(args.folderPath, train=True), batch_size=100)
    testLoader = torch.utils.data.DataLoader(MyIterableDataset(args.folderPath, train=False), batch_size=100)
    classes = [36, 40, 41, 46, 49]

    dataiter = iter(trainLoader)
    images, labels = dataiter.next()
    print("Done!")


if __name__ == '__main__':
    main()
