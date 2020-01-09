#!/usr/bin/env python
# encoding: utf-8
"""
TODO
"""
import argparse
import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import sklearn

import pandas as pd
from adtof import config
from adtof.deepModels.peakPicking import PeakPicking
from adtof.deepModels.rv1pt import RV1Torch
from adtof.io.converters.converter import Converter
from adtof.io.mir import MIR
from adtof.io.myMidi import MidiProxy

logging.basicConfig(filename='logs/conversion.log', level=logging.DEBUG)


def getDataset(folderPath, sampleRate=25, context=25, midiLatency=0, classWeight=[2 / 16, 8 / 16, 16 / 16, 2 / 16, 4 / 16]):
    """
    sampleRate = if the highest speed is a note each 20ms,    
                the sr should be 1/0.02=50 
    context = how many frames are given with each samples
    midiLatency = how many frames the onsets are offseted to make sure that the transient is not discarded
    """
    tracks = config.getFilesInFolder(folderPath, config.AUDIO)
    midis = config.getFilesInFolder(folderPath, config.MIDI_CONVERTED)
    alignments = config.getFilesInFolder(folderPath, config.MIDI_ALIGNED)
    mir = MIR(sampleRate=sampleRate)
    for track, midi, alignment in zip(tracks, midis, alignments):
        alignmentInput = pd.read_csv(alignment, escapechar=" ")
        y = MidiProxy(midi).getDenseEncoding(sampleRate=sampleRate)
        # y = MidiProxy(midi).getDenseEncoding(sampleRate=sampleRate, offset=-alignmentInput.offset[0], playback= 1/alignmentInput.playback[0])
        for i, row in enumerate(y):
            if max(row) == 1:
                firstNoteIdx = i
                break
        x = mir.open(track)
        for i in range(firstNoteIdx, min(len(y) - 1, len(x) - 26)):
            yield x[i - midiLatency:i + context - midiLatency], y[i]

def vizDataset(dataset, samples = 50):
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
    dataset = getDataset(args.folderPath)
    while(True):
        vizDataset(dataset, samples=2000)

    print("Done!")


if __name__ == '__main__':
    main()
