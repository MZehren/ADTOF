#!/usr/bin/env python
# encoding: utf-8
"""
WIP: convert the midi files with the alignment into text files
"""
import argparse
import csv
import os
import numpy as np

from adtof import config

import pretty_midi


def main():
    # load the arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('inputFolder', type=str, help="Path to the chart folder.")
    args = parser.parse_args()
    # parameters of the evaluation

    # load the file path
    # gtPaths = config.getFilesInFolder(args.inputFolder, config.MIDI_CONVERTED)
    # for f in gtPaths:
    #     midi = pretty_midi.PrettyMIDI(f)
    #     midi.get_beats(start_time=midi.get_onsets()[0])
    gtPaths = config.getFilesInFolder(args.inputFolder, "midi_aligned")
    errors = []
    for f in gtPaths:
        with open(f) as csvFile:
            reader = csv.reader(csvFile)
            errors.append([float(list(reader)[1][0]), f])

    errors.sort(key=lambda x: np.abs(x[0]))
    print(errors)


if __name__ == '__main__':
    main()
