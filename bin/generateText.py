#!/usr/bin/env python
# encoding: utf-8
"""
WIP: convert the midi files with the alignment into text files
"""
import argparse
import csv
import os

from adtof import config

import pretty_midi

def main():
    # load the arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('inputFolder', type=str, help="Path to the chart folder.")
    args = parser.parse_args()
    # parameters of the evaluation

    # load the file path
    gtPaths = config.getFilesInFolder(args.inputFolder, config.MIDI_CONVERTED)
    for f in gtPaths:
        midi = pretty_midi.PrettyMIDI(f)
        midi.get_beats(start_time=midi.get_onsets()[0])
    # files = [
    #     os.path.join(args.inputFolder, "midi_aligned", f) for f in os.listdir(os.path.join(args.inputFolder, "midi_aligned"))
    # ]  #TODO: hardcoded .ogg extension
    # errors = []
    # for f in files:
    #     with open(f) as csvFile:
    #         reader = csv.reader(csvFile)
    #         errors.append(float(list(reader)[1][0]))

    # print(len([1 for error in errors if error > 0.01]) / len(errors))
    # print("Done!")


if __name__ == '__main__':
    main()
