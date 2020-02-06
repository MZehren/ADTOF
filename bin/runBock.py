#!/usr/bin/env python
# encoding: utf-8
"""

"""
import argparse
import os

import mir_eval
import numpy as np
import pandas as pd
import pretty_midi

from adtof import config
from automix.model.classes.track import Track


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('inputFolder', type=str, help="Path to the chart folder.")
    args = parser.parse_args()

    audios = config.getFilesInFolder(args.inputFolder, config.AUDIO)
    midis = config.getFilesInFolder(args.inputFolder, config.MIDI_CONVERTED)
    alignments = config.getFilesInFolder(args.inputFolder, config.MIDI_ALIGNED)

    F=[]
    for audioPath, midiPath, offsetPath in zip(audios, midis, alignments):
        track = Track(audioPath, config.getFileBasename(audioPath))
        audioBeats = track.getBeats()
        
        alignmentInput = pd.read_csv(offsetPath, escapechar=" ")
        midiBeats = pretty_midi.PrettyMIDI(midiPath).get_beats() * alignmentInput.playback[0] - alignmentInput.offset[0]

        f, p, r = mir_eval.onset.f_measure(midiBeats, audioBeats.times)
        F.append(f)
        print(np.mean(F))


if __name__ == '__main__':
    main()
