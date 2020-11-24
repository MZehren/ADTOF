#!/usr/bin/env python
# encoding: utf-8
"""
TODO
"""
import argparse
import datetime
import logging
import os

import numpy as np
import pandas as pd
import pretty_midi
import sklearn

from adtof import config
from automix.model.classes.deck import Deck
from automix.model.classes.track import Track
from automix.model.inputOutput.serializer.reaperProxy import ReaperProxy


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    parser = argparse.ArgumentParser(description='todo')
    parser.add_argument('folderPath', type=str, help="Path.")
    args = parser.parse_args()

    rpp = ReaperProxy()

    tracks = config.getFilesInFolder(args.folderPath, config.AUDIO)
    midis = config.getFilesInFolder(args.folderPath, config.MIDI_CONVERTED)
    alignments = config.getFilesInFolder(args.folderPath, config.MIDI_ALIGNED)

    for i, track in enumerate(tracks):
        name = config.getFileBasename(track)
        changes, tempi = pretty_midi.PrettyMIDI(midis[i]).get_tempo_changes()
        alignmentInput = pd.read_csv(alignments[i], escapechar=" ")

        rpp.serialize(
            os.path.join(args.folderPath, "RPP", name + ".rpp"), [
                Deck(
                    tracks=[
                        Track(tracks[i], populateFeatures=False, position=alignmentInput.offset[0], playRate=alignmentInput.playback[0], length=1000)
                    ]
                ),
                Deck(tracks=[Track(midis[i], populateFeatures=False, length=1000)])
            ],
            tempoMarkers=[(changes[i], tempi[i]) for i, _ in enumerate(changes)]
        )

        print("bla")


if __name__ == '__main__':
    main()
