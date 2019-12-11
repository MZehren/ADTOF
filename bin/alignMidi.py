#!/usr/bin/env python
# encoding: utf-8
"""

"""
import argparse
import os

from adtof import config
from adtof.io.converters.onsetsAlignementConverter import OnsetsAlignementConverter


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('inputFolder', type=str, help="Path to the chart folder.")
    args = parser.parse_args()

    oac = OnsetsAlignementConverter()

    files = [f[:-4] for f in os.listdir(os.path.join(args.inputFolder, "audio"))] #TODO: hardcoded .ogg extension
    data = [[
        os.path.join(args.inputFolder, "audio", f + ".ogg"),
        os.path.join(args.inputFolder, "midi_converted", f + ".midi"),
        os.path.join(args.inputFolder, "midi_aligned", f + ".midi")
    ] for f in files]
    for audio, midiIn, midiOut in data:
        oac.convert(audio, midiIn, midiOut)
    print("Done!")


if __name__ == '__main__':
    main()
