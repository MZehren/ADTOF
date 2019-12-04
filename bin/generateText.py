#!/usr/bin/env python
# encoding: utf-8
"""
WIP: convert the midi files with the alignment into text files
"""
import argparse
import logging
import os
import csv
from adtof.io.converters import OnsetsAlignementConverter


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('inputFolder', type=str, help="Path to the chart folder.")
    args = parser.parse_args()

    files = [
        os.path.join(args.inputFolder, "midi_aligned", f) for f in os.listdir(os.path.join(args.inputFolder, "midi_aligned"))
    ]  #TODO: hardcoded .ogg extension
    errors = []
    for f in files:
        with open(f) as csvFile:
            reader = csv.reader(csvFile)
            errors.append(float(list(reader)[1][0]))

    print(len([1 for error in errors if error > 0.01]) / len(errors))
    print("Done!")


if __name__ == '__main__':
    main()
