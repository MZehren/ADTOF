#!/usr/bin/env python
# encoding: utf-8
"""
Convert the PhaseShift's MIDI file format into real MIDI
"""
import argparse
import concurrent.futures
import logging

from adtof.converters.converter import Converter
import os


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description="Process a Phase Shift chart folder and convert the MIDI file to standard MIDI")
    parser.add_argument("inputFolder", type=str, help="Path to the chart folder.")
    parser.add_argument("outputFolder", type=str, help="Path to the destination folder")
    args = parser.parse_args()

    Converter.convertAll(args.inputFolder, args.outputFolder)

    print("Done!")


if __name__ == "__main__":
    main()
