#!/usr/bin/env python
# encoding: utf-8
"""
Convert the PhaseShift's MIDI file format into real MIDI
"""
import argparse
import concurrent.futures
import logging
import os

import pandas as pd
from adtof import config
from adtof.converters.converter import Converter
from adtof.model.dataLoader import DataLoader


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description="Process a Phase Shift chart folder and convert the MIDI file to standard MIDI")
    parser.add_argument("inputFolder", type=str, help="Path to the chart folder.")
    parser.add_argument("outputFolder", type=str, help="Path to the destination folder")
    parser.add_argument("-p", "--parallel", action="store_true", help="Set if the conversion is ran in parallel")
    args = parser.parse_args()

    Converter.convertAll(args.inputFolder, args.outputFolder, parallelProcess=args.parallel)
    genSplits(args.outputFolder)

    print("Done!")


def genSplits(outputFolder, nFolds=10):
    dl = DataLoader(outputFolder)
    trainIndexes, valIndexes, testIndexes = dl.getSplit(nFolds=nFolds)

    path = os.path.join(outputFolder, config.SPLIT, str(nFolds) + "cv_test_split")
    Converter.checkPathExists(path)
    pd.DataFrame([dl.audioPaths[i] for i in testIndexes]).to_csv(path, header=None, index=False)


if __name__ == "__main__":
    main()
