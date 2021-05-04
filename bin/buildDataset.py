#!/usr/bin/env python
# encoding: utf-8

import argparse
import logging
import os
import shutil

import pandas as pd
from adtof import config
from adtof.converters.converter import Converter
from adtof.model.dataLoader import DataLoader

cwd = os.path.abspath(os.path.dirname(__file__))
all_logs = os.path.join(cwd, "..", "logs/")
tensorboardLogs = os.path.join(all_logs, "fit/")
hparamsLogs = os.path.join(all_logs, "hparam/")
Converter.checkPathExists(all_logs)
logging.basicConfig(filename=os.path.join(all_logs, "buildDataset.log"), level=logging.DEBUG, filemode="w")


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description="Process a Phase Shift chart folder and convert the MIDI file to standard MIDI")
    parser.add_argument("inputFolder", type=str, help="Path to the chart folder.")
    parser.add_argument("outputFolder", type=str, help="Path to the destination folder.")
    parser.add_argument("-p", "--parallel", action="store_true", help="Set if the conversion is ran in parallel")
    args = parser.parse_args()

    Converter.convertAll(args.inputFolder, args.outputFolder, parallelProcess=args.parallel)
    genSplits(args.outputFolder)

    print("Done!")


def genSplits(outputFolder, nFolds=10):

    # Get split
    dl = DataLoader.factoryADTOF(outputFolder, lazyLoading=True)
    tracks = [dl.audioPaths[i] for i in dl.testIndexes]

    # text
    path = os.path.join(outputFolder, config.SPLIT, str(nFolds) + "cv_test_split")
    Converter.checkPathExists(path)
    pd.DataFrame(tracks).to_csv(path, header=None, index=False)

    # audio
    audioPath = os.path.join(outputFolder, config.SPLIT, "audio")
    for source in tracks:
        target = os.path.join(audioPath, os.path.basename(source))
        Converter.checkPathExists(target)
        shutil.copy(source, target)


if __name__ == "__main__":
    main()
