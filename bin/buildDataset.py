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
Converter.checkPathExists(all_logs)
logging.basicConfig(filename=os.path.join(all_logs, "buildDataset.log"), level=logging.DEBUG, filemode="w")


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description="Process a chart folder with the automatic cleaning")
    parser.add_argument("inputFolder", type=str, help="Path to the chart folder.")
    parser.add_argument("outputFolder", type=str, help="Path to the destination folder.")
    parser.add_argument("-p", "--parallel", action="store_true", help="Set to run the cleaning in parallel")
    args = parser.parse_args()

    Converter.convertAll(args.inputFolder, args.outputFolder, parallelProcess=args.parallel)
    print("Done!")


if __name__ == "__main__":
    main()
