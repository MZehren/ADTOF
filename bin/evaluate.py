#!/usr/bin/env python
# encoding: utf-8
"""
TODO
"""
from adtof.io.textReader import TextReader
import argparse
import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import mir_eval

from adtof import config
from adtof.deepModels import dataLoader
from adtof.deepModels.peakPicking import PeakPicking
from adtof.deepModels.rv1tf import RV1TF
from adtof.converters.converter import Converter
from adtof.io.mir import MIR
from adtof.io import eval


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    parser = argparse.ArgumentParser(description="todo")
    parser.add_argument("groundTruthPath", type=str, help="Path to music or folder containing music to transcribe")
    parser.add_argument("estimationsPath", type=str, help="Path to output folder")
    parser.add_argument("-w", "--window", type=float, default=0.05, help="window size for hit rate")
    args = parser.parse_args()

    # Get the data
    groundTruths = config.getFilesInFolder(args.groundTruthPath)
    estimations = [path for path in config.getFilesInFolder(args.estimationsPath) if os.path.splitext(path)[1] == ".txt"]
    groundTruths, estimations = config.getIntersectionOfPaths(groundTruths, estimations)
    tr = TextReader()
    groundTruths = [tr.getOnsets(grounTruth, separated=True) for grounTruth in groundTruths]
    estimations = [tr.getOnsets(estimation, separated=True) for estimation in estimations]

    result = eval.runEvaluation(groundTruths, estimations)


if __name__ == "__main__":
    main()
