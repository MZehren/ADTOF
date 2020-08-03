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
    # parser.add_argument("-c", "--convertInput", action="store_true", help="Convert the pitches 0, 1, 2 to the correct 36, 40, 42")
    args = parser.parse_args()

    # Get the data
    groundTruthsPaths = config.getFilesInFolder(args.groundTruthPath)
    estimationsPaths = [path for path in config.getFilesInFolder(args.estimationsPath) if os.path.splitext(path)[1] == ".txt"]
    groundTruthsPaths, estimationsPaths = config.getIntersectionOfPaths(groundTruthsPaths, estimationsPaths)
    tr = TextReader()
    groundTruths = [tr.getOnsets(grounTruth) for grounTruth in groundTruthsPaths]
    estimations = [tr.getOnsets(estimation) for estimation in estimationsPaths]

    result = eval.runEvaluation(groundTruths, estimations, groundTruthsPaths)
    print(result)
    plot(result, prefix="mean")
    plot(result, prefix="sum")


def plot(result, prefix="mean", bars=["F", "P", "R"], groups=["all", "35", "38", "47", "42", "49"]):
    """
    {'mean F': 0.4400290519510308, 'mean P': 0.6659775429409889, 'mean R': 0.4096366303496082, 'mean F 35': 0.5989156395480592, 'mean P 35': 0.8660609488134552, 'mean R 35': 0.5240203566565339, 'mean F 38': 0.6032228067405822, 'mean P 38': 0.7579546096714384, 'mean R 38': 0.5846379121932345, 'mean F 42': 0.11794870956445104, 'mean P 42': 0.37391707033807325, 'mean R 42': 0.12025162219905618}
    """
    fig, ax = plt.subplots()
    ind = np.arange(len(groups))  # the x locations for the groups
    width = 1 / (len(groups) + 1)  # the width of the bars

    for i, bar in enumerate(bars):
        X = ind + (i * width)
        Y = [result[" ".join([prefix, bar, group])] for group in groups]
        ax.bar(
            X, Y, width=width, edgecolor="black", label=" ".join([prefix, bar]),
        )
        for x, y in zip(X, Y):
            plt.annotate(np.format_float_positional(y, precision=2), xy=(x - width / 2, y + 0.01))

    plt.xticks(ind + width, groups)
    plt.grid(axis="y", linestyle="--")
    plt.legend()
    plt.ylim(0, 1)
    plt.show()


if __name__ == "__main__":
    main()
