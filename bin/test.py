#!/usr/bin/env python
# encoding: utf-8
"""
TODO
"""
import argparse
import datetime
import io
import itertools
import logging
import os
import shutil
import timeit

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import tensorflow as tf

from adtof import config
from adtof.converters.converter import Converter
from adtof.deepModels import dataLoader
from adtof.deepModels.peakPicking import PeakPicking
from adtof.deepModels.rv1tf import RV1TF
from adtof.io import mir

tf.config.threading.set_intra_op_parallelism_threads(32)
tf.config.threading.set_inter_op_parallelism_threads(32)
# tf.config.experimental_run_functions_eagerly(True)
if not os.path.exists("logs"):
    os.makedirs("logs")
logging.basicConfig(filename="logs/conversion.log", level=logging.DEBUG)


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    parser = argparse.ArgumentParser(description="todo")
    parser.add_argument("folderPath", type=str, help="Path.")
    args = parser.parse_args()

    paramGrid = {
        "labels": [config.LABELS_5],
        "classWeights": [config.WEIGHTS_5],
        "sampleRate": [100],
        "diff": [False],
        "samplePerTrack": [1],
        "batchSize": [100],
        "context": [25],
        "labelOffset": [0],
        "labelRadiation": [1],
        "learningRate": [0.001 / 2],
        "normalize": [False],
    }

    # Get the data
    # classWeight = dataLoader.getClassWeight(args.folderPath, sampleRate=sampleRate, labels=labels)
    for paramIndex, params in enumerate(list(sklearn.model_selection.ParameterGrid(paramGrid))):
        trainGen, valGen, testGen = dataLoader.getSplit(args.folderPath, **params)
        bla = trainGen()
        print(timeit.timeit(lambda: next(bla), number=2000))
        bli = valGen()
        print(timeit.timeit(lambda: next(bli), number=2000))
        blu = testGen()
        print(timeit.timeit(lambda: next(blu), number=2000))


if __name__ == "__main__":
    main()
