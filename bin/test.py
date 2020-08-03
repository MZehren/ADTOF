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

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import tensorflow as tf

from adtof.deepModels import dataLoader
from adtof.deepModels.peakPicking import PeakPicking
from adtof.deepModels.rv1tf import RV1TF
from adtof.io import mir
from adtof.converters.converter import Converter
from adtof import config

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
    parser.add_argument("-r", "--restart", action="store_true", help="Override the model if present")
    parser.add_argument("-d", "--deleteLogs", action="store_true", help="Delete the logs")
    parser.add_argument("-l", "--limit", type=int, default=-1, help="Limit the number of tracks used in training and eval")
    args = parser.parse_args()
    labels = config.LABELS_5
    classWeights = config.WEIGHTS_5
    sampleRate = 100

    # Get the data
    classWeight = dataLoader.getClassWeight(args.folderPath, sampleRate=sampleRate, labels=labels)

    generator = dataLoader.getTFGenerator(
        args.folderPath, train=True, labels=labels, classWeights=classWeights, sampleRate=sampleRate, limitInstances=50
    )
    while True:
        bla = generator()
        for i in range(1000):
            next(bla)

        print("next")
        bla = generator()
        for i in range(1000):
            next(bla)


if __name__ == "__main__":
    main()
