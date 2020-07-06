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
    labels = ["36"]  # [36, 40, 41, 46, 49]
    sampleRate = 100

    # dataLoader.vizDataset(args.folderPath, labels=labels, sampleRate=sampleRate)
    # Plot the first image of the dataset
    # for x, y in dataset:
    #     file_writer = tf.summary.create_file_writer(log_dir)
    #     with file_writer.as_default():
    #         tf.summary.image(str(list(np.reshape(y, (batch_size)))), x, step=0, max_outputs=20, description=str(list(np.reshape(y, (batch_size)))))

    # Get the data
    # classWeight = dataLoader.getClassWeight(args.folderPath)
    bla = dataLoader.getTFGenerator(args.folderPath, train=True, labels=labels, sampleRate=sampleRate, limitInstances=50)()
    while True:
        next(bla)


if __name__ == "__main__":
    main()
