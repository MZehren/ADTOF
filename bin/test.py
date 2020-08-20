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
from adtof.deepModels.dataLoader import DataLoader
from adtof.deepModels.peakPicking import PeakPicking
from adtof.deepModels.rv1tf import RV1TF
from adtof.io.mir import MIR

tf.config.threading.set_intra_op_parallelism_threads(32)
tf.config.threading.set_inter_op_parallelism_threads(32)
# # tf.config.experimental_run_functions_eagerly(True)
# if not os.path.exists("logs"):
#     os.makedirs("logs")
# logging.basicConfig(filename="logs/conversion.log", level=logging.DEBUG)


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    parser = argparse.ArgumentParser(description="todo")
    parser.add_argument("folderPath", type=str, help="Path.")
    args = parser.parse_args()

    dl = DataLoader(args.folderPath)
    buffer = {}
    # gen = dl.getGen(repeat=False, samplePerTrack=None)
    def gen():
        mir = MIR()
        for audioPath, cachePath in zip(dl.audioPaths, dl.featurePaths):
            buffer[audioPath] = mir.open(audioPath, cachePath=cachePath)
            yield buffer[audioPath]

    i = 0
    for value in gen():
        print(i)
        i += 1


if __name__ == "__main__":
    main()
