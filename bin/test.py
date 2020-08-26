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

import pickle

# file="/home/mickael/Documents/programming/madmom-0.16.dev0/madmom/models/drums/2018/drums_cnn0_O8_S0.pkl"
file = "/Users/mzehren/Programming/ADTOF/vendors/madmom-0.16.dev0/madmom/models/drums/2018/drums_crnn1_O8_S0.pkl"
with open(file, "rb") as f:
    u = pickle._Unpickler(f)
    u.encoding = "latin1"
    p = u.load()
    print(p)


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    parser = argparse.ArgumentParser(description="todo")
    parser.add_argument("folderPath", type=str, help="Path.")
    args = parser.parse_args()

    dl = DataLoader(args.folderPath)
    gen = dl.getSplit(validationFold=1)


if __name__ == "__main__":
    main()
