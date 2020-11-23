#!/usr/bin/env python
# encoding: utf-8
"""
TODO
"""
import argparse
from bin.trainModelTF import hp
import datetime
import logging
import os

import madmom
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import sklearn
from tensorboard.plugins.hparams.summary_v2 import hparams
import tensorflow as tf

from adtof import config
from adtof.converters.converter import Converter
from adtof.model.dataLoader import DataLoader
from adtof.model.model import Model
from adtof.model import peakPicking
from adtof.io.textReader import TextReader
from adtof.converters.RVCRNNConverter import RVCRNNConverter

# TODO: needed because error is thrown:
# Check failed: ret == 0 (11 vs. 0)Thread creation via pthread_create() failed.
# See: https://github.com/tensorflow/tensorflow/issues/41532
# tf.config.threading.set_intra_op_parallelism_threads(32)
# tf.config.threading.set_inter_op_parallelism_threads(32)


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    parser = argparse.ArgumentParser(description="todo")
    parser.add_argument("inputPath", type=str, help="Path to music or folder containing music to transcribe")
    parser.add_argument("outputPath", type=str, default="./", help="Path to output folder")
    args = parser.parse_args()

    Converter.checkPathExists(os.path.join(args.outputPath, "filePlaceholder"))
    rvConverter = RVCRNNConverter()
    rvConverter.convertAll(args.inputPath, args.outputPath)


if __name__ == "__main__":
    main()
