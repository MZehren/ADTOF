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


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    parser = argparse.ArgumentParser(description="todo")
    parser.add_argument("inputPath", type=str, help="Path to music or folder containing music to transcribe")
    parser.add_argument("outputPath", type=str, default="./", help="Path to output folder")
    parser.add_argument("model", type=str, help="name of the nodel to train, possible choice", default="crnn-ADTOF")
    args = parser.parse_args()

    # Get the model
    model, hparams = next(Model.modelFactory(fold=0))
    assert "peakThreshold" in hparams

    model.predictFolder(args.inputPath, args.outputPath, **hparams)


if __name__ == "__main__":
    main()
