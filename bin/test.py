#!/usr/bin/env python
# encoding: utf-8
"""
TODO
"""
import argparse
import datetime
import logging
import os
import shutil

import numpy as np
import six
import sklearn
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from adtof import config
from adtof.converters.converter import Converter
from adtof.model.dataLoader import DataLoader
from adtof.model.model import Model

# TODO: needed because error is thrown:
# Check failed: ret == 0 (11 vs. 0)Thread creation via pthread_create() failed.
# See: https://github.com/tensorflow/tensorflow/issues/41532
tf.config.threading.set_intra_op_parallelism_threads(32)
tf.config.threading.set_inter_op_parallelism_threads(32)

# Setup logs
cwd = os.path.abspath(os.path.dirname(__file__))
all_logs = os.path.join(cwd, "..", "logs/")
tensorboardLogs = os.path.join(all_logs, "fit/")
hparamsLogs = os.path.join(all_logs, "hparam/")
Converter.checkPathExists(all_logs)
logging.basicConfig(filename=os.path.join(all_logs, "training.log"), level=logging.DEBUG, filemode="w")


def main():
    """
    Entry point of the program
    """
    parser = argparse.ArgumentParser(description="todo")
    parser.add_argument("folderPath", type=str, help="Path.")
    args = parser.parse_args()

    for fold in range(2):
        for model, hparams in Model.modelFactory(fold=fold):
            dl = DataLoader(args.folderPath, **hparams)
            trainGen, valGen, valFullGen, _ = dl.getTrainValTestGens(validationFold=fold, **hparams)
            model.evaluate(valFullGen, **hparams)
            # trainGen = trainGen()
            # for i in range(1000):
            #     next(trainGen)

            dataset_train = tf.data.Dataset.from_generator(
                valFullGen,
                (tf.float32, tf.float32, tf.float32),
                output_shapes=(tf.TensorShape((None, None, 1)), tf.TensorShape((len(hparams["labels"]),)), tf.TensorShape(1)),
            )
            dataset_train = dataset_train.batch(hparams["batchSize"]).repeat()
            model.vizPredictions(dataset_train, **hparams)


if __name__ == "__main__":
    main()
