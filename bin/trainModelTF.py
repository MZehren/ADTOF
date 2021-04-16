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
from adtof.model import peakPicking

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
    parser.add_argument("folderPath", type=str, help="Path to the training dataset.")
    args = parser.parse_args()

    for fold in [0, 1, 2]:
        for model, hparams in Model.modelFactory(fold=fold):
            score = train_test_model(hparams, args, fold, model)

            if score is not None:
                with tf.summary.create_file_writer(hparamsLogs + model.name).as_default():
                    hp.hparams(
                        {k: v if isinstance(v, (bool, float, int, six.string_types)) else str(v) for k, v in hparams.items()},
                        trial_id=model.name,
                    )
                    for key, value in score.items():
                        tf.summary.scalar(key, value, step=fold)


def train_test_model(hparams, args, fold, model: Model):
    """
    Train procedure for one model with one set of hparam for one fold.
    Learn the best peak threshold on the validation data
    Compute the score on the test data 
    """
    # Get the data
    # (dataset_train, dataset_val, valFullGen, trainTracksCount, valTracksCount, testFullNamedGen) = DataLoader.factoryADTOF(
    #     args.folderPath, testFold=fold, **hparams
    # )

    (dataset_train, dataset_val, valFullGen, trainTracksCount, valTracksCount, testFullNamedGen,) = DataLoader.factoryPublicDatasets(
        args.folderPath, testFold=fold, **hparams
    )

    if not model.weightLoadedFlag:  # if model is not trained, do the fitting
        # number of minibatches per epoch = number of tracks * samples per tracks / samples per bacth
        # This is not really an epoch, since we do see all the tracks, but only a few sample of each track
        steps_per_epoch = trainTracksCount * hparams["samplePerTrack"] / hparams["batchSize"] * hparams["training_epoch"]
        validation_steps = valTracksCount * hparams["samplePerTrack"] / hparams["batchSize"] * hparams["validation_epoch"]
        model.fit(dataset_train, dataset_val, tensorboardLogs, steps_per_epoch, validation_steps, **hparams)

    if os.path.exists(hparamsLogs + model.name):  # If the model is already evaluated, skip the evaluation
        logging.info("Skipping evaluation of model %s", model.name)
        results = None
    else:
        logging.info("Evaluating model %s", model.name)
        # model.vizPredictions(dataset_train, **hparams)

        results = {}
        if "peakThreshold" in hparams:  # Predict "peakThreshold" on validation data
            del hparams["peakThreshold"]
        scoreVal = model.evaluate(valFullGen(), **hparams)
        logging.info("Best PeakThreshold is " + str(scoreVal["peakThreshold"]))
        hparams["peakThreshold"] = scoreVal["peakThreshold"]
        for k, v in scoreVal.items():
            results["validation_" + k] = v

        # scoreTest = model.evaluate(testFullGen(), **hparams)
        for dataset, gen in testFullNamedGen.items():
            scoreTest = model.evaluate(gen(), **hparams)
            for k, v in scoreTest.items():
                results[dataset + "_" + k] = v

        logging.info(str(results))

    # return results


if __name__ == "__main__":
    main()
