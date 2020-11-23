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
    parser.add_argument("folderPath", type=str, help="Path.")
    args = parser.parse_args()

    for fold in range(3):
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

    # score = test_enssemble_models(args)
    # logging.info(str(score))
    # with tf.summary.create_file_writer(hparamsLogs + "ensemble model").as_default():
    #     # hp.hparams(
    #     #     {k: v if isinstance(v, (bool, float, int, six.string_types)) else str(v) for k, v in hparams.items()}, trial_id=model.name,
    #     # )
    #     for key, value in score.items():
    #         tf.summary.scalar(key, value, step=0)


def test_enssemble_models(args):
    """
    TODO factorise
    Used to evaluate an ensemble of models
    """
    models = [list(Model.modelFactory(fold=fold))[0][0] for fold in range(2)]
    hparams = list(Model.modelFactory(fold=0))[0][1]
    dl = DataLoader(args.folderPath, **hparams)

    trainGen, valGen, valFullGen, testFullGen = dl.getTrainValTestGens(validationFold=0, **hparams)
    predictions = []
    Y = []
    for i, (x, y) in enumerate(testFullGen()):
        predictions.append(Model.predictEnsemble(models, x))
        Y.append(y)

    return peakPicking.fitPeakPicking(predictions, Y, peakPickingSteps=[0.3], **hparams)


def train_test_model(hparams, args, fold, model):
    """
    Train procedure for one model with one set of hparam for one fold.
    Learn the best peak threshold on the validation data
    Compute the score on the test data 
    """
    # Get the data
    dl = DataLoader(args.folderPath, **hparams)
    trainGen, valGen, valFullGen, testFullGen = dl.getTrainValTestGens(validationFold=fold, **hparams)
    dataset_train = dl.getDataset(trainGen, **hparams)
    dataset_val = dl.getDataset(valGen, **hparams)
    dataset_train = dataset_train.batch(hparams["batchSize"]).repeat()
    dataset_val = dataset_val.batch(hparams["batchSize"]).repeat()
    dataset_train = dataset_train.prefetch(buffer_size=2)
    dataset_val = dataset_val.prefetch(buffer_size=2)

    # if model is not trained, do the fitting
    if not model.weightLoadedFlag:
        # number of minibatches per epoch = number of tracks * samples per tracks / samples per bacth
        # This is not really an epoch, since we do see all the tracks, but only a few sample of each tracks
        # limit #steps just to make sure that it progresses
        train, val, test = dl.getSplit(**hparams)
        steps_per_epoch = len(train) * hparams["samplePerTrack"] / hparams["batchSize"]
        maxStepPerEpoch = 300
        if steps_per_epoch > maxStepPerEpoch:
            logging.info("The step per epoch is set at %s, seing all tracks would really take %s steps", maxStepPerEpoch, steps_per_epoch)
            steps_per_epoch = maxStepPerEpoch
        validation_steps = min(len(val) * hparams["samplePerTrack"] / hparams["batchSize"], maxStepPerEpoch)
        model.fit(dataset_train, dataset_val, tensorboardLogs, steps_per_epoch, validation_steps, **hparams)
        # TODO: need to call reset state?
    # If the model is already evaluated, skip the evaluation
    # Predict on validation data
    if os.path.exists(hparamsLogs + model.name):
        logging.info("Skipping evaluation of model %s", model.name)
        return None
    else:
        logging.info("Evaluating model %s", model.name)
        # model.vizPredictions(dataset_train, **hparams)
        scoreVal = model.evaluate(valFullGen, **hparams)
        logging.info("Best PeakThreshold is " + str(scoreVal["peakThreshold"]))
        hparams["peakThreshold"] = scoreVal["peakThreshold"]
        scoreTest = model.evaluate(testFullGen, **hparams)

        # Merge the validation results for Hparam selection
        for k, v in scoreVal.items():
            scoreTest["validation_" + k] = v
        return scoreTest


if __name__ == "__main__":
    main()
