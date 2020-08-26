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
from adtof.deepModels.dataLoader import DataLoader
from adtof.deepModels.peakPicking import PeakPicking
from adtof.deepModels.rv1tf import RV1TF
from adtof.io import eval

# TODO: needed because error is thrown:
# Check failed: ret == 0 (11 vs. 0)Thread creation via pthread_create() failed.
# See: https://github.com/tensorflow/tensorflow/issues/41532
tf.config.threading.set_intra_op_parallelism_threads(32)
tf.config.threading.set_inter_op_parallelism_threads(32)
# tf.config.experimental_run_functions_eagerly(True)

# When tf.config.threading.set_intra_op_parallelism_threads(32) and tf 2.2
# LLVM ERROR: out of memory

# When tf.config.threading.set_intra_op_parallelism_threads(32) and tf 2.3
# tensorflow terminate called after throwing an instance of 'std::bad_alloc'

# When tf.config.threading.set_intra_op_parallelism_threads(1)
# tensorflow.python.framework.errors_impl.ResourceExhaustedError:  OOM when allocating tensor with shape[100,21,164,32] and type float on /job:localhost/replica:0/task:0/device:CPU:0 by allocator cpu
#         [[node sequential_1/conv12/Relu (defined at bin/trainModelTF.py:180) ]]
# Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info.
#  [Op:__inference_test_function_8151]
# Function call stack:
# test_function
# 2020-08-17 17:57:46.592177: W tensorflow/core/kernels/data/generator_dataset_op.cc:103] Error occurred when finalizing GeneratorDataset iterator: Failed precondition: Python interpreter state is not initialized. The process may be terminated.
#         [[{{node PyFunc}}]]

# When tf.config.threading.set_intra_op_parallelism_threads(64) and tf 2.3
# tensorflow terminate called after throwing an instance of 'std::bad_alloc'


# Setup logs
cwd = os.path.abspath(os.path.dirname(__file__))
all_logs = os.path.join(cwd, "..", "logs/")
tensorboardLogs = os.path.join(all_logs, "fit/")
hparamsLogs = os.path.join(all_logs, "hparam/")
checkpoint_dir = os.path.join(cwd, "..", "models")
Converter.checkPathExists(all_logs)
logging.basicConfig(filename=os.path.join(all_logs, "training.log"), level=logging.DEBUG, filemode="w")

paramGrid = [
    (
        "cnn-offset5",
        {
            "labels": config.LABELS_5,
            "classWeights": config.WEIGHTS_5 / 2,
            "sampleRate": 100,
            "diff": True,
            "samplePerTrack": 20,
            "batchSize": 100,
            "context": 25,
            "labelOffset": 5,
            "labelRadiation": 2,
            "learningRate": 0.0001,
            "normalize": False,
            "model": "CNN",
            "fmin": 20,
            "fmax": 20000,
            "pad": False,
            "beat_targ": False,
        },
    )
    # "crnn",
    # {
    #     "labels": config.LABELS_5,
    #     "classWeights": config.WEIGHTS_5 / 2,
    #     "sampleRate": 100,
    #     "diff": True,
    #     "samplePerTrack": 60000,
    #     "batchSize": 8,
    #     "context": 13,
    #     "labelOffset": 0,
    #     "labelRadiation": 2,
    #     "learningRate": 0.0001,
    #     "normalize": False,
    #     "model": "CNN",
    #     "fmin": 20,
    #     "fmax": 20000,
    #     "pad": False,
    #     "beat_targ": False,
    # },
    # ),
]


def removeFolder(path):
    """
    Delete the content of the folder at path
    """
    if os.path.exists(tensorboardLogs):
        try:
            shutil.rmtree(path, ignore_errors=True)
            logging.info("Removing path %s", path)
        except Exception as e:
            logging.warning("Couldn't remove folder %s \n%s", path, e)
    Converter.checkPathExists(path)  # Create the base folder


def train_test_model(hparams, args, fold, modelName):
    """
    TODO 
    """
    # Get the data TODO: the buffer is getting destroyed after each fold
    trainGen, valGen, valFullGen = DataLoader(args.folderPath).getThreeSplitGen(validationFold=fold, limit=args.limit, **hparams)
    dataset_train = tf.data.Dataset.from_generator(
        trainGen,
        (tf.float32, tf.float32, tf.float32),
        output_shapes=(tf.TensorShape((None, None, 1)), tf.TensorShape((len(hparams["labels"]),)), tf.TensorShape(1)),
    )
    dataset_val = tf.data.Dataset.from_generator(
        valGen,
        (tf.float32, tf.float32, tf.float32),
        output_shapes=(tf.TensorShape((None, None, 1)), tf.TensorShape((len(hparams["labels"]),)), tf.TensorShape(1)),
    )
    dataset_train = dataset_train.batch(hparams["batchSize"]).repeat()
    dataset_val = dataset_val.batch(hparams["batchSize"]).repeat()
    dataset_train = dataset_train.prefetch(buffer_size=2)
    dataset_val = dataset_val.prefetch(buffer_size=2)

    # Set the log
    log_dir = os.path.join(tensorboardLogs, modelName + "_" + datetime.datetime.now().strftime("%d-%m-%H:%M"),)
    # Get the model
    checkpoint_path = os.path.join(checkpoint_dir, modelName + ".ckpt")
    nBins = 168 if hparams["diff"] else 84
    modelHandler = RV1TF(**hparams)
    model = modelHandler.createModel(n_bins=nBins, output=len(hparams["labels"]), **hparams)

    # if model is already trained, load the weights else fit
    if os.path.exists(checkpoint_path + ".index") and not args.restart:
        logging.info("Loading model weights %s", checkpoint_path)
        model.load_weights(checkpoint_path)
    else:
        logging.info("Training model %s", modelName)
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_images=False),
            tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True,),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.2),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=30, verbose=1, restore_best_weights=True),
            # tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: log_layer_activation(epoch, viz_example, model, activation_model, file_writer))
        ]
        model.fit(
            dataset_train,
            epochs=200,
            initial_epoch=0,
            steps_per_epoch=100,
            callbacks=callbacks,
            validation_data=dataset_val,
            validation_steps=100
            # class_weight=classWeight
        )

    # If the model is already evaluated, skip the evaluation
    # Predict on validation data
    if os.path.exists(hparamsLogs + modelName) and not args.restart:
        logging.info("Skipping evaluation of model %s", modelName)
        return None
    else:
        logging.info("Evaluating model %s", modelName)
        YHat, Y = np.array([[modelHandler.predictWithPP(model, x, **hparams), y] for x, y in valFullGen()]).T
        score = eval.runEvaluation(Y, YHat)
        return score


def vizPredictions(dataset, model, params, nBins):
    """
    Plot the input, output and target of the model
    """
    for x, y, w in dataset:
        predictions = model.predict(x)
        import matplotlib.pyplot as plt

        f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        ax1.plot(predictions)
        ax1.set_ylabel("Prediction")
        ax2.plot(y)
        ax2.set_ylabel("Truth")
        ax2.set_xlabel("Time step")
        ax3.matshow(tf.transpose(tf.reshape(x[:, 0], (params["batchSize"], nBins))), aspect="auto")
        ax1.legend(params["labels"])
        plt.show()


def main():
    """
    Entry point of the program
    """
    parser = argparse.ArgumentParser(description="todo")
    parser.add_argument("folderPath", type=str, help="Path.")
    parser.add_argument(
        "-r", "--restart", action="store_true", help="Override the model and logs if present. Default is to resume training"
    )
    parser.add_argument("-l", "--limit", type=int, default=None, help="Limit the number of tracks used in training and eval")
    args = parser.parse_args()

    if args.restart and os.path.exists(tensorboardLogs):
        removeFolder(tensorboardLogs)
        removeFolder(hparamsLogs)
        removeFolder(checkpoint_dir)

    for modelName, params in paramGrid:
        for fold in range(2):
            modelNameComp = modelName + "_Limit" + str(args.limit) + "_Fold" + str(fold)
            score = train_test_model(params, args, fold, modelNameComp)

            if score is not None:
                with tf.summary.create_file_writer(hparamsLogs + modelNameComp).as_default():
                    hp.hparams(
                        {k: v if isinstance(v, (bool, float, int, six.string_types)) else str(v) for k, v in params.items()},
                        trial_id=modelNameComp,
                    )
                    for key, value in score.items():
                        tf.summary.scalar(key, value, step=fold)


if __name__ == "__main__":
    main()
