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

from adtof import config
from adtof.converters.converter import Converter
from adtof.deepModels import dataLoader
from adtof.deepModels.peakPicking import PeakPicking
from adtof.deepModels.rv1tf import RV1TF
from adtof.io import mir

tf.config.threading.set_intra_op_parallelism_threads(32)
tf.config.threading.set_inter_op_parallelism_threads(32)
# tf.config.experimental_run_functions_eagerly(True)
if not os.path.exists("logs"):
    os.makedirs("logs")
logging.basicConfig(filename="logs/training.log", level=logging.DEBUG)


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

    paramGrid = {
        "labels": [config.LABELS_5],
        "classWeights": [config.WEIGHTS_5],
        "sampleRate": [100],
        "diff": [False],
        "samplePerTrack": [100],
        "batchSize": [100],
        "context": [25],
        "labelOffset": [0],
        "labelRadiation": [1],
        "learningRate": [0.0005],
    }

    # Set the logs
    cwd = os.path.abspath(os.path.dirname(__file__))
    all_logs = os.path.join(cwd, "..", "logs")
    if args.restart and os.path.exists(all_logs):
        try:
            shutil.rmtree(all_logs)
        except:
            logging.warning("Couldn't remove folder %s", all_logs)
    Converter.checkPathExists(all_logs)

    for paramIndex, params in enumerate(list(sklearn.model_selection.ParameterGrid(paramGrid))):
        for fold in range(2):
            # Get the data TODO: the buffer is getting destroyed after each fold
            trainGen, valGen, testGen = dataLoader.getSplit(args.folderPath, randomState=fold, limit=args.limit, **params)

            dataset_train = tf.data.Dataset.from_generator(
                trainGen,
                (tf.float32, tf.float32, tf.float32),
                output_shapes=(tf.TensorShape((None, None, 1)), tf.TensorShape((len(params["labels"]),)), tf.TensorShape(1)),
            )
            dataset_val = tf.data.Dataset.from_generator(
                valGen,
                (tf.float32, tf.float32, tf.float32),
                output_shapes=(tf.TensorShape((None, None, 1)), tf.TensorShape((len(params["labels"]),)), tf.TensorShape(1)),
            )
            dataset_train = dataset_train.batch(params["batchSize"]).repeat()
            dataset_val = dataset_val.batch(params["batchSize"]).repeat()
            # dataset_train = dataset_train.prefetch(buffer_size=batch_size // 2)
            # dataset_val = dataset_val.prefetch(buffer_size=batch_size // 2)

            # Get the model
            checkpoint_dir = os.path.join(cwd, "..", "models")
            checkpoint_path = os.path.join(checkpoint_dir, "rv1.ckpt")
            nBins = 168 if params["diff"] else 84
            model = RV1TF().createModel(n_bins=nBins, output=len(params["labels"]), **params)
            model.summary()
            latest = tf.train.latest_checkpoint(checkpoint_dir)
            # if latest and not args.restart:
            #     model.load_weights(latest)

            # Set the log
            log_dir = os.path.join(
                all_logs, "fit", datetime.datetime.now().strftime("%m%d-%H%M") + "-paramIndex:" + str(paramIndex) + "-fold:" + str(fold)
            )

            # Fit the model
            callbacks = [
                tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=True),
                # tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True,),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.2)
                # tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=1),
                # tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: log_layer_activation(epoch, viz_example, model, activation_model, file_writer))
            ]

            model.fit(
                dataset_train,
                epochs=100,
                initial_epoch=0,
                steps_per_epoch=100,
                callbacks=callbacks,
                validation_data=dataset_val,
                validation_steps=100
                # class_weight=classWeight
            )

            # for x, y, w in dataset_val:
            #     predictions = model.predict(x)

            #     import matplotlib.pyplot as plt

            #     f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

            #     ax1.plot(predictions)
            #     ax1.set_ylabel("Prediction")
            #     ax2.plot(y)
            #     ax2.set_ylabel("Truth")
            #     ax2.set_xlabel("Time step")
            #     ax3.matshow(tf.transpose(tf.reshape(x[:, 0], (params["batchSize"], nBins))), aspect="auto")
            #     ax1.legend(params["labels"])
            #     plt.show()


if __name__ == "__main__":
    main()
