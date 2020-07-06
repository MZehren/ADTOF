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
    dataset = tf.data.Dataset.from_generator(
        dataLoader.getTFGenerator(args.folderPath, train=True, labels=labels, sampleRate=sampleRate, limitInstances=args.limit),
        (tf.float32, tf.float32),
        output_shapes=(tf.TensorShape((None, None, 1)), tf.TensorShape((len(labels),))),
    )
    dataset_test = tf.data.Dataset.from_generator(
        dataLoader.getTFGenerator(args.folderPath, train=False, labels=labels, sampleRate=sampleRate, limitInstances=args.limit),
        (tf.float32, tf.float32),
        output_shapes=(tf.TensorShape((None, None, 1)), tf.TensorShape((len(labels),))),
    )
    batch_size = 100
    dataset = dataset.batch(batch_size).repeat()
    dataset_test = dataset_test.batch(batch_size).repeat()
    # dataset = dataset.prefetch(buffer_size=batch_size // 2)
    # dataset_test = dataset_test.prefetch(buffer_size=batch_size // 2)

    # Get the model
    cwd = os.path.abspath(os.path.dirname(__file__))
    checkpoint_dir = os.path.join(cwd, "..", "models")
    checkpoint_path = os.path.join(checkpoint_dir, "rv1.ckpt")
    model = RV1TF().createModel(output=len(labels))
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest and not args.restart:
        model.load_weights(latest)

    # Set the logs
    all_logs = os.path.join(cwd, "..", "logs")
    log_dir = os.path.join(all_logs, "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    if args.deleteLogs:
        try:
            shutil.rmtree(all_logs)
        except:
            print("couldn't remove folder", all_logs)

    Converter.checkPathExists(all_logs)
    file_writer = tf.summary.create_file_writer(log_dir)

    # Get the debug activation model
    layer_outputs = [layer.output for layer in model.layers]  # Extracts the outputs of the top 12 layers
    activation_model = tf.keras.models.Model(
        inputs=model.input, outputs=layer_outputs
    )  # Creates a model that will return these outputs, given the model input
    viz_example, _ = next(dataLoader.getTFGenerator(args.folderPath, train=False, labels=labels, sampleRate=sampleRate)())
    viz_example = viz_example.reshape([1] + list(viz_example.shape))  # Adding mini-batch

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=True),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True,),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2)
        # tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=1),
        # tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: log_layer_activation(epoch, viz_example, model, activation_model, file_writer))
    ]

    model.fit(
        dataset,
        epochs=100,
        initial_epoch=0,
        steps_per_epoch=1,
        callbacks=callbacks,
        validation_data=dataset_test,
        validation_steps=30
        # class_weight=classWeight
    )

    # for x, y in dataset_test:
    #     predictions = model.predict(x)

    #     import matplotlib.pyplot as plt

    #     f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    #     ax1.plot(predictions)
    #     ax1.set_ylabel("Prediction")
    #     ax2.plot(y)
    #     ax2.set_ylabel("Truth")
    #     ax2.set_xlabel("Time step")
    #     plt.show()
    #     print("Done!")


if __name__ == "__main__":
    main()
