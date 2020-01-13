#!/usr/bin/env python
# encoding: utf-8
"""
TODO
"""
import argparse
import datetime
import logging
import os

import numpy as np
import sklearn
import tensorflow as tf

from adtof.deepModels import dataLoader
from adtof.deepModels.peakPicking import PeakPicking
from adtof.deepModels.rv1tf import RV1TF
from adtof.io import mir
from adtof.io.converters.converter import Converter

logging.basicConfig(filename='logs/conversion.log', level=logging.DEBUG)


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    parser = argparse.ArgumentParser(description='todo')
    parser.add_argument('folderPath', type=str, help="Path.")
    args = parser.parse_args()

    # Get the data
    dataset = tf.data.Dataset.from_generator(
        dataLoader.getTFGenerator(args.folderPath, train=True), (tf.float64, tf.float64),
        output_shapes=(tf.TensorShape((None, None, 1)), tf.TensorShape((5, )))
    )
    dataset_test = tf.data.Dataset.from_generator(
        dataLoader.getTFGenerator(args.folderPath, train=False), (tf.float64, tf.float64),
        output_shapes=(tf.TensorShape((None, None, 1)), tf.TensorShape((5, )))
    )
    #     testDS = tf.data.Dataset.from_generator(generateGenerator(test), (tf.float64, tf.int64))
    #     return trainDS, testDS

    dataset = dataset.batch(100).repeat()
    dataset_test = dataset_test.batch(100).repeat()
    dataset = dataset.prefetch(buffer_size=100)
    dataset_test = dataset_test.prefetch(buffer_size=100)

    # Get the model
    model = RV1TF().createModel()
    checkpoint_path = "models/rv1.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest:
        model.load_weights(latest)

    log_dir = os.path.join("logs", "fit", "rv1.2")  #datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=False, profile_batch=0),
        # tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True)
    ]

    model.fit(
        dataset,
        epochs=60,
        steps_per_epoch=100,
        callbacks=callbacks,
        validation_data=dataset_test,
        validation_steps=10,
    )

    for x, y in dataset_test:
        predictions = model.predict(x)

        import matplotlib.pyplot as plt
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.plot(predictions)
        ax2.plot(y)
        plt.show()
        print("Done!")


if __name__ == '__main__':
    main()
