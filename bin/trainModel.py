#!/usr/bin/env python
# encoding: utf-8
"""
TODO
"""
import logging
logging.basicConfig(filename='logs/conversion.log', level=logging.DEBUG)

import argparse
import datetime
import os

import numpy as np
import sklearn
import tensorflow as tf

from adtof.deepModels import RV1
from adtof.io import MIR
from adtof.io.converters import Converter, PhaseShiftConverter


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    parser = argparse.ArgumentParser(description='todo')
    parser.add_argument('folderPath', type=str, help="Path.")
    args = parser.parse_args()

    dataset, dataset_test = Converter.convertAll(args.folderPath)
    dataset = dataset.batch(400).repeat()
    dataset_test = dataset_test.batch(400).repeat()

    model = RV1().createModel()
    log_dir = os.path.join("logs", "fit", "rv3") #datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = "models/rv3"

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=False),
        # tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True)
    ]
    # model.load_weights(checkpoint_path, )

    model.fit(dataset,
              epochs=60,
              steps_per_epoch=100,
              callbacks=callbacks,
              validation_data=dataset_test,
              validation_steps=10,
              )

    it = dataset_test.make_one_shot_iterator()
    x, y = next(it)
    predictions = model.predict(x)

    import matplotlib.pyplot as plt
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(predictions)
    ax2.plot(y)
    plt.show()
    print("Done!")


if __name__ == '__main__':
    main()
