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
    dataset = dataset.batch(1200).repeat()
    dataset_test = dataset_test.batch(1200).repeat()

    model = RV1().createModel()
    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=2, verbose=1),
        tf.keras.callbacks.ModelCheckpoint("models/rv1", load_weights_on_restart=True, save_weights_only=True)
    ]

    # model.fit(dataset,
    #           epochs=50,
    #           steps_per_epoch=1000,
    #           callbacks=callbacks,
    #           validation_data=dataset_test,
    #           validation_steps=10,
    #           )

    predictions = model.predict(dataset_test, steps=1)
    import matplotlib.pyplot as plt
    plt.plot(predictions)
    plt.show()
    print("Done!")


if __name__ == '__main__':
    main()
