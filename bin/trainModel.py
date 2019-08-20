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
from adtof.io import CQT
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
    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(dataset,
              epochs=5,
              steps_per_epoch=1000,
              callbacks=[tensorboard_callback],
              validation_data=dataset_test,
              validation_steps=10)
    print("Done!")


if __name__ == '__main__':
    main()
