#!/usr/bin/env python
# encoding: utf-8
"""
TODO
"""
import argparse
import datetime
import os

import numpy as np
import sklearn
import tensorflow as tf

from adtof.deepModels import RV1
from adtof.io import CQT
from adtof.io.converters import Converter, PhaseShiftConverter


def loadData(path, context=25, limit=None):
    psc = PhaseShiftConverter()
    cqt = CQT()

    X = None
    Y = None
    i = 0
    for root, dirs, files in os.walk(path):
        # fullPath = os.sep.join(path)
        midi, audio = psc.getConvertibleFiles(root)
        if audio and midi:
            try:
                # Get the midi in dense matrix representation
                y = psc.convert(root).getDenseEncoding(sampleRate=98.4375, timeShift=0)

                # Get the CQT with a context
                x = cqt.open(os.sep.join([root, audio]))
                x = np.array([x[i:i + context] for i in range(len(x) - context)])

                # Add the channel dimension
                x = x.reshape(x.shape + (1,))

                x = x[:min(len(y), len(x))]
                y = y[:min(len(y), len(x))]
                X = np.concatenate((X, x), axis=0) if X is not None else x
                Y = np.concatenate((Y, y), axis=0) if Y is not None else y
                print(root + " OK")
                i += 1
            except Exception as e: 
                print(root, e)
        
        if limit is not None and i >= limit:
            break
    # x = fe.open(path)
    # y = []

    return X, Y


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    parser = argparse.ArgumentParser(description='todo')
    parser.add_argument('folderPath', type=str, help="Path.")
    args = parser.parse_args()

    Converter.convertAll(args.folderPath)
    exit(1)
    model = RV1().createModel()
    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    X, Y = loadData(args.folderPath, limit=3)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.33, random_state=42)
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.batch(2048).repeat()
    dataset_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    dataset_test = dataset.batch(2048).repeat()

    model.fit(dataset, epochs=5, steps_per_epoch=30, callbacks=[tensorboard_callback], validation_data=dataset_test)

    print("Done!")


if __name__ == '__main__':
    main()

