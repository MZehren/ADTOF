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

from adtof import config
from adtof.deepModels import dataLoader
from adtof.deepModels.peakPicking import PeakPicking
from adtof.deepModels.rv1tf import RV1TF
from adtof.io.mir import MIR
from adtof.io.converters.converter import Converter

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# tf.config.experimental_run_functions_eagerly(True)
logging.basicConfig(filename='logs/conversion.log', level=logging.DEBUG)


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    parser = argparse.ArgumentParser(description='todo')
    parser.add_argument('folderPath', type=str, help="Path.")
    args = parser.parse_args()
    # TODO: save the meta parameters in a costr(ig file
    sampleRate = 50
    context = 25
    classLabels = [35]

    # Get the model
    model = RV1TF().createModel(output=len(classLabels))
    checkpoint_path = "models/rv1.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest:
        model.load_weights(latest)

    # Get the data
    tracks = config.getFilesInFolder(args.folderPath, config.AUDIO)
    mir = MIR(frameRate=sampleRate)

    # Predict the file and write the output
    for track in tracks:
        X = mir.open(track)
        X = X.reshape(X.shape + (1, ))
        Y = model.predict(np.array([X[i:i + context] for i in range(len(X) - context)]))
        # plt.plot([i/sampleRate for i in range(len(Y))], Y)
        
        denseResult = PeakPicking()._dense_peak_picking(Y) # denseResult is of the shape:(timestep, class)
        sparseResult = [str(i / sampleRate) + "\t" + str(classLabels[j]) for i, y in enumerate(denseResult.numpy()) for j, isPeak in enumerate(y) if isPeak]
        with open(os.path.join(args.folderPath, "MZ-CNN_1", config.getFileBasename(track) + ".txt"), "w") as outputFile:
            outputFile.write("\n".join(sparseResult))

if __name__ == '__main__':
    main()
