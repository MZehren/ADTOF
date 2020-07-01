#!/usr/bin/env python
# encoding: utf-8
"""
TODO
"""
import argparse
import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import sklearn
import tensorflow as tf

from adtof import config
from adtof.deepModels import dataLoader
from adtof.deepModels.peakPicking import PeakPicking
from adtof.deepModels.rv1tf import RV1TF
from adtof.converters.converter import Converter
from adtof.io.mir import MIR

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# tf.config.experimental_run_functions_eagerly(True)
logging.basicConfig(filename="logs/conversion.log", level=logging.DEBUG)


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    parser = argparse.ArgumentParser(description="todo")
    parser.add_argument("inputPath", type=str, help="Path to music or folder containing music to transcribe")
    parser.add_argument("outputPath", type=str, default="./", help="Path to output folder")
    args = parser.parse_args()
    # TODO: save the meta parameters in a config file
    sampleRate = 100
    context = 25
    classLabels = [36]
    plot = False

    # Get the model
    model = RV1TF().createModel(output=len(classLabels))
    checkpoint_path = "models/rv1.ckpt"  # TODO make the path relative to the code and not the shell
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest:
        model.load_weights(latest)

    # Get the data
    tracks = config.getFilesInFolder(args.inputPath)
    tracks = tracks[int(len(tracks) * 0.85) :]
    mir = MIR(frameRate=sampleRate)

    # Predict the file and write the output
    for track in tracks:
        if not os.path.exists(args.outputPath):
            os.makedirs(args.outputPath)
        if os.path.exists(os.path.join(args.outputPath, config.getFileBasename(track) + ".txt")):
            continue

        X = mir.open(track)
        X = X.reshape(X.shape + (1,))
        Y = model.predict(np.array([X[i : i + context] for i in range(len(X) - context)]))

        # TODO make it work for matrix
        sparseResultIdx = [PeakPicking().serialPeakPicking(Y[:, column]) for column in range(Y.shape[1])]

        # write text
        with open(os.path.join(args.outputPath, config.getFileBasename(track) + ".txt"), "w") as outputFile:
            outputFile.write(
                "\n".join(
                    [
                        str(i / sampleRate) + "\t" + str(classLabels[classIdx])
                        for classIdx, classPeaks in enumerate(sparseResultIdx)
                        for i in classPeaks
                    ]
                )
            )

        #  write midi
        midi = pretty_midi.PrettyMIDI()
        instrument = cello = pretty_midi.Instrument(program=1, is_drum=True)
        midi.instruments.append(instrument)
        for classi, notes in enumerate(sparseResultIdx):
            for i in notes:
                note = pretty_midi.Note(velocity=100, pitch=classLabels[classi], start=i / sampleRate, end=i / sampleRate)
                instrument.notes.append(note)
        midi.write(os.path.join(args.outputPath, config.getFileBasename(track) + ".mid"))

        # plot
        if plot:
            plt.imshow(X.reshape(X.shape[:-1]).T)
            plt.plot(Y * len(X[0]))
            denseResult = np.zeros(X.shape[0])
            for p in sparseResultIdx:
                denseResult[p] = X.shape[1]
            plt.plot(denseResult)
            plt.show()


if __name__ == "__main__":
    main()
