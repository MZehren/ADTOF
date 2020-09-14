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
from adtof.converters.converter import Converter
from adtof.model.dataLoader import DataLoader
from adtof.model.model import Model

# TODO: needed because error is thrown:
# Check failed: ret == 0 (11 vs. 0)Thread creation via pthread_create() failed.
# See: https://github.com/tensorflow/tensorflow/issues/41532
tf.config.threading.set_intra_op_parallelism_threads(32)
tf.config.threading.set_inter_op_parallelism_threads(32)


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    parser = argparse.ArgumentParser(description="todo")
    parser.add_argument("inputPath", type=str, help="Path to music or folder containing music to transcribe")
    parser.add_argument("outputPath", type=str, default="./", help="Path to output folder")
    args = parser.parse_args()

    # Get the model
    models = [Model.modelFactory(fold=fold)[0] for fold in range(2)]
    ppp = madmom.features.notes.NotePeakPickingProcessor(
        threshold=peakThreshold, smooth=0, pre_avg=0.1, post_avg=0.01, pre_max=0.02, post_max=0.01, combine=0.02, fps=sampleRate
    )
    # Get the data
    dl = DataLoader(args.inputPat, loadLabels=False)
    tracks = dl.getGen(repeat=False, samplePerTrack=None, yDense=False)

    # Predict the file and write the output
    for track in tracks:
        if not os.path.exists(args.outputPath):
            os.makedirs(args.outputPath)
        if os.path.exists(os.path.join(args.outputPath, config.getFileBasename(track) + ".txt")):
            continue

        Y = Model.predictEnsemble(track)

        # TODO make it work for matrix
        sparseResultIdx = [PeakPicking().serialPeakPicking(Y[:, column]) for column in range(Y.shape[1])]

        # write text
        textFormatedResult = [
            (i / sampleRate, classLabels[classIdx]) for classIdx, classPeaks in enumerate(sparseResultIdx) for i in classPeaks
        ]
        textFormatedResult.sort(key=lambda x: (x[0], x[1]))
        with open(os.path.join(args.outputPath, config.getFileBasename(track) + ".txt"), "w") as outputFile:
            outputFile.write("\n".join([str(el[0]) + "\t" + str(el[1]) for el in textFormatedResult]))

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
