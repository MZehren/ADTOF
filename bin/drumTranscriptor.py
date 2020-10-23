#!/usr/bin/env python
# encoding: utf-8
"""
TODO
"""
import argparse
from bin.trainModelTF import hp
import datetime
import logging
import os

import madmom
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import sklearn
from tensorboard.plugins.hparams.summary_v2 import hparams
import tensorflow as tf

from adtof import config
from adtof.converters.converter import Converter
from adtof.model.dataLoader import DataLoader
from adtof.model.model import Model
from adtof.model import peakPicking
from adtof.io.textReader import TextReader

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
    model, hparams = next(Model.modelFactory(fold=0))
    ppp = madmom.features.notes.NotePeakPickingProcessor(
        threshold=0.3, smooth=0, pre_avg=0.1, post_avg=0.01, pre_max=0.02, post_max=0.01, combine=0.02, fps=hparams["sampleRate"]
    )
    # Get the data
    dl = DataLoader(args.inputPath, loadLabels=False)
    hparams["samplePerTrack"] = None
    tracks = dl.getGen(repeat=False, yDense=False, **hparams)

    # Predict the file and write the output
    for (x, _), track in zip(tracks(), dl.audioPaths):
        if not os.path.exists(args.outputPath):
            os.makedirs(args.outputPath)
        outputTrackPath = os.path.join(args.outputPath, config.getFileBasename(track) + ".txt")
        if os.path.exists(outputTrackPath):
            continue

        Y = model.predict(x, **hparams)
        sparseResultIdx = peakPicking.peakPicking(Y, ppProcess=ppp, timeOffset=hparams["labelOffset"] / hparams["sampleRate"], **hparams)

        # write text
        formatedOutput = [(time, pitch) for pitch, times in sparseResultIdx.items() for time in times]
        formatedOutput.sort(key=lambda x: x[0])
        TextReader().writteBeats(outputTrackPath, formatedOutput)

        #  write midi
        midi = pretty_midi.PrettyMIDI()
        instrument = cello = pretty_midi.Instrument(program=1, is_drum=True)
        midi.instruments.append(instrument)
        for pitch, notes in sparseResultIdx.items():
            for i in notes:
                note = pretty_midi.Note(velocity=100, pitch=pitch, start=i, end=i)
                instrument.notes.append(note)
        midi.write(os.path.join(args.outputPath, config.getFileBasename(track) + ".mid"))

        # # plot
        # if plot:
        #     plt.imshow(X.reshape(X.shape[:-1]).T)
        #     plt.plot(Y * len(X[0]))
        #     denseResult = np.zeros(X.shape[0])
        #     for p in sparseResultIdx:
        #         denseResult[p] = X.shape[1]
        #     plt.plot(denseResult)
        #     plt.show()


if __name__ == "__main__":
    main()
