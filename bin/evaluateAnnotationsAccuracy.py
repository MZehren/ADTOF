#!/usr/bin/env python
# encoding: utf-8
"""
TODO
"""
import argparse
import logging

import matplotlib.pyplot as plt


from adtof import config

# from adtof.converters.correctAlignmentConverter import CorrectAlignmentConverter
from adtof.io.textReader import TextReader
from adtof.model.model import Model
from adtof.model.dataLoader import DataLoader
import os

cwd = os.path.abspath(os.path.dirname(__file__))
all_logs = os.path.join(cwd, "..", "logs/")
logging.basicConfig(filename=os.path.join(all_logs, "evalAnnotations.log"), level=logging.DEBUG, filemode="w")


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion

    ideas for prunning the bad annotations:
    - FMeasure too low between the beats annotated and estimated with madmom: 
    If there is an octave issue, the precision or recall will go down to 0.5 at most, but the fMeasure can be above
    - Standard deviation. if the std is too high, it means that the annotations are constantly shifting back and forth compared to the estimations
    Thus, it is not a constant offset through the whole track which allow a better correction
    - Instant std: removing the tracks which are varying a lot in a small time frame
    - residual error after correction: The correction is smoothed on a 5s window, 
    checking if any track after the correction has still error could be a good idea to remove it
    """
    parser = argparse.ArgumentParser(description="todo")
    parser.add_argument("folderPath", type=str, help="Path.")
    # parser.add_argument("-d", "--distance", type=float, default=0.05, help="Hit rate distance for the precision and recall")
    args = parser.parse_args()

    # evalCAC(args.folderPath)
    evalADT(args.folderPath)


def evalADT(folderPath):
    """
    Check suspicious tracks having a low ADT score
    """
    model, hparams = next(Model.modelFactory())
    dl = DataLoader(folderPath, **hparams)
    # TODO: factorise this code
    fullGenParams = {k: v for k, v in hparams.items()}
    fullGenParams["repeat"] = False
    fullGenParams["samplePerTrack"] = None
    fullGenParams["yDense"] = False
    fullGen = dl.getGen(**fullGenParams)
    model.evaluate(fullGen, **hparams, paths=dl.audioPaths)


def evalCAC(folderPath):
    """
    Evaluate the alignment corrrection converter by looking at the improvement in F measure of the beat detection
    """
    annotatedMidis = config.getFilesInFolder(folderPath, config.CONVERTED_MIDI)
    estimatedBeats = config.getFilesInFolder(folderPath, config.BEATS_ESTIMATIONS)
    annotatedMidis, estimatedBeats = config.getIntersectionOfPaths(annotatedMidis, estimatedBeats)

    cac = CorrectAlignmentConverter()
    tr = TextReader()
    qualities = []
    for annotatedMidiPath, estimatedBeatPath in zip(annotatedMidis, estimatedBeats):
        # qualities5s.append(cac.convert(estimatedBeatPath, annotatedMidiPath, "", "", "", smoothingCorrectionWindow=5))
        qualities.append(
            cac.convert(estimatedBeatPath, annotatedMidiPath, "", "", "", smoothingCorrectionWindow=10, thresholdCorrectionWindow=0.1)
        )

    boxplots([qualities], ["F with 5s smoothing and 10s"])
    # boxplots([precisions, recalls, fMeasures], ["precision, recall, fMeasure"])
    # boxplots([[v for v in meanDiffs if not np.isnan(v)], [v for v in stds if not np.isnan(v)]], ["meanDiffs", "meanStds"])
    # scatterPlot(fMeasures, stds, "fMeasures", "stds")


def boxplots(values, labels):
    fig, ax = plt.subplots()
    ax.set_title("box plot of " + ", ".join(labels))
    ax.boxplot(values)
    plt.show()


def scatterPlot(x, y, xLabel, yLabel):
    plt.scatter(x, y)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()


if __name__ == "__main__":
    main()
