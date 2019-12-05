#!/usr/bin/env python
# encoding: utf-8
"""
WIP: evaluate an algorithm
"""
import argparse
import csv
import logging
import os
from collections import defaultdict

import mir_eval
import numpy as np

from adtof import config
from adtof.io.converters import OnsetsAlignementConverter, TextConverter
from adtof.io.myMidi import MidoProxy as midi  # mido seems faster here


def main():
    # load the arguments
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('inputFolder', type=str, help="Path to the chart folder.")
    args = parser.parse_args()
    # parameters of the evaluation
    removeStart = True
    window = 0.03
    onsetOffset = True

    # load the file path
    gtPaths = config.getFilesInFolder(args.inputFolder, config.MIDI_CONVERTED)
    estimationPaths = [config.getFilesInFolder(args.inputFolder, algo) for algo in config.THREE_CLASS_EVAL]
    offsetPaths = config.getFilesInFolder(args.inputFolder, config.OD_OFFSET)
    assert len(gtPaths) == len(estimationPaths[0])

    # eval
    tc = TextConverter()
    meanResults = {}
    sumResults = {}
    for algoI, algo in enumerate(estimationPaths):
        algoName = config.THREE_CLASS_EVAL[algoI]
        meanResults[algoName] = {40: defaultdict(list), 36: defaultdict(list), 46: defaultdict(list)}
        sumResults[algoName] = {40: defaultdict(int), 36: defaultdict(int), 46: defaultdict(int)}
        for i, _ in enumerate(gtPaths):
            gt = midi(gtPaths[i]).getOnsets(separated=True)
            estimations = tc.getOnsets(algo[i], separated=True)

            # Clean the files
            if removeStart:
                firstOnset = min([v[0] for k, v in gt.items()])
                #TODO: Slow implementation
                estimations = {k: [t for t in v if t > firstOnset] for k, v in estimations.items()}

            if onsetOffset:
                with open(offsetPaths[i], mode="r") as csvFile:
                    reader = csv.reader(csvFile)
                    offset = float(list(reader)[1][2])  #TODO Read second row, second column
                gt = {k: [t + offset for t in v] for k, v in gt.items()}

            for pitch in meanResults[algoName].keys():
                y_truth = np.array(gt[pitch]) if pitch in gt else np.array([])
                y_pred = np.array(estimations[pitch]) if pitch in estimations else np.array([])

                tp = len(mir_eval.util.match_events(y_truth, y_pred, window))
                fp = len(y_pred) - tp
                fn = len(y_truth) - tp
                f, p, r = getF(tp, fp, fn)
                meanResults[algoName][pitch]["F"].append(f)
                meanResults[algoName][pitch]["P"].append(p)
                meanResults[algoName][pitch]["R"].append(r)
                sumResults[algoName][pitch]["TP"] += tp
                sumResults[algoName][pitch]["FP"] += fp
                sumResults[algoName][pitch]["FN"] += fn
                if f == 0:

                    print(config.getFileBasename(gtPaths[i]), pitch, p, r, f)

    for algoName in meanResults:
        print(algoName)
        print("mean fm", np.mean([np.array(pitch["F"]) for pitch in meanResults[algoName].values()]))
        print(
            "sum fm",
            getF(np.sum([pitch["TP"] for pitch in sumResults[algoName].values()]),
                    np.sum([pitch["FP"] for pitch in sumResults[algoName].values()]),
                    np.sum([pitch["FN"] for pitch in sumResults[algoName].values()]))[0])
        print("mean KD fm", np.mean(meanResults[algoName][36]["F"]))
        print("sum KD fm",
                getF(sumResults[algoName][36]["TP"], sumResults[algoName][36]["FP"], sumResults[algoName][36]["FN"])[0])
        print("mean SD fm", np.mean(meanResults[algoName][40]["F"]))
        print("sum SD fm",
                getF(sumResults[algoName][40]["TP"], sumResults[algoName][40]["FP"], sumResults[algoName][40]["FN"])[0])
        print("mean HH fm", np.mean(meanResults[algoName][46]["F"]))
        print("sum HH fm",
                getF(sumResults[algoName][46]["TP"], sumResults[algoName][46]["FP"], sumResults[algoName][46]["FN"])[0])
        
        import matplotlib.pyplot as plt
        plt.hist(meanResults[algoName][36]["F"])
        plt.show()
    print("Done!")


def hit(y_truth: np.array, y_pred: np.array, window: float):
    """
    bla
    """
    count = 0
    cursor = 0
    for a in y_pred:
        for i, c in enumerate(y_truth[cursor:]):
            if np.abs(a - c) <= window:
                count += 1
                cursor += i
                continue
    return count


def getF(tp, fp, fn):
    if tp == 0:
        return 0, 0, 0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f = 2 * tp / (2 * tp + fp + fn)
    return f, p, r


if __name__ == '__main__':
    main()
