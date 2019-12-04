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
    results = {}
    for algoI, algo in enumerate(estimationPaths):
        algoName = config.THREE_CLASS_EVAL[algoI]
        results[algoName] = {40: defaultdict(list), 36: defaultdict(list), 46: defaultdict(list)}
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

            for pitch in results[algoName].keys():
                if pitch not in estimations or pitch not in gt:
                    f, p, r = 0, 0, 0
                else:
                    f, p, r = mir_eval.onset.f_measure(np.array(gt[pitch]), np.array(estimations[pitch]), window=window)
                results[algoName][pitch]["F"].append(f)
                results[algoName][pitch]["P"].append(p)
                results[algoName][pitch]["R"].append(r)
                print(config.getFileBasename(gtPaths[i]), pitch, p, r, f)

    for algoName in results:
        print(algoName)
        print("mean F", np.mean([np.array(pitch["F"]) for pitch in results[algoName].values()]))
        print("mean KD", np.mean(results[algoName][36]["F"]))
        print("mean SD", np.mean(results[algoName][40]["F"]))
        print("mean HH", np.mean(results[algoName][46]["F"]))
    print("Done!")


if __name__ == '__main__':
    main()
