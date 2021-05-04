#!/usr/bin/env python
# encoding: utf-8
"""
TODO
"""
import argparse
import datetime
import json
import logging
import os

import matplotlib.pyplot as plt
import mir_eval
import numpy as np
import pandas as pd
from adtof import config
from adtof.converters.converter import Converter
from adtof.io.mir import MIR
from adtof.io.textReader import TextReader
from adtof.model import eval


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    parser = argparse.ArgumentParser(description="todo")
    parser.add_argument("groundTruthPath", type=str, help="Path to music or folder containing music to transcribe")
    parser.add_argument("estimationsPath", type=str, help="Path to output folder")
    parser.add_argument(
        "-w",
        "--distance",
        type=float,
        default=0.05,
        help="distance allowed for hit rate. 0.03 is MIREX; 0.05 (50ms window) is Cartwright and Bello and multiple other; ",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default=None, help="specifies the dataset used for the evaluation to setup the good mapping"
    )
    args = parser.parse_args()
    classes = config.LABELS_5
    if args.dataset == "RBMA":
        mappingDictionaries = [config.RBMA_MIDI_8, config.MIDI_REDUCED_5]
        sep = "\t"
    elif args.dataset == "MDB":
        mappingDictionaries = [config.MDBS_MIDI, config.MIDI_REDUCED_5]
        sep = "\t"
    elif args.dataset == "ENST":
        mappingDictionaries = [config.ENST_MIDI, config.MIDI_REDUCED_5]
        sep = " "
    else:
        mappingDictionaries = [config.RBMA_MIDI_8, config.MIDI_REDUCED_5]
        sep = "\t"

    # Get the paths
    groundTruthsPaths = config.getFilesInFolder(args.groundTruthPath)
    estimationsPaths = config.getFilesInFolder(args.estimationsPath, allowedExtension=[".txt"])
    if args.dataset != "MDB":  # MDB doesn't have the same file name. but the order checks out
        groundTruthsPaths, estimationsPaths = config.getIntersectionOfPaths(groundTruthsPaths, estimationsPaths)

    # Decode
    tr = TextReader()
    groundTruths = [tr.getOnsets(grounTruth, mappingDictionaries=mappingDictionaries, sep=sep) for grounTruth in groundTruthsPaths]
    estimations = [tr.getOnsets(estimation) for estimation in estimationsPaths]

    result = eval.runEvaluation(groundTruths, estimations, paths=groundTruthsPaths, classes=classes, distanceThreshold=args.distance)
    print(args.dataset)
    print(result)


def plot(dict, title, ylim=True, legend=True, sort=False, ylabel="F-measure", text=True):
    """
    TODO
    """
    import pandas as pd

    df = pd.DataFrame(dict)
    if sort:
        df = df.sort_values("", ascending=False)

    cm = 1 / 2.54  # centimeters in inches
    fullpageWidth = 17.2 * cm
    oneColumn = fullpageWidth / 2
    height = fullpageWidth / 5
    width = fullpageWidth
    ax = df.plot.bar(edgecolor="black", legend=legend, figsize=(width, width / 5))

    # for i, patch in enumerate(ax.patches):
    #     patch.set_alpha(0.25 if i < 6 else 1)

    # Set correct size for the font
    # plt.rc("font", family="serif")
    # plt.rc("ytick", labelsize="x-small")

    if legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)

    # plt.xticks(ind + width, groups)
    plt.grid(axis="y", linestyle="--")
    if ylim:
        plt.ylim(0, 1)
    plt.ylabel(ylabel)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xticks(rotation=0)
    plt.title(title)

    if text:
        for p in ax.patches:
            sizeLimit = 0.35
            ax.text(
                p.get_x() + p.get_width() / 2,
                p.get_height() + 0.03 if p.get_height() < sizeLimit else p.get_height() - 0.03,
                "%.2f" % round(p.get_height(), 2),
                rotation="vertical",
                horizontalalignment="center",
                verticalalignment="bottom" if p.get_height() < sizeLimit else "top",
                color="black" if p.get_height() < sizeLimit else "w",
            )

    plt.savefig(title + ".pdf", dpi=600, bbox_inches="tight")
    # plt.savefig(title + ".png", dpi=600, bbox_inches="tight")
    # plt.show()


def plotResults():
    # Get results we computed
    with open("evaluation/crnn-all.json") as f:
        crnnAllJSON = json.load(f)
    crnnAll = {key: np.mean([values[key] for values in crnnAllJSON.values()]) for key in crnnAllJSON["crnn-all_Fold0"].keys()}

    with open("evaluation/crnn-ptTMIDT.json") as f:
        crnnPtJSON = json.load(f)
    crnnPt = {key: np.mean([values[key] for values in crnnPtJSON.values()]) for key in crnnPtJSON["crnn-ptTMIDT_Fold0"].keys()}

    with open("evaluation/crnn-ptTMIDT_log71.json") as f:
        crnnPtJSON = json.load(f)
    crnnPt71 = {key: np.mean([values[key] for values in crnnPtJSON.values()]) for key in crnnPtJSON["crnn-ptTMIDT_Fold0"].keys()}

    with open("evaluation/crnn-ADTOF.json") as f:
        crnnADTOFJSON = json.load(f)
    crnnADTOF = {key: np.mean([values[key] for values in crnnADTOFJSON.values()]) for key in crnnADTOFJSON["crnn-CC_Fold0"].keys()}

    with open("evaluation/crnn-ADTOF_log71.json") as f:
        crnnADTOFJSON = json.load(f)
    crnnADTOF71 = {key: np.mean([values[key] for values in crnnADTOFJSON.values()]) for key in crnnADTOFJSON["crnn-CC_Fold0"].keys()}

    # Get VOGL's results for comparison
    # From the website: http://ifs.tuwien.ac.at/~vogl/dafx2018/
    # Values from the website
    VOGL_ALLMIDI_RBMA = {
        "sum F all": 0.52,
        "sum F 35": 0.77,
        "sum F 38": 0.49,
        "sum F 47": 0.25,
        "sum F 42": 0.58,
        "sum F 49": np.mean([0.6, 0.03, 0.05]),
    }
    VOGL_ALLMIDI_MDB = {
        "sum F all": 0.57,
        "sum F 35": 0.64,
        "sum F 38": 0.55,
        "sum F 47": 0.25,
        "sum F 42": 0.70,
        "sum F 49": np.mean([0.11, 0.23, 0.00]),
    }
    VOGL_ALLMIDI_ENSTWET = {
        "sum F all": 0.62,
        "sum F 35": 0.79,
        "sum F 38": 0.53,
        "sum F 47": 0.16,
        "sum F 42": 0.75,
        "sum F 49": np.mean([0.07, 0.21, 0.02]),
    }
    VOGL_ALL_RBMA = {
        "sum F all": 0.52,
        "sum F 35": 0.79,
        "sum F 38": 0.51,
        "sum F 47": 0.24,
        "sum F 42": 0.59,
        "sum F 49": np.mean([0.06, 0.05, 0.05]),
    }
    VOGL_ALL_MDB = {
        "sum F all": 0.64,
        "sum F 35": 0.68,
        "sum F 38": 0.60,
        "sum F 47": 0.20,
        "sum F 42": 0.76,
        "sum F 49": np.mean([0.16, 0.46, 0.10]),
    }
    VOGL_ALL_ENSTWET = {
        "sum F all": 0.64,
        "sum F 35": 0.79,
        "sum F 38": 0.58,
        "sum F 47": 0.23,
        "sum F 42": 0.77,
        "sum F 49": np.mean([0.17, 0.33, 0.13]),
    }

    # Values from the pre-trained model
    VOGL_ENSEMBLE_CC0 = {
        "mean F all": 0.508802504159026,
        "mean P all": 0.588064022893335,
        "mean R all": 0.6081516499677526,
        "sum F all": 0.6241227123538473,
        "sum P all": 0.6557991131826109,
        "sum R all": 0.5953653809290537,
        "mean F 35": 0.7871626597721907,
        "mean P 35": 0.8286981823311089,
        "mean R 35": 0.8015920494560249,
        "sum F 35": 0.8157544501746798,
        "sum P 35": 0.8341650127587185,
        "sum R 35": 0.7981390049373338,
        "mean F 38": 0.7530013627048892,
        "mean P 38": 0.7834727246098705,
        "mean R 38": 0.767979481529634,
        "sum F 38": 0.7353791456429226,
        "sum P 38": 0.7713198734367938,
        "sum R 38": 0.7026387125553306,
        "mean F 47": 0.1968128055707403,
        "mean P 47": 0.18212634461155575,
        "mean R 47": 0.6191549361024239,
        "sum F 47": 0.18646813763308132,
        "sum P 47": 0.11808676959155755,
        "sum R 47": 0.44299853372434017,
        "mean F 42": 0.5186306456026357,
        "mean P 42": 0.6527325950325911,
        "mean R 42": 0.5177904233962511,
        "sum F 42": 0.5937763553402677,
        "sum P 42": 0.7149962242533136,
        "sum R 42": 0.5077011260470018,
        "mean F 49": 0.28840504714467463,
        "mean P 49": 0.49329026788154823,
        "mean R 49": 0.3342413593544291,
        "sum F 49": 0.3633130307206687,
        "sum P 49": 0.5687639198218263,
        "sum R 49": 0.26690182245737804,
    }
    VOGL_ENSEMBLE_CCLog70 = {
        "mean F all": 0.4835573579227426,
        "mean P all": 0.5772585073439122,
        "mean R all": 0.612169729098965,
        "sum F all": 0.6353351898704862,
        "sum P all": 0.6628751955637976,
        "sum R all": 0.6099922734423204,
        "mean F 35": 0.7742571981005972,
        "mean P 35": 0.8405772247597515,
        "mean R 35": 0.7880573943045629,
        "sum F 35": 0.8180421446915407,
        "sum P 35": 0.8444938408817264,
        "sum R 35": 0.793197190143755,
        "mean F 38": 0.7287393604411647,
        "mean P 38": 0.7792499000278864,
        "mean R 38": 0.7518049896084068,
        "sum F 38": 0.7411163174910629,
        "sum P 38": 0.7812274801825606,
        "sum R 38": 0.704922918529356,
        "mean F 47": 0.17658785560610887,
        "mean P 47": 0.15248053142326345,
        "mean R 47": 0.6681185238365472,
        "sum F 47": 0.27186581534407617,
        "sum P 47": 0.1805546207175069,
        "sum R 47": 0.5500307566126718,
        "mean F 42": 0.4968396292680449,
        "mean P 42": 0.6533048261120569,
        "mean R 42": 0.5423565402959439,
        "sum F 42": 0.6336857691891757,
        "sum P 42": 0.7321377083213194,
        "sum R 42": 0.5585733560157212,
        "mean F 49": 0.24136274619779724,
        "mean P 49": 0.46068005439660303,
        "mean R 49": 0.31051119744936495,
        "sum F 49": 0.2875876444949782,
        "sum P 49": 0.5184476632959825,
        "sum R 49": 0.19898253527036242,
    }

    # Values from MIREX 2018 competition
    VOGL_MIREX_MDB = {
        "sum F all": 0.60,
        "sum F 35": 0.72,
        "sum F 38": 0.63,
        "sum F 47": 0.60,
        "sum F 42": 0.60,
        "sum F 49": np.mean([0.27, 0.66, 0.91]),
    }
    VOGL_MIREX_RBMA = {
        "sum F all": 0.58,
        "sum F 35": 0.85,
        "sum F 38": 0.27,
        "sum F 47": 0.13,
        "sum F 42": 0.48,
        "sum F 49": np.mean([0.64, 0.79, 0.82]),
    }

    # Approximative value from the plot, just to get an idea of the performance
    # No values are reported in text
    VOGL_PTMIDI_RBMA = {
        "sum F all": 0.56,
        "sum F 35": 0.80,
        "sum F 38": 0.55,
        "sum F 47": 0.28,
        "sum F 42": 0.61,
        "sum F 49": np.mean([0.21, 0.07, 0.23]),
    }
    VOGL_PTMIDI_MDB = {
        "sum F all": 0.68,
        "sum F 35": 0.7,
        "sum F 38": 0.61,
        "sum F 47": 0.33,
        "sum F 42": 0.79,
        "sum F 49": np.mean([0.39, 0.55, 0.18]),
    }
    VOGL_PTMIDI_ENSTWET = {
        "sum F all": 0.68,
        "sum F 35": 0.79,
        "sum F 38": 0.58,
        "sum F 47": 0.29,
        "sum F 42": 0.80,
        "sum F 49": np.mean([0.24, 0.5, 0.16]),
    }

    def map(dict, keyPrefixes=[""], score="F"):
        mapping = {
            "sum " + score + " all": "SUM",
            "sum " + score + " 35": "BD",
            "sum " + score + " 38": "SD",
            "sum " + score + " 47": "TT",
            "sum " + score + " 42": "HH",
            "sum " + score + " 49": "CY + RD",
        }
        return {v: np.mean([dict[pre + k] for pre in keyPrefixes]) for k, v in mapping.items()}

    plot(
        {
            # "Train on ENST, MDB, and RBMA": map(crnnAll, keyPrefixes=["adtof_"]),
            "Train on TMIDT and refinement on ENST, MDB, and RBMA": map(crnnPt, keyPrefixes=["adtof_"]),
            "Train on TMIDT and refinement on ENST, MDB, and RBMA71": map(crnnPt71, keyPrefixes=["adtof_"]),
            "Train on ADTOF": map(crnnADTOF, keyPrefixes=["adtof_"]),
            "Train on ADTOF71": map(crnnADTOF71, keyPrefixes=["adtof_"]),
            # "Ensemble of models trained on TMIDT, RBMA, MDB, and ENST": map(VOGL_ENSEMBLE_CCLog70),
        },
        "Test on ADTOF",
        legend=False,
    )
    plot(
        {
            # "Train on ENST, MDB, and RBMA": map(crnnAll, keyPrefixes=["rbma_"]),
            "Train on TMIDT and refinement on ENST, MDB, and RBMA": map(crnnPt, keyPrefixes=["rbma_"]),
            "Train on TMIDT and refinement on ENST, MDB, and RBMA71": map(crnnPt71, keyPrefixes=["rbma_"]),
            "Train on ADTOF": map(crnnADTOF, keyPrefixes=["rbma_"]),
            "Train on ADTOF71": map(crnnADTOF71, keyPrefixes=["rbma_"]),
            # "Train on all Vogl": map(VOGL_ALL_RBMA),
            # "Train on pt MIDI Vogl": map(VOGL_PTMIDI_RBMA),
            # "Train on RBMA, ENST, MDB, and TMIDT": map(VOGL_ALLMIDI_RBMA),
        },
        "Test on RBMA",
        legend=True,
    )
    plot(
        {
            # "Train on RBMA, MDB, and ENST": map(crnnAll, keyPrefixes=["mdb_full_mix_"]),
            "Train on TMIDT and refinement on RBMA, MDB, and ENST": map(crnnPt, keyPrefixes=["mdb_full_mix_"]),
            "Train on TMIDT and refinement on RBMA, MDB, and ENST71": map(crnnPt71, keyPrefixes=["mdb_full_mix_"]),
            "Train on ADTOF": map(crnnADTOF, keyPrefixes=["mdb_full_mix_"]),
            "Train on ADTOF71": map(crnnADTOF71, keyPrefixes=["mdb_full_mix_"]),
            # "Train on all Vogl": map(VOGL_ALL_MDB),
            # "Train on pt MIDI Vogl": map(VOGL_PTMIDI_MDB),
            # "Train on RBMA, ENST, MDB, and TMIDT": map(VOGL_ALLMIDI_MDB),
        },
        "Test on MDB",
        legend=False,
    )
    plot(
        {
            # "Train on RBMA, MDB, and ENST": map(crnnAll, keyPrefixes=["enst_sum_"]),
            "Train on TMIDT and refinement on RBMA, MDB, and ENST": map(crnnPt, keyPrefixes=["enst_sum_"]),
            "Train on TMIDT and refinement on RBMA, MDB, and ENST71": map(crnnPt71, keyPrefixes=["enst_sum_"]),
            "Train on ADTOF": map(crnnADTOF, keyPrefixes=["enst_sum_"]),
            "Train on ADTOF71": map(crnnADTOF71, keyPrefixes=["enst_sum_"]),
        },
        "Test on ENST",
        legend=False,
    )
    # plot(
    #     {
    #         "Train on RBMA, MDB, and ENST": map(crnnAll, keyPrefixes=["adtof_", "rbma_", "mdb_full_mix_", "enst_sum_"]),
    #         "Train on TMIDT and refinement on RBMA, MDB, and ENST": map(
    #             crnnPt, keyPrefixes=["adtof_", "rbma_", "mdb_full_mix_", "enst_sum_"]
    #         ),
    #         "Train on ADTOF": map(crnnADTOF, keyPrefixes=["adtof_", "rbma_", "mdb_full_mix_", "enst_sum_"]),
    #     },
    #     "Overall",
    #     legend=False,
    # )
    plt.show()


def plotInstrumentClasses():
    """
    Count the number of onsets for each instrument in the datet
    """
    parser = argparse.ArgumentParser(description="todo")
    parser.add_argument("groundTruthPath", type=str, help="Path to music or folder containing music to transcribe")
    parser.add_argument("estimationsPath", type=str, help="Path to output folder")
    parser.add_argument(
        "-w",
        "--distance",
        type=float,
        default=0.03,
        help="distance allowed for hit rate. 0.03 is MIREX; 0.025 (50ms window) is Cartwright and Bello; ",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default=None, help="specifies the dataset used for the evaluation to setup the good mapping"
    )
    args = parser.parse_args()
    classes = config.LABELS_5
    if args.dataset == "RBMA":
        mappingDictionaries = [config.RBMA_MIDI_8]
        sep = "\t"
    elif args.dataset == "MDB":
        mappingDictionaries = []
        sep = "\t"
    elif args.dataset == "ENST":
        mappingDictionaries = [config.ENST_MIDI, config.MIDI_REDUCED_5]
        sep = " "
    else:
        mappingDictionaries = [config.RBMA_MIDI_8, config.MIDI_REDUCED_5]
        sep = "\t"

    # Get the paths
    groundTruthsPaths = config.getFilesInFolder(args.groundTruthPath)
    estimationsPaths = [path for path in config.getFilesInFolder(args.estimationsPath) if os.path.splitext(path)[1] == ".txt"]
    if args.dataset != "MDB":  # MDB doesn't have the same file name. but the order checks out
        groundTruthsPaths, estimationsPaths = config.getIntersectionOfPaths(groundTruthsPaths, estimationsPaths)

    # Decode
    tr = TextReader()
    groundTruths = [tr.getOnsets(grounTruth, mappingDictionaries=mappingDictionaries, sep=sep) for grounTruth in groundTruthsPaths]
    estimations = [tr.getOnsets(estimation) for estimation in estimationsPaths]

    sum = {}
    for est in groundTruths:
        for inst, oc in est.items():
            if inst not in sum:
                sum[inst] = 0
            sum[inst] += len(oc)

    # def map(dict, add=""):
    #     mapping = {42: "HH", 35: "BD", 38: "SD", 47: "TT", 75: "CL", 53: "BE", 51: "RD", 49: "CY"}
    #     return {v: dict[k] for k, v in mapping.items()}

    plot({"": sum}, "Instrument classes", ylim=False, legend=False, sort=True, ylabel="Count")
    plt.show()


if __name__ == "__main__":
    # main()
    plotResults()
