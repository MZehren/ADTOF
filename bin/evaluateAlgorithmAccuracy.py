#!/usr/bin/env python
# encoding: utf-8
"""
TODO
"""
from adtof.io.textReader import TextReader
import argparse
import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import mir_eval

from adtof import config

from adtof.converters.converter import Converter
from adtof.io.mir import MIR
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
        default=0.03,
        help="distance allowed for hit rate. 0.03 is MIREX; 0.025 (50ms window) is Cartwright and Bello; ",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default=None, help="specifies the dataset used for the evaluation to setup the good mapping"
    )
    args = parser.parse_args()
    classes = config.LABELS_3
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
    estimationsPaths = [path for path in config.getFilesInFolder(args.estimationsPath) if os.path.splitext(path)[1] == ".txt"]
    if args.dataset != "MDB":  # MDB doesn't have the same file name. but the order checks out
        groundTruthsPaths, estimationsPaths = config.getIntersectionOfPaths(groundTruthsPaths, estimationsPaths)

    # Decode
    tr = TextReader()
    groundTruths = [tr.getOnsets(grounTruth, mappingDictionaries=mappingDictionaries, sep=sep) for grounTruth in groundTruthsPaths]
    estimations = [tr.getOnsets(estimation) for estimation in estimationsPaths]

    result = eval.runEvaluation(groundTruths, estimations, paths=groundTruthsPaths, classes=classes, distance=args.distance)
    print(args.dataset)
    print(result)
    plot(result, prefix="mean", groups=["all"] + [str(e) for e in classes])
    plot(result, prefix="sum", groups=["all"] + [str(e) for e in classes])
    plt.show()


def plot(result, prefix="mean", bars=["F", "P", "R"], groups=["all", "35", "38", "47", "42", "49"]):
    """
    test

    Parameters
    ----------
    result
    prefix
    bars
    groups

    Returns
    -------

    """
    fig, ax = plt.subplots()
    ind = np.arange(len(groups))  # the x locations for the groups
    width = 1 / (len(groups) + 1)  # the width of the bars

    for i, bar in enumerate(bars):
        X = ind + (i * width)
        Y = [result[" ".join([prefix, bar, group])] for group in groups]
        ax.bar(
            X, Y, width=width, edgecolor="black", label=" ".join([prefix, bar]),
        )
        for x, y in zip(X, Y):
            plt.annotate(np.format_float_positional(y, precision=2), xy=(x - width / 2, y + 0.01))

    plt.xticks(ind + width, groups)
    plt.grid(axis="y", linestyle="--")
    plt.legend()
    plt.ylim(0, 1)


def newPlot(dict, title, ylim=True, legend=True, sort=False, ylabel="F-measure"):
    """
    dictionary:
        {
            RBMA:{  # plot
                MZ:{  # group
                    All: value  # bar 
                    KD: value
                    ...
                }
                Vogl:{
                    All: value
                    KD: value
                    ...
                }
            }
            ENST:{ ... }
        }
    """
    import pandas as pd

    df = pd.DataFrame(dict)
    if sort:
        df = df.sort_values("", ascending=False)
    ax = df.plot.bar(edgecolor="black", legend=legend, figsize=(10, 2))

    # for i, patch in enumerate(ax.patches):
    #     patch.set_alpha(0.25 if i < 6 else 1)

    # plt.xticks(ind + width, groups)
    plt.grid(axis="y", linestyle="--")
    if ylim:
        plt.ylim(0, 1)
    plt.ylabel(ylabel)
    plt.xticks(rotation=0)
    plt.title(title)

    plt.savefig(title + ".pdf", dpi=600)


def plotResults():

    # MZ
    MZ_ADTOF = {
        "sum F all": 0.67680,
        "sum F 35": 0.82921,
        "sum F 38": 0.75134,
        "sum F 47": 0.20568,
        "sum F 42": 0.63277,
        "sum F 49": 0.43859,
    }
    MZ_RBMA = {
        "mean F all": 0.41724200722886723,
        "mean P all": 0.4789603867137693,
        "mean R all": 0.6553960804728254,
        "sum F all": 0.6040870576507726,
        "sum P all": 0.5484824714897725,
        "sum R all": 0.6722378346398462,
        "mean F 35": 0.797350918151105,
        "mean P 35": 0.7276143192983746,
        "mean R 35": 0.9440304358114014,
        "sum F 35": 0.7983332381985273,
        "sum P 35": 0.6888976455521624,
        "sum R 35": 0.9491042345276873,
        "mean F 38": 0.349059579905169,
        "mean P 38": 0.34417343377375537,
        "mean R 38": 0.6924806314557463,
        "sum F 38": 0.56391970417327,
        "sum P 38": 0.49409858828974773,
        "sum R 38": 0.6567210089203322,
        "mean F 47": 0.23895618894714168,
        "mean P 47": 0.42822780173901076,
        "mean R 47": 0.38238624603515414,
        "sum F 47": 0.24012393493415957,
        "sum P 47": 0.33101975440469833,
        "sum R 47": 0.18839258584017016,
        "mean F 42": 0.538877934032345,
        "mean P 42": 0.5100097880426789,
        "mean R 42": 0.7420741543359596,
        "sum F 42": 0.5869136498098324,
        "sum P 42": 0.5158155369854535,
        "sum R 42": 0.6807450371701659,
        "mean F 49": 0.1619654151085756,
        "mean P 49": 0.38477659071502673,
        "mean R 49": 0.516008934725866,
        "sum F 49": 0.1286549707602339,
        "sum P 49": 0.15325077399380804,
        "sum R 49": 0.11086226203807391,
    }
    MZ_RBMA_3 = {
        "mean F all": 0.4400764804719079,
        "mean P all": 0.43370172930439743,
        "mean R all": 0.8629157613633572,
        "sum F all": 0.6319200083614953,
        "sum P all": 0.548954329469447,
        "sum R all": 0.7444284834804022,
        "mean F 35": 0.7979872053672956,
        "mean P 35": 0.728334676526333,
        "mean R 35": 0.9431471908581083,
        "sum F 35": 0.7997026702498714,
        "sum P 35": 0.6910079051383399,
        "sum R 35": 0.9489754376441851,
        "mean F 38": 0.458201661703798,
        "mean P 38": 0.5221524684678828,
        "mean R 38": 0.6376646507315469,
        "sum F 38": 0.6365693595836162,
        "sum P 38": 0.6504046242774566,
        "sum R 38": 0.6233104365167295,
        "mean F 47": 0.18518518518518517,
        "mean P 47": 0.18518518518518517,
        "mean R 47": 1.0,
        "sum F 47": 0.0,
        "sum P 47": 0.0,
        "sum R 47": 1.0,
        "mean F 42": 0.5367861278810384,
        "mean P 42": 0.5106140941203643,
        "mean R 42": 0.733766965227131,
        "sum F 42": 0.5818435754189943,
        "sum P 42": 0.5156897938973819,
        "sum R 42": 0.6674677561483617,
        "mean F 49": 0.2222222222222222,
        "mean P 49": 0.2222222222222222,
        "mean R 49": 1.0,
        "sum F 49": 0.0,
        "sum P 49": 0.0,
        "sum R 49": 1.0,
    }
    MZ_MDB = {
        "mean F all": 0.6205394267178704,
        "mean P all": 0.7551034093457771,
        "mean R all": 0.7085228534718282,
        "sum F all": 0.680201604316036,
        "sum P all": 0.7773811455460004,
        "sum R all": 0.6046188793538617,
        "mean F 35": 0.8036480487828286,
        "mean P 35": 0.8668979250810638,
        "mean R 35": 0.7708456143473722,
        "sum F 35": 0.802924791086351,
        "sum P 35": 0.8649662415603901,
        "sum R 35": 0.7491877842755036,
        "mean F 38": 0.6562078114690553,
        "mean P 38": 0.9104705355705572,
        "mean R 38": 0.6171317584690054,
        "sum F 38": 0.5826155050900548,
        "sum P 38": 0.9481733220050977,
        "sum R 38": 0.42049736247174074,
        "mean F 47": 0.6405493861008907,
        "mean P 47": 0.7555946291560103,
        "mean R 47": 0.7943640637875042,
        "sum F 47": 0.38961038961038963,
        "sum P 47": 0.46875,
        "sum R 47": 0.3333333333333333,
        "mean F 42": 0.6096038245437088,
        "mean P 42": 0.6441685373593644,
        "mean R 42": 0.8147168240426901,
        "sum F 42": 0.7802660380535444,
        "sum P 42": 0.7021212121212121,
        "sum R 42": 0.8779840848806366,
        "mean F 49": 0.3926880626928691,
        "mean P 49": 0.5983854195618902,
        "mean R 49": 0.5455560067125687,
        "sum F 49": 0.2711076684740511,
        "sum P 49": 0.6055363321799307,
        "sum R 49": 0.17465069860279442,
    }
    MZ_ENST_SUM = {
        "mean F all": 0.41937460483851,
        "mean P all": 0.7031660852442764,
        "mean R all": 0.3906405077619682,
        "sum F all": 0.5605160414191139,
        "sum P all": 0.7608294930875577,
        "sum R all": 0.44369793066380003,
        "mean F 35": 0.6722256200093613,
        "mean P 35": 0.9095032151911238,
        "mean R 35": 0.5961506507083166,
        "sum F 35": 0.7230587041109797,
        "sum P 35": 0.9348206474190727,
        "sum R 35": 0.5895172413793104,
        "mean F 38": 0.35372965938726814,
        "mean P 38": 0.8749765929617295,
        "mean R 38": 0.28347707608749456,
        "sum F 38": 0.32900432900432897,
        "sum P 38": 0.9510807736063709,
        "sum R 38": 0.19890554365929097,
        "mean F 47": 0.21856691175187223,
        "mean P 47": 0.6531099257884971,
        "mean R 47": 0.2257805091733663,
        "sum F 47": 0.1527272727272727,
        "sum P 47": 0.3783783783783784,
        "sum R 47": 0.09567198177676538,
        "mean F 42": 0.6452080092899054,
        "mean P 42": 0.6955183321066826,
        "mean R 42": 0.6376495415384413,
        "sum F 42": 0.6825396825396826,
        "sum P 42": 0.7071473750790639,
        "sum R 42": 0.6595870206489676,
        "mean F 49": 0.20714282375414328,
        "mean P 49": 0.3827223601733487,
        "mean R 49": 0.21014476130222173,
        "sum F 49": 0.21431828545371634,
        "sum P 49": 0.3555219364599092,
        "sum R 49": 0.15339425587467362,
    }
    MZ_ENST_WET = {
        "mean F all": 0.5020027804384568,
        "mean P all": 0.7909520556324104,
        "mean R all": 0.4987024232900214,
        "sum F all": 0.6837135196051762,
        "sum P all": 0.8108581436077058,
        "sum R all": 0.5910373555495835,
        "mean F 35": 0.7638537439584842,
        "mean P 35": 0.9754528575404453,
        "mean R 35": 0.7196237112388244,
        "sum F 35": 0.8169507726620997,
        "sum P 35": 0.9668174962292609,
        "sum R 35": 0.7073103448275863,
        "mean F 38": 0.5358861547887056,
        "mean P 38": 0.8140092099368544,
        "mean R 38": 0.4786149097057931,
        "sum F 38": 0.5403252572187188,
        "sum P 38": 0.8930334613274822,
        "sum R 38": 0.3873423744944088,
        "mean F 47": 0.19890546419160124,
        "mean P 47": 0.798469387755102,
        "mean R 47": 0.23235776128633276,
        "sum F 47": 0.20233463035019456,
        "sum P 47": 0.6933333333333334,
        "sum R 47": 0.11845102505694761,
        "mean F 42": 0.7480107746301297,
        "mean P 42": 0.7094225081310045,
        "mean R 42": 0.8407725213069168,
        "sum F 42": 0.7849083852333243,
        "sum P 42": 0.725392058725392,
        "sum R 42": 0.8550639134709931,
        "mean F 49": 0.26335776462336347,
        "mean P 49": 0.6574063147986468,
        "mean R 49": 0.22214321291224,
        "sum F 49": 0.2231899836690256,
        "sum P 49": 0.6721311475409836,
        "sum R 49": 0.13381201044386423,
    }

    # VOGL
    # From the website: http://ifs.tuwien.ac.at/~vogl/dafx2018/
    # correlation found is:

    # all
    # MIDI
    # MIDI bal.
    # all+MIDI
    # pt MIDI
    # pt MIDI bal.

    # enst
    # enst(all)
    # enst(all+md1%)
    # (all+md_bm1%)
    # enst(mdb)
    # enst(midi)
    # enst(midi_10p)
    # enst(midi_1p)
    # (midi_bal_m)
    # enst(rbma)

    VOGL_ADTOF_ALLMIDI = {
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
    VOGL_RBMA_MIREX = {
        "sum F all": 0.58,
        "sum F 35": 0.85,
        "sum F 38": 0.27,
        "sum F 47": 0.13,
        "sum F 42": 0.48,
        "sum F 49": np.mean([0.64, 0.79, 0.82]),
    }
    VOGL_RBMA_ALLMIDI = {
        "sum F all": 0.52,
        "sum F 35": 0.77,
        "sum F 38": 0.49,
        "sum F 47": 0.25,
        "sum F 42": 0.58,
        "sum F 49": np.mean([0.6, 0.03, 0.05]),
    }
    VOGL_MDB_ALLMIDI = {
        "sum F all": 0.57,
        "sum F 35": 0.64,
        "sum F 38": 0.55,
        "sum F 47": 0.25,
        "sum F 42": 0.70,
        "sum F 49": np.mean([0.11, 0.23, 0.00]),
    }
    VOGL_MDB_MIREX = {
        "sum F all": 0.60,
        "sum F 35": 0.72,
        "sum F 38": 0.63,
        "sum F 47": 0.60,
        "sum F 42": 0.60,
        "sum F 49": np.mean([0.27, 0.66, 0.91]),
    }
    VOGL_ENST_ALLMIDI = {
        "sum F all": 0.62,
        "sum F 35": 0.79,
        "sum F 38": 0.53,
        "sum F 47": 0.16,
        "sum F 42": 0.75,
        "sum F 49": np.mean([0.07, 0.21, 0.02]),
    }

    # Approximative value from the plot TODO: double check, there was an error
    VOGL_RBMA_PTMIDI = {
        "sum F all": 0.56,
        "sum F 35": 0.80,
        "sum F 38": 0.55,
        "sum F 47": 0.28,
        "sum F 42": 0.61,
        "sum F 49": np.mean([0.21, 0.07, 0.23]),
    }
    VOGL_MDB_PTMIDI = {
        "sum F all": 0.68,
        "sum F 35": 0.7,
        "sum F 38": 0.61,
        "sum F 47": 0.33,
        "sum F 42": 0.58,
        "sum F 49": np.mean([0.29, 0.55, 0.18]),
    }
    VOGL_ENST_PTMIDI = {
        "sum F all": 0.68,
        "sum F 35": 0.79,
        "sum F 38": 0.58,
        "sum F 47": 0.29,
        "sum F 42": 0.80,
        "sum F 49": np.mean([0.21, 0.5, 0.13]),
    }

    def map(dict, add=""):
        mapping = {
            "sum F all": "SUM" + add,
            "sum F 35": "BD",
            "sum F 38": "SD",
            "sum F 47": "TT",
            "sum F 42": "HH",
            "sum F 49": "CY + RD" + add,
        }
        return {v: dict[k] for k, v in mapping.items()}

    # results = {
    #     "ADTOF": {"ADTOF": map(MZ_ADTOF), "All+MIDI": map(VOGL_ADTOF)},
    #     "RBMA": {"ADTOF": map(MZ_RBMA), "all+MIDI": map(VOGL_RBMA)},
    #     "MDB": {"ADTOF": map(MZ_MDB), "all+MIDI": map(VOGL_MDB)},
    #     "ENST": {"ADTOF": map(MZ_ENST_WET), "all+MIDI": map(VOGL_ENST)},
    # }
    # newPlot(results, "test")

    # newPlot(
    #     {
    #         "Test on ADTOF": map(VOGL_ADTOF_ALLMIDI, add="*"),
    #         "Test on RBMA": map(VOGL_RBMA_ALLMIDI, add="*"),
    #         "Test on MDB": map(VOGL_MDB_ALLMIDI, add="*"),
    #         "Test on ENST": map(VOGL_ENST_ALLMIDI, add="*"),
    #     },
    #     "",
    #     legend=True,
    # )

    # Plot only one bar
    # newPlot({"Train on RBMA, ENST, MDB, and TMIDT": map(VOGL_ADTOF_ALLMIDI)}, "Test on ADTOF", legend=True)
    # newPlot(
    #     {"Train on RBMA, ENST, MDB, and TMIDT": map(VOGL_RBMA_ALLMIDI, add="*")}, "Test on RBMA", legend=False,
    # )  # "pt MIDI": map(VOGL_RBMA_PTMIDI)
    # newPlot(
    #     {"Train on RBMA, ENST, MDB, and TMIDT": map(VOGL_MDB_ALLMIDI, add="*")}, "Test on MDB", legend=False,
    # )  # , "pt MIDI": map(VOGL_MDB_PTMIDI)
    # newPlot(
    #     {"Train on RBMA, ENST, MDB, and TMIDT": map(VOGL_ENST_ALLMIDI, add="*")}, "Test on ENST", legend=False,
    # )  # , "pt MIDI": map(VOGL_ENST_PTMIDI)
    # plt.show()

    newPlot({"Train on ADTOF": map(MZ_ADTOF), "Train on RBMA, ENST, MDB, and TMIDT": map(VOGL_ADTOF_ALLMIDI)}, "Test on ADTOF", legend=True)
    newPlot(
        {"Train on ADTOF": map(MZ_RBMA, add="*"), "Train on RBMA, ENST, MDB, and TMIDT": map(VOGL_RBMA_ALLMIDI, add="*")},
        "Test on RBMA",
        legend=False,
    )  # "pt MIDI": map(VOGL_RBMA_PTMIDI)
    newPlot(
        {"Train on ADTOF": map(MZ_MDB, add="*"), "Train on RBMA, ENST, MDB, and TMIDT": map(VOGL_MDB_ALLMIDI, add="*")},
        "Test on MDB",
        legend=False,
    )  # , "pt MIDI": map(VOGL_MDB_PTMIDI)
    newPlot(
        {"Train on ADTOF": map(MZ_ENST_WET, add="*"), "Train on RBMA, ENST, MDB, and TMIDT": map(VOGL_ENST_ALLMIDI, add="*")},
        "Test on ENST",
        legend=False,
    )  # , "pt MIDI": map(VOGL_ENST_PTMIDI)
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
    classes = config.LABELS_3
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

    newPlot({"": sum}, "Instrument classes", ylim=False, legend=False, sort=True, ylabel="Count")
    plt.show()


if __name__ == "__main__":
    # main()
    plotResults()
    # plotInstrumentClasses()
