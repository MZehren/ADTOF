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


def newPlot(dict, title, ylim=True, legend=True, sort=False, ylabel="F-measure", text=True):
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

    if text:
        for p in ax.patches:
            ax.text(p._x0 + 0.01, p._y1 + 0.03, "%.2f" % round(p._y1, 2), rotation="vertical")

    plt.savefig(title + ".pdf", dpi=600)


def plotResults():

    # MZ
    # Trained on CC0
    MZ_CC0_CC0 = {
        "sum F all": 0.67680,
        "sum F 35": 0.82921,
        "sum F 38": 0.75134,
        "sum F 47": 0.20568,
        "sum F 42": 0.63277,
        "sum F 49": 0.43859,
    }
    MZ_CC0_RBMA = {
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
    MZ_CC0_RBMA_3 = {
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
    MZ_CC0_MDB = {
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
    MZ_CC0_ENSTSUM = {
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
    MZ_CC0_ENSTWET = {
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

    # Trained on CCLog70
    MZ_CCLog70_CCLog70_Fold0 = {
        "sum F all": 0.70698,
        "sum F 35": 0.84282,
        "sum F 38": 0.76681,
        "sum F 47": 0.41445,
        "sum F 42": 0.67301,
        "sum F 49": 0.51137,
    }
    MZ_CCLog70_CCLog70_Fold1 = {
        "sum F all": 0.73212,
        "sum F 35": 0.86235,
        "sum F 38": 0.79119,
        "sum F 47": 0.42968,
        "sum F 42": 0.69679,
        "sum F 49": 0.54307,
    }
    MZ_CCLog70_RBMA = {
        "mean F all": 0.3922029919368118,
        "mean P all": 0.37681240460499443,
        "mean R all": 0.7219618160956532,
        "sum F all": 0.5793083697286462,
        "sum P all": 0.47271622537449465,
        "sum R all": 0.7479662771779323,
        "mean F 35": 0.7520736255652231,
        "mean P 35": 0.6560526878133559,
        "mean R 35": 0.9650936727884376,
        "sum F 35": 0.745805106826472,
        "sum P 35": 0.6053121299272542,
        "sum R 35": 0.9712269272529859,
        "mean F 38": 0.34901580093749573,
        "mean P 38": 0.3087176045032811,
        "mean R 38": 0.7711492469145923,
        "sum F 38": 0.5402015677491602,
        "sum P 38": 0.4247226624405705,
        "sum R 38": 0.7419255613657336,
        "mean F 47": 0.18704786376197882,
        "mean P 47": 0.22557732182276416,
        "mean R 47": 0.4939689036904128,
        "sum F 47": 0.271450133519099,
        "sum P 47": 0.2196542653137918,
        "sum R 47": 0.35521118201154667,
        "mean F 42": 0.5565325929719253,
        "mean P 42": 0.5020487787110031,
        "mean R 42": 0.8094110744647783,
        "sum F 42": 0.604788476511809,
        "sum P 42": 0.50158747242103,
        "sum R 42": 0.7614573972714648,
        "mean F 49": 0.11634507644743575,
        "mean P 49": 0.1916656301745679,
        "mean R 49": 0.5701861826200452,
        "sum F 49": 0.1492537313432836,
        "sum P 49": 0.12274368231046931,
        "sum R 49": 0.19036954087346025,
    }
    MZ_CCLog70_MDB = {
        "mean F all": 0.5282000585965535,
        "mean P all": 0.575047844244843,
        "mean R all": 0.7812763764892944,
        "sum F all": 0.6851131654446924,
        "sum P all": 0.6923076923076923,
        "sum R all": 0.6780666330136295,
        "mean F 35": 0.8021576494072292,
        "mean P 35": 0.7796572101883068,
        "mean R 35": 0.8656330587147185,
        "sum F 35": 0.7722536469187259,
        "sum P 35": 0.7126373626373627,
        "sum R 35": 0.8427550357374919,
        "mean F 38": 0.7081209968262244,
        "mean P 38": 0.8802185216926451,
        "mean R 38": 0.6801057480520032,
        "sum F 38": 0.6610251450676983,
        "sum P 38": 0.9224021592442645,
        "sum R 38": 0.5150715900527506,
        "mean F 47": 0.2422040812409473,
        "mean P 47": 0.26122079572289947,
        "mean R 47": 0.8634591149241433,
        "sum F 47": 0.18003913894324852,
        "sum P 47": 0.10926365795724466,
        "sum R 47": 0.5111111111111111,
        "mean F 42": 0.576721535405176,
        "mean P 42": 0.6287904469237549,
        "mean R 42": 0.7828469995851287,
        "sum F 42": 0.7812877610909719,
        "sum P 42": 0.7175015852885225,
        "sum R 42": 0.8575217885562714,
        "mean F 49": 0.3117960301031906,
        "mean P 49": 0.32535224669660884,
        "mean R 49": 0.7143369611704782,
        "sum F 49": 0.42417815482502647,
        "sum P 49": 0.45248868778280543,
        "sum R 49": 0.3992015968063872,
    }
    MZ_CCLog70_ENSTWET = {
        "mean F all": 0.5971456958906342,
        "mean P all": 0.6879683019091486,
        "mean R all": 0.6759343123162999,
        "sum F all": 0.7160638151560995,
        "sum P all": 0.7333098591549295,
        "sum R all": 0.6996103198065037,
        "mean F 35": 0.8433611720963299,
        "mean P 35": 0.9613196880412467,
        "mean R 35": 0.8029443177894431,
        "sum F 35": 0.8676337262012693,
        "sum P 35": 0.9592382225192115,
        "sum R 35": 0.792,
        "mean F 38": 0.5952916727620785,
        "mean P 38": 0.7865781020036148,
        "mean R 38": 0.5511236323001116,
        "sum F 38": 0.6169244767970883,
        "sum P 38": 0.8506900878293601,
        "sum R 38": 0.48394004282655245,
        "mean F 47": 0.37916883693241754,
        "mean P 47": 0.588932375318471,
        "mean R 47": 0.4268060709132137,
        "sum F 47": 0.3108695652173913,
        "sum P 47": 0.2972972972972973,
        "sum R 47": 0.32574031890660593,
        "mean F 42": 0.7745759735550063,
        "mean P 42": 0.733701160322037,
        "mean R 42": 0.8661951499591457,
        "sum F 42": 0.8026494873423464,
        "sum P 42": 0.7451145552560647,
        "sum R 42": 0.8698131760078662,
        "mean F 49": 0.3933308241073381,
        "mean P 49": 0.36931018386037306,
        "mean R 49": 0.7326023906195848,
        "sum F 49": 0.4792673619944035,
        "sum P 49": 0.39266360983743226,
        "sum R 49": 0.6148825065274152,
    }

    # Trained on RBLog70
    MZ_RBLog70_RBLog70_Fold0 = {
        "sum F all": 0.78584,
        "sum F 35": 0.89532,
        "sum F 38": 0.85273,
        "sum F 47": 0.41485,
        "sum F 42": 0.76056,
        "sum F 49": 0.67877,
    }
    MZ_RBLog70_RBMA = {
        "mean F all": 0.3648957854365652,
        "mean P all": 0.4555417535624794,
        "mean R all": 0.6292162196704163,
        "sum F all": 0.5331215080059263,
        "sum P all": 0.5153397522172758,
        "sum R all": 0.5521742345806834,
        "mean F 35": 0.7895104384606559,
        "mean P 35": 0.7224963937305193,
        "mean R 35": 0.9311508685364723,
        "sum F 35": 0.7862227324913891,
        "sum P 35": 0.6812574612017509,
        "sum R 35": 0.9294245385450597,
        "mean F 38": 0.3919219333778435,
        "mean P 38": 0.38977789379385785,
        "mean R 38": 0.7165180988885461,
        "sum F 38": 0.5647892871209137,
        "sum P 38": 0.49267063673843337,
        "sum R 38": 0.6616425715164564,
        "mean F 47": 0.15065488470875033,
        "mean P 47": 0.16268382627546055,
        "mean R 47": 0.5132478064503081,
        "sum F 47": 0.2871287128712871,
        "sum P 47": 0.21733021077283374,
        "sum R 47": 0.422971741112124,
        "mean F 42": 0.4067067805041898,
        "mean P 42": 0.598782708377947,
        "mean R 42": 0.4550734497430231,
        "sum F 42": 0.4513991538814414,
        "sum P 42": 0.6001626457034427,
        "sum R 42": 0.36173515235683357,
        "mean F 49": 0.08568489013138653,
        "mean P 49": 0.40396794563461236,
        "mean R 49": 0.5300908747337318,
        "sum F 49": 0.1366087477531456,
        "sum P 49": 0.14690721649484537,
        "sum R 49": 0.1276595744680851,
    }
    MZ_RBLog70_MDB = {
        "mean F all": 0.5874008427982267,
        "mean P all": 0.7186588648450278,
        "mean R all": 0.7151419236016247,
        "sum F all": 0.7026482440990212,
        "sum P all": 0.8174815807099799,
        "sum R all": 0.616102978293791,
        "mean F 35": 0.8225345932063831,
        "mean P 35": 0.8439446549307993,
        "mean R 35": 0.8219728592317641,
        "sum F 35": 0.8118032786885245,
        "sum P 35": 0.8193249503639973,
        "sum R 35": 0.8044184535412605,
        "mean F 38": 0.6940212486789892,
        "mean P 38": 0.913841739577458,
        "mean R 38": 0.650734029546513,
        "sum F 38": 0.62015503875969,
        "sum P 38": 0.9219330855018587,
        "sum R 38": 0.46721929163526754,
        "mean F 47": 0.3697863853550484,
        "mean P 47": 0.4083496897242213,
        "mean R 47": 0.8459478454752557,
        "sum F 47": 0.19954648526077098,
        "sum P 47": 0.12535612535612536,
        "sum R 47": 0.4888888888888889,
        "mean F 42": 0.5931015992067475,
        "mean P 42": 0.8236797291257602,
        "mean R 42": 0.6416788751534388,
        "sum F 42": 0.7998359310910583,
        "sum P 42": 0.8717031738936075,
        "sum R 42": 0.7389162561576355,
        "mean F 49": 0.45756038754396516,
        "mean P 49": 0.6034785108669007,
        "mean R 49": 0.6153760086011519,
        "sum F 49": 0.5359477124183006,
        "sum P 49": 0.7765151515151515,
        "sum R 49": 0.4091816367265469,
    }
    MZ_RBLog70_ENSTWET = {
        "mean F all": 0.5529391484430326,
        "mean P all": 0.7064142490462004,
        "mean R all": 0.5830517272597928,
        "sum F all": 0.6932890942844729,
        "sum P all": 0.8411770344100451,
        "sum R all": 0.5896264445041656,
        "mean F 35": 0.8033651254317642,
        "mean P 35": 0.9803418163532182,
        "mean R 35": 0.7566000524740717,
        "sum F 35": 0.8480499219968799,
        "sum P 35": 0.9759425493716337,
        "sum R 35": 0.7497931034482759,
        "mean F 38": 0.5850324459736427,
        "mean P 38": 0.7916161447338146,
        "mean R 38": 0.5367915315550712,
        "sum F 38": 0.5993545412632549,
        "sum P 38": 0.8463541666666666,
        "sum R 38": 0.46395431834404,
        "mean F 47": 0.2600140772110721,
        "mean P 47": 0.2630045742088334,
        "mean R 47": 0.5086219336219336,
        "sum F 47": 0.2270947533281128,
        "sum P 47": 0.1730310262529833,
        "sum R 47": 0.33029612756264237,
        "mean F 42": 0.6811349671089179,
        "mean P 42": 0.9240533245422214,
        "mean R 42": 0.6166212590765251,
        "sum F 42": 0.7549791570171376,
        "sum P 42": 0.9180512531681216,
        "sum R 42": 0.6411012782694199,
        "mean F 49": 0.4351491264897662,
        "mean P 49": 0.5730553853929151,
        "mean R 49": 0.49662385957136257,
        "sum F 49": 0.5653397667872939,
        "sum P 49": 0.7361256544502618,
        "sum R 49": 0.45887728459530025,
    }

    # Trained on YTLog70
    MZ_YTLog70_YTLog70_Fold0 = {
        "sum F all": 0.83251,
        "sum F 35": 0.96154,
        "sum F 38": 0.88414,
        "sum F 47": 0.64987,
        "sum F 42": 0.59957,
        "sum F 49": 0.68209,
    }
    MZ_YTLog70_RBMA = {
        "mean F all": 0.34614124200597507,
        "mean P all": 0.3851640471740806,
        "mean R all": 0.6185162517564566,
        "sum F all": 0.4822807431635027,
        "sum P all": 0.4432290052680508,
        "sum R all": 0.5288788640733619,
        "mean F 35": 0.7565053014420519,
        "mean P 35": 0.6950482237491655,
        "mean R 35": 0.8969942175289584,
        "sum F 35": 0.7667826384532025,
        "sum P 35": 0.6612887358583375,
        "sum R 35": 0.9123235613463626,
        "mean F 38": 0.32390969003972175,
        "mean P 38": 0.3446707845862405,
        "mean R 38": 0.6584639434081203,
        "sum F 38": 0.5060331825037708,
        "sum P 38": 0.42784272051009564,
        "sum R 38": 0.6191940941248847,
        "mean F 47": 0.15728503937350524,
        "mean P 47": 0.19135040645455945,
        "mean R 47": 0.41503107625974495,
        "sum F 47": 0.29615060410227595,
        "sum P 47": 0.27541154951659264,
        "sum R 47": 0.3202673959282893,
        "mean F 42": 0.3757249046559339,
        "mean P 42": 0.5509137572728342,
        "mean R 42": 0.44697395523504707,
        "sum F 42": 0.4271281280762655,
        "sum P 42": 0.5838526912181303,
        "sum R 42": 0.3367371946736378,
        "mean F 49": 0.11728127451866267,
        "mean P 49": 0.14383706380760314,
        "mean R 49": 0.6751180663504124,
        "sum F 49": 0.10586011342155009,
        "sum P 49": 0.06018731767234761,
        "sum R 49": 0.43896976483762595,
    }
    MZ_YTLog70_MDB = {
        "mean F all": 0.5299765731578892,
        "mean P all": 0.6364772204163129,
        "mean R all": 0.7197380687301489,
        "sum F all": 0.6843483515745651,
        "sum P all": 0.7312383412254269,
        "sum R all": 0.6431095406360424,
        "mean F 35": 0.7219983086346532,
        "mean P 35": 0.842222322255237,
        "mean R 35": 0.6737598236509639,
        "sum F 35": 0.7691187181354698,
        "sum P 35": 0.8748964374482188,
        "sum R 35": 0.6861598440545809,
        "mean F 38": 0.684621437782956,
        "mean P 38": 0.8356788938743932,
        "mean R 38": 0.6875599170976832,
        "sum F 38": 0.6775413711583924,
        "sum P 38": 0.9092639593908629,
        "sum R 38": 0.539939713639789,
        "mean F 47": 0.40570603735565375,
        "mean P 47": 0.4738724281058378,
        "mean R 47": 0.8366426736464543,
        "sum F 47": 0.26586102719033233,
        "sum P 47": 0.1825726141078838,
        "sum R 47": 0.4888888888888889,
        "mean F 42": 0.5941111175093938,
        "mean P 42": 0.7656595117444098,
        "mean R 42": 0.6792387930963699,
        "sum F 42": 0.8307381193124368,
        "sum P 42": 0.8907198612315698,
        "sum R 42": 0.7783251231527094,
        "mean F 49": 0.24344596450678943,
        "mean P 49": 0.2649529461016863,
        "mean R 49": 0.7214891361592733,
        "sum F 49": 0.38546005301022346,
        "sum P 49": 0.3105552165954851,
        "sum R 49": 0.5079840319361277,
    }
    MZ_YTLog70_ENSTWET = {
        "mean F all": 0.5992164847055067,
        "mean P all": 0.7368144788418904,
        "mean R all": 0.6614117536021175,
        "sum F all": 0.7057959354996677,
        "sum P all": 0.7361546880700475,
        "sum R all": 0.6778419779629132,
        "mean F 35": 0.7933397077859619,
        "mean P 35": 0.9295962252896087,
        "mean R 35": 0.7740001095582029,
        "sum F 35": 0.828869723104857,
        "sum P 35": 0.9178954423592494,
        "sum R 35": 0.7555862068965518,
        "mean F 38": 0.7098269232336427,
        "mean P 38": 0.85632120856465,
        "mean R 38": 0.6474420894258163,
        "sum F 38": 0.7269082056422741,
        "sum P 38": 0.9129496402877698,
        "sum R 38": 0.6038543897216274,
        "mean F 47": 0.40636412737521743,
        "mean P 47": 0.7440391156462585,
        "mean R 47": 0.38962378890950317,
        "sum F 47": 0.36950146627565983,
        "sum P 47": 0.5185185185185185,
        "sum R 47": 0.2870159453302961,
        "mean F 42": 0.710602975467719,
        "mean P 42": 0.8539015315390502,
        "mean R 42": 0.666731140429614,
        "sum F 42": 0.7605323946760532,
        "sum P 42": 0.8629555666500249,
        "sum R 42": 0.67984267453294,
        "mean F 49": 0.37594868966499256,
        "mean P 49": 0.3002143131698852,
        "mean R 49": 0.8292616396874507,
        "sum F 49": 0.4705206738131699,
        "sum P 49": 0.3328819068255688,
        "sum R 49": 0.8022193211488251,
    }

    # Trained on all
    MZ_All_CCLog70 = {
        "mean F all": 0.392398195063965,
        "mean P all": 0.5891009158881002,
        "mean R all": 0.4655871898611807,
        "sum F all": 0.5433482039248048,
        "sum P all": 0.6466666155156408,
        "sum R all": 0.46849621176326717,
        "mean F 35": 0.6192618897399833,
        "mean P 35": 0.7688964557602093,
        "mean R 35": 0.587078313927906,
        "sum F 35": 0.6584904696203178,
        "sum P 35": 0.7980182690819012,
        "sum R 35": 0.5604923772862704,
        "mean F 38": 0.6142011533516701,
        "mean P 38": 0.6353954777267268,
        "mean R 38": 0.6647722660050283,
        "sum F 38": 0.6311995027967682,
        "sum P 38": 0.643762677484787,
        "sum R 38": 0.6191172884662277,
        "mean F 47": 0.1903619567600459,
        "mean P 47": 0.4101312300581168,
        "mean R 47": 0.40725672722802286,
        "sum F 47": 0.17394848386044995,
        "sum P 47": 0.42442322991249004,
        "sum R 47": 0.10939101906909986,
        "mean F 42": 0.464346194739601,
        "mean P 42": 0.5415020565843117,
        "mean R 42": 0.5565537472597777,
        "sum F 42": 0.5385587466543175,
        "sum P 42": 0.5793633369923161,
        "sum R 42": 0.5031237167830117,
        "mean F 49": 0.07381978072852445,
        "mean P 49": 0.5895793593111361,
        "mean R 49": 0.11227489488516865,
        "sum F 49": 0.028672032193158958,
        "sum P 49": 0.3505535055350554,
        "sum R 49": 0.014947291131273929,
    }
    MZ_All_ENSTSUM = {
        "sum F 35": np.mean([0.48145, 0.88883, 0.89134]),
        "sum F 38": np.mean([0.51427, 0.63337, 0.58313]),
        "sum F 42": np.mean([0.75783, 0.74634, 0.76273]),
        "sum F 47": np.mean([0.079646, 0.099792, 0.041176]),
        "sum F 49": np.mean([0.18781, 0.40146, 0.33707]),
        "sum F all": np.mean([0.58237, 0.67458, 0.64888]),
    }
    MZ_ALL_ENSTWET = {
        "sum F 35": np.mean([0.72552, 0.86706, 0.94400]),
        "sum F 38": np.mean([0.62189, 0.80939, 0.77926]),
        "sum F 42": np.mean([0.76624, 0.75939, 0.86471]),
        "sum F 47": np.mean([0.11842, 0.11494, 0.18617]),
        "sum F 49": np.mean([0.097149, 0.39800, 0.43856]),
        "sum F all": np.mean([0.66195, 0.71654, 0.77032]),
    }
    MZ_ALL_MDB = {
        "sum F 35": np.mean([0.72744, 0.49153, 0.64407]),
        "sum F 38": np.mean([0.59672, 0.58068, 0.68506]),
        "sum F 42": np.mean([0.82405, 0.49786, 0.61980]),
        "sum F 47": np.mean([0.0000, 0.056180, 0.19231]),
        "sum F 49": np.mean([0.074074, 0.33849, 0.58707]),
        "sum F all": np.mean([0.71766, 0.48382, 0.62356]),
    }
    MZ_ALL_RBMA = {
        "sum F 35": np.mean([0.68778, 0.59864, 0.62517]),
        "sum F 38": np.mean([0.41213, 0.14032, 0.25509]),
        "sum F 42": np.mean([0.46014, 0.51216, 0.61412]),
        "sum F 47": np.mean([0.35810, 0.38914, 0.019429]),
        "sum F 49": np.mean([0.016997, 0.063872, 0.14724]),
        "sum F all": np.mean([0.51217, 0.43499, 0.50053]),
    }
    with open("crnn-all") as f:
        crnnAllJSON = json.load(f)
    crnnAll = {key: np.mean([values[key] for values in crnnAllJSON.values()]) for key in crnnAllJSON["crnn-all_Fold0"].keys()}

    with open("crnn-ptTMIDT") as f:
        crnnPtJSON = json.load(f)
    crnnPt = {key: np.mean([values[key] for values in crnnPtJSON.values()]) for key in crnnPtJSON["crnn-ptTMIDT_Fold0"].keys()}

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

    def map(dict, add="", keyPrefix=""):

        mapping = {
            "sum F all": "SUM" + add,
            "sum F 35": "BD",
            "sum F 38": "SD",
            "sum F 47": "TT",
            "sum F 42": "HH",
            "sum F 49": "CY + RD" + add,
        }
        return {v: dict[keyPrefix + k] for k, v in mapping.items()}

    # results = {
    #     "ADTOF": {"ADTOF": map(MZ_ADTOF), "All+MIDI": map(VOGL_ADTOF)},
    #     "RBMA": {"ADTOF": map(MZ_RBMA), "all+MIDI": map(VOGL_RBMA)},
    #     "MDB": {"ADTOF": map(MZ_MDB), "all+MIDI": map(VOGL_MDB)},
    #     "ENST": {"ADTOF": map(MZ_ENST_WET), "all+MIDI": map(VOGL_ENST)},
    # }
    # newPlot(results, "test")

    # newPlot(
    #     {
    #         "Test on ADTOF": map(VOGL_ADTOF_ALLMIDI),
    #         "Test on RBMA": map(VOGL_RBMA_ALLMIDI),
    #         "Test on MDB": map(VOGL_MDB_ALLMIDI),
    #         "Test on ENST": map(VOGL_ENST_ALLMIDI),
    #     },
    #     "",
    #     legend=True,
    # )

    # Plot only one bar
    # newPlot({"Train on RBMA, ENST, MDB, and TMIDT": map(VOGL_ADTOF_ALLMIDI)}, "Test on ADTOF", legend=True)
    # newPlot(
    #     {"Train on RBMA, ENST, MDB, and TMIDT": map(VOGL_RBMA_ALLMIDI)}, "Test on RBMA", legend=False,
    # )  # "pt MIDI": map(VOGL_RBMA_PTMIDI)
    # newPlot(
    #     {"Train on RBMA, ENST, MDB, and TMIDT": map(VOGL_MDB_ALLMIDI)}, "Test on MDB", legend=False,
    # )  # , "pt MIDI": map(VOGL_MDB_PTMIDI)
    # newPlot(
    #     {"Train on RBMA, ENST, MDB, and TMIDT": map(VOGL_ENST_ALLMIDI)}, "Test on ENST", legend=False,
    # )  # , "pt MIDI": map(VOGL_ENST_PTMIDI)
    # plt.show()

    # newPlot(
    #     {"Train on ADTOF CC0": map(MZ_CC0_CC0), "Ensemble": map(VOGL_ENSEMBLE_CC0),}, "Test on ADTOF CC0", legend=True,
    # )
    newPlot(
        {
            "Train on ADTOF CCLog70": map(MZ_CCLog70_CCLog70_Fold0),
            "Train on ADTOF CCLog70 fold 1": map(MZ_CCLog70_CCLog70_Fold1),
            "Train on all MZ": map(MZ_All_CCLog70),
            "Train on all MZ good save": map(crnnAll, keyPrefix="adtof_"),
            "Train on pt MIDI MZ": map(crnnPt, keyPrefix="adtof_"),
            "Ensemble": map(VOGL_ENSEMBLE_CCLog70),
        },
        "Test on ADTOF CCLog70",
        legend=True,
    )
    newPlot(
        {
            "Train on ADTOF CC0": map(MZ_CC0_RBMA),
            "Train on ADTOF CCLog70": map(MZ_CCLog70_RBMA),
            # "Train on ADTOF RBLog70": map(MZ_RBLog70_RBMA),
            # "Train on ADTOF YTLog70": map(MZ_YTLog70_RBMA),
            "Train on all MZ": map(MZ_ALL_RBMA),
            "Train on all MZ good save": map(crnnAll, keyPrefix="rbma_"),
            "Train on all Vogl": map(VOGL_ALL_RBMA),
            "Train on pt MIDI MZ": map(crnnPt, keyPrefix="rbma_"),
            "Train on pt MIDI Vogl": map(VOGL_PTMIDI_RBMA),
            # "Train on RBMA, ENST, MDB, and TMIDT": map(VOGL_ALLMIDI_RBMA),
        },
        "Test on RBMA",
        legend=True,
    )  # "pt MIDI": map(VOGL_RBMA_PTMIDI)
    newPlot(
        {
            "Train on ADTOF CC0": map(MZ_CC0_MDB),
            "Train on ADTOF CCLog70": map(MZ_CCLog70_MDB),
            # "Train on ADTOF RBLog70": map(MZ_RBLog70_MDB),
            # "Train on ADTOF YTLog70": map(MZ_YTLog70_MDB),
            "Train on all MZ": map(MZ_ALL_MDB),
            "Train on all MZ good save": map(crnnAll, keyPrefix="mdb_"),
            "Train on all Vogl": map(VOGL_ALL_MDB),
            "Train on pt MIDI MZ": map(crnnPt, keyPrefix="mdb_"),
            "Train on pt MIDI Vogl": map(VOGL_PTMIDI_MDB),
            # "Train on RBMA, ENST, MDB, and TMIDT": map(VOGL_ALLMIDI_MDB),
        },
        "Test on MDB",
        legend=False,
    )  # , "pt MIDI": map(VOGL_MDB_PTMIDI)
    newPlot(
        {
            "Train on ADTOF CC0": map(MZ_CC0_ENSTWET),
            "Train on ADTOF CCLog70": map(MZ_CCLog70_ENSTWET),
            # "Train on ADTOF RBLog70": map(MZ_RBLog70_ENSTWET),
            # "Train on ADTOF YTLog70": map(MZ_YTLog70_ENSTWET),
            "Train on all MZ": map(MZ_ALL_ENSTWET),
            "Train on all MZ good save": map(crnnAll, keyPrefix="enst_wet_"),
            "Train on all Vogl": map(VOGL_ALL_ENSTWET),
            "Train on pt MIDI MZ": map(crnnPt, keyPrefix="enst_wet_"),
            "Train on pt MIDI Vogl": map(VOGL_PTMIDI_ENSTWET),
            # "Train on RBMA, ENST, MDB, and TMIDT": map(VOGL_ALLMIDI_ENSTWET),
        },
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

    newPlot({"": sum}, "Instrument classes", ylim=False, legend=False, sort=True, ylabel="Count")
    plt.show()


if __name__ == "__main__":
    # main()
    plotResults()
    # plotInstrumentClasses()
