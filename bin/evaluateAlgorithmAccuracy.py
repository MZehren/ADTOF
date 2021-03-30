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
            ax.text(p._x0 + 0.01, p._y1, str(np.round(p._y1, decimals=2)))

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
        "mean F all": 0.389084027985125,
        "mean P all": 0.37401950901504505,
        "mean R all": 0.7178835123027728,
        "sum F all": 0.5721199971361065,
        "sum P all": 0.4668505059475123,
        "sum R all": 0.7386851057535867,
        "mean F 35": 0.7500398928298642,
        "mean P 35": 0.6543587091469685,
        "mean R 35": 0.9624583330116844,
        "sum F 35": 0.7440333507034914,
        "sum P 35": 0.6038741329724243,
        "sum R 35": 0.9689196525515744,
        "mean F 38": 0.3460437395835728,
        "mean P 38": 0.30605289214283954,
        "mean R 38": 0.767359182702478,
        "sum F 38": 0.5363941769316908,
        "sum P 38": 0.42172917767212537,
        "sum R 38": 0.7366964011073516,
        "mean F 47": 0.18655666847758873,
        "mean P 47": 0.2251196505370512,
        "mean R 47": 0.4917934313395019,
        "sum F 47": 0.26982468361778705,
        "sum P 47": 0.21833897031191282,
        "sum R 47": 0.35308416894560923,
        "mean F 42": 0.5471539283548732,
        "mean P 42": 0.4934885525505764,
        "mean R 42": 0.7985463577660803,
        "sum F 42": 0.5912276148455748,
        "sum P 42": 0.49034063391271593,
        "sum R 42": 0.7443836287884976,
        "mean F 49": 0.11562591067972629,
        "mean P 49": 0.19107774069778952,
        "mean R 49": 0.5692602566941192,
        "sum F 49": 0.14837576821773485,
        "sum P 49": 0.1220216606498195,
        "sum R 49": 0.18924972004479285,
    }
    MZ_CCLog70_MDB = {
        "mean F all": 0.5252136502801041,
        "mean P all": 0.5715850573418201,
        "mean R all": 0.7782816628325353,
        "sum F all": 0.6803952821166719,
        "sum P all": 0.6875402654297127,
        "sum R all": 0.6733972741039879,
        "mean F 35": 0.7944250232577793,
        "mean P 35": 0.7713145855717282,
        "mean R 35": 0.8582764360881706,
        "sum F 35": 0.7662994938969933,
        "sum P 35": 0.7071428571428572,
        "sum R 35": 0.8362573099415205,
        "mean F 38": 0.7066810301925202,
        "mean P 38": 0.8768000526246594,
        "mean R 38": 0.6791899279558886,
        "sum F 38": 0.6571566731141198,
        "sum P 38": 0.917004048582996,
        "sum R 38": 0.5120572720422004,
        "mean F 47": 0.2422040812409473,
        "mean P 47": 0.26122079572289947,
        "mean R 47": 0.8634591149241433,
        "sum F 47": 0.18003913894324852,
        "sum P 47": 0.10926365795724466,
        "sum R 47": 0.5111111111111111,
        "mean F 42": 0.5727534859130108,
        "mean P 42": 0.6251999511293609,
        "mean R 42": 0.7782408768341418,
        "sum F 42": 0.7761090971862592,
        "sum P 42": 0.7127457197209892,
        "sum R 42": 0.8518378173550587,
        "mean F 49": 0.31000463079626334,
        "mean P 49": 0.32338990166045184,
        "mean R 49": 0.7122419583603324,
        "sum F 49": 0.4199363732767763,
        "sum P 49": 0.4479638009049774,
        "sum R 49": 0.39520958083832336,
    }
    MZ_CCLog70_ENSTWET = {
        "mean F all": 0.5931138875568629,
        "mean P all": 0.6841634887811615,
        "mean R all": 0.671099801839817,
        "sum F all": 0.71035620959978,
        "sum P all": 0.7274647887323944,
        "sum R all": 0.6940338618650901,
        "mean F 35": 0.8423576217679951,
        "mean P 35": 0.9601678782379797,
        "mean R 35": 0.8020235619820981,
        "sum F 35": 0.8664249017830159,
        "sum P 35": 0.9579017707985299,
        "sum R 35": 0.790896551724138,
        "mean F 38": 0.5947897175940463,
        "mean P 38": 0.7858955623210752,
        "mean R 38": 0.5507233937265306,
        "sum F 38": 0.6163178647255081,
        "sum P 38": 0.849853617733166,
        "sum R 38": 0.4834641922436355,
        "mean F 47": 0.37916883693241754,
        "mean P 47": 0.588932375318471,
        "mean R 47": 0.4268060709132137,
        "sum F 47": 0.3108695652173913,
        "sum P 47": 0.2972972972972973,
        "sum R 47": 0.32574031890660593,
        "mean F 42": 0.7588238460240796,
        "mean P 42": 0.7193526048178925,
        "mean R 42": 0.8467537131288879,
        "sum F 42": 0.7903094093095,
        "sum P 42": 0.7336590296495957,
        "sum R 42": 0.856440511307768,
        "mean F 49": 0.39042941546577575,
        "mean P 49": 0.36646902321038893,
        "mean R 49": 0.7291922694483545,
        "sum F 49": 0.4746883744594251,
        "sum P 49": 0.38891204668611923,
        "sum R 49": 0.6090078328981723,
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
        "mean F all": 0.362491645129928,
        "mean P all": 0.4526536330008492,
        "mean R all": 0.6266340798459303,
        "sum F all": 0.5279091769157995,
        "sum P all": 0.5103012734237499,
        "sum R all": 0.546775624907558,
        "mean F 35": 0.7873470792808488,
        "mean P 35": 0.720558976494605,
        "mean R 35": 0.928657433116183,
        "sum F 35": 0.784385763490241,
        "sum P 35": 0.6796657381615598,
        "sum R 35": 0.9272529858849077,
        "mean F 38": 0.389138246511222,
        "mean P 38": 0.38644830157537674,
        "mean R 38": 0.7136775445406704,
        "sum F 38": 0.5608507286333202,
        "sum P 38": 0.48923499770957396,
        "sum R 38": 0.65702860658259,
        "mean F 47": 0.15036101572931435,
        "mean P 47": 0.16232578648210488,
        "mean R 47": 0.5121305351575982,
        "sum F 47": 0.28630363036303635,
        "sum P 47": 0.21670569867291178,
        "sum R 47": 0.42175630507444545,
        "mean F 42": 0.40069859893513965,
        "mean P 42": 0.5906285304789232,
        "mean R 42": 0.4495399376073942,
        "sum F 42": 0.44018553443090885,
        "sum P 42": 0.5852534562211982,
        "sum R 42": 0.35274895841842985,
        "mean F 49": 0.0849132851931149,
        "mean P 49": 0.4033065699732366,
        "mean R 49": 0.529164948807806,
        "sum F 49": 0.1354104254044338,
        "sum P 49": 0.14561855670103094,
        "sum R 49": 0.1265397536394177,
    }
    MZ_RBLog70_MDB = {
        "mean F all": 0.5846350750697675,
        "mean P all": 0.7152132703370354,
        "mean R all": 0.7126497059879666,
        "sum F all": 0.698762233736327,
        "sum P all": 0.8129604822505023,
        "sum R all": 0.6126956082786471,
        "mean F 35": 0.8141293253882745,
        "mean P 35": 0.8351107652528044,
        "mean R 35": 0.8138881792550913,
        "sum F 35": 0.8052459016393442,
        "sum P 35": 0.8127068166776968,
        "sum R 35": 0.7979207277452891,
        "mean F 38": 0.6931204983559974,
        "mean P 38": 0.9118477476449559,
        "mean R 38": 0.6500778502887234,
        "sum F 38": 0.6186546636659165,
        "sum P 38": 0.9197026022304833,
        "sum R 38": 0.4660889223813112,
        "mean F 47": 0.3697863853550484,
        "mean P 47": 0.4083496897242213,
        "mean R 47": 0.8459478454752557,
        "sum F 47": 0.19954648526077098,
        "sum P 47": 0.12535612535612536,
        "sum R 47": 0.4888888888888889,
        "mean F 42": 0.590502054008424,
        "mean P 42": 0.8206605785400758,
        "mean R 42": 0.6393075477182084,
        "sum F 42": 0.7961443806398687,
        "sum P 42": 0.867679928475637,
        "sum R 42": 0.7355058734369079,
        "mean F 49": 0.45563711224109305,
        "mean P 49": 0.6000975705231192,
        "mean R 49": 0.6140271072025542,
        "sum F 49": 0.5294117647058824,
        "sum P 49": 0.7670454545454546,
        "sum R 49": 0.4041916167664671,
    }
    MZ_RBLog70_ENSTWET = {
        "mean F all": 0.5508239506913521,
        "mean P all": 0.7041608957832477,
        "mean R all": 0.5808873757894419,
        "sum F all": 0.6904451554291582,
        "sum P all": 0.8377264449343429,
        "sum R all": 0.5872077398548777,
        "mean F 35": 0.8028843344457467,
        "mean P 35": 0.9798521904073877,
        "mean R 35": 0.7561269948299613,
        "sum F 35": 0.8474258970358816,
        "sum P 35": 0.9752244165170556,
        "sum R 35": 0.7492413793103448,
        "mean F 38": 0.5838895386046511,
        "mean P 38": 0.789808614552304,
        "mean R 38": 0.5359374077236831,
        "sum F 38": 0.5981250960504073,
        "sum P 38": 0.8446180555555556,
        "sum R 38": 0.46300261717820607,
        "mean F 47": 0.2600140772110721,
        "mean P 47": 0.2630045742088334,
        "mean R 47": 0.5086219336219336,
        "sum F 47": 0.2270947533281128,
        "sum P 47": 0.1730310262529833,
        "sum R 47": 0.33029612756264237,
        "mean F 42": 0.6735261950759188,
        "mean P 42": 0.916719583127259,
        "mean R 42": 0.6082818831183332,
        "sum F 42": 0.7487262621584068,
        "sum P 42": 0.9104477611940298,
        "sum R 42": 0.6357915437561455,
        "mean F 49": 0.4338056081193722,
        "mean P 49": 0.5714195166204549,
        "mean R 49": 0.49546865965329834,
        "sum F 49": 0.5629272215520708,
        "sum P 49": 0.7329842931937173,
        "sum R 49": 0.45691906005221933,
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
        "mean F all": 0.3434484701568534,
        "mean P all": 0.37982152389913804,
        "mean R all": 0.6155783402979002,
        "sum F all": 0.47776241696732646,
        "sum P all": 0.43907654167957855,
        "sum R all": 0.5239239757432332,
        "mean F 35": 0.7543998121287548,
        "mean P 35": 0.6930502177385811,
        "mean R 35": 0.894594436577568,
        "sum F 35": 0.7650715793075914,
        "sum P 35": 0.6598130841121496,
        "sum R 35": 0.9102877307274702,
        "mean F 38": 0.31939845684884965,
        "mean P 38": 0.3293555399691705,
        "mean R 38": 0.6538042926782713,
        "sum F 38": 0.5017596782302665,
        "sum P 38": 0.42422954303931987,
        "sum R 38": 0.6139649338665026,
        "mean F 47": 0.15700795430839026,
        "mean P 47": 0.19119099309571994,
        "mean R 47": 0.4139728752015439,
        "sum F 47": 0.2953076706940152,
        "sum P 47": 0.2746276456754638,
        "sum R 47": 0.3193558189000304,
        "mean F 42": 0.3700165079221191,
        "mean P 42": 0.542317039762662,
        "mean R 42": 0.4420013572810321,
        "sum F 42": 0.41707683539712964,
        "sum P 42": 0.5701133144475921,
        "sum R 42": 0.328813005473409,
        "mean F 49": 0.11641961957615343,
        "mean P 49": 0.1431938289295567,
        "mean R 49": 0.6735187397510857,
        "sum F 49": 0.10532001080205237,
        "sum P 49": 0.059880239520958084,
        "sum R 49": 0.43673012318029114,
    }
    MZ_YTLog70_MDB = {
        "mean F all": 0.5262600259454832,
        "mean P all": 0.6322947982505607,
        "mean R all": 0.7161602098420154,
        "sum F all": 0.6785738266299605,
        "sum P all": 0.725068158989812,
        "sum R all": 0.6376829883897022,
        "mean F 35": 0.7134746947414591,
        "mean P 35": 0.8336248702230507,
        "mean R 35": 0.6653025047441973,
        "sum F 35": 0.7625637290604516,
        "sum P 35": 0.8674399337199669,
        "sum R 35": 0.6803118908382066,
        "mean F 38": 0.6826378451718385,
        "mean P 38": 0.8320955207258026,
        "mean R 38": 0.686146359260846,
        "sum F 38": 0.6718676122931443,
        "sum P 38": 0.9016497461928934,
        "sum R 38": 0.5354182366239638,
        "mean F 47": 0.40570603735565375,
        "mean P 47": 0.4738724281058378,
        "mean R 47": 0.8366426736464543,
        "sum F 47": 0.26586102719033233,
        "sum P 47": 0.1825726141078838,
        "sum R 47": 0.4888888888888889,
        "mean F 42": 0.590663936407466,
        "mean P 42": 0.7619250408758486,
        "mean R 42": 0.6759818026647035,
        "sum F 42": 0.8258847320525784,
        "sum P 42": 0.8855160450997398,
        "sum R 42": 0.7737779461917393,
        "mean F 49": 0.23881761605099835,
        "mean P 49": 0.2599561313222628,
        "mean R 49": 0.7167277088938766,
        "sum F 49": 0.3778871639530481,
        "sum P 49": 0.30445393532641857,
        "sum R 49": 0.49800399201596807,
    }
    MZ_YTLog70_ENSTWET = {
        "mean F all": 0.5963817630272593,
        "mean P all": 0.7338586417508117,
        "mean R all": 0.658025859913766,
        "sum F all": 0.7020882157473154,
        "sum P all": 0.7322874863188618,
        "sum R all": 0.6742811072292395,
        "mean F 35": 0.7921081806923658,
        "mean P 35": 0.9278955450174998,
        "mean R 35": 0.7730348585929521,
        "sum F 35": 0.8285671054622484,
        "sum P 35": 0.9175603217158177,
        "sum R 35": 0.7553103448275862,
        "mean F 38": 0.7082911957601177,
        "mean P 38": 0.854322896589763,
        "mean R 38": 0.6460925835227652,
        "sum F 38": 0.7249033366747817,
        "sum P 38": 0.910431654676259,
        "sum R 38": 0.602188912681418,
        "mean F 47": 0.40636412737521743,
        "mean P 47": 0.7440391156462585,
        "mean R 47": 0.38962378890950317,
        "sum F 47": 0.36950146627565983,
        "sum P 47": 0.5185185185185185,
        "sum R 47": 0.2870159453302961,
        "mean F 42": 0.7014682779351258,
        "mean P 42": 0.8449568134703036,
        "mean R 42": 0.6550536119900396,
        "sum F 42": 0.7528324716752833,
        "sum P 42": 0.854218671992012,
        "sum R 42": 0.6729596853490659,
        "mean F 49": 0.3736770333734696,
        "mean P 49": 0.2980788380302335,
        "mean R 49": 0.8263244565535703,
        "sum F 49": 0.4666921898928024,
        "sum P 49": 0.3301733477789816,
        "sum R 49": 0.7956919060052219,
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
    VOGL_ALLMIDI_ENST = {
        "sum F all": 0.62,
        "sum F 35": 0.79,
        "sum F 38": 0.53,
        "sum F 47": 0.16,
        "sum F 42": 0.75,
        "sum F 49": np.mean([0.07, 0.21, 0.02]),
    }

    # Values from the pre-trained model
    VOGL_ENSEMBLE_ADTOF = {
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

    # Approximative value from the plot TODO: double check, there was an error
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
        "sum F 42": 0.58,
        "sum F 49": np.mean([0.29, 0.55, 0.18]),
    }
    VOGL_PTMIDI_ENST = {
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

    newPlot(
        {
            "Train on ADTOF CC0": map(MZ_CC0_CC0),
            "Train on ADTOF CCLog70": map(MZ_CCLog70_CCLog70_Fold0),
            "Train on ADTOF RBLog70": map(MZ_RBLog70_RBLog70_Fold0),
            "Train on ADTOF YTLog70": map(MZ_YTLog70_YTLog70_Fold0),
            "Train on RBMA, ENST, MDB, and TMIDT": map(VOGL_ENSEMBLE_ADTOF),
        },
        "Test on ADTOF CC0, ",
        legend=True,
    )
    newPlot(
        {
            "Train on ADTOF CC0": map(MZ_CC0_RBMA, add="*"),
            "Train on ADTOF CCLog70": map(MZ_CCLog70_RBMA, add="*"),
            "Train on ADTOF RBLog70": map(MZ_RBLog70_RBMA, add="*"),
            "Train on ADTOF YTLog70": map(MZ_YTLog70_RBMA, add="*"),
            "Train on RBMA, ENST, MDB, and TMIDT": map(VOGL_ALLMIDI_RBMA, add="*"),
        },
        "Test on RBMA",
        legend=False,
    )  # "pt MIDI": map(VOGL_RBMA_PTMIDI)
    newPlot(
        {
            "Train on ADTOF CC0": map(MZ_CC0_MDB, add="*"),
            "Train on ADTOF CCLog70": map(MZ_CCLog70_MDB, add="*"),
            "Train on ADTOF RBLog70": map(MZ_RBLog70_MDB, add="*"),
            "Train on ADTOF YTLog70": map(MZ_YTLog70_MDB, add="*"),
            "Train on RBMA, ENST, MDB, and TMIDT": map(VOGL_ALLMIDI_MDB, add="*"),
        },
        "Test on MDB",
        legend=False,
    )  # , "pt MIDI": map(VOGL_MDB_PTMIDI)
    newPlot(
        {
            "Train on ADTOF CC0": map(MZ_CC0_ENSTWET, add="*"),
            "Train on ADTOF CCLog70": map(MZ_CCLog70_ENSTWET, add="*"),
            "Train on ADTOF RBLog70": map(MZ_RBLog70_ENSTWET, add="*"),
            "Train on ADTOF YTLog70": map(MZ_YTLog70_ENSTWET, add="*"),
            "Train on RBMA, ENST, MDB, and TMIDT": map(VOGL_ALLMIDI_ENST, add="*"),
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
    main()
    plotResults()
    # plotInstrumentClasses()
