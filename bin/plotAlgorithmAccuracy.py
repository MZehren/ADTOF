#!/usr/bin/env python
# encoding: utf-8

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(dict, title, ylim=True, legend=True, sort=False, ylabel="F-measure", text=True):
    """
    Plot the dict as a barchart
    """
    # Load the dictionary into pandas
    df = pd.DataFrame(dict)
    if sort:
        df = df.sort_values("", ascending=False)

    # Specify the size for the paper
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

    # Add the legend to the bottom
    if legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)

    # add the background lines and labels
    # plt.xticks(ind + width, groups)
    plt.grid(axis="y", linestyle="--")
    if ylim:
        plt.ylim(0, 1)
    plt.ylabel(ylabel)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xticks(rotation=0)
    plt.title(title)

    # Add the text on top of the bars
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


def map(dict, keyPrefixes=[""], score="F"):
    """
    Retrieve the json fields used for the plot and map their name to the paper nomenclature
    """
    mapping = {
        "sum " + score + " all": "SUM",
        "sum " + score + " 35": "BD",
        "sum " + score + " 38": "SD",
        "sum " + score + " 47": "TT",
        "sum " + score + " 42": "HH",
        "sum " + score + " 49": "CY + RD",
    }
    return {v: np.mean([dict[pre + k] for pre in keyPrefixes]) for k, v in mapping.items()}


def plotResults():
    """
    Load the json files containing the cross-validation result and plot them
    """
    # Get results as the mean of the three evaluations
    with open("evaluation/crnn-all.json") as f:
        crnnAllJSON = json.load(f)
    crnnAll = {key: np.mean([values[key] for values in crnnAllJSON.values()]) for key in crnnAllJSON["crnn-all_Fold0"].keys()}

    with open("evaluation/crnn-ptTMIDT.json") as f:
        crnnPtJSON = json.load(f)
    crnnPt = {key: np.mean([values[key] for values in crnnPtJSON.values()]) for key in crnnPtJSON["crnn-ptTMIDT_Fold0"].keys()}

    with open("evaluation/crnn-ADTOF.json") as f:
        crnnADTOFJSON = json.load(f)
    crnnADTOF = {key: np.mean([values[key] for values in crnnADTOFJSON.values()]) for key in crnnADTOFJSON["crnn-CC_Fold0"].keys()}

    plot(
        {
            "Train on ENST, MDB, and RBMA": map(crnnAll, keyPrefixes=["adtof_"]),
            "Train on TMIDT and refinement on ENST, MDB, and RBMA": map(crnnPt, keyPrefixes=["adtof_"]),
            "Train on ADTOF": map(crnnADTOF, keyPrefixes=["adtof_"]),
        },
        "Test on ADTOF",
        legend=False,
    )
    plot(
        {
            "Train on ENST, MDB, and RBMA": map(crnnAll, keyPrefixes=["rbma_"]),
            "Train on TMIDT and refinement on ENST, MDB, and RBMA": map(crnnPt, keyPrefixes=["rbma_"]),
            "Train on ADTOF": map(crnnADTOF, keyPrefixes=["rbma_"]),
        },
        "Test on RBMA",
        legend=True,
    )
    plot(
        {
            "Train on RBMA, MDB, and ENST": map(crnnAll, keyPrefixes=["mdb_full_mix_"]),
            "Train on TMIDT and refinement on ENST, MDB, and RBMA": map(crnnPt, keyPrefixes=["mdb_full_mix_"]),
            "Train on ADTOF": map(crnnADTOF, keyPrefixes=["mdb_full_mix_"]),
        },
        "Test on MDB",
        legend=False,
    )
    plot(
        {
            "Train on RBMA, MDB, and ENST": map(crnnAll, keyPrefixes=["enst_sum_"]),
            "Train on TMIDT and refinement on ENST, MDB, and RBMA": map(crnnPt, keyPrefixes=["enst_sum_"]),
            "Train on ADTOF": map(crnnADTOF, keyPrefixes=["enst_sum_"]),
        },
        "Test on ENST",
        legend=False,
    )

    plt.show()


if __name__ == "__main__":
    plotResults()
