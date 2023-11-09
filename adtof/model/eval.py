import numpy as np
import mir_eval
from collections import defaultdict
from adtof import config
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering


def runEvaluation(groundTruths, estimations, paths=[], distanceThreshold=0.05, classes=config.LABELS_5, octave=False):
    """
    Evaluate predictions according to ground truth estimations. Uses the hit rate metric with a small window.

    Parameters
    ----------
    groundTruths : list[Dict]
        Expects a list of tracks, each track is a dictionary mapping the classes to the onsets {class: [positions]}. See textReader.getOnsets()
    estimations : list[Dict]
        Expects a list of tracks, each track is a dictionary mapping the classes to the onsets {class: [positions]}. See textReader.getOnsets()
    paths : list[string], optional
        For debug logging when a track performs badly, by default []
    distanceThreshold : float, optional
        Aka Window: Tolerance distance to match predict and annotated onsets (see mir_eval.util.match_events),
        0.05 is similar to prior studies, 0.03 is similar to MIREX. By default 0.05
    classes : [type], optional
        On which classes to compute the results (ie. if the model predicted more classes), by default config.LABELS_5

    Returns
    -------
    dict
        return the sum and mean F,P,R scores per classes and for all classes.
    """
    assert len(groundTruths) == len(estimations)

    meanResults = {pitch: defaultdict(list) for pitch in classes}
    sumResults = {pitch: defaultdict(int) for pitch in classes + ["all"]}
    for i, (groundTruth, estimation) in enumerate(zip(groundTruths, estimations)):
        for pitch in classes:
            y_truth = np.array(groundTruth[pitch]) if pitch in groundTruth else np.array([])
            y_pred = np.array(estimation[pitch]) if pitch in estimation else np.array([])

            # Compute the metric with allowed rock beat
            if octave:
                # Double the predictions and the GT to emmulate a rock beat in both
                y_pred_octave = sorted(list(y_pred) + [(y_pred[i] + y_pred[i + 1]) / 2 for i in range(len(y_pred) - 1)])
                y_truth_octave = sorted(list(y_truth) + [(y_truth[i] + y_truth[i + 1]) / 2 for i in range(len(y_truth) - 1)])
                # Compute the number of matches added by either doubling the predictions or the GT
                matches_pred_octave = [(y_truth[i], y_pred_octave[j]) for i, j in mir_eval.util.match_events(y_truth, y_pred_octave, distanceThreshold)]
                matches_truth_octave = [(y_truth_octave[i], y_pred[j]) for i, j in mir_eval.util.match_events(y_truth_octave, y_pred, distanceThreshold)]
                # The number of TP now also considers the fake GT events in between two annotations (The GT did not annotate a rock beat)
                tp = len(matches_truth_octave)
                # The number of FP is still all the events without a match
                fp = len(y_pred) - tp
                # the nuber of FN takes into account the fake prediction event in between two predictions (the prediction missed the rock beat)
                fn = len(y_truth) - len(matches_pred_octave)
            else:
                matches = [(y_truth[i], y_pred[j]) for i, j in mir_eval.util.match_events(y_truth, y_pred, distanceThreshold)]
                tp = len(matches)
                fp = len(y_pred) - tp
                fn = len(y_truth) - tp

            f, p, r = getF1(tp, fp, fn)
            meanResults[pitch]["F"].append(f)
            meanResults[pitch]["P"].append(p)
            meanResults[pitch]["R"].append(r)
            sumResults[pitch]["TP"] += tp
            sumResults[pitch]["FP"] += fp
            sumResults[pitch]["FN"] += fn
            sumResults["all"]["TP"] += tp
            sumResults["all"]["FP"] += fp
            sumResults["all"]["FN"] += fn

        trackF = np.mean([meanResults[pitch]["F"][-1] for pitch in classes])
        # if trackF < 0.1:
        #     logging.debug("Alert, track performed badly. Score: %s, Track:%s", str(trackF), paths[i] if i in paths else i)

    result = {}

    # score for all classes
    for F in ["F", "P", "R"]:
        result["mean " + str(F) + " " + "all"] = np.mean([meanResults[pitch][F] for pitch in classes])
    f, p, r = getF1(sumResults["all"]["TP"], sumResults["all"]["FP"], sumResults["all"]["FN"])
    result["sum F all"] = f
    result["sum P all"] = p
    result["sum R all"] = r
    result["sum TP all"] = sumResults["all"]["TP"]
    result["sum FP all"] = sumResults["all"]["FP"]
    result["sum FN all"] = sumResults["all"]["FN"]

    # Score for specific pitches
    for pitch in classes:
        for F in ["F", "P", "R"]:
            result["mean " + F + " " + str(pitch)] = np.mean(meanResults[pitch][F])

        f, p, r = getF1(sumResults[pitch]["TP"], sumResults[pitch]["FP"], sumResults[pitch]["FN"])
        result["sum F " + str(pitch)] = f
        result["sum P " + str(pitch)] = p
        result["sum R " + str(pitch)] = r

    return result


def getF1(tp, fp, fn):
    """Compute the precision, recall and F-Measure from the number of true positives,
    false positives and false negatives. Based on Madmom implementation which is different from mir_eval

    Args:
        tp (int): true positives
        fp (int): false positives
        fn (int): false negatives

    Returns:
        (int, int, int): f, p, r
    """
    # MIR eval
    # if (tp + fn) == 0 or (tp + fp) == 0:
    #     return 0., 0., 0.

    # if there are no positive predictions, none of them are wrong
    if (tp + fp) == 0:
        p = 1.0
    else:
        p = tp / (tp + fp)

    # if there are no positive annotations, we recalled all of them
    if (tp + fn) == 0:
        r = 1.0
    else:
        r = tp / (tp + fn)

    # f = 2 * tp / (2 * tp + fp + fn)
    numerator = 2 * (p * r)
    if numerator == 0:
        f = 0.0
    else:
        f = numerator / (p + r)
    return f, p, r


def plotPseudoConfusionMatrices(groundTruths, estimations, distanceThreshold=0.05, **kwargs):
    """
    Densify the GTs and estimations and then call plotPseudoConfusionMatricesFromDense
    GTs and estimations are in a sparse shape: [{35:[0,1], 38:[0,1]}, {35:[0,1], 38:[0,1]}]
    The densification works by clustering all the events (GT and estimations) close together
    """
    assert len(groundTruths) == len(estimations)
    denseGT = []
    denseEstimation = []

    # For each track
    for i, (groundTruth, estimation) in enumerate(zip(groundTruths, estimations)):
        # Cluster the estimated and annotated event
        try:
            times = [time for label, times in estimation.items() for time in times]
            labels = [(False, label) for label, times in estimation.items() for time in times]
            times += [time for label, times in groundTruth.items() for time in times]
            labels += [(True, label) for label, times in groundTruth.items() for time in times]
            clustersLabel = AgglomerativeClustering(distance_threshold=distanceThreshold, n_clusters=None).fit_predict(np.array(times).reshape(-1, 1))
            clustering = defaultdict(list)
            for i, cluster in enumerate(clustersLabel):
                clustering[cluster].append(labels[i])

            # Put the labels form the cluster in the GT and estimation with matching index
            for clusterLabels in clustering.values():
                denseGT.append(sorted([v[1] for v in clusterLabels if v[0]]))
                denseEstimation.append(sorted([v[1] for v in clusterLabels if not v[0]]))
        except Exception as e:
            logging.debug("Error in track %s: %s", i, e)

    plotPseudoConfusionMatricesFromDense(denseGT, denseEstimation, **kwargs)


def plotPseudoConfusionMatricesFromDense(denseGT, denseEstimation, saveFigure=None, **kwargs):
    """
    Plot the different from the GTs and estimations, inspired by Vogl: Towards multi-instrument drum transcription, 2018
    The GTs and estimations are in the shape: [[35,42],[35,42,47]].
    The same index in GT and estimation should correspond to the same location
    """
    assert len(denseGT) == len(denseEstimation)
    difference = defaultdict(lambda: defaultdict(int))
    classicConfusion = defaultdict(lambda: defaultdict(int))
    onsetMasking = defaultdict(lambda: defaultdict(int))
    positiveMasking = defaultdict(lambda: defaultdict(int))
    for i in range(len(denseGT)):
        if denseGT[i] != denseEstimation[i]:
            FP = [v for v in denseEstimation[i] if v not in denseGT[i]]
            FN = [v for v in denseGT[i] if v not in denseEstimation[i]]
            TP = [v for v in denseEstimation[i] if v in denseGT[i]]
            FP = [0] if len(FP) == 0 else FP
            FN = [0] if len(FN) == 0 else FN
            TP = [0] if len(TP) == 0 else TP

            # raw missmatch between GT / Estimation
            # Remember that Pandas is column first (i.e, specify dict[x][y])
            difference[str(denseEstimation[i])][str(denseGT[i])] += 1

            # Confusion (one class is confused with another)
            for fp in FP:
                for fn in FN:
                    classicConfusion[fp][fn] += 1
            # classicConfusion[str(FP)][str(FN)] += 1

            # onsetmasking (one class hides another)
            for tp in TP:
                for fn in FN:
                    onsetMasking[tp][fn] += 1
            # onsetMasking[str(TP)][str(FN)] += 1

            # Positive masking (one class creates a false detection to another)
            for tp in TP:
                for fp in FP:
                    positiveMasking[tp][fp] += 1

    if np.sum([v for rows in difference.values() for v in rows.values()]) > 50:
        plotHeatmap(difference, ylabel="Ground Truth", xlabel="Estimations", saveFigure=saveFigure + "-confusion.pdf", **kwargs)
    plotHeatmap(
        classicConfusion,
        ylabel="Missing (fn)",
        xlabel="Additional (fp)",
        saveFigure=saveFigure + "-PseudoConfusion.pdf",
    )
    plotHeatmap(
        onsetMasking,
        ylabel="Missing (fn)",
        xlabel="Detected (tp)",
        saveFigure=saveFigure + "-masking.pdf",
    )
    plotHeatmap(
        positiveMasking,
        ylabel="Additional (fp)",
        xlabel="Detected (tp)",
        saveFigure=saveFigure + "-excitation.pdf",
    )


def plotHeatmap(heatmap, ylabel="Ground Truth", xlabel="Estimation", title="", saveFigure=None, text=False, **kwargs):
    width = 426 / 72.27 / 2
    plt.figure(figsize=(width, width))
    # sort heatmap
    index = sorted(list(set([k2 for k, v in heatmap.items() for k2, v2 in v.items()])), key=lambda item: str(10 + len(str(item))) + str(item))
    cols = sorted([k for k in heatmap.keys()], key=lambda item: str(10 + len(str(item))) + str(item))
    # Change the labels to text
    labels = {0: "∅", 35:"KD", 38:"SD", 42:"HH", 47:"TT", 49:"CY+RD"}
    index = [labels[i] if i in labels else i for i in index]
    cols = [labels[i] if i in labels else i for i in cols]
    heatmap = {labels[k] if k in labels else k : {labels[k2] if k2 in labels else k2 : v2 for k2, v2 in v.items()} for k, v in heatmap.items()}
    # Build the heatmap
    df = pd.DataFrame(heatmap, index=index, columns=cols)
    df = df.fillna(0)

    # plot the heatmap
    # sns.heatmap(df, annot=False)
    plt.ion()
    plt.title(title)
    plt.pcolor(df)
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns, rotation=90)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()
    # plt.grid(color="grey", linestyle="--", linewidth=1)
    plt.draw()
    # plt.show()
    if text:
        for y in range(df.shape[0]):
            for x in range(df.shape[1]):
                plt.text(x + 0.5, y + 0.5, str(df.__array__()[y, x]), horizontalalignment="center", verticalalignment="center")

    if saveFigure is not None:
        plt.savefig(saveFigure)
    plt.close()


def plotActivation(predictions, Y, sparsePredictions, trackI=0, labels=config.LABELS_5, limit=100, **kwargs):
    plt.figure()
    plt.plot(predictions[trackI][:limit] + range(len(labels)))  # range(len(predictions[i].T))
    plt.yticks(range(len(labels)), labels)

    # for labelI, label in enumerate(labels):
    #     gt = Y[trackI][label]
    #     est = sparsePredictions[trackI][label]
    #     matches = mir_eval.util.match_events(gt, est, 0.05)
    #     estMatch = set(j for i, j in matches)
    #     gtMatch = set(i for i, j in matches)
    #     TP = [np.mean((gt[i], est[j])) for i, j in matches]
    #     FP = [t for i, t in enumerate(est) if i not in estMatch]
    #     FN = [t for i, t in enumerate(gt) if i not in gtMatch]
    #     for points, color in zip([TP, FP, FN], ["tab:green", "tab:red", "tab:purple"]):
    #         plt.plot(
    #             np.array(points) * 100,  # x
    #             np.array([predictions[trackI][min(int(np.round(p * 100)), len(predictions[trackI]))][labelI] for p in points]) + labelI,  # y
    #             "o",
    #             color=color,
    #         )
