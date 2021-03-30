import numpy as np
import mir_eval
from collections import defaultdict
from adtof import config
import logging


def runEvaluation(groundTruths, estimations, paths=[], distanceThreshold=0.05, removeStart=True, classes=config.LABELS_5):
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
    removeStart : bool, optional
        If we discard the estimations before the first ground truth onset in each track. 
        This is to prevent false positives due to "count-in" from charts which are not annotated, by default True
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
        annotationStart = min([v[0] for k, v in groundTruth.items() if len(v)], default=5)
        for pitch in classes:
            if removeStart and pitch in estimation:
                estimation[pitch] = [e for e in estimation[pitch] if e > annotationStart]  # TODO: can make faster

            y_truth = np.array(groundTruth[pitch]) if pitch in groundTruth else np.array([])
            y_pred = np.array(estimation[pitch]) if pitch in estimation else np.array([])

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
        if trackF < 0.1:
            logging.debug("Alert, track performed badly. Score: %s, Track:%s", str(trackF), paths[i] if i in paths else i)

    result = {}

    # score for all classes
    for F in ["F", "P", "R"]:
        result["mean " + str(F) + " " + "all"] = np.mean([meanResults[pitch][F] for pitch in classes])
    f, p, r = getF1(sumResults["all"]["TP"], sumResults["all"]["FP"], sumResults["all"]["FN"])
    result["sum F all"] = f
    result["sum P all"] = p
    result["sum R all"] = r

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
    false positives and false negatives. Based on Madmom implementatio which is different from mir_eval

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

