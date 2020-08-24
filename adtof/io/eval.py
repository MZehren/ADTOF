import numpy as np
import mir_eval
from collections import defaultdict
from adtof import config
import logging


def runEvaluation(groundTruths, estimations, paths=[], window=0.05, removeStart=True, classes=config.LABELS_5):
    """
    TODO
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

            matches = [(y_truth[i], y_pred[j]) for i, j in mir_eval.util.match_events(y_truth, y_pred, window)]
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
        if trackF < 0.05:
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
    false positives and false negatives.

    Args:
        tp (int): true positives
        fp (int): false positives
        fn (int): false negatives

    Returns:
        (int, int, int): f, p, r
    """
    if tp == 0:
        return 0, 0, 0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    # f = 2 * tp / (2 * tp + fp + fn)
    f = 2 * (p * r) / (p + r)
    return f, p, r
