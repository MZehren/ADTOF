import numpy as np
import mir_eval
from collections import defaultdict


def runEvaluation(groundTruths, estimations, window=0.05, removeStart=True, classes=[35, 38, 42]):
    """
    """
    assert len(groundTruths) == len(estimations)

    meanResults = {pitch: defaultdict(list) for pitch in classes}
    sumResults = {pitch: defaultdict(int) for pitch in classes}
    for groundTruth, estimation in zip(groundTruths, estimations):
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
            f, p, r = getF(tp, fp, fn)
            meanResults[pitch]["F"].append(f)
            meanResults[pitch]["P"].append(p)
            meanResults[pitch]["R"].append(r)
            sumResults[pitch]["TP"] += tp
            sumResults[pitch]["FP"] += fp
            sumResults[pitch]["FN"] += fn

    result = {}
    for F in ["F", "P", "R"]:
        result["mean " + str(F)] = np.mean([meanResults[pitch][F] for pitch in classes])
    for pitch in classes:
        for F in ["F", "P", "R"]:
            result["mean " + F + " " + str(pitch)] = np.mean(meanResults[pitch][F])

    return result


def getF(tp, fp, fn):
    """
    """
    if tp == 0:
        return 0, 0, 0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f = 2 * tp / (2 * tp + fp + fn)
    return f, p, r
