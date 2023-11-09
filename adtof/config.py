"""
Config file to store constants and utility functions
"""
import logging
import os
import numpy as np
from typing import List, Dict
import glob
import pandas as pd

AUDIO = "audio/audio"  # Folder containging .ogg files
PROCESSED_AUDIO = "audio/processed"  # Folder containing .npy files
FEATURES = "audio/features"  # Folder containging features extracted for the neural network
RAW_MIDI = "annotations/raw_midi"  # Original midi in PhaseShift format
CONVERTED_MIDI = "annotations/converted_midi"  # Converted midi in standard midi format with all class when available (need desambiguation)
CONVERTED_DRUM = "annotations/converted_drum"  # onsets of the converted annotations with all class when available (need desambiguation)
ALIGNED_DRUM = "annotations/aligned_drum"  # aligned onsets of the converted annotations with all class when available (need desambiguation)
ALIGNED_BEATS = "annotations/aligned_beats"  # aligned beats of the converted annotations
ALIGNED_MIDI = "annotations/aligned_midi"  # aligned midi of the converted annotations for debug purposes
RV_ESTIMATIONS = "estimations/RV_CRNN8"  # Richard Vogl's CRNN8 estimations
BEATS_ESTIMATIONS = "estimations/beats"  # Madmom's beat estimations
BEATS_ACTIVATION = "estimations/beats_activation"  # Madmom's beat DNN output
MANUAL_SUBSTRACTION = "annotations/manual_substraction"  # Files to remove after manual check

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

cwd = os.path.abspath(os.path.dirname(__file__))
CHECKPOINT_DIR = os.path.join(cwd, "models")


def checkAllPathsExist(*outputs):
    """
    Check if all the paths exist and generate the tree of folders if they doesn't
    (see checkPathExists)
    """
    allPathsExist = True
    for output in outputs:
        if not checkPathExists(output):
            allPathsExist = False
    return allPathsExist


def plot(dict, title, ylim=True, sort=False, ylabel="F-measure", text=True, width=6.8, height=6.8 / 5, ax=None, fill=True, zorder=3, linestyle="-"):
    """
    Plot the dict as a barchart
    """
    # Load the dictionary into pandas
    df = pd.DataFrame(dict)
    if sort:
        df = df.sort_values("", ascending=False)

    # Specify the size for the paper
    dfAx = df.plot.bar(edgecolor="black", legend=False, figsize=(width, height), ax=ax, fill=fill, zorder=zorder, linestyle=linestyle)

    # add the background lines and labels
    dfAx.set_title("Test on " + title)
    dfAx.grid(axis="y", linestyle="--", zorder=0)
    if ylim:
        dfAx.set_ylim(0, 1)
    dfAx.set_ylabel(ylabel)
    dfAx.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    dfAx.tick_params(axis="x", labelrotation=0)

    # Add the text on top of the bars
    if text:
        for p in dfAx.patches:
            sizeLimit = 0.35
            dfAx.text(
                p.get_x() + p.get_width() / 2,
                p.get_height() + 0.03 if p.get_height() < sizeLimit else p.get_height() - 0.03,
                "%.2f" % round(p.get_height(), 2),
                rotation="vertical",
                horizontalalignment="center",
                verticalalignment="bottom" if p.get_height() < sizeLimit else "top",
                color="black" if p.get_height() < sizeLimit else "w",
            )


def checkPathExists(path):
    """
    return if the path exists and generate the tree of folder if they doesn't
    """
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return os.path.exists(path)


def getFilesInFolder(path):
    """
    Get the files in the folder, or the file pointed by the path
    """
    result = glob.glob(path)
    result.sort()
    return np.array(result)
    # if os.path.exists(path) == False:  # if it doesn't exist
    #     return np.array([])
    # elif os.path.isfile(path):  # if it's a file
    #     return np.array([path])
    # else:  # if it's a folder
    #     result = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    #     result = [path for path in result if os.path.splitext(os.path.basename(path))[1] in allowedExtension]
    #     result.sort()
    #     return np.array(result)


def getIntersectionOfPaths(*listOfList):
    """
    Return the list of paths A and B where file exist in both directories (without taking the extension into account)
    """
    listOfList = [np.array(L) for L in listOfList]  # Change list to np array
    listOfSet = [set([getFileBasename(item) for item in L]) for L in listOfList]  # Save the basename of the file in a set for lookup
    listOfList = [[item for item in L if all([getFileBasename(item) in S for S in listOfSet])] for L in listOfList]  # get the item which are in all the lookup set
    return listOfList


def getFileBasename(path):
    """
    Return the name of the file without path or extension
    """
    basename = os.path.splitext(os.path.basename(path))[0]
    if basename[-6:] == ".drums":
        return basename[:-6]
    else:
        return basename


def getBand(path):
    """
    return the name of the band from the path to the file
    """
    basename = getFileBasename(path)
    return basename.split(" - ")[0].lower()


def remapPitches(pitches, mappings, removeIfUnknown=True):
    """
    Map pitches to a value from a mapping. and return only the converted pitches

    All the pitches from the same timestamp have to be converted at once because of modifiers pitches in the mappings
    (ie: 98 = 45 if 110 in pitches else 46)

    pitches(int or list[int]): pitches to be converted with the mapping. If only one pitch is send instead of an iterable,
    only one pitch is returned (or None)
    mappings(dict or list[dict]): a (list of) mapping from one pitch to another. See config.EXPERT_MIDI or config.MIDI_REDUCED_3 for examples of mappings.
    Use a list (i.e.[EXPERT_MIDI, MIDI_REDUCED_3]) to chain them.
    removeIfUnknown(boolean): Specifies if any pitch not known in the mappings is removed or kept as it is.
    """
    if not isinstance(mappings, list):
        mappings = [mappings]

    if not isinstance(pitches, list):
        pitches = [pitches]

    for mapping in mappings:
        pitchRemap = getPitchesRemap(pitches, mapping)
        pitches = [pitchRemap[pitch] if pitch in pitchRemap else pitch for pitch in pitches if pitch in pitchRemap or not removeIfUnknown]
        # Handle cases where a pitch is mapped to a list of pitches
        pitches = np.array(pitches).flatten()

    return pitches


def getPitchesRemap(pitches: List[int], mapping: Dict[int, int]):
    """
    Return a dictionnary of mapping from input pitches to target pitches

    All the pitches from the same timestamp have to be converted at once because of modifiers pitches in the mappings
    (ie: 98 = 45 if 110 in pitches else 46)
    """
    result = {}
    setPitches = set(pitches)
    for pitch in pitches:
        if pitch not in mapping or mapping[pitch] is None:
            continue

        mapped = mapping[pitch]
        if isinstance(mapped, dict):  # If the target pitch is conditionned by other pitches
            conditionFound = [key for key in mapped.keys() if key in setPitches]
            if len(conditionFound):
                mapped = mapped[conditionFound[0]]
            else:
                mapped = mapped["default"]

        result[pitch] = mapped

    return result


def update(A, B):
    """
    Update the dictionary A with the dictionary B
    """
    for k, v in B.items():
        if k not in A:
            A[k] = v
    return A


# Labels and class weights for the 5 output of the neural network
LABELS_5 = [35, 38, 47, 42, 49]
LABELS_5TXT = ["BD", "SD", "TT", "HH", "CY+RD"]
LABELS_3 = [35, 38, 42]

# Vogl weights Bass drum (1.0), snare drum (4.0), and hi-hat (1.5)
# Our set of weights computed with the approach from https://markcartwright.com/files/cartwright2018increasing.pdf section 3.4.1 Task weights
# Compute the inverse estimated entropy of each label activity distribution
WEIGHTS_5 = np.array([1.0780001453213364, 1.3531086684241876, 3.413723052423422, 1.144276962353584, 1.76755104053326])
