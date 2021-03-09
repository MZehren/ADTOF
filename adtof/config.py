"""
Config file keeping folder name
"""
import logging
import os
import numpy as np
from typing import List, Dict, Iterable

AUDIO = "audio/audio"  # Folder containging .ogg files
PROCESSED_AUDIO = "audio/processed"  # Folder containing .npy files
FEATURES = "audio/features"  # Folder containging features extracted for the neural network
RAW_MIDI = "annotations/raw_midi"  # Original midi in PhaseShift format
CONVERTED_MIDI = "annotations/converted_midi"  # Converted midi in standard midi format with all class when available (need desambiguation)
ALIGNED_DRUM = "annotations/aligned_drum"  # aligned onsets of the converted annotations with all class when available (need desambiguation)
ALIGNED_BEATS = "annotations/aligned_beats"  # aligned beats of the converted annotations
ALIGNED_MIDI = "annotations/aligned_midi"  # aligned midi of the converted annotations for debug purposes
RV_ESTIMATIONS = "estimations/RV_CRNN8"  # Richard Vogl's CRNN8 estimations
BEATS_ESTIMATIONS = "estimations/beats"  # Madmom's beat estimations
BEATS_ACTIVATION = "estimations/beats_activation"  # Madmom's beat DNN output
MANUAL_SUBSTRACTION = "annotations/manual_substraction"  # Files to remove after manual check
SPLIT = "split"

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

cwd = os.path.abspath(os.path.dirname(__file__))
CHECKPOINT_DIR = os.path.join(cwd, "..", "models")


def getFilesInFolder(*path):
    """
    Get the files in the folder, or the file pointed by the path
    """
    if os.path.exists(os.path.join(*path)) == False:  # if it doesn't exist
        return np.array([])
    elif os.path.isfile(os.path.join(*path)):  # if it's a file
        return np.array([os.path.join(*path)])
    else:  # if it's a folder
        result = [os.path.join(*path, f) for f in os.listdir(os.path.join(*path)) if os.path.isfile(os.path.join(*path, f))]
        result.sort()
        return np.array(result)


def getIntersectionOfPaths(A, B):
    """
    Return the list of paths A and B where file exist in both directories (without taking the extension into account)
    """
    A = np.array(A)
    B = np.array(B)
    ASet = set([getFileBasename(a) for a in A])
    BSet = set([getFileBasename(b) for b in B])
    A = A[[getFileBasename(a) in BSet for a in A]]
    B = B[[getFileBasename(b) in ASet for b in B]]
    return A, B


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
    return basename.split(" - ")[0]


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

    returnIterable = True
    if not isinstance(pitches, Iterable):
        pitches = [pitches]
        returnIterable = False

    for mapping in mappings:
        pitchRemap = getPitchesRemap(pitches, mapping)
        pitches = [pitchRemap[pitch] if pitch in pitchRemap else pitch for pitch in pitches if pitch in pitchRemap or not removeIfUnknown]

    if returnIterable:
        return pitches
    else:
        return pitches[0] if len(pitches) else None


def getPitchesRemap(pitches: List[int], mapping: Dict[int, int]):
    """
    Return a dictionnary of mapping from input pitches to target pitches

    All the pitches from the same timestamp have to be converted at once because of modifiers pitches in the mappings 
    (ie: 98 = 45 if 110 in pitches else 46)
    """
    result = {}
    for pitch in pitches:
        if pitch not in mapping or mapping[pitch] is None:
            continue

        mapped = mapping[pitch]
        if isinstance(mapped, dict):  # If the target pitch is conditionned by other pitches
            conditionFound = [key for key in mapped.keys() if key in pitches]
            if len(conditionFound):
                mapped = mapped[conditionFound[0]]
            else:
                mapped = mapped["default"]

        result[pitch] = mapped

    return result


# Maps Phase shift/rock band expert difficulty to std midi
# For more documentation on the MIDI specifications for PhaseShift or RockBand, check http://docs.c3universe.com/rbndocs/index.php?title=Drum_Authoring
# TODO: DRUM_ROLLS = 126
# TODO: CYMBAL_SWELL = 127
EXPERT_MIDI = {
    95: 35,  # Kick note, orange
    96: 35,  # Kick note, orange
    97: {"disco": 42, "default": 38},  # Snare, red
    98: {110: 45, "disco": 38, "default": 42},  # hi-hat / high tom, yellow
    99: {111: 43, "default": 57},  # Open Hi-hat / ride / cowbell / Medium tom, blue
    100: {112: 41, "default": 49},  # Crash / low tom, green
}

# Test of using the expert annotation augmented with animation checks
# {"expert": {97: 38, 98: 47}, "animation": {27: 38, 26: 38}, "result": {97: 38}},  # Snare flam
# {"expert": {98:42, 100:49}, "animation": {49}, "result": {98: 42}},  # Double crashs
EXPERT_ANIMATION_MIDI = {
    95: 35,  # Kick note, orange
    96: 35,  # Kick note, orange
    97: {"disco": 42, "default": 38},  # Snare, red
    98: {110: 45, "disco": 38, "default": 42},  # hi-hat / high tom, yellow
    99: {111: 43, "default": 57},  # Open Hi-hat / ride / cowbell / Medium tom, blue
    100: {112: 41, "default": 49},  # Crash / low tom, green
}


# Maps PS/RB animation pitches to the standard midi pitches. The animation seems to contain a better representation of the real notes
# Not available on all charts
ANIMATIONS_MIDI = {
    51: 41,  # Floor Tom hit w/RH
    50: 41,  # Floor Tom hit w/LH
    49: 43,  # Tom2 hit w/RH
    48: 43,  # Tom2 hit w/LH
    47: 45,  # Tom1 hit w/RH
    46: 45,  # Tom1 hit w/LH
    45: 57,  # A soft hit on crash 2 with the left hand
    44: 57,  # A hit on crash 2 with the left hand
    43: 51,  # A ride hit with the left hand
    42: 51,  # Ride Cym hit w/RH
    41: 57,  # Crash2 Choke (hit w/RH, choke w/LH)
    40: 49,  # Crash1 Choke (hit w/RH, choke w/LH)
    39: 57,  # Crash2 (near Ride Cym) soft hit w/RH
    38: 57,  # Crash2 hard hit w/RH
    37: 49,  # Crash1 (near Hi-Hat) soft hit w/RH
    36: 49,  # Crash1 hard hit w/RH
    35: 49,  # Crash1 soft hit w/LH
    34: 49,  # Crash1 hard hit w/LH
    32: 60,  # Percussion w/ RH
    31: {25: 46, "default": 42,},  # Hi-Hat hit w/RH # 25:Hi-Hat pedal up (hat open for the duration of the note) w/LF.
    30: {25: 46, "default": 42},  # Hi-Hat hit w/LH
    29: 38,  # A soft snare hit with the right hand
    28: 38,  # A soft snare hit with the left hand
    27: 38,  # Snare hit w/RH
    26: 38,  # Snare hit w/LH
    24: 35,  # Kick hit w/RF
}
# Maps all the standard midi pitches to more general consistant ones
# ie.: converts all tom-tom to the 47 tom, converts all hi-hat to 42 hi-hat
MIDI_REDUCED_3 = {
    35: 35,  # BD
    36: 35,
    37: 38,  # SD
    38: 38,
    40: 38,
    # 41: None,
    # 43: None,
    # 45: None,
    # 47: None,
    # 48: None,
    # 50: None,
    42: 42,  # HH
    44: 42,
    46: 42,
    # 49: None,
    # 51: None,
    # 52: None,
    # 53: None,
    # 55: None,
    # 57: None,
    # 59: None,
    # 60: None,  # Don't remap the "percussion" as it is inconsistant
}

MIDI_REDUCED_5 = {
    35: 35,  # BD
    36: 35,
    # 37: None,
    38: 38,  # SD
    40: 38,
    41: 47,  # TT
    43: 47,
    45: 47,
    47: 47,
    48: 47,
    50: 47,
    42: 42,  # HH
    44: 42,
    46: 42,
    49: 49,  # CY
    51: 49,
    52: 49,
    53: 49,
    55: 49,
    57: 49,
    59: 49,
    # 60: None,
}

MIDI_REDUCED_6 = {
    35: 35,
    36: 35,
    37: 38,
    38: 38,
    40: 38,
    41: 47,
    43: 47,
    45: 47,
    47: 47,
    48: 47,
    50: 47,
    42: 46,
    44: 46,
    46: 46,
    49: 49,
    51: 51,
    52: 49,
    53: 51,
    55: 49,
    57: 49,
    59: 49,
    # 60: None,
}

MIDI_REDUCED_8 = {
    35: 35,
    36: 35,
    37: 38,
    38: 38,
    40: 38,
    41: 41,
    43: 41,
    45: 45,
    47: 45,
    48: 45,
    50: 45,
    42: 42,
    44: 42,
    46: 46,
    49: 49,
    51: 51,
    52: 49,
    53: 51,
    55: 49,
    57: 49,
    59: 49,
    # 60: None,
}

# MDBS text event to midi pitches
# MDBS_MIDI = {"KD": 35, "SD": 38, "HH": 42, "TT": 47, "CY": 49, "OT": 37}
# From Vogl
MDBS_MIDI = {
    "KD": 35,
    "SD": 38,
    "SDB": 38,
    "SDD": 38,
    "SDF": 38,
    "SDG": 38,
    "SDNS": 38,
    "CHH": 42,
    "OHH": 46,
    "PHH": 44,
    "HIT": 50,
    "MHT": 48,
    "HFT": 43,
    "LFT": 41,
    "RDC": 51,
    "RDB": 53,
    "CRC": 49,
    "CHC": 52,
    "SPC": 55,
    "SST": 37,
    "TMB": 54,
}


# RBMA event to midi pitches
RBMA_MIDI_3 = {0: 35, 1: 38, 2: 42}
RBMA_MIDI_8 = {
    0: 35,  # BD
    1: 38,  # SD
    2: 47,  # TT  (lft)
    3: 42,  # HH
    4: 49,  # CY
    5: 51,  # RD
    6: 53,  # ride bell / bells / etc
    7: 75,  # claves
}
# FROM VOGL
RBMA_MIDI = {
    0: 36,  # bass drum
    1: 38,  # snare drum
    2: 42,  # closed hi-hat
    3: 46,  # open hi-hat
    4: 44,  # pedal hi-hat
    5: 56,  # cowbell
    6: 53,  # ride bell
    7: 41,  # low floor tom
    9: 43,  # high floor tom
    10: 45,  # low tom
    11: 47,  # low-mid tom
    12: 48,  # high-mid tom
    13: 50,  # high tom
    14: 37,  # side stick
    15: 39,  # hand clap
    16: 51,  # ride cymbal
    17: 49,  # crash cymbal
    18: 55,  # splash cymbal
    19: 52,  # chinese cymbal
    20: 70,  # shaker, maracas
    21: 54,  # tambourine
    22: 75,  # claves, stick click
    23: 81,  # high bells / triangle
}


# FROM VOGL
ENST_MIDI = {
    "bd": 35,
    "cs": 37,
    "rs": 38,
    "sd": 38,
    "sd-": 38,
    "lft": 41,
    "chh": 42,
    "lt": 45,
    "ohh": 46,
    "lmt": 47,
    "mt": 48,
    "cr": 49,
    "c1": 49,
    "cr1": 49,
    "cr5": 49,
    "rc": 51,
    "rc1": 51,
    "rc3": 51,
    "ch": 52,
    "ch1": 52,
    "ch5": 52,
    "spl": 55,
    "spl2": 55,
    "cb": 56,
    "cr2": 57,
    "c": 57,
    "c4": 57,
    "rc2": 59,
    "rc4": 59,
    "sticks": 75,
    #'mtr': -1,
    #'sweep': -1,
    #'ltr': -1,
}


# Labels and class weights for the 5 output of the neural network
LABELS_5 = [35, 38, 47, 42, 49]
LABELS_3 = [35, 38, 42]
# measure frequency {0: 5.843319324520516, 1: 7.270538125118844, 2: 50.45626814462919, 3: 3.5409710967670245, 4: 24.28284008637114}
# Vogl weights Bass drum (1.0), snare drum (4.0), and hi-hat (1.5)
WEIGHTS_5 = np.array([10.780001453213364, 13.531086684241876, 34.13723052423422, 11.44276962353584, 17.6755104053326])

