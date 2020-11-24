"""
Config file keeping folder name
"""
import logging
import os
import numpy as np
from typing import List, Dict

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
    if not isinstance(pitches, list):
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
        if isinstance(mapped, dict):
            mapped = mapped[mapped["modifier"] in pitches]

        result[pitch] = mapped

    return result


# Maps Phase shift/rock band expert difficulty to std midi
# For more documentation on the MIDI specifications for PhaseShift or RockBand, check http://docs.c3universe.com/rbndocs/index.php?title=Drum_Authoring
# TODO: handle "disco flip" event
# TODO: DRUM_ROLLS = 126
# TODO: CYMBAL_SWELL = 127
EXPERT_MIDI = {
    95: 35,
    96: 35,
    97: 38,
    98: {"modifier": 110, True: 45, False: 42},
    99: {"modifier": 111, True: 43, False: 57},
    100: {"modifier": 112, True: 41, False: 49},
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
    # 25:Hi-Hat pedal up (hat open) w/LF. The hat will stay open for the duration of the note. The default is pedal down (hat closed).
    31: {"modifier": 25, True: 46, False: 42},  # Hi-Hat hit w/RH
    30: {"modifier": 25, True: 46, False: 42},  # Hi-Hat hit w/LH
    27: 38,  # Snare hit w/RH
    26: 38,  # Snare hit w/LH
    24: 35,  # Kick hit w/RF
    28: 38,  # A soft snare hit with the left hand
    29: 38,  # A soft snare hit with the right hand
    43: 51,  # A ride hit with the left hand
    44: 57,  # A hit on crash 2 with the left hand
    45: 57,  # A soft hit on crash 2 with the left hand
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
    37: 38,  # SD
    38: 38,
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
MDBS_MIDI = {"KD": 35, "SD": 38, "HH": 42, "TT": 47, "CY": 49, "OT": 37}

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

ENST_MIDI = {
    "bd": 35,  # Bass drum
    # "sweep": None,  # Brush sweep sticks Sticks hit together
    "sd": 38,  # Snare drum
    "rs": 38,  # Rim shot
    "cs": 75,  # Cross stick
    "chh": 42,  # Hi-hat (closed)
    "ohh": 46,  # Hi-hat (open)
    # "cb": None,  # Cowbell
    "c": 49,  # Other cymbals
    "lmt": 43,  # Low-mid tom
    "mt": 45,  # Mid tom
    # "mtr": None,  # Mid tom, hit on the rim
    "lt": 41,  # Low tom
    # "ltr": None,  # Low tom, hit on the rim
    "lft": 41,  # Lowest tom
    "rc": 51,  # Ride cymbal
    "ch": 52,  # Chinese ride cymbal
    "cr": 49,  # Crash cymbal
    "spl": 55,  # Splash cymbal
}
# Labels and class weights for the 5 output of the neural network
LABELS_5 = [35, 38, 47, 42, 49]
LABELS_3 = [35, 38, 42]
# measure frequency {0: 5.843319324520516, 1: 7.270538125118844, 2: 50.45626814462919, 3: 3.5409710967670245, 4: 24.28284008637114}
# Vogl weights Bass drum (1.0), snare drum (4.0), and hi-hat (1.5)
WEIGHTS_5 = np.array([10.780001453213364, 13.531086684241876, 34.13723052423422, 11.44276962353584, 17.6755104053326])

