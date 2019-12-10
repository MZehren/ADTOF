"""
Config file keeping folder name
"""
import os

# Folder containging .ogg files
AUDIO = "audio"
# Midi converted from PS charts
MIDI_CONVERTED = "midi_converted"
# Offset needed for the files
OD_OFFSET = "od_offset"
# Algo to eval
THREE_CLASS_EVAL = ["RV-CRNN_3"]


def getFilesInFolder(*path):
    return [os.path.join(*path, f) for f in os.listdir(os.path.join(*path)) if os.path.isfile(os.path.join(*path, f))]


def getFileBasename(path):
    return os.path.splitext(os.path.basename(path))[0]


# Maps Phase shift/rock band expert difficulty to std midi
# For more documentation on the MIDI specifications for PhaseShift or RockBand, check http://docs.c3universe.com/rbndocs/index.php?title=Drum_Authoring
# TODO: handle "disco flip" event
EXPERT_MIDI = {
    95: 35,
    96: 35,
    97: 38,
    98: {
        "modifier": 110,
        True: 45,
        False: 46
    },
    99: {
        "modifier": 111,
        True: 43,
        False: 57
    },
    100: {
        "modifier": 112,
        True: 41,
        False: 49
    }
}

# Maps PS/RB animation to the notes. The animation seems to display a better representation of the real notes
# on the official charts released. Not available on all charts
ANIMATIONS_MIDI = {
    51: 41,
    50: 41,
    49: 43,
    48: 43,
    47: 45,
    46: 45,
    42: 51,
    41: 57,
    40: 49,
    39: 57,
    38: 57,
    37: 49,
    36: 49,
    35: 49,
    34: 49,
    32: 60,  # Percussion w/ RH
    31: {
        "modifier": 25,
        True: 46,
        False: 42
    },
    30: {
        "modifier": 25,
        True: 46,
        False: 42
    },
    27: 40,
    26: 40,
    24: 36,
    28: 40,
    29: 40,
    43: 51,
    44: 57,
    45: 57
}
# Maps all the midi classes to more general consistant ones
# ie.: converts all tom-tom to a low tom, converts all hi-hat to open hi-hat 
MIDI_REDUCED = {
    35: 36,
    36: 36,
    37: 40,
    38: 40,
    40: 40,
    41: 41,
    43: 41,
    45: 41,
    47: 41,
    48: 41,
    50: 41,
    42: 46,
    44: 46,
    46: 46,
    49: 49,
    51: 49,
    52: 49,
    53: 49,
    55: 49,
    57: 49,
    59: 49,
    60: 0  # Don't remap the "percussion" as it is inconsistant 
}

# MDBS text event to midi pitches
MDBS_MIDI = {"KD": 36, "SD": 40, "HH": 46, "TT": 41, "CY": 49, "OT": 37}

# RBMA event to midi pitches
RBMA_MIDI = {0: 36, 1: 40, 2: 46}
