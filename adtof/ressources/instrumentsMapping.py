from adtof.config import update

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

# Maps PS/RB animation pitches to the standard midi pitches. The animations contain a better representation of the real notes
# Not available on all charts
ANIMATIONS_MIDI = {
    51: 41,  # Floor Tom hit w/RH
    50: 41,  # Floor Tom hit w/LH
    49: 45,  # Tom2 hit w/RH
    48: 45,  # Tom2 hit w/LH
    47: 47,  # Tom1 hit w/RH
    46: 47,  # Tom1 hit w/LH
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
    31: {
        25: 46,
        "default": 42,
    },  # Hi-Hat hit w/RH # 25:Hi-Hat pedal up (hat open for the duration of the note) w/LF.
    30: {25: 46, "default": 42},  # Hi-Hat hit w/LH
    29: 38,  # A soft snare hit with the right hand
    28: 38,  # A soft snare hit with the left hand
    27: 38,  # Snare hit w/RH
    26: 38,  # Snare hit w/LH
    # 25: 44,  # HH pedal up. Invert this event
    24: 35,  # Kick hit w/RF
}

# With the animation there are also information on velocity (not available for all charts)
DEFAULT_VELOCITY = 96
SOFT_VELOCITY = 49
ANIMATIONS_VELOCITY = {
    51: DEFAULT_VELOCITY,  # Floor Tom hit w/RH
    50: DEFAULT_VELOCITY,  # Floor Tom hit w/LH
    49: DEFAULT_VELOCITY,  # Tom2 hit w/RH
    48: DEFAULT_VELOCITY,  # Tom2 hit w/LH
    47: DEFAULT_VELOCITY,  # Tom1 hit w/RH
    46: DEFAULT_VELOCITY,  # Tom1 hit w/LH
    45: SOFT_VELOCITY,  # A soft hit on crash 2 with the left hand
    44: DEFAULT_VELOCITY,  # A hit on crash 2 with the left hand
    43: DEFAULT_VELOCITY,  # A ride hit with the left hand
    42: DEFAULT_VELOCITY,  # Ride Cym hit w/RH
    41: DEFAULT_VELOCITY,  # Crash2 Choke (hit w/RH, choke w/LH)
    40: DEFAULT_VELOCITY,  # Crash1 Choke (hit w/RH, choke w/LH)
    39: SOFT_VELOCITY,  # Crash2 (near Ride Cym) soft hit w/RH
    38: DEFAULT_VELOCITY,  # Crash2 hard hit w/RH
    37: SOFT_VELOCITY,  # Crash1 (near Hi-Hat) soft hit w/RH
    36: DEFAULT_VELOCITY,  # Crash1 hard hit w/RH
    35: SOFT_VELOCITY,  # Crash1 soft hit w/LH
    34: DEFAULT_VELOCITY,  # Crash1 hard hit w/LH
    32: DEFAULT_VELOCITY,  # Percussion w/ RH
    31: DEFAULT_VELOCITY,  # Hi-Hat hit w/RH # 25:Hi-Hat pedal up (hat open for the duration of the note) w/LF.
    30: DEFAULT_VELOCITY,  # Hi-Hat hit w/LH
    29: SOFT_VELOCITY,  # A soft snare hit with the right hand
    28: SOFT_VELOCITY,  # A soft snare hit with the left hand
    27: DEFAULT_VELOCITY,  # Snare hit w/RH
    26: DEFAULT_VELOCITY,  # Snare hit w/LH
    24: DEFAULT_VELOCITY,  # Kick hit w/RF
}

# Maps all the general midi pitches to consistant groupes (see https://en.wikipedia.org/wiki/General_MIDI#Percussion)
# ie.: converts all tom-tom to the 47 tom, converts all hi-hat to 42 hi-hat
MIDI_REDUCED_3 = {  # Std 3 classes ADT, we ignore the rest
    35: 35,  # Acoustic Bass Drum
    36: 35,  # Electric Bass Drum
    37: 38,  # Side Stick
    38: 38,  # Acoustic Snare
    39: 38,  # Hand Clap
    40: 38,  # Electric Snare
    42: 42,  # Closed Hi-hat
    44: 42,  # Pedal Hi-hat
    46: 42,  # Open Hi-hat
}
MIDI_REDUCED_5 = update(  # Adding TT and CY + RD
    {
        41: 47,  # Low Floor Tom
        43: 47,  # High Floor Tom
        45: 47,  # Low Tom
        47: 47,  # Low-Mid Tom
        48: 47,  # High-Mid Tom
        50: 47,  # High Tom
        49: 49,  # Crash Cymbal 1
        51: 49,  # Ride Cymbal 1
        52: 49,  # Chinese Cymbal
        53: 49,  # Ride Bell
        55: 49,  # Splash Cymbal
        57: 49,  # Crash Cymbal 2
        59: 49,  # Ride Cymbal 2
    },
    MIDI_REDUCED_3,
)
MIDI_REDUCED_6 = update(  # Splitting CY and RD
    {
        51: 51,  # Ride Cymbal 1
        53: 51,  # Ride Bell
        59: 51,  # Ride Cymbal 2
    },
    MIDI_REDUCED_5,
)
MIDI_REDUCED_7 = update(  # Splitting OH and CH
    {
        46: 46,  # Open Hi-hat
    },
    MIDI_REDUCED_6,
)
MIDI_REDUCED_8 = update(  # Splitting floor tom and rack tom
    {
        41: 41,  # Low Floor Tom
        43: 41,  # High Floor Tom
    },
    MIDI_REDUCED_7,
)
MIDI_REDUCED_9 = update(  # Adding percussion
    {
        60: 60,  # High Bongo
        # 25: None,  # Snare Roll
        # 26: None,  # Finger Snap
        # 27: None,  # High Q
        # 28: None,  # Slap
        # 29: None,  # Scratch Pull
        # 30: None,  # Scratch Push
        # 31: None,  # Sticks
        # 32: None,  # Square Click
        # 33: None,  # Metronome Bell
        # 34: None,  # Metronome Click
        # 54: None,  # Tambourine
        # 56: None,  # Cowbell
        # 58: None,  # Vibraslap
        # 69: None,  # Cabasa
        # 70: None,  # Maracas
        # 71: None,  # Short Whistle
        # 72: None,  # Long Whistle
        # 73: None,  # Short Guiro
        # 74: None,  # Long Guiro
        # 75: None,  # Claves
        # 76: None,  # High Woodblock
        # 77: None,  # Low Woodblock
        # 78: None,  # Mute Cuica
        # 79: None,  # Open Cuica
        # 80: None,  # Mute Triangle
        # 81: None,  # Open Triangle
        # 82: None,  # Shaker
        # 83: None,  # Jingle Bell
        # 84: None,  # Belltree
        # 85: None,  # Castanets
        # 86: None,  # Mute Surdo
        # 87: None,  # Open Surdo
        # 61: None,  # Low Bongo
        # 62: None,  # Mute High Conga
        # 63: None,  # Open High Conga
        # 64: None,  # Low Conga
        # 65: None,  # High Timbale
        # 66: None,  # Low Timbale
        # 67: None,  # High Agogô
        # 68: None,  # Low Agogô
    },
    MIDI_REDUCED_8,
)

# MDBS text event to midi pitches
# From https://carlsouthall.files.wordpress.com/2017/12/ismir2017mdbdrums.pdf
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
# FROM VOGL http://www.ifs.tuwien.ac.at/~vogl/dafx2018/mappings.py
RBMA_MIDI_3 = {0: 35, 1: 38, 2: 42}  # drums_3
RBMA_MIDI_8 = {  # drums_m
    0: 35,  # BD
    1: 38,  # SD
    2: 47,  # TT  (lft)
    3: 42,  # HH
    4: 49,  # CY
    5: 51,  # RD
    6: 53,  # ride bell / bells / etc
    7: 75,  # claves/sticks
}
RBMA_MIDI_18 = {  # drums_l
    0: 35,  # BD
    1: 38,  # SD
    2: 37,  # side stick
    3: 39,  # clap
    4: 41,  # TT  (lft)
    5: 45,  #     (lt)
    6: 48,  #     (hmt)
    7: 42,  # HH
    8: 44,  # pedal hh
    9: 46,  # open hh
    10: 54,  # tamborine
    11: 51,  # RD
    12: 53,  # ride bell
    13: 49,  # crash
    14: 55,  # splash
    15: 52,  # chinese
    16: 56,  # cowbell
    17: 75,  # click/sticks
}
RBMA_FULL_MIDI = {  # drums_f
    0: 35,  # "B0", "Acoustic Bass Drum"],
    1: 36,  # "C1", "Bass Drum 1"],
    2: 37,  # "C#1", "Side Stick"],
    3: 38,  # "D1", "Acoustic Snare"],
    4: 39,  # "Eb1", "Hand Clap"],
    5: 40,  # "E1", "Electric Snare"],
    6: 41,  # "F1", "Low Floor Tom"],
    7: 42,  # "F#1", "Closed Hi Hat"],
    8: 43,  # "G1", "High Floor Tom"],
    9: 44,  # "Ab1", "Pedal Hi-Hat"],
    10: 45,  # "A1", "Low Tom"],
    11: 46,  # "Bb1", "Open Hi-Hat"],
    12: 47,  # "B1", "Low-Mid Tom"],
    13: 48,  # "C2", "Hi Mid Tom"],
    14: 49,  # "C#2", "Crash Cymbal 1"],
    15: 50,  # "D2", "High Tom"],
    16: 51,  # "Eb2", "Ride Cymbal 1"],
    17: 52,  # "E2", "Chinese Cymbal"],
    18: 53,  # "F2", "Ride Bell"],
    19: 54,  # "F#2", "Tambourine"],
    20: 55,  # "G2", "Splash Cymbal"],
    21: 56,  # "Ab2", "Cowbell"],
    22: 57,  # "A2", "Crash Cymbal 2"],
    23: 58,  # "Bb2", "Vibraslap"],
    24: 59,  # "B2", "Ride Cymbal 2"],
    25: 60,  # "C3", "Hi Bongo"],
    26: 61,  # "C#3", "Low Bongo"],
    27: 62,  # "D3", "Mute Hi Conga"],
    28: 63,  # "Eb3", "Open Hi Conga"],
    29: 64,  # "E3", "Low Conga"],
    30: 65,  # "F3", "High Timbale"],
    31: 66,  # "F#3", "Low Timbale"],
    32: 67,  # "G3", "High Agogo"],
    33: 68,  # "Ab3", "Low Agogo"],
    34: 69,  # "A3", "Cabasa"],
    35: 70,  # "Bb3", "Maracas"],
    36: 71,  # "B3", "Short Whistle"],
    37: 72,  # "C4", "Long Whistle"],
    38: 73,  # "C#4", "Short Guiro"],
    39: 74,  # "D4", "Long Guiro"],
    40: 75,  # "Eb4", "Claves"],
    41: 76,  # "E4", "Hi Wood Block"],
    42: 77,  # "F4", "Low Wood Block"],
    43: 78,  # "F#4", "Mute Cuica"],
    44: 79,  # "G4", "Open Cuica"],
    45: 80,  # "Ab4", "Mute Triangle"],
    46: 81,  # "A4", "Open Triangle"],
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
    # "mtr": None,
    # "sweep": None,
    # "ltr": None,
}

# Slakh2100 to General MIDI
SLAKH2100_MIDI = {
    # From Kontakt Factory
    "funk_kit.nkm": {
        24: 39,  # clap
        25: 39,  # clap
        26: 39,  # clap
        27: 39,  # clap
        28: 39,  # clap
        29: 39,  # clap
        30: 39,  # clap
        31: 39,  # clap
        32: 39,  # clap
        33: 39,  # clap
        34: 39,  # clap
        35: 39,  # clap
        36: 36,  # Bass drum
        37: 37,  # Side stick
        38: 38,  # Snare Left
        39: 38,  # Rim Shot
        40: 40,  # Snare right
        41: 41,  # Low tom left
        42: 42,  # Closed HiHat
        43: 43,  # Low tom right
        44: 44,  # Pedal HiHat
        45: 45,  # Mid tom 1 left
        46: 46,  # Open HiHat
        47: 47,  # Mid tom 1 right
        48: 48,  # Mid tom 2 left
        49: 49,  # Crash Cymbal
        50: 50,  # Mid tom 2 right
        51: 51,  # Ride Cymbal
        52: 50,  # Hi tom left
        53: 50,  # Hi tom right
        54: 53,  # Ride bell
        55: 55,  # Splash Cymbal
        56: 56,  # Cowbell
        57: 57,  # Crash Cymbal 2
        58: 55,  # Splash Cymbal 2
        59: 59,  # Ride Cymbal 2
        60: (36, 42),  # seq: Closed HiHat + Bass drum, then Open HH ...
        61: (36, 44),
        62: (36, 46),
        63: 36,
        64: (36, 42),
        65: (36, 41),
        66: (36, 44),
        67: (36, 42),
        68: (36, 42, 47),
        69: (36, 42, 43),
        70: (36, 42),
        71: (36, 42),
    },
    # From Kontakt Factory
    "pop_kit.nkm": {
        35: 35,  # Bass drum 2
        36: 36,  # Bass drum
        37: 37,  # Side stick
        38: 38,  # Snare Left
        39: 38,  # Rim Shot
        40: 40,  # Snare right
        41: 41,  # Low tom left
        42: 42,  # Closed HiHat
        43: 43,  # Low tom right
        44: 44,  # Pedal HiHat
        45: 45,  # Mid tom 1 left
        46: 46,  # Open HiHat
        47: 47,  # Mid tom 1 right
        48: 48,  # Mid tom 2 left
        49: 49,  # Crash Cymbal
        50: 50,  # Mid tom 2 right
        51: 51,  # Ride Cymbal
        52: 50,  # Hi tom left
        53: 50,  # Hi tom right
        54: 53,  # Ride bell
        55: 55,  # Splash Cymbal
        56: 56,  # Cowbell
        57: 57,  # Crash Cymbal 2
        58: 55,  # Splash Cymbal 2
        59: 59,  # Ride Cymbal 2
        60: (36, 42),  # seq: Closed HiHat + Bass drum, then Open HH ...
        61: (36, 44),
        62: 36,
        63: 36,
        64: (36, 42),
        65: 42,
        66: (36, 44),
        67: (36, 42),
        68: (36, 42, 47),
        69: (36, 42, 43),
        70: (36, 42),
        71: 36,
    },
    # From Kontakt Factory
    "street_knowledge_kit.nkm": {
        # 35: None,# Tones
        36: 36,  # Bass Drum
        37: 38,  # Snare Drum
        38: 38,  # Snare Drum 2
        39: 39,  # Clap
        40: 40,  # Snare drum 3
        41: 82,  # Shaker
        42: 42,  # Closed Hihat
        43: 34,  # Click
        44: 42,  # Closed HiHat 2
        # 45: None,  # FX 1
        46: 42,  # Closed Hihat 3
        47: 36,  # Bass Drum 2
        48: 36,  # Bass Drum 3
        49: 80,  # Muted Triangle
        50: 40,  # Snare Drum 4
        51: 81,  # Open Triangle
        52: 39,  # Clap 2
        53: 54,  # Tambourine
        54: 82,  # Shaker
        55: 34,  # Clik 2
        56: 82,  # Shaker 2
        57: 57,  # Revese Crash
        58: 69,  # Cabasa
        59: 36,  # Bass Drum 4
        60: (36, 42),  # seq: Closed HiHat + Bass drum, then Open HH ...
        61: (36, 44),
        62: 36,
        63: 36,
        64: (36, 42),
        65: (36, 42),
        66: 36,
        67: 36,
        68: 36,
        69: 36,
        70: 36,
        # 71: None, #FX
    },
    # From: https://medias.audiofanzine.com/files/the-garage-kit-default-mapping-280614.pdf
    "garage_kit_lite.nkm": {
        22: 57,  # 	High Crash - Choke
        23: 51,  # 	Ride - Choke
        24: 57,  # 	Low Crash - Choke
        25: 52,  # 	China - Choke
        26: 55,  # 	Splash - Choke
        27: 56,  # 	Cowbell Low - Open
        28: 56,  # 	Cowbell Low - Muted
        29: 56,  # 	Cowbell Hi - Open
        30: 56,  # 	Cowbell Hi - Muted
        31: 54,  # 	Tambourine - Tap
        32: 54,  # 	Tambourine - Shake
        33: 39,  # 	Clap - Solo
        34: 39,  # 	Clap - Multi
        35: 31,  # 	Stick - Hit
        36: 36,  # 	Kick - Dampened
        37: 37,  # 	Snare - Sidestick
        38: 38,  # 	Snare - Center L/R
        39: 38,  # 	Snare - Rimshot
        40: 40,  # 	Snare - Halfway L/R
        41: 41,  # 	Tom 3 - Center L/R (double)
        42: 42,  # 	Closed Hihat - Tip L/R
        43: 43,  # 	Tom 3 - Center L/R
        44: 44,  # 	Closed Hihat - Pedal
        45: 45,  # 	Tom 2 - Center L/R
        46: 46,  # 	Open Hihat - Contro
        47: 47,  # 	Tom 1 - Center L/R
        48: 57,  # 	High Crash - Tip
        49: 57,  # 	High Crash - Edge
        50: 57,  # 	High Crash - Bell
        51: 51,  # 	Ride - Tip
        52: 51,  # 	Ride - Edge
        53: 53,  # 	Ride - Bell
        54: 49,  # 	Low Crash - Tip
        55: 49,  # 	Low Crash - Edge
        56: 49,  # 	Low Crash - Bell
        57: 52,  # 	China - Edge
        58: 52,  # 	China - Tip
        59: 55,  # 	Splash - Edge
        60: 36,  # 	Kick - Open
        61: 37,  # 	Snare - Rim Only
        62: 38,  # 	Snare Flam
        63: 38,  # 	Snare - Roll
        64: 38,  # 	Snare - Wires Off
        65: 43,  # 	Tom 3 - Rimshot (double)
        66: 42,  # 	Closed Hihat - Tight Tip L/R
        67: 43,  # 	Tom 3 - Rimshot
        68: 42,  # 	Closed Hihat - Shank L/R
        69: 45,  # 	Tom 2 - Rimshot
        70: 46,  # 	Open Hihat - Peda
        71: 41,  # 	Tom 1 - Rimshot
        72: 37,  # 	Tom 3 - Rim Only (double)
        73: 37,  # 	Tom 3 - Rim Only
        74: 37,  # 	Tom 2 - Rim Only
        75: 37,  # 	Tom 1 - Rim Only
        76: 46,  # 	Open Hihat - 1/4
        77: 46,  # 	Open Hihat - 1/2
        78: 46,  # 	Open Hihat - 3/4
        79: 46,  # 	Open Hihat - Loose
        80: 46,  # 	Open Hihat - Full
        81: 38,  # 	Snare - Center L
        # 82: None,  # 	[Empty]
        83: 38,  # 	Snare - Center R
        84: 38,  # 	Snare - Halfway L
        85: 42,  # 	Closed Hihat - Tight Tip L
        86: 38,  # 	Snare - Halfway R
        87: 42,  # 	Closed Hihat - Tight Tip R
        88: 43,  # 	Tom 3 - Center L (double)
        89: 43,  # 	Tom 3 - Center R (double)
        90: 42,  # 	Closed Hihat - Tip L
        91: 43,  # 	Tom 3 - Center L
        92: 42,  # 	Closed Hihat - Tip R
        93: 43,  # 	Tom 3 - Center R
        94: 42,  # 	Closed Hihat - Shank L
        95: 45,  # 	Tom 2 - Center L
        96: 45,  # 	Tom 2 - Center R
        97: 42,  # 	Closed Hihat - Shank R
        98: 41,  # 	Tom 1 - Center L
        # 99: None,  # 	[Empty]
        100: 41,  # 	Tom 1 - Center R
    },
    # From: https://medias.audiofanzine.com/files/the-session-kit-default-mapping-280615.pdf
    "session_kit_full.nkm": {
        22: 57,  # 	High Crash - Choke
        23: 51,  # 	Ride - Choke
        24: 57,  # 	Low Crash - Choke
        25: 52,  # 	China - Choke
        26: 55,  # 	Splash - Choke
        27: 77,  # 	Woodblock Low
        28: 77,  # 	Woodblock Low (double)
        29: 76,  # 	Woodblock Hi
        30: 76,  # 	Woodblock Hi (double)
        31: 54,  # 	Tambourine - Tap
        32: 54,  # 	Tambourine - Shake
        33: 39,  # 	Clap - Solo
        34: 39,  # 	Clap - Multi
        35: 31,  # 	Stick - Hit
        36: 36,  # 	Kick - Dampened
        37: 37,  # 	Snare - Sidestick
        38: 38,  # 	Snare - Center L/R
        39: 38,  # 	Snare - Rimshot
        40: 40,  # 	Snare - Halfway L/R
        41: 41,  # 	Tom 4 - Center L/R
        42: 42,  # 	Closed Hihat - Tip L/R
        43: 43,  # 	Tom 3 - Center L/R
        44: 44,  # 	Closed Hihat - Pedal
        45: 45,  # 	Tom 2 - Center L/R
        46: 46,  # 	Open Hihat - Contro
        47: 47,  # 	Tom 1 - Center L/R
        48: 57,  # 	High Crash - Tip
        49: 57,  # 	High Crash - Edge
        50: 57,  # 	High Crash - Bell
        51: 51,  # 	Ride - Tip
        52: 51,  # 	Ride - Edge
        53: 53,  # 	Ride - Bell
        54: 49,  # 	Low Crash - Tip
        55: 49,  # 	Low Crash - Edge
        56: 49,  # 	Low Crash - Bell
        57: 52,  # 	China - Edge
        58: 52,  # 	China - Tip
        59: 55,  # 	Splash - Edge
        60: 36,  # 	Kick - Open
        61: 37,  # 	Snare - Rim Only
        62: 38,  # 	Snare Flam
        63: 38,  # 	Snare - Roll
        64: 38,  # 	Snare - Wires Off
        65: 43,  # 	Tom 4 - Rimshot
        66: 42,  # 	Closed Hihat - Tight Tip L/R
        67: 43,  # 	Tom 3 - Rimshot
        68: 42,  # 	Closed Hihat - Shank L/R
        69: 45,  # 	Tom 2 - Rimshot
        70: 46,  # 	Open Hihat - Peda
        71: 41,  # 	Tom 1 - Rimshot
        72: 37,  # 	Tom 4 - Rim Only
        73: 37,  # 	Tom 3 - Rim Only
        74: 37,  # 	Tom 2 - Rim Only
        75: 37,  # 	Tom 1 - Rim Only
        76: 46,  # 	Open Hihat - 1/4
        77: 46,  # 	Open Hihat - 1/2
        78: 46,  # 	Open Hihat - 3/4
        79: 46,  # 	Open Hihat - Loose
        80: 46,  # 	Open Hihat - Full
        81: 38,  # 	Snare - Center L
        # 82: None,  # 	[Empty]
        83: 38,  # 	Snare - Center R
        84: 38,  # 	Snare - Halfway L
        85: 42,  # 	Closed Hihat - Tight Tip L
        86: 38,  # 	Snare - Halfway R
        87: 42,  # 	Closed Hihat - Tight Tip R
        88: 43,  # 	Tom 4 - Center L
        89: 43,  # 	Tom 4 - Center R
        90: 42,  # 	Closed Hihat - Tip L
        91: 43,  # 	Tom 3 - Center L
        92: 42,  # 	Closed Hihat - Tip R
        93: 43,  # 	Tom 3 - Center R
        94: 42,  # 	Closed Hihat - Shank L
        95: 45,  # 	Tom 2 - Center L
        96: 45,  # 	Tom 2 - Center R
        97: 42,  # 	Closed Hihat - Shank R
        98: 41,  # 	Tom 1 - Center L
        # 99: None,  # 	[Empty]
        100: 41,  # 	Tom 1 - Center R
    },
    # From: https://medias.audiofanzine.com/files/the-stadium-kit-default-mapping-280613.pdf
    "stadium_kit_full.nkm": {
        22: 57,  # 	High Crash - Choke
        23: 51,  # 	Ride - Choke
        24: 57,  # 	Low Crash - Choke
        25: 52,  # 	China - Choke
        26: 55,  # 	Splash - Choke
        27: 56,  # 	Cowbell Open
        28: 56,  # 	Cowbell - Muted
        29: 56,  # 	Cowbell Open (double)
        30: 56,  # 	Cowbell - Muted (double)
        31: 54,  # 	Tambourine - Tap
        32: 54,  # 	Tambourine - Shake
        33: 39,  # 	Clap - Solo
        34: 39,  # 	Clap - Multi
        35: 31,  # 	Stick - Hit
        36: 36,  # 	Kick - Dampened
        37: 37,  # 	Snare - Sidestick
        38: 38,  # 	Snare - Center L/R
        39: 38,  # 	Snare - Rimshot
        40: 40,  # 	Snare - Halfway L/R
        41: 41,  # 	Tom 4 - Center L/R
        42: 42,  # 	Closed Hihat - Tip L/R
        43: 43,  # 	Tom 3 - Center L/R
        44: 44,  # 	Closed Hihat - Pedal
        45: 45,  # 	Tom 2 - Center L/R
        46: 46,  # 	Open Hihat - Contro
        47: 47,  # 	Tom 1 - Center L/R
        48: 57,  # 	High Crash - Tip
        49: 57,  # 	High Crash - Edge
        50: 57,  # 	High Crash - Bell
        51: 51,  # 	Ride - Tip
        52: 51,  # 	Ride - Edge
        53: 53,  # 	Ride - Bell
        54: 49,  # 	Low Crash - Tip
        55: 49,  # 	Low Crash - Edge
        56: 49,  # 	Low Crash - Bell
        57: 52,  # 	China - Edge
        58: 52,  # 	China - Tip
        59: 55,  # 	Splash - Edge
        60: 36,  # 	Kick - Open
        61: 37,  # 	Snare - Rim Only
        62: 38,  # 	Snare Flam
        63: 38,  # 	Snare - Roll
        64: 38,  # 	Snare - Wires Off
        65: 43,  # 	Tom 4 - Rimshot
        66: 42,  # 	Closed Hihat - Tight Tip L/R
        67: 43,  # 	Tom 3 - Rimshot
        68: 42,  # 	Closed Hihat - Shank L/R
        69: 45,  # 	Tom 2 - Rimshot
        70: 46,  # 	Open Hihat - Peda
        71: 41,  # 	Tom 1 - Rimshot
        72: 37,  # 	Tom 4 - Rim Only
        73: 37,  # 	Tom 3 - Rim Only
        74: 37,  # 	Tom 2 - Rim Only
        75: 37,  # 	Tom 1 - Rim Only
        76: 46,  # 	Open Hihat - 1/5
        77: 46,  # 	Open Hihat - 1/3
        78: 46,  # 	Open Hihat - 3/5
        79: 46,  # 	Open Hihat - Loose
        80: 46,  # 	Open Hihat - Full
        81: 38,  # 	Snare - Center L
        # 82: None,  # 	[Empty]
        83: 38,  # 	Snare - Center R
        84: 38,  # 	Snare - Halfway L
        85: 42,  # 	Closed Hihat - Tight Tip L
        86: 38,  # 	Snare - Halfway R
        87: 42,  # 	Closed Hihat - Tight Tip R
        88: 43,  # 	Tom 4 - Center L
        89: 43,  # 	Tom 4 - Center R
        90: 42,  # 	Closed Hihat - Tip L
        91: 43,  # 	Tom 3 - Center L
        92: 42,  # 	Closed Hihat - Tip R
        93: 43,  # 	Tom 3 - Center R
        94: 42,  # 	Closed Hihat - Shank L
        95: 45,  # 	Tom 2 - Center L
        96: 45,  # 	Tom 2 - Center R
        97: 42,  # 	Closed Hihat - Shank R
        98: 41,  # 	Tom 1 - Center L
        # 99: None,  # 	[Empty]
        100: 41,  # 	Tom 1 - Center R
    },
    # From: https://www.native-instruments.com/fileadmin/ni_media/downloads/manuals/Abbey_Road_Modern_Drummer_Manual_English_2012_07.zip
    "ar_modern_sparkle_kit_full.nkm": {
        16: 56,  # 	Perc 4 (Cowbell) Low Open
        17: 56,  # 	Perc 4 (Cowbell) Low Muted
        19: 56,  # 	Perc 4 (Cowbell) High Open
        21: 56,  # 	Perc 4 (Cowbell) High Muted
        22: 57,  # 	Cymbal 1 (High Crash) Choke ***
        23: 51,  # 	Cymbal 3 (Ride) Choke ***
        24: 49,  # 	Cymbal 2 (Low Crash) Choke ***
        25: 52,  # 	Cymbal 4 (China) Choke ***
        26: 55,  # 	Cymbal 5 (Splash) Choke ***
        27: 55,  # 	Perc 3 (Chopper) Low
        28: 55,  # 	Perc 3 (Chopper) Mid
        29: 55,  # 	Perc 3 (Chopper) High
        31: 54,  # 	Perc 5 (Tambourine) Tap
        32: 54,  # 	Perc 5 (Tambourine) Shake
        33: 39,  # 	Perc 2 (Clap) Solo
        34: 39,  # 	Perc 2 (Clap) Multi
        35: 31,  # 	Perc 1 (Stick) Hit
        36: 36,  # 	Kick Drum Dampened
        37: 37,  # 	Snare Drum 1, 2 & 3 Sidestick
        38: 38,  # 	Snare Drum 1, 2 & 3 Center Right/Left Alternating *
        39: 38,  # 	Snare Drum 1, 2 & 3 Rimshot
        40: 38,  # 	Snare Drum 1, 2 & 3 Halfway Right/Left Alternating *
        41: 41,  # 	Tom 4 (Floor Tom) Center Right/Left Alternating *
        42: 42,  # 	Hihat Closed Tip Right/Left Alternating*
        43: 43,  # 	Tom 3 (Rack Tom Low) Center Right/Left Alternating *
        44: 44,  # 	Hihat Closed Pedal
        45: 45,  # 	Tom 2 (Rack Tom Mid) Center Right/Left Alternating *
        46: 46,  # 	Hihat Open Controller**
        47: 47,  # 	Tom 1 (Rack Tom High) Center Right/Left Alternating *
        48: 57,  # 	Cymbal 1 (High Crash) Tip
        49: 57,  # 	Cymbal 1 (High Crash) Edge
        50: 57,  # 	Cymbal 1 (High Crash) Bell
        51: 51,  # 	Cymbal 3 (Ride) Tip
        52: 51,  # 	Cymbal 3 (Ride) Edge
        53: 53,  # 	Cymbal 3 (Ride) Bell
        54: 49,  # 	Cymbal 2 (Low Crash) Tip
        55: 49,  # 	Cymbal 2 (Low Crash) Edge
        56: 49,  # 	Cymbal 2 (Low Crash) Bell
        57: 52,  # 	Cymbal 4 (China) Edge
        58: 52,  # 	Cymbal 4 (China) Tip
        59: 52,  # 	Cymbal 5 (Splash) Edge
        60: 36,  # 	Kick Drum Open
        61: 37,  # 	Snare Drum 1, 2 & 3 Rim Only
        62: 38,  # 	Snare Drum 1, 2 & 3 Flam
        63: 38,  # 	Snare Drum 1, 2 & 3 Roll
        64: 38,  # 	Snare Drum 1, 2 & 3 Wires Off
        65: 41,  # 	Tom 4 (Floor Tom) Rimshot
        66: 42,  # 	Hihat Closed Tight Tip Right/Left Alternating *
        67: 43,  # 	Tom 3 (Rack Tom Low) Rimshot
        68: 42,  # 	Hihat Closed Shank Right/Left Alternating *
        69: 45,  # 	Tom 2 (Rack Tom Mid) Rimshot
        70: 44,  # 	Hihat Open Pedal
        71: 47,  # 	Tom 1 (Rack Tom High) Rimshot
        72: 37,  # 	Tom 4 (Floor Tom) Rim Only
        73: 37,  # 	Tom 3 (Rack Tom Low) Rim Only
        74: 37,  # 	Tom 2 (Rack Tom Mid) Rim Only
        75: 37,  # 	Tom 1 (Rack Tom High) Rim Only
        76: 46,  # 	Hihat Open Quarter
        77: 46,  # 	Hihat Open Half
        78: 46,  # 	Hihat Open Three
        79: 46,  # 	Hihat Open Loose
        80: 46,  # 	Hihat Open Full
        81: 38,  # 	Snare Drum 1, 2 & 3 Center Left Hand
        82: 36,  # 	Kick Half Open
        83: 38,  # 	Snare Drum 1, 2 & 3 Center Right Hand
        84: 38,  # 	Snare Drum 1, 2 & 3 Halfway Left Hand
        85: 42,  # 	Hihat Closed Tight Tip Left Hand
        86: 38,  # 	Snare Drum 1, 2 & 3 Halfway Right Hand
        87: 42,  # 	Hihat Closed Tight Tip Right Hand
        88: 41,  # 	Tom 4 (Floor Tom) Center Left Hand
        89: 41,  # 	Tom 4 (Floor Tom) Center Right Hand
        90: 42,  # 	Hihat Closed Tip Left Hand
        91: 43,  # 	Tom 3 (Rack Tom Low) Center Left Hand
        92: 42,  # 	Hihat Closed Tip Right Hand
        93: 43,  # 	Tom 3 (Rack Tom Low) Center Right Hand
        94: 42,  # 	Hihat Closed Shank Left Hand
        95: 45,  # 	Tom 2 (Rack Tom Mid) Center Left Hand
        96: 45,  # 	Tom 2 (Rack Tom Mid) Center Right Hand
        97: 42,  # 	Hihat Closed Shank Right Hand
        98: 47,  # 	Tom 1 (Rack Tom High) Center Left Hand
        100: 47,  # Tom 1 (Rack Tom High) Center Right Hand
        101: 38,  # Snare Drum 1, 2 & 3 Splash Off
        102: 38,  # Snare Drum 1, 2 & 3 Splash Rim
        103: 38,  # Snare Drum 1, 2 & 3 Splash On
    },
    # From: https://www.native-instruments.com/fileadmin/ni_media/downloads/manuals/Abbey_Road_Modern_Drummer_Manual_English_2012_07.zip
    "ar_modern_white_kit_full.nkm": {
        17: 55,  # 	Perc 4 (Chopper) Low
        19: 55,  # 	Perc 4 (Chopper) Mid
        21: 55,  # 	Perc 4 (Chopper) High
        22: 57,  # 	Cymbal 1 (High Crash) Choke ***
        23: 51,  # 	Cymbal 3 (Ride) Choke ***
        24: 49,  # 	Cymbal 2 (Low Crash) Choke ***
        25: 52,  # 	Cymbal 4 (China) Choke ***
        26: 55,  # 	Cymbal 5 (Splash) Choke ***
        27: 56,  # 	Perc 3 (Cowbell) Low Open
        28: 56,  # 	Perc 3 (Cowbell) Low Muted
        29: 56,  # 	Perc 3 (Cowbell) High Open
        30: 56,  # 	Perc 3 (Cowbell) High Muted
        31: 52,  # 	Perc 5 (Spiral) Stick
        32: 52,  # 	Perc 5 (Spiral) Mallet
        33: 39,  # 	Perc 2 (Clap) Solo
        34: 39,  # 	Perc 2 (Clap) Multi
        35: 31,  # 	Perc 1 (Stick) Hit
        36: 36,  # 	Kick Drum Dampened
        37: 37,  # 	Snare Drum 1, 2 & 3 Sidestick
        38: 38,  # 	Snare Drum 1, 2 & 3 Center Right/Left Alternating *
        39: 38,  # 	Snare Drum 1, 2 & 3 Rimshot
        40: 38,  # 	Snare Drum 1, 2 & 3 Halfway Right/Left Alternating *
        41: 41,  # 	Tom 4 (Floor Tom) Center Right/Left Alternating *
        42: 42,  # 	Hihat Closed Tip Right/Left Alternating*
        43: 43,  # 	Tom 3 (Rack Tom Low) Center Right/Left Alternating *
        44: 44,  # 	Hihat Closed Pedal
        45: 45,  # 	Tom 2 (Rack Tom Mid) Center Right/Left Alternating *
        46: 46,  # 	Hihat Open Controller**
        47: 47,  # 	Tom 1 (Rack Tom High) Center Right/Left Alternating *
        48: 57,  # 	Cymbal 1 (High Crash) Tip
        49: 57,  # 	Cymbal 1 (High Crash) Edge
        50: 57,  # 	Cymbal 1 (High Crash) Bell
        51: 51,  # 	Cymbal 3 (Ride) Tip
        52: 51,  # 	Cymbal 3 (Ride) Edge
        53: 53,  # 	Cymbal 3 (Ride) Bell
        54: 49,  # 	Cymbal 2 (Low Crash) Tip
        55: 49,  # 	Cymbal 2 (Low Crash) Edge
        56: 49,  # 	Cymbal 2 (Low Crash) Bell
        57: 52,  # 	Cymbal 4 (China) Edge
        58: 52,  # 	Cymbal 4 (China) Tip
        59: 52,  # 	Cymbal 5 (Splash) Edge
        60: 36,  # 	Kick Drum Open
        61: 37,  # 	Snare Drum 1, 2 & 3 Rim Only
        62: 38,  # 	Snare Drum 1, 2 & 3 Flam
        63: 38,  # 	Snare Drum 1, 2 & 3 Roll
        64: 38,  # 	Snare Drum 1, 2 & 3 Wires Off
        65: 41,  # 	Tom 4 (Floor Tom) Rimshot
        66: 42,  # 	Hihat Closed Tight Tip Right/Left Alternating *
        67: 43,  # 	Tom 3 (Rack Tom Low) Rimshot
        68: 42,  # 	Hihat Closed Shank Right/Left Alternating *
        69: 45,  # 	Tom 2 (Rack Tom Mid) Rimshot
        70: 44,  # 	Hihat Open Pedal
        71: 47,  # 	Tom 1 (Rack Tom High) Rimshot
        72: 37,  # 	Tom 4 (Floor Tom) Rim Only
        73: 37,  # 	Tom 3 (Rack Tom Low) Rim Only
        74: 37,  # 	Tom 2 (Rack Tom Mid) Rim Only
        75: 37,  # 	Tom 1 (Rack Tom High) Rim Only
        76: 46,  # 	Hihat Open Quarter
        77: 46,  # 	Hihat Open Half
        78: 46,  # 	Hihat Open Three
        79: 46,  # 	Hihat Open Loose
        80: 46,  # 	Hihat Open Full
        81: 38,  # 	Snare Drum 1, 2 & 3 Center Left Hand
        82: 36,  # 	Kick Half Open
        83: 38,  # 	Snare Drum 1, 2 & 3 Center Right Hand
        84: 38,  # 	Snare Drum 1, 2 & 3 Halfway Left Hand
        85: 42,  # 	Hihat Closed Tight Tip Left Hand
        86: 38,  # 	Snare Drum 1, 2 & 3 Halfway Right Hand
        87: 42,  # 	Hihat Closed Tight Tip Right Hand
        88: 41,  # 	Tom 4 (Floor Tom) Center Left Hand
        89: 41,  # 	Tom 4 (Floor Tom) Center Right Hand
        90: 42,  # 	Hihat Closed Tip Left Hand
        91: 43,  # 	Tom 3 (Rack Tom Low) Center Left Hand
        92: 42,  # 	Hihat Closed Tip Right Hand
        93: 43,  # 	Tom 3 (Rack Tom Low) Center Right Hand
        94: 42,  # 	Hihat Closed Shank Left Hand
        95: 45,  # 	Tom 2 (Rack Tom Mid) Center Left Hand
        96: 45,  # 	Tom 2 (Rack Tom Mid) Center Right Hand
        97: 42,  # 	Hihat Closed Shank Right Hand
        98: 47,  # 	Tom 1 (Rack Tom High) Center Left Hand
        100: 47,  # Tom 1 (Rack Tom High) Center Right Hand:
    },
}


EGMD_MIDI = {
    # From https://rolandus.zendesk.com/hc/en-us/articles/360005173411-TD-17-Default-Factory-MIDI-Note-Map
    36: 36,  # KICK
    38: 38,  # SNARE (HEAD)
    40: 38,  # SNARE  (RIM)
    37: 37,  # SNARE X-Stick
    48: 48,  # TOM 1
    50: 48,  # TOM 1 (RIM)
    45: 45,  # TOM 2
    47: 45,  # TOM 2 (RIM)
    43: 43,  # TOM 3 (HEAD)
    58: 43,  # TOM 3 (RIM)
    46: 46,  # HH OPEN (BOW)
    26: 46,  # HH OPEN (EDGE)
    42: 42,  # HH CLOSED (BOW)
    22: 42,  # HH CLOSED (EDGE)
    44: 44,  # HH PEDAL
    49: 49,  # CRASH 1 (BOW)
    55: 49,  # CRASH 1 (EDGE)
    57: 49,  # CRASH 2 (BOW)
    52: 49,  # CRASH 2 (EDGE)
    51: 51,  # RIDE (BOW)
    59: 51,  # RIDE (EDGE)
    53: 53,  # RIDE (BELL)
    # 27: None, #AUX (HEAD)
    # 28: None, #AUX (RIM)
    # From Table 2: https://arxiv.org/pdf/2004.00188.pdf
    54: 54,  # TODO convert to HH?
    39: 39,  # TODO clap?
}
