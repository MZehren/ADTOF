# from python-midi in vendors
import midi
import json
import os
import warnings
import argparse

# Load static variables
# For more documentation on the MIDI specifications for PhaseShift or RockBand, check http://docs.c3universe.com/rbndocs/index.php?title=Drum_Authoring
INI_NAME = "song.ini"
PS_MIDI_NAME = "notes.mid"
PS_DRUM_TRACK_NAMES = ["PART REAL_DRUMS_PS", "PART DRUMS_2X","PART DRUMS"] # By order of quality
TOMS_MODIFIER = {110: 98, 111: 99, 112: 100} #ie.: When the 110 is on, changes the note 98 from hi-hat to high tom for the duration of the note.
TOMS_MODIFIER_LOOKUP = {v:k for k,v in TOMS_MODIFIER.items()}
DRUM_ROLLS = 126 #TODO: implement
CYMBAL_SWELL = 127 #TODO: implement

# midi notes used by the game PhaseShift and RockBand
with open(os.path.join(os.path.dirname(__file__), "./conversionDictionnaries/PhaseShiftMidiToStandard.json"), 'r') as outfile:
    PS_MIDI = {int(key): int(value) for key, value in json.load(outfile).items()}


# Convert the redundant classes of midi to the more general one (ie.: the bass drum 35 and 36 are converted to 36)
# See https://en.wikipedia.org/wiki/General_MIDI#Percussion for the full list of events
with open(os.path.join(os.path.dirname(__file__), "./conversionDictionnaries/StandardMidiToReduced.json"), 'r') as outfile:
    REDUCED_MIDI = {int(key): int(value) for key, value in json.load(outfile).items()}


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    parser = argparse.ArgumentParser(description='Process a Phase Shift chart folder and convert the midi file to standard midi')
    parser.add_argument('folderPath', type=str, help="The path to the Phase Shift chart folder.")
    parser.add_argument(
        '-o',
        dest='outputName',
        type=str,
        default="notes_std.mid",
        help="Name of the MIDI file created from the conversion. Default to 'notes_std.mid'"
    )
    args = parser.parse_args()
    
    convertFolder(args.folderPath, args.outputName)

def convertFolder(folderPath, outputName):
    """
    Read the ini file and convert the midi file to the standard events
    """
    metadata = readIni(os.path.join(folderPath, INI_NAME))
    delay = float(metadata["delay"]) / 1000 if "delay" in metadata else 0.

    if not metadata["pro_drums"] or metadata["pro_drums"] != "True":
        warnings.warn("song.ini doesn't contain pro_drums = True")

    # Read the midi file
    pattern = midi.read_midifile(os.path.join(folderPath, PS_MIDI_NAME))

    #clean the midi
    pattern = cleanMidi(pattern, delay=delay)

    # Write the resulting file
    midi.write_midifile(os.path.join(folderPath, outputName), pattern)


def readIni(iniPath):
    """
    the ini file is of shape:

    [song]
    delay = 0
    multiplier_note = 116
    artist = Acid Bath
    ...

    """
    with open(iniPath, "rU") as iniFile:
        rows = iniFile.read().split("\n")
        items = [row.split(" = ") for row in rows]
        return {item[0]: item[1] for item in items if len(item) == 2}


def cleanMidi(pattern, delay=0):
    """
    Clean the midi file to a standard file with correct pitches, only on drum track, and remove the duplicated events.

    Arguments:
        pattern: midi file from python-midi
        delay (seconds): add this delay at the start of the midi file
    """
    # Check if the format of the midi file is supported
    if pattern.format != 1 or pattern.resolution < 0 or not pattern.tick_relative:
        Exception("ERROR MIDI format not implemented, Expecting a format 1 MIDI")

    # Remove the non-drum tracks
    tracksName = [[event.text for event in track if event.name == "Track Name"] for track in pattern]
    tracksName = [names[0] if names else None for names in tracksName]
    for name in PS_DRUM_TRACK_NAMES:
        if name in tracksName:
            tracksToRemove = [i for i, trackName in enumerate(tracksName) if trackName != None and trackName != name]
            break
    for trackId in sorted(tracksToRemove, reverse=True):
        del pattern[trackId]

    # for each track
    for i, track in enumerate(pattern):

        # add the delay
        if delay != 0:
            ticks = secondToTick(delay, ppq=pattern.resolution)
            track[0].tick += ticks

        # Keep track of the simultaneous notes playing 
        notesOn = {}
        notesOff = {}
        for event in track:
            # Before the start of a new time step, do the conversion
            if event.tick > 0:
                convertPitches(notesOn)
                convertPitches(notesOff) # Convert the note off events to the same pitches
                notesOff = {} # Note off events have no duration, so we remove them

            # Keep track of all the notes
            if event.name == "Note On" and event.velocity > 0:
                if event.pitch in notesOn:
                    warnings.warn("error MIDI Note On overriding existing note")
                notesOn[event.pitch] = event
            if (event.name == "Note On" and event.velocity == 0) or event.name == "Note Off":
                if event.pitch not in notesOn:
                    warnings.warn("error MIDI Note Off not existing")
                notesOn.pop(event.pitch, None)
                notesOff[event.pitch] = event

        # Remove empty events with a pitch set to None from the convertPitches method:
        eventsToRemove = [j for j, event in enumerate(track) if (event.name == "Note On" or event.name == "Note Off") and event.data[0] == None]
        for j in sorted(eventsToRemove, reverse=True):
            #Save to time information from the event removed in the next event
            if track[j].tick and len(track) > j + 1:
                track[j + 1].tick += track[j].tick
            del track[j]
    return pattern


def convertPitches(events):
    """
    Convert the notes from a list of simultaneous events to standard pitches.
    The events which should be removed have a pitch set to None.
    
    This function is not pure, it's going to change the items in the dictionnary of events

    Arguments:
        events: dictionnary of simultaneous midi notes. The key has to be the pitch of the note
    """
    #All pitches played at this time
    allPitches = events.keys()
    #keeping track of duplicated pitches after the classes reduction
    existingPitches = set([])

    for pitch, event in events.items():

        # Convert to standard midi pitches and apply the tom modifiers
        if pitch in TOMS_MODIFIER:
            pitch = None #this is not a real note played, but a modifier
        elif pitch in TOMS_MODIFIER_LOOKUP and TOMS_MODIFIER_LOOKUP[pitch] in allPitches:
            pitch = PS_MIDI[TOMS_MODIFIER_LOOKUP[pitch]] #this pitch is played with his modifier
        elif pitch in PS_MIDI:
            pitch = PS_MIDI[pitch] #this pitch doesn't have a modifier
        else:
            pitch = None

        # Remove ambiguous notes (tom alto or tom medium) by converting to base classes (toms)
        pitch = REDUCED_MIDI[pitch] if pitch in REDUCED_MIDI else None

        # Remove duplicated pitches
        if pitch in existingPitches:
            pitch = None

        existingPitches.add(pitch)
        event.data[0] = pitch

    return events


def secondToTick(time, mpqn=500000, ppq=960):
    """
    convert from seconds to midi ticks based on midi resolution

    arguments:
        time: time in seconds
        mpqn: microseconds per quarter note
        ppq: pulses (ticks) per quarter notes
    """
    return int(float(time) / (float(mpqn) / 1000000) * float(ppq))


if __name__ == '__main__':
    main()