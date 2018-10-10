# from python-midi in vendors
import midi
import mido
import json
import os
import warnings
import argparse

# Load static variables
INI_NAME = "song.ini"
PS_MIDI_NAME = "notes.mid"
PS_DRUM_TRACK_NAMES = ["PART DRUMS_2X","PART DRUMS"]

# When the 112 is played, the 100 is played too but shouldn't
with open(os.path.join(os.path.dirname(__file__), "./conversionDictionnaries/PhaseShiftArtefacts.json"), 'r') as outfile:
    PS_CYMBAL_DETECTION = {int(key): int(value) for key, value in json.load(outfile).items()}

# Convert the redondant classes of midi to the same base (ie.: the bass drum 35 or 36 are converted to 36)
# See https://en.wikipedia.org/wiki/General_MIDI#Percussion for the full list of events
with open(os.path.join(os.path.dirname(__file__), "./conversionDictionnaries/StandardMidiToReduced.json"), 'r') as outfile:
    REDUCED_MIDI = {int(key): int(value) for key, value in json.load(outfile).items()}

# midi notes used by the game Phase Shifter
# the controller doesn't have a precise representation of each drums
with open(os.path.join(os.path.dirname(__file__), "./conversionDictionnaries/PhaseShiftMidiToStandard.json"), 'r') as outfile:
    PS_MIDI = {int(key): int(value) for key, value in json.load(outfile).items()}


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
    # I delete them instead of creating a new pattern because pattern is more than a list and list comprehension wouldn't work
    tracksToRemove = [i for i, track in enumerate(pattern) if "text" in dir(track[0]) and track[0].text != PS_DRUM_TRACK_NAMES]
    for trackId in sorted(tracksToRemove, reverse=True):
        del pattern[trackId]

    # for each track
    for i, track in enumerate(pattern):

        # add the delay
        if delay != 0:
            ticks = secondToTick(delay, ppq=pattern.resolution)
            track[0].tick += ticks

        # Change the note on each event to standard midi pitches
        simultaneousEvents = []
        for event in track:
            simultaneousEvents.append(event)
            if event.tick != 0:
                convertPitches([localEvent for localEvent in simultaneousEvents if localEvent.name == "Note On"])
                convertPitches([localEvent for localEvent in simultaneousEvents if localEvent.name == "Note Off"])
                simultaneousEvents = []

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

    Arguments:
        events: Synchronous midi events
    """
    #All pitches played at this time
    allPitches = set([event.data[0] for event in events])
    #keeping track of duplicated pitches after the classes reduction
    existingPitches = set([])

    for event in events:
        pitch = event.data[0]

        # remove the extra pitches for Phase Shift cympal detection
        if pitch in PS_CYMBAL_DETECTION and PS_CYMBAL_DETECTION[pitch] in allPitches:
            pitch = None
        # Convert to standard midi pitches
        pitch = PS_MIDI[pitch] if pitch in PS_MIDI else pitch
        # Reduce the number of pitches by converting to base classes
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