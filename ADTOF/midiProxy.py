# from python-midi in vendors
import midi
import json
import os

# Load notes dictionaries
# when the 112 is played, the 100 is played too but shouldn't
# I think it is an artefact from the pads used to tab the musics
with open(os.path.join(os.path.dirname(__file__), "./conversionDictionnaries/PhaseShifterArtefacts.json"), 'r') as outfile:
    PADARTEFACTS = {int(key): int(value) for key, value in json.load(outfile).items()}
# see https://en.wikipedia.org/wiki/General_MIDI#Percussion for the full list of events
with open(os.path.join(os.path.dirname(__file__), "./conversionDictionnaries/StandardMidiToReduced.json"), 'r') as outfile:
    MIDIREDUCED = {int(key): int(value) for key, value in json.load(outfile).items()}
# midi notes used by the game Phase Shifter
# the controller doesn't have a precise representation of each drums
with open(os.path.join(os.path.dirname(__file__), "./conversionDictionnaries/PhaseShifterMidiToStandard.json"), 'r') as outfile:
    PHASEMIDI = {int(key): int(value) for key, value in json.load(outfile).items()}

# def readEofIni(iniPath):
#     with open(iniPath, "r") as iniFile:
#         rows = iniFile.read().split("\n")
#         return {row[0], row[1] for  }

def convertMidi(phaseShiftMidiPath, outputMidiPath):
    """
    TODO
    """
    # Read the midi file
    pattern = midi.read_midifile(phaseShiftMidiPath)

    # Check if the format of the midi file is supported
    if pattern.format != 1 and pattern.resolution < 0:
        Exception("ERROR midi format not implemented")

    # Change the note of each event to the standard midi notation
    for i,track in enumerate(pattern):
        # remove the first track which is the tempo track
        if i == 0: continue

        simultaneousEvents = []
        for event in track:
            simultaneousEvents.append(event)

            if event.tick != 0:
                convertSimultaneousEvents([localEvent for localEvent in simultaneousEvents if localEvent.name == "Note On"])
                convertSimultaneousEvents([localEvent for localEvent in simultaneousEvents if localEvent.name == "Note Off"])
                simultaneousEvents = []

        # Remove empty events:
        pattern[i] = [event for event in pattern[i] if (event.name != "Note On" and event.name != "Note Off") or event.data[0] != None]

    # Write the resulting file
    midi.write_midifile(outputMidiPath, pattern)


def convertSimultaneousEvents(events):
    """
    TODO : add documentation
    """
    allnotes = [event.data[0] for event in events]

    for event in events:
        # remove the notes wich shouldn't be there
        if event.data[0] in PADARTEFACTS and PADARTEFACTS[event.data[0]] in allnotes:
            event.data[0] = 0
        # Convert to standard midi pitch
        event.data[0] = PHASEMIDI[event.data[0]] if event.data[0] in PHASEMIDI else event.data[0]
        # Reduce the number of pitches by converting to base classes
        event.data[0] = MIDIREDUCED[event.data[0]] if event.data[0] in MIDIREDUCED else 0

    # # remove duplicated notes
    # return list(set(events))
    return events


convertMidi("notes.mid", 'result.mid')
