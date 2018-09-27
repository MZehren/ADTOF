# from python-midi in vendors
import midi
import json
import os

# when the 112 is played, the 100 is played too but shouldn't
# I think it is an artefact from the pads used to tab the musics
with open(os.path.join(os.path.dirname(__file__),"./conversionDictionnaries/PhaseShifterArtefacts.json"), 'r') as outfile:
    PADARTEFACTS = {int(key):int(value) for key,value in json.load(outfile).items()}
# see https://en.wikipedia.org/wiki/General_MIDI#Percussion for the full list of events
with open(os.path.join(os.path.dirname(__file__),"./conversionDictionnaries/StandardMidiToReduced.json"), 'r') as outfile:
    MIDIREDUCED = {int(key):int(value) for key,value in json.load(outfile).items()}
# midi notes used by the game Phase Shifter
# the controller doesn't have a precise representation of each drums
with open(os.path.join(os.path.dirname(__file__),"./conversionDictionnaries/PhaseShifterMidiToStandard.json"), 'r') as outfile:
    PHASEMIDI = {int(key):int(value) for key,value in json.load(outfile).items()}



def main(phaseShiftMidiPath, outputMidiPath):
    """
    TODO
    """
    # Read the midi file
    pattern = midi.read_midifile(phaseShiftMidiPath)

    # Check if the format of the midi file is supported
    if pattern.format != 1 and pattern.resolution < 0:
        print("ERROR midi format not implemented")
        return

    # Change the pitch of each event to the standard one
    for track in pattern[1:]:
        for event in track:
            if event.name == "Note Off" or event.name == "Note On":
                event.data = convertMidiEvent(event.data)
                
    # Write the resulting file
    midi.write_midifile(outputMidiPath, pattern)

def convertMidiEvent(notes):
    """
    TODO
    """
    # remove the events wich shouldn't be there
    notes = [note for note in notes if not(note in PADARTEFACTS and PADARTEFACTS[note] in notes)]
    
    # Convert to standard midi events
    notes = [PHASEMIDI[note] if note in PHASEMIDI else note for note in notes]

    # Reduce the number of classes by merging drums
    notes = [MIDIREDUCED[note] if note in MIDIREDUCED else note for note in notes]

    # Remove duplicated events
    notes = list(set(notes))
    return notes

main("notes.mid", 'result.mid')
