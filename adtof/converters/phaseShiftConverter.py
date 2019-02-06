#!/usr/bin/env python

import argparse
import json
import os
import warnings

import mido
import pkg_resources

from adtof.converters import Converter

# from adtof.io import MidiFile


class PhaseShiftConverter(Converter):
    """
    Convert PhaseShift MIDI files into standard MIDI files based on mapping dictionaries
    """
    # Static variables
    # For more documentation on the MIDI specifications for PhaseShift or RockBand, check http://docs.c3universe.com/rbndocs/index.php?title=Drum_Authoring
    INI_NAME = "song.ini"
    PS_MIDI_NAME = "notes.mid"
    PS_DRUM_TRACK_NAMES = ["PART REAL_DRUMS_PS", "PART DRUMS_2X", "PART DRUMS"]  # By order of quality
    # ie.: When the 110 is on, changes the note 98 from hi-hat to high tom for the duration of the note.
    TOMS_MODIFIER = {110: 98, 111: 99, 112: 100}
    TOMS_MODIFIER_LOOKUP = {v: k for k, v in TOMS_MODIFIER.items()}
    DRUM_ROLLS = 126  # TODO: implement
    CYMBAL_SWELL = 127  # TODO: implement

    # midi notes used by the game PhaseShift and RockBand
    _ps2m = pkg_resources.resource_string(__name__, "mappingDictionaries/phaseShiftMidiToStandard.json").decode()
    PS_MIDI = {int(key): int(value) for key, value in json.loads(_ps2m).items()}

    # Convert the redundant classes of midi to the more general one (ie.: the bass drum 35 and 36 are converted to 36)
    # See https://en.wikipedia.org/wiki/General_MIDI#Percussion for the full list of events
    _m2r = pkg_resources.resource_string(__name__, "mappingDictionaries/standardMidiToReduced.json").decode()
    REDUCED_MIDI = {int(key): int(value) for key, value in json.loads(_m2r).items()}

    def convert(self, folderPath, inputFile, outputName, addDelay=True):
        """
        Read the ini file and convert the midi file to the standard events
        """
        #read the ini file
        delay = None
        try:
            metadata = self.readIni(os.path.join(folderPath, PhaseShiftConverter.INI_NAME))
            delay = float(metadata["delay"]) / 1000 if "delay" in metadata else 0.

            if not metadata["pro_drums"] or metadata["pro_drums"] != "True":
                warnings.warn("song.ini doesn't contain pro_drums = True")
        except:
            warnings.warn("song.ini not found")

        if not addDelay or not delay:
            delay = 0

        # Read the midi file
        midi = mido.MidiFile(os.path.join(folderPath, inputFile))

        # clean the midi
        midi = self.cleanMidi(midi, delay=delay)

        # Write the resulting file
        midi.save(os.path.join(folderPath, outputName))

    def isConvertible(self, folderPath, inputFile):
        files = os.listdir(folderPath)
        return inputFile == PhaseShiftConverter.PS_MIDI_NAME and  "song.ini" in files and ("song.ogg" in files or "guitar.ogg" in files)


    def readIni(self, iniPath):
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

    def cleanMidi(self, midi, delay=0):
        """
        Clean the midi file to a standard file with correct pitches, only on drum track, and remove the duplicated events.

        Arguments:
            pattern: midi file from python-midi
            delay (seconds): add this delay at the start of the midi file
        """
        # Check if the format of the midi file is supported
        if midi.type != 1:
            raise Exception("ERROR: MIDI format not implemented, Expecting a format 1 MIDI")

        # Remove the non-drum tracks
        tracksName = [[event.name for event in track if event.type == "track_name"] for track in midi.tracks]
        tracksName = [names[0] if names else None for names in tracksName]
        drumTrackFlag = False
        for name in PhaseShiftConverter.PS_DRUM_TRACK_NAMES:
            if name in tracksName:
                drumTrackFlag = True
                tracksToRemove = [i for i, trackName in enumerate(tracksName) if trackName != None and trackName != name]
                for trackId in sorted(tracksToRemove, reverse=True):
                    del midi.tracks[trackId]
                break

        if not drumTrackFlag:
            raise Exception("ERROR: No drum track in the MIDI file")

        # for each track
        for i, track in enumerate(midi.tracks):

            # add the delay
            if delay != 0:
                if i == 0: 
                    #If this is the tempo track, add a set tempo meta event event such as the delay is a 4 beats
                    event = mido.MetaMessage("set_tempo")
                    event.tempo = int(delay * 1000000 / 4)
                    track.insert(0, event)
                    track[1].time += 4 * midi.ticks_per_beat
                else:
                    #If this is a standard track, add a delay of 4 beats to the first event
                    track[0].time += 4 * midi.ticks_per_beat

            # Keep track of the simultaneous notes playing
            notesOn = {}
            notesOff = {}
            for event in track:
                # Before the start of a new time step, do the conversion
                if event.time > 0:
                    self.convertPitches(notesOn)
                    # Convert the note off events to the same pitches
                    self.convertPitches(notesOff)
                    notesOff = {}  # Note off events have no duration, so we remove them

                # Keep track of all the notes
                if event.type == "note_on" and event.velocity > 0:
                    if event.note in notesOn:
                        warnings.warn("error MIDI Note On overriding existing note")
                    notesOn[event.note] = event
                if (event.type == "note_on" and event.velocity == 0) or event.type == "note_off":
                    if event.note not in notesOn:
                        warnings.warn("error MIDI Note Off not existing")
                    notesOn.pop(event.note, None)
                    notesOff[event.note] = event

            # Remove empty events with a pitch set to None from the convertPitches method:
            eventsToRemove = [j for j, event in enumerate(track) if (event.type == "note_on" or event.type == "note_off") and event.note == 0]
            for j in sorted(eventsToRemove, reverse=True):
                # Save to time information from the event removed in the next event
                if track[j].time and len(track) > j + 1:
                    track[j + 1].time += track[j].time
                del track[j]
        return midi

    def convertPitches(self, events):
        """
        Convert the notes from a list of simultaneous events to standard pitches.
        The events which should be removed have a pitch set to None.

        This function is not pure, it's going to change the items in the dictionnary of events

        Arguments:
            events: dictionnary of simultaneous midi notes. The key has to be the pitch of the note
        """
        # All pitches played at this time
        allPitches = events.keys()
        # keeping track of duplicated pitches after the classes reduction
        existingPitches = set([])

        for pitch, event in events.items():

            # Convert to standard midi pitches and apply the tom modifiers
            if pitch in PhaseShiftConverter.TOMS_MODIFIER:
                pitch = 0  # this is not a real note played, but a modifier
            elif pitch in PhaseShiftConverter.TOMS_MODIFIER_LOOKUP and PhaseShiftConverter.TOMS_MODIFIER_LOOKUP[pitch] in allPitches:
                # this pitch is played with his modifier
                pitch = PhaseShiftConverter.PS_MIDI[PhaseShiftConverter.TOMS_MODIFIER_LOOKUP[pitch]]
            elif pitch in PhaseShiftConverter.PS_MIDI:
                pitch = PhaseShiftConverter.PS_MIDI[pitch]  # this pitch doesn't have a modifier
            else:
                pitch = 0

            # Remove ambiguous notes (tom alto or tom medium) by converting to base classes (toms)
            pitch = PhaseShiftConverter.REDUCED_MIDI[pitch
                                                     ] if pitch in PhaseShiftConverter.REDUCED_MIDI and PhaseShiftConverter.REDUCED_MIDI[pitch] else 0

            # Remove duplicated pitches
            if pitch in existingPitches:
                pitch = 0

            existingPitches.add(pitch)
            event.note = pitch

        return events

    def secondTotime(self, time, mpqn=500000, ppq=960):
        """
        convert from seconds to midi times based on midi resolution

        arguments:
            time: time in seconds
            mpqn: microseconds per quarter note
            ppq: pulses (times) per quarter notes
        """
        return int(float(time) / (float(mpqn) / 1000000) * float(ppq))