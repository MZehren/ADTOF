#!/usr/bin/env python

from collections import defaultdict
import os
import warnings
from shutil import copyfile
from typing import List, Tuple

import ffmpeg
from adtof import config
from adtof.config import ANIMATIONS_MIDI, EXPERT_MIDI, MIDI_REDUCED_5, MIDI_REDUCED_8
from adtof.converters.converter import Converter
from adtof.io.midiProxy import PythonMidiProxy
import pandas as pd


class PhaseShiftConverter(Converter):
    """
    Convert PhaseShift MIDI files into standard MIDI files based on mapping dictionaries
    """

    # Static variables
    INI_NAME = "song.ini"
    PS_MIDI_NAME = "notes.mid"  # TODO remove this field
    PS_MIDI_NAMES = ["notes.mid"]
    PS_AUDIO_NAMES = ["song.ogg", "drums.ogg", "guitar.ogg"]
    PS_DRUM_TRACK_NAMES = ["PART REAL_DRUMS_PS", "PART DRUMS_2X", "PART DRUMS"]  # By order of quality

    def convert(self, inputFolder, outputMidiPath, outputRawMidiPath, outputAudioPath, addDelay=True):
        """
        Read the ini file and convert the midi file to the standard events
        """
        # read the ini file
        delay = None
        self.name = ""
        try:
            metadata = self.readIni(os.path.join(inputFolder, PhaseShiftConverter.INI_NAME))
            delay = float(metadata["delay"]) / 1000 if "delay" in metadata and metadata["delay"] != "" else 0.0
            self.name = metadata["name"]
            if "pro_drums" not in metadata or not metadata["pro_drums"] or metadata["pro_drums"] != "True":
                warnings.warn("song.ini doesn't contain pro_drums = True " + self.name)
        except:
            warnings.warn("song.ini not found " + inputFolder)

        if not addDelay or not delay:
            delay = 0

        # Read the midi file
        inputMidiPath = os.path.join(inputFolder, PhaseShiftConverter.PS_MIDI_NAME)
        midi = PythonMidiProxy(inputMidiPath)

        # clean the midi
        midi = self.cleanMidi(midi, delay=delay)

        # Write the resulting file
        if outputMidiPath:
            trackName = self.getMetaInfo(inputFolder)["name"]
            _, audioFiles, _ = self.getConvertibleFiles(inputFolder)

            inputAudioFiles = [os.path.join(inputFolder, audioFile) for audioFile in audioFiles]
            midi.save(outputMidiPath)
            copyfile(inputMidiPath, outputRawMidiPath)
            self.cleanAudio(inputAudioFiles, outputAudioPath)

        return midi

    def cleanAudio(self, audioFiles, outputAudioPath):
        """
        Copy the audio file or generate one from multi inputs
        """
        if len(audioFiles) == 1:
            copyfile(os.path.join(audioFiles[0]), outputAudioPath)
        else:
            ffmpeg.filter([ffmpeg.input(audioFile) for audioFile in audioFiles], "amix", inputs=len(audioFiles)).filter(
                "volume", len(audioFiles)
            ).output(outputAudioPath, **{"b:a": "128k"}).run(overwrite_output=True)

    def isConvertible(self, inputFolder):
        """
        Check if the path provided is a convertible PhaseShift custom
        """
        return all(self.getConvertibleFiles(inputFolder))

    def getMetaInfo(self, inputFolder):
        """
        Read the ini file to get meta infos
        """
        ini = self.readIni(os.path.join(inputFolder, PhaseShiftConverter.INI_NAME))

        meta = {
            "name": ini["artist"].replace("/", "-") + " - " + ini["name"].replace("/", "-")
            if "name" in ini and "artist" in ini
            else os.path.basename(inputFolder),
            "genre": ini["genre"] if "genre" in ini else None,
            "pro_drums": ini["pro_drums"] if "pro_drums" in ini else None,
        }
        return meta

    def getTrackName(self, inputFolder):
        return self.getMetaInfo(inputFolder)["name"]

    def getFirstOccurenceOfIntersection(self, A: list, B: list):
        """ Util function to select the first file"""

        def myIn(a: str, B: List[str]):
            """
            Quick and dirty check of string  bieng included in one of the element of B
            """
            for b in B:
                if b is not None and a in b:
                    return b
            return False

        for a in A:
            b = myIn(a, B)
            if b:
                return b
        return None

    def getConvertibleFiles(self, inputFolder) -> Tuple[str, List[str], str]:
        """
        Return the files from 
        """
        if os.path.isdir(inputFolder) == False:
            return None, None, None

        files = os.listdir(inputFolder)
        midiFile = self.getFirstOccurenceOfIntersection(PhaseShiftConverter.PS_MIDI_NAMES, files)
        audioFiles = [
            file for file in files if ".ogg" in file and file != "preview.ogg"
        ]  # getFirstOccurenceOfIntersection(PhaseShiftConverter.PS_AUDIO_NAMES, files)
        iniFile = PhaseShiftConverter.INI_NAME if PhaseShiftConverter.INI_NAME in files else None
        return midiFile, audioFiles, iniFile

    def readIni(self, iniPath):
        """
        the ini file is of shape:

        [song]
        delay = 0
        multiplier_note = 116
        artist = Acid Bath
        ...

        """
        with open(iniPath, "rU", errors="ignore") as iniFile:
            rows = iniFile.read().split("\n")
            items = [row.split(" = ") for row in rows]
            return {item[0]: item[1] for item in items if len(item) == 2}

    def cleanMidi(self, midi, delay=0):
        """
        Clean the midi file to a standard file with standard pitches, only one drum track, and remove the duplicated events.

        Arguments:
            midi: midi file from python-midi
            delay (seconds): add this delay at the start of the midi file
        """
        # Check if the format of the midi file is supported
        if midi.type != 1:
            raise NotImplementedError("ERROR: MIDI format not implemented, Expecting a format 1 MIDI in " + midi)

        # Remove the non-drum tracks
        self.removeUnwantedTracks(midi)

        # add the delay
        midi.addDelay(delay)

        # Convert the pitches
        self.convertTracks(midi)

        return midi

    def removeUnwantedTracks(self, midi):
        """Delete tracks without drums

        Parameters
        ----------
        midi : the midi object

        Raises
        ------
        ValueError
            raises an error when no drum track has been found
        """
        tracksName = midi.getTracksName()
        drumTrack = self.getFirstOccurenceOfIntersection(PhaseShiftConverter.PS_DRUM_TRACK_NAMES, tracksName)
        if drumTrack is None:
            raise ValueError("ERROR: No drum track in the MIDI file " + midi)
        tracksToRemove = [i for i, trackName in enumerate(tracksName) if trackName != None and trackName != drumTrack and i != 0]
        for trackId in sorted(tracksToRemove, reverse=True):
            del midi.tracks[trackId]

    def convertTracks(self, midi):
        """Convert the pitches from the midi tracks

        Parameters
        ----------
        midi : the midi object to convert
        """
        # TODO: clean the code
        self.expertDiscrepancies = defaultdict(int)
        self.animDiscrepancies = defaultdict(int)
        for track in midi.tracks:
            notesOn = {}
            hasAnimation = any([True for event in track if midi.getEventPith(event) in config.ANIMATIONS_MIDI])
            for event in track:
                # Keep the original pitch as a key
                notePitch = midi.getEventPith(event)

                # Before the start of a new time step, do the conversion
                if midi.getEventTick(event) > 0:
                    # Convert the note on and off events to the same pitches
                    conversion = self.convertPitches(notesOn.keys(), hasAnimation)
                    for pitch, passedEvent in notesOn.items():
                        # Set the pitch, if the note is not converted we set it to 0 and remove it later
                        midi.setEventPitch(passedEvent, conversion.get(pitch, 0))

                # Keep track of the notes currently playing
                if midi.isEventNoteOn(event):
                    if notePitch in notesOn:
                        warnings.warn("error MIDI Note On overriding existing note")
                    else:
                        notesOn[notePitch] = event
                elif midi.isEventNoteOff(event):
                    if notePitch not in notesOn:
                        warnings.warn("error MIDI Note Off not existing")
                    else:
                        midi.setEventPitch(event, notesOn[notePitch].pitch)
                    notesOn.pop(notePitch, None)

            # Remove empty events with a pitch set to 0 from the convertPitches method:
            eventsToRemove = [
                j
                for j, event in enumerate(track)
                if (midi.isEventNoteOn(event) or midi.isEventNoteOff(event)) and midi.getEventPith(event) == 0
            ]
            for j in sorted(eventsToRemove, reverse=True):
                # Save to time information from the event removed in the next event
                if midi.getEventTick(track[j]) and len(track) > j + 1:
                    midi.setEventTick(track[j + 1], midi.getEventTick(track[j]) + midi.getEventTick(track[j + 1]))
                del track[j]

        if len(self.expertDiscrepancies) or len(self.animDiscrepancies):
            table = pd.DataFrame({"Expert addition": self.expertDiscrepancies, "anim addition": self.animDiscrepancies})
            warnings.warn("Discrepancie between expert and animation annotations: " + self.name + "\n" + str(table))

    def convertPitches(self, pitches, useAnimation):
        """
        Convert the notes from a list of simultaneous events to standard pitches.
        The events which should be removed have a pitch set to 0.
        """

        converted = config.getPitchesRemap(pitches, EXPERT_MIDI)

        if useAnimation:
            animation = config.getPitchesRemap(pitches, ANIMATIONS_MIDI)
            # Check if there are discrepeancies between expert and animation
            simpleExpert = set(config.remapPitches(converted.values(), MIDI_REDUCED_5))
            simpleAnim = set(config.remapPitches(animation.values(), MIDI_REDUCED_5))

            for pitch in simpleExpert:
                if pitch not in simpleAnim:
                    self.expertDiscrepancies[pitch] += 1
            for pitch in simpleAnim:
                if pitch not in simpleExpert:
                    self.animDiscrepancies[pitch] += 1

            converted = animation

        # TODO This dict generation is very close to config.getPitchesRemap (the diff is replacement of unknown to 0)
        return {k: MIDI_REDUCED_8[v] if v in MIDI_REDUCED_8 else 0 for k, v in converted.items()}

