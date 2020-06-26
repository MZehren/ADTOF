#!/usr/bin/env python

import argparse
import json
import logging
import ntpath
import os
import sys
import warnings
from shutil import copyfile
from typing import List

import ffmpeg
import numpy as np
import pkg_resources

from adtof import config
from adtof.config import ANIMATIONS_MIDI, EXPERT_MIDI, MIDI_REDUCED_8
from adtof.converters.converter import Converter
from adtof.io.midiProxy import MidiProxy


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

    DRUM_ROLLS = 126  # TODO: implement
    CYMBAL_SWELL = 127  # TODO: implement

    def convert(self, inputFolder, outputMidiPath, outputRawMidiPath, outputAudioPath, addDelay=True):
        """
        Read the ini file and convert the midi file to the standard events
        """
        # read the ini file
        delay = None
        try:
            metadata = self.readIni(os.path.join(inputFolder, PhaseShiftConverter.INI_NAME))
            delay = float(metadata["delay"]) / 1000 if "delay" in metadata and metadata["delay"] != "" else 0.0

            if "pro_drums" not in metadata or not metadata["pro_drums"] or metadata["pro_drums"] != "True":
                warnings.warn("song.ini doesn't contain pro_drums = True  " + inputFolder)
        except:
            warnings.warn("song.ini not found " + inputFolder)

        if not addDelay or not delay:
            delay = 0

        # Read the midi file
        inputMidiPath = os.path.join(inputFolder, PhaseShiftConverter.PS_MIDI_NAME)
        midi = MidiProxy(inputMidiPath)

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

    def getConvertibleFiles(self, inputFolder) -> (str, List[str], str):
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
        Clean the midi file to a standard file with correct pitches, only on drum track, and remove the duplicated events.

        Arguments:
            pattern: midi file from python-midi
            delay (seconds): add this delay at the start of the midi file
        """
        # Check if the format of the midi file is supported
        if midi.type != 1:
            raise NotImplementedError("ERROR: MIDI format not implemented, Expecting a format 1 MIDI in " + midi)

        # Remove the non-drum tracks
        tracksName = midi.getTracksName()
        drumTrack = self.getFirstOccurenceOfIntersection(PhaseShiftConverter.PS_DRUM_TRACK_NAMES, tracksName)
        if drumTrack is None:
            raise ValueError("ERROR: No drum track in the MIDI file " + midi)
        tracksToRemove = [i for i, trackName in enumerate(tracksName) if trackName != None and trackName != drumTrack and i != 0]
        for trackId in sorted(tracksToRemove, reverse=True):
            del midi.tracks[trackId]

        # add the delay
        midi.addDelay(delay)

        # Convert the pitches
        for i, track in enumerate(midi.tracks):
            notesOn = {}
            for i, event in enumerate(track):
                # Keep the original pithc as a key
                notePitch = midi.getEventPith(event)

                # Before the start of a new time step, do the conversion
                if midi.getEventTick(event) > 0:
                    # Convert the note on and off events to the same pitches
                    conversion = self.convertPitches(notesOn.keys())
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
        return midi

    def convertPitches(self, pitches):
        """
        Convert the notes from a list of simultaneous events to standard pitches.
        The events which should be removed have a pitch set to None.
        """
        converted = self.remap(pitches, ANIMATIONS_MIDI)
        if len(converted) == 0:
            converted = self.remap(pitches, EXPERT_MIDI)
        return {k: MIDI_REDUCED_8[v] for k, v in converted.items()}

    def remap(self, pitches, mapping):
        """
        Map pitches to a mapped value from a mapping
        """
        result = {}
        for pitch in pitches:
            if pitch not in mapping:
                continue
            mapped = mapping[pitch]
            if isinstance(mapped, dict):
                mapped = mapped[mapped["modifier"] in pitches]

            result[pitch] = mapped

        return result
