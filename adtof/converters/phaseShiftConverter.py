#!/usr/bin/env python

import logging
import os
import warnings
from collections import defaultdict
from shutil import copyfile
from typing import List, Tuple

import ffmpeg
import pandas as pd
from adtof import config
from adtof.config import ANIMATIONS_MIDI, EXPERT_MIDI, MIDI_REDUCED_5, MIDI_REDUCED_8
from adtof.converters.converter import Converter
from adtof.io.midiProxy import PrettyMidiWrapper


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
            delay = float(metadata["delay"]) / 1000 if "delay" in metadata and metadata["delay"] != "" else 0.0  # Get the delay in s
            self.name = metadata["name"]
            if "pro_drums" not in metadata or not metadata["pro_drums"] or metadata["pro_drums"] != "True":
                warnings.warn("song.ini doesn't contain pro_drums = True " + self.name)
        except:
            warnings.warn("song.ini not found " + inputFolder)

        if not addDelay or not delay:
            delay = 0

        # Read the midi file
        inputMidiPath = os.path.join(inputFolder, PhaseShiftConverter.PS_MIDI_NAME)
        midi = PrettyMidiWrapper(inputMidiPath)

        # clean the midi
        debug = self.cleanMidi(midi, delay=delay)

        # Write the resulting file
        if outputMidiPath:
            trackName = self.getMetaInfo(inputFolder)["name"]
            _, audioFiles, _ = self.getConvertibleFiles(inputFolder)

            inputAudioFiles = [os.path.join(inputFolder, audioFile) for audioFile in audioFiles]
            midi.write(outputMidiPath)
            copyfile(inputMidiPath, outputRawMidiPath)
            self.cleanAudio(inputAudioFiles, outputAudioPath, delay)

        return debug

    def cleanAudio(self, audioFiles, outputAudioPath, delay):
        """
        Copy the audio file or generate one from multi inputs
        If there is a delay in the song.ini, trim the beginning of the audio (delaying the midi file is harder)
        """
        if len(audioFiles) == 1 and delay == 0:
            copyfile(os.path.join(audioFiles[0]), outputAudioPath)
        else:
            outputArgs = {"b:a": "128k"}  # TODO can we keep the original bitrate?
            if delay > 0:
                outputArgs["ss"] = str(delay)

            ffmpeg.filter([ffmpeg.input(audioFile) for audioFile in audioFiles], "amix", inputs=len(audioFiles)).filter(
                "volume", len(audioFiles)
            ).output(outputAudioPath, **outputArgs).global_args("-loglevel", "error").run(overwrite_output=True)

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
        # Remove the non-drum tracks
        self.removeUnwantedTracks(midi)

        # add the delay is removed from the midi. Instead let's trunk the start of the audio track
        # midi.addDelay(delay)

        # Convert the pitches
        if len(midi.instruments) != 1:
            raise ValueError("number of drum tracks in the midi file != 1")  # TODO: should I use the track.is_drum?
        self.convertTrack(midi, midi.instruments[0], useAnimation=False)

        # Debug issues
        # return self.diffTwoTracks(midi.instruments[0])

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
        tracksName = [instrument.name for instrument in midi.instruments]
        drumTrack = self.getFirstOccurenceOfIntersection(PhaseShiftConverter.PS_DRUM_TRACK_NAMES, tracksName)
        if drumTrack is None:
            raise ValueError("ERROR: No drum track in the MIDI file " + midi)
        tracksToRemove = [i for i, trackName in enumerate(tracksName) if trackName != None and trackName != drumTrack]
        for trackId in sorted(tracksToRemove, reverse=True):
            del midi.instruments[trackId]

    def convertTrack(self, midi, track, useAnimation=True):
        """Convert the pitches from the midi tracks

        Parameters
        ----------
        midi : the midi object to convert
        """
        from pretty_midi.containers import Note

        events = [note for note in track.notes] + midi.discoFlip
        events.sort(key=lambda x: x.start)
        events += [Note(0, 0, events[-1].start + 1, events[-1].end + 1)]  # Add a fake event to parse the last note as well

        cursor = 0
        notesOn = {}
        # hasAnimation = any([True for event in track.notes if event.pitch in config.ANIMATIONS_MIDI])
        for event in events:
            # Copy the original pitch
            notePitch = event.pitch

            # Before the start of a new time step, or at the last event, do the conversion
            if event.start > cursor:
                cursor = event.start

                # Convert the note on and off events to the same pitches
                conversions = self.convertPitches(notesOn.keys(), useAnimation)
                for pitch, passedEvent in notesOn.items():
                    # Set the pitch, if the note is not converted we set it to 0 and remove it later
                    passedEvent.pitch = conversions.get(pitch, 0)

                # remove finished events
                notesOn = {k: v for k, v in notesOn.items() if v.end > event.start}

            # Keep track of the notes currently playing
            assert notePitch not in notesOn
            notesOn[notePitch] = event

        # Remove empty events with a pitch set to 0 from the convertPitches method:
        track.notes = [note for note in track.notes if note.pitch != 0]  # TODO change the method to pure

    def convertPitches(self, pitches, useAnimation=True):
        """
        TODO better comments
        Convert the notes from a list of simultaneous events to standard pitches.
        The events which should be removed are not mapped
        """
        conv = config.getPitchesRemap(pitches, EXPERT_MIDI)
        conv = {k: MIDI_REDUCED_5[v] for k, v in conv.items() if v in MIDI_REDUCED_5}

        # TODO: check flams and other stuff when animation and expert don't agree
        # if useAnimation:
        #     animconv = config.getPitchesRemap(pitches, ANIMATIONS_MIDI)
        #     animconv = {k: MIDI_REDUCED_5[v] for k, v in animconv.items() if v in MIDI_REDUCED_5}

        return conv

    def diffTwoTracks(self, track):
        errors = defaultdict(list)
        for i, event in enumerate(track.notes):
            if event.velocity != 1:
                errors[event.start].append(event)

        name = {64: "expert", 127: "animation"}
        log = defaultdict(int)
        for error in errors.values():
            if len(error) == 1:
                log[str(name[error[0].velocity]) + "_" + str(error[0].pitch)] += 1
            elif len(error) == 2 and error[0].velocity != error[1].velocity:
                error.sort(key=lambda e: e.velocity)
                log[str(error[0].pitch) + "<->" + str(error[1].pitch)] += 1
            else:
                log["other"] += 1

        debug = [(k, v) for k, v in log.items()]
        debug.sort(key=lambda e: e[1], reverse=True)
        if len(errors):
            logging.debug("difference in the notes" + ", ".join([k + ": " + str(v) for k, v in debug]))
        return log

