#!/usr/bin/env python

import logging
import os
import warnings
from collections import defaultdict
from shutil import copyfile
from typing import List, Tuple

import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import angle
import pandas as pd
from adtof import config
from adtof.config import ANIMATIONS_MIDI, EXPERT_MIDI, MIDI_REDUCED_5, MIDI_REDUCED_8
from adtof.converters.converter import Converter
from adtof.io.midiProxy import PrettyMidiWrapper
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer


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

    # Debug difference between expert and anim
    ANIM = []
    EXPERT = []
    ANIM_KEY = []
    EXPERT_KEY = []
    HAS_ANIM = []

    def convert(self, inputFolder, outputMidiPath, outputRawMidiPath, outputAudioPath, addDelay=True):
        """
        Read the ini file and convert the midi file to the standard events
        """
        # read the ini file
        delay = None
        self.name = self.getMetaInfo(inputFolder)["name"]
        try:
            metadata = self.readIni(os.path.join(inputFolder, PhaseShiftConverter.INI_NAME))
            delay = float(metadata["delay"]) / 1000 if "delay" in metadata and metadata["delay"] != "" else 0.0  # Get the delay in s
            if "pro_drums" not in metadata or not metadata["pro_drums"] or metadata["pro_drums"] != "True":
                warnings.warn("song.ini doesn't contain pro_drums = True " + self.name)
        except Exception as e:
            warnings.warn("song.ini not found " + inputFolder + str(e))

        if not addDelay or not delay:
            delay = 0

        # Read the midi file
        inputMidiPath = os.path.join(inputFolder, PhaseShiftConverter.PS_MIDI_NAME)
        midi = PrettyMidiWrapper(inputMidiPath)

        # clean the midi
        debug = self.cleanMidi(midi)

        # Write the resulting file
        if outputMidiPath:

            _, audioFiles, _ = self.getConvertibleFiles(inputFolder)

            inputAudioFiles = [os.path.join(inputFolder, audioFile) for audioFile in audioFiles]
            midi.write(outputMidiPath)
            copyfile(inputMidiPath, outputRawMidiPath)
            self.cleanAudio(inputAudioFiles, outputAudioPath, delay)

        return delay

    def cleanAudio(self, audioFiles, outputAudioPath, delay):
        """
        Copy the audio file or generate one from multi inputs
        If there is a delay in the song.ini, trim the beginning of the audio (delaying the midi file was harder)
        """
        if len(audioFiles) == 1 and delay == 0:
            copyfile(os.path.join(audioFiles[0]), outputAudioPath)
        else:
            outputArgs = {"b:a": "128k"}  # TODO can we keep the original bitrate?
            if delay > 0:
                outputArgs["ss"] = str(delay)
            elif delay < 0:
                raise ValueError("midi delay is negative")

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

    def cleanMidi(self, midi):
        """
        Clean the midi file to a standard file with standard pitches, only one drum track, and remove the duplicated events.

        Arguments:
            midi: midi file from python-midi
        """
        # Remove the non-drum tracks
        self.removeUnwantedTracks(midi)

        # Convert the pitches
        self.convertTrack(midi)

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

        # Pretty midi split the notes from the same track on drum and not drum channel.
        # We should merge them in case of error of channel from the annotator
        if len(midi.instruments) != 1:
            logging.debug("number of drum tracks in the midi file != 1, merging the tracks")
            midi.instruments[0].notes = [note for instrument in midi.instruments for note in instrument.notes]
            midi.instruments[0].notes.sort(key=lambda n: n.start)
            for trackId in sorted(range(1, len(midi.instruments)), reverse=True):
                del midi.instruments[trackId]

    def convertTrack(self, midi, debug=False):
        """Convert the pitches from the midi tracks

        Parameters
        ----------
        midi : the midi object to convert
        debug: If plotting the "animation" and "expert" mismatch
        """
        from pretty_midi.containers import Note

        # Add the discoFlip events as notes
        track = midi.instruments[0]
        events = [note for note in track.notes] + midi.discoFlip
        events.sort(key=lambda x: x.start)
        # DIRTY code: Adding a fake event at the end so the for loop works
        events += [Note(0, 0, events[-1].start + 1, events[-1].end + 1)]

        # Check if the track has "annimation" annotations or only "expert" ones
        if debug:
            hasAnimation = len([1 for event in track.notes if event.pitch in config.ANIMATIONS_MIDI]) > 100
            PhaseShiftConverter.HAS_ANIM.append(int(hasAnimation))

        # Convert all the pitches
        cursor = 0
        notesOn = {}
        for event in events:
            # Copy the original pitch
            notePitch = event.pitch

            # At the start of a new time step, do the conversion of the previous events
            if event.start > cursor:
                cursor = event.start

                # Convert the note on events to the same pitches
                animConversions, conversions = self.convertPitches(notesOn.keys())
                for pitch, passedEvent in notesOn.items():
                    # convert the pitch, if the note is not converted we set it to 0 and remove it later
                    passedEvent.pitch = conversions.get(pitch, 0)

                # Save the discrepancies between "expert" and "animation" for debugging
                if debug and hasAnimation:
                    PhaseShiftConverter.EXPERT.append(sorted(list(set(conversions.values()))))
                    PhaseShiftConverter.ANIM.append(sorted(list(set(animConversions.values()))))
                    PhaseShiftConverter.EXPERT_KEY.append(sorted(list(conversions.keys())))
                    PhaseShiftConverter.ANIM_KEY.append(sorted(list(animConversions.keys())))

                # Remove finished events
                notesOn = {k: v for k, v in notesOn.items() if v.end > event.start}

            # Keep track of the notes currently playing
            # don't register duplicate note on the same location
            if notePitch not in notesOn:
                notesOn[notePitch] = event

        if debug:
            self._plotDebugMatrices()
        # Remove empty events with a pitch set to 0 from the convertPitches method:
        track.notes = [note for note in track.notes if note.pitch != 0]  # TODO change the method to be pure?

    def convertPitches(self, pitches):
        """
        Return a mapping converting the notes from a list of simultaneous events to standard pitches.
        The events which should be removed are not mapped.

        There are discrepancies between expert and animation annotations, tries to correct them, but return both mapings for debugging.
        If unsure, use the "expert" mapping:
        - The animations are not always provided
        - When provided, the animations are not always accurate
        - The expert annotations have innacuracies to enhance the gameplay
            In RB set
            - Open HH is played as a crash (+2000 occurences)
            - Flam is played as Snare + tom (+2000 occurences)
            In YT set
            - The annimations seems to have been automatically created from the expert with errors
            In CC set
            - Open HH is played as a crash (+2000 occurences)
        """

        # Get the animation and expert notes mapped to standard events
        animation = config.getPitchesRemap(pitches, ANIMATIONS_MIDI)
        expert = config.getPitchesRemap(pitches, EXPERT_MIDI)
        # Reduce the vocabulary to only 5 classes to remove ambiguity
        animation = {k: MIDI_REDUCED_5[v] for k, v in animation.items() if v in MIDI_REDUCED_5}
        expert = {k: MIDI_REDUCED_5[v] for k, v in expert.items() if v in MIDI_REDUCED_5}
        animationValues = set(animation.values())
        expertValues = set(expert.values())

        # Correct "animation" and "expert" notes mismatch to get more precise annotations
        # TODO add the conditions in an external mapping
        for k, v in expert.items():
            # Animations have only simplified BD
            if v == 35 and 35 not in animationValues:
                animation[k] = v

            # Real open HH on pads CY
            if v == 49 and 49 not in animationValues and 42 in animationValues:
                expert[k] = 42

            # Real double CY on pads CY + HH
            # TODO: add "49 in expertValues"?
            if v == 42 and 42 not in animationValues and 49 in animationValues:
                expert[k] = 49

            # Real flam on SD on pads TT + SD
            if v == 47 and 47 not in animationValues and 38 in animationValues:
                expert[k] = 38

        delAnim = []
        for k, v in animation.items():
            # False positive BD in animation
            if v == 35 and 35 not in expertValues:
                delAnim.append(k)

            # Real TT on RD because of issue? (converted to CY with 5 classes vocabulary)
            if v == 49 and 49 not in expertValues and 47 in expertValues:
                animation[k] = 47
        for k in delAnim:
            del animation[k]

        # Outro of the track not annotated for gameplay but annotated for annimation
        # TODO Correction not compatible with the "false positive BD"
        RB = False
        if RB and len(expert) == 0 and len(animation) > 0:
            expert = animation

        return animation, expert

    def _plotDebugMatrices(self):
        """
        Plot expert and anim mismatches
        """
        logging.debug("ratio of tracks with annimation", np.mean(PhaseShiftConverter.HAS_ANIM))

        difference = defaultdict(lambda: defaultdict(int))
        for i in range(len(PhaseShiftConverter.EXPERT)):
            if PhaseShiftConverter.EXPERT[i] != PhaseShiftConverter.ANIM[i]:  # and len(PhaseShiftConverter.ANIM[i]) > 0
                # difference[str(PhaseShiftConverter.ANIM_KEY[i])][str(PhaseShiftConverter.EXPERT_KEY[i])] += 1
                difference[str(PhaseShiftConverter.ANIM[i])][str(PhaseShiftConverter.EXPERT[i])] += 1
        df = pd.DataFrame(difference)
        # df = df.div(df.sum(axis=1), axis=0)  # norm
        # df.sort_index(level=0, ascending=True, inplace=True)
        df = df.fillna(0)
        logging.debug(
            "Expert / Anim matching ratio", len(PhaseShiftConverter.EXPERT) / (df._values.sum() + len(PhaseShiftConverter.EXPERT)),
        )
        plt.ion()
        plt.show()

        plt.pcolor(df)

        plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
        plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns, rotation=90)
        plt.ylabel("Expert")
        plt.xlabel("Anim")

        # Change minor ticks for grid
        # plt.yticks(np.arange(0, len(df.index), 1), df.index, minor=True)
        # plt.xticks(np.arange(0, len(df.columns), 1), df.columns, minor=True)
        plt.grid(color="grey", linestyle="--", linewidth=1)
        plt.draw()

        plt.pause(0.001)
