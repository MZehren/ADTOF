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
import pandas as pd
from adtof import config
from adtof.ressources.instrumentsMapping import ANIMATIONS_MIDI, ANIMATIONS_VELOCITY, EXPERT_MIDI, MIDI_REDUCED_5, MIDI_REDUCED_7, MIDI_REDUCED_8, DEFAULT_VELOCITY
from adtof.converters.converter import Converter
from adtof.io.midiProxy import PrettyMidiWrapper
from adtof.model.eval import plotPseudoConfusionMatrices, plotPseudoConfusionMatricesFromDense


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

    def convert(self, inputFolder, outputMidiPath, outputRawMidiPath, outputAudioPath, addDelay=True, **kwargs):
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
        debug = self.cleanMidi(midi, **kwargs)

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
        if config.checkPathExists(outputAudioPath):  # File already exists
            return
        if len(audioFiles) == 1 and delay == 0:  # If it's only one file, copy it
            copyfile(os.path.join(audioFiles[0]), outputAudioPath)
        else:  # If it's multiple audio file, merge them
            # Not sure of those parameters, it seems that libopus might have a better quality, and increasing the bitrate helps reducing the difference from the merged sources
            outputArgs = {"c:a": "libopus", "b:a": "256k"}
            if delay > 0:
                outputArgs["ss"] = str(delay)
            elif delay < 0:
                raise ValueError("midi delay is negative")

            ffmpeg.filter([ffmpeg.input(audioFile) for audioFile in audioFiles], "amix", inputs=len(audioFiles)).filter("volume", len(audioFiles)).output(
                outputAudioPath, **outputArgs
            ).global_args("-loglevel", "error").run(overwrite_output=True)

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
            "name": ini["artist"].replace("/", "-") + " - " + ini["name"].replace("/", "-") if "name" in ini and "artist" in ini else os.path.basename(inputFolder),
            "genre": ini["genre"] if "genre" in ini else None,
            "pro_drums": ini["pro_drums"] if "pro_drums" in ini else None,
        }
        return meta

    def getTrackName(self, inputFolder):
        return self.getMetaInfo(inputFolder)["name"]

    def getFirstOccurenceOfIntersection(self, A: list, B: list):
        """Util function to select the first file"""

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
        audioFiles = [file for file in files if ".ogg" in file and file != "preview.ogg"]  # getFirstOccurenceOfIntersection(PhaseShiftConverter.PS_AUDIO_NAMES, files)
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

    def cleanMidi(self, midi, **kwargs):
        """
        Clean the midi file to a standard file with standard pitches, only one drum track, and remove the duplicated events.

        Arguments:
            midi: midi file from python-midi
        """
        # Remove the non-drum tracks
        self.removeUnwantedTracks(midi)

        # Convert the pitches
        self.convertTrack(midi, **kwargs)

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
            raise ValueError("ERROR: No drum track in the MIDI file " + self.name)
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

    def convertTrack(self, midi, useAnimation=False, task="5", debug=False):
        """Convert the pitches from the midi tracks

        Parameters
        ----------
        midi : the midi object to convert
        debug: If plotting the "animation" and "expert" mismatch
        """
        from pretty_midi.containers import Note

        if task == "5":
            map = MIDI_REDUCED_5
        elif task == "7":
            map = MIDI_REDUCED_7
        else:
            raise ValueError("task should be 5 or 7")

        # Add the discoFlip events as notes
        track = midi.instruments[0]
        events = [note for note in track.notes] + midi.discoFlip
        events.sort(key=lambda x: x.start)
        # DIRTY code: Adding a fake event at the end so the for loop works
        events += [Note(0, 0, events[-1].start + 1, events[-1].end + 1)]

        # Check if the track has "annimation" annotations or only "expert" ones
        hasAnimation = len([1 for event in track.notes if event.pitch in ANIMATIONS_MIDI]) > 100
        if useAnimation and not hasAnimation:
            raise ValueError("The track has no animation annotations")
        if debug:
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
                animConversions, conversions, velocity, invertTiming = self.convertPitches(notesOn.keys(), map)
                for pitch, passedEvent in notesOn.items():
                    # convert the pitch, if the note is not converted we set it to 0 and remove it later
                    if useAnimation:
                        passedEvent.pitch = animConversions.get(pitch, 0)
                        passedEvent.velocity = velocity.get(pitch, DEFAULT_VELOCITY)
                        passedEvent.invert = invertTiming.get(pitch, False)
                    else:
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
            # TODO: there are notes duplicated after the conversion, depending on the mapping (e.g. 47+45 happening together tranlated to two 47 notes)
            if notePitch not in notesOn:
                notesOn[notePitch] = event

        if debug:
            # self._plotDebugMatrices()
            # plotPseudoConfusionMatricesFromDense(PhaseShiftConverter.EXPERT, PhaseShiftConverter.ANIM, ylabel="Expert", xlabel="Anim", distanceThreshold=0.01)
            pass

        # Remove empty events with a pitch set to 0 from the convertPitches method:
        track.notes = [note for note in track.notes if note.pitch != 0]  # TODO change the method to be pure?

        # Set the start of HH pedal event to the closing time
        # for note in track.notes:
        #     if hasattr(note, "invert") and note.invert:
        #         note.start = note.end

    def convertPitches(self, pitches, map):
        """
        Return a mapping converting the notes from a list of simultaneous events to standard pitches.
        The events which should be removed are not mapped.

        There are discrepancies between expert and animation annotations, tries to correct them, but return both mapings for debugging.
        If unsure, use the "expert" mapping:
        - The animations are not always provided
        - When provided, the animations are not always accurate
        - The expert annotations have innacuracies to enhance the gameplay
            In RB set
            - Open HH is played as an expert CY but animation OH (+2000 occurences)
            - Flam is played as expert SN + TT but animation SN (+2000 occurences)
            In YT set: I recommend using expert with automatic correction from animation
            - The annimations seems to have been automatically created from the expert with errors (i.e.: no left BD, but other errors too making me not willing to trust this set)
            - The RD can be wrongly annotated and used for a third crash
            - The OH can be wrongly annotated as a CH
            - The ghost note have a low recall (often annotated as standard velocity), and even not a perfect precision (some normal notes are annotated as ghost)
            In CC set
            - Open HH is played as a crash (+2000 occurences)
        """

        # Get the animation and expert notes mapped to standard events
        animation = config.getPitchesRemap(pitches, ANIMATIONS_MIDI)
        expert = config.getPitchesRemap(pitches, EXPERT_MIDI)
        velocity = config.getPitchesRemap(pitches, ANIMATIONS_VELOCITY)
        invertTiming = {}

        # Reduce the vocabulary to only 5 classes to remove ambiguity
        animation = {k: map[v] for k, v in animation.items() if v in map}
        expert = {k: map[v] for k, v in expert.items() if v in map}
        animationValues = set(animation.values())
        expertValues = set(expert.values())

        # Correct "animation" and "expert" notes mismatch to get more precise annotations
        # TODO add the conditions in an external mapping
        for k, v in expert.items():
            # Animations have only simplified BD
            if v == 35 and 35 not in animationValues:
                animation[k] = v

            # Real open HH on pads CY
            if v == 49 and 49 not in animationValues and 46 in animationValues:
                expert[k] = 42

            # Real double CY on pads CY + HH
            if v == 42 and 42 not in animationValues and 49 in animationValues:
                expert[k] = 49

            # Real flam SD on pads TT + SD
            if v == 47 and 47 not in animationValues and 38 in animationValues:
                expert[k] = 38

        delAnim = []
        for k, v in animation.items():
            # False positive BD in animation
            if v == 35 and 35 not in expertValues:
                delAnim.append(k)

            # Real TT on RD because of issue? (converted to CY with 5 classes vocabulary)
            # TODO double check the validity
            if (v == 49 or v == 51) and 49 not in expertValues and 47 in expertValues:
                animation[k] = 47

            # Flam annotated with both hand on the snare
            # if k == 27 and 26 in animation:

            # HH pedal in anim starts when openning and ends on closing. In MIDI it should only be when closing
            if k == 25:
                invertTiming[k] = True
        for k in delAnim:
            del animation[k]

        # Outro of the track not annotated for gameplay but annotated for annimation
        # TODO Correction not compatible with the "false positive BD"
        RB = False
        if RB and len(expert) == 0 and len(animation) > 0:
            expert = animation

        return animation, expert, velocity, invertTiming
