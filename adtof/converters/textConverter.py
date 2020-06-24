#!/usr/bin/env python

import argparse
import json
import logging
import os
import sys
import warnings
from collections import defaultdict

import pkg_resources

from adtof.config import MDBS_MIDI, MIDI_REDUCED_3, RBMA_MIDI
from adtof.io.midiProxy import MidiProxy
from adtof.converters.converter import Converter


class TextConverter(Converter):
    """
    Convert the text format from rbma_13 and MDBDrums to midi
    """

    def castInt(self, s):
        """
        Try to convert a string in int if possible
        """
        try:
            casted = int(s)
            return casted
        except ValueError:
            return s

    def getOnsets(self, txtFilePath, separated=False):
        """
        Parse the file and return a list of {"time": int, "pitch": int}

        separated= return {pitch: [events]} instead of a flat array
        """
        events = []
        with open(txtFilePath, "r") as f:
            for line in f:
                time, pitch = line.replace(" ", "").replace("\r\n", "").replace("\n", "").split("\t")
                time = float(time)
                pitch = self.castInt(pitch)

                if pitch in MDBS_MIDI:
                    pitch = MDBS_MIDI[pitch]
                elif pitch in RBMA_MIDI:
                    pitch = RBMA_MIDI
                pitch = MIDI_REDUCED_3[pitch]

                events.append({"time": time, "pitch": pitch})

        if separated:
            result = defaultdict(list)
            for e in events:
                result[e["pitch"]].append(e["time"])
            return result

        return events

    def convert(self, txtFilePath, outputName=None):
        """
        Convert a text file of the shape:
        float string/int\n

        returns and save a midi object 
        """
        # read the file
        events = self.getOnsets(txtFilePath)

        # create the midi
        midi = MidiProxy(None)
        for event in events:
            midi.addNote(e["time"], e["pitch"])

        # return
        if outputName:
            midi.save(outputName)
        return midi
