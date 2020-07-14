#!/usr/bin/env python

import argparse
import json
import logging
import os
import sys
import warnings
from collections import defaultdict

import pkg_resources

from adtof import config
from adtof.io.midiProxy import MidiProxy


class TextReader(object):
    """
    Convert the text format from rbma_13 and MDBDrums to midi
    """

    def castInt(self, s):
        """
        Try to convert a string in int if possible
        """
        try:
            casted = int(float(s))
            return casted
        except ValueError:
            return s

    def getOnsets(self, txtFilePath, mappingDictionaries=[config.RBMA_MIDI_8, config.MIDI_REDUCED_5], group=True):
        """
        Parse the file and return a list of {"time": int, "pitch": int}

        separated= return {pitch: [events]} instead of a flat array
        """
        events = []
        with open(txtFilePath, "r") as f:
            for line in f:
                try:
                    time, pitch = line.replace(" ", "").replace("\r\n", "").replace("\n", "").split("\t")
                except:
                    print("Line couldn't be decoded, passing.", line)
                    continue
                time = float(time)

                pitch = self.castInt(pitch)
                pitch = config.remapPitches(pitch, mappingDictionaries, removeIfUnknown=False)
                if pitch != None:
                    events.append({"time": time, "pitch": pitch})

        if group is False:
            return events
        result = defaultdict(list)
        for e in events:
            result[e["pitch"]].append(e["time"])
        return result

    def writteBeats(self, path, beats):
        """

        """
        with open(path, "w") as f:
            f.write("\n".join([str(time) + "\t" + str(beatNumber) for time, beatNumber in beats]))
