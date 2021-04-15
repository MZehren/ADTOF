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

    def decode(self, line, sep):
        time, pitch = line.replace("\r\n", "").replace("\n", "").split(sep)
        time = time.replace(" ", "")
        pitch = pitch.replace(" ", "")
        time = float(time)
        pitch = self.castInt(pitch)

        return (time, pitch)

    def getOnsets(
        self,
        txtFilePath,
        mappingDictionaries=[config.RBMA_MIDI_8, config.MIDI_REDUCED_5],
        group=True,
        sep="\t",
        removeIfClassUnknown=False,
        **kwargs
    ):
        """
            Parse the text file following Mirex encoding:
            [time]\t[class]\n 

        Args:
            txtFilePath (string): path to the text file.
            mappingDictionaries (list, optional): Mapping to convert the class of events into other classes (ie: config.RBMA_MIDI_8 mapping class 0 to 35). 
            It is a list of dictionaries to chain multiple mappings one after the other.
            group (bool, optional): If true, returns {class: [position]}. Else, returns [{position: class}] . Defaults to True.

        Returns:
            Dictionary of the shape {class: [positions]}
        """
        events = []
        with open(txtFilePath, "r") as f:
            for line in f:
                try:
                    time, pitch = self.decode(line, sep)
                except Exception as e:
                    print("Line couldn't be decoded, passing.", repr(line), str(e))
                    continue

                pitch = config.remapPitches(pitch, mappingDictionaries, removeIfUnknown=removeIfClassUnknown)
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
