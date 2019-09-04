#!/usr/bin/env python

import argparse
import json
import logging
import os
import sys
import warnings

import pkg_resources

from adtof.io.converters import Converter
from adtof.io import MidiProxy


class TextConverter(Converter):
    """
    Convert the text format from rbma_13 and MDBDrums to midi
    """
    ressource = pkg_resources.resource_string(__name__, "mappingDictionaries/rbma13ToMidi.json").decode()
    RBMA_MIDI = {int(key): int(value) for key, value in json.loads(ressource).items()}

    ressource = pkg_resources.resource_string(__name__, "mappingDictionaries/MDBDrumsToMidi.json").decode()
    MDBS_MIDI = {key: int(value) for key, value in json.loads(ressource).items()}

    _m2r = pkg_resources.resource_string(__name__, "mappingDictionaries/standardMidiToReduced.json").decode()
    REDUCED_MIDI = {int(key): int(value) for key, value in json.loads(_m2r).items()}

    def convert(self, txtFilePath, outputName=None):
        """
        Read the ini file and convert the midi file to the standard events
        """
        # read the file
        events = []
        with open(txtFilePath, "r") as f:
            for line in f:
                time, pitch = line.replace(" ", "").replace("\r\n", "").replace("\n", "").split("\t")
                
                time = float(time)
                pitch = self.MDBS_MIDI[pitch] if pitch in self.MDBS_MIDI else self.RBMA_MIDI[int(pitch)]
                events.append([time, pitch])

        # create the midi
        midi = MidiProxy(None)
        for time, pitch in events:
            midi.addNote(time, pitch)
        
        #return
        if outputName:
            midi.save(outputName)
        return midi
