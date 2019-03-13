import os
from bisect import bisect_left

import librosa
import numpy as np

from adtof.converters import Converter
from adtof.io import MidoProxy
from adtof.converters import PhaseShiftConverter

# from dtw import dtw


class OnsetsAlignementConverter(Converter):
    """
    Converter which tries to align the midi file to the music file as close as possible 
    by looking at the difference between MIDI note_on events and librosa.onsets
    """

    def convertRecursive(self, rootFodler, outputName, midiCandidates=None, musicCandidates=None):
        converted = 0
        failed = 0
        if midiCandidates is None:
            midiCandidates = PhaseShiftConverter.PS_MIDI_NAMES
        if musicCandidates is None:
            musicCandidates = PhaseShiftConverter.PS_MUSIC_NAMES
        for root, _, files in os.walk(rootFodler):
            midiFiles = [file for file in midiCandidates if file in files]
            musicFiles = [file for file in musicCandidates if file in files]

            if midiFiles and musicFiles:
                try:
                    self.convert(os.path.join(root, musicFiles[0]), os.path.join(root, midiFiles[0]), os.path.join(root, outputName))
                    print("converted", root)
                    converted += 1
                except ValueError:
                    print(ValueError)
                    failed += 1
        print("converted", converted, "failed", failed)

    def convert(self, inputMusicPath, inputMidiPath, outputPath):
        midi = MidoProxy(inputMidiPath)
        midiOnsets = midi.getOnsets()

        y, sr = librosa.load(inputMusicPath)
        musicOnsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")

        error, correction = self.getError(midiOnsets, musicOnsets)


        midi.addDelay(correction)
        midi.save(outputPath)

    def getError(self, onsetsA, onsetsB, maxThreshold=0.02):
        """
        Compute the average error of onsets close to each other bellow a maxThreshold

        """
        tuples = [(onsetA, self.findNeighboor(onsetsB, onsetA)) for onsetA in onsetsA]
        tuplesThresholded = [(onsets[0], onsets[1]) for onsets in tuples if np.abs(onsets[0] - onsets[1]) < maxThreshold]
        diff = [onsets[0] - onsets[1] for onsets in tuplesThresholded]
        error = np.mean(np.abs(diff))
        correction = np.mean(diff)
        return error, correction

    def findNeighboor(self, grid, value):
        """
        Assumes grid is sorted. Returns closest grid tick to each values.
        If two ticks are equally close, return the smallest one.
        """
        pos = bisect_left(grid, value)

        if pos == 0:
            return grid[0]
        if pos == len(grid):
            return grid[-1]
        before = grid[pos - 1]
        after = grid[pos]
        if after - value < value - before:
            return after
        else:
            return before
