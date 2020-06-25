import logging
import os
from bisect import bisect_left

import librosa
import madmom
import numpy as np
import pandas as pd
import pretty_midi
from mir_eval.onset import f_measure

from adtof import config
from adtof.converters.converter import Converter
from adtof.io.textReader import TextReader
import matplotlib.pyplot as plt


class CorrectAlignmentConverter(Converter):
    """
    Converter which tries to align the midi file to the music file as close as possible 
    by looking at the difference between MIDI note_on events and librosa.onsets
    """

    def convert(self, alignedDrumsInput, alignedBeatInput, missalignedMidiInput, alignedDrumOutput, alignedBeatOutput):
        # get midi kicks
        midi = pretty_midi.PrettyMIDI(missalignedMidiInput)
        kicks_midi = [note.start for note in midi.instruments[0].notes if note.pitch == 36]

        # get midi beats
        beats_midi = midi.get_beats()

        tr = TextReader()
        # get audio beats
        musicBeats = [el["time"] for el in tr.getOnsets(alignedBeatInput, convertPitches=False)]

        # get audio estimated kicks
        kicks_audio = tr.getOnsets(alignedDrumsInput, separated=True)[36]

        self.computeAlignment(beats_midi, musicBeats)
        error, offset, diffPlayback = self.computeAlignment(kicks_midi, kicks_audio)  # musicBeats[:, 0], midiBeats)
        # f, p, r = f_measure(np.array(kicks_midi) * diffPlayback - offset, np.array(kicks_audio), window=0.01)
        # pd.DataFrame(
        #     {"MAE": [error], "offset": [offset], "playback": [diffPlayback], "F-measure": [f], "precision": [p], "recall": [r]}
        # ).to_csv(outputPath + ".txt")

    def computeAlignment(self, onsetsA, onsetsB, maxThreshold=0.05):
        """
        Compute the average error of onsets close to each other bellow a maxThreshold

        the correction tells how much you should shift B to align with A
        or -correction you need to shift A to align with B 
        """
        # Get the alignment between the notes and the onsets
        tuples = [(onsetA, self.findNeighboor(onsetsB, onsetA)) for onsetA in onsetsA]
        tuplesThresholded = np.array([[onsets[0], onsets[1]] for onsets in tuples if np.abs(onsets[0] - onsets[1]) < maxThreshold])

        if len(tuplesThresholded) < 2:
            return

        self.computeDynamicOffset(tuplesThresholded)

    def computeDynamicOffset(self, tuplesThresholded, smoothWindow=5):
        """
        Compute the difference between all the tuples, smoothed on 10s windows
        """
        # annotations - estimations =
        diff = np.array([a - b for a, b in tuplesThresholded])
        averagedDiff = [
            np.mean(diff[np.logical_and(tuplesThresholded[:, 0] > a - smoothWindow, tuplesThresholded[:, 0] < a + smoothWindow)])
            for a, b in tuplesThresholded
        ]
        weightedAverage = []
        for a, b in tuplesThresholded:
            mask = np.logical_and(tuplesThresholded[:, 0] > a - smoothWindow, tuplesThresholded[:, 0] < a + smoothWindow)
            localDiff = diff[mask]
            weights = smoothWindow - np.abs(a - tuplesThresholded[:, 0][mask])
            weightedAverage.append(np.average(localDiff, weights=weights))

        # plt.plot([a for a, b in tuplesThresholded], diff)
        # plt.plot([a for a, b in tuplesThresholded], averagedDiff)
        plt.plot([a for a, b in tuplesThresholded], weightedAverage)
        # plt.show()

    def computeOffset(self, tuplesThresholded):
        """
        Compute on average the offset of the best alignment
        """
        diff = [a - b for a, b in tuplesThresholded]
        offset = np.mean(diff)
        previousError = np.mean(np.abs(diff))
        remainingError = np.mean(np.abs(diff - offset))  # TODO: MAE vs RMSE?
        if previousError < remainingError:
            logging.error("Check the computation: " + str(remainingError - previousError))
        return offset, remainingError

    def computeRate(self, tuplesThresholded):
        """
        Compute on average the playback speed of the best alignment
        """
        playback = np.mean(np.diff(tuplesThresholded[:, 1])) / np.mean(np.diff(tuplesThresholded[:, 0]))
        tuplesThresholded = [(a, b / playback) for a, b in tuplesThresholded]
        offset, remainingError = self.computeOffset(tuplesThresholded)
        return playback, offset, remainingError

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
