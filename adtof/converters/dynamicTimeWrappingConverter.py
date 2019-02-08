from bisect import bisect_left

import librosa
import numpy as np

from adtof.converters import Converter
from adtof.io import MidoProxy
from dtw import dtw


class DynamicTimeWrappingConverter(Converter):
    """
    Converter which tries to align the music file to the midi file as close as possible
    """

    def parseToDense(self, sparseArray, sr, length):
        dense = np.zeros(length)

        for event in sparseArray:
            index = int(np.round(event * sr))
            dense[index] = 1

        return dense

    def convert(self, inputMusicPath, inputMidiPath, outputName):
        midi = MidoProxy(inputMidiPath)
        y, sr = librosa.load(inputMusicPath)

        midiOnsets = midi.getOnsets()
        musicOnsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")

        self.getError(midiOnsets, musicOnsets)

        # midiDense = self.parseToDense(midiOnsets, sr, len(y)).reshape(-1, 1)
        # musicDense = self.parseToDense(musicOnsets, sr, len(y)).reshape(-1, 1)

        # d, cost_matrix, acc_cost_matrix, path = dtw(midiDense[50000:51000], musicDense[50000:51000], dist=lambda x, y: (x - y)**2)
        # pass

    def getError(self, onsetsA, onsetsB, maxThreshold=0.02):
        """
        Compute the average error of onsets close to each other bellow a maxThreshold
        """
        tuples = [(onsetA, self.findNeighboor(onsetsB, onsetA)) for onsetA in onsetsA]
        tuplesThresholded = [(onsets[0], onsets[1]) for onsets in tuples if np.abs(onsets[0] - onsets[1]) < maxThreshold]
        diff = [onsets[0] - onsets[1] for onsets in tuplesThresholded]
        error = np.mean(np.abs(diff))
        correction = np.mean(diff)
        return error

    def findNeighboor(self, grid, value):
        """
        Assumes grid is sorted. Returns closest value to value.
        If two numbers are equally close, return the smallest number.
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

    # def isConvertible(self, folder, inputFile):
    #     files = os.listdir(folderPath)
    #     return inputFile == PhaseShiftConverter.PS_MIDI_NAME and  "song.ini" in files and ("song.ogg" in files or "guitar.ogg" in files)
