import logging
import os
from bisect import bisect_left

import librosa
import madmom
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import quantile
import pandas as pd
import pretty_midi
import mir_eval
from scipy.interpolate import interp1d

from adtof import config
from adtof.converters.converter import Converter
from adtof.io.textReader import TextReader
from adtof.io.midiProxy import PrettyMidiWrapper


class CorrectAlignmentConverter(Converter):
    """
    Converter which tries to align the midi file to the music file as close as possible 
    by looking at the difference between annotated MIDI beats and estimated beats from Madmom
    """

    def convert(
        self,
        refBeatInput,
        missalignedMidiInput,
        alignedDrumTextOutput,
        alignedBeatTextOutput,
        alignedMidiOutput,
        thresholdFMeasure=0.9,
        sampleRate=100,
        fftSize=2048,
        thresholdCorrectionWindow=0.05,
        smoothingCorrectionWindow=5,
    ):
        """
        ThresholdFMeasure = the limit at which the track will not get rendered because there are too many beats missed
        """
        # Get annotations and estimation data used as reference.
        # kicks_midi = [note.start for note in midi.instruments[0].notes if note.pitch == 36]
        # kicks_audio = tr.getOnsets(alignedDrumsInput, separated=True)[36]
        midi = PrettyMidiWrapper(missalignedMidiInput)
        beats_midi, beatIdx = midi.get_beats_with_index()
        tr = TextReader()
        beats_audio = [el["time"] for el in tr.getOnsets(refBeatInput, mappingDictionaries=[], group=False)]

        # Get the list of unique match between annotations and estimations
        matches = self.getEventsMatch(beats_midi, beats_audio, thresholdCorrectionWindow)

        # Compute the dynamic offset for the best alignment
        # correction = self.computeAlignment(kicks_midi, kicks_audio)
        correction = self.getDynamicOffset(matches, smoothWindow=smoothingCorrectionWindow)

        # Apply the dynamic offset to beat and notes
        correctedBeatTimes = self.setDynamicOffset(correction, beats_midi, thresholdCorrectionWindow)

        # Measure if the annotations are of good quality and do not writte the output if needed.
        quality = self.getAnnotationsQuality(correctedBeatTimes, beats_audio, sampleRate, fftSize)
        if quality < thresholdFMeasure:
            print("Not enough overlap between track's estimated and annotated beats to ensure alignment (overlap of " + str(quality) + "%)")
            return quality

        # writte the output
        if len(midi.instruments) == 0:  # If there is no drums
            return quality
        drumsPitches = [note.pitch for note in midi.instruments[0].notes]
        drumstimes = [note.start for note in midi.instruments[0].notes]
        correctedDrumsTimes = self.setDynamicOffset(correction, drumstimes, thresholdCorrectionWindow)
        tr.writteBeats(alignedDrumTextOutput, [(correctedDrumsTimes[i], drumsPitches[i]) for i in range(len(correctedDrumsTimes))])
        tr.writteBeats(alignedBeatTextOutput, [(correctedBeatTimes[i], beatIdx[i]) for i in range(len(correctedBeatTimes))])
        newMidi = PrettyMidiWrapper.fromListOfNotes(
            [(correctedDrumsTimes[i], drumsPitches[i]) for i in range(len(correctedDrumsTimes))],
            beats=[(correctedBeatTimes[i], beatIdx[i]) for i in range(len(correctedBeatTimes))],
        )
        newMidi.write(alignedMidiOutput)
        return quality

    def getAnnotationsQuality(self, onsetsA, onsetsB, sampleRate, fftSize):
        """
        Estimate the quality of the annotations with the f_measure based on the 
        """
        # For the tolerance window, either we want to ensure that the annotation is in the same sample than the actual onset
        # Or we want to make sure that the fft overlaps with the actual onset
        # toleranceWindow = 1 / (sampleRate * 2)
        toleranceWindow = (fftSize / 44100) * 0.8
        f, p, r = mir_eval.onset.f_measure(np.array(onsetsA), np.array(onsetsB), window=toleranceWindow)
        # TODO see std ?
        return f

    def setDynamicOffset(self, offset, onsets, maxOffsetThreshold):
        """
        Shift the onsets with the offset linearly interpolated
        """
        try:
            x = [o["time"] for o in offset]
            y = [o["diff"] for o in offset]
            interpolation = interp1d(x, y, kind="cubic", fill_value="extrapolate")(onsets)
        except Exception as e:
            print("Interpolation of the annotation offset failed", str(e))
            return []

        if max(np.abs(interpolation)) > maxOffsetThreshold:
            print("Extrapolation of annotations offset seems too extreme " + max(np.abs(interpolation)))
            return []

        converted = onsets - interpolation
        converted[converted < 0] = 0
        return converted

    def getDynamicOffset(self, tuplesThresholded, smoothWindow=5):
        """Compute the difference between all the tuples, smoothed on a 5s windows by default

        Args:
            tuplesThresholded ([(float, float)]): list of neighboor positions
            smoothWindow (int, optional): [description]. Defaults to 5.

        Returns:
            [type]: [description]
        """
        # annotations - estimations =
        diff = np.array([a - b for a, b in tuplesThresholded])
        # averagedDiff = [
        #     np.mean(diff[np.logical_and(tuplesThresholded[:, 0] > a - smoothWindow, tuplesThresholded[:, 0] < a + smoothWindow)])
        #     for a, b in tuplesThresholded
        # ]
        weightedAverage = []
        for a, b in tuplesThresholded:
            mask = np.logical_and(tuplesThresholded[:, 0] > a - smoothWindow, tuplesThresholded[:, 0] < a + smoothWindow)
            localDiff = diff[mask]
            weights = smoothWindow - np.abs(a - tuplesThresholded[:, 0][mask])
            weightedAverage.append(np.average(localDiff, weights=weights))

        # plt.plot([a for a, b in tuplesThresholded], diff)
        # plt.plot([a for a, b in tuplesThresholded], averagedDiff)
        # plt.plot([a for a, b in tuplesThresholded], weightedAverage)
        # plt.show()
        return [{"time": tuplesThresholded[i][0], "diff": weightedAverage[i]} for i in range(len(weightedAverage))]

    def computeOffset(self, tuplesThresholded):
        """
        Compute on average the offset of the best alignment
        """
        raise DeprecationWarning()
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
        raise DeprecationWarning()
        playback = np.mean(np.diff(tuplesThresholded[:, 1])) / np.mean(np.diff(tuplesThresholded[:, 0]))
        tuplesThresholded = [(a, b / playback) for a, b in tuplesThresholded]
        offset, remainingError = self.computeOffset(tuplesThresholded)
        return playback, offset, remainingError

    def getEventsMatch(self, onsetsA, onsetsB, window):
        """Compute the closest position in B for each element of A

        Args:
            onsetsA ([float]): list of positions in A
            onsetsB ([float]): list of positions in B

        Returns:
            [(float, float)]: list of tuples with each element of A next to the closest element in B
        """
        matchIdx = mir_eval.util.match_events(onsetsA, onsetsB, window)
        matchPositions = np.array([[onsetsA[t[0]], onsetsB[t[1]]] for t in matchIdx])

        # Get the matches with own method (not single matches)
        # tuples = [(onsetA, self.findNeighboor(onsetsB, onsetA)) for onsetA in onsetsA]
        # tuplesThresholded = np.array([[onsets[0], onsets[1]] for onsets in tuples if np.abs(onsets[0] - onsets[1]) < maxThreshold])

        # # Get the matches with DTW (not single matches),
        # #  This is slower and doesn't change the result since we are using euclidean distances on position for the cost matrix
        # _, tuples = librosa.sequence.dtw(X=onsetsA, Y=onsetsB)
        # tuples = [(onsetsA[t[0]], onsetsB[t[1]]) for t in tuples]
        # tuplesThresholded1 = np.array([[onsets[0], onsets[1]] for onsets in tuples if np.abs(onsets[0] - onsets[1]) < maxThreshold])
        # tuplesThresholded1 = tuplesThresholded1[::-1]
        return matchPositions

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
