import logging
import os
from bisect import bisect_left

import librosa
import madmom
import matplotlib.pyplot as plt
import mir_eval
import numpy as np
import pandas as pd
import pretty_midi
import tapcorrect
from numpy.lib.function_base import quantile
from scipy.interpolate import interp1d

from adtof import config
from adtof.converters.converter import Converter
from adtof.io.midiProxy import PrettyMidiWrapper
from adtof.io.textReader import TextReader


class CorrectAlignmentConverter(Converter):
    """
    Converter which tries to align the midi file to the music file as close as possible 
    by looking at the difference between annotated MIDI beats and estimated beats from Madmom
    """

    def convert(
        self,
        refBeatInput,
        refBeatActivationInput,
        missalignedMidiInput,
        alignedDrumTextOutput,
        alignedBeatTextOutput,
        alignedMidiOutput,
        deviationDerivation="act",
        thresholdFMeasure=0.9,
        sampleRate=100,
        fftSize=2048,
        thresholdCorrectionWindow=0.1,
        smoothingCorrectionWindow=5,
    ):
        """        
        # TODO: possibility to compute a dynamic smoothing window depending on the variability of the offset
        # TODO: utiliser un jeu de données non aligné pour le test

        Parameters
        ----------
        deviationDerivation : str, optional
            Select the method used to estimate the deviation of the notes. 
            "act" to use the method in [TOWARDS AUTOMATICALLY CORRECTING TAPPED BEAT ANNOTATIONS FOR MUSIC RECORDINGS, imsir 2019]
            "track" to use the difference between the annotated and tracked beats (after Madmom's DBNDownBeatTrackingProcessor) averaged on a small window.
            by default "act"
        thresholdFMeasure : float, optional
            ThresholdFMeasure = the limit at which the track will not get rendered because there are too many beats missed, by default 0.9
            The score uses the F-Measure with a size equal to the fftSize
        thresholdCorrectionWindow : float, optional
            Threshold below which the estimated beat (or activation peak) is a match with the neighbooring annotated beat, by default 0.05
        smoothingCorrectionWindow : int, optional
            [description], by default 5
        """
        # Get annotations and estimation data used as reference.
        # kicks_midi = [note.start for note in midi.instruments[0].notes if note.pitch == 36]
        # kicks_audio = tr.getOnsets(alignedDrumsInput, separated=True)[36]
        midi = PrettyMidiWrapper(missalignedMidiInput)
        beats_midi, beatIdx = midi.get_beats_with_index()
        tr = TextReader()
        beats_audio = [el["time"] for el in tr.getOnsets(refBeatInput, mappingDictionaries=[], group=False)]

        # Compute the dynamic offset for the best alignment
        if deviationDerivation == "act":
            beatAct = np.load(refBeatActivationInput, allow_pickle=True)
            correction = self.computeDNNActivationDeviation(beats_midi, beatAct, matchWindow=thresholdCorrectionWindow)
        elif deviationDerivation == "track":
            correction = self.computeTrackedBeatsDeviation(
                beats_midi, beats_audio, matchWindow=thresholdCorrectionWindow, smoothWindow=smoothingCorrectionWindow
            )
        else:
            raise Exception("deviationDerivation is set to an unknown value")
        # self.plot([correctionAct, correctionTrack], ["activation", "tracked onset"], refBeatInput)

        # Apply the dynamic offset to beat and notes
        correctedBeatTimes = self.setDynamicOffset(correction, beats_midi, thresholdCorrectionWindow)

        # Measure if the annotations are of good quality and do not writte the output if needed.
        quality = self.getAnnotationsQuality(correctedBeatTimes, beats_audio, sampleRate, fftSize)
        if quality < thresholdFMeasure:
            logging.error(
                "Not enough overlap between track's estimated and annotated beats to ensure alignment (overlap of " + str(quality) + "%)"
            )
            return quality

        # Merging the drum tracks and getting the events
        if len(midi.instruments) == 0:  # If there is no drums
            logging.error("No drum track in the midi.")
            return quality
        elif len(midi.instruments) > 1:  # There are multiple drums
            logging.info("multiple drums tracks on the midi file. They are merged : " + str(midi.instruments))
        drumsPitches = [
            note.pitch for instrument in midi.instruments for note in instrument.notes
        ]  # [note.pitch for note in midi.instruments[0].notes]
        drumstimes = [
            note.start for instrument in midi.instruments for note in instrument.notes
        ]  # [note.start for note in midi.instruments[0].notes]

        # writte the output
        correctedDrumsTimes = self.setDynamicOffset(correction, drumstimes, thresholdCorrectionWindow)
        tr.writteBeats(alignedDrumTextOutput, [(correctedDrumsTimes[i], drumsPitches[i]) for i in range(len(correctedDrumsTimes))])
        tr.writteBeats(alignedBeatTextOutput, [(correctedBeatTimes[i], beatIdx[i]) for i in range(len(correctedBeatTimes))])
        newMidi = PrettyMidiWrapper.fromListOfNotes(
            [(correctedDrumsTimes[i], drumsPitches[i]) for i in range(len(correctedDrumsTimes))],
            beats=[(correctedBeatTimes[i], beatIdx[i]) for i in range(len(correctedBeatTimes))],
        )
        newMidi.write(alignedMidiOutput)

        # # TESSSSST
        # correctedDrumsTimes = self.setDynamicOffset(correctionAct, drumstimes, thresholdCorrectionWindow, interpolation="linear")
        # newMidi = PrettyMidiWrapper.fromListOfNotes(
        #     [(correctedDrumsTimes[i], drumsPitches[i]) for i in range(len(correctedDrumsTimes))],
        #     beats=[(correctedBeatTimes[i], beatIdx[i]) for i in range(len(correctedBeatTimes))],
        # )
        # newMidi.write(alignedMidiOutput[:-5] + "test.midi")

        return quality

    def getAnnotationsQuality(self, onsetsA, onsetsB, sampleRate, fftSize):
        """
        Estimate the quality of the annotations with the f_measure with a tolerance computed depending on the FFT size for training
        # TODO: see if it is possible to compute a correlation score betweeen annotations and estimations which is not F-measure
        """
        # For the tolerance window, either we want to ensure that the annotation is in the same sample than the actual onset
        # Or we want to make sure that the fft overlaps with the actual onset
        # toleranceWindow = 1 / (sampleRate * 2)
        toleranceWindow = (fftSize / 44100) * 0.8 / 2
        f, p, r = mir_eval.onset.f_measure(np.array(onsetsA), np.array(onsetsB), window=toleranceWindow)
        # TODO see std ?
        return f

    def setDynamicOffset(self, offset, onsets, maxOffsetThreshold, interpolation="cubic"):
        """
        Shift the onsets with the offset linearly interpolated
        """
        try:
            x = [o["time"] for o in offset]
            y = [o["diff"] for o in offset]
            interpolation = interp1d(x, y, kind="linear", fill_value="extrapolate")(onsets)
            # self.plot(offset, interp1d(x, y, kind="linear", fill_value="extrapolate"))
        except Exception as e:
            logging.error("Interpolation of the annotation offset failed", str(e))
            return []

        if max(np.abs(interpolation)) > maxOffsetThreshold:
            logging.error("Extrapolation of annotations offset seems too extreme " + max(np.abs(interpolation)))
            return []

        converted = onsets - interpolation
        converted[converted < 0] = 0
        return converted

    def computeDNNActivationDeviation(self, beats_midi, act, fs_act=100, matchWindow=0.05, lambda_transition=0.1):
        """TODO

        Parameters
        ----------
        beats_midi : [type]
            [description]
        act : [type]
            [description]
        fs_act : int, optional
            [description], by default 100
        matchWindow : float, optional
            [description], by default 0.05
        lambda_transition : float, optional
            [description], by default 0.1

        Returns
        -------
        [type]
            [description]
        """
        # compute deviation matrix
        max_deviation = int(matchWindow * fs_act)
        D_pre, list_iois_pre = tapcorrect.tapcorrection.compute_deviation_matrix(act, beats_midi, fs_act, max_deviation)
        # compute deviation sequence
        dev_sequence = tapcorrect.tapcorrection.compute_score_maximizing_dev_sequence(D_pre, lambda_transition)
        final_beat_times, mu, sigma = tapcorrect.tapcorrection.convert_dev_sequence_to_corrected_tap_times(
            dev_sequence, beats_midi, max_deviation, fs_act
        )
        return [{"time": beats_midi[i], "diff": beats_midi[i] - final_beat_times[i]} for i in range(len(beats_midi))]

    def computeTrackedBeatsDeviation(self, beats_midi, beats_audio, matchWindow=0.05, smoothWindow=5):
        """Compute the difference between all the tuples, smoothed on a 5s windows by default

        Args:
            tuplesThresholded ([(float, float)]): list of neighboor positions
            smoothWindow (int, optional): [description]. Defaults to 5.

        Returns:
            [type]: [description]
        """
        # Get the list of unique match between annotations and estimations
        matches = self.getEventsMatch(beats_midi, beats_audio, matchWindow)

        # annotations - estimations =
        diff = np.array([a - b for a, b in matches])
        # averagedDiff = [
        #     np.mean(diff[np.logical_and(tuplesThresholded[:, 0] > a - smoothWindow, tuplesThresholded[:, 0] < a + smoothWindow)])
        #     for a, b in tuplesThresholded
        # ]
        weightedAverage = []
        for a, b in matches:
            mask = np.logical_and(matches[:, 0] > a - smoothWindow, matches[:, 0] < a + smoothWindow)
            localDiff = diff[mask]
            weights = smoothWindow - np.abs(a - matches[:, 0][mask])
            weightedAverage.append(np.average(localDiff, weights=weights))

        # plt.plot([a for a, b in tuplesThresholded], diff)
        # plt.plot([a for a, b in tuplesThresholded], averagedDiff)
        # plt.plot([a for a, b in tuplesThresholded], weightedAverage)
        # plt.show()
        return [{"time": matches[i][0], "diff": weightedAverage[i]} for i in range(len(weightedAverage))]

    def plot(self, correction, interp):
        correction = [e for e in correction if e["time"] <= 10]
        interValues = np.arange(0, 10, 0.05)

        plt.figure(figsize=(10, 4))
        plt.plot([t["time"] for t in correction], [t["diff"] for t in correction], "+", zorder=5, label="Beats deviation")
        plt.plot(interValues, interp(interValues), "--", label="Interpolation")
        plt.legend()
        plt.ylabel("Deviation (s)")
        plt.xlabel("Position (s)")
        plt.savefig("alignment.png", dpi=600)
        # plt.show()

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
