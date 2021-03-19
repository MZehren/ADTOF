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
        thresholdQuality=0.5,
        sampleRate=100,
        fftSize=2048,
        thresholdCorrectionWindow=0.05,
        smoothingCorrectionWindow=5,
        audioPath=None,
    ):
        """
        # TODO: utiliser un jeu de données non aligné pour le test

        Parameters
        ----------
        deviationDerivation : str, optional
            Select the method used to estimate the deviation of the notes.
            "act" (recommended) to use the method in [TOWARDS AUTOMATICALLY CORRECTING TAPPED BEAT ANNOTATIONS FOR MUSIC RECORDINGS, imsir 2019]
            "track" (not recommended) to use the difference between the annotated and tracked beats (after Madmom's DBNDownBeatTrackingProcessor) averaged on a small window.
            by default "act"
        thresholdQuality : float, optional
            the limit at which the track will get discarded because there are too many beats annotated are not identified and aligned to the computer estimations
            TODO: The score uses either the precision of the beats hit rate, or the value of the beat activation function after the alignment, or the maximum correction offset. 
        thresholdCorrectionWindow : float, optional
            Threshold below which the estimated beat (or activation peak) is a match with the neighbooring annotated beat, by default 0.05
            TODO: see Toward automatically correcting tapped beat... 
            If the value is too large, the quality score will artificially increase, but the beats are going to be aligned to wrong estimations as well. 
        smoothingCorrectionWindow : int, optional
            used only for "track" deviationDerivation, smooth the correction over that many beats (by default 5)
        """
        # Get annotations and estimation data used as reference.
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

        # Apply the dynamic offset to beat
        correctedBeatTimes = self.setDynamicOffset(correction, beats_midi, thresholdCorrectionWindow)

        # Apply the dynamic offset to onsets
        if len(midi.instruments) > 1:  # There are multiple drums
            raise ValueError("multiple drums tracks on the midi file")
            logging.debug("multiple drums tracks on the midi file. They are merged : " + str(midi.instruments))
        drumsPitches = [
            note.pitch for instrument in midi.instruments for note in instrument.notes
        ]  # [note.pitch for note in midi.instruments[0].notes]
        drumstimes = [
            note.start for instrument in midi.instruments for note in instrument.notes
        ]  # [note.start for note in midi.instruments[0].notes]
        correctedDrumsTimes = self.setDynamicOffset(correction, drumstimes, thresholdCorrectionWindow)

        # Measure if the annotations are of good quality.
        midiLimit = midi.get_end_time()
        qualityAct = self.getAnnotationsQualityAct(correctedBeatTimes, correctedDrumsTimes, beatAct, sampleRate)
        qualityHit = self.getAnnotationsQualityHit([t for t in beats_audio if t <= midiLimit], correctedBeatTimes, sampleRate, fftSize)
        # Get the beats with a huge correction which doesn't seem correct (intersecting with a drums onset to remove wrong estimations from madmom)
        drumstimesSet = set(drumstimes)
        fishy_corrections = [c for c in correction if np.abs(c["diff"]) > 0.025 and c["time"] in drumstimesSet]

        # Discard the tracks with a low quality and with extreme corrections (set to 25ms)
        if qualityAct < thresholdQuality:
            # debug
            # print("investigate", os.path.basename(audioPath))
            # print(
            #     "odd_Time",
            #     len([t for t in midi.time_signature_changes if t.numerator != 4]) > 0,
            #     "/ fast_tempo",
            #     max(midi.get_tempo_changes()[1]) > 215,
            # )
            # print("activation_quality", qualityAct, "/ hitRate_quality", qualityHit)
            # correctedDrumsTimesSet = set(correctedDrumsTimes)
            # print("fishy_corrections", [c for c in correction if np.abs(c["diff"]) > 0.03 and c["time"].in(correctedDrumsTimesSet)])

            raise ValueError("Not enough overlap between track's estimated and annotated beats to ensure alignment")
        elif len(fishy_corrections) > 2:  # TODO: Why 2 again?
            # debug
            # for c in fishy_corrections:
            #     print(
            #         "correction",
            #         np.round(c["diff"], decimals=2),
            #         str(int(c["time"] // 60)) + ":" + str(np.round(c["time"] % 60, decimals=2)),
            #     )
            raise ValueError("Extreme correction needed for this track")

        # writte the output
        tr.writteBeats(alignedDrumTextOutput, [(correctedDrumsTimes[i], drumsPitches[i]) for i in range(len(correctedDrumsTimes))])
        tr.writteBeats(alignedBeatTextOutput, [(correctedBeatTimes[i], beatIdx[i]) for i in range(len(correctedBeatTimes))])
        newMidi = PrettyMidiWrapper.fromListOfNotes(
            [(correctedDrumsTimes[i], drumsPitches[i]) for i in range(len(correctedDrumsTimes))],
            beats=[(correctedBeatTimes[i], beatIdx[i]) for i in range(len(correctedBeatTimes))],
        )
        newMidi.write(alignedMidiOutput)

        return qualityAct

    def getAnnotationsQualityAct(self, correctedBeats, correctedOnsets, act, sampleRate, threshold=0.1):
        """
        Check the activation amplitude for each position of the annotated beats (after the correction).
        This gives an indication of how many beats were effectively next to an audio cue and correctly aligned to it.
        
        Because some beats are in silent times were no audio cues are available, the activation can be low even if the beat is correctly annotated.
        To take this into account, only the beats intersecting with a drum onset are used. The drum onset insure that the activate will not be zero. 

        The threshold specifies the minimum activation value on each annotation to consider the beat correctly annotated
        """
        # Round the timings to milliseconds to correct float imprecisions possibly creating an empty intersection
        intersection = set(np.round(correctedBeats, decimals=4)).intersection(np.round(correctedOnsets, decimals=4))
        beatsAct = [act[int(np.round(t * sampleRate))] for t in intersection if int(np.round(t * sampleRate)) < len(act)]

        if len(beatsAct) == 0:
            raise ValueError("no score at the position of the midi beats. there is an issue with the computation of the beat")

        return len([1 for ba in beatsAct if ba >= threshold]) / len(beatsAct)

    def getAnnotationsQualityHit(self, refBeats, estBeats, sampleRate, fftSize):
        """
        Estimate the quality of the annotations with their precision in relation to the algorithm estimations.
        It uses a hit rate tolerance computed depending on the FFT size used for preprocessing the data later on

        Limitation: This method doesn't handle octave issues where the beats annotated are twice as fast as the beats estimated.
        When that's the case, the precision drops and the score decreases even though the activation function used to correct the beats migh work.
        See "getAnntotationQualityAct"
        """
        # For the tolerance window, either we want to ensure that the annotation is in the same sample than the actual onset
        # toleranceWindow = 1 / (sampleRate * 2)
        # Or we want to make sure that the fft overlaps with the actual onset
        toleranceWindow = (fftSize / 44100) * 0.8 / 2

        # we want to return precision (how many annotation was corrected)
        # Since it's ok to have low recall (more beats detected by the computer because of odd time signature or octave problem)
        f, p, r = mir_eval.onset.f_measure(np.array(refBeats), np.array(estBeats), window=toleranceWindow)
        return p

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
            raise ValueError("Interpolation of the annotation offset failed")

        if max(np.abs(interpolation)) > maxOffsetThreshold:
            raise ValueError("Extrapolation of annotations offset seems too extreme ")

        converted = onsets - interpolation
        converted[converted < 0] = 0
        return converted

    def computeDNNActivationDeviation(self, beats_midi, act, fs_act=100, matchWindow=0.05, lambda_transition=0.1):
        """
        From tapcorrect, compute the most likely offset of the annotated beats according to an activation function

        Parameters
        ----------
        beats_midi : [type]
            [description]
        act : [type]
            [description]
        fs_act : int, optional
            sample rate of the activation. If using Madmom, it should be 100Hz. By default 100
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

    def plotCorrection(self, correction, interp):
        """
        Debug method used to plot the correction applied
        """
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

    def plotActivation(self, audioPath, beatAct):
        """
        Debug Method used to plot the beat activation
        """
        audio = librosa.load(audioPath)
        sStart = 5
        sStop = 7

        fig, axs = plt.subplots(2)
        axs[0].plot(np.arange(0, sStop - sStart, 1 / 22050), audio[0][int(sStart * 22050) : int(sStop * 22050)])
        axs[0].set(ylabel="Amplitude")
        axs[1].plot(np.arange(0, sStop - sStart, 1 / 100), beatAct[int(sStart * 100) : int(sStop * 100)])
        axs[1].set(xlabel="Time (s)", ylabel="Audio cue")
        plt.show()

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
