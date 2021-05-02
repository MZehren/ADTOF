import logging
import os
from bisect import bisect_left

import librosa
import madmom
import matplotlib.pyplot as plt
import mir_eval
import numpy as np
from numpy import ma
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
        refBeatActivationInput,
        missalignedMidiInput,
        alignedDrumTextOutput,
        alignedBeatTextOutput,
        alignedMidiOutput,
        thresholdQuality=0.95,
        sampleRate=100,
        thresholdCorrectionWindow=0.05,
        actThreshold=0.1,
        debug=False,
    ):
        """
        # TODO: utiliser un jeu de données non aligné pour le test

        Parameters
        ----------
        thresholdQuality : float, optional
            the limit at which the track will get discarded because there are too many beats annotated are not identified and aligned to the computer estimations
            TODO: The score uses either the precision of the beats hit rate, or the value of the beat activation function after the alignment, or the maximum correction offset. 
        thresholdCorrectionWindow : float, optional
            TODO: see Toward automatically correcting tapped beat... 
            If the value is too large, the quality score will artificially increase, but the beats are going to be aligned to wrong estimations as well. 
        """
        # Get annotations and estimation data used as reference.
        midi = PrettyMidiWrapper(missalignedMidiInput)
        beatAct = np.load(refBeatActivationInput, allow_pickle=True)
        beats_midi, beatIdx = midi.get_beats_with_index(stopTime=len(beatAct) / sampleRate)
        tr = TextReader()

        # Compute the dynamic offset for the best alignment
        correction = self.computeDNNActivationDeviation(beats_midi, beatAct, max_deviation=50)
        ## Deprecated Method
        # smoothingCorrectionWindow = 5
        # beats_audio = [el["time"] for el in tr.getOnsets(refBeatInput, mappingDictionaries=[], group=False)]
        # correction = self.computeTrackedBeatsDeviation(
        #     beats_midi, beats_audio, matchWindow=thresholdCorrectionWindow, smoothWindow=smoothingCorrectionWindow
        # )

        # Apply the dynamic offset to beat
        correctedBeatTimes = self.setDynamicOffset(correction, beats_midi)

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
        correctedDrumsTimes = self.setDynamicOffset(correction, drumstimes)

        # # Measure if the annotations are of good quality.
        # qualityAct = self.getAnnotationsQualityAct(
        #     beats_midi, correctedBeatTimes, correctedDrumsTimes, beatAct, sampleRate, actThreshold=actThreshold
        # )
        # ## Deprecated Method
        # # qualityHit = self.getAnnotationsQualityHit([t for t in beats_audio if t <= midi.get_end_time()], correctedBeatTimes, sampleRate)
        # if debug:
        #     return qualityAct

        # # Get the beats with a huge correction which doesn't seem correct (intersecting with a drums onset to remove wrong estimations from madmom)
        # drumstimesSet = set(drumstimes)
        # fishy_corrections = [c for c in correction if np.abs(c["diff"]) > 0.025 and c["time"] in drumstimesSet]

        # # Discard the tracks with a low quality and with extreme corrections (set to 25ms)
        # if qualityAct < thresholdQuality:
        #     # debug
        #     # print("investigate", os.path.basename(audioPath))
        #     # print(
        #     #     "odd_Time",
        #     #     len([t for t in midi.time_signature_changes if t.numerator != 4]) > 0,
        #     #     "/ fast_tempo",
        #     #     max(midi.get_tempo_changes()[1]) > 215,
        #     # )
        #     # print("activation_quality", qualityAct, "/ hitRate_quality", qualityHit)
        #     # correctedDrumsTimesSet = set(correctedDrumsTimes)
        #     # print("fishy_corrections", [c for c in correction if np.abs(c["diff"]) > 0.03 and c["time"].in(correctedDrumsTimesSet)])

        #     raise ValueError("Not enough overlap between track's estimated and annotated beats to ensure alignment")
        # elif len(fishy_corrections) > 0:  # TODO: Why 2 again?
        #     # debug
        #     # for c in fishy_corrections:
        #     #     print(
        #     #         "correction",
        #     #         np.round(c["diff"], decimals=2),
        #     #         str(int(c["time"] // 60)) + ":" + str(np.round(c["time"] % 60, decimals=2)),
        #     #     )
        #     raise ValueError("Extreme correction needed for this track")

        # writte the output
        tr.writteBeats(alignedDrumTextOutput, [(correctedDrumsTimes[i], drumsPitches[i]) for i in range(len(correctedDrumsTimes))])
        tr.writteBeats(alignedBeatTextOutput, [(correctedBeatTimes[i], beatIdx[i]) for i in range(len(correctedBeatTimes))])
        newMidi = PrettyMidiWrapper.fromListOfNotes(
            [(correctedDrumsTimes[i], drumsPitches[i]) for i in range(len(correctedDrumsTimes))],
            beats=[(correctedBeatTimes[i], beatIdx[i]) for i in range(len(correctedBeatTimes))],
        )
        newMidi.write(alignedMidiOutput)

    def getAnnotationsQualityAct(self, originalBeats, correctedBeats, correctedOnsets, act, sampleRate, actThreshold=0.1):
        """
        Check the beat activation amplitude for each position of the annotated beats after the correction.
        This gives an indication of how many beats were effectively next to an audio cue and correctly aligned to it.
        
        Because some beats are in silent times were no audio cues are available, the activation can be low even if the beat is correctly annotated.
        To take this into account, only the beats intersecting with a drum onset are used. The drum onset insure that the activate will not be zero. 

        The threshold specifies the minimum activation value on each annotation to consider the beat correctly annotated
        """
        # Round the timings to 4 decimals to correct float imprecisions possibly creating an empty intersection
        intersection = list(set(np.round(correctedBeats, decimals=4)).intersection(np.round(correctedOnsets, decimals=4)))
        intersection.sort()
        beatsAct = [act[int(np.round(t * sampleRate))] for t in intersection if int(np.round(t * sampleRate)) < len(act)]
        beatsActIdx = [int(np.round(t * sampleRate)) for t in intersection if int(np.round(t * sampleRate)) < len(act)]

        if len(beatsAct) == 0:
            raise ValueError("no score at the position of the midi beats. there is an issue with the computation of the beat")

        # # try with a threshold set to the mean of the act
        # actThreshold = np.mean(act)
        pWindow = 1
        margin = 0.05
        isPeak = [act[i] + margin >= max(act[max(i - pWindow, 0) : i + 1 + pWindow]) for i in beatsActIdx]
        actThreshold = 0.1
        isAbove = [act[i] >= actThreshold for i in beatsActIdx]

        peakConfidence = len([1 for i in range(len(isPeak)) if isPeak[i]]) / len(intersection)
        confidence = len([1 for ba in beatsAct if ba >= actThreshold]) / len(beatsAct)
        # if peakConfidence < 0.90:
        #     # self.plotActivation([np.round(i, decimals=2) for i in intersection], act)
        #     from automix.model.classes.signal import Signal

        #     Signal(act, sampleRate=100).plot(maxSamples=9999999999999)
        #     originalBeats = set([np.round(i, decimals=2) for i in originalBeats])
        #     correctedBeats = set([np.round(i, decimals=2) for i in correctedBeats])
        #     Signal(1, times=[i for i in originalBeats if i not in correctedBeats]).plot(
        #         maxSamples=9999999999, asVerticalLine=True, color="red"
        #     )
        #     Signal(1, times=[i for i in correctedBeats if i not in originalBeats]).plot(maxSamples=9999999999, asVerticalLine=True)
        #     Signal(1, times=[i for i in correctedBeats if i in originalBeats]).plot(
        #         maxSamples=9999999999, asVerticalLine=True, color="green", show=True
        #     )
        #     # Signal(1, times=[time for i, time in enumerate(intersection) if beatsAct[i] <= actThreshold]).plot(
        #     #     maxSamples=9999999999, asVerticalLine=True, show=True, color="red"
        #     # )

        #     # Signal(1, times=[intersection[i] for i in range(len(isPeak)) if not isPeak[i] or not isAbove[i]]).plot(
        #     #     maxSamples=9999999999, asVerticalLine=True, show=True, color="red"
        #     # )
        return confidence

    def getAnnotationsQualityHit(self, refBeats, estBeats, sampleRate, fftSize=2048):
        """
        Estimate the quality of the annotations with their hit rate to the algorithm estimations.
        It uses a hit rate window computed depending on the FFT size used for preprocessing the data later on

        Limitation: This method doesn't handle octave issues where the beats annotated are twice as fast as the beats estimated.
        When that's the case, the precision drops and the score decreases even though the activation function used to correct the beats migh work.
        See "getAnntotationQualityAct" for an improvement of the method
        """
        # For the tolerance window, either we want to ensure that the annotation is in the same sample than the actual onset
        # toleranceWindow = 1 / (sampleRate * 2)
        # Or we want to make sure that the fft overlaps with the actual onset
        toleranceWindow = (fftSize / 44100) * 0.8 / 2

        # we want to return precision (how many annotation was corrected)
        # Since it's ok to have low recall (more beats detected by the computer because of odd time signature or octave problem)
        f, p, r = mir_eval.onset.f_measure(np.array(refBeats), np.array(estBeats), window=toleranceWindow)
        return p

    def setDynamicOffset(self, offset, onsets):
        """
        Shift the onsets with the offset linearly interpolated
        
        Raises an exception if the interpolation is outside of the range of known values (should correspond to the length of the audio)
        Raises an exception if the interpolation is above a maxOffsetThreshold
        """

        x = [o["time"] for o in offset]
        y = [o["diff"] for o in offset]
        minBeatInter = min(np.diff(x))
        maxTempo = 60 / minBeatInter
        if x[-1] + minBeatInter < onsets[-1]:  # Check how far the correction has to be extrapolated
            raise ValueError(
                "Extrapolation of the annotation is too far from the ground truth with a distance of {:.2f}s".format(onsets[-1] - x[-1])
            )

        interpolation = interp1d(x, y, kind="linear", fill_value="extrapolate")(onsets)
        # self.plotCorrection(offset, interp1d(x, y, kind="linear", fill_value=0.0))

        if max(np.abs(interpolation)) > minBeatInter * 0.5:
            raise ValueError(
                "Interpolation of annotations offset seems too extreme ({:.2f}s) wereas the min beat interval is {:.2f}s ({:.2f} bpm)".format(
                    max(np.abs(interpolation)), minBeatInter, maxTempo
                )
            )

        converted = onsets - interpolation
        converted[converted < 0] = 0
        return converted

    def computeDNNActivationDeviation(self, beats_midi, act, fs_act=100, max_deviation=50, lambda_transition=0.1):
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
            see tapcorrect, keeping the default 50
        lambda_transition : float, optional
            see tapcorrect, keeping the default 0.1

        Returns
        -------
        [type]
            [description]
        """
        # compute deviation matrix
        D_pre, list_iois_pre = tapcorrect.tapcorrection.compute_deviation_matrix(act, beats_midi, fs_act, max_deviation)
        # compute deviation sequence
        dev_sequence = tapcorrect.tapcorrection.compute_score_maximizing_dev_sequence(D_pre, lambda_transition)
        final_beat_times, mu, sigma = tapcorrect.tapcorrection.convert_dev_sequence_to_corrected_tap_times(
            dev_sequence, beats_midi, max_deviation, fs_act
        )
        result = [{"time": beats_midi[i], "diff": beats_midi[i] - final_beat_times[i]} for i in range(len(beats_midi))]
        # plt.plot([r["diff"] for r in result], label=str(max_deviation))
        # plt.legend()
        # plt.show()
        return result

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
        interValues = np.arange(-10, 10, 0.05)

        cm = 1 / 2.54  # centimeters in inches
        width = 17.2 * cm
        plt.figure(figsize=(width, width * 0.4))
        plt.plot([t["time"] for t in correction], [t["diff"] for t in correction], "+", zorder=5, label="Beats deviation")
        plt.plot(interValues, interp(interValues), "--", label="Interpolation")
        plt.legend()
        plt.ylabel("Deviation (s)")
        plt.xlabel("Position (s)")
        plt.savefig("alignment.pdf", dpi=600, bbox_inches="tight")
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
