import logging
import os

import matplotlib.pyplot as plt
import mir_eval
import numpy as np
import tapcorrect
from adtof.converters.converter import Converter
from adtof.io.midiProxy import PrettyMidiWrapper
from adtof.io.textReader import TextReader
from scipy.interpolate import interp1d


class CorrectAlignmentConverter(Converter):
    """
    Converter which tries to align the midi file to the audio file as close as possible
    by looking at the difference between annotated MIDI beats and estimated beats from Madmom
    """

    def convert(
        self,
        refBeatActivationInput,
        missalignedMidiInput,
        convertedDrumTextOutput,
        alignedDrumTextOutput,
        alignedBeatTextOutput,
        alignedMidiOutput,
        maxDeviation=10,
        activationThreshold=0.2,
        thresholdQuality=0.5,
        maxCorrectionDistance=0.08,
        sampleRate=100,
    ):
        """

        Parameters
        ----------
        maxDeviation : int, optional
            Max deviation for the correction computed by tap correct in frames
            In tap correct, the default is 50, meaning a maximum of 0.5s of correction
        activationThreshold: float, optional
            Minimum activation from the beat detection algorithm, at the location of a beat, to consider it well corrected.
            Only the beats intersecting with an onset are used to ensure that an audio cue is present.
        thresholdQuality : float, optional
            The ratio of well aligned beats (with an activation above the activationThreshold) required to keep the track.
            Below this threshold, the track is considered not well corrected and thus it is discarded.
            Only the beats intersecting with an onset are used to ensure that an audio cue is present.
        maxCorrectionThreshold : float, optional
            maximum distance
        """
        # Get annotations and estimation data used as reference.
        midi = PrettyMidiWrapper(missalignedMidiInput)
        beatAct = np.load(refBeatActivationInput, allow_pickle=True)
        beats_midi, beatIdx = midi.get_beats_with_index(stopTime=len(beatAct) / sampleRate)
        tr = TextReader()

        # Compute the dynamic offset for the best alignment
        correction = self.computeDNNActivationDeviation(beats_midi, beatAct, max_deviation=maxDeviation)

        # Apply the dynamic offset to beat
        correctedBeatTimes = self.setDynamicOffset(correction, beats_midi)

        # Apply the dynamic offset to onsets
        if len(midi.instruments) > 1:  # There are multiple drums
            raise ValueError("multiple drums tracks on the midi file")

        # Get the notes from the midi and writte the original alignment
        drumsNotes = [note for instrument in midi.instruments for note in instrument.notes]  # [note.pitch for note in midi.instruments[0].notes]
        drumstimes = [note.start for instrument in midi.instruments for note in instrument.notes]  # [note.start for note in midi.instruments[0].notes]
        tr.writteBeats(convertedDrumTextOutput, [(drumstimes[i], drumsNotes[i].pitch) for i in range(len(drumstimes))])

        # Compute the alignment and measure if it worked
        correctedDrumsTimes = self.setDynamicOffset(correction, drumstimes)
        self.getAnnotationsQualityAct(
            correction,
            drumstimes,
            beatAct,
            sampleRate,
            activationThreshold,
            thresholdQuality,
            maxCorrectionDistance,
            os.path.basename(missalignedMidiInput),
        )

        # writte the aligned output
        tr.writteBeats(alignedDrumTextOutput, [(correctedDrumsTimes[i], drumsNotes[i].pitch) for i in range(len(correctedDrumsTimes))])
        tr.writteBeats(alignedBeatTextOutput, [(correctedBeatTimes[i], beatIdx[i]) for i in range(len(correctedBeatTimes))])
        newMidi = PrettyMidiWrapper.fromListOfNotes(
            [(correctedDrumsTimes[i], drumsNotes[i]) for i in range(len(correctedDrumsTimes))],
            beats=[(correctedBeatTimes[i], beatIdx[i]) for i in range(len(correctedBeatTimes))],
        )
        newMidi.write(alignedMidiOutput)

    def getAnnotationsQualityAct(self, correction, originalOnsets, act, sampleRate, activationThreshold, thresholdQuality, maxCorrectionDistance, trackName):
        """
        Check the beat activation amplitude for each position of the annotated beats after the correction.
        This gives an indication of how many beats were effectively next to an audio cue and correctly aligned to it.

        Because some beats are in silent times where no audio cues are available, the activation can be low even if the beat is correctly annotated.
        To take this into account, only the beats intersecting with a drum onset are used. The drum onset insure that the activate will not be zero.

        The threshold specifies the minimum activation value on each annotation to consider the beat correctly annotated
        """
        onsetsSet = set(np.round(originalOnsets, decimals=4))
        # Round the timings to 4 decimals to correct float imprecisions possibly creating an empty intersection
        correctionAtOnset = [c for c in correction if np.round(c["time"], decimals=4) in onsetsSet]
        activationAtCorrection = [act[int(np.round((c["time"] - c["diff"]) * sampleRate))] for c in correctionAtOnset if int(np.round(c["time"] * sampleRate)) < len(act)]

        if len(activationAtCorrection) == 0:
            raise ValueError("no score at the position of the midi beats. there is an issue with the computation of the beat")

        ratioBeatsWithHighActivation = len([1 for ba in activationAtCorrection if ba >= activationThreshold]) / len(activationAtCorrection)

        # Discard the tracks with a low quality and with extreme corrections
        if ratioBeatsWithHighActivation < thresholdQuality:
            raise ValueError("Little overlap between track's estimated and annotated beats to ensure alignment on " + trackName)
        elif len(correctionAtOnset) / len(correction) < 0.5:
            raise ValueError("The majority of the beats are not overlapping a drum onset on " + trackName)
        elif len([c for c in correctionAtOnset if np.abs(c["diff"]) > maxCorrectionDistance]) > 2:
            raise ValueError("Extreme correction applied to align the beat for " + trackName)

        # self.debugPlot(act, [t["time"] for t in correctionAtOnset], [t["time"] - t["diff"] for t in correctionAtOnset])
        # self.debugPlot(
        #     act,
        #     [
        #         correctionAtOnset[i]["time"] - correctionAtOnset[i]["diff"]
        #         for i, ba in enumerate(activationAtCorrection)
        #         if ba <= activationThreshold
        #     ],
        # )

    def getAnnotationsQualityHit(self, refBeats, estBeats, sampleRate, fftSize=2048):
        """
        Estimate the quality of the annotations with their hit rate to the algorithm estimations.
        It uses a hit rate window computed depending on the FFT size used for preprocessing the data later on

        Limitation: This method doesn't handle octave issues where the beats annotated are twice as fast as the beats estimated.
        When that's the case, the precision drops and the score decreases even though the activation function used to correct the beats migh work.
        See "getAnntotationQualityAct" for an improvement of the method
        """
        import warnings

        warnings.warn("deprecated", DeprecationWarning)
        # For the tolerance window, either we want to ensure that the annotation is in the same sample than the actual onset
        # toleranceWindow = 1 / (sampleRate * 2)
        # Or we want to make sure that the fft overlaps with the actual onset
        toleranceWindow = (fftSize / 44100) * 0.8 / 2

        # we want to return precision (how many annotation was corrected)
        # Since it's ok to have low recall (more beats detected by the computer because of odd time signature or octave problem)
        f, p, r = mir_eval.onset.f_measure(np.array(refBeats), np.array(estBeats), window=toleranceWindow)
        return p

    def debugPlot(self, act, refBeats, corrrectedBeat=None):
        from automix.model.classes.signal import Signal

        Signal(act, sampleRate=100).plot(maxSamples=9999999)
        if corrrectedBeat is not None:
            Signal(1, times=corrrectedBeat).plot(maxSamples=9999999, asVerticalLine=True, color="red")
        Signal(1, times=refBeats).plot(maxSamples=9999999, asVerticalLine=True, show=True)

    def setDynamicOffset(self, offset, onsets):
        """
        Shift the onsets with the offset linearly interpolated

        Raises an exception if the interpolation is outside of the range of known values (should correspond to the length of the audio)
        Raises an exception if the interpolation is above a maxOffsetThreshold
        """

        x = [o["time"] for o in offset]
        y = [o["diff"] for o in offset]

        if x[-1] + np.diff(x)[-1] < onsets[-1]:  # raise an error if the extrapolation is above one beat of length
            raise ValueError("Extrapolation of the annotation is too far from the ground truth with a distance of {:.2f}s".format(onsets[-1] - x[-1]))

        interpolation = interp1d(x, y, kind="linear", fill_value="extrapolate")(onsets)

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
        final_beat_times, mu, sigma = tapcorrect.tapcorrection.convert_dev_sequence_to_corrected_tap_times(dev_sequence, beats_midi, max_deviation, fs_act)
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
        import warnings

        warnings.warn("deprecated", DeprecationWarning)
        # Get the list of unique match between annotations and estimations
        matchIdx = mir_eval.util.match_events(beats_midi, beats_audio, matchWindow)
        matches = np.array([[beats_midi[t[0]], beats_audio[t[1]]] for t in matchIdx])

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
