import logging
import os
from bisect import bisect_left

import librosa
import madmom
import numpy as np
import pandas as pd
import pretty_midi
from mir_eval.onset import f_measure

from adtof.io.converters.converter import Converter
from adtof.io.converters.textConverter import TextConverter
from adtof import config

class OnsetsAlignementConverter(Converter):
    """
    Converter which tries to align the midi file to the music file as close as possible 
    by looking at the difference between MIDI note_on events and librosa.onsets
    """
    # CNNPROC = madmom.features.onsets.CNNOnsetProcessor()
    # PEAKPROC = madmom.features.onsets.OnsetPeakPickingProcessor(fps=100)

    DBPROC = madmom.features.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    DBACT = madmom.features.RNNDownBeatProcessor()

    # def convertRecursive(self, rootFodler, outputName, midiCandidates=None, musicCandidates=None):
    #     converted = 0
    #     failed = 0
    #     if midiCandidates is None:
    #         midiCandidates = PhaseShiftConverter.PS_MIDI_NAMES
    #     if musicCandidates is None:
    #         musicCandidates = PhaseShiftConverter.PS_AUDIO_NAMES
    #     for root, _, files in os.walk(rootFodler):
    #         midiFiles = [file for file in midiCandidates if file in files]
    #         musicFiles = [file for file in musicCandidates if file in files]

    #         if midiFiles and musicFiles:
    #             try:
    #                 self.convert(os.path.join(root, musicFiles[0]), os.path.join(root, midiFiles[0]), os.path.join(root, outputName))
    #                 print("converted", root)
    #                 converted += 1
    #             except ValueError:
    #                 print(ValueError)
    #                 failed += 1
    #     print("converted", converted, "failed", failed)

    def convert(self, inputMusicPath, inputMidiPath, outputPath, beatsPath):
        # Get midi onsets
        # midi = MidiProxy(inputMidiPath)
        # midiOnsets = midi.getOnsets()

        # get midi beats
        # midi = pretty_midi.PrettyMIDI(inputMidiPath)
        # startNote = midi.get_onsets()[0]
        # midiBeats = [b for b in midi.get_beats() if b > startNote]

        # get midi kicks
        midi = pretty_midi.PrettyMIDI(inputMidiPath)
        kicks_midi = [note.start for note in midi.instruments[0].notes if note.pitch == 36]

        # get audio onsets
        # y, sr = librosa.load(inputMusicPath)
        # musicOnsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
        # act = OnsetsAlignementConverter.CNNPROC(inputMusicPath)
        # musicOnsets = OnsetsAlignementConverter.PEAKPROC(act)

        # get audio beats
        # if os.path.exists(beatsPath):
        #     musicBeats = np.load(beatsPath)
        # else:
        #     act = OnsetsAlignementConverter.DBACT(inputMusicPath)
        #     musicBeats = OnsetsAlignementConverter.DBPROC(act)
        #     np.save(beatsPath, musicBeats)

        # get audio estimated kicks
        tc = TextConverter()
        kicks_audio = tc.getOnsets(inputMusicPath, separated=True)[36]

        error, offset, diffPlayback = self.computeAlignment(kicks_midi, kicks_audio)  #musicBeats[:, 0], midiBeats)
        f, p, r = f_measure(np.array(kicks_midi) * diffPlayback - offset, np.array(kicks_audio), window=0.01)
        pd.DataFrame({
            "MAE": [error],
            "offset": [offset],
            "playback": [diffPlayback],
            "F-measure": [f],
            "precision": [p],
            "recall": [r]
        }).to_csv(outputPath + ".txt")

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
            return 0, 0, 1

        # Compute two versions of the alignement
        offset1, error1 = self.computeOffset(tuplesThresholded)
        playback2, offset2, error2 = self.computeRate(tuplesThresholded)

        assert np.isnan(offset1) == False and np.isnan(offset2) == False

        if error1 < error2:  #TODO: There has to be something wrong in the computation
            return error1, offset1, 1
        else:
            return error2, offset2, playback2

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
