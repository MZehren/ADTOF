import os
from bisect import bisect_left

import librosa
import madmom
import numpy as np
import pretty_midi

from adtof.io.converters.converter import Converter
from adtof.io.converters.textConverter import TextConverter


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

        print(outputPath)
        error, offset, diffPlayback = self.getError(kicks_midi, kicks_audio)  #musicBeats[:, 0], midiBeats)
        assert np.isnan(offset) == False

        with open(outputPath + ".txt", "w") as file:
            file.write("MAE, offset, playback\n" + str(error) + "," + str(offset) + "," + str(diffPlayback))
        # midi.addDelay(-offset)
        # midi.save(outputPath)

    def getError(self, onsetsA, onsetsB, maxThreshold=0.05):
        """
        Compute the average error of onsets close to each other bellow a maxThreshold

        the correction tells how much you should shift B to align with A
        or -correction you need to shift A to align with B 
        """
        # Get the alignment between the notes and the onsets
        tuples = [(onsetA, self.findNeighboor(onsetsB, onsetA)) for onsetA in onsetsA]
        tuplesThresholded = np.array(
            [[onsets[0], onsets[1]] for onsets in tuples if np.abs(onsets[0] - onsets[1]) < maxThreshold])

        def getOffset()
        # get the offset difference
        diffOffset = [onsets[0] - onsets[1] for onsets in tuplesThresholded]
        correctionOffset = np.mean(diffOffset)
        remainingError0 = np.mean(np.abs(diffOffset))  # TODO: MAE vs RMSE?
        remainingError1 = np.mean(np.abs(diffOffset - correctionOffset))  # TODO: MAE vs RMSE?

        # Get the playback difference and apply it
        correctionPlayback = np.mean(np.diff(tuplesThresholded[:, 1])) / np.mean(np.diff(tuplesThresholded[:, 0]))
        midOffset = (tuplesThresholded[-1][1] / correctionPlayback - tuplesThresholded[-1][1]) / 2
        tuplesThresholded = [(a, b / correctionPlayback + midOffset) for a, b in tuplesThresholded]

        # get the offset difference
        diffOffset = [onsets[0] - onsets[1] for onsets in tuplesThresholded]
        correctionOffset = np.mean(diffOffset)
        remainingError2 = np.mean(np.abs(diffOffset - correctionOffset))  # TODO: MAE vs RMSE?

        if remainingError0 > 0.020:
            print(remainingError0, remainingError1, remainingError2)
            print()
        return remainingError2, correctionOffset, correctionPlayback

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
