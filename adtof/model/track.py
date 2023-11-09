import numpy as np
from adtof import config
from adtof.io import mir
from adtof.io.textReader import TextReader
from adtof.io.midiProxy import PrettyMidiWrapper
from adtof.ressources import instrumentsMapping
import os
import logging
from collections import defaultdict
import copy
import itertools


class Track(object):
    def __init__(
        self,
        audioPath,
        preprocessPath=None,
        beatPath=None,
        annotationPath=None,
        removeStart=False,
        labelOffset=0,
        sampleRate=100,
        sampleWeight=None,
        emptyWeight=1,
        tatumSubdivision=None,
        tempoInterval=None,
        minVelocity=0,
        **kwargs
    ):
        """
        Class handling the reading of a single track.

        Parameters
        removeStart: Remove start is required in ADTOF set to remove the sonified count-in in the tracks which is not annotated
        """
        # Read audio
        self.path = audioPath
        self.title = os.path.basename(audioPath)
        x = self.readAudio(audioPath, preprocessPath, sampleRate=sampleRate, **kwargs)

        # Read beats
        if beatPath is not None:
            # TODO handle compound time signature: The beats are not divided equally in those cases.
            # Dict of beat number (1-4) to time
            beats = self.readBeats(beatPath, **kwargs)
            self.beats = beats

        # read sparse target
        if annotationPath is not None:
            notes = self.readLabels(annotationPath, **kwargs)
            timeOffset = 0
            if labelOffset:  # We offset label to compensate for the valid padding of the encoder
                timeOffset = labelOffset / sampleRate
                for note in notes:
                    note["time"] -= timeOffset

            if removeStart:  # We remove start for the "count-in" in rhythm game tracks. Not necessary for the other datasets
                maxTime = [n["time"] for n in beats if n["pitch"] == 1][2] - timeOffset  # get the time of the 3rd downbeat
                x, notes, beats = self.removeStart(x, notes, beats, sampleRate=sampleRate, maxTime=maxTime, **kwargs)  # Trim x, Offset labels and beats

            # Build a label to time dictionary
            y = defaultdict(list)
            for note in notes:
                y[note["pitch"]].append(note["time"])
            self.y = y

        self.x = x
        self.samplesCardinality = len(x)

        # Read tatum
        if tatumSubdivision is not None:
            # List of beats (0-n) to time
            # Some beats can be duplicated in RBMA beats annotations. Remove the duplications with a call to set()
            self.beatsTime = sorted(set([n["time"] for n in beats]))

            # List of tatums to time
            # Double the tempo of the track to get the tatum time when it is too slow
            if tempoInterval is not None:
                tatumsTime = []
                tatumSubdivisionDivisors = [1 / i for i in range(2, tatumSubdivision // 2 + 1) if tatumSubdivision % i == 0] + [1 / tatumSubdivision]
                for start, stop in zip(self.beatsTime, self.beatsTime[1:]):
                    tempo = 60 / (stop - start)
                    if tempo < tempoInterval[0]:
                        factor = [factor for factor in [2, 4, 8] if tempo * factor >= tempoInterval[0]][0]
                    elif tempo >= tempoInterval[1]:
                        bestFactors = [factor for factor in tatumSubdivisionDivisors if tempo * factor < tempoInterval[1]]
                        if len(bestFactors):
                            factor = bestFactors[0]
                        else:
                            logging.warning("Track {} has a tempo of {} bpm. The tempo is too fast to spawn 1 tatum".format(self.title, tempo))
                            continue
                    else:
                        factor = 1
                    tatumsTime += list(np.arange(start, stop, (stop - start) / (tatumSubdivision * factor)))
                tatumsTime += [self.beatsTime[-1]]
            else:
                tatumsTime = [tatum for start, stop in zip(self.beatsTime, self.beatsTime[1:]) for tatum in np.arange(start, stop, (stop - start) / tatumSubdivision)] + [
                    self.beatsTime[-1]
                ]
            self.tatumsTime = tatumsTime
            # List of tatums to mid point time with the previous tatum
            tatumsBoundarieTime = [tatumsTime[0]] + [(start + stop) / 2 for start, stop in zip(tatumsTime, tatumsTime[1:])] + [tatumsTime[-1]]
            # List of tatums to the closest boundary frames: [start, stop[
            self.tatumsBoundariesFrame = np.array(
                [[int(np.round(boundary * sampleRate)) for boundary in boundaries] for boundaries in zip(tatumsBoundarieTime, tatumsBoundarieTime[1:])]
            )

            # Check that the tatum will spawn at least one frame
            if min(self.tatumsBoundariesFrame[:, 1] - self.tatumsBoundariesFrame[:, 0]) <= 0:
                logging.warning("Tatums are too close together. Check that the beats are correct" + str(self.path))

            # List of frames to corresponding tatum
            self.framesTatum = [i for i, values in enumerate(self.tatumsBoundariesFrame) for v in range(*values)]

            # If the first beat is not at the start of the track (e.g., when the beat detection is not perfect), we need to extend the list of framesTatum to include the frames before the first detected tatum. We attribute them to the first tatum.
            if self.tatumsBoundariesFrame[0][0] != 0:
                self.framesTatum = [0] * self.tatumsBoundariesFrame[0][0] + self.framesTatum

            self.samplesCardinality = len(self.tatumsBoundariesFrame)

            # Get target for each tatum
            if "labelRadiation" in kwargs and kwargs["labelRadiation"] != 0:
                raise NotImplementedError("labelRadiation is not implemented for tatum-level predictions")

        # Compute dense target
        if annotationPath is not None:
            if tatumSubdivision is not None:  # Get target for each tatum
                yDense, velocityDense = Track.getDenseEncoding(notes, sampleRate=sampleRate, frameToTatum=self.framesTatum, **kwargs)
            else:  # Get target for each frame
                yDense, velocityDense = Track.getDenseEncoding(notes, sampleRate=sampleRate, length=len(x) + 1, **kwargs)

            # if beatPath is not None:
            #     for b in beats:
            #         b["pitch"] = 1 if b["pitch"] == 1 else 0
            #     newkwargs = copy.copy(kwargs)
            #     newkwargs["labels"] = [0, 1]
            #     beatsDense, _ = Track.getDenseEncoding(beats, sampleRate=sampleRate, length=len(x) + 1, **newkwargs)
            self.yDense = yDense

            # Get weight for each target
            if sampleWeight is not None:
                self.sampleWeight = np.maximum(np.sum(yDense * sampleWeight, axis=1), emptyWeight)
                if minVelocity:
                    # get minimum velocity for each frame
                    frameVelocity = np.array([np.min(velocityDense[frame]) for frame in range(len(velocityDense))])
                    binaryDensity = (frameVelocity > minVelocity).astype(np.int16)
                    self.sampleWeight *= binaryDensity

    def readAudio(self, audioPath, preprocessPath=None, **kwargs):
        """
        Read the track audio
        """
        x = mir.preProcess(audioPath, cachePath=preprocessPath, **kwargs)
        return x

    def readBeats(self, beatPath, **kwargs):
        """
        get the beat annotations
        """
        if ".mid" in beatPath[-4:] or ".midi" in beatPath[-5:]:
            beats, beatIdx = PrettyMidiWrapper(beatPath).get_beats_with_index()
            return [{"time": b, "pitch": i} for b, i in zip(beats, beatIdx)]
        else:
            return TextReader().getOnsets(beatPath, **kwargs)

    def readLabels(self, annotationPath, mappingDictionaries=[instrumentsMapping.MIDI_REDUCED_5], removeIfClassUnknown=False, **kwargs):
        """
        get the track annotations
        """
        # Read the annotations
        if ".mid" in annotationPath[-4:] or ".midi" in annotationPath[-5:]:
            notes = PrettyMidiWrapper(annotationPath).getOnsets(**kwargs)
        else:
            notes = TextReader().getOnsets(annotationPath, **kwargs)

        # Remap the annotations
        self.logMapped = defaultdict(int)
        sparseNotes = []
        for note in notes:
            targetPitches = config.remapPitches(note["pitch"], mappingDictionaries, removeIfUnknown=removeIfClassUnknown)
            self.logMapped[str((note["pitch"], list(targetPitches)))] += 1
            for targetPitch in targetPitches:
                if targetPitch is not None:
                    sparseNotes.append({"pitch": targetPitch, "time": note["time"], "velocity": note["velocity"]})

        # remove duplicated notes (same time and same pitch) due to simplified mapping (and possibly other reasons)
        uniqueNotes = {(note["time"], note["pitch"]): note for note in sparseNotes}
        sparseNotes = sorted(uniqueNotes.values(), key=lambda x: x["time"])

        return sparseNotes

    def removeStart(self, x, notes, beats, sampleRate=100, context=25, maxTime=99, trimEnd=False, **kwargs):
        """
        Trim x to either the first annotation or the two first bars, whatever comes first.
        Change the time of notes to start at 0
        """
        # Trim before the first note to remove count in
        # Move the trim by the offset amount to keep the first notation
        firstNoteTime = notes[0]["time"] if len(notes) else 0
        firstNoteTime = min(maxTime, firstNoteTime)
        firstNoteTime = max(0, firstNoteTime)
        firstNoteIdx = int(round(firstNoteTime * sampleRate))

        # Trim after the last note to remove all part of the track not annotated
        # Make sure the index doesn't exceed any boundaries
        # TODO is it necessary, or do we want to keep all the audio?
        if trimEnd:
            lastNoteTime = notes[-1]["time"] if len(notes) else len(x) / sampleRate
            lastNoteIdx = min(int(lastNoteTime * sampleRate) + 1, len(x) - 1 - context)
            x = x[firstNoteIdx : lastNoteIdx + context]
        else:
            x = x[firstNoteIdx:]

        # Change the timing in the annotations
        for note in notes:
            note["time"] -= firstNoteTime

        # Change the timing in the beats annotations
        beats = [beat for beat in beats if beat["time"] >= firstNoteTime]
        for beat in beats:
            beat["time"] = beat["time"] - firstNoteTime

        return (x, notes, beats)

    @staticmethod
    def getDenseEncoding(notes, length=None, sampleRate=100, frameToTatum=None, labels=config.LABELS_5, labelRadiation=1, **kwargs):
        """
        Encode in a dense matrix the midi onsets
        labelRadiation = how many samples from the event have a non-null target
        if frameToTatum is specified, return an array of #tatum length

        TODO: Move that to io.midi
        """

        if length == None:  # if length not specified, make it big enough to store all the notes
            if frameToTatum is not None:  # Either it is the frame of the last tatum
                length = frameToTatum[-1] + 1
            else:  # or the frame of the last note
                lastNoteTime = notes[-1]["time"]  # TODO: break if there is no note
                length = int(np.round((lastNoteTime) * sampleRate)) + 1

        result = []
        resultVelocity = []
        for key in labels:
            # Size of the dense matrix
            row = np.zeros(length)
            rowVelocity = np.ones(length)
            for note in [n for n in notes if n["pitch"] == key]:
                # index of the event
                time = note["time"]
                index = int(np.round(time * sampleRate))

                if frameToTatum is None:
                    if index < 0 or index >= len(row):
                        continue
                else:
                    if index < 0 or index >= len(frameToTatum):
                        continue
                    index = frameToTatum[index]

                if labelRadiation == 0:
                    target = [1]
                elif labelRadiation == 1:
                    target = [0.5, 1, 0.5]
                elif labelRadiation == 2:
                    target = [0.25, 0.5, 1.0, 0.5, 0.25]
                elif labelRadiation == 3:
                    target = [0.14644661, 0.5, 0.85355339, 1.0, 0.85355339, 0.5, 0.14644661]
                else:
                    raise NotImplementedError("Radiation of this size not implemented: " + str(labelRadiation))

                for i in range(-labelRadiation, labelRadiation + 1):
                    if index + i >= 0 and index + i < len(row):
                        row[index + i] = target[labelRadiation + i]
                        rowVelocity[index + i] = note["velocity"]
                row[index] = 1
                rowVelocity[index] = note["velocity"]
            result.append(row)
            resultVelocity.append(rowVelocity)

        return np.array(result).T, np.array(resultVelocity).T

    def getAvailableSliceIndexes(self, trainingSequence=400, tatumSubdivision=None, context=9, samePadding=False, **kwargs):
        """
        Get the indexes of the start of possible slices fitting training sequence.
        This handles the match between input and target size when applying tatum-synchronicity or frame-synchronicity with same padding,
        and missmatch between input and target size when applying frame-synchronicity with valid padding.
        """
        if tatumSubdivision is not None or samePadding == True:
            return range(0, self.samplesCardinality - (trainingSequence + 1), (trainingSequence + 1))
        else:  # If valid padding, the input is larger by context - 1 than the target.
            return range(0, self.samplesCardinality - (trainingSequence + (context - 1)), (trainingSequence + (context - 1)))

    def getSlice(self, sampleIdx, trainingSequence, tatumSubdivision=None, context=9, sampleWeight=None, samePadding=False, **kwargs):
        """
        Split the track into sequences
        """
        if tatumSubdivision is None:
            # TODO do we change the offset to be applied in the generator? This will require to compute the offset again during infering?
            # raise DeprecationWarning()
            if samePadding == True:
                input = {"x": self.x[sampleIdx : sampleIdx + trainingSequence]}
            else:
                input = {"x": self.x[sampleIdx : sampleIdx + trainingSequence + (context - 1)]}
        else:
            # Get the frame index of the tatums' boundary used for training
            tatumsBoundariesFrame = self.tatumsBoundariesFrame[sampleIdx : sampleIdx + trainingSequence]
            # Give the input sampes for the encoder
            x = self.x[tatumsBoundariesFrame[0][0] : tatumsBoundariesFrame[-1][1]]
            # Give the indexes for the tatum max-pooling used on the embedded space
            tatumsBoundariesFrame = tatumsBoundariesFrame - tatumsBoundariesFrame[0][0]
            input = {"x": x, "tatumsBoundariesFrame": tatumsBoundariesFrame}

        #  Check if self has yDense attribute
        if not hasattr(self, "yDense"):
            return (input,)

        target = self.yDense[sampleIdx : sampleIdx + trainingSequence]

        if sampleWeight is not None:
            return input, target, self.sampleWeight[sampleIdx : sampleIdx + trainingSequence]
        else:
            return input, target

    def getUniqueSequences(self, beatNumberAsBoundary=None, distanceThreshold=0.02, subdivision=12):
        """
        Returns a list of unique sequences from the given grid
        """

        groups = defaultdict(int)
        grid = [n["time"] for n in self.beats if beatNumberAsBoundary is None or n["pitch"] == beatNumberAsBoundary]
        for start, end in zip(grid, grid[1:]):
            notes = []
            for pitch, onsets in self.y.items():
                pitchNotes = [(o, pitch) for o in itertools.takewhile(lambda x: x < end - distanceThreshold, (o for o in onsets if start - distanceThreshold <= o))]
                onsets = onsets[len(pitchNotes) :]
                notes += pitchNotes
            # notes = [(o, pitch) for pitch, onsets in self.y.items() for o in onsets if start - distanceThreshold <= o < end - distanceThreshold]
            notesRelative = tuple(sorted((np.round(abs(start - o) / (end - start) * subdivision) / subdivision, p) for o, p in notes))
            groups[notesRelative] += 1
        return groups
