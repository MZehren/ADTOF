import logging
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import sklearn

from adtof import config
from adtof.io.mir import MIR
from adtof.io.textReader import TextReader


class DataLoader(object):
    def __init__(self, folderPath):
        """
        Class for the handling of the worflow while loading the dataset

        Parameters
        ----------
        folderPath : [str]
            Path to the root folder containing the dataset
        """
        self.folderPath = folderPath

        # Getting the intersection of audio and annotations files
        self.audioPaths = config.getFilesInFolder(self.folderPath, config.AUDIO)
        self.annotationPaths = config.getFilesInFolder(self.folderPath, config.ALIGNED_DRUM)
        self.audioPaths, self.annotationPaths = config.getIntersectionOfPaths(self.audioPaths, self.annotationPaths)
        self.featurePaths = np.array(
            [os.path.join(folderPath, config.PROCESSED_AUDIO, config.getFileBasename(track) + ".npy") for track in self.audioPaths]
        )

    def readTrack(self, i, sampleRate=100, context=25, labelOffset=0, labels=[36, 40, 41, 46, 49], yDense=True, removeStart=True, **kwargs):
        """
        Read the track and the midi to return X and Y 
        """
        mir = MIR(frameRate=sampleRate, **kwargs)
        x = mir.open(self.audioPaths[i], cachePath=self.featurePaths[i])
        x = x.reshape(x.shape + (1,))  # Add the channel dimension

        # read files
        notes = TextReader().getOnsets(self.annotationPaths[i])
        # Trim before the first midi note (to remove the unannotated count in) and after the last uncovered part
        # Default to index 5s if there are not pitch with event.
        firstNoteTime = np.min([v[0] for v in notes.values() if len(v)])
        lastNoteTime = np.max([v[-1] for v in notes.values() if len(v)])
        firstNoteIdx = int(firstNoteTime * sampleRate)
        lastNoteIdx = min(int(lastNoteTime * sampleRate) + 1, len(x) - 1 - context)

        X = x[firstNoteIdx - labelOffset : lastNoteIdx + context - labelOffset]
        if yDense:  # Return dense GT
            y = self.getDenseEncoding(self.annotationPaths[i], notes, sampleRate=sampleRate, keys=labels, **kwargs)
            Y = y[firstNoteIdx:lastNoteIdx]
            return (X, Y)
        else:  # Return sparse GT
            notes = {k: v - firstNoteTime for k, v in notes.items()}
            return (X, notes)

    def getDenseEncoding(
        self, filename, notes, sampleRate=100, offset=0, playback=1, keys=[36, 40, 41, 46, 49], labelRadiation=1, **kwargs
    ):
        """
        Encode in a dense matrix the midi onsets

        sampleRate = sampleRate
        timeShift = offset of the midi, so the event is actually annotated later
        keys = pitch of the offset in each column of the matrix
        labelRadiation = how many samples from the event have a non-null target
        TODO: Move that to io.midi
        """
        length = np.max([values[-1] for values in notes.values()])
        result = []
        for key in keys:
            # Size of the dense matrix
            row = np.zeros(int(np.round((length / playback + offset) * sampleRate)) + 1)
            for time in notes[key]:
                # indexs at 1 in the dense matrix
                index = int(np.round((time / playback + offset) * sampleRate))
                assert index >= 0
                # target = [(np.cos(np.arange(-radiation, radiation + 1) * (np.pi / (radiation + 1))) + 1) / 2]
                if labelRadiation == 0:
                    target = [1]
                elif labelRadiation == 1:
                    target = [0.5, 1, 0.5]
                elif labelRadiation == 2:
                    target = [0.25, 0.75, 1.0, 0.75, 0.25]
                elif labelRadiation == 3:
                    target = [0.14644661, 0.5, 0.85355339, 1.0, 0.85355339, 0.5, 0.14644661]
                else:
                    raise NotImplementedError("Radiation of this size not implemented: " + str(labelRadiation))

                for i in range(-labelRadiation, labelRadiation + 1):
                    if index + i >= 0 and index + i < len(row):
                        row[index + i] = target[labelRadiation + i]
                row[index] = 1

            result.append(row)
            if len(notes[key]) == 0:
                logging.debug("Pitch %s is not represented in the track %s", key, filename)
        return np.array(result).T

    def getSplit(self, trainNSplit=10, validationSplit=0.20, randomState=1, limit=None, **kwargs):
        """
        TODO
        """

        # Split the data in train, validation and test, without same band in test and train+test
        groups = [config.getBand(path) for path in self.audioPaths]
        groupKFold = sklearn.model_selection.GroupKFold(n_splits=trainNSplit)
        groupKFold.get_n_splits(self.audioPaths, self.annotationPaths, groups)
        trainValIndexes, testIndexes = next(groupKFold.split(self.audioPaths, self.annotationPaths, groups))
        trainIndexes, valIndexes = sklearn.model_selection.train_test_split(
            trainValIndexes, test_size=validationSplit, random_state=randomState, shuffle=True
        )
        if limit is not None:
            trainIndexes = trainIndexes[:limit]
            valIndexes = valIndexes[:limit]
            testIndexes = testIndexes[:limit]

        return (trainIndexes, valIndexes, testIndexes)

    def getThreeSplitGen(self, **kwargs):
        """[summary]
        TODO
        """
        (trainIndexes, valIndexes, testIndexes) = self.getSplit(**kwargs)

        fullGenParams = {k: v for k, v in kwargs.items()}
        fullGenParams["repeat"] = False
        fullGenParams["samplePerTrack"] = None
        fullGenParams["yDense"] = False

        return (
            self.getGen(trainIndexes, **kwargs),
            self.getGen(valIndexes, **kwargs),
            self.getGen(valIndexes, **fullGenParams),
        )

    def getGen(
        self,
        trackIndexes=None,
        samplePerTrack=100,
        context=25,
        balanceClassesDistribution=False,
        classWeights=[2, 4],
        repeat=True,
        **kwargs,
    ):
        """
        TODO
        """
        buffer = {}
        if trackIndexes is None:
            trackIndexes = list(range(len(self.audioPaths)))

        def gen():
            cursors = {}  # The cursors dictionnary are stored in the gen to make it able to reinitialize
            while True:  # Infinite yield of samples
                for trackIdx in trackIndexes:  # go once each track in the split before restarting
                    # Cache dictionnary for lazy loading. Stored outside of the gen function to persist between dataset reset.
                    # Get the current track in the buffer, or load it from disk if the buffer is empty
                    if trackIdx not in buffer:
                        X, Y = self.readTrack(trackIdx, context=context, **kwargs)
                        indexes = self._balanceDistribution(X, Y) if balanceClassesDistribution else []
                        buffer[trackIdx] = {"x": X, "y": Y, "indexes": indexes, "name": self.audioPaths[trackIdx]}

                    # Set the cursor to the beginning of the track if it has not been read since the last reinitialisation
                    if trackIdx not in cursors:
                        cursors[trackIdx] = 0
                    track = buffer[trackIdx]

                    # Yield the specified number of samples per track, save the cursor to resume on the same location,
                    if samplePerTrack is not None:
                        for _ in range(samplePerTrack):
                            cursor = cursors[trackIdx]
                            if balanceClassesDistribution:
                                # track["cursor"] = (cursor + 1) % len(track["indexes"])
                                # sampleIdx = track["indexes"][cursor]
                                raise NotImplementedError()
                            else:
                                if cursor + 1 >= (len(track["x"]) - context) or cursor + 1 >= len(track["y"]):
                                    cursors[trackIdx] = 0
                                else:
                                    cursors[trackIdx] = cursor + 1
                                sampleIdx = cursor

                            y = track["y"][sampleIdx]
                            sampleWeight = np.array([max(np.sum(y * classWeights), 1)])
                            yield track["x"][sampleIdx : sampleIdx + context], y, sampleWeight
                    else:  # Yield the full track split in overlapping chunks with context
                        yield np.array([track["x"][i : i + context] for i in range(len(track["x"]) - context)]), track["y"]
                if not repeat:
                    break

        return gen

    def _balanceDistribution(self, X, Y):
        """ 
        balance the distribution of the labels Y by removing the labels without events such as there is only half of them empty.
        """
        nonEmptyIndexes = [i for i, row in enumerate(Y) if np.max(row) == 1]
        emptyIndexes = [(nonEmptyIndexes[i] + nonEmptyIndexes[i + 1]) // 2 for i in range(len(nonEmptyIndexes) - 1)]
        idxUsed = np.array(list(zip(nonEmptyIndexes, emptyIndexes))).flatten()
        return np.unique(idxUsed)

    def getClassWeight(self, sampleRate=100, labels=[36]):
        """
        Approach from https://markcartwright.com/files/cartwright2018increasing.pdf section 3.4.1 Task weights, adapted to compute class weights
        Compute the inverse estimated entropy of each label activity distribution

        """
        tr = TextReader()

        tracks = config.getFilesInFolder(self.folderPath, config.AUDIO)
        drums = config.getFilesInFolder(self.folderPath, config.ALIGNED_DRUM)
        # Getting the intersection of audio and annotations files
        tracks, drums = config.getIntersectionOfPaths(tracks, drums)

        tracksNotes = [tr.getOnsets(drumFile) for drumFile in drums]  # Get the dict of notes' events
        timeSteps = np.sum([librosa.get_duration(filename=track) for track in tracks]) * sampleRate
        result = []
        for label in labels:
            n = np.sum([len(trackNotes[label]) for trackNotes in tracksNotes])
            p = n / timeSteps
            y = 1 / (-p * np.log(p) - (1 - p) * np.log(1 - p))
            result.append(y)
        print(result)  # [10.780001453213364, 13.531086684241876, 34.13723052423422, 11.44276962353584, 17.6755104053326]
        return result

    def vizDataset(self, samples=100, labels=[36], sampleRate=50, condensed=False):
        gen = getGen(train=False, labels=labels, sampleRate=sampleRate, midiLatency=10)()

        X = []
        Y = []
        if condensed:
            for i in range(samples):
                x, y = next(gen)
                X.append(x[0].reshape((168)))
                Y.append(y)

            plt.matshow(np.array(X).T)
            print(np.sum(Y))
            for i in range(len(Y[0])):
                times = [t for t, y in enumerate(Y) if y[i]]
                plt.plot(times, np.ones(len(times)) * i * 10, "or")
            plt.show()
        else:
            fig = plt.figure(figsize=(8, 8))
            columns = 2
            rows = 5
            for i in range(1, columns * rows + 1):
                fig.add_subplot(rows, columns, i)
                x, y = next(gen)

                plt.imshow(np.reshape(x, (x.shape[0], x.shape[2])))
                if y[0]:
                    plt.plot([0], [0], "or")
            plt.show()
