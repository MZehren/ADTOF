from collections import defaultdict
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
    def __init__(self, folderPath, loadLabels=True):
        """
        Class for the handling of the workflow while loading the dataset

        Parameters
        ----------
        folderPath : [str]
            Path to the root folder containing the dataset
        """
        self.folderPath = folderPath
        self.loadLabels = loadLabels
        if loadLabels:
            # Getting the intersection of audio and annotations files
            self.audioPaths = config.getFilesInFolder(self.folderPath, config.AUDIO)
            self.annotationPaths = config.getFilesInFolder(self.folderPath, config.ALIGNED_DRUM)
            self.audioPaths, self.annotationPaths = config.getIntersectionOfPaths(self.audioPaths, self.annotationPaths)
            self.featurePaths = np.array(
                [os.path.join(folderPath, config.PROCESSED_AUDIO, config.getFileBasename(track) + ".npy") for track in self.audioPaths]
            )
        else:
            self.audioPaths = config.getFilesInFolder(self.folderPath)

    def readTrack(self, trackIdx, removeStart=True, yDense=True, labelOffset=0, sampleRate=100, **kwargs):
        """
        Read all the info of the track used for training and evaluation
        """
        name = self.audioPaths[trackIdx]
        X = self.readAudio(trackIdx, sampleRate=sampleRate, **kwargs)
        if self.loadLabels:
            notes = self.readLabels(trackIdx, **kwargs)
            if labelOffset:
                timeOffset = labelOffset / sampleRate
                for k, v in notes.items():
                    notes[k] = np.array(v) - timeOffset
            if removeStart:
                X, notes = self.removeStart(X, notes, sampleRate=sampleRate, **kwargs)
            if yDense:
                notes = self.getDenseEncoding(name, notes, sampleRate=sampleRate, **kwargs)
            # indexes = self._balanceDistribution(X, Y) if balanceClassesDistribution else []
            return {"x": X, "y": notes, "name": name}
        else:
            return {"x": X, "y": None, "name": name}

    def readAudio(self, i, sampleRate=100, **kwargs):
        """
        Read the track audio
        """
        mir = MIR(frameRate=sampleRate, **kwargs)
        x = mir.open(self.audioPaths[i], cachePath=self.featurePaths[i] if self.loadLabels else None)
        x = x.reshape(x.shape + (1,))  # Add the channel dimension
        return x

    def readLabels(self, i, **kwargs):
        """
        get the track annotations
        """
        notes = TextReader().getOnsets(self.annotationPaths[i])
        return notes

    def removeStart(self, x, notes, sampleRate=100, context=25, **kwargs):
        """
        Trim X to start and end on notes from notes
        Change the time of notes to start at 0
        """
        # Trim before the first note to remove count in
        # Move the trim by the offset amount to keep the first notation
        firstNoteTime = np.min([v[0] for v in notes.values() if len(v)])
        firstNoteTime = max(0, firstNoteTime)
        firstNoteIdx = int(round(firstNoteTime * sampleRate))

        # Trim after the last note to remove all part of the track not annotated
        # Make sure the index doesn't exceed any boundaries
        # TODO is it necessary, or do we want to keep all the audio?
        lastNoteTime = np.max([v[-1] for v in notes.values() if len(v)])
        lastNoteIdx = min(int(lastNoteTime * sampleRate) + 1, len(x) - 1 - context)

        X = x[firstNoteIdx : lastNoteIdx + context]
        for k, v in notes.items():
            notes[k] = v - (firstNoteTime)
        return (X, notes)

    def getDenseEncoding(self, filename, notes, length=None, sampleRate=100, labels=[36, 40, 41, 46, 49], labelRadiation=1, **kwargs):
        """
        Encode in a dense matrix the midi onsets

        sampleRate = sampleRate
        offset = offset of the midi in s, so the event is actually annotated later
        keys = pitch of the offset in each column of the matrix
        labelRadiation = how many samples from the event have a non-null target
        TODO: Move that to io.midi
        """
        # if length not specified, make it big enough to store all the notes
        if length == None:
            lastNoteTime = np.max([values[-1] for values in notes.values()])
            length = int(np.round((lastNoteTime) * sampleRate)) + 1

        result = []
        for key in labels:
            # Size of the dense matrix
            row = np.zeros(length)
            for time in notes[key]:
                # indexs at 1 in the dense matrix
                index = int(np.round(time * sampleRate))
                if index < 0 or index >= len(row):
                    continue
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

    def getSplit(self, nFolds=10, validationFold=0, tracksLimit=None, **kwargs):
        """Return indexes of tracks for the train, validation and test splits from a k-fold scheme.
        There are no group overlap between folds

        Parameters
        ----------
        trainNSplit : int, optional
            Number of splits for the data, has to be >=3, by default 10. Keep it the same during training as changing it would shuffle the splits.
        validationFold : int, optional
            fold of the validation, has to be < trainNSplit -1 since one fold is saved for testing, by default 0
        limit : int, optional
            if present, limit the size of each split, by default None

        Returns
        -------
        (trainIndexes, validationIndexes, testIndexes)

        TODO: how to handle
        - Early stopping for the final model trained on the full training+validation fold wich overfit the test set
        - Multiple test set: Do the full hparam search on train+validation and then eval on test, change the split and start again.
        """

        # Split the data in train, validation and test, without same band in each split
        groups = [config.getBand(path) for path in self.audioPaths]
        groupKFold = sklearn.model_selection.GroupKFold(n_splits=nFolds)
        realNFolds = groupKFold.get_n_splits(self.audioPaths, self.annotationPaths, groups)

        # The data is split in train, val, test. Hence at least three groups are needed
        assert realNFolds >= 3
        assert validationFold < realNFolds - 1

        folds = [fold for _, fold in groupKFold.split(self.audioPaths, self.annotationPaths, groups)]
        testIndexes = folds.pop(0)
        valIndexes = folds.pop(validationFold)
        trainIndexes = np.concatenate(folds)

        # Shuffle indexes such as the tracks from the same band will not occur in sequence
        trainIndexes = sklearn.utils.shuffle(trainIndexes, random_state=0)
        valIndexes = sklearn.utils.shuffle(valIndexes, random_state=0)
        testIndexes = sklearn.utils.shuffle(testIndexes, random_state=0)

        # Limit the number of files as it can be memory intensive
        # The limit is done after the split to ensure no test data leak in the train set with different values of limit.
        if tracksLimit is not None:
            trainIndexes = trainIndexes[:tracksLimit]
            valIndexes = valIndexes[:tracksLimit]
            testIndexes = testIndexes[:tracksLimit]

        return (trainIndexes, valIndexes, testIndexes)

    def getTrainValTestGens(self, **kwargs):
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
            self.getGen(testIndexes, **fullGenParams),
        )

    def getGen(
        self,
        trackIndexes=None,
        samplePerTrack=100,
        context=25,
        balanceClassesDistribution=False,
        classWeights=np.array([1, 1, 1, 1, 1]),
        repeat=True,
        batchSize=1,
        **kwargs,
    ):
        """
        Return an infinite generator yielding samples

        Parameters
        ----------
        trackIndexes : list(int), optional
            Specifies which track have to be selected based on their index. (see self.getSplit to get indexes), by default None
        samplePerTrack : int, optional
            Specifies how many samples has to be returned before moving to the next track, the generator is infinite and will resume where it stopped at each track.
            If None, the full track is returned without windowing. by default 100 TODO: Make this behavior more explicit
        context : int, optional
            Specifies how large the context is,, by default 25
        balanceClassesDistribution : bool, optional
            Not implemented, by default False
        classWeights : [type], optional
            the sample weight is computed by min(1, sum(classWeights * y)), by default np.array([1, 1, 1, 1, 1])
        repeat : bool, optional
            If False, stop after seeing each track., by default True
        batchSize : int, optional
            Not implemented, by default 1

        Returns
        -------
        generator yielding unique samples(x, y, w)
        """
        cache = {}
        if trackIndexes is None:
            trackIndexes = list(range(len(self.audioPaths)))

        def gen():
            cursors = {}  # The cursors dictionnary are stored in the gen to make it able to reinitialize
            while True:  # Infinite yield of samples
                for trackIdx in trackIndexes:  # go once each track in the split before restarting
                    # Cache dictionnary for lazy loading. Stored outside of the gen function to persist between dataset reset.
                    # Get the current track in the buffer, or load it from disk if the buffer is empty
                    if trackIdx not in cache:
                        cache[trackIdx] = self.readTrack(trackIdx, **kwargs)
                    track = cache[trackIdx]

                    # Yield the specified number of samples per track, save the cursor to resume on the same location,
                    if samplePerTrack is not None:
                        # Set the cursor in the middle of the track if it has not been read since the last reinitialisation
                        if trackIdx not in cursors:
                            cursors[trackIdx] = min((len(track["x"]) - context), len(track["y"])) // 2
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
                            # sampleWeight = np.array([max(np.sum(y * classWeights), 1)])
                            # TODO Could be faster by caching the results since the weight or target is not changing.
                            sampleWeight = sum([act * classWeights[i] for i, act in enumerate(y) if act > 0])
                            sampleWeight = np.array([max(sampleWeight, 1)])

                            yield track["x"][sampleIdx : sampleIdx + context], y, sampleWeight
                    else:  # Yield the full track split in overlapping chunks with context
                        # Returning a number of samples multiple of batch size to enable hardcoded batchSize in the model.
                        # used by the tf statefull RNN
                        totalSamples = len(track["x"]) - context
                        usableSamples = totalSamples - totalSamples % batchSize
                        yield np.array([track["x"][i : i + context] for i in range(totalSamples)]), track["y"]
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
        gen = self.getGen(train=False, labels=labels, sampleRate=sampleRate, midiLatency=10)()

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
