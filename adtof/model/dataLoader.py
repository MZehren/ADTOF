import logging
import os
from collections import defaultdict
from typing import List

import librosa
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import tensorflow as tf
from adtof import config
from adtof.io.mir import MIR
from adtof.io.textReader import TextReader
import pandas as pd


class DataLoader(object):
    @classmethod
    def factoryADTOF(cls, folderPath: str, **kwargs):
        """instantiate a DataLoader following ADTOF folder hierarchy

        Parameters
        ----------
        folderPath : path to the root folder of the ADTOF dataset
        """
        return cls(
            os.path.join(folderPath, config.AUDIO),
            os.path.join(folderPath, config.ALIGNED_DRUM),
            os.path.join(folderPath, config.MANUAL_SUBSTRACTION),
            os.path.join(folderPath, config.PROCESSED_AUDIO),
            **kwargs,
        )

    @classmethod
    def factoryPublicDatasets(cls, folderPath: str, **kwargs):
        """instantiate a DataLoader following RBMA, ENST, and MDB folder hierarchies

        Parameters
        ----------
        folderPath : path to the root folder containing the public datasets
        """
        rbma = cls(
            os.path.join(folderPath, "rbma_13/audio"),
            os.path.join(folderPath, "rbma_13/annotations/drums_m"),
            None,
            os.path.join(folderPath, "rbma_13/preprocess"),
            folds=[[1, 7, 12, 21, 3, 10, 29, 24, 20], [11, 28, 18, 13, 23, 9, 4, 14, 26], [0, 2, 15, 17, 19, 8, 27, 22, 16]],
            mappingDictionaries=[config.RBMA_MIDI_8, config.MIDI_REDUCED_5],
            sep="\t",
            **kwargs,
        )
        return rbma

    def __init__(
        self,
        audioFolder: str,
        annotationFolder: str = None,
        blockListPath: str = None,
        cachePreprocessPath: str = None,
        checkFilesNameMatch: bool = True,
        folds: List = None,
        fmin=20,
        fmax=20000,
        **kwargs,
    ):
        """
        Class for the handling of building a dataset for training or infering
        TODO
        """
        # Fetch audio paths
        self.audioPaths = config.getFilesInFolder(audioFolder)
        self.annotationPaths = None
        self.preprocessPaths = None

        # Fetch annotation paths
        if annotationFolder is not None:
            # Getting the intersection of audio and annotations files
            self.annotationPaths = config.getFilesInFolder(annotationFolder)
            if checkFilesNameMatch:
                self.audioPaths, self.annotationPaths = config.getIntersectionOfPaths(self.audioPaths, self.annotationPaths)

        # Remove track from the blocklist manually checked
        if blockListPath is not None and os.path.exists(blockListPath):
            toRemove = set(pd.read_csv(blockListPath, sep="No separator workaround", header=None)[0])
            self.audioPaths = [f for f in self.audioPaths if config.getFileBasename(f) not in toRemove]
            if annotationFolder is not None:
                self.annotationPaths = [f for f in self.annotationPaths if config.getFileBasename(f) not in toRemove]

        # Set the cache paths to store pre processed audio
        if cachePreprocessPath is not None:
            self.preprocessPaths = np.array(
                [
                    os.path.join(cachePreprocessPath + str(fmin) + "-" + str(fmax), config.getFileBasename(track) + ".npy")
                    for track in self.audioPaths
                ]
            )

        # split the data according to the specified folds or generate then
        if folds == None:
            self.folds = self._getFolds(**kwargs)
        else:
            self.folds = folds

        # load in memory all the data to build the samples
        self.data = [self.readTrack(i, **kwargs) for i in range(len(self.audioPaths))]

    def readTrack(self, trackIdx, removeStart=True, labelOffset=0, sampleRate=100, **kwargs):
        """
        Read all the info of the track used for training and evaluation
        """
        name = self.audioPaths[trackIdx]
        x = self.readAudio(trackIdx, sampleRate=sampleRate, **kwargs)
        if self.annotationPaths is not None:
            y = self.readLabels(trackIdx, **kwargs)
            if labelOffset:
                timeOffset = labelOffset / sampleRate
                for k, v in y.items():
                    y[k] = np.array(v) - timeOffset
            if removeStart:
                x, y = self.removeStart(x, y, sampleRate=sampleRate, **kwargs)
            # TODO Is it optimised to keep y in dense and sparse form?
            yDense = self.getDenseEncoding(name, y, sampleRate=sampleRate, **kwargs)
            return {"x": x, "y": y, "yDense": yDense, "name": name}
        else:
            return {"x": x, "y": None, "name": name}

    def readAudio(self, i, sampleRate=100, **kwargs):
        """
        Read the track audio
        """
        mir = MIR(frameRate=sampleRate, **kwargs)
        x = mir.open(self.audioPaths[i], cachePath=self.preprocessPaths[i] if self.preprocessPaths is not None else None)
        x = x.reshape(x.shape + (1,))  # Add the channel dimension
        return x

    def readLabels(self, i, **kwargs):
        """
        get the track annotations
        """
        y = TextReader().getOnsets(self.annotationPaths[i], **kwargs)
        return y

    def removeStart(self, x, y, sampleRate=100, context=25, **kwargs):
        """
        Trim X to start and end on notes from notes
        Change the time of notes to start at 0
        """
        # Trim before the first note to remove count in
        # Move the trim by the offset amount to keep the first notation
        firstNoteTime = np.min([v[0] for v in y.values() if len(v)])
        firstNoteTime = max(0, firstNoteTime)
        firstNoteIdx = int(round(firstNoteTime * sampleRate))

        # Trim after the last note to remove all part of the track not annotated
        # Make sure the index doesn't exceed any boundaries
        # TODO is it necessary, or do we want to keep all the audio?
        lastNoteTime = np.max([v[-1] for v in y.values() if len(v)])
        lastNoteIdx = min(int(lastNoteTime * sampleRate) + 1, len(x) - 1 - context)

        X = x[firstNoteIdx : lastNoteIdx + context]
        for k, v in y.items():
            y[k] = v - (firstNoteTime)
        return (X, y)

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
            # if len(notes[key]) == 0:
            #     logging.debug("Pitch %s is not represented in the track %s", key, filename)
        return np.array(result).T

    def _getFolds(self, nFolds=10, **kwargs):
        """
        Split the data in groups without band overlap between split
        """
        groups = [config.getBand(path) for path in self.audioPaths]
        groupKFold = sklearn.model_selection.GroupKFold(n_splits=nFolds)
        # realNFolds = groupKFold.get_n_splits(self.audioPaths, self.annotationPaths, groups)

        return [fold for _, fold in groupKFold.split(self.audioPaths, self.annotationPaths, groups)]

    def getSplit(self, testFold=0, validationFold=0, **kwargs):
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
        folds = self.folds
        testIndexes = folds.pop()
        valIndexes = folds.pop(validationFold)
        trainIndexes = np.concatenate(folds)

        # Shuffle indexes such as the tracks from the same band will not occur in sequence
        trainIndexes = sklearn.utils.shuffle(trainIndexes, random_state=0)
        valIndexes = sklearn.utils.shuffle(valIndexes, random_state=0)
        testIndexes = sklearn.utils.shuffle(testIndexes, random_state=0)

        # Limit the number of files as it can be memory intensive
        # The limit is done after the split to ensure no test data leak in the train set with different values of limit.
        # if tracksLimit is not None:
        #     trainIndexes = trainIndexes[:tracksLimit]
        #     valIndexes = valIndexes[:tracksLimit]
        #     testIndexes = testIndexes[:tracksLimit]

        return (trainIndexes, valIndexes, testIndexes)

    def _getDataset(self, gen, trainingSequence=1, labels=[1, 1, 1, 1, 1], **kwargs):
        """
        Get a tf.dataset from the generator.
        Fill the right size for each dim
        """
        # time, feature, channel dimension
        xShape = (None, None, 1)
        yShape = (len(labels),) if trainingSequence == 1 else (trainingSequence, len(labels))
        wShape = ((1),)
        return tf.data.Dataset.from_generator(
            gen,
            (tf.float32, tf.float32, tf.float32),
            output_shapes=(tf.TensorShape(xShape), tf.TensorShape(yShape), tf.TensorShape(wShape)),
        )

    def getTrainValTestGens(self, batchSize=None, **kwargs):
        """
        Return 4 generators to perform training and validation:
        trainGen : dataset for training 
        valGen : dataset for validation 
        valFullGen : Finit generator giving full tracks for fitting peak picking
        testFullGen : Finit generator giving full tracks for computing final result
        """
        (trainIndexes, valIndexes, testIndexes) = self.getSplit(**kwargs)

        fullGenParams = {k: v for k, v in kwargs.items()}
        fullGenParams["repeat"] = False
        fullGenParams["samplePerTrack"] = None

        trainGen, valGen, valFullGen, testFullGen = (
            self.getGen(trainIndexes, **kwargs),
            self.getGen(valIndexes, **kwargs),
            self.getGen(valIndexes, **fullGenParams),
            self.getGen(testIndexes, **fullGenParams),
        )

        dataset_train = self._getDataset(trainGen, **kwargs)
        dataset_val = self._getDataset(valGen, **kwargs)
        dataset_train = dataset_train.batch(batchSize).repeat()
        dataset_val = dataset_val.batch(batchSize).repeat()
        dataset_train = dataset_train.prefetch(buffer_size=2)
        dataset_val = dataset_val.prefetch(buffer_size=2)

        return (dataset_train, dataset_val, valFullGen(), testFullGen())  # TODO should be datasets instead of gen?

    def getGen(
        self,
        trackIndexes=None,
        samplePerTrack=100,
        context=25,
        trainingSequence=1,
        classWeights=np.array([1, 1, 1, 1, 1]),
        repeat=True,
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
            Specifies how large the context is, by default 25
        trainingSequence : int, optional
            Specifies how many context frames are merged together, by default 1
        classWeights : [type], optional
            the sample weight is computed by min(1, sum(classWeights * y)), by default np.array([1, 1, 1, 1, 1])
        repeat : bool, optional
            If False, stop after seeing each track., by default True

        Returns
        -------
        generator yielding unique samples(x, y, w)

        TODO:
        if balanceClassesDistribution:
            # track["cursor"] = (cursor + 1) % len(track["indexes"])
            # sampleIdx = track["indexes"][cursor]
            raise NotImplementedError()
        """
        if trackIndexes is None:
            trackIndexes = list(range(len(self.audioPaths)))

        # Compute how large the x time dimension has to be by merging [trainingSequence] frames of context
        xWindowSize = context + (trainingSequence - 1)
        yWindowSize = trainingSequence

        def gen():
            cursors = {}  # The cursors dictionnary are stored in the gen to make it able to reinitialize
            while True:  # Infinite yield of samples
                for trackIdx in trackIndexes:  # go once each track in the split before restarting
                    track = self.data[trackIdx]
                    # Set the cursor in the middle of the track if it has not been read since the last reinitialisation
                    if trackIdx not in cursors:
                        cursors[trackIdx] = (len(track["x"]) - xWindowSize) // 2

                    # If the track is shorter than the context used to compute the output, we skip it
                    if (len(track["x"]) - xWindowSize) < 0:
                        continue

                    # Yield the specified number of samples per track, save the cursor to resume on the same location,
                    if samplePerTrack is not None:
                        for _ in range(samplePerTrack):
                            # Compute the next index
                            sampleIdx = cursors[trackIdx]
                            nextIdx = sampleIdx + yWindowSize
                            if nextIdx + xWindowSize >= len(track["x"]) or nextIdx + yWindowSize >= len(track["y"]):
                                cursors[trackIdx] = 0
                            else:
                                cursors[trackIdx] = nextIdx

                            x = track["x"][sampleIdx : sampleIdx + xWindowSize]
                            y = track["yDense"][sampleIdx] if yWindowSize == 1 else track["y"][sampleIdx : sampleIdx + yWindowSize]

                            # TODO Could be faster by caching the results since the weight or target is not changing.
                            sampleWeight = np.array([max(np.sum(y * classWeights), 1)])  # /yWindowSize
                            # sampleWeight = sum([act * classWeights[i] for i, act in enumerate(y) if act > 0])
                            # sampleWeight = np.array([max(sampleWeight, 1)])

                            yield x, y, sampleWeight

                    else:  # Yield the full track split in overlapping chunks with context
                        # Returning a number of samples multiple of batch size to enable hardcoded batchSize in the model.
                        # used by the tf statefull RNN
                        # TODO: correct valid padding alignment?
                        yield (track["x"], track["y"])
                        # totalSamples = len(track["x"]) - context
                        # usableSamples = totalSamples - totalSamples % batchSize
                        # yield np.array([track["x"][i : i + context] for i in range(totalSamples)]), track["y"]
                if not repeat:
                    break

        return gen

    def _balanceDistribution(self, X, Y):
        """ 
        balance the distribution of the labels Y by removing the labels without events such as there is only half of them empty.
        """
        raise NotImplementedError()
        nonEmptyIndexes = [i for i, row in enumerate(Y) if np.max(row) == 1]
        emptyIndexes = [(nonEmptyIndexes[i] + nonEmptyIndexes[i + 1]) // 2 for i in range(len(nonEmptyIndexes) - 1)]
        idxUsed = np.array(list(zip(nonEmptyIndexes, emptyIndexes))).flatten()
        return np.unique(idxUsed)

    def getClassWeight(self, sampleRate=100, labels=[36]):
        """
        Approach from https://markcartwright.com/files/cartwright2018increasing.pdf section 3.4.1 Task weights, adapted to compute class weights
        Compute the inverse estimated entropy of each label activity distribution
        """
        raise NotImplementedError()
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
