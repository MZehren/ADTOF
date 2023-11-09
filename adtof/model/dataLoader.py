import collections
import logging
import os
import random
from collections import defaultdict, namedtuple
from typing import Iterable, List

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from adtof import config
from adtof.ressources import instrumentsMapping, splits
from adtof.converters.madmomBeatConverter import MadmomBeatConverter
from adtof.io import mir
from adtof.io.textReader import TextReader
from adtof.model.lazyDict import LazyDict
from adtof.model.track import Track
from sklearn.utils import shuffle
import glob
import re
from adtof.model import dataAugmentation as da
import sklearn.model_selection
from collections import Counter
import time


class DataLoader(object):
    @classmethod
    def getAllDatasets(cls, folderPath, testFold=0, validationRatio=0.15, GTBeats=False, n_splits=10, **kwargs):
        """
        Build a dataset for all the known datasets

        folderPath: location where all the datasets (adtof, ENST, MDB, rbma, TMIDT) are stored
        testFold: which fold to test on
        validationRatio: the ratio of the training set to use for validation if given a float. Or which fold to use for training if given an integer.
        GTBeats: If using ground truth beats location or Madmom's
        """
        dataLoaded = {}

        # try:
        #     dataLoaded["slakh2100_solo"], dataLoaded["slakh2100_acc"] = cls.getSlakh2100(folderPath, n_splits, testFold, **kwargs)
        # except Exception as e:
        #     logging.error("Could not load slakh2100 dataset", e)

        # try:
        #     dataLoaded["egmd"] = cls.getEGMD(folderPath, n_splits, testFold, **kwargs)
        # except Exception as e:
        #     logging.error("Could not load egmd dataset", e)

        try:
            dataLoaded["adtos"] = cls.getADTOS(folderPath, n_splits, testFold, GTBeats=GTBeats, **kwargs)
        except Exception as e:
            logging.error("Could not load adtos dataset", e)

        try:
            dataLoaded["tmidt_solo"], dataLoaded["tmidt_acc"] = cls.getTMIDT(folderPath, n_splits, testFold, **kwargs)
        except Exception as e:
            logging.error("Could not load tmidt dataset", e)

        try:
            dataLoaded["adtof_F70"], dataLoaded["adtof_yt"] = cls.getADTOF(folderPath, n_splits, testFold, GTBeats=GTBeats, **kwargs)
        except Exception as e:
            logging.error("Could not load adtof dataset", e)

        try:
            dataLoaded["rbma"] = cls(
                os.path.join(folderPath, "rbma_13/audio/*.mp3"),
                os.path.join(folderPath, "rbma_13/annotations/drums_f/*.txt"),
                cachePreprocessFolders=os.path.join(folderPath, "rbma_13/preprocess/"),
                beatPaths=os.path.join(folderPath, "rbma_13/annotations public/beats", "*.txt") if GTBeats else os.path.join(folderPath, "rbma_13/madmom_beats", "*.txt"),
                splits=splits.RBMA_SPLITS,
                mappingDictionaries=[instrumentsMapping.RBMA_FULL_MIDI, instrumentsMapping.MIDI_REDUCED_5],
                sep="\t",
                validationFold=validationRatio,
                testFold=testFold,
                **kwargs,
            )
            dataLoaded["mdb_full_mix"] = cls(
                os.path.join(folderPath, "MDB Drums/audio/full_mix/*"),
                os.path.join(folderPath, "MDB Drums/annotations/subclass/*"),
                cachePreprocessFolders=os.path.join(folderPath, "MDB Drums/audio/preprocess"),
                beatPaths=os.path.join(folderPath, "MDB Drums/madmom_beats/full_mix", "*.txt"),
                splits=splits.MDB_SPLITS_MIX,
                mappingDictionaries=[instrumentsMapping.MDBS_MIDI, instrumentsMapping.MIDI_REDUCED_5],
                sep="\t",
                validationFold=validationRatio,
                testFold=testFold,
                checkFilesNameMatch=False,
                **kwargs,
            )
            # dataLoaded["mdb_drum_solo"] = cls(
            #     os.path.join(folderPath, "MDB Drums/audio/drum_only/*"),
            #     os.path.join(folderPath, "MDB Drums/annotations/subclass/*"),
            #     cachePreprocessFolders=os.path.join(folderPath, "MDB Drums/audio/preprocess_solo"),
            #     beatPaths=os.path.join(folderPath, "MDB Drums/madmom_beats/drum_only", "*.txt"),
            #     splits=splits.MDB_SPLITS_DRUM_ONLY,
            #     mappingDictionaries=[instrumentsMapping.MDBS_MIDI, instrumentsMapping.MIDI_REDUCED_5],
            #     sep="\t",
            #     validationFold=validationRatio,
            #     testFold=testFold,
            #     checkFilesNameMatch=False,
            #     **kwargs,
            # )
            dataLoaded["enst_sum"] = cls(
                os.path.join(folderPath, "ENST-drums-public/audio_sum/*"),
                os.path.join(folderPath, "ENST-drums-public/annotations/*"),
                cachePreprocessFolders=os.path.join(folderPath, "ENST-drums-public/preprocessSum"),
                beatPaths=os.path.join(folderPath, "ENST-drums-public/madmom_beats/audio_sum", "*.txt"),
                splits=splits.ENST_SPLITS,
                mappingDictionaries=[instrumentsMapping.ENST_MIDI, instrumentsMapping.MIDI_REDUCED_5],
                sep=" ",
                validationFold=validationRatio,
                testFold=testFold,
                **kwargs,
            )
            dataLoaded["enst_wet"] = cls(
                os.path.join(folderPath, "ENST-drums-public/audio_wet/*"),
                os.path.join(folderPath, "ENST-drums-public/annotations/*"),
                cachePreprocessFolders=os.path.join(folderPath, "ENST-drums-public/preprocessWet"),
                beatPaths=os.path.join(folderPath, "ENST-drums-public/madmom_beats/audio_wet", "*.txt"),
                splits=splits.ENST_SPLITS,
                mappingDictionaries=[instrumentsMapping.ENST_MIDI, instrumentsMapping.MIDI_REDUCED_5],
                sep=" ",
                validationFold=validationRatio,
                testFold=testFold,
                **kwargs,
            )
        except Exception as e:
            logging.error("Error while loading rbma, tmidt, or enst dataset", e)

        for name, dataset in dataLoaded.items():
            logging.debug("Dataset {} loaded, {} test, {} val, {} train".format(name, len(dataset.testIndexes), len(dataset.valIndexes), len(dataset.trainIndexes)))
            # dataset.data[0]
        return dataLoaded

    @classmethod
    def getEGMD(cls, folderPath, n_splits, testFold, ratioKits=1, ratioSequences=1, scenario="all", **kwargs):
        # Get all tracks
        tracks = config.getFilesInFolder(os.path.join(folderPath, "e-gmd-v1.0.0", "drummer*", "session*", "*.wav"))
        tracks = [t for t in tracks if librosa.get_duration(path=t) > 4]  # Remove track smaller than 4s
        plugin = [re.findall("_([0-9]+).wav", path)[0] for path in tracks]  # Get plugin number
        egmd_splits = None
        validationFold = testFold + 1

        if scenario == "egmd":  # Get only a subset of kits and sequences
            sequence = ["_".join(path.split("/")[-1].split("_")[:-1]) for path in tracks]  # Get sequence name
            allowedPlugins = sorted(list(set(plugin)))
            allowedPlugins = set(allowedPlugins[: int(len(allowedPlugins) * ratioKits)])
            allowedSequences = sorted(list(set(sequence)))
            allowedSequences = set(allowedSequences[: int(len(allowedSequences) * ratioSequences)])
            tracks = [str(t) for t, p, s in zip(tracks, plugin, sequence) if p in allowedPlugins and s in allowedSequences]
            testFold = -1  # No test set
            validationFold = 0.0  # No validation set
        else:  # Split data without overlap of kits
            groupKFold = sklearn.model_selection.GroupKFold(n_splits=n_splits)
            egmd_splits = [[config.getFileBasename(tracks[i]) for i in fold] for _, fold in groupKFold.split(tracks, groups=plugin)]
            # Split data without overlap of sequence
            availableSeqences = sorted(list(set(["_".join(config.getFileBasename(t).split("_")[:-1]) for t in tracks])))
            n_sequences_per_split = len(availableSeqences) / n_splits
            splitSequences = [set(availableSeqences[int(i * n_sequences_per_split) : int((i + 1) * n_sequences_per_split)]) for i in range(n_splits)]
            egmd_splits = [[t for t in split if "_".join(t.split("_")[:-1]) in splitSequences[i]] for i, split in enumerate(egmd_splits)]

        return cls(
            tracks,
            annotationPaths=os.path.join(folderPath, "e-gmd-v1.0.0", "drummer*", "*", "*.midi"),
            cachePreprocessFolders=os.path.join(folderPath, "e-gmd-v1.0.0", "preprocess"),
            mappingDictionaries=[instrumentsMapping.EGMD_MIDI, instrumentsMapping.MIDI_REDUCED_5],
            splits=egmd_splits,
            testFold=testFold,
            validationFold=validationFold,
            # minVelocity=0.33,
            **kwargs,
        )

    @classmethod
    def getADTOS(cls, folderPath, n_splits, testFold, **kwargs):
        # Get all tracks
        tracks = config.getFilesInFolder(os.path.join(folderPath, "adtos/master/*.mp3"))
        plugins = ["_".join(p.split("_")[1:]) for p in tracks]

        # Split data without overlap of kits
        groupKFold = sklearn.model_selection.GroupKFold(n_splits=n_splits)
        splits = [[config.getFileBasename(tracks[i]) for i in fold] for _, fold in groupKFold.split(tracks, groups=plugins)]

        acc = cls(
            tracks,
            annotationPaths=[os.path.join(folderPath, "adtos", "midi", os.path.basename(t).split("_")[0] + "_drums.mid") for t in tracks],
            beatPaths=[os.path.join(folderPath, "adtos", "midi", os.path.basename(t).split("_")[0] + "_drums.mid") for t in tracks],
            cachePreprocessFolders=os.path.join(folderPath, "adtos", "preprocessedAcc"),
            mappingDictionaries=[instrumentsMapping.EZDRUMMER_MIDI, instrumentsMapping.MIDI_REDUCED_5],
            splits=splits,
            testFold=testFold,
            validationFold=testFold + 1,
            getGroup=plugins,
            checkFilesNameMatch=False,
            checkInstrumentIsDrum=False,
            **kwargs,
        )
        return acc

    @classmethod
    def getSlakh2100(cls, folderPath, n_splits, testFold, **kwargs):
        # Get all tracks
        def getPlugin(path):
            with open(path) as f:
                data = yaml.load(f, Loader=SafeLoader)
                return {v["inst_class"]: v["plugin_name"] for k, v in data["stems"].items()}["Drums"]

        tracks = config.getFilesInFolder(os.path.join(folderPath, "Slakh/metadata/Track*.yaml"))
        plugins = [getPlugin(p) for p in tracks]
        slakhTrackMapping = [[instrumentsMapping.SLAKH2100_MIDI[plugin], instrumentsMapping.MIDI_REDUCED_5] for plugin in plugins]

        # Split data without overlap of kits
        groupKFold = sklearn.model_selection.GroupKFold(n_splits=n_splits)
        slakh_splits = [[config.getFileBasename(tracks[i]) for i in fold] for _, fold in groupKFold.split(tracks, groups=plugins)]
        # validationFold = testFold + 1

        # Train on all data
        testFold = -1
        validationFold = 0.0

        solo = cls(
            [os.path.join(folderPath, "Slakh", "solo", config.getFileBasename(t) + ".flac") for t in tracks],
            annotationPaths=os.path.join(folderPath, "Slakh", "midi", "Track*.mid"),
            cachePreprocessFolders=os.path.join(folderPath, "Slakh", "preprocessedSolo"),
            trackMapping=slakhTrackMapping,
            splits=slakh_splits,
            testFold=testFold,
            validationFold=validationFold,
            getGroup=plugins,
            # minVelocity=0.33,
            **kwargs,
        )

        acc = cls(
            [os.path.join(folderPath, "Slakh", "acc", config.getFileBasename(t) + ".flac") for t in tracks],
            annotationPaths=os.path.join(folderPath, "Slakh", "midi", "Track*.mid"),
            cachePreprocessFolders=os.path.join(folderPath, "Slakh", "preprocessedAcc"),
            trackMapping=slakhTrackMapping,
            splits=slakh_splits,
            testFold=testFold,
            validationFold=validationFold,
            getGroup=plugins,
            # minVelocity=0.33,
            **kwargs,
        )
        return solo, acc

    @classmethod
    def getADTOF(cls, folderPath, n_splits, testFold, removeStart=True, GTBeats=False, **kwargs):
        # Generate the splits for adtof and adtof_yt to avoid band overlap
        mergedAudioPath = (
            config.getFilesInFolder(os.path.join(folderPath, "adtofParsedCC", config.AUDIO, "*.ogg")).tolist()
            + config.getFilesInFolder(os.path.join(folderPath, "adtofParsedYT", config.AUDIO, "*.ogg")).tolist()
        )
        groups = [config.getBand(path).lower() for path in mergedAudioPath]
        groupKFold = sklearn.model_selection.GroupKFold(n_splits=n_splits)
        adtof_splits = [[config.getFileBasename(mergedAudioPath[i]) for i in fold] for _, fold in groupKFold.split(mergedAudioPath, groups=groups)]

        adtof_F70 = cls(
            os.path.join(folderPath, "adtofParsedCC", config.AUDIO, "*.ogg"),
            os.path.join(folderPath, "adtofParsedCC", config.ALIGNED_DRUM, "*.txt"),
            os.path.join(folderPath, "adtofParsedCC", config.MANUAL_SUBSTRACTION),
            os.path.join(folderPath, "adtofParsedCC", config.PROCESSED_AUDIO),
            beatPaths=os.path.join(folderPath, "adtofParsedCC", config.ALIGNED_BEATS, "*.txt") if GTBeats else os.path.join(folderPath, "adtofParsedCC/estimations/beats", "*.txt"),
            testFold=testFold,
            validationFold=testFold + 1,  # Validation set is not 15% of training, but a separated split without leaked bands
            splits=adtof_splits,
            removeStart=removeStart,
            mappingDictionaries=[instrumentsMapping.MIDI_REDUCED_5],
            **kwargs,
        )

        adtof_yt = cls(
            os.path.join(folderPath, "adtofParsedYT", config.AUDIO, "*.ogg"),
            annotationPaths=os.path.join(folderPath, "adtofParsedYT", config.ALIGNED_DRUM, "*.txt"),
            cachePreprocessFolders=os.path.join(folderPath, "adtofParsedYT/" + config.PROCESSED_AUDIO),
            beatPaths=os.path.join(folderPath, "adtofParsedYT", config.ALIGNED_BEATS, "*.txt") if GTBeats else os.path.join(folderPath, "adtofParsedYT/estimations/beats", "*.txt"),
            testFold=testFold,
            validationFold=testFold + 1,
            splits=adtof_splits,
            removeStart=removeStart,
            mappingDictionaries=[instrumentsMapping.MIDI_REDUCED_5],
            **kwargs,
        )

        # TODO: compute the estimation of the beats for adtof_rb
        # TODO: test with converted drums before alignemnt.
        # adtof_rb = cls(
        #     os.path.join(folderPath, "adtofParsedRB", config.AUDIO, "*.ogg"),
        #     annotationPaths=os.path.join(folderPath, "adtofParsedRB", config.ALIGNED_DRUM, "*.txt"),
        #     cachePreprocessFolders=os.path.join(folderPath, "adtofParsedRB/" + config.PROCESSED_AUDIO),
        #     beatPaths=os.path.join(folderPath, "adtofParsedRB", config.ALIGNED_BEATS, "*.txt") if True else os.path.join(folderPath, "adtofParsedRB/estimations/beats", "*.txt"),
        #     testFold=testFold,
        #     validationFold=testFold + 1,
        #     # splits=adtof_splits,
        #     removeStart=removeStart,
        #     mappingDictionaries=[instrumentsMapping.MIDI_REDUCED_5],
        #     **kwargs,
        # )

        # # check if files are overlapping
        # testA = [adtof_F70.audioPaths[i] for i in adtof_F70.testIndexes]
        # trainB = [adtof_yt.audioPaths[i] for i in adtof_yt.trainIndexes]
        # overlap = config.getIntersectionOfPaths(testA, trainB)
        # print("Overlapping files between adtof and adtof_YT: {} tracks".format(len(overlap[0])))

        # # check if bands are overlapping
        # A = collections.Counter([config.getBand(path) for path in testA])
        # B = collections.Counter([config.getBand(path) for path in trainB])
        # overlap = {k: (A[k], B[k]) for k in A if k in B}
        # print("Overlapping artists between adtof and adtof_YT: {} bands".format(len(overlap)))

        return adtof_F70, adtof_yt

    @classmethod
    def getTMIDT(cls, folderPath, n_splits, testFold, ratioKits=1, ratioSequences=1, scenario="all", **kwargs):
        # Get all tracks
        tracks = config.getFilesInFolder(os.path.join(folderPath, "TMIDT/mp3/*.mp3"))
        tmidt_splits = None
        validationFold = testFold + 1

        def getGroup(p):
            return "".join(p.split("_")[1:]).split("-")[0]

        groups = [getGroup(p) for p in tracks]
        groupKFold = sklearn.model_selection.GroupKFold(n_splits=n_splits)
        tmidt_splits = [[config.getFileBasename(tracks[i]) for i in fold] for _, fold in groupKFold.split(tracks, groups=groups)]
        # splits = [
        #     list(pd.read_csv(os.path.join(folderPath, path), sep="no separator hack", header=None)[0])
        #     for path in ["TMIDT/splits/3-fold_cv_acc_0.txt", "TMIDT/splits/3-fold_cv_acc_1.txt", "TMIDT/splits/3-fold_cv_acc_2.txt"]
        # ]

        # testFold = -1  # No test set
        # validationFold = 0.0  # No validation set

        solo = cls(
            [t for t in tracks if t[:-11] != "_accomp.mp3"],
            os.path.join(folderPath, "TMIDT/annotations/drums_f/*"),
            os.path.join("./adtof/ressources/TMIDT_ignore"),
            cachePreprocessFolders=os.path.join(folderPath, "TMIDT/preprocess"),
            mappingDictionaries=[instrumentsMapping.RBMA_FULL_MIDI, instrumentsMapping.MIDI_REDUCED_5],
            sep="\t",
            splits=tmidt_splits,
            getGroup=getGroup,
            testFold=testFold,
            validationFold=validationFold,
            **kwargs,
        )

        acc = cls(
            [t for t in tracks if t[-11:] == "_accomp.mp3"],
            os.path.join(folderPath, "TMIDT/annotations/drums_f/*"),
            os.path.join("./adtof/ressources/TMIDT_ignore"),  # Filter out the files with bad encoding
            cachePreprocessFolders=os.path.join(folderPath, "TMIDT/preprocess"),
            mappingDictionaries=[instrumentsMapping.RBMA_FULL_MIDI, instrumentsMapping.MIDI_REDUCED_5],
            sep="\t",
            splits=tmidt_splits,
            getGroup=getGroup,
            testFold=testFold,
            validationFold=validationFold,
            **kwargs,
        )

        return solo, acc

    @classmethod
    def factoryMixedDatasets(cls, folderPath: str, testFold=0, scenario="all", regulateSamplingProbability=1, dataAugmentation=False, **kwargs):
        """returns all the access function required to train/validate/test on mixed datasets

        Parameters
        ----------
        folderPath : path to the root folder of the ADTOF dataset
        """
        # Load all the different datasets
        datasets = cls.getAllDatasets(folderPath, testFold, scenario=scenario, **kwargs)

        # Specify datasets used for training, validation and testing
        if scenario == "public":
            trainingSets = ["rbma", "enst_wet", "enst_sum", "mdb_full_mix", "mdb_drum_solo"]  # The public training data is generated by mixing the corresponding folds together
            # The pick probability is equal to the length of the dataset to reproduce merging them equally (ENST and MDB appears twice, so we reduce the probability by half)
            trainingSetsCardinality = [1.72, 0.175, 0.175, 0.51, 0.51]
            validationSets = ["rbma", "enst_sum", "mdb_full_mix"]
            validationSetsPickCardinality = [1.72, 0.35, 1.02]
        elif scenario == "tmidt":
            trainingSets = "tmidt_acc"
            validationSets = "tmidt_acc"
        elif scenario == "egmd":
            trainingSets = "egmd"
            validationSets = ["tmidt_solo", "slakh2100_solo", "enst_wet", "mdb_drum_solo"]
            validationSetsPickCardinality = [1, 1, 1, 1]
        elif scenario == "tmidt-acc":
            trainingSets = "tmidt_acc"
            validationSets = ["slakh2100_acc", "enst_sum", "mdb_full_mix", "adtof_F70"]
            validationSetsPickCardinality = [145, 0.35, 1.02, 94]  # TODO does it make sense to use these values for validation? Are the splits taken into account?
        elif scenario == "slakh2100-acc":
            trainingSets = "slakh2100_acc"
            validationSets = ["rbma", "enst_sum", "mdb_full_mix", "adtof_F70"]
            validationSetsPickCardinality = [1.72, 0.35, 1.02, 94]
        elif scenario == "tmidt-valAdtofAll":
            trainingSets = "tmidt_acc"
            validationSets = ["adtof_yt", "adtof_F70"]
            validationSetsPickCardinality = [208.40, 94.09]
        elif scenario == "adtof":
            trainingSets = "adtof_F70"
            validationSets = "adtof_F70"
        elif scenario == "adtof-all":
            trainingSets = ["adtof_yt", "adtof_F70"]
            trainingSetsCardinality = [208.40, 94.09]
            validationSets = ["adtof_yt", "adtof_F70"]
            validationSetsPickCardinality = [208.40, 94.09]
        elif scenario == "adtof-rgw-yt-rb":
            trainingSets = ["adtof_yt", "adtof_F70", "adtof_rb"]
            trainingSetsCardinality = [208.40, 94.09, 125]
            validationSets = ["adtof_yt", "adtof_F70", "adtof_rb"]
            validationSetsPickCardinality = [208.40, 94.09, 125]
        elif scenario == "adtof-tmidt":
            trainingSets = ["adtof_yt", "adtof_F70", "tmidt_acc"]
            trainingSetsCardinality = [1, 1, 1]  # Same probability for all datasets to reproduce Cartwright's results
            validationSets = ["adtof_yt", "adtof_F70", "tmidt_acc"]
            validationSetsPickCardinality = [1, 1, 1]
        elif scenario == "all":
            trainingSets = ["adtof_yt", "adtof_F70", "rbma", "enst_wet", "enst_sum", "mdb_full_mix", "mdb_drum_solo"]
            trainingSetsCardinality = [208.40, 94.09, 1.72, 0.35, 0.35, 1.02, 1.02]
            validationSets = ["adtof_yt", "adtof_F70", "rbma", "enst_sum", "mdb_full_mix"]
            validationSetsPickCardinality = [208.40, 94.09, 1.72, 0.35, 1.02]
        elif scenario in datasets:
            trainingSets = scenario
            validationSets = scenario
        else:
            raise ValueError("training scenario not implemented")

        def getPickProbability(samples, regulateSamplingProbability):
            """
            boost lower-resource languages by sampling examples according to the probability p(L) ∝ |L|^α,
            where p(L) is the probability of sampling text from
            a given language during pre-training and |L| is the
            number of examples in the language. The hyperparameter α (typically with α < 1)
            """
            return [(p / sum(samples)) ** regulateSamplingProbability for p in samples]

        # Prune the training sets
        trainDataLoaders = [datasets[name] for name in trainingSets] if isinstance(trainingSets, list) else [datasets[trainingSets]]
        for trainDL in trainDataLoaders:
            trainDL.pruneTrainSet(**kwargs)
        # Split the data in train, test and val sets with generators
        datasetsGenerators = {setName: set.getTrainValTestGens(**kwargs) for setName, set in datasets.items()}

        # Load training data
        # Also create a train generator giving full tracks to compute the results on training data and judge over or under fitting
        # Count the number of tracks to set the epoch size
        if isinstance(trainingSets, list):
            trainGen = cls._mixingGen(
                [datasetsGenerators[name].trainGen for name in trainingSets], pickProbability=getPickProbability(trainingSetsCardinality, regulateSamplingProbability)
            )
            trainFullGen = cls._roundRobinGen([datasetsGenerators[name].trainFullGen for name in trainingSets])
            trainTracksCount = np.sum([len(datasets[name].trainIndexes) for name in trainingSets])
        else:
            trainGen = datasetsGenerators[trainingSets].trainGen
            trainFullGen = datasetsGenerators[trainingSets].trainFullGen
            trainTracksCount = len(datasets[trainingSets].trainIndexes)
        if dataAugmentation:
            trainGen = da.dataAugmentationGen(trainGen, **kwargs)
            # [e for e in trainGen()]
        train_dataset = cls._getDataset(trainGen, **kwargs)  # Create a dataset from the generators for compatibility with tf.model.fit

        # Load validation data
        # Also create a validation generator giving full tracks to compute the peak picking parameters
        # Count the number of tracks to set the validation epoch size
        if isinstance(validationSets, list):
            valGen = cls._mixingGen(
                [datasetsGenerators[name].valGen for name in validationSets], pickProbability=getPickProbability(validationSetsPickCardinality, regulateSamplingProbability)
            )
            valFullGen = cls._roundRobinGen([datasetsGenerators[name].valFullGen for name in validationSets])
            valTracksCount = np.sum([len(datasets[name].valIndexes) for name in validationSets])
        else:
            valGen = datasetsGenerators[validationSets].valGen
            valFullGen = datasetsGenerators[validationSets].valFullGen
            valTracksCount = len(datasets[validationSets].valIndexes)
        val_dataset = cls._getDataset(valGen, **kwargs)  # Create a dataset from the generators for compatibility with tf.model.fit

        # build a dict of test data for evaluation
        testFullNamedGen = {name: datasetsGenerators[name].testFullGen for name in datasets.keys() if name != "tmidt"}

        # Return all the access functions required
        dataAccess = namedtuple(
            "dataAccess", ["train_dataset", "val_dataset", "trainFullGen", "valFullGen", "trainTracksCount", "valTracksCount", "testFullNamedGen", "trainDataLoaders"]
        )
        return dataAccess(train_dataset, val_dataset, trainFullGen, valFullGen, trainTracksCount, valTracksCount, testFullNamedGen, trainDataLoaders)

    @classmethod
    def _roundRobinGen(cls, generators):
        """
        return an element of each generator until they are all exhausted
        """

        def gen():
            instantiatedGen = [gen() for gen in generators]  # invoke the generators when invoked to reset the iteration at the beginning
            while True:  # loop until all generators are exhausted
                allExhausted = True
                for pickedGen in instantiatedGen:
                    e = next(pickedGen, -1)
                    if e != -1:
                        yield e
                        allExhausted = False

                if allExhausted:
                    break

        return gen

    @classmethod
    def _mixingGen(cls, generators, pickProbability=None):
        """
        return elements from generator with a weighted probability
        TODO: When mixing gen, the order of the draw is not preserved
        """

        def gen():
            # invoke the generators to set the iteration at the beginning
            instantiatedGen = [gen() for gen in generators]
            indexes = list(range(len(generators)))
            while True:
                pickedGen = random.choices(indexes, weights=pickProbability, k=1)[0]
                try:
                    yield next(instantiatedGen[pickedGen])
                except StopIteration:
                    instantiatedGen[pickedGen] = generators[pickedGen]()

        return gen

    @classmethod
    def _getDataset(cls, gen, trainingSequence=1, labels=[1, 1, 1, 1, 1], batchSize=8, prefetch=tf.data.AUTOTUNE, tatumSubdivision=None, sampleWeight=None, n_channels=1, **kwargs):
        """
        Get a tf.dataset from the generator.
        Fill the right size for each dim
        """
        if trainingSequence == 1:
            raise DeprecationWarning("CNN architecture are not supported anymore")

        # Create dataset input, target and weight signature
        if tatumSubdivision is None:  # for a frame-level network
            inputs_signature = {"x": tf.TensorSpec(shape=(None, mir.getDim(**kwargs), n_channels), dtype=tf.float32)}  # shape: (time, feature, channel dimension)
            targets_signature = tf.TensorSpec((trainingSequence, len(labels)), dtype=tf.float32)  # shape: (time - valid padding, classes)
            sample_weights_signature = tf.TensorSpec((trainingSequence,), dtype=tf.float32)  # shape: (time - valid padding,)
        else:  # For a tatum-level network
            inputs_signature = {
                "x": tf.TensorSpec(shape=(None, mir.getDim(**kwargs), n_channels), dtype=tf.float32),
                "tatumsBoundariesFrame": tf.TensorSpec(shape=(trainingSequence, 2), dtype=tf.int32),
            }  # shape:(number of tatum in the sequence,)
            targets_signature = tf.TensorSpec((trainingSequence, len(labels)), dtype=tf.float32)  # shape: (tatum, classes)
            sample_weights_signature = tf.TensorSpec((trainingSequence,), dtype=tf.float32)  # shape: (tatum,)
        signature = (inputs_signature, targets_signature, sample_weights_signature) if sampleWeight is not None else (inputs_signature, targets_signature)
        dataset = tf.data.Dataset.from_generator(gen, output_signature=signature)

        # Batch
        if batchSize != None:
            if tatumSubdivision is None:
                dataset = dataset.batch(batchSize).repeat()
            else:
                dataset = dataset.padded_batch(
                    batchSize, padded_shapes=tuple([{k: list(v.shape) for k, v in s.items()} if isinstance(s, dict) else list(s.shape) for s in signature])
                ).repeat()
        if prefetch != None:
            dataset = dataset.prefetch(buffer_size=prefetch)
        return dataset

    def __init__(
        self,
        audioPaths: str,
        annotationPaths: str = None,
        blockListPath: str = None,
        cachePreprocessFolders: str = None,
        checkFilesNameMatch: bool = True,
        crossValidation: bool = True,
        beatPaths: str = None,
        fmin=20,
        fmax=20000,
        bandsPerOctave=12,
        n_channels=1,
        trackMapping=None,
        mappingDictionaries=None,
        **kwargs,
    ):
        """
        Class for the handling of building a dataset for training and infering.
        Loads a folder of tracks, split the data, and provides generator to iterate over everything.

        cachePreprocessPath if not None, save the processed data in a cache folder to speed up the loading of the data
        store if True, lazy load the data and store it in memory
        """
        # Fetch audio paths
        self.audioPaths = config.getFilesInFolder(audioPaths) if isinstance(audioPaths, str) else audioPaths
        self.annotationPaths = None
        self.beatPaths = None
        self.preprocessPaths = None

        # Fetch annotation paths
        if annotationPaths is not None:
            self.annotationPaths = config.getFilesInFolder(annotationPaths) if isinstance(annotationPaths, str) else annotationPaths
            if checkFilesNameMatch:  # Getting the intersection of audio and annotations files
                self.audioPaths, self.annotationPaths = config.getIntersectionOfPaths(self.audioPaths, self.annotationPaths)

        # Fetch beat paths
        if beatPaths is not None:
            # Computing the beats if needed
            if isinstance(beatPaths, str):
                for audioPath in self.audioPaths:
                    beatsEstimationsPath = os.path.join(os.path.dirname(beatPaths), config.getFileBasename(audioPath) + ".txt")
                    if not config.checkAllPathsExist(beatsEstimationsPath):
                        mbc = MadmomBeatConverter()
                        logging.debug("MadmomBeatConverter on %s", audioPath)
                        mbc.convert(audioPath, None, beatsEstimationsPath, None)
            self.beatPaths = config.getFilesInFolder(beatPaths) if isinstance(beatPaths, str) else beatPaths
            if checkFilesNameMatch:  # Getting the intersection of audio, annotations, and beat files
                self.audioPaths, self.annotationPaths, self.beatPaths = config.getIntersectionOfPaths(self.audioPaths, self.annotationPaths, self.beatPaths)

        # Remove track from the blocklist manually checked
        if blockListPath is not None and os.path.exists(blockListPath):
            toRemove = set(pd.read_csv(blockListPath, sep="\t", header=None, on_bad_lines="skip")[0])
            self.audioPaths = [f for f in self.audioPaths if config.getFileBasename(f) not in toRemove]
            if self.annotationPaths is not None:
                self.annotationPaths = [f for f in self.annotationPaths if config.getFileBasename(f) not in toRemove]
            if self.beatPaths is not None:
                self.beatPaths = [f for f in self.beatPaths if config.getFileBasename(f) not in toRemove]

        # Set the cache paths to store pre processed audio
        if cachePreprocessFolders is not None:
            self.preprocessPaths = np.array(
                [
                    os.path.join(
                        (cachePreprocessFolders if isinstance(cachePreprocessFolders, str) else cachePreprocessFolders[i])
                        + "-".join([str(fmin), str(fmax), str(bandsPerOctave), str(n_channels)]),
                        config.getFileBasename(track) + ".npy",
                    )
                    for i, track in enumerate(self.audioPaths)
                ]
            )

        # Split the data according to the specified folds or generate then
        if crossValidation:
            self.folds = self._getFolds(**kwargs)
            self.trainIndexes, self.valIndexes, self.testIndexes = self._getSubsetsFromFolds(self.folds, **kwargs)
        else:  # If there is no cross validation, put every tracks in the test split
            self.trainIndexes, self.valIndexes = [], []
            self.testIndexes = list(range(len(self.audioPaths)))

        # define the lazy loader
        self.data = LazyDict(
            lambda i: Track(
                self.audioPaths[i],
                self.preprocessPaths[i] if self.preprocessPaths is not None else None,
                self.beatPaths[i] if self.beatPaths is not None else None,
                self.annotationPaths[i] if self.annotationPaths is not None else None,
                bandsPerOctave=bandsPerOctave,
                n_channels=n_channels,
                fmin=fmin,
                fmax=fmax,
                mappingDictionaries=trackMapping[i] if trackMapping is not None else mappingDictionaries,
                **kwargs,
            )
        )

    def getTotalDuration(self, tracks=None):
        """
        Compute the total duration of the dataset in seconds
        """
        duration = 0
        tracks = tracks if tracks is not None else range(len(self.audioPaths))
        for i in tracks:
            track = self.data.getWithoutSaving(i)
            duration += track.samplesCardinality / 100
        return duration

    def _getFolds(self, nFolds=10, isGroupKFold=True, splits=None, getGroup=config.getBand, **kwargs):
        """
         Split the data in groups without band overlap between split

        Parameters
        ----------
        nFolds : int, optional
            number of folds to create (how many splits are made)
        isGroupKFold : bool, optional
            if True, split the data in groups without band overlap between split
        splits : list of list of str (track name), optional
            if not None, use the given splits instead of generating them
        Returns
        -------
        nFolds list of int, representing the index of the track in each fold
        """
        if isGroupKFold:
            self.groups = getGroup if isinstance(getGroup, list) else [getGroup(path) for path in self.audioPaths]
        else:
            self.groups = [i for i in range(len(self.audioPaths))]

        self.keptGroups = self.groups

        if splits is not None:
            fileLookup = {config.getFileBasename(file): i for i, file in enumerate(self.audioPaths)}
            return [[fileLookup[name] for name in split if name in fileLookup] for split in splits]

        groupKFold = sklearn.model_selection.GroupKFold(n_splits=nFolds)
        return [fold for _, fold in groupKFold.split(self.audioPaths, self.annotationPaths, self.groups)]

    def _getSubsetsFromFolds(self, folds, testFold=-1, validationFold=0, randomState=0, **kwargs):
        """
        Return indexes of tracks for the train, validation and test splits from a k-fold scheme.

        Parameters
        ----------
        folds :
            list of indexes returned by _getFolds(), or config.DB_SPLITS given at dataLoader.__init__
        testFold : int, optional
            index of the split saved for test
            By default 0
        validationFold : int/float, optional
            if int, index of the remaining split saved for validation.
            if float, ratio of training data saved for validation.
            By default 0

        Returns
        -------
        (trainIndexes, validationIndexes, testIndexes)
        """
        if testFold != -1:
            testIndexes = folds.pop(testFold)
        else:
            testIndexes = []

        if isinstance(validationFold, int):  # save a split for validation
            valIndexes = folds.pop(validationFold)
            trainIndexes = np.concatenate(folds)
        elif isinstance(validationFold, float):  # save some % of the training data for validation
            trainData = np.concatenate(folds)
            trainData = sklearn.utils.shuffle(trainData, random_state=randomState)
            splitIndex = int(len(trainData) * validationFold)
            valIndexes = trainData[:splitIndex]
            trainIndexes = trainData[splitIndex:]

        # Shuffle indexes such as the tracks from the same band will not occur in sequence
        trainIndexes = sklearn.utils.shuffle(trainIndexes, random_state=randomState)
        valIndexes = sklearn.utils.shuffle(valIndexes, random_state=randomState)
        testIndexes = sklearn.utils.shuffle(testIndexes, random_state=randomState)

        return (trainIndexes, valIndexes, testIndexes)

    def pruneTrainSet(self, nSequences=None, nGroups=None, pruningData=None, **kwargs):
        """
        prune tracks of the training set to perform an ablation study
        Specify either the number of sequences and groups, or the pruningData which creates a search spaces: [(1, 1), (1, 2), ... (1, 512), (2, 1), (2, 2), ...]
        """
        if (nSequences is None or nGroups is None) and pruningData is None:
            return

        # Identify the group of each track
        groupToTrack = defaultdict(list)
        self.trainIndexes = list(set(self.trainIndexes))  # TODO trainIndexes is not unique before this step
        for i in self.trainIndexes:
            groupToTrack[self.groups[i]].append(i)
        availableGroups = sorted(list(groupToTrack.keys()), key=lambda x: len(groupToTrack[x]), reverse=True)  # prioritise the groups with the highest number of sequences

        # Select the number of groups and sequences to keep from pruningData
        if pruningData is not None:
            samplingTracks = [2**i for i in range(20) if 2**i < len(self.trainIndexes)]
            samplingGroups = [2**i for i in range(20) if 2**i < len(availableGroups)]
            searchSpace = [(i, j) for i in samplingGroups for j in samplingTracks]
            nGroups, nSequences = searchSpace[pruningData]
            print(f"Prunning searchSpace {pruningData}/{len(searchSpace)-1}: {nGroups} groups, {nSequences} sequences")

        # Limit number of groups
        keptGroups = availableGroups[:nGroups]

        # Limit number of sequences
        maxSequences = len(self.trainIndexes[:nSequences])

        if len(keptGroups) < nGroups:
            raise ValueError("Not enough groups to keep the desired number of groups")
        if maxSequences < nSequences:
            raise ValueError("Not enough tracks to keep the desired number of tracks")
        if np.sum([len(groupToTrack[group]) for group in keptGroups]) < maxSequences:
            raise ValueError("Not enough groups to keep the desired number of tracks")

        # Pop tracks in the most common group until the number of sequences is below the limit
        while np.sum([len(groupToTrack[group]) for group in keptGroups]) > maxSequences:
            mostCommonGroup = keptGroups[0]
            for group in keptGroups[1:]:
                if len(groupToTrack[group]) >= len(groupToTrack[mostCommonGroup]):
                    mostCommonGroup = group
                else:
                    break
            groupToTrack[mostCommonGroup].pop()
            if len(groupToTrack[mostCommonGroup]) == 0:
                raise ValueError("Not enough tracks to keep the desired number of groups")

        self.trainIndexes = [i for group in keptGroups for i in groupToTrack[group]]
        self.keptGroups = keptGroups

    @staticmethod
    def mergeDiversityStatistics(dataloaders):
        """
        Get the diversity statistics of multiple dataloaders (See _getDiversityStatistics()).
        And merge them into a single dict
        """
        # Merge stats
        results = {}
        for dataloader in dataloaders:
            for key, value in dataloader._getDiversityStatistics().items():
                results[key] = results[key] + value if key in results else value

        # Return length for counter type
        results = {key: len(value) if isinstance(value, Counter) else value for key, value in results.items()}
        return results

    def _getDiversityStatistics(self):
        """
        Estimates the diversity of this dataset to identify the scaling of the model wrt the number of:
        tracks, groups (Kits or bands), bars, beats, unique bars, unique beats, duration
        """
        results = {}
        results["trainTracksCount"] = len(self.trainIndexes)
        results["trainGroupsCount"] = len(self.keptGroups)

        results["trainBeats"] = sum([len(self.data[idx].beats) for idx in self.trainIndexes])
        results["trainBars"] = sum([len([1 for b in self.data[idx].beats if b["pitch"] == 1]) for idx in self.trainIndexes])

        sequences = Counter()
        for idx in self.trainIndexes:
            sequences += Counter(self.data[idx].getUniqueSequences())
        results["trainUniqueBeats"] = sequences

        sequences = Counter()
        for idx in self.trainIndexes:
            beatsPerBars = max([b["pitch"] for b in self.data[idx].beats])
            sequences += Counter(self.data[idx].getUniqueSequences(beatNumberAsBoundary=1, subdivision=12 * beatsPerBars))
        results["trainUniqueBars"] = sequences

        results["trainingDuration"] = self.getTotalDuration(self.trainIndexes)
        return results

    def getTrainValTestGens(self, batchSize=None, **kwargs):
        """
        Return 4 generators to perform training and validation:
        trainGen : Finite generator for training
        valGen : Finite generator for validation
        trainFullGen: Finite generator giving full tracks for computing precision and recall on train data
        valFullGen : Finite generator giving full tracks for fitting peak picking
        testFullGen : Finite generator giving full tracks for computing final result
        """

        trainGen, valGen, trainFullGen, valFullGen, testFullGen = (
            self.getGen(self.trainIndexes, **kwargs),
            self.getGen(self.valIndexes, **kwargs),
            self.getGen(self.trainIndexes, training=False, **kwargs),
            self.getGen(self.valIndexes, training=False, **kwargs),
            self.getGen(self.testIndexes, training=False, **kwargs),
        )

        datasetGenerators = namedtuple("datasetGenerators", ["trainGen", "valGen", "trainFullGen", "valFullGen", "testFullGen"])
        return datasetGenerators(trainGen, valGen, trainFullGen, valFullGen, testFullGen)

    def getGen(
        self,
        trackIndexes=None,
        training=True,
        trainingSequence=400,
        **kwargs,
    ):
        """
        Return an infinite generator yielding samples

        Parameters
        ----------
        trackIndexes : list(int), optional
            Specifies which track have to be selected based on their index. (see self.getSplit to get indexes), by default None
        training : bool, optional
            Specifies if a generator returning training samples is constructed, output shape compatible with tf.model.fit: (inputs, target, weights)
            If false, yield object of type ADTOF.model.Track
        context : int, optional
            Specifies how large the context is, by default 25
        trainingSequence : int, optional
            Specifies how many samples are in the target

        Returns
        -------
        generator yielding unique samples(x, y, w)
        """
        if trackIndexes is None:
            trackIndexes = list(range(len(self.audioPaths)))

        def genWithoutReplacement():
            """
            Draw randomly from all the tracks without replacement (longer tracks are seen more often)
            """
            logging.debug("reset the generator with %s tracks, %s", len(trackIndexes), self.audioPaths[trackIndexes[0]])
            allTrainingIndexes = shuffle(
                [(trackIdx, sampleIdx) for trackIdx in trackIndexes for sampleIdx in self.data[trackIdx].getAvailableSliceIndexes(trainingSequence, **kwargs)],
                random_state=0,
            )
            for trackIdx, sampleIdx in allTrainingIndexes:
                track: Track = self.data[trackIdx]
                yield track.getSlice(sampleIdx, trainingSequence, **kwargs)

        def genWholeTracks():
            """
            Return whole tracks one by one
            """
            for trackIdx in trackIndexes:
                yield self.data.getWithoutSaving(trackIdx)

        # next(genWithoutReplacement())
        if training:
            return genWithoutReplacement
        else:
            return genWholeTracks

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

        tracksNotes = [tr.getOnsets(drumFile)[0] for drumFile in drums]  # Get the dict of notes' events
        timeSteps = np.sum([librosa.get_duration(filename=track) for track in tracks]) * sampleRate
        result = []
        for label in labels:
            n = np.sum([len(trackNotes[label]) for trackNotes in tracksNotes])
            p = n / timeSteps
            y = 1 / (-p * np.log(p) - (1 - p) * np.log(1 - p))
            result.append(y)
        print(result)  # [10.780001453213364, 13.531086684241876, 34.13723052423422, 11.44276962353584, 17.6755104053326]
        return result

    def vizDataset(self, samples=100, labels=[36], sampleRate=50, condensed=False, title=""):
        gen = self.getGen(training=False, labels=labels, sampleRate=sampleRate, midiLatency=10)()

        fig = plt.figure(figsize=(8, 8))
        fig.suptitle(title)
        columns = 2
        rows = 5
        for i in range(1, columns * rows + 1):
            fig.add_subplot(rows, columns, i)
            track = next(gen)
            X, Y = track.getSlice(400, 400)

            plt.imshow(np.reshape(X["x"], (X["x"].shape[0], X["x"].shape[1])).T)
            for i in range(len(Y[0])):
                times = [t for t, y in enumerate(Y) if y[i]]
                plt.plot(times, np.ones(len(times)) * i * 10, "or")
            plt.title(track.title)
        plt.savefig(title + ".png")
        # plt.show()
