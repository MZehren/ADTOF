import logging
import os
import sys
import warnings
from collections import defaultdict

import jellyfish
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import tensorflow as tf

from adtof.io import MIR


class Converter(object):
    """
    Base class to convert file formats
    """

    def convert(self, inputPath, outputFolder):
        """
        Base method to convert a file
        if the outputName is not None, the file is also written to the disk
        """
        raise NotImplementedError()

    def isConvertible(self, path):
        """
        return True or False if it's convertible
        """
        raise NotImplementedError()

    def getTrackName(self, path):
        """
        return the name of the track 
        """
        raise NotImplementedError()

    def checkPathExists(self, path):
        """ 
            Generate the tree of folder if it doesn't exist
        """
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def _getFileCandidates(rootFolder):
        """
        go recursively inside all folders, identify the format available and list all the tracks
        """
        # TODO clean the circle dependency by ,oving the code in a better location
        from adtof.io.converters import ArchiveConverter, RockBandConverter, PhaseShiftConverter, TextConverter

        # Decompress all the files
        ac = ArchiveConverter()
        for root, _, files in os.walk(rootFolder):
            for file in files:
                fullPath = os.path.join(root, file)
                try:
                    ac.convert(fullPath)
                except Exception as e:
                    logging.info(e)
                    logging.info("Archive not working: " + file)

        rbc = RockBandConverter()
        psc = PhaseShiftConverter()
        tc = TextConverter()
        results = defaultdict(list)
        #Check anything convertible
        for root, _, files in os.walk(rootFolder):
            # if os.path.split(root)[1] == "rbma_13" or (root.split("/")[-2] == "rbma_13" and
            #                                            root.split("/")[-1] == ""):  # Add rbma files
            #     midifolder = os.path.join(root, "annotations/drums")
            #     midiFiles = [os.path.join(midifolder, f) for f in os.listdir(midifolder) if f.endswith(".txt")]

            #     audioFolder = os.path.join(root, "audio")
            #     audioFiles = [
            #         os.path.join(audioFolder, f)
            #         for f in os.listdir(audioFolder)
            #         if f.endswith(".mp3") or f.endswith(".wav")
            #     ]

            #     if len(midiFiles) != len(audioFiles):
            #         raise Exception("the rbma audio files without drums are not removed (6-7-26) ")
            #     for i, _ in enumerate(midiFiles):
            #         results[midiFiles[i]].append((midiFiles[i], audioFiles[i], tc))

            if psc.isConvertible(root):
                results[psc.getTrackName(root)].append(
                    (root, psc))  # TODO: change the hardcoded format to something simpler to use
            # else:
            #     for file in files:
            #         path = os.path.join(root, file)
            #         if rbc.isConverinputFolder):
            #             results[rbc.getTrackName(path)].append((path, "", psc))

        # Remove duplicate
        return results

    @staticmethod
    def _cleanName(name):
        """
        Look at keywords in the name.
        if it contains ainy, remove them and return a priority score
        """
        keywords = [
            "2xBP_Plus", "2xBP", "2xBPv3", "2xBPv1a", "2xBPv2", "2xBPv1", "(2x Bass Pedal+)", "(2x Bass Pedal)",
            "(2x Bass Pedals)", "2xbp", "2x", "X+", "Expert+", "Expert_Plus", "(Expert+G)", "Expert", "(Expert G)",
            "(Reduced 2x Bass Pedal+)", "(Reduced 2x Bass Pedal)", "1x", "(B)"
        ]

        contained = [k for k in keywords if k in name]
        if len(contained):
            longest = max(contained, key=lambda k: len(k))
            return name.replace(longest, '').replace(' ', ''), keywords.index(longest)
        else:
            return name.replace(' ', ''), 10000

    @staticmethod
    def _mergeFileNames(candidates, similitudeThreshold=0.9):
        """
        Merge the multiple version of the tracks between "foo_expert" and "foo_expert+"
        1: remove the keywords like "expert" or "(double_bass)"
        2: look at the distance between the names
        3: group the track with similar names and keep the highest priority one (double bass > single bass)

        TODO: make it clear
        """
        names = candidates.keys()
        names = [n for n in names if n is not None]
        cleanedNames = [Converter._cleanName(name) for name in names]
        analysed = set([])
        group = []
        for i, a in enumerate(names):
            if i in analysed:
                continue
            analysed.add(i)
            aClean, priorityA = cleanedNames[i]
            row = [(a, priorityA)]
            for j, b in enumerate(names):
                if j in analysed:
                    continue

                bClean, priorityB = cleanedNames[j]
                similitude = jellyfish.jaro_distance(aClean, bClean)
                if similitude > similitudeThreshold:
                    analysed.add(j)
                    row.append((b, priorityB))
            group.append(row)

        result = {}
        for row in group:
            if len(row) == 1:
                result[row[0][0]] = candidates[row[0][0]]
            else:
                key = min(row, key=lambda k: k[1])[0]
                result[key] = candidates[key]
                logging.debug(("removing doubles: ", key, row))
        return result

    @staticmethod
    def _pickVersion(candidates):
        """
        in case there are multiple version of the same track, pick the best version to use
        PhaseShift > rockBand
        PhaseShift with more notes > Phase shift with less notes 
        """
        from adtof.io.converters import PhaseShiftConverter
        # from adtof.io.converters import RockBandConverter

        for candidate in candidates:
            psTrakcs = [
                convertor for convertor in candidates[candidate] if isinstance(convertor[1], PhaseShiftConverter)
            ]
            if len(psTrakcs) > 0:
                # TODO: select the best one
                candidates[candidate] = psTrakcs[0]
            else:
                # TODO: convert Rockband
                # del candidates[candidate]
                candidates[candidate] = candidates[candidate][0]
        return candidates

    @staticmethod
    def convertAll(inputFolder, outputFolder):
        """
        convert all tracks in the good format
        """

        # Get all possible convertible files and remove duplicated ones
        candidates = Converter._getFileCandidates(inputFolder)
        candidates = Converter._mergeFileNames(candidates)
        candidates = Converter._pickVersion(candidates)
        candidateName = list(candidates.values())
        candidateName.sort(key=lambda x: x[0])
        logging.info("number of tracks in the dataset: " + str(len(candidates)))

        # Do the conversion
        for path, converter in candidates.values():
            converter.convert(path, outputFolder)

    @staticmethod
    def getTFGenerator(candidateName, test_size=0.1):
        """
        WIP: Create a generator dynamically generating converted tracks        
        """

        def generateGenerator(data):
            """
            Create a generator with the tracks in data
            TODO: this is ugly
            """

            def gen(context=25, midiLatency=12, classWeight=[2 / 16, 8 / 16, 16 / 16, 2 / 16, 4 / 16]):
                """
                [36, 40, 41, 46, 49]
                """
                mir = MIR()
                for midiPath, audiPath, converter in data:
                    try:
                        # TODO: update: _, audio, _ = converter.getConvertibleFiles(path)
                        # Get the y: midi in dense matrix representation
                        y = converter.convert(midiPath).getDenseEncoding(sampleRate=100, timeShift=0, radiation=0)
                        y = y[midiLatency:]
                        if np.sum(y) == 0:
                            warnings.warn("Midi doesn't have notes " + midiPath)
                            continue

                        # Get the x: audio with stft or cqt or whatever + overlqp windows to get some context
                        x = mir.open(audiPath)
                        x = np.array([x[i:i + context] for i in range(len(x) - context)])
                        x = x.reshape(x.shape + (1,))  # Add the channel dimension

                        for i in range(min(len(y) - 1, len(x) - 1)):
                            # sampleWeight = 1  #max(1/16, np.sum(classWeight * y[i])) #TODO: compute the ideal weight based on the distribution of the samples
                            yield x[i], y[i]
                    except Exception as e:
                        print(midiPath, e)
                print("DEBUG: real new epoch")

            return gen

        train, test = sklearn.model_selection.train_test_split(candidateName, test_size=test_size, random_state=1)

        # next(Converter.generateGenerator(train)())
        trainDS = tf.data.Dataset.from_generator(generateGenerator(train), (tf.float64, tf.int64))
        testDS = tf.data.Dataset.from_generator(generateGenerator(test), (tf.float64, tf.int64))
        return trainDS, testDS

    @staticmethod
    def vizDataset(iterator):
        X, Y = iterator.get_next()
        plt.matshow(np.array([np.reshape(x[0], 84) for x in X]).T)
        print(np.sum(Y))
        for i in range(len(Y[0])):
            times = [t for t, y in enumerate(Y) if y[i]]
            plt.plot(times, np.ones(len(times)) * i * 10, "or")
        plt.show()

    @staticmethod
    def vizNumpy(X, Y, title="bla"):
        plt.title(title)
        plt.matshow(X.T)
        for i in range(len(Y[0])):
            times = [t for t, y in enumerate(Y) if y[i]]
            plt.plot(times, np.ones(len(times)) * i * 10, "or")
        plt.show()
