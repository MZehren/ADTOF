import logging
import os
import sys
from collections import defaultdict

import jellyfish
import numpy as np
import sklearn
import tensorflow as tf

from adtof.io import CQT


class Converter(object):
    """
    Base class to convert file formats
    """

    def convert(self, inputPath, outputName=None):
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

    @staticmethod
    def _getFileCandidates(rootFolder):
        """
        go recursively inside all folders, identify the format available and list all the tracks
        """
        # Decompress all the files
        from adtof.io.converters import ArchiveConverter
        from adtof.io.converters import RockBandConverter
        from adtof.io.converters import PhaseShiftConverter

        ac = ArchiveConverter()
        for root, dirs, files in os.walk(rootFolder):
            for file in files:
                fullPath = os.path.join(root, file)
                ac.convert(fullPath)

        rbc = RockBandConverter()
        psc = PhaseShiftConverter()

        results = defaultdict(list)
        #Check anything convertible
        for root, dirs, files in os.walk(rootFolder):
            if psc.isConvertible(root):
                results[psc.getTrackName(root)].append((root, psc))
            else:
                for file in files:
                    path = os.path.join(root, file)
                    if rbc.isConvertible(path):
                        results[rbc.getTrackName(path)].append((path, psc))

        # Remove duplicate
        return results

    @staticmethod
    def _cleanName(name):
        """
        Look at keywords in the name.
        if it contains ainy, remove them and return a priority score
        """
        keywords = [
            "2xBP_Plus", "2xBP", "2xBPv3", "2xBPv1a", "2xBPv2", "2xBPv1", "(2x Bass Pedal+)", "(2x Bass Pedal)", "(2x Bass Pedals)", "2xbp", "2x",
            "X+", "Expert+", "Expert_Plus", "(Expert+G)", "Expert", "(Expert G)", "(Reduced 2x Bass Pedal+)", "1x", "(B)"
        ]

        contained = [k for k in keywords if k in name]
        if len(contained):
            longest = max(contained, key=lambda k: len(k))
            return name.replace(longest, ''), keywords.index(longest)
        else:
            return name, 10000

    @staticmethod
    def _mergeFileNames(candidates, similitudeThreshold=0.8):
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
        from adtof.io.converters import RockBandConverter

        for candidate in candidates:
            psTrakcs = [convertor for convertor in candidates[candidate] if isinstance(convertor[1], PhaseShiftConverter)]
            if len(psTrakcs) > 0:
                # TODO: select the best one
                candidates[candidate] = psTrakcs[0]
            else:
                # TODO: convert Rockband
                del candidates[candidate]

        return candidates

    @staticmethod
    def generateGenerator(data):
        """
        Create a generator with the tracks in data
        """

        def gen(context=25):
            cqt = CQT()
            for path, converter in data:
                midi, audio, ini = converter.getConvertibleFiles(path)
                try:
                    # Get the midi in dense matrix representation
                    y = converter.convert(path).getDenseEncoding(sampleRate=98.4375, timeShift=0)

                    # Get the CQT with a context
                    x = cqt.open(os.sep.join([path, audio]))
                    x = np.array([x[i:i + context] for i in range(len(x) - context)])

                    # Add the channel dimension
                    x = x.reshape(x.shape + (1, ))

                    for i in range(min(len(y), len(x))):
                        yield x[i], y[i]
                except Exception as e:
                    print(path, e)

        return gen

    @staticmethod
    def convertAll(rootFolder, test_size=0.2):
        """
        convert all tracks in the good format
        and return a dataset.
        """
        logging.basicConfig(filename='log/conversion.log', level=logging.DEBUG)

        candidates = Converter._getFileCandidates(rootFolder)
        candidates = Converter._mergeFileNames(candidates, similitudeThreshold=0.8)
        candidates = Converter._pickVersion(candidates)

        train, test = sklearn.model_selection.train_test_split(list(candidates.values()), test_size=test_size)
        return tf.data.Dataset.from_generator(Converter.generateGenerator(train), (tf.float64, tf.int64)), tf.data.Dataset.from_generator(Converter.generateGenerator(test), (tf.float64, tf.int64))
