import concurrent.futures
import logging
import os
import sys
import warnings
from collections import defaultdict

import jellyfish
import matplotlib.pyplot as plt
import numpy as np
import sklearn


class Converter(object):
    """
    Base class to convert file formats
    """

    def convert(self, inputs, outputs):
        """
        Base method to convert a file
        """
        raise NotImplementedError()

    def isConvertible(self, path):
        """
        return wether the path is a suitable input
        """
        raise NotImplementedError()

    @staticmethod
    def checkPathExists(path):
        """ 
        return if the path exists and generate the tree of folder if it doesn't
        """
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return os.path.exists(path)

    @staticmethod
    def checkAllPathsExist(*outputs):
        """
        Check if all the path exist
        """
        allPathsExist = True
        for output in outputs:
            if not Converter.checkPathExists(output):
                allPathsExist = False
        return allPathsExist

    @staticmethod
    def _getFileCandidates(rootFolder):
        """
        go recursively inside all folders, identify the format available and list all the tracks
        """
        # TODO clean the circle dependency by ,oving the code in a better location
        from adtof.converters.archiveConverter import ArchiveConverter
        from adtof.converters.rockBandConverter import RockBandConverter
        from adtof.converters.phaseShiftConverter import PhaseShiftConverter

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
        results = defaultdict(list)
        genres = defaultdict(list)
        # Check anything convertible
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
            # else:
            #     for file in files:
            #         path = os.path.join(root, file)
            #         if rbc.isConverinputFolder):
            #             results[rbc.getTrackName(path)].append((path, "", psc))

            if psc.isConvertible(root):
                meta = psc.getMetaInfo(root)
                if not meta["pro_drums"]:
                    # genres["pro_drums:False"].append(root)
                    logging.info("track not contianing pro_drums tag " + meta["name"])
                    continue
                # else:
                #     genres["pro_drums:True"].append(root)
                genres[meta["genre"]].append(root)
                # if meta["genre"] == "Pop/Dance/Electronic":
                results[meta["name"]].append({"path": root, "convertor": psc})

        # # Plot
        # values = [[k, len(v)] for k, v in genres.items()]
        # values.sort(key=lambda e: -e[1])
        # plt.figure(figsize=(10, 4))
        # plt.bar(range(len(values)), [v[1] for v in values], edgecolor="black")
        # plt.grid(axis="y", linestyle="--")
        # plt.xticks(range(len(values)), [v[0] for v in values], rotation=90)
        # plt.ylabel("Count")
        # plt.gcf().subplots_adjust(bottom=0.4)
        # plt.savefig("Genre distribution.png", dpi=600)
        return results

    @staticmethod
    def _cleanName(name):
        """
        Look at keywords in the name.
        if it contains any, remove them and return a priority score
        """
        keywords = [
            "2xBP_Plus",
            "2xBP",
            "2xBPv3",
            "2xBPv1a",
            "2xBPv2",
            "2xBPv1",
            "(2x Bass Pedal+)",
            "(2x Bass Pedal)",
            "(2x Bass Pedals)",
            "2xbp",
            "2x",
            "X+",
            "Expert+",
            "Expert_Plus",
            "(Expert+G)",
            "Expert",
            "(Expert G)",
            "(Reduced 2x Bass Pedal+)",
            "(Reduced 2x Bass Pedal)",
            "1x",
            "(B)",
        ]

        contained = [k for k in keywords if k in name]
        if len(contained):
            longest = max(contained, key=lambda k: len(k))
            return name.replace(longest, "").replace(" ", ""), keywords.index(longest)
        else:
            return name.replace(" ", ""), 10000

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
                # similitude = jellyfish.jaro_distance(aClean, bClean)
                if aClean == bClean:  # similitude > similitudeThreshold:
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
        from adtof.converters.phaseShiftConverter import PhaseShiftConverter

        # from adtof.io.converters import RockBandConverter

        for candidate in candidates:
            psTrakcs = [convertor for convertor in candidates[candidate] if isinstance(convertor["convertor"], PhaseShiftConverter)]
            if len(psTrakcs) > 0:
                # TODO: select the best one
                candidates[candidate] = psTrakcs[0]
            else:
                # TODO: convert Rockband
                # del candidates[candidate]
                candidates[candidate] = candidates[candidate][0]
        return candidates

    @staticmethod
    def convertAll(inputFolder, outputFolder, parallelProcess=False):
        """
        convert all tracks in the good format
        """
        from adtof.converters.RVCRNNConverter import RVCRNNConverter
        from adtof.converters.madmomBeatConverter import MadmomBeatConverter
        from adtof.converters.correctAlignmentConverter import CorrectAlignmentConverter
        from adtof.converters.featuresExctractorConverter import FeaturesExtractorConverter
        from adtof import config

        # Get all possible convertible files
        candidates = Converter._getFileCandidates(inputFolder)
        # remove duplicated ones
        candidates = Converter._mergeFileNames(candidates)
        candidates = Converter._pickVersion(candidates)
        candidateName = list(candidates.values())
        candidateName.sort(key=lambda x: x["path"])
        logging.info("number of tracks in the dataset after selection: " + str(len(candidates)))

        # Do the conversion
        if parallelProcess:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(Converter.runConvertors, candidate, outputFolder, trackName)
                    for trackName, candidate in list(candidates.items())
                ]
                concurrent.futures.wait(futures)
        else:
            for i, (trackName, candidate) in enumerate(list(candidates.items())):
                Converter.runConvertors(candidate, outputFolder, trackName)

    @staticmethod
    def runConvertors(candidate, outputFolder, trackName):
        try:
            from adtof.converters.madmomBeatConverter import MadmomBeatConverter
            from adtof.converters.correctAlignmentConverter import CorrectAlignmentConverter
            from adtof import config

            mbc = MadmomBeatConverter()
            ca = CorrectAlignmentConverter()
            # # Convert the chart into standard midi
            inputChartPath = candidate["path"]
            convertedMidiPath = os.path.join(outputFolder, config.CONVERTED_MIDI, trackName + ".midi")
            rawMidiPath = os.path.join(outputFolder, config.RAW_MIDI, trackName + ".midi")
            audioPath = os.path.join(outputFolder, config.AUDIO, trackName + ".ogg")
            if not Converter.checkAllPathsExist(convertedMidiPath, rawMidiPath, audioPath):
                # print(trackName + "TODO ")
                candidate["convertor"].convert(inputChartPath, convertedMidiPath, rawMidiPath, audioPath)

            # Align the annotations by looking at the average beat estimation difference
            # RVDrumsEstimationPath = os.path.join(outputFolder, config.RV_ESTIMATIONS, trackName + ".drums.txt")
            beatsEstimationsPath = os.path.join(outputFolder, config.BEATS_ESTIMATIONS, trackName + ".txt")
            beatsActivationPath = os.path.join(outputFolder, config.BEATS_ACTIVATION, trackName + ".npy")
            alignedBeatsAnnotationsPath = os.path.join(outputFolder, config.ALIGNED_BEATS, trackName + ".txt")
            alignedDrumAnotationsPath = os.path.join(outputFolder, config.ALIGNED_DRUM, trackName + ".txt")
            alignedMidiAnotationsPath = os.path.join(outputFolder, config.ALIGNED_MIDI, trackName + ".midi")
            if not Converter.checkAllPathsExist(beatsEstimationsPath, beatsActivationPath):
                mbc.convert(audioPath, beatsEstimationsPath, beatsActivationPath)
            if True:  # not Converter.checkAllPathsExist(alignedDrumAnotationsPath, alignedBeatsAnnotationsPath, alignedMidiAnotationsPath):
                ca.convert(
                    beatsEstimationsPath,
                    beatsActivationPath,
                    convertedMidiPath,
                    alignedDrumAnotationsPath,
                    alignedBeatsAnnotationsPath,
                    alignedMidiAnotationsPath,
                    audioPath=audioPath,
                )

            # # Extract Features
            # featuresExtractedPath = os.path.join(outputFolder, config.FEATURES, trackName + ".npy")
            # if not Converter.checkPathExists(featuresExtractedPath):
            #     fe.convert(audioPath, featuresExtractedPath)
            logging.info(trackName + " Done.")
        except Exception as e:
            logging.warning(trackName + " not converted: " + str(e))

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
