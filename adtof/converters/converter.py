import concurrent.futures
import logging
import os
from collections import defaultdict, Counter
import re
import jellyfish
import matplotlib.pyplot as plt
from adtof.config import plot
import pandas as pd


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
    def _getFileCandidates(rootFolder):
        """
        go recursively inside all folders, identify the format available and list all the tracks
        """
        # TODO clean the circle dependency by moving the code in a better location
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
                    logging.warning(e)
                    logging.warning("Archive not working: " + file)

        rbc = RockBandConverter()
        psc = PhaseShiftConverter()
        results = defaultdict(list)
        genres = defaultdict(list)
        # Check anything convertible
        for root, _, files in os.walk(rootFolder):
            if psc.isConvertible(root):
                meta = psc.getMetaInfo(root)
                if not meta["pro_drums"]:
                    # genres["pro_drums:False"].append(root)
                    logging.warning(meta["name"] + "track not containing pro_drums tag ")
                    continue
                # else:
                #     genres["pro_drums:True"].append(root)
                genres[meta["genre"]].append(root)
                results[meta["name"]].append({"path": root, "convertor": psc})
        # # Plot
        values = [[k, len(v)] for k, v in genres.items()]
        values.sort(key=lambda e: -e[1])
        return results, values

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
    def _mergeFileNames(candidates, similitudeThreshold=1):
        """
        Merge the multiple version of the tracks between "foo_expert" and "foo_expert+"
        1: remove the keywords like "expert" or "(double_bass)"
        2: look at the distance between the names
        3: group the track with similar names and keep the highest priority one (double bass > single bass)
        """
        names = candidates.keys()
        names = [n for n in names if n is not None]
        cleanedNames = [Converter._cleanName(name) for name in names]
        analysed = set([])
        group = []
        # Group names by similitude
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
                if similitude >= similitudeThreshold:  # aClean == bClean:
                    analysed.add(j)
                    row.append((b, priorityB))
            group.append(row)

        # Select the best version of each track when there are duplicates
        result = {}
        for row in group:
            if len(row) == 1:
                result[row[0][0]] = candidates[row[0][0]]
            else:
                key = min(row, key=lambda k: k[1])[0]
                result[key] = candidates[key]
                logging.warning("Duplicated tracks, (keeping min value): " + str(row))
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
    def plotGenres(inputFolders):
        dist = [Converter._getFileCandidates(inputFolder)[1] for inputFolder in inputFolders]
        genres = [set([genre[0] for genre in genres]) for genres in dist]
        genreMap = {
            "(9)": "Other",
            "1": "Other",
            "Alternative": "Alternative",
            "Alternative Rock": "Alternative",
            "Blues": "Rock",
            "Breakcore": "Metal",
            "Classic Rock": "Rock",
            "Classical": "Other",
            "Country": "Rock",
            "Death Metal": "Metal",
            "Deathcore": "Metal",
            "Djent": "Metal",
            "Easycore": "Metal",
            "Emo": "Metal",
            "Fusion": "Jazz",
            "Glam": "Metal",
            "Grunge": "Rock",
            "Hard Rock": "Rock",
            "Hardcore": "Metal",
            "Heavy Metal": "Metal",
            "Hip-Hop/Rap": "Hip-Hop/Rap",
            "Indie Rock": "Rock",
            "Inspirational": "Alternative",
            "J-Rock": "Rock",
            "Jazz": "Jazz",
            "Math Rock": "Rock",
            "Mathcore": "Rock",
            "Melodic Death Metal": "Metal",
            "Meoldic Death Metal": "Metal",
            "Metal": "Metal",
            "Metal5": "Metal",
            "Metalcore": "Metal",
            "New Wave": "New Wave",
            None: "Other",
            "Novelty": "Other",
            "Nu-Metal": "Metal",
            "Other": "Other",
            "Pop Punk": "Punk",
            "Pop-Rock": "Pop-Rock",
            "Pop/Dance/Electronic": "Pop/Dance/Electronic",
            "Post-Hardcore": "Metal",
            "Post-Metal": "Metal",
            "Power Metal": "Metal",
            "Prog": "Prog",
            "Prog Metal": "Metal",
            "Prog Rock": "Rock",
            "Progressive Death Metal": "Metal",
            "Progressive Metal": "Metal",
            "Progressive Rock": "Rock",
            "Progressive Thrash Metal": "Metal",
            "Punk": "Punk",
            "R&B/Soul/Funk": "R&B/Soul/Funk",
            "Reggae/Ska": "Other",
            "Rock": "Rock",
            "Rock Fusion": "Rock",
            "Southern Rock": "Rock",
            "Symphonic Metal": "Metal",
            "Tech Death": "Metal",
            "Techdeath": "Metal",
            "Thrash Metal": "Metal",
            "Video Game": "Other",
            "World": "Other",
        }

        plotData = {}
        for name, d in zip(["ADTOF-YT", "ADTOF-RGW"], dist):
            mappedGenres = defaultdict(int)
            for genre, count in d:
                mappedGenres[genreMap[genre]] += count

            # Relative frequency
            mappedGenres = {k: v / sum(mappedGenres.values()) for k, v in mappedGenres.items()}
            plotData[name] = mappedGenres

        plotData["ADTOF-YT"] = {k: v for k, v in sorted(plotData["ADTOF-YT"].items(), key=lambda x: plotData["ADTOF-YT"][x[0]] + plotData["ADTOF-RGW"][x[0]], reverse=True)}
        # Rotate the x-axis labels
        plot(plotData, "", sort=False, ylabel="Relative frequency", text=False, ylim=False)
        plt.xticks(rotation=90)
        plt.ylim(0, 0.8)
        plt.title("")
        plt.legend(["ADTOF-YT", "ADTOF-RGW"], loc="upper right")
        plt.savefig("GenreDist.pdf", bbox_inches="tight")

    @staticmethod
    def convertAll(inputFolder, outputFolder, parallelProcess=False, **kwargs):
        """
        convert all tracks in the good format
        """
        # Get all possible convertible files
        candidates, _ = Converter._getFileCandidates(inputFolder)
        # remove duplicated ones
        candidates = Converter._mergeFileNames(candidates)
        candidates = Converter._pickVersion(candidates)
        candidateName = list(candidates.values())
        candidateName.sort(key=lambda x: x["path"])
        logging.info("number of tracks in the dataset after selection: " + str(len(candidates)))

        # Do the conversion
        results = []
        if parallelProcess:
            with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
                raise DeprecationWarning("parallel process is likely not working with the added optional kwargs")
                futures = [executor.submit(Converter.runConvertors, candidate, outputFolder, trackName, **kwargs) for trackName, candidate in candidates.items()]
                concurrent.futures.wait(futures)
                results = [f._result for f in futures]
        else:
            for i, (trackName, candidate) in enumerate(list(candidates.items())):
                results.append(Converter.runConvertors(candidate, outputFolder, trackName, **kwargs))
        logging.info(str(Counter(results)))
        print(str(Counter(results)))

    @staticmethod
    def runConvertors(candidate, outputFolder, trackName, **kwargs):
        try:
            from adtof.converters.madmomBeatConverter import MadmomBeatConverter
            from adtof.converters.correctAlignmentConverter import CorrectAlignmentConverter
            from adtof.converters.phaseShiftConverter import PhaseShiftConverter
            from adtof import config

            mbc = MadmomBeatConverter()
            ca = CorrectAlignmentConverter()
            psc = PhaseShiftConverter()
            # Convert the chart into standard midi
            inputChartPath = candidate["path"]
            convertedMidiPath = os.path.join(outputFolder, config.CONVERTED_MIDI, trackName + ".midi")
            rawMidiPath = os.path.join(outputFolder, config.RAW_MIDI, trackName + ".midi")
            audioPath = os.path.join(outputFolder, config.AUDIO, trackName + ".ogg")
            if not config.checkAllPathsExist(convertedMidiPath, rawMidiPath, audioPath):
                psc.convert(inputChartPath, convertedMidiPath, rawMidiPath, audioPath, debug=True, **kwargs)

            # Align the annotations by looking at the average beat estimation difference
            # RVDrumsEstimationPath = os.path.join(outputFolder, config.RV_ESTIMATIONS, trackName + ".drums.txt")
            beatsEstimationsPath = os.path.join(outputFolder, config.BEATS_ESTIMATIONS, trackName + ".txt")
            beatsActivationPath = os.path.join(outputFolder, config.BEATS_ACTIVATION, trackName + ".npy")
            convertedDrumAnotationsPath = os.path.join(outputFolder, config.CONVERTED_DRUM, trackName + ".txt")
            alignedBeatsAnnotationsPath = os.path.join(outputFolder, config.ALIGNED_BEATS, trackName + ".txt")
            alignedDrumAnotationsPath = os.path.join(outputFolder, config.ALIGNED_DRUM, trackName + ".txt")
            alignedMidiAnotationsPath = os.path.join(outputFolder, config.ALIGNED_MIDI, trackName + ".midi")
            if not config.checkAllPathsExist(beatsEstimationsPath, beatsActivationPath):
                mbc.convert(audioPath, convertedMidiPath, beatsEstimationsPath, beatsActivationPath)
            if not config.checkAllPathsExist(alignedDrumAnotationsPath, alignedBeatsAnnotationsPath, alignedMidiAnotationsPath, convertedDrumAnotationsPath):
                ca.convert(
                    beatsActivationPath,
                    convertedMidiPath,
                    convertedDrumAnotationsPath,
                    alignedDrumAnotationsPath,
                    alignedBeatsAnnotationsPath,
                    alignedMidiAnotationsPath,
                )

            return "converted"

        except Exception as e:
            logging.warning(trackName + " not converted: " + str(e))
            return str(e)
