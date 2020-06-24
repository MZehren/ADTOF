import os

from adtof.converters.converter import Converter


class RockBandConverter(Converter):
    """
    Convert a RockBand file in TODO
    """

    BEGINNING_OF_FILES = ["CON", "RBSF"]
    FILE_EXTENSION = [".rba", "_rb3con"]

    def isConvertible(self, file):
        # Check if the extension is known
        if any([True for i in RockBandConverter.FILE_EXTENSION if i in file]):
            return True

        # Check if the file starts with the good string
        try:
            with open(file, "r") as f:
                firstLine = f.readline()
                if any([True for i in RockBandConverter.BEGINNING_OF_FILES if i in firstLine[:4]]):
                    return True
        except:
            return False

    def getTrackName(self, file):
        # TODO: improve that by reading the binary file maybe
        name = os.path.split(file)[-1]
        for ext in RockBandConverter.FILE_EXTENSION:
            name = name.split(ext)[0]
        return name
