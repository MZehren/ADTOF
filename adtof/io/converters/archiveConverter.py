import os

from pyunpack import Archive

from adtof.io.converters import Converter


class ArchiveConverter(Converter):
    """
    uncompress an archive file such a .zip or .rar
    """

    def convert(self, inputPath, outputName=None):
        if self.isConvertible(inputPath):
            folderName = outputName if outputName is not None else inputPath[:-4]
            Archive(inputPath).extractall(folderName, auto_create_dir="True")
            os.remove(inputPath)

    def isConvertible(self, path):
        if path[-3:] == "zip" or path[-3:] == "rar":
            return path
        else:
            return None
