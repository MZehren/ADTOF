import os
import sys


class Converter(object):
    """
    Base class to convert file formats
    """

    def convert(self, inputPath, outputName):
        """
        Base method to convert a file
        """
        raise NotImplementedError()



    def convertRecursive(self, rootFodler, outputName):
        """
        Go recursively inside the folders and convert everything convertible
        """
        raise NotImplementedError
