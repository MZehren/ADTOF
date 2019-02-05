import os
import sys


class Converter(object):
    """
    Base class to convert file formats
    """

    def convert(self, folder, inputFile, outputName):
        """
        Base method to convert a file
        """
        raise NotImplementedError()

    def isConvertible(self, folder, inputFile):
        """
        Base method to check if a file is convertible
        """
        raise NotImplementedError()

    def convertRecursive(self, rootFodler, outputName):
        """
        Go recursively inside the folders and convert everything convertible
        """
        converted = 0
        failed = 0
        for root, _, files in os.walk(rootFodler):
            for file in files:
                if self.isConvertible(root, file):
                    try:
                        self.convert(root, file, outputName)
                        print("converted", root, file)
                        converted += 1
                    except ValueError:
                        print("Unexpected error:", sys.exc_info()[0], sys.exc_info()[1])
                        print("for file:", root, file)
                        failed += 1

        print("converted:", converted, "failed:", failed)
