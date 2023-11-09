#!/usr/bin/env python


from collections import defaultdict

from adtof import config
from adtof.ressources import instrumentsMapping


class TextReader(object):
    """
    Convert the text format from rbma_13 and MDBDrums to midi
    """

    def castInt(self, s):
        """
        Try to convert a string in int if possible
        """
        try:
            casted = int(float(s))
            return casted
        except ValueError:
            return s

    def decode(self, line, sep):
        values = line.replace("\r\n", "").replace("\n", "").split(sep)
        values = [v.replace(" ", "") for v in values]
        time = values[0]
        pitch = values[1]
        time = float(time)
        pitch = self.castInt(pitch)
        velocity = float(values[2]) if len(values) > 2 else 1

        return {"time": time, "pitch": pitch, "velocity": velocity}

    def getOnsets(self, txtFilePath, sep="\t", **kwargs):
        """
        Parse the text file following Mirex encoding:
        [time]\t[class]\t[velocity]\n
        """
        events = []
        with open(txtFilePath, "r") as f:
            for line in f:
                try:
                    events.append(self.decode(line, sep))
                except Exception as e:
                    print("Line couldn't be decoded, passing.", repr(line), str(e))
                    continue

        return events

    def writteBeats(self, path, beats):
        """ """
        with open(path, "w") as f:
            f.write("\n".join([str(time) + "\t" + str(beatNumber) for time, beatNumber in beats]))
