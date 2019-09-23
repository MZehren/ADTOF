from struct import unpack

from adtof.io import MidiProxy
from adtof.io.converters import Converter, PhaseShiftConverter
import string

# from dtw import dtw


class GuitarProToMidiConverter(Converter):
    """
    Convert a guitar pro file (.gp5) to midi
    see https://github.com/alexsteb/GuitarPro-to-Midi
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cursor = 0
        self.data = None

    def convert(self, filePath, outputName=None):
        with open(filePath, mode='rb') as file:  # b is important -> binary
            self.data = file.read()
        props = {}

        # read Infos
        version = self._readByteString(30)
        props["versionMajor"], props["versionMinor"] = version[-4:].split(".")
        for key in ["title", "subtitle", "interpret", "album", "words", "music", "copyright", "tab", "instructional"]:
            props[key] = self._readIntByteString()
        props["notesCount"] = self._readInt()
        for i in range(props["notesCount"]):
            self._readIntByteString()
        props["lyricsTrackChoice"] = self._readInt()
        for i in range(5):
            lyricStartingMeasure = self._readInt()
            lyrics = self._readString(self._readInt())
        props["masterEffectVolume"] = self._readInt()
        self._readInt()
        for i in range(11):
            rseEqualizer = self._readSignedByte()
        props["pageSetup"] = {
            "pageSize": [self._readInt(), self._readInt()],
            "padding": [self._readInt(), self._readInt(), self._readInt(), self._readInt()],
            "scoreSizeProportion": self._readInt(),
            "headerAndFooter": self._readShort()
        }
        for key in ["title", "subtitle", "artist", "album", "words", "music", "wordsAndMusic", "copyright1", "copyright2", "pageNumber"]:
            props["pageSetup"][key] = self._readIntByteString()
        props["tempoName"] = self._readIntByteString()
        tempo = self._readInt()
        hideTempo = self._readBool() if int(props["versionMinor"]) > 0 else False
        key = self._readSignedByte()
        octabe = self._readInt()
        for i in range(64):  # midiChannels
            instrument = self._readInt()
            isPercussion = i % 16 == 9
            for key in ["volume", "balance", "chorus", "reverb", "phaser", "tremolo"]:
                self._readByte()
            self.cursor += 2
        for i in range(19):  # Directions is a list of 19 short each pointing at the number of measure.
            self._readShort()
        reverb = self._readInt()
        measureCount = self._readInt()
        trackCount = self._readInt()

        # read measureHeaders
        headers = []
        for i in range(measureCount):
            if i != 0:
                self.cursor += 1
            flags = self._readByte()
            header = {
                "tempo": tempo,
                "timeSignature.numerator": self._readSignedByte() if flags & 0x01 else None,
                "timeSignature.denominator": self._readSignedByte() if flags & 0x02 else None,
                "isRepeatOpen": (flags & 0x04) != 0,
                "repeatClose": self._readSignedByte() if flags & 0x08 else None,
                "repeatAlternatives": self._readByte() if flags & 0x10 else None,
                "marker": [self._readIntByteString(),
                           self._readByte(), self._readByte(),
                           self._readByte(), self._readByte()] if flags & 0x20 else None,
                "root": [self._readSignedByte(), self._readSignedByte()] if flags & 0x40 else None,
                "hasDoubleBar": (flags & 0x80) != 0
            }

            if header["repeatClose"] and header["repeatClose"] > -1:
                header["repeatClose"] -= 1
            header["timeSignature.beams"] = [self._readByte() for i in range(4)] if flags & 0x03 else None
            if (flags & 0x10) == 0:
                self.cursor += 1
            header["tripleFeel"] = self._readByte()
            headers.append(header)

        print("lol ?")

    def _readIntByteString(self):
        """
        read the size(int) length(byte) string(char*length)
        and move the cursor by size +4 ?
        """
        size = self._readInt() - 1
        return self._readByteString(size)

    def _readByteString(self, size):
        """
        read the length(byte) string(char*length)
        and move the cursor by size
        """
        length = self._readByte()
        return self._readString(size)[:length]

    def _readString(self, length):
        """
        read string and increase the cursor
        """
        result = "".join([c.decode("latin-1") for c in unpack("c" * length, self.data[self.cursor:self.cursor + length])])
        self.cursor += length
        return result

    def _readInt(self):
        """
        read an int and increase the curose
        """
        result = unpack("i", self.data[self.cursor:self.cursor + 4])[0]
        self.cursor += 4
        return result

    def _readByte(self):
        """
        read a byte and increase the curose
        """
        result = self.data[self.cursor]
        self.cursor += 1
        return result

    def _readSignedByte(self):
        """
        TODO
        """
        result = self.data[self.cursor]
        self.cursor += 1
        return result

    def _readShort(self):
        """
        read a short and increase the cursor
        """
        result = unpack("h", self.data[self.cursor:self.cursor + 2])[0]
        self.cursor += 2
        return result

    def _readBool(self):
        """
        read a bool and increase the cursor
        """
        result = unpack("?", self.data[self.cursor:self.cursor + 1])[0]
        self.cursor += 1
        return result


g = GuitarProToMidiConverter()
g.convert("/home/mickael/Documents/Datasets/drumsTranscription/Transcriptions [By Alex Rudinger]/01 AAL/01 GuitarPro Files/01 Tempting Time.gp5")
