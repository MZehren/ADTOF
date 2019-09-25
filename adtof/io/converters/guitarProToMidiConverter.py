from struct import unpack

from adtof.io import MidiProxy
from adtof.io.converters import Converter, PhaseShiftConverter
import string

# from dtw import dtw

class binaryReader(Object):

    def __init__(self, path):
        with open(path, mode='rb') as file: 
            self.binary = file.read()
        self.cursor = 0
        self.data = {}
        self.types = {
            "ibs": self._readIntByteString,
            "bs": self._readByteString,
            "s":self._readString,
            "i":self._readInt,
            "B":self._readByte,
            "sB":self._readSignedByte,
            "S":self._readShort,
            "?":self._readBool,
            "s":self._skip
        }
    
    def readField(name, type, *args):
        String
        self.data[name] = self.types[type](*args)

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
        result = "".join([c.decode("latin-1") for c in unpack("c" * length, self.binary[self.cursor:self.cursor + length])])
        self.cursor += length
        return result

    def _readInt(self):
        """
        read an int and increase the curose
        """
        result = unpack("i", self.binary[self.cursor:self.cursor + 4])[0]
        self.cursor += 4
        return result

    def _readByte(self):
        """
        read a byte and increase the curose
        """
        result = self.binary[self.cursor]
        self.cursor += 1
        return result

    def _readSignedByte(self):
        """
        TODO
        """
        result = self.binary[self.cursor]
        self.cursor += 1
        return result

    def _readShort(self):
        """
        read a short and increase the cursor
        """
        result = unpack("h", self.binary[self.cursor:self.cursor + 2])[0]
        self.cursor += 2
        return result

    def _readBool(self):
        """
        read a bool and increase the cursor
        """
        result = unpack("?", self.binary[self.cursor:self.cursor + 1])[0]
        self.cursor += 1
        return result
    
    def _skip(self, inc):
        """
        move the cursor
        """
        self.cursor += inc


class GuitarProToMidiConverter(Converter):
    """
    Convert a guitar pro file (.gp5) to midi
    see https://github.com/alexsteb/GuitarPro-to-Midi
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cursor = 0
        self.data = None

    def otherConvert(self):
        pass
    def convert(self, filePath, outputName=None):
        
        props = {}



        # read Infos
        # TODO split that into multiple functions
        version = self._readByteString(30)
        props["versionMajor"], props["versionMinor"] = [int(v) for v in version[-4:].split(".")]
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

        # read tracks
        for i in range(trackCount):
            if (i == 0 or props["versionMinor"] == 0):
                self.cursor += 1
            flags1 = self._readByte()
            isPercussionTrack = flags1 & 0x01
            is12StringedGuitarTrack = flags1 & 0x02
            isBanjoTrack = flags1 & 0x03
            isVisible = flags1 & 0x04
            isSolo = flags1 & 0x05
            isMute = flags1 & 0x06
            useRSE = flags1 & 0x07
            indicateTuning = flags1 & 0x08

            name =  self._readByteString(40)
            stringCount = self._readInt()

            for j in range(7):
                iTuning = self._readInt()
            
            port = self._readInt()
            # channel
            index = self._readInt()
            effectChannel = self._readInt() -1
            fretCount = self._readInt()
            offset = self._readInt()
            color = [self._readByte() for c in range(4)]

            # gp5 part
            flags2 = self._readShort()
            tablature = ((flags2 & 0x0001) != 0)
            notation = ((flags2 & 0x0002) != 0)
            diagramsAreBelow = ((flags2 & 0x0004) != 0)
            showRyhthm = ((flags2 & 0x0008) != 0)
            forceHorizontal = ((flags2 & 0x0010) != 0)
            forceChannels = ((flags2 & 0x0020) != 0)
            diagramList = ((flags2 & 0x0040) != 0)
            diagramsInScore = ((flags2 & 0x0080) != 0)
            autoLetRing = ((flags2 & 0x0200) != 0)
            autoBrush = ((flags2 & 0x0400) != 0)
            extendRhythmic = ((flags2 & 0x0800) != 0)

            RSEautoAccentuation = self._readByte()
            channelBank = self._readByte()

            # readTrackRSE
            humanize = self._readByte()
            self._readInt()
            self._readInt()
            self._readInt()
            self.cursor += 12
            # read RSE Instrument
            instrument = self._readInt()
            _ = self._readInt()
            soundbank = self._readInt()
            if props["versionMinor"] == 0:
                effectNumber = self._readShort()
                self.cursor += 1
            else:
                effectNumber = self._readInt()
            if props["versionMinor"] >= 10:
                self.cursor += 4
            if props["versionMinor"] > 0:
                effect = self._readIntByteString()
                effectCategory = self._readIntByteString()

        if props["versionMinor"] == 0:
            self.cursor += 2
        else:
            self.cursor += 1
        
        # readMeasures
        for header in headers:
            # if header == 0x60:
            #     pass
            for track in range(trackCount):
                for voice in range(2):
                    beats = self._readInt()
                    for beat in range(beats):
                        flags = self._readByte()
                        if (flags & 0x40) != 0:
                            status = self._readByte()
                        durationValue = 1 << self._readSignedByte() + 2
                        durationIsDoted = flags & 0x01 != 0
                        if flags & 0x20:
                            iTuplet = self._readInt()
                        if flags & 0x02: #readChord
                            newFormat = self._readBool()
                            if newFormat:
                                isSharp = self._readBool()
                                self.cursor += 3
                                root = self._readByte()
                                typeOf = self._readByte()
                                extension = self._readByte()
                                bass = self._readInt()
                                tonality = = self._readInt()
                                add = self._readBool()
                                name = self._readIntByteString()
                                fifth = self._readByte()
                                ninth = self._readByte()
                                eleventh = self._readByte()
                                firstFret = self._readInt()
                                strings = [self._readInt() for fret in range(7)]
                                barresCount = self._readByte()
                                barreFrets = [self._readByte() for bla in range(5)]
                                barreStarts = [self._readByte() for bla in range(5)]
                                barreEnds = [self._readByte() for bla in range(5)]
                                omission = [self._readBool() for bla in range(7)]
                                self.cursor += 1
                                for fingering in range(7):
                                    finger = self._readSignedByte()
                                show = self._readBool()
                            else:
                                name = self._readIntByteString()
                                firstFret = self._readInt()
                                if firstFret > 0:
                                    for fret in range(6):
                                        string = self._readInt()

                        # if ((flags & 0x04) != 0) beat.text = readText();
                        # if ((flags & 0x08) != 0) beat.effect = readBeatEffects(effect);
                        # if ((flags & 0x10) != 0)


        print("lol ?")




g = GuitarProToMidiConverter()
# "/home/mickael/Documents/Datasets/drumsTranscription/Transcriptions [By Alex Rudinger]/01 AAL/01 GuitarPro Files/01 Tempting Time.gp5"
g.convert("E:/ADTSets/Transcriptions [By Alex Rudinger]/01 AAL/01 GuitarPro Files/01 Tempting Time.gp5")
