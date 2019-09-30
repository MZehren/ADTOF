from collections import OrderedDict
from typing import List

import guitarpro
import midi

from adtof.io import MidiProxy
from adtof.io.converters import Converter, PhaseShiftConverter


class GuitarProToMidiConverter(Converter):
    """
    Convert a guitar pro file (.gp5) to midi
    see https://github.com/alexsteb/GuitarPro-to-Midi
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def convert(self, filePath, outputName=None):

        with open(filePath, mode='rb') as file:
            binary = file.read()

        gp = guitarpro.parse(filePath)
        midi = self.generateMidi(gp)
        return

    def generateMidi(self, gp):
        print(gp)
        pattern = midi.Pattern()
        tick = 0
        for track in gp.tracks:
            if not track.channel.isPercussionChannel:
                continue
            for measure in track.measures:
                for voice in measure.voices:
                    for beat in voice.beats:
                        on = midi.NoteOnEvent(tick=0, velocity=20, pitch=midi.G_3)



        

g = GuitarProToMidiConverter()
g.convert("/home/mickael/Documents/Datasets/drumsTranscription/Transcriptions [By Alex Rudinger]/01 AAL/01 GuitarPro Files/01 Tempting Time.gp5")
# g.convert("E:/ADTSets/Transcriptions [By Alex Rudinger]/01 AAL/01 GuitarPro Files/01 Tempting Time.gp5")
