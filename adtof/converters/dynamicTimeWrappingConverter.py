import librosa

from adtof.converters import Converter
from adtof.io import MidoProxy
from dtw import dtw


class DynamicTimeWrappingConverter(Converter):
    """
    Converter which tries to align the music file to the midi file as close as possible
    """

    def convert(self, inputMusicPath, inputMidiPath, outputName):
        midi = MidoProxy(inputMidiPath)
        midiEvents = midi.getEventsPosition()

        y, sr =  librosa.load(inputMusicPath)
        onsets = librosa.onset.onset_detect(y=y, sr=sr)
        pass

    # def isConvertible(self, folder, inputFile):
    #     files = os.listdir(folderPath)
    #     return inputFile == PhaseShiftConverter.PS_MIDI_NAME and  "song.ini" in files and ("song.ogg" in files or "guitar.ogg" in files)
