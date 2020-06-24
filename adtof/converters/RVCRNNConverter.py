import os
from adtof.converters.converter import Converter
import subprocess
from pkg_resources import resource_filename


class RVCRNNConverter(Converter):
    """
    Annotate from RV-CRNN8 an audio file with the drum transcrption
    """

    def convert(self, inputAudio, outputFile):
        # python3 DrumTranscriptor -m CRNN_8 batch -o /mnt/e/ADTSets/adtof/parsed/RV-CRNN_8 /mnt/e/ADTSets/adtof/parsed/audio/*.ogg
        args = [
            resource_filename(__name__, "../../vendors/madmomDrumsEnv/bin/python"),
            resource_filename(__name__, "../../vendors/madmom-0.16.dev0/bin/DrumTranscriptor"),
            "-m",
            "CRNN_8",
            "single",
            inputAudio,
        ]  # Calling python from python, Yay...
        process = subprocess.Popen(args, stdout=subprocess.PIPE)
        output = process.stdout.read().decode()
        with open(outputFile, "w") as file:
            file.write(output)
