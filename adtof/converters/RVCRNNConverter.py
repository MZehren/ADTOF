from adtof.converters.converter import Converter
import subprocess
from pkg_resources import resource_filename


class RVCRNNConverter(Converter):
    """
    Annotate from RV-CRNN8 an audio file with the drum transcrption
    """

    def convertAll(self, inputFolder, outputFolder):
        """
        ConvertAll!
        """
        # python3 DrumTranscriptor -m CRNN_8 batch -o /mnt/e/ADTSets/adtof/parsed/RV-CRNN_8 /mnt/e/ADTSets/adtof/parsed/audio/*.ogg
        args = [
            resource_filename(__name__, "../../vendors/madmomDrumsEnv/bin/python"),
            resource_filename(__name__, "../../vendors/madmom-0.16.dev0/bin/DrumTranscriptor"),
            "-m",
            "CRNN_8",
            "batch",
            "-o",
            outputFolder,
            inputFolder + "*.ogg",
        ]  # Calling python from python, Yay...

        # Doing that manually while I am ot sure to keep that.
        # TODO: Remmove the need to manually copy and paste the line in cmd
        print(" ".join(args))

        # process = subprocess.Popen(args, stdout=subprocess.PIPE)
        # output = process.stdout.read().decode()
        # print(output)
