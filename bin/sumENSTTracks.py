import argparse
import os

import ffmpeg
from adtof import config


def main(
    wetFolder, accompaniementFolder, outputFolder,
):
    wetTracks = config.getFilesInFolder(wetFolder)
    accompaniementTracks = config.getFilesInFolder(accompaniementFolder)

    wetTracks, accompaniementTracks = config.getIntersectionOfPaths(wetTracks, accompaniementTracks)
    for wetTrack, accompaniementTrack in zip(wetTracks, accompaniementTracks):
        mergeTracks([str(wetTrack), str(accompaniementTrack)], os.path.join(outputFolder, os.path.basename(wetTrack)))


def mergeTracks(audioFiles, pathOutput, weights="2 1"):
    """
    Using ffmpeg to sum tracks with identical names in two different folders. 
    Use the weights parameter to specify how loud the tracks from a specific folder are going to be (see Wu et. al.)
    """
    ffmpeg.filter([ffmpeg.input(audioFile) for audioFile in audioFiles], "amix", inputs=len(audioFiles), weights=weights).output(
        pathOutput
    ).global_args("-loglevel", "error").run(overwrite_output=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script used to sum drum and accompaniment files in ENST. Simply copy the ")
    parser.add_argument("ENSTFolder", type=str, help="Path to ENST dataset.")
    args = parser.parse_args()
    main(
        os.path.join(args.ENSTFolder, "audio_wet"),
        os.path.join(args.ENSTFolder, "audio_accompagniement"),
        os.path.join(args.ENSTFolder, "audio_sum"),
    )
