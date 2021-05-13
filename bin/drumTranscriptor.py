#!/usr/bin/env python
# encoding: utf-8
import argparse

from adtof.model.model import Model


def main():
    parser = argparse.ArgumentParser(description="Use one of the three trained model to perform ADT")
    parser.add_argument("inputPath", type=str, help="Path to a music file or folder containing music")
    parser.add_argument("outputPath", type=str, default="./", help="Path to output folder")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Name of the pre-trained model used for the transcription. Values: crnn-ADTOF, crnn-all, crnn-ptTMIDT. (default: crnn-ADTOF)",
        default="crnn-ADTOF",
    )
    args = parser.parse_args()

    # Get the model
    model, hparams = Model.modelFactory(modelName=args.model, fold=0)
    assert "peakThreshold" in hparams
    assert model.weightLoadedFlag

    model.predictFolder(args.inputPath, args.outputPath, **hparams)


if __name__ == "__main__":
    main()
