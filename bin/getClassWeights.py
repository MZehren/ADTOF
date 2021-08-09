#!/usr/bin/env python
# encoding: utf-8
import argparse

from adtof import config
from adtof.model import dataLoader


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    parser = argparse.ArgumentParser(description="todo")
    parser.add_argument("folderPath", type=str, help="Path.")
    args = parser.parse_args()
    labels = config.LABELS_5
    sampleRate = 100

    # Get the data
    classWeight = dataLoader.getClassWeight(args.folderPath, sampleRate=sampleRate, labels=labels)
    print(classWeight)


if __name__ == "__main__":
    main()
