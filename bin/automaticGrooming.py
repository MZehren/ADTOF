#!/usr/bin/env python
# encoding: utf-8

import argparse
import logging

from adtof.converters.converter import Converter


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description="Process a chart folder with the automatic cleaning procedure")
    parser.add_argument("inputFolder", type=str, help="Path to the chart folder.")
    parser.add_argument("outputFolder", type=str, help="Path to the destination folder of the dataset.")
    parser.add_argument("-p", "--parallel", action="store_true", help="Run the cleansing in parallel")
    parser.add_argument("-t", "--task", type=str, help="Number of classes for the conversion", default="5")
    parser.add_argument(
        "-a",
        "--animation",
        action="store_true",
        help="If the notes converted come from the animation annotations. (default: False, but the discrepancies between animation and gameplay are still corrected)",
    )

    args = parser.parse_args()

    Converter.convertAll(args.inputFolder, args.outputFolder, parallelProcess=args.parallel, task=args.task, useAnimation=args.animation)
    print("Done!")


if __name__ == "__main__":
    main()
