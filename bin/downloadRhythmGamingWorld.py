#!/usr/bin/env python
# encoding: utf-8

import argparse
import logging
import os
import shutil

import pandas as pd
from adtof import config
from adtof import config
from adtof.converters.converter import Converter
from adtof.io.ccDownloader import scrapIndex


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("outputFolder", type=str, help="Path to the destination folder.")
    # parser.add_argument("-p", "--parallel", action="store_true", help="Set if the conversion is ran in parallel")
    args = parser.parse_args()

    scrapIndex(path=args.outputFolder)

    print("Done!")


if __name__ == "__main__":
    main()
