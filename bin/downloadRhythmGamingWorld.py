#!/usr/bin/env python
# encoding: utf-8

import argparse
import logging

from adtof.io.ccDownloader import scrapIndex


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description="Download custom charts fron the website https://rhythmgamingworld.com/")
    parser.add_argument("outputFolder", type=str, help="Path to the destination folder where the files are downloaded.")
    args = parser.parse_args()

    scrapIndex(path=args.outputFolder)
    print("Done!")


if __name__ == "__main__":
    main()
