#!/usr/bin/env python
# encoding: utf-8
"""
TODO
"""
import argparse

from adtof.io import FeatureExtraction


def loadData(path):
    fe = FeatureExtraction()
    x = fe.open(path)
    y = []

    return x, y


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    parser = argparse.ArgumentParser(description='todo')
    parser.add_argument('folderPath', type=str, help="Path.")
    args = parser.parse_args()

    X, Y = loadData(args.folderPath)

    print("Done!")


if __name__ == '__main__':
    main()
