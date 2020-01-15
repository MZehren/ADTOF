#!/usr/bin/env python
# encoding: utf-8
"""
TODO
"""
import argparse
import datetime
import logging
import os
import random


import numpy as np
import pandas as pd
import sklearn
import torch

from adtof import config
from adtof.deepModels.peakPicking import PeakPicking
from adtof.deepModels.rv1pt import RV1Torch
from adtof.io.converters.converter import Converter
from adtof.io.mir import MIR
from adtof.io.myMidi import MidiProxy

logging.basicConfig(filename='logs/conversion.log', level=logging.DEBUG)




def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    parser = argparse.ArgumentParser(description='todo')
    parser.add_argument('folderPath', type=str, help="Path.")
    args = parser.parse_args()

    # Get the data
    trainLoader = torch.utils.data.DataLoader(TorchIterableDataset(args.folderPath, train=True), batch_size=100)
    testLoader = torch.utils.data.DataLoader(TorchIterableDataset(args.folderPath, train=False), batch_size=100)
    classes = [36, 40, 41, 46, 49]

    dataiter = iter(trainLoader)
    images, labels = dataiter.next()
    print("Done!")


if __name__ == '__main__':
    main()
