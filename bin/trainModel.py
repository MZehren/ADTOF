#!/usr/bin/env python
# encoding: utf-8
"""
TODO
"""
import argparse
import os

from adtof.io import CQT
from adtof.io.converters import PhaseShiftConverter
from adtof.deepModels import RV1

def loadData(path):
    psc = PhaseShiftConverter()
    cqt = CQT()

    for root, dirs, files in os.walk(path):
        # fullPath = os.sep.join(path)
        midi, audio = psc.getConvertibleFiles(root)
        if audio and midi:
            try:
                y = psc.convert(root).getDenseEncoding(sampleRate=98.4375, timeShift=0)
                x = cqt.open(os.sep.join([root, audio]))[:len(y)]
                return x, y
            except:
                print(root + " not working")
    # x = fe.open(path)
    # y = []

    return x, y


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    parser = argparse.ArgumentParser(description='todo')
    parser.add_argument('folderPath', type=str, help="Path.")
    args = parser.parse_args()

    model = RV1().createModel()
    X, Y = loadData(args.folderPath)
    model.fit(X, Y, epochs=5)
    
    print("Done!")


if __name__ == '__main__':
    main()
