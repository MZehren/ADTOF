#!/usr/bin/env python
# encoding: utf-8
"""
Convert the PhaseShift's MIDI file format into real MIDI
"""
import argparse

from adtof.io.converters import OnsetsAlignementConverter


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('rootFolder', type=str,
                        help="Path to the root folder of the conversion.")
    parser.add_argument("-i", '--input', dest="inputName", type=str,
                        default="notes_std.mid", help="name of the midi files to convert.")
    parser.add_argument(
        '-o',
        '--output',
        dest='outputName',
        type=str,
        default="notes_aligned.mid",
        help="Name of the MIDI file created from the alignment. Default to 'notes_aligned.mid'"
    )
    args = parser.parse_args()

    oac = OnsetsAlignementConverter()
    oac.convertRecursive(args.rootFolder, args.outputName, [args.inputName])
    print("Done!")


if __name__ == '__main__':
    main()