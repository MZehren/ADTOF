ADTOF
=====
Automatic Drum Transcription On Fire (ADTOF) is an effort to list existing formats of drum transcriptions available online and utilize them as ADT training data by converting them to a standard MIDI file aligned to an audio file.

Right now, we offer to convert charts from the game Phase Shift to an easy to use file format.

Documentation
-------------
Documentation can be found online at https://adtof.readthedocs.io/en/latest/

Installation
------------
This package needs python-midi_

To install the dependencies run:

>>> pip install -r requirements.txt

.. _python-midi: https://github.com/vishnubob/python-midi

Usage
-----

To convert a chart run:

>>> cd ADTOF
>>> python PhaseShiftConverter.py [-h] [-o OUTPUTNAME] [-d] folderPath
usage: PhaseShiftConverter.py [-h] [-o OUTPUTNAME] [-d] folderPath
Process a Phase Shift chart folder and convert the MIDI file to standard MIDI
positional arguments:
  folderPath            Path to the Phase Shift chart folder.
optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUTNAME, --output OUTPUTNAME
                        Name of the MIDI file created from the conversion.
                        Default to 'notes_std.mid'
  -d, --delay           Add a delay in the MIDI file according to the
                        song.ini's delay property