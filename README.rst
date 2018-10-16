ADTOF
=====
Automatic Drum Transcription On Fire (ADTOF) is an effort to list existing format of drum transcriptions available online and utilize them as ADT training data by converting them to a standard MIDI file aligned to an audio file.

Right now, we offer to convert charts from the game Phase Shift to an easy to use file format.

Documentation
-------------
Documentation can be found online at https://adtof.readthedocs.io/en/latest/

Installation
------------
This package needs python-midi_ which may only be available for python2.

To install the dependencies run:

>>> pip install -r requirements.txt

.. _python-midi: https://github.com/vishnubob/python-midi

Usage
-----

To convert a chart run:

>>> cd ADTOF
>>> python PhaseShiftConverter.py [-h] [-o OUTPUTNAME] [-d] folderPath
