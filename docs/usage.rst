Usage
=====

There are multiple file formats you could import from, each one comes from a specific game. 
In this page, you will see a little summary on how to handle those formats.
For now, we offer to import files from the games Phase Shift and we are pointing to a solution to handle Rock Band (1 to 4).
With those two formats we are covering the majority of ressources available online. 
The charts are then converted to a standard MIDI file aligned to an audio file.

Standard MIDI
-------------

We aim to convert the games' file format to standard midi files to maximises the compatibility with other softwares or tools.
The resulting file is a standard `format 1 MIDI file`_ with two tracks:

 - The 'tempo map' containing all the tempo and time signature events.
 - The drums track containing all the drums note on events.

This MIDI file is going to be aligned with the music file included in the charts of the game.


Phase Shift
-----------

Format
~~~~~~

`Phase Shift`_ is a free Guitar Hero clone. 
The charts distributed for this game are contained in a folder with multiple files:

 - **guitar.ogg**: the audio from the track annotated. This file can sometimes be in multitracks Mogg when separated stems are available, which is not common for custom tracks.
 - **notes.mid**: a format 1 MIDI file containing the notes to be played by the different instruments such as the drums or the guitars. The pitches in this file are not standard and need to be converted with the script in this repository. The MIDI specifications of this format seem to be the same as Rock Band's format, which is described on c3universe.com_.
 - **song.ini**: a file containing some meta information such as the delay between the audio and the MIDI files.

Conversion
~~~~~~~~~~

To convert a Phase Shift MIDI file to a standard one, run the script:

>>> cd ADTOF
>>> python PhaseShiftConverter.py -h
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



This script is going to convert the Phase Shift midi file to a "note_std.mid" file by applying multiple corrections:

 - Only one track containing drums events is kept and all the other instruments are discarded.
 - Toms notes which are cymbal events played during a modifying note are converted to standard events.
 - The Phase Shift pitches are converted to standard pitches following the mapping in PhaseShiftMidiToStandard.json_. You can see the list of standard pitches on Wikipedia_.
 - The ambiguous midi classes are reduced to general ones following StandardMidiToReduced.json_. For examples, all the toms (notes 41, 43, 45, 47, 48, and 50) are converted to floor tom events (note 41). Feel free to change this mapping to fit your needs and your own classes.
 - If the argument *-d* is present, a delay at the start of the MIDI file is added to match the eventual delay in **song.ini** .

.. _PhaseShiftMidiToStandard.json: https://github.com/MZehren/ADTOF/blob/master/ADTOF/mappingDictionaries/PhaseShiftMidiToStandard.json
.. _StandardMidiToReduced.json: https://github.com/MZehren/ADTOF/blob/master/ADTOF/mappingDictionaries/StandardMidiToReduced.json

Rock Band
-----------

Format
~~~~~~

Rock Band is a series of video games released on multiples consoles such as the PlayStation 3 or the Xbox 360.
The charts used are either in **con** or **rb3con** formats, which are binary files encapsulating the audio file as well as the MIDI events. 

Conversion
~~~~~~~~~~

The tool C3CONTools_ can be used to convert Rock Band format to Phase Shift in batches. 
Phase Shift format can then be converted to a standard MIDI file with the method described above.



.. _format 1 MIDI file: https://www.csie.ntu.edu.tw/~r92092/ref/midi/#mff1   
.. _Phase Shift: http://www.dwsk.co.uk/index_phase_shift.html
.. _Wikipedia: https://en.wikipedia.org/wiki/General_MIDI#Percussive
.. _C3CONTools: http://customscreators.com/index.php?/topic/9095-c3-con-tools-v400-012518/
.. _c3universe.com: http://docs.c3universe.com/rbndocs/index.php?title=Drum_Authoring