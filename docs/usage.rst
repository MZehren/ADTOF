Usage
=====

There are multiple file formats you could import from, each one comes from a specific game. 
In this page, you will see a little summary on how to handle those formats.
For now, we offer to import files from the games Phase Shift and Rock Band (1 to 4) and then convert them to a standard MIDI file aligned to an audio file.

Standard MIDI
-------------

We aim to convert the games' file format to standard midi files to maximises the compatibility with other softwares.
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
 - **notes.mid**: a format 1 MIDI file containing the notes to be played by the different instruments such as the drums or the guitars. The note from the events are not standard and need to be converted with the script in this repository. The MIDI specifications of this format seem to be the same as Rock Band's format, which is described on c3universe.com_.
 - **song.ini**: a file containing some meta information such as the delay between the audio and the MIDI files.

Conversion
~~~~~~~~~~

To convert a Phase Shift MIDI file to a standard one, run the script:

>>> cd ADTOF
>>> python PhaseShiftConverter.py [-h] [-o OUTPUTNAME] [-d] folderPath

This script is going to convert the Phase Shift midi file to a "note_std.mid" file by applying multiple corrections:

 - Only one track containing drums event is kept and all the other instruments are discarded.
 - Phase Shift format handles toms notes with the same MIDI pitch as cymbal notes played at the same time as a modifying note. We convert the events with modifiers to the standard ones.
 - The Phase Shift pitches are converted to standard pitches following the mapping in PhaseShiftMidiToStandard.json_. You can see the list of standard pitches on Wikipedia_.
 - The ambiguous midi classes are reduced to the general ones following StandardMidiToReduced.json_. Feel free to change this mapping to fit your needs.
 - A delay at the start of the MIDI file is added to match the eventual delay in **song.ini** if the argument *-d* is present.

.. _PhaseShiftMidiToStandard.json: https://github.com/MZehren/ADTOF/blob/master/ADTOF/conversionDictionnaries/PhaseShiftMidiToStandard.json
.. _StandardMidiToReduced.json: https://github.com/MZehren/ADTOF/blob/master/ADTOF/conversionDictionnaries/StandardMidiToReduced.json

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