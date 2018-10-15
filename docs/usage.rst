Usage
=====

There are multiple formats you could import from, each one comes from a specific game. 
In this page you will see a little summary on how to handle those formats as well as a conversion we offer in this repository.

Standard MIDI
-------------

We offer to convert the most common game formats to standard midi files to maximises the compatibility with any other softwares.
The resulting file is a standard `format 1 MIDI file`_ with two tracks:

 - The 'tempo map' containing all the tempo and time signature events.
 - The drums track containing all the drums note on events.

This MIDI file is going to be aligned to the music file distributed.


Phase Shift
-----------

Format
~~~~~~

`Phase Shift`_ (abreviated PS) is a free Guitar Hero clone. 
The charts distributed for this game are contained in a folder with multiple files:

 - **guitar.ogg**: the audio from the source track. This file can sometimes be in multitracks Mogg when separated stems are availabe, which is not common for custom tracks.
 - **notes.mid**: a format 1 MIDI file containing the notes to be played by the differents instruments such as the drums or the guitars. The note from the events are not standard and need to be converted with the script in this repository. The MIDI specifications of this format seems to be the same as Rock Band's format, which is described on c3universe.com_.
 - **song.ini**: a file containing some meta informations suchs as the delay between the audio and the MIDI files.

Conversion
~~~~~~~~~~

To convert a Phase Shift MIDI file to a standard one, run the script:

>>> cd ADTOF
>>> python PhaseShiftConverter.py [pathToPhaseShiftChartFolder]

This script is going to convert the Phase Shift midi file to a "note_std.mid" file by applying multiple corrections:

 - Only the tracks containing drums event are kept and all the other instruments are discarded.
 - Phase Shift format handles toms notes with a modifier applied to a cymbal events. We convert the events with modifiers to the real ones.
 - The Phase Shift pitches are converted to standard pitches following the dictionnary PhaseShiftMidiToStandard.json_. You can see the list of standard pitches on Wikipedia_.
 - The ambiguous midi classes are reduced to the general ones following StandardMidiToReduced.json_. Feel free to change this dictionnary to fit your needs.

.. _PhaseShiftMidiToStandard.json: https://github.com/MZehren/ADTOF/blob/master/ADTOF/conversionDictionnaries/PhaseShiftMidiToStandard.json
.. _StandardMidiToReduced.json: https://github.com/MZehren/ADTOF/blob/master/ADTOF/conversionDictionnaries/StandardMidiToReduced.json

Rock Band
-----------

Format
~~~~~~

Rock Band is a serie of video games released on multiples consoles such as the PlayStation 3 or the Xbox 360.
The format used is either in **con** or **rb3con** which are binary files encapsulating the audio file as well as midi events. 

Conversion
~~~~~~~~~~

The tool C3CONTools_ can be used to convert Rock Band format to Phase Shift in batches. 
Phase Shift format can then be converted to a standard midi file with the method described above



.. _format 1 MIDI file: https://www.csie.ntu.edu.tw/~r92092/ref/midi/#mff1   
.. _Phase Shift: http://www.dwsk.co.uk/index_phase_shift.html
.. _Wikipedia: https://en.wikipedia.org/wiki/General_MIDI#Percussive
.. _C3CONTools: http://customscreators.com/index.php?/topic/9095-c3-con-tools-v400-012518/
.. _c3universe.com: http://docs.c3universe.com/rbndocs/index.php?title=Drum_Authoring