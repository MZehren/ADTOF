Usage
=====

There are multiple formats you could import from, each one comes from a specific game. 
In this page you will see a little summary on how to handle those formats as well as a conversion we offer in this repository.

Standard MIDI
------------

We offer to convert the most common game formats to standard midi files to maximises the compatibility with any other softwares.
The resulting file is a standard `format 1 MIDI file`_ with two tracks: 
 - The 'tempo map' containing all the tempo and time signature events.
 - The drums track containing all the drums note on events.

This MIDI file is going to be aligned to the music file distributed.


Phase Shift
-----------

Format
~~~~~~

`Phase Shift`_ is a free Guitar Hero clone. 
I did not find any documentation online about the format they use. Here's my attempt to reverse engineer their format.
The charts distributed are contained in a folder with multiple files:
 - guitar.ogg: the source track from the source. This file can sometimes be in multitracks Mogg when the track is available with separated stems, but it's very uncommon for custom tracks.
 - notes.mid: a format 1 MIDI file containing the notes to be played by the differents instruments such as the drums or the guitars. The note from the events are not standard and need to be converted with the script in this repository.
 - song.ini: a file containing some meta inforations suchs as the delay between the audio and the MIDI files.




.. _format 1 MIDI file: https://www.csie.ntu.edu.tw/~r92092/ref/midi/#mff1   
.. _Phase Shift: http://www.dwsk.co.uk/index_phase_shift.html