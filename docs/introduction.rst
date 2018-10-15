Introduction
============

ADTOF is an effort to list existing format of drum transcriptions available online and utilize them as ADT training data.

Automatic drum transcription (ADT) is the problem of automatically creating a transcription with the help of algorithms.
Usualy, the softwares doing ADT need a music file as an input and produce a midi file from it.
One of the current issue in this field is the lack of training data which consist of music already annotated manualy. 

This project rised from the idea that finding a transcription from any moderately famous song is easy with a simple google search. 
The user can then access a human made transcription which is usually in text, html, or guitar pro format (`example 1`_).
The issue here is that this kind of data is not synchronized to the original source audio, making it very hard to use as training data for ADT.

The alternative is to look at video games such as Guitar Hero or Rock band where the player can perform a popular song on a controller of the shape of a guitar or drums. 
The transcriptions available for those games have the advantage of being precisely synchronized to the original audio to insure the best gameplay experience to the user (`example 2`_).
A community of players started to create their own transcriptions of their favorite bands and shared them in different format. 

.. _example 1: https://www.songsterr.com/a/wsa/gojira-lenfant-sauvage-drum-tab-s381936t5
.. _example 2: https://www.youtube.com/watch?v=26vfTMXLlV4

Limitations
~~~~~~~~~~~

The game charts are made for players on a `toy drum set`_, thus it can be an inexact representation of what is played in the track annotated. 
This game controller is similar to an electronic drum set but is meant to be versatile and only contains 8 pads loosely defined.
The general template of the correlation between the game controller and a real drum kit is:

 - Orange drum = Kick
 - Red drum = Snare
 - Yellow drum = Rack Tom 1
 - Blue drum = Rack Tom 2
 - Green drum = Floor tom
 - Yellow cymbal = Hihat
 - Blue cymbal = Ride Cymbal or Crash Cymbal
 - Green cymbal = Crash Cymbal 

This template is not necesseraly followed by the annotator as the gameplay experienced can be favorised against a more precise annotation. 
We propose to remove all the ambiguity in the annotations by mapping the precise classes to general ones (see :ref:`Usage <usage>`).


All the charts transcribed by humans don't necesseraly match the same level of quality.
In the :ref:`Datasets <datasets>` section we tried to subjectively grade the quality of the annotations, 
but we think that manually checking the tracks yielding bad precision with an ADT algorithm will have to be done.

.. _toy drum set: https://www.amazon.com/Rock-Band-Wireless-Pro-Drum-PlayStation-4/dp/B019GMR9WE

Existing work
~~~~~~~~~~~~~

Other tools made to create or convert between different games file formats already exist. 
If this project is not enough for you, you may want to look at those tools. An even more exhaustive list of tools can also be found on `customscreators.com`_.

 - `Editor On Fire`_: Editor to create songs. Can import, analyse, and export between most of the games formats. 
 - C3CONTools_: A collection of tools to create, analyse, clean and convert most of the games formats. Most of the tools can work with batches making it easy to covert multiple files together.
 - `Magma Rock Band Network tools`_: Official tool to create song packages from stems and MIDI files in Rock Band Specifications. 
 - `Magma C3 Roks Edition`_: Clone of the official tool made by independant developpers.

.. _Editor On Fire: http://ignition.customsforge.com/eof
.. _C3CONTools: http://customscreators.com/index.php?/topic/9095-c3-con-tools-v400-012518/
.. _Magma Rock Band Network tools: https://forums.harmonixmusic.com/discussion/167159/rock-band-network-tools-and-documentaion-released
.. _Magma C3 Roks Edition: http://customscreators.com/index.php?/topic/9257-magma-c3-roks-edition-v332-072815/
.. _customscreators.com: http://customscreators.com/index.php?/forum/7-authoring-tools-support-advice/