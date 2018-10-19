Introduction
============

ADTOF is an effort to list existing format of drum transcription available online and utilize them as ADT training data by converting them to a standard MIDI file aligned to an audio file.

Automatic drum transcription (ADT) is the problem of automatically creating a transcription with the help of algorithms.
Usually, softwares doing ADT need a music file as an input and produce a midi file from it.
One of the current issues in this field is the lack of training data which consist of music already annotated manually. 

This project rose from the idea that finding a transcription from any moderately famous song is easy with a simple google search. 
Online services offer crowdsourced human-made transcriptions which qre usually in text, HTML, or guitar pro formats (`example 1`_).
The issue here is that this kind of annotations are not synchronized to the source audio, making them very hard to use as training data for ADT.

The alternative is to look at video games such as Guitar Hero, Phase Shift or Rock Band where the player can perform a popular song on a controller of the shape of a guitar or drums. 
The transcriptions available for those games have the advantage of being precisely synchronized to the original audio to ensure the best gameplay experience to the user (`example 2`_).
A community of players started to create their own transcriptions of their favourite music and shared them online. 

.. _example 1: https://www.songsterr.com/a/wsa/gojira-lenfant-sauvage-drum-tab-s381936t5
.. _example 2: https://www.youtube.com/watch?v=26vfTMXLlV4

Limitations
~~~~~~~~~~~

The game charts are made for players on a `toy drum set`_, thus it can be an inexact representation of what is played in the track annotated. 
This game controller is similar to an electronic drum set but is meant to be versatile and only contains 8 pads which can represent different drums.
The general template of the correlation between the game controller and a real drum kit is:

 - Orange drum = Kick
 - Red drum = Snare
 - Yellow drum = Rack Tom 1
 - Blue drum = Rack Tom 2
 - Green drum = Floor tom
 - Yellow cymbal = Hihat
 - Blue cymbal = Ride Cymbal or Crash Cymbal
 - Green cymbal = Crash Cymbal 

This template is not necessarily followed by the annotator as the gameplay experienced can be favoured against a more precise annotation. 
To overcome this problem, we propose to remove all the ambiguity in the annotations by mapping the precise classes to general ones (see :ref:`Usage <usage>`).


All the charts transcribed by humans don't necessarily match the same level of quality.
In the :ref:`Datasets <datasets>` section, we tried to subjectively grade the quality of the annotations, 
but we think that manually checking the tracks yielding bad precision with an ADT algorithm will have to be done.

.. _toy drum set: https://www.amazon.com/Rock-Band-Wireless-Pro-Drum-PlayStation-4/dp/B019GMR9WE

Existing work
~~~~~~~~~~~~~

Other tools made to create or convert between different games file formats already exist. 
If this project is not enough for you, you may want to look at those tools. Another list of tools can also be found on `customscreators.com`_.
To our knowledge, no tool offers to convert a game format to a standard midi format.

 - `Editor On Fire`_: Editor to create songs. Can import, analyse, and export between most of the games formats. 
 - C3CONTools_: A collection of tools to create, analyse, clean and convert most of the games formats. Most of the tools can work in batches making it easy to convert multiple files at the same time.
 - `Magma Rock Band Network tools`_: Official tool to create song packages from stems and MIDI files in Rock Band Specifications. 
 - `Magma C3 Roks Edition`_: Clone of the official tool made by independant developpers.

.. _Editor On Fire: http://ignition.customsforge.com/eof
.. _C3CONTools: http://customscreators.com/index.php?/topic/9095-c3-con-tools-v400-012518/
.. _Magma Rock Band Network tools: https://forums.harmonixmusic.com/discussion/167159/rock-band-network-tools-and-documentaion-released
.. _Magma C3 Roks Edition: http://customscreators.com/index.php?/topic/9257-magma-c3-roks-edition-v332-072815/
.. _customscreators.com: http://customscreators.com/index.php?/forum/7-authoring-tools-support-advice/
