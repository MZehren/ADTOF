Introduction
============

ADTOF is an effort to list existing format of drum transcriptions available online and utilize them as ADT training data.

Automatic drum transcription (ADT) is the problem of automatically creating a transcription with the help of algorithms.
Usualy, the softwares doing ADT need a music file as an input and produce a midi file from it.
One of the current issue in this field is the lack of training data which consist of music already annotated manualy. 

This project rised from the idea that finding a transcription from any moderately famous song is easy with a simple google search. 
The user can then access a human made transcription which is usually in text, html, or guitar pro format such as https://www.songsterr.com/a/wsa/gojira-lenfant-sauvage-drum-tab-s381936t5.
The issue here is that this kind of data is not synchronized to the original source audio, making it very hard to use as training data for ADT.

The alternative is to look at video games such as Guitar Hero or Rock band where the player can perform a popular song on a controller of the shape of a guitar or drums. 
The transcriptions available for those games have the advantage of being precisely synchronized to the original audio to insure the best gameplay experience to the user such as https://www.youtube.com/watch?v=26vfTMXLlV4.
A community of players started to create their own transcriptions of their favorite bands and shared them in different format. 

Existing work
~~~~~~~~~~~~~

Other tools made to create or convert between different games file formats already exist. 
If this project is not enough for you, you may want to look at those tools:

 - `Editor On Fire`_: Editor to create songs. Can import, analyse, and export between most of the games formats. 
 - C3CONTools_: A collection of tools to create, analyse, clean and convert most of the games formats. Most of the tools can work with batches of files.
 - `Magma C3 Roks Edition`_: bla
 - `Moonscraper Chart File Creator`_: bla

.. _Editor On Fire: http://ignition.customsforge.com/eof
.. _CCONTools: http://customscreators.com/index.php?/topic/9095-c3-con-tools-v400-012518/
.. _Magma C3 Roks Edition: http://customscreators.com/index.php?/topic/9257-magma-c3-roks-edition-v332-072815/