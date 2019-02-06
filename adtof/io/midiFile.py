"""
Deprecated. It's better for the simplicity of the code to directly interface with mido instead of having a proxy such as this one.

The only advantage of the proxy would be to be able to easily switch between mido and any other midi library. but chances of that occuring avere very slim
"""
# import mido
# # import midi as pythonmidi
# import collections


# class MidiFile(collections.Iterable):
#     """
#     Class making the proxy with the available midi modules
#     See:
#     - mido (doesn't work with all the midi files)
#     - python-midi (not compatible with python3)
#     """

#     def __init__(self, path: str):
#         # TODO add midi-python capabilities ? It's only python2, but knows how to handle SYX messages
#         # try:
#         midoMidi = mido.MidiFile(path)
#         # except:
#         #     midoMidi = self.cleanMidi(path)
#         # midiMidi = pythonmidi.read_midifile(str)
#         self.format = midoMidi.type
#         self.length = midoMidi.length
#         self.tracks:List[Event] = midoMidi.tracks
#         self.ticks_per_beat = midoMidi.ticks_per_beat

#     def cleanMidi(self, str):
#         #TODO inplement that
#         #likely an issue because of SYX messages in python file
#         raise NotImplementedError()


#     def __iter__(self):
#         # return (x for x in list.__iter__(self.values) if x is not None)
#         return self.tracks.__iter__()

#     def __len__(self):
#         return len(self.tracks)

#     def __getitem__(self, i):
#         return self.tracks[i]

#     def write(self, path: str):
#         """
#         serialize this class in midi to the path location
#         """
#         midoMidi = mido.MidiFile()

#         for track in self.tracks:
#             midoTrack = mido.MidiTrack()
#             for event in track:
#                 midoTrack.append(event)
#             midoMidi.add_track(track)

#         midoMidi.save(path)

