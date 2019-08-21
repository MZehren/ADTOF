# #!/usr/bin/env python

# import midi
# from adtof.io.converters import Converter


# class pythonMidiConverter(Converter):

#     def convert(self, inputPath, outputName=None):
#         """[summary]
        
#         Arguments:
#             Converter {[type]} -- [description]
#             inputPath {[type]} -- [description]
        
#         Keyword Arguments:
#             outputName {[type]} -- [description] (default: {None})
#         """
#         pattern = midi.read_midifile(inputPath)
#         for track in pattern:
#             for event in track:
#                 if event.name  == "SysEx":
#                     event.data = [1]
#                     event.statusmsg = 1
#                 if any([bla for bla in event.data if bla > 127]):
#                     print("bla")
#         midi.write_midifile(outputName, pattern)

# pythonMidiConverter().convert(
#     "E:/ADTSets/sodamlazy/lamb of god/Lamb of God - Overlord (rd1.0)/notes.mid",
#     outputName="E:/ADTSets/sodamlazy/lamb of god/Lamb of God - Overlord (rd1.0)/notes.mid")
