"""
Identify what is the smallest time interval between notes in our dataset 
"""
import os
import pretty_midi
import numpy as np

TracksLocations = "/Users/mzehren/Programming/adtofParsed/midi_converted"
midis = [os.path.join(TracksLocations, path) for path in os.listdir(TracksLocations)]
midis.sort()

# 20 Hz == 1/20s = 50 ms = 300*4 bpm
# The violation = 134*8 bpm  = 270 * 4 bpm
def getMinimumPitchInterval(notes, pitch, minThreshold=0.050):
    notes = [note for note in notes if note.pitch == pitch]
    onsets = [note.start for note in notes]
    diff = [value for value in np.diff(onsets) if value != 0 and value > minThreshold]
    minimum = np.min(diff) if len(diff) else np.nan
    return minimum


minimums = []
for file in midis[:100]:
    midi = pretty_midi.PrettyMIDI(file)
    instrument = midi.instruments[0]
    minimum = np.nanmin([getMinimumPitchInterval(instrument.notes, pitch) for pitch in [36, 42]])
    print(file, minimum)
    minimums.append(minimum)
print(np.min(minimums))
