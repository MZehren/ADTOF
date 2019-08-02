from collections import defaultdict

import mido
import numpy as np
from mido import MidiFile


def lazy_property(fn):
    '''Decorator that makes a property lazy-evaluated.
    TODO change the location ?
    '''
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


class MidoProxy(MidiFile):
    """
    Encapsulating MIDI functionalities in this proxy class to open the possibility of supporting another midi library more easily.
    """

    @lazy_property
    def tempoEvents(self):
        """
        Return a list a tuples
        [(ticks, mpqn)]
        """
        tempoEvents = [(0, 500000)]  #Start with the default microseconds/beats, which is 120 bpm
        tickCursor = 0
        if self.type != 1:
            raise NotImplementedError()

        for tempoEvent in self.tracks[0]:
            tickCursor += tempoEvent.time
            if tempoEvent.type == 'set_tempo':
                currentMpqn = tempoEvent.tempo
                if tempoEvents[-1][0] == tickCursor:  #Make sure there are no overlapping tempo events
                    tempoEvents[-1] = (tickCursor, currentMpqn)
                else:
                    tempoEvents.append((tickCursor, currentMpqn))

        return tempoEvents

    @lazy_property
    def positionsLookup(self):
        """
        Compute the time in seconds of each event

        store that in a 3D list of [track, event, property]
        """
        positionsLookup = []
        for i, track in enumerate(self.tracks):
            positionsLookup.append([])
            tickCursor = 0
            timeCursor = 0
            for j, event in enumerate(track):
                timeIncrement = 0
                if event.time:
                    timeIncrement = self.getTicksToSecond(tickCursor + event.time, start=tickCursor)
                    timeCursor += timeIncrement
                    tickCursor += event.time

                positionsLookup[i].append({"tickAbsolute": tickCursor, "timeAbsolute": timeCursor, "tick": event.time, "time": timeCursor})

        return positionsLookup

    def getTicksToSecond(self, ticks, start=0, tempo=None):
        """
        Transform a number of ticks with a tempo in microseconds per beat into seconds based on the resolution of the midi file

        if no tempo is provided, the track's tempo is going to be used
        It's possible to specify a start value if the event is not occuring from the start
        """
        end = ticks
        delta = end - start
        if tempo is None:  #get all the tempo changes occuring between the start and end locations. The resulting tempo is the weighted average
            tempoEvents = self.tempoEvents
            selectedTempo = [t for t in tempoEvents if t[0] > start and t[0] < end]  #get the tempo changes during the event
            tempo0 = [t[1] for t in tempoEvents if t[0] <= start][-1]  #get the tempo at the start location
            Tempi = [tempo0] + [t[1] for t in selectedTempo]
            weight = np.diff([start] + [t[0] for t in selectedTempo] + [end]) / delta  #get the weighted average of all the tempi
            tempo = np.sum(Tempi * weight)

        beat = delta / self.ticks_per_beat  #convert thegetraiseticks in beat and then seconds
        usPerBeats = beat * tempo
        return usPerBeats / 1000000

    def getSecondToTicks(self, second, start=0, tempo=0):
        """
        Compute the numbs a delta  er of ticks equal to the duration in second in function of the MIDI resolution

        If no tempo is provided, uses track's tempo.
        If start in ticks is provided, the value take into account the tempo 
        """
        if not tempo:
            raise NotImplementedError()
            # tempoEvents = self.tempoEvents
            # tempoDuration = [self.getTicksToSecond(tempoEvent[0], tempo=tempoEvent[1]) for tempoEvent in tempoEvents]
            # weight =
        return int(float(second) / (float(tempo) / 1000000) * float(self.ticks_per_beat))

    def getOnsets(self, separated=False):
        """
        Return a list a positions in seconds of the notes_on events
        """
        allNotesPositions = []
        notesPositions = defaultdict(list)

        for i, track in enumerate(self.tracks):
            for j, event in enumerate(track):
                if event.type == "note_on" and event.velocity > 0:
                    notesPositions[event.note].append(self.positionsLookup[i][j]["timeAbsolute"])
                    if event.time:
                        allNotesPositions.append(self.positionsLookup[i][j]["timeAbsolute"])

        if separated:
            return notesPositions
        else:
            return allNotesPositions

    def addDelay(self, delta):
        """
        Increment the position of the midi events by a delay
        """
        for track in self.tracks:
            increment = self.getSecondToTicks(delta, tempo=self.tempoEvents[0][1])
            for event in track:
                event.time += increment
                if event.time < 0:
                    increment = event.time
                    event.time = 0
                else:
                    break

    def getTrackNames(self):
        """
        Return the first name event of each track
        """
        tracksNames = [[event.name for event in track if event.type == "track_name"] for track in self.tracks]
        return [names[0] if names else None for names in tracksNames]

    def getDenseEncoding(self, sampleRate=100, timeShift=0):
        """
        Encode in a dense matrix
        from [0.1, 0.5]
        to [0, 1, 0, 0, 0, 1]
        """
        notes = self.getOnsets(separated=True)
        result = []
        for key in notes.keys():
            row = np.zeros(int(np.round((self.length + timeShift) * sampleRate))+1)
            for time in notes[key]:
                row[int(np.round((time + timeShift) * sampleRate))] = 1
            result.append(row)

        return np.array(result).T

    @staticmethod
    def fromDenseEncoding(sampleRate, timeShift=0):
        raise NotImplementedError()
      
