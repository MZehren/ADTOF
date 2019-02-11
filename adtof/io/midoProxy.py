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

    def getTicksToSecond(self, ticks, start=0, tempo=0):
        """
        Transform a number of ticks with a tempo in microseconds per beat into seconds based on the resolution of the midi file

        if no tempo is provided, the track's tempo is going to be used
        It's possible to specify a start value if the event is not occuring at the m
        """
        end = ticks
        delta = end - start
        if not tempo:  #get all the tempo changes occuring between the start and end locations. The resulting tempo is the weighted average
            tempoEvents = self.tempoEvents
            selectedTempo = [t for t in tempoEvents if t[0] > start and t[0] < end]  #get the tempo changes during the event
            tempo0 = [t[1] for t in tempoEvents if t[0] <= start][0]  #get the tempo at the start location
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

    def getOnsets(self):
        """
        Return a list a positions in seconds of the notes_on events
        """
        notesPositions = []
        for i, track in enumerate(self.tracks):
            for j, event in enumerate(track):
                if event.time:
                    if event.type == "note_on" and event.velocity > 0:
                        notesPositions.append(self.positionsLookup[i][j]["timeAbsolute"])

        return notesPositions

    def addDelay(self, delta):
        """
        Increment the position of the midi events by a delay
        TODO check if negative value works
        """
        for i, track in enumerate(self.tracks):
            track[0].time += self.getSecondToTicks(delta, tempo=self.tempoEvents[0][1])

    def getTrackNames(self):
        """
        Return the first name event of each track
        """
        tracksNames = [[event.name for event in track if event.type == "track_name"] for track in self.tracks]
        return [names[0] if names else None for names in tracksNames]
