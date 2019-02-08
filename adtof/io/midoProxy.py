import mido
import numpy as np
from mido import MidiFile


class MidoProxy(MidiFile):
    """
    Encapsulating MIDI functionalities in this proxy class to open the possibility of supporting another midi library more easily.
    """

    def getTempoEvent(self):
        """
        Return a list a tuples
        [(ticks, mpqn)]
        """
        tempoEvents = [(0, 500000)] #Start with the default microseconds/beats, which is 120 bpm
        tickCursor = 0
        if self.type != 1:
            raise NotImplementedError()

        for tempoEvent in self.tracks[0]:
            tickCursor += tempoEvent.time
            if tempoEvent.type == 'set_tempo':
                currentMpqn = tempoEvent.tempo
                if tempoEvents[-1][0] == tickCursor: #Make sure there are no overlapping tempo events
                    tempoEvents[-1] = (tickCursor, currentMpqn)
                else:
                    tempoEvents.append((tickCursor, currentMpqn))

        return tempoEvents

    def getTicksToSecond(self, ticks, tempo):
        """
        Transform a number of ticks with a tempo in microseconds per beat into seconds based on the resolution of the midi file
        """
        beat = ticks / self.ticks_per_beat
        usPerBeats = beat * tempo
        return usPerBeats / 1000000

    def getEventsPosition(self):
        """
        Compute the time in seconds of each event
        """
        tempoEvents = self.getTempoEvent()

        for track in self.tracks[1:]:
            tickCursor = 0
            timeCursor = 0

            for event in track:
                if event.time:
                    start = tickCursor
                    end = tickCursor + event.time
                    selectedTempo =  [t for t in tempoEvents if t[0] > start and t[0] < end]  

                    tempo0 = [t[1] for t in tempoEvents if t[0] <= start][0]
                    Tempi = [tempo0] + [t[1] for t in selectedTempo]
                    weight = np.diff([start] + [t[0] for t in selectedTempo] + [end]) / event.time

                    timeCursor += self.getTicksToSecond(event.time, np.sum(Tempi * weight))

                tickCursor += event.time
                #TODO find a better method than: event.time = timeCursor
                


    def getTrackNames(self):
        """
        Return the first name of each track
        """
        tracksNames = [[event.name for event in track if event.type == "track_name"] for track in self.tracks]
        return [names[0] if names else None for names in tracksNames]

class timeValue(Int)