import logging
import warnings
from collections import defaultdict

import numpy as np
import midi
from mido import MidiFile
import mido


def lazy_property(fn):
    '''
    Decorator that makes a property lazy-evaluated.
    TODO change the location ?
    '''
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


# class MidiProxy(object):

#     def _getProxy(self):
#         try:
#             import midi
#             return PythonMidiProxy
#         except ImportError:
#             logging.info("python-midi import error, trying mido")

#         try:
#             from mido import MidiFile
#             import mido
#             return MidoProxy
#         except ImportError:
#             logging.info("python-midi import error, trying mido")


class MidoProxy(MidiFile):
    """
    Encapsulating MIDI functionalities in this proxy class to open the possibility of supporting another midi library more easily.
    """

    def getTracksName(self):
        tracksName = [[event.name for event in track if event.type == midi.TRACK_NAME_EVENT] for track in self.tracks]
        tracksName = [names[0] if names else None for names in tracksName]
        return tracksName

    def addDelay(self, delay):
        """Add delay at the start of the midi 
        
        Arguments:
            delay {float} -- delay in second 
        """
        for i, track in enumerate(self.tracks):
            # add the delay
            if delay != 0:
                if i == 0:
                    # If this is the tempo track, add a set tempo meta event event such as the delay is a 4 beats
                    event = mido.MetaMessage("set_tempo")
                    event.tempo = int(delay * 1000000 / 4)
                    track.insert(0, event)
                    track[1].time += 4 * midi.ticks_per_beat
                else:
                    # If this is a standard track, add a delay of 4 beats to the first event
                    track[0].time += 4 * midi.ticks_per_beat

    def getEventTick(self, event):
        return event.time

    def setEventTick(self, event, tick):
        event.time = tick

    def isEventNoteOn(self, event):
        return event.type == "note_on" and event.velocity > 0

    def isEventNoteOff(self, event):
        return (event.type == "note_on" and event.velocity == 0) or event.type == "note_off"

    def getEventPith(self, event):
        try:
            return event.note
        except:
            return None

    def setEventPitch(self, event, pitch):
        event.note = pitch

    @lazy_property
    def tempoEvents(self):
        """
        Return a list a tuples
        [(ticks, mpqn)]
        """
        tempoEvents = [(0, 500000)]  # Start with the default microseconds/beats, which is 120 bpm
        tickCursor = 0
        if self.type != 1:
            raise NotImplementedError()

        for tempoEvent in self.tracks[0]:
            tickCursor += tempoEvent.time
            if tempoEvent.type == 'set_tempo':
                currentMpqn = tempoEvent.tempo
                # Make sure there are no overlapping tempo events
                if tempoEvents[-1][0] == tickCursor:
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

                positionsLookup[i].append({
                    "tickAbsolute": tickCursor,
                    "timeAbsolute": timeCursor,
                    "tick": event.time,
                    "time": timeCursor
                })

        return positionsLookup

    def getTicksToSecond(self, ticks, start=0, tempo=None):
        """
        Transform a number of ticks with a tempo in microseconds per beat into seconds based on the resolution of the midi file

        if no tempo is provided, the track's tempo is going to be used
        It's possible to specify a start value if the event is not occuring from the start
        """
        end = ticks
        delta = end - start
        if tempo is None:  # get all the tempo changes occuring between the start and end locations. The resulting tempo is the weighted average
            tempoEvents = self.tempoEvents
            selectedTempo = [t for t in tempoEvents if t[0] > start and t[0] < end
                            ]  # get the tempo changes during the event
            # get the tempo at the start location
            tempo0 = [t[1] for t in tempoEvents if t[0] <= start][-1]
            Tempi = [tempo0] + [t[1] for t in selectedTempo]
            weight = np.diff([start] + [t[0] for t in selectedTempo] +
                             [end]) / delta  # get the weighted average of all the tempi
            tempo = np.sum(Tempi * weight)

        # convert thegetraiseticks in beat and then seconds
        beat = delta / self.ticks_per_beat
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

    # def addDelay(self, delta):
    #     """
    #     Increment the position of the midi events by a delay
    #     """
    #     for track in self.tracks:
    #         increment = self.getSecondToTicks(
    #             delta, tempo=self.tempoEvents[0][1])
    #         for event in track:
    #             event.time += increment
    #             if event.time < 0:
    #                 increment = event.time
    #                 event.time = 0
    #             else:
    #                 break

    def getTrackNames(self):
        """
        Return the first name event of each track
        """
        tracksNames = [[event.name for event in track if event.type == "track_name"] for track in self.tracks]
        return [names[0] if names else None for names in tracksNames]

    def getDenseEncoding(self, sampleRate=100, timeShift=0, keys=[36, 40, 41, 46, 49]):
        """
        Encode in a dense matrix
        from [0.1, 0.5]
        to [0, 1, 0, 0, 0, 1]
        """
        notes = self.getOnsets(separated=True)
        result = []
        for key in keys:
            row = np.zeros(int(np.round((self.length + timeShift) * sampleRate)) + 1)
            for time in notes[key]:
                row[int(np.round((time + timeShift) * sampleRate))] = 1
            result.append(row)
            if len(notes[key]) == 0:
                logging.info(self.filename + " " + str(key) + " is not represented in this track")

        return np.array(result).T

    @staticmethod
    def fromDenseEncoding(sampleRate, timeShift=0):
        raise NotImplementedError()


class PythonMidiProxy():
    """
    Encapsulating MIDI functionalities in this proxy class to open the possibility of supporting another midi library more easily.
    """

    def __init__(self, path):
        self.tracks = midi.read_midifile(path)
        self.type = self.tracks.format
        self.filename = path

    def getTracksName(self):
        """Return the name of the tracks
        
        Returns:
            list -- name of each track
        """
        tracksName = [[event.text for event in track if event.name == "Track Name"] for track in self.tracks]
        tracksName = [names[0] if names else None for names in tracksName]
        return tracksName

    def addDelay(self, delay):
        """Add delay at the start of the midi 
        
        Arguments:
            delay {float} -- delay in second 
        """
        for i, track in enumerate(self.tracks):
            # add the delay
            if delay != 0:
                if i == 0:
                    # If this is the tempo track, add a set tempo meta event event such as the delay is a 4 beats
                    event = midi.SetTempoEvent()
                    event.mpqn = int(delay * 1000000 / 4)
                    track.insert(0, event)
                    track[1].tick += 4 * self.tracks.resolution
                else:
                    # If this is a standard track, add a delay of 4 beats to the first event
                    track[0].tick += 4 * self.tracks.resolution

    def getEventTick(self, event):
        return event.tick

    def setEventTick(self, event, tick):
        event.tick = tick

    def isEventNoteOn(self, event):
        return event.name == "Note On" and event.velocity > 0

    def isEventNoteOff(self, event):
        return (event.name == "Note On" and event.velocity == 0) or event.name == "Note Off"

    def getEventPith(self, event):
        try:
            return event.pitch
        except:
            return None

    def setEventPitch(self, event, pitch):
        event.pitch = pitch

    def save(self, path):
        midi.write_midifile(path, self.tracks)

    def tempoEvents(self):
        """
        Return a list a tuples
        [(ticks, mpqn)]
        """
        tempoEvents = [(0, 500000)]  # Start with the default microseconds/beats, which is 120 bpm
        tickCursor = 0
        if self.tracks.format != 1:
            raise NotImplementedError()

        for tempoEvent in self.tracks[0]:
            tickCursor += tempoEvent.tick
            if tempoEvent.name == 'Set Tempo':
                currentMpqn = tempoEvent.mpqn
                # Make sure there are no overlapping tempo events
                if tempoEvents[-1][0] == tickCursor:
                    tempoEvents[-1] = (tickCursor, currentMpqn)
                else:
                    tempoEvents.append((tickCursor, currentMpqn))

        return tempoEvents

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
                if event.tick:
                    timeIncrement = self.getTicksToSecond(tickCursor + event.tick, start=tickCursor)
                    timeCursor += timeIncrement
                    tickCursor += event.tick

                positionsLookup[i].append({
                    "tickAbsolute": tickCursor,
                    "timeAbsolute": timeCursor,
                    "tick": event.tick,
                    "time": timeCursor
                })

        return positionsLookup

    def getTicksToSecond(self, ticks, start=0, tempo=None):
        """
        Transform a number of ticks with a tempo in microseconds per beat into seconds based on the resolution of the midi file

        if no tempo is provided, the track's tempo is going to be used
        It's possible to specify a start value if the event is not occuring from the start
        """
        end = ticks
        delta = end - start
        if tempo is None:  # get all the tempo changes occuring between the start and end locations. The resulting tempo is the weighted average
            tempoEvents = self.tempoEvents()
            # get the tempo changes during the event
            selectedTempo = [t for t in tempoEvents if t[0] > start and t[0] < end]
            # get the tempo at the start location
            tempo0 = [t[1] for t in tempoEvents if t[0] <= start][-1]
            Tempi = [tempo0] + [t[1] for t in selectedTempo]
            weight = np.diff([start] + [t[0] for t in selectedTempo] +
                             [end]) / delta  # get the weighted average of all the tempi
            tempo = np.sum(Tempi * weight)

        # convert thegetraiseticks in beat and then seconds
        beat = delta / self.tracks.resolution
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
        return int(float(second) / (float(tempo) / 1000000) * float(self.tracks.resolution))

    def getOnsets(self, separated=False):
        """
        Return a list a positions in seconds of the notes_on events
        """
        allNotesPositions = []
        notesPositions = defaultdict(list)
        positionsLookup = self.positionsLookup()

        for i, track in enumerate(self.tracks):
            for j, event in enumerate(track):
                if self.isEventNoteOn(event):
                    notesPositions[event.pitch].append(positionsLookup[i][j]["timeAbsolute"])
                    if event.tick:
                        allNotesPositions.append(positionsLookup[i][j]["timeAbsolute"])

        if separated:
            return notesPositions
        else:
            return allNotesPositions

    def getTrackNames(self):
        """
        Return the first name event of each track
        """
        tracksNames = [[event.name for event in track if event.type == "track_name"] for track in self.tracks]
        return [names[0] if names else None for names in tracksNames]

    def getDenseEncoding(self, sampleRate=100, timeShift=0, keys=[36, 40, 41, 46, 49], radiation=2):
        """
        Encode in a dense matrix
        from [0.1, 0.5]
        to [0, 1, 0, 0, 0, 1]
        """
        notes = self.getOnsets(separated=True)
        length = np.max([values[-1] for values in notes.values()])
        result = []
        for key in keys:
            row = np.zeros(int(np.round((length + timeShift) * sampleRate)) + 1)
            for time in notes[key]:
                index = int(np.round((time + timeShift) * sampleRate))
                for i in range(index - radiation, min(index + radiation + 1, len(row))):
                    row[i] = 1
            result.append(row)
            if len(notes[key]) == 0:
                logging.info(self.filename + " " + str(key) + " is not represented in this track")

        return np.array(result).T

    @staticmethod
    def fromDenseEncoding(sampleRate, timeShift=0):
        raise NotImplementedError()


MidiProxy = PythonMidiProxy
