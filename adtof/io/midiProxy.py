import logging
import re
import warnings
from collections import defaultdict

import midi
import mido
import numpy as np
import pretty_midi
from mido import MidiFile

# TODO Remove those two implementations and use only one class with pretty midi


def lazy_property(fn):
    """
    Decorator that makes a property lazy-evaluated.
    TODO change the location ?
    """
    attr_name = "_lazy_" + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


class MidoProxy(MidiFile):
    """
    TODO: DEPRECATED
    Encapsulating MIDI functionalities in this proxy class to open the possibility of supporting another midi library more easily.
    """

    def getBeats(self):
        """
        Returns the beat number and time of the track
        """
        # tpb = self.ticks_per_beat
        # timeEvents = (self.tempoEvents + self.timeSignatureEvents)
        # timeEvents.sort(key = lambda y: y[0])

        raise NotImplementedError()

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
    def timeSignatureEvents(self):
        """
        Return a list a tuples
        [(ticks, numerator, denumerator)]
        """
        events = [(0, mido.MetaMessage("time_signature"))]
        tickCursor = 0
        if self.type != 1:
            raise NotImplementedError()

        for event in self.tracks[0]:
            tickCursor += event.time
            if event.type == "time_signature":
                # Make sure there are no overlapping tempo events
                if events[-1][0] == tickCursor:
                    events[-1] = (tickCursor, event)
                else:
                    events.append((tickCursor, event))
        return events

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
            if tempoEvent.type == "set_tempo":
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
        if (
            tempo is None
        ):  # get all the tempo changes occuring between the start and end locations. The resulting tempo is the weighted average
            tempoEvents = self.tempoEvents
            selectedTempo = [t for t in tempoEvents if t[0] > start and t[0] < end]  # get the tempo changes during the event
            # get the tempo at the start location
            tempo0 = [t[1] for t in tempoEvents if t[0] <= start][-1]
            Tempi = [tempo0] + [t[1] for t in selectedTempo]
            weight = np.diff([start] + [t[0] for t in selectedTempo] + [end]) / delta  # get the weighted average of all the tempi
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


class PythonMidiProxy:
    """
    Encapsulating MIDI functionalities in this proxy class to open the possibility of supporting another midi library more easily.
    """

    def __init__(self, path):
        if path is not None:
            self.tracks = midi.read_midifile(path)
            self.type = self.tracks.format
            self.filename = path
        else:
            self.tracks = midi.Pattern()
            track = midi.Track()
            self.tracks.append(track)
            self.filename = ""

    def addNote(self, deltaTime, pitch, duration=0, velocity=90, trackIndex=-1):
        """
        add a note in the specified track
        the time is in absolute second
        """
        track = self.tracks[trackIndex]

        tickOn = self.getSecondToTicks(deltaTime)
        if len(track):
            tickOn -= np.sum([n.tick for n in track])
        tickOff = 0
        if duration:  # TODO
            raise NotImplementedError()

        on = midi.NoteOnEvent(tick=tickOn, velocity=velocity, pitch=pitch)
        off = midi.NoteOnEvent(tick=tickOff, pitch=pitch)

        track.append(on)
        track.append(off)

    def getTracksName(self):
        """Return the name of the tracks
        
        Returns:
            list -- name of each track
        """
        tracksName = [[event.text for event in track if event.name == "Track Name"] for track in self.tracks]
        tracksName = [names if names else None for names in tracksName]
        return tracksName

    def addDelay(self, delay):
        """Add delay at the start of the midi 
        
        Arguments:
            delay {float} -- delay in second 
        """
        # add the delay
        if delay > 0:
            for i, track in enumerate(self.tracks):
                if i == 0:
                    # If this is the tempo track, add a set tempo meta event event such as the delay is a 4 beats
                    event = midi.SetTempoEvent()
                    event.mpqn = int(delay * 1000000 / 4)
                    track.insert(0, event)
                    track[1].tick += 4 * self.tracks.resolution
                else:
                    # If this is a standard track, add a delay of 4 beats to the first event
                    track[0].tick += 4 * self.tracks.resolution
        elif delay < 0:
            # reduce the position of the events until the delay is consummed
            warnings.warn("Track with negatie delay, the corrected midi has now misaligned beats")
            ticksToRemove = self.getSecondToTicks(-delay)
            for i, track in enumerate(self.tracks):
                decrement = ticksToRemove
                for event in track:
                    if event.tick >= decrement:
                        event.tick -= decrement
                        break
                    else:
                        decrement -= event.tick
                        event.tick = 0

    def getEventTick(self, event):
        return event.tick

    def setEventTick(self, event, tick):
        event.tick = tick

    def isEventNoteOn(self, event):
        return event.name == "Note On" and event.velocity > 0

    def isEventNoteOff(self, event):
        return (event.name == "Note On" and event.velocity == 0) or event.name == "Note Off"

    def getEventPith(self, event):
        return event.pitch if hasattr(event, "pitch") else None

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
            if tempoEvent.name == "Set Tempo":
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

                positionsLookup[i].append({"tickAbsolute": tickCursor, "timeAbsolute": timeCursor, "tick": event.tick, "time": timeCursor})

        return positionsLookup

    def getTicksToSecond(self, ticks, start=0, tempo=None):
        """
        Transform a number of ticks with a tempo in microseconds per beat into seconds based on the resolution of the midi file

        if no tempo is provided, the track's tempo is going to be used
        It's possible to specify a start value if the event is not occuring from the start
        """
        end = ticks
        delta = end - start
        if (
            tempo is None
        ):  # get all the tempo changes occuring between the start and end locations. The resulting tempo is the weighted average
            tempoEvents = self.tempoEvents()
            # get the tempo changes during the event
            selectedTempo = [t for t in tempoEvents if t[0] > start and t[0] < end]
            # get the tempo at the start location
            tempo0 = [t[1] for t in tempoEvents if t[0] <= start][-1]
            Tempi = [tempo0] + [t[1] for t in selectedTempo]
            weight = np.diff([start] + [t[0] for t in selectedTempo] + [end]) / delta  # get the weighted average of all the tempi
            tempo = np.sum(Tempi * weight)

        # convert thegetraiseticks in beat and then seconds
        beat = delta / self.tracks.resolution
        usPerBeats = beat * tempo
        return usPerBeats / 1000000

    def getSecondToTicks(self, second, start=0, tempo=None):
        """
        Compute the numbs a delta  er of ticks equal to the duration in second in function of the MIDI resolution

        If no tempo is provided, uses track's tempo.
        If start in ticks is provided, the value take into account the tempo 
        """
        if tempo is None:  # TODO: test if it's working
            tempo = self.tempoEvents()[0][1]
            warnings.warn("naive tempo implementation: get secondToTicks doesn't take into account changes in tempo")
            # tempoEvents = self.tempoEvents()

            # # get the tempo changes during the event
            # selectedTempo = [t for t in tempoEvents if t[0] > start and t[0] < end]
            # # get the tempo at the start location
            # tempo0 = [t[1] for t in tempoEvents if t[0] <= start][-1]
            # Tempi = [tempo0] + [t[1] for t in selectedTempo]
            # weight = np.diff([start] + [t[0] for t in selectedTempo] +
            #                  [end]) / delta  # get the weighted average of all the tempi
            # tempo = np.sum(Tempi * weight)

        return int(second / (tempo / 1000000) * self.tracks.resolution)

    def getOnsets(self, separated=False):
        """
        Return a list a positions in seconds of the notes_on events

        - separated (default False): return a dict for the position of each notes
        """
        allNotesPositions = set([])
        notesPositions = defaultdict(list)
        positionsLookup = self.positionsLookup()

        for i, track in enumerate(self.tracks):
            for j, event in enumerate(track):
                if self.isEventNoteOn(event):
                    notesPositions[event.pitch].append(positionsLookup[i][j]["timeAbsolute"])
                    allNotesPositions.add(positionsLookup[i][j]["timeAbsolute"])

        if separated:
            return notesPositions
        else:
            return allNotesPositions

    def getBeats(self):
        """
        Returns the beat number and time of the track
        """
        raise NotImplementedError()

    def getTrackNames(self):
        """
        Return the first name event of each track
        """
        tracksNames = [[event.name for event in track if event.type == "track_name"] for track in self.tracks]
        return [names[0] if names else None for names in tracksNames]

    @staticmethod
    def fromDenseEncoding(sampleRate, timeShift=0):
        raise NotImplementedError()


class PrettyMidiWrapper(pretty_midi.PrettyMIDI):
    @classmethod
    def fromListOfNotes(cls, notes, beats=[]):
        """Instantiate a pretty_midi.PrettyMIDI class with the list of notes. 
        TODO: If the beats are provided, set tempo events as well

        Parameters
        ----------
        notes : [(position, pitch)]
            list of tuples with a position and pitch
        beats : [(position, beatNumber)], optional
            List of position and beat number in the bar (ie. 1 to 4), by default []

        Returns
        -------
        [pretty_midi.PrettyMIDI]
            the midi object which can be serialised
        """
        midi = cls()
        instrument = pretty_midi.Instrument(program=1, is_drum=True)
        midi.instruments.append(instrument)
        for time, pitch in notes:
            note = pretty_midi.Note(velocity=100, pitch=pitch, start=time, end=time)
            instrument.notes.append(note)

        for time, beatIdx in beats:
            note = pretty_midi.Note(velocity=100, pitch=beatIdx, start=time, end=time)
            instrument.notes.append(note)

        return midi

    def get_beats_with_index(self, ingoreStart=False):
        """call get_beats and get_downbeats to return the list of beats and the list of beats index

        Returns
        -------
        (list, list)
            tuple with list of beats position and list of beats index
        """
        beats = self.get_beats()
        downbeats = set(self.get_downbeats())
        beatCursor = -1
        beatIdx = []
        for beat in beats:
            if beat in downbeats:
                beatCursor = 1
            beatIdx.append(beatCursor)
            beatCursor = beatCursor + 1 if beatCursor != -1 else -1

        if ingoreStart:
            firstNote = self.get_onsets()[0]
        return beats, beatIdx

    def _load_metadata(self, midi_data):
        """
        Call base class load_meatadata and add text events as well

        See: http://docs.c3universe.com/rbndocs/index.php?title=Drum_Authoring#Pro_Drum_and_Disco_Flip
        """
        from pretty_midi.containers import Note

        super()._load_metadata(midi_data)
        self.discoFlip = []
        for track in midi_data.tracks:
            flipStart = None
            for event in track:
                if event.type == "text":
                    if re.search("\[mix 3 drums[0-3]d\]", event.text) is not None:
                        assert flipStart == None
                        flipStart = self._PrettyMIDI__tick_to_time[event.time]
                    elif (
                        re.search("\[mix 3 drums[0-3]dnoflip\]", event.text) is not None
                        or re.search("\[mix 3 drums[0-3]\]", event.text) is not None
                    ):
                        # assert flipStart != None
                        if flipStart != None:
                            self.discoFlip.append(Note(0, "disco", flipStart, self._PrettyMIDI__tick_to_time[event.time]))
                        flipStart = None

                    if event.text == "[mix 2 drums2d]" or event.text == "[mix 4 drums4d]":
                        raise NotImplementedError("mix not implemented")

            if flipStart != None:
                self.discoFlip.append(Note(0, "disco", flipStart, self._PrettyMIDI__tick_to_time[track[-1].time]))

        self.discoFlip.sort(key=lambda x: x.start)
        if len(self.discoFlip) != 0:
            logging.debug("Disco Flip found in track ")

    def addDelay(self, delay):
        """
        move the notes according to a delay

        Parameters
        ----------
        delay : int
            delay in ms

        Raises
        ------
        NotImplementedError
            todo
        """
        if delay != 0:
            raise NotImplementedError("Add delay with PrettyMIDI not implemented")
            logging.debug("Delay in raw midi")
            originalTimes = np.array([0, midi.get_end_time()])
            midi.adjust_times(originalTimes, originalTimes + delay)
