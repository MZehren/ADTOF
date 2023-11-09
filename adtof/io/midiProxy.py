import logging
import re
from bisect import bisect_left
from collections import namedtuple
from collections import defaultdict

import numpy as np
import pretty_midi

from adtof import config
from adtof.ressources import instrumentsMapping


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


class PrettyMidiWrapper(pretty_midi.PrettyMIDI):
    @classmethod
    def fromDict(cls, sparseResultIdx):
        """
        Write
        """
        midi = cls()
        instrument = pretty_midi.Instrument(program=1, is_drum=True)
        midi.instruments.append(instrument)
        for pitch, notes in sparseResultIdx.items():
            for i in notes:
                note = pretty_midi.Note(velocity=100, pitch=pitch, start=i, end=i + 0.1)
                instrument.notes.append(note)
        return midi

    @classmethod
    def fromListOfNotes(cls, notes, beats=[]):
        """Instantiate a pretty_midi.PrettyMIDI class with the list of notes.
        TODO: If the beats are provided, set tempo events as well

        Parameters
        ----------
        notes : [(position, pretttyMidi.Note)]
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
        for time, note in notes:
            note = pretty_midi.Note(velocity=note.velocity, pitch=note.pitch, start=time, end=time + 0.1)
            instrument.notes.append(note)

        for time, beatIdx in beats:
            note = pretty_midi.Note(velocity=100, pitch=beatIdx, start=time, end=time + 0.1)
            instrument.notes.append(note)

        return midi

    def get_beats_with_index(self, ingoreStart=False, stopTime=None):
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
        if stopTime is not None:
            beats = [t for t in beats if t <= stopTime]
            beatIdx = beatIdx[: len(beats)]
        return beats, beatIdx

    def _load_metadata(self, midi_data):
        """
        Call base class load_meatadata

        This implementation add text events as well to search for discoflip events:
        See: http://docs.c3universe.com/rbndocs/index.php?title=Drum_Authoring#Pro_Drum_and_Disco_Flip

        looking for [mix <difficulty> drums<configuration>] text events
        <difficulty> is a value, 0 through 3, where 0 is Easy, 1 is Medium, 2 is Hard, and 3 is Expert:
        We only look for expert difficulty: 3

        <configuration> refers to the configuration of the drum audio streams.
        is a value between 0 though 4 with different configuration of stems
        or a value of "0d" to "4d" for a "discoflip" mix inverting snare and HH
        """
        from pretty_midi.containers import Note

        super()._load_metadata(midi_data)
        self.discoFlip = []
        for track in midi_data.tracks:
            flipStart = None
            for event in track:
                if event.type == "text":
                    if re.search("\[mix 3 drums[0-4]d\]", event.text) is not None:
                        if flipStart is not None:
                            raise ValueError("DiscoFlip event before the end of the previous one")

                        flipStart = self._PrettyMIDI__tick_to_time[event.time]
                    elif re.search("\[mix 3 drums[0-4]dnoflip\]", event.text) is not None or re.search("\[mix 3 drums[0-4]\]", event.text) is not None:
                        # assert flipStart != None
                        if flipStart != None:
                            self.discoFlip.append(Note(0, "disco", flipStart, self._PrettyMIDI__tick_to_time[event.time]))
                        flipStart = None

            if flipStart != None:  # insterting end of disco flip if needed at the end
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
            delay in s

        Raises
        ------
        NotImplementedError
            todo
        """
        raise NotImplementedError("pretty_midi doesn't adjust the timing of the beats")
        # This correctly change the time of the notes, but doesn't adapt the time of the beats
        # if delay != 0:
        #     intervals = [0, self.get_end_time()]
        #     self.adjust_times(intervals, [i + delay for i in intervals])
        #     self.adjust_times(intervals, intervals)

        # # add the delay
        # if delay > 0:
        #     self.get_tempo_changes
        #     for i, track in enumerate(self.tracks):
        #         if i == 0:
        #             # If this is the tempo track, add a set tempo meta event event such as the delay is a 4 beats
        #             event = midi.SetTempoEvent()
        #             event.mpqn = int(delay * 1000000 / 4)
        #             track.insert(0, event)
        #             track[1].tick += 4 * self.tracks.resolution
        #         else:
        #             # If this is a standard track, add a delay of 4 beats to the first event
        #             track[0].tick += 4 * self.tracks.resolution
        # elif delay < 0:
        #     # reduce the position of the events until the delay is consummed
        #     warnings.warn("Track with negatie delay, the corrected midi has now misaligned beats")
        #     ticksToRemove = self.getSecondToTicks(-delay)
        #     for i, track in enumerate(self.tracks):
        #         decrement = ticksToRemove
        #         for event in track:
        #             if event.tick >= decrement:
        #                 event.tick -= decrement
        #                 break
        #             else:
        #                 decrement -= event.tick
        #                 event.tick = 0

    def quantizeNotes(self, subdivision=12):
        """
        Quantize the notes to the closest beat subdivision specified in parameters
        """
        notes = self.instruments[0].notes
        beats = self.get_beats()
        grid = []
        for i in range(len(beats) - 1):
            step = (beats[i + 1] - beats[i]) / subdivision
            grid += [beats[i] + j * step for j in range(subdivision)]

        for note in notes:
            note.start = self._take_closest(grid, note.start)

    def _take_closest(self, myList, myNumber):
        """
        Assumes myList is sorted. Returns closest value to myNumber.
        If two numbers are equally close, return the smallest number.
        """
        pos = bisect_left(myList, myNumber)
        if pos == 0:
            return myList[0]
        if pos == len(myList):
            return myList[-1]
        before = myList[pos - 1]
        after = myList[pos]
        if after - myNumber < myNumber - before:
            return after
        else:
            return before

    def getNotesGroup(self, grid):
        """
        return the notes in between the ticks of the grid.
        return the position of the note as a ratio relative to the ticks interval
        """
        notes = self.instruments[0].notes
        groups = {}
        Note = namedtuple("Note", ["start", "pitch"])
        for i in range(len(grid) - 1):
            start = grid[i]
            stop = grid[i + 1]
            group = [note for note in notes if note.start >= start and note.start < stop and note.velocity]
            groupRelative = [Note(np.round((note.start - start) / (stop - start), decimals=2), note.pitch) for note in group]
            groups[start] = groupRelative
        return groups

    def getOnsets(self, checkInstrumentIsDrum=True, **kwargs):
        """
        Returns:
            Dictionary of the shape {class: [positions]}
        """
        # Get all notes from all instruments
        events = []
        for instrument in [instrument for instrument in self.instruments if instrument.is_drum or not checkInstrumentIsDrum]:
            for note in instrument.notes:
                events.append({"time": note.start, "pitch": note.pitch, "velocity": note.velocity / 127})

        return events
