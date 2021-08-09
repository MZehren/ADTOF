import midi


class GuitarProToMidiConverter(Converter):
    """
    Convert a guitar pro file (.gp5) to midi
    using https://github.com/Perlence/PyGuitarPro with doc in https://pyguitarpro.readthedocs.io/en/stable/
    see https://github.com/alexsteb/GuitarPro-to-Midi for alternative source
    
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def convert(self, filePath, outputName=None):

        with open(filePath, mode="rb") as file:
            binary = file.read()

        gp = guitarpro.parse(filePath)
        generatedMidi = self.generateMidi(gp)
        return generatedMidi

    def generateMidi(self, gp):

        tracks = [track for track in gp.tracks if track.channel.isPercussionChannel]
        assert len(tracks) == 1

        # Duplicate measures with repeat
        measures = []
        repeatAlternative = 0
        repeatStart = 0
        i = 0
        while i < len(tracks[0].measures):
            measure = tracks[0].measures[i]
            measures.append(measure)
            assert measure.header.repeatAlternative == 0

            if measure.header.isRepeatOpen:
                repeatStart = i
            if measure.header.repeatClose > 0:
                assert measure.header.repeatAlternative == -1 or measure.header.repeatAlternative == 0
                if repeatAlternative < measure.header.repeatClose:
                    i = repeatStart
                    repeatAlternative += 1
                    continue
                else:
                    repeatAlternative = 0
            i += 1

        # Get the notes from the duplicated measures
        notes = [[beat.start, note] for measure in measures for voice in measure.voices for beat in voice.beats for note in beat.notes]
        # notes.sort(key=lambda note: note[0])
        tempi = [[measure.start, measure.tempo] for measure in measures]

        #
        track = midi.Track()
        cursor = 0
        for start, note in notes:
            track.append(midi.NoteOnEvent(tick=start - cursor, velocity=note.velocity, pitch=note.value))
            track.append(midi.NoteOffEvent(tick=0, pitch=note.value))
            cursor = start

        track.append(midi.EndOfTrackEvent(tick=1))
        pattern = midi.Pattern()
        pattern.append(track)
        return pattern
