from collections import defaultdict
import bisect
from functools import lru_cache

import mido

from anticipation.config import MAX_DUR, MAX_PITCH, TIME_RESOLUTION
from anticipation.vocab import DUR_OFFSET, NOTE_OFFSET, TIME_OFFSET


DEFAULT_TEMPO_US_PER_BEAT = 500000
DEFAULT_UNCLOSED_DURATION_SEC = (TIME_RESOLUTION // 4) / TIME_RESOLUTION


def _time_token(seconds):
    return TIME_OFFSET + max(0, round(seconds * TIME_RESOLUTION))


def _dur_token(seconds):
    dur_units = max(0, round(seconds * TIME_RESOLUTION))
    return DUR_OFFSET + min(dur_units, MAX_DUR - 1)


class AsapScoreTiming:
    def __init__(self, raw_triplets, normalized_triplets, segment_starts_sec,
                 segment_starts_quarter, segment_tempos_us_per_beat):
        self.raw_triplets = [list(triplet) for triplet in raw_triplets]
        self.normalized_triplets = [list(triplet) for triplet in normalized_triplets]
        self.alignment_tuples = []
        self._segment_starts_sec = list(segment_starts_sec)
        self._segment_starts_quarter = list(segment_starts_quarter)
        self._segment_tempos_us_per_beat = list(segment_tempos_us_per_beat)
        self._normalized_lookup = {}

        for raw_triplet, normalized_triplet in zip(self.raw_triplets, self.normalized_triplets):
            self._normalized_lookup.setdefault(tuple(raw_triplet), list(normalized_triplet))

    def raw_seconds_to_normalized_seconds(self, raw_seconds):
        if not self._segment_starts_sec:
            return max(0.0, float(raw_seconds))

        idx = bisect.bisect_right(self._segment_starts_sec, raw_seconds) - 1
        idx = max(0, min(idx, len(self._segment_starts_sec) - 1))

        start_sec = self._segment_starts_sec[idx]
        start_quarter = self._segment_starts_quarter[idx]
        tempo_us_per_beat = self._segment_tempos_us_per_beat[idx]
        sec_per_quarter = tempo_us_per_beat / 1_000_000.0

        if sec_per_quarter <= 0:
            return max(0.0, float(start_quarter))

        quarter_pos = start_quarter + (raw_seconds - start_sec) / sec_per_quarter
        return max(0.0, float(quarter_pos))

    def normalize_raw_triplet(self, raw_triplet):
        normalized_triplet = self._normalized_lookup.get(tuple(raw_triplet))
        if normalized_triplet is not None:
            return list(normalized_triplet)

        raw_time_sec = (raw_triplet[0] - TIME_OFFSET) / TIME_RESOLUTION
        raw_duration_sec = max(0.0, (raw_triplet[1] - DUR_OFFSET) / TIME_RESOLUTION)

        normalized_time_sec = self.raw_seconds_to_normalized_seconds(raw_time_sec)
        if raw_duration_sec > 0:
            normalized_end_sec = self.raw_seconds_to_normalized_seconds(
                raw_time_sec + raw_duration_sec
            )
            normalized_duration_sec = max(0.0, normalized_end_sec - normalized_time_sec)
        else:
            normalized_duration_sec = 0.0

        return [
            _time_token(normalized_time_sec),
            _dur_token(normalized_duration_sec),
            int(raw_triplet[2]),
        ]


@lru_cache(maxsize=128)
def load_asap_score_timing(score_midi):
    midi = mido.MidiFile(score_midi)
    merged_track = mido.merge_tracks(midi.tracks)

    abs_ticks = 0
    abs_seconds = 0.0
    current_tempo = DEFAULT_TEMPO_US_PER_BEAT

    segment_starts_sec = [0.0]
    segment_starts_quarter = [0.0]
    segment_tempos_us_per_beat = [current_tempo]

    instruments = defaultdict(int)
    notes = []
    open_notes = defaultdict(list)

    for message in merged_track:
        delta_ticks = int(message.time)
        if delta_ticks:
            abs_seconds += mido.tick2second(delta_ticks, midi.ticks_per_beat, current_tempo)
            abs_ticks += delta_ticks

        if message.type == 'set_tempo':
            current_tempo = message.tempo
            current_quarter = abs_ticks / midi.ticks_per_beat
            if (
                segment_starts_sec
                and abs(segment_starts_sec[-1] - abs_seconds) < 1e-12
                and abs(segment_starts_quarter[-1] - current_quarter) < 1e-12
            ):
                segment_tempos_us_per_beat[-1] = current_tempo
            else:
                segment_starts_sec.append(abs_seconds)
                segment_starts_quarter.append(current_quarter)
                segment_tempos_us_per_beat.append(current_tempo)
            continue

        if message.type == 'program_change':
            instruments[message.channel] = message.program
            continue

        if message.type == 'note_on' and message.velocity > 0:
            instrument = 128 if message.channel == 9 else instruments[message.channel]
            note_token = NOTE_OFFSET + MAX_PITCH * instrument + message.note
            notes.append(
                {
                    'start_tick': abs_ticks,
                    'start_sec': abs_seconds,
                    'start_quarter': abs_ticks / midi.ticks_per_beat,
                    'tempo_us_per_beat': current_tempo,
                    'note_token': note_token,
                    'dur_tick': None,
                    'dur_sec': None,
                    'dur_quarter': None,
                }
            )
            open_notes[(instrument, message.note, message.channel)].append(len(notes) - 1)
            continue

        if message.type in ('note_off', 'note_on') and (
            message.type == 'note_off' or message.velocity == 0
        ):
            instrument = 128 if message.channel == 9 else instruments[message.channel]
            key = (instrument, message.note, message.channel)
            if key in open_notes and open_notes[key]:
                note_idx = open_notes[key].pop(0)
                note = notes[note_idx]
                note['dur_tick'] = abs_ticks - note['start_tick']
                note['dur_sec'] = max(0.0, abs_seconds - note['start_sec'])
                note['dur_quarter'] = max(0.0, note['dur_tick'] / midi.ticks_per_beat)

    raw_triplets = []
    normalized_triplets = []
    alignment_tuples = []

    for note in notes:
        raw_duration_sec = note['dur_sec']
        normalized_duration_sec = note['dur_quarter']

        if raw_duration_sec is None or normalized_duration_sec is None:
            raw_duration_sec = DEFAULT_UNCLOSED_DURATION_SEC
            sec_per_quarter = note['tempo_us_per_beat'] / 1_000_000.0
            if sec_per_quarter > 0:
                normalized_duration_sec = raw_duration_sec / sec_per_quarter
            else:
                normalized_duration_sec = raw_duration_sec

        raw_triplets.append(
            [
                _time_token(note['start_sec']),
                _dur_token(raw_duration_sec),
                note['note_token'],
            ]
        )
        alignment_tuples.append(
            [
                note['start_sec'],
                raw_triplets[-1][1] - DUR_OFFSET,
                raw_triplets[-1][2] - NOTE_OFFSET,
            ]
        )
        normalized_triplets.append(
            [
                _time_token(note['start_quarter']),
                _dur_token(normalized_duration_sec),
                note['note_token'],
            ]
        )

    timing = AsapScoreTiming(
        raw_triplets,
        normalized_triplets,
        segment_starts_sec,
        segment_starts_quarter,
        segment_tempos_us_per_beat,
    )
    timing.alignment_tuples = alignment_tuples
    return timing
