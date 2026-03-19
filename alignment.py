import numpy as np
import scipy.interpolate
from functools import lru_cache
from anticipation.convert import midi_to_events
from anticipation.config import *
from anticipation.vocab import *
from itertools import combinations

try:
    from alignment_cython import match_perf_to_score as _match_perf_to_score_fast
except ImportError:
    _match_perf_to_score_fast = None


@lru_cache(maxsize=512)
def load_annotation_file(file_path):
    annotations = []

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                timestamp = float(parts[0])
                annotation = parts[2]
                annotations.append((timestamp, annotation))

    return annotations


@lru_cache(maxsize=512)
def compare_annotations(file1_path, file2_path, interpolate=True):
    """
    Creates a mapping between downbeat and beat times in two annotation files.
    Inputs are timestamps in the first file, outputs are timestamps in the second file
    """

    annotations1 = load_annotation_file(file1_path)
    annotations2 = load_annotation_file(file2_path)

    min_length = min(len(annotations1), len(annotations2))
    if len(annotations1) != len(annotations2):
        shorter_file = file1_path if len(annotations1) == min_length else file2_path
        print(f'Number of annotations in {file1_path} and {file2_path} do not match.')
        print(f"Proceeding with the first {min_length} annotations from {shorter_file}.")

    data = []

    for i in range(min_length):
        data.append((annotations1[i][0], annotations2[i][0]))

    x, y = list(zip(*data))

    if interpolate:
        map = scipy.interpolate.interp1d(x, y)
        return map

    return x, y


@lru_cache(maxsize=512)
def _load_midi_note_tuples(file_path):
    events = midi_to_events(file_path, quantize=False)
    tuples = []
    for i in range(int(len(events) / 3)):
        tuples.append((
            events[3 * i] / TIME_RESOLUTION,
            events[3 * i + 1] - DUR_OFFSET,
            events[3 * i + 2] - NOTE_OFFSET,
        ))
    return tuple(tuples)


def power_set(lst, min_length=2, max_length=6):
    result = []
    # Only iterate from min_length to max_length (inclusive)
    for i in range(min_length, min(max_length + 1, len(lst) + 1)):
        result.extend(combinations(lst, i))
    return result


def _project_time_between_annotation_grids(time_point, src_grid, dst_grid):
    if len(src_grid) == 0 or len(dst_grid) == 0:
        return None

    if len(src_grid) == 1 or len(dst_grid) == 1:
        return float(dst_grid[0])

    if time_point <= src_grid[0]:
        src_span = src_grid[1] - src_grid[0]
        dst_span = dst_grid[1] - dst_grid[0]
        progress = (time_point - src_grid[0]) / src_span if src_span != 0 else 0.0
        return float(dst_grid[0] + progress * dst_span)

    for i in range(len(src_grid) - 1):
        if src_grid[i] <= time_point <= src_grid[i + 1]:
            src_span = src_grid[i + 1] - src_grid[i]
            dst_span = dst_grid[i + 1] - dst_grid[i]
            progress = (time_point - src_grid[i]) / src_span if src_span != 0 else 0.0
            return float(dst_grid[i] + progress * dst_span)

    src_span = src_grid[-1] - src_grid[-2]
    dst_span = dst_grid[-1] - dst_grid[-2]
    progress = (time_point - src_grid[-1]) / src_span if src_span != 0 else 0.0
    return float(dst_grid[-1] + progress * dst_span)


def _match_perf_to_score_python(p_tuples, s_tuples, p_min, p_max, mapped_times, thres):
    matched_indices = [-1] * len(p_tuples)
    available_score = [True] * len(s_tuples)
    score_indices_by_pitch = {}
    for idx, s_tuple in enumerate(s_tuples):
        score_indices_by_pitch.setdefault(s_tuple[2], []).append(idx)

    for i, p_tuple in enumerate(p_tuples):
        p_time, p_note = p_tuple[0], p_tuple[2]
        if not (p_min <= p_time <= p_max):
            continue

        mapped_time = mapped_times[i]
        best_dist = np.inf
        best_index = -1

        for k in score_indices_by_pitch.get(p_note, ()):
            if not available_score[k]:
                continue
            dist = np.abs(mapped_time - s_tuples[k][0])
            if dist <= thres and dist <= best_dist:
                best_dist = dist
                best_index = k

        if best_index >= 0:
            matched_indices[i] = best_index
            available_score[best_index] = False

    return matched_indices


def align_tokens(file1, file2, file3, file4, skip_Nones=True):
    # turn midi into events, without quantizing so we can get 16 digits of precision in arrival time
    perf = midi_to_events(file1, quantize=False)
    score = midi_to_events(file2, quantize=False)

    p_beats, s_beats = compare_annotations(file3, file4, interpolate=False)
    s_beats = np.array(s_beats)
    p_beats = np.array(p_beats)
    map = compare_annotations(file3, file4)

    # create tuples, scaling arrival time back to seconds, which is the unit the annotation mapping uses
    p_tuples = [[perf[3 * i] / TIME_RESOLUTION, perf[3 * i + 1] - DUR_OFFSET, perf[3 * i + 2] - NOTE_OFFSET]
                for i in range(int(len(perf) / 3))]
    s_tuples = [[score[3 * i] / TIME_RESOLUTION, score[3 * i + 1] - DUR_OFFSET, score[3 * i + 2] - NOTE_OFFSET]
                for i in range(int(len(score) / 3))]
    p_times = [tup[0] for tup in p_tuples]

    tol = 1e-4

    # match score notes with corresponding beats in annotation file
    s_tuples_b = []
    assigned = []

    for tup in s_tuples:
        mask = np.abs(tup[0] - s_beats) <= tol
        if sum(mask):
            beat = list(np.where(mask)[0])[0]
            s_tuples_b.append((tup[0], tup[1], tup[2], beat))
            assigned.append(beat)
        else:
            s_tuples_b.append(tup)

    for i in range(len(s_beats)):
        if i not in assigned:
            print(f'could not find notes in score associated with beat {i}')

    # match perf notes with corresponding beats in annotation file
    p_tuples_b = []
    assigned = []

    for tup in p_tuples:
        mask = np.abs(tup[0] - p_beats) <= tol
        if sum(mask):
            beat = list(np.where(mask)[0])[0]
            p_tuples_b.append((tup[0], tup[1], tup[2], beat))
            assigned.append(beat)
        else:
            p_tuples_b.append(tup)

    for j in [i for i in range(len(p_beats)) if i not in assigned]:
        beat = p_beats[j]

        candidates = [tup[0] for tup in p_tuples_b if len(tup) == 3 and abs(tup[0] - beat) <= 0.5]

        success = False
        for subset in power_set(candidates):
            if np.abs(np.average(subset) - beat) <= tol:
                for time in subset:
                    k = p_times.index(time)
                    p_tuples_b[k] = (p_tuples_b[k][0], p_tuples_b[k][1], p_tuples_b[k][2], j)
                success = True
                break
        if not success:
            print(f'could not find notes in perf associated with beat {j}')

    # match score and perf notes that occurred on the same beats, then between
    # (almost all correctly) matched beats, we want to use the map to match off-beat notes
    # outside of the mapping range and domain, just match notes with the same pitch
    matched_tuples = []

    s_tuples_b_copy = s_tuples_b.copy()

    p_min = map.x.min()
    p_max = map.x.max()
    s_min = map.y.min()
    s_max = map.y.max()

    for i, p_tuple in enumerate(p_tuples_b):
        for j, s_tuple in enumerate(s_tuples_b_copy):
            p_time, p_note = p_tuple[0], p_tuple[2]
            s_time, s_note = s_tuple[0], s_tuple[2]

            k = s_tuples_b.index(s_tuple)

            if len(p_tuple) == 4 and len(s_tuple) == 4 and p_tuple[2:] == s_tuple[2:]:
                matched_tuples.append([p_tuple, i, s_tuple, k])
                s_tuples_b_copy.remove(s_tuple)
            elif len(p_tuple) == 3 and len(s_tuple) == 3 and p_time < p_min and s_time < s_min and p_note == s_note:
                matched_tuples.append([p_tuple, i, s_tuple, k])
                s_tuples_b_copy.remove(s_tuple)
            elif len(p_tuple) == 3 and len(s_tuple) == 3 and p_time > p_max and s_time > s_max and p_note == s_note:
                matched_tuples.append([p_tuple, i, s_tuple, k])
                s_tuples_b_copy.remove(s_tuple)
            elif len(p_tuple) == 3 and len(s_tuple) == 3 and p_min <= p_time <= p_max and s_min <= s_time <= s_max and \
                    np.abs(map(p_time) - s_time) < .1 and p_note == s_note:
                matched_tuples.append([p_tuple, i, s_tuple, k])
                s_tuples_b_copy.remove(s_tuple)
        if p_tuple not in [l[0] for l in matched_tuples] and not skip_Nones:
            matched_tuples.append([p_tuple, i, [None, None, None], None])

    # revert back to token format and remove beat indices
    for i, l in enumerate(matched_tuples):
        # performance tokens should have control offset
        l[0] = [round(l[0][0] * TIME_RESOLUTION), l[0][1] + DUR_OFFSET, l[0][2] + NOTE_OFFSET]
        l[0] = [CONTROL_OFFSET + t for t in l[0]]

        if l[2][0] is not None:
            l[2] = [round(l[2][0] * TIME_RESOLUTION), l[2][1] + DUR_OFFSET, l[2][2] + NOTE_OFFSET]
        matched_tuples[i] = l

    return matched_tuples


def align_tokens2(file1, file2, file3, file4, skip_Nones=True, thres=0.1,
                  preserve_unmatched_perf=False, score_tuples=None):
    # turn midi into events, without quantizing so we can get 16 digits of precision in arrival time
    p_tuples = _load_midi_note_tuples(file1)

    p_beats, s_beats = compare_annotations(file3, file4, interpolate=False)
    map = compare_annotations(file3, file4)

    if score_tuples is None:
        s_tuples = _load_midi_note_tuples(file2)
    else:
        s_tuples = tuple(tuple(tup) for tup in score_tuples)

    p_min = map.x.min()
    p_max = map.x.max()
    mapped_times = [
        float(map(p_tuple[0])) if p_min <= p_tuple[0] <= p_max else 0.0
        for p_tuple in p_tuples
    ]

    match_impl = _match_perf_to_score_fast or _match_perf_to_score_python
    matched_indices = match_impl(p_tuples, s_tuples, p_min, p_max, mapped_times, thres)
    matched_tuples = []

    for i, p_tuple in enumerate(p_tuples):
        best_match = [None, None, None]
        best_index = matched_indices[i]

        if best_index >= 0:
            best_match = s_tuples[best_index]
            matched_tuples.append([p_tuple, i, best_match, best_index])
        elif not skip_Nones:
            if preserve_unmatched_perf:
                projected_score_time = _project_time_between_annotation_grids(
                    p_tuple[0],
                    p_beats,
                    s_beats,
                )
                if projected_score_time is not None:
                    best_match = [projected_score_time, 0, REST - NOTE_OFFSET]
            matched_tuples.append([p_tuple, i, best_match, None])

    # revert back to token format
    for i, l in enumerate(matched_tuples):
        # performance tokens should have control offset
        l[0] = [round(l[0][0] * TIME_RESOLUTION), l[0][1] + DUR_OFFSET, l[0][2] + NOTE_OFFSET]
        l[0] = [CONTROL_OFFSET + t for t in l[0]]

        if l[2][0] is not None:
            l[2] = [
                max(0, round(l[2][0] * TIME_RESOLUTION)),
                l[2][1] + DUR_OFFSET,
                l[2][2] + NOTE_OFFSET,
            ]

        matched_tuples[i] = l

    return matched_tuples
