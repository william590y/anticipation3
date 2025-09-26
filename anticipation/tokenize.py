"""
Top-level functions for preprocessing data to be used for training.
"""

from tqdm import tqdm

import numpy as np

from anticipation import ops
from anticipation.config import *
from anticipation.vocab import *
from anticipation.convert import compound_to_events, midi_to_interarrival, midi_to_compound
from alignment import *


def extract_spans(all_events, rate):
    events = []
    controls = []
    span = True
    next_span = end_span = TIME_OFFSET+0
    for time, dur, note in zip(all_events[0::3],all_events[1::3],all_events[2::3]):
        assert(note not in [SEPARATOR, REST]) # shouldn't be in the sequence yet

        # end of an anticipated span; decide when to do it again (next_span)
        if span and time >= end_span:
            span = False
            next_span = time+int(TIME_RESOLUTION*np.random.exponential(1./rate))

        # anticipate a 3-second span
        if (not span) and time >= next_span:
            span = True
            end_span = time + DELTA*TIME_RESOLUTION

        if span:
            # mark this event as a control
            controls.extend([CONTROL_OFFSET+time, CONTROL_OFFSET+dur, CONTROL_OFFSET+note])
        else:
            events.extend([time, dur, note])

    return events, controls


ANTICIPATION_RATES = 10
def extract_random(all_events, rate):
    events = []
    controls = []
    for time, dur, note in zip(all_events[0::3],all_events[1::3],all_events[2::3]):
        assert(note not in [SEPARATOR, REST]) # shouldn't be in the sequence yet

        if np.random.random() < rate/float(ANTICIPATION_RATES):
            # mark this event as a control
            controls.extend([CONTROL_OFFSET+time, CONTROL_OFFSET+dur, CONTROL_OFFSET+note])
        else:
            events.extend([time, dur, note])

    return events, controls


def extract_instruments(all_events, instruments):
    events = []
    controls = []
    for time, dur, note in zip(all_events[0::3],all_events[1::3],all_events[2::3]):
        assert note < CONTROL_OFFSET         # shouldn't be in the sequence yet
        assert note not in [SEPARATOR, REST] # these shouldn't either

        instr = (note-NOTE_OFFSET)//2**7
        if instr in instruments:
            # mark this event as a control
            controls.extend([CONTROL_OFFSET+time, CONTROL_OFFSET+dur, CONTROL_OFFSET+note])
        else:
            events.extend([time, dur, note])

    return events, controls


def maybe_tokenize(compound_tokens):
    """
    Tokenizes a sequence of compound tokens if the length is appropriate.
    Returns the list of events and truncations (number of notes above 10s that were truncated)
    """
    # skip sequences with very few events
    if len(compound_tokens) < COMPOUND_SIZE*MIN_TRACK_EVENTS:
        return None, None, 1 # short track

    events, truncations = compound_to_events(compound_tokens, stats=True)
    end_time = ops.max_time(events, seconds=False)

    # don't want to deal with extremely short tracks
    if end_time < TIME_RESOLUTION*MIN_TRACK_TIME_IN_SECONDS:
        return None, None, 1 # short track

    # don't want to deal with extremely long tracks
    if end_time > TIME_RESOLUTION*MAX_TRACK_TIME_IN_SECONDS:
        return None, None, 2 # long track

    # skip sequences more instruments than MIDI channels (16)
    if len(ops.get_instruments(events)) > MAX_TRACK_INSTR:
        return None, None, 3 # too many instruments

    return events, truncations, 0


def tokenize_ia(datafiles, output, augment_factor, idx=0, debug=False):
    assert augment_factor == 1 # can't augment interarrival-tokenized data

    all_truncations = 0
    seqcount = rest_count = 0
    stats = 4*[0] # (short, long, too many instruments, inexpressible)
    np.random.seed(0)

    with open(output, 'w') as outfile:
        concatenated_tokens = []
        for j, filename in tqdm(list(enumerate(datafiles)), desc=f'#{idx}', position=idx+1, leave=True):
            with open(filename, 'r') as f:
                _, _, status = maybe_tokenize([int(token) for token in f.read().split()])

            if status > 0:
                stats[status-1] += 1
                continue

            filename = filename[:-len('.compound.txt')] # get the original MIDI

            # already parsed; shouldn't raise an exception
            tokens, truncations = midi_to_interarrival(filename, stats=True)
            tokens[0:0] = [MIDI_SEPARATOR]
            concatenated_tokens.extend(tokens)
            all_truncations += truncations

            # write out full sequences to file
            while len(concatenated_tokens) >= CONTEXT_SIZE:
                seq = concatenated_tokens[0:CONTEXT_SIZE]
                concatenated_tokens = concatenated_tokens[CONTEXT_SIZE:]
                outfile.write(' '.join([str(tok) for tok in seq]) + '\n')
                seqcount += 1

    if debug:
        fmt = 'Processed {} sequences (discarded {} tracks, discarded {} seqs, added {} rest tokens)'
        print(fmt.format(seqcount, stats[0]+stats[1]+stats[2], stats[3], rest_count))

    return (seqcount, rest_count, stats[0], stats[1], stats[2], stats[3], all_truncations)


def tokenize(datafiles, output, augment_factor, idx=0, debug=False):
    """
    Applies anticipatory tokenization to a list of datafiles, writing the results to output.
    1. These datafiles should be .txt files containing compound tokenizations, which are converted
       to events via maybe_tokenize.
    2. Creates controls out of the events via augment_factor, or no augmentation (pure autoregression)
       if augment_factor == 1.
    3. Calls anticipate() to interleave controls and events
    4. Splits the tokens into sequences of length 1023, which are written to the output
    """
    tokens = []
    all_truncations = 0
    seqcount = rest_count = 0
    stats = 4*[0] # (short, long, too many instruments, inexpressible)
    np.random.seed(0)

    with open(output, 'w') as outfile:
        concatenated_tokens = []
        for j, filename in tqdm(list(enumerate(datafiles)), desc=f'#{idx}', position=idx+1, leave=True):
            with open(filename, 'r') as f:
                all_events, truncations, status = maybe_tokenize([int(token) for token in f.read().split()])

            if status > 0:
                stats[status-1] += 1
                continue

            instruments = list(ops.get_instruments(all_events).keys())
            end_time = ops.max_time(all_events, seconds=False)

            # different random augmentations
            for k in range(augment_factor):
                if k % 10 == 0:
                    # no augmentation
                    events = all_events.copy()
                    controls = []
                elif k % 10 == 1:
                    # span augmentation
                    lmbda = .05
                    events, controls = extract_spans(all_events, lmbda)
                elif k % 10 < 6:
                    # random augmentation
                    r = np.random.randint(1,ANTICIPATION_RATES)
                    events, controls = extract_random(all_events, r)
                else:
                    if len(instruments) > 1:
                        # instrument augmentation: at least one, but not all instruments
                        u = 1+np.random.randint(len(instruments)-1)
                        subset = np.random.choice(instruments, u, replace=False)
                        events, controls = extract_instruments(all_events, subset)
                    else:
                        # no augmentation
                        events = all_events.copy()
                        controls = []

                if len(concatenated_tokens) == 0:
                    z = ANTICIPATE if k % 10 != 0 else AUTOREGRESS

                all_truncations += truncations
                events = ops.pad(events, end_time)
                rest_count += sum(1 if tok == REST else 0 for tok in events[2::3])
                tokens, controls = ops.anticipate(events, controls)
                assert len(controls) == 0 # should have consumed all controls (because of padding)
                tokens[0:0] = [SEPARATOR, SEPARATOR, SEPARATOR]
                concatenated_tokens.extend(tokens)

                # write out full sequences to file
                while len(concatenated_tokens) >= EVENT_SIZE*M:
                    seq = concatenated_tokens[0:EVENT_SIZE*M]
                    concatenated_tokens = concatenated_tokens[EVENT_SIZE*M:]

                    # relativize time to the context
                    seq = ops.translate(seq, -ops.min_time(seq, seconds=False), seconds=False)
                    assert ops.min_time(seq, seconds=False) == 0
                    if ops.max_time(seq, seconds=False) >= MAX_TIME:
                        stats[3] += 1
                        continue

                    # if seq contains SEPARATOR, global controls describe the first sequence
                    seq.insert(0, z)

                    outfile.write(' '.join([str(tok) for tok in seq]) + '\n')
                    seqcount += 1

                    # grab the current augmentation controls if we didn't already
                    z = ANTICIPATE if k % 10 != 0 else AUTOREGRESS

    if debug:
        fmt = 'Processed {} sequences (discarded {} tracks, discarded {} seqs, added {} rest tokens)'
        print(fmt.format(seqcount, stats[0]+stats[1]+stats[2], stats[3], rest_count))

    return (seqcount, rest_count, stats[0], stats[1], stats[2], stats[3], all_truncations)


def tokenize2(datafiles, output, idx=0, debug=False):
    """
    Applies anticipatory tokenization to a list of datafiles where each is a tuple
    (file1, file2, file3, file4) with 
    1. file1 being the path to the performance MIDI file
    2. file2 being the path to the score MIDI file
    3. file3 being the path to the performance annotation file
    4. file4 being the path to the score annotation file

    Note: This is the old tokenization process that uses anticipation with mapping
    """
    tokens = []
    all_truncations = 0
    seqcount = rest_count = 0
    stats = 4*[0] # (short, long, too many instruments, inexpressible)
    np.random.seed(0)

    with open(output, 'w') as outfile:
        concatenated_tokens = []
        for j, filegroup in tqdm(list(enumerate(datafiles)), desc=f'#{idx}', position=idx+1, leave=True):

            file1, file2 = midi_to_compound(filegroup[0]), midi_to_compound(filegroup[1])
            file3, file4 = filegroup[2], filegroup[3]

            controls, truncations_c, _ = maybe_tokenize(file1)
            controls = [CONTROL_OFFSET+token for token in controls] # mark these tokens as controls
            all_events, truncations_e, _ = maybe_tokenize(file2)

            z = ANTICIPATE

            all_truncations += truncations_c + truncations_e

            # only need to pad the events 
            events = ops.pad(all_events, end_time=ops.max_time(all_events, seconds=False))

            rest_count += sum(1 if tok == REST else 0 for tok in events[2::3])

            map = compare_annotations(file4, file3) # create mapping from score to performance
            tokens, controls = ops.anticipate2(events, controls, map)

            assert len(controls) == 0 # should have consumed all controls (because of padding)
            tokens[0:0] = [SEPARATOR, SEPARATOR, SEPARATOR]
            concatenated_tokens.extend(tokens)

            # write sequences of length EVENT_SIZE*M = 1023 to the output file,
            # any extra remain in concatenated_tokens for the next input file.      
            while len(concatenated_tokens) >= EVENT_SIZE*M:
                seq = concatenated_tokens[0:EVENT_SIZE*M]
                concatenated_tokens = concatenated_tokens[EVENT_SIZE*M:]

                # make sure each sequence starts at time 0
                seq = ops.translate(seq, -ops.min_time(seq, seconds=False), seconds=False)
                assert ops.min_time(seq, seconds=False) == 0
                if ops.max_time(seq, seconds=False) >= MAX_TIME:
                    stats[3] += 1
                    continue

                # if seq contains SEPARATOR, global controls describe the first sequence
                seq.insert(0, z)

                outfile.write(' '.join([str(tok) for tok in seq]) + '\n')
                seqcount += 1

    if debug:
        fmt = 'Processed {} sequences (discarded {} tracks, discarded {} seqs, added {} rest tokens)'
        print(fmt.format(seqcount, stats[0]+stats[1]+stats[2], stats[3], rest_count))

    return (seqcount, rest_count, stats[0], stats[1], stats[2], stats[3], all_truncations)

def tokenize3(datafiles, output, idx=0, debug=False, skip_Nones=True):
    """
    Applies anticipatory tokenization to a list of datafiles where each is a tuple
    (file1, file2, file3, file4) with 
    1. file1 being the path to the performance MIDI file
    2. file2 being the path to the score MIDI file
    3. file3 being the path to the performance annotation file
    4. file4 being the path to the score annotation file

    Note: This is the new tokenization process that alternates score and perf tokens and inserts 
          None,None,None tokens whenver a corresponding score token cannot be found.
    """
    tokens = []
    all_truncations = 0
    seqcount = rest_count = 0
    stats = 4*[0] # (short, long, too many instruments, inexpressible)
    np.random.seed(0)

    with open(output, 'w') as outfile:
        concatenated_tokens = []
        for j, filegroup in tqdm(list(enumerate(datafiles)), desc=f'#{idx}', position=idx+1, leave=True):

            file1,file2,file3,file4 = filegroup

            print(f'Now aligning {file1} and {file2}')
            matched_tuples = align_tokens2(file1,file2,file3,file4,skip_Nones=skip_Nones)

            # interleave the tokens via alternation
            interleaved_tokens = []

            for i, l in enumerate(matched_tuples):
                if l[0][0]-CONTROL_OFFSET <= DELTA*TIME_RESOLUTION:
                    interleaved_tokens.extend(l[0])

            prefix_len = int(len(interleaved_tokens)/3)

            for i, l in enumerate(matched_tuples):
                if i < len(matched_tuples)-prefix_len:
                    interleaved_tokens.extend(l[2])
                    interleaved_tokens.extend(matched_tuples[i+prefix_len][0])
                else:
                    interleaved_tokens.extend(l[2])

            # print(interleaved_tokens)

            # because we already have a sequence of interleaved tokens, don't want to make any truncations
            # controls, truncations_c, _ = maybe_tokenize(file1)
            # controls = [CONTROL_OFFSET+token for token in controls] # mark these tokens as controls
            # all_events, truncations_e, _ = maybe_tokenize(file2)

            z = ANTICIPATE

            # all_truncations += truncations_c + truncations_e

            # only need to pad the events 
            # events = ops.pad(all_events, end_time=ops.max_time(all_events, seconds=False))

            # rest_count += sum(1 if tok == REST else 0 for tok in events[2::3])

            # map = compare_annotations(file4, file3) # create mapping from score to performance
            # tokens, controls = ops.anticipate2(events, controls, map)

            # assert len(controls) == 0 # should have consumed all controls (because of padding)

            # separator is a special token with value 55025
            # NOTE: prepend separators to the actual interleaved stream
            interleaved_tokens[0:0] = [SEPARATOR, SEPARATOR, SEPARATOR]
            concatenated_tokens.extend(interleaved_tokens)

            # write sequences of length EVENT_SIZE*M = 1023 to the output file,
            # any extra remain in concatenated_tokens for the next input file.      
            while len(concatenated_tokens) >= EVENT_SIZE*M:
                seq = concatenated_tokens[0:EVENT_SIZE*M]
                concatenated_tokens = concatenated_tokens[EVENT_SIZE*M:]

                # make sure each sequence starts at time 0 (shifts each token's arrival time by the 
                # min time of the sequence, accounting for control offsets)
                seq = ops.translate(seq, -ops.min_time(seq, seconds=False), seconds=False)
                assert ops.min_time(seq, seconds=False) == 0
                if ops.max_time(seq, seconds=False) >= MAX_TIME:
                    stats[3] += 1
                    continue

                # if seq contains SEPARATOR, global controls describe the first sequence
                seq.insert(0, z)

                outfile.write(' '.join([str(tok) for tok in seq]) + '\n')
                seqcount += 1

    if debug:
        fmt = 'Processed {} sequences (discarded {} tracks, discarded {} seqs, added {} rest tokens)'
        print(fmt.format(seqcount, stats[0]+stats[1]+stats[2], stats[3], rest_count))

    return (seqcount, rest_count, stats[0], stats[1], stats[2], stats[3], all_truncations)


def tokenize4(datafiles, output, idx=0, debug=False, skip_Nones=True, prefix_controls=33):
    """
    Like tokenize3, but instead of taking ~5 seconds of control tokens up front, it
    uses the first `prefix_controls` control triples as a bootstrap prefix and
    alternates each control with a REST padding event. After that, it alternates
    score and (future) control tokens with a fixed offset of `prefix_controls`.

    This makes the initial conditioning period a fixed length by count rather than time.
    """
    tokens = []
    all_truncations = 0
    seqcount = rest_count = 0
    stats = 4*[0] # (short, long, too many instruments, inexpressible)
    np.random.seed(0)

    with open(output, 'w') as outfile:
        concatenated_tokens = []
        for j, filegroup in tqdm(list(enumerate(datafiles)), desc=f'#{idx}', position=idx+1, leave=True):
            file1, file2, file3, file4 = filegroup

            print(f'Now aligning {file1} and {file2}')
            matched_tuples = align_tokens2(file1, file2, file3, file4, skip_Nones=skip_Nones)

            # Build the interleaved stream with a fixed-length control+padding prefix
            interleaved_tokens = []

            # prefix of control tokens by count (not time), alternating with REST pads
            k = min(prefix_controls, len(matched_tuples))
            for t in matched_tuples[:k]:
                cc = t[0]  # control triple (ATIME/ADUR/ANOTE with CONTROL offsets)
                interleaved_tokens.extend(cc)
                # Create a REST event at the same absolute time as control (duration=0)
                cc_time = cc[0] - CONTROL_OFFSET
                interleaved_tokens.extend([TIME_OFFSET + cc_time, DUR_OFFSET + 0, REST])

            prefix_len = k

            # Main alternation: score_i followed by control_{i+prefix_len}
            for i, t in enumerate(matched_tuples):
                score = t[2]
                if score[0] is not None:
                    interleaved_tokens.extend(score)
                # Append future control with fixed offset when in range
                ii = i + prefix_len
                if ii < len(matched_tuples):
                    interleaved_tokens.extend(matched_tuples[ii][0])

            # Prepend separators to delineate a new sequence
            interleaved_tokens[0:0] = [SEPARATOR, SEPARATOR, SEPARATOR]

            z = ANTICIPATE

            concatenated_tokens.extend(interleaved_tokens)

            # Chunk into fixed-length sequences
            while len(concatenated_tokens) >= EVENT_SIZE*M:
                seq = concatenated_tokens[0:EVENT_SIZE*M]
                concatenated_tokens = concatenated_tokens[EVENT_SIZE*M:]

                # Translate times so the minimum time in the slice is zero
                seq = ops.translate(seq, -ops.min_time(seq, seconds=False), seconds=False)
                assert ops.min_time(seq, seconds=False) == 0
                if ops.max_time(seq, seconds=False) >= MAX_TIME:
                    stats[3] += 1
                    continue

                # Insert the global mode flag at the start to get 1024 tokens total
                seq.insert(0, z)

                outfile.write(' '.join([str(tok) for tok in seq]) + '\n')
                seqcount += 1

    if debug:
        fmt = 'Processed {} sequences (discarded {} tracks, discarded {} seqs, added {} rest tokens)'
        print(fmt.format(seqcount, stats[0]+stats[1]+stats[2], stats[3], rest_count))

    return (seqcount, rest_count, stats[0], stats[1], stats[2], stats[3], all_truncations)