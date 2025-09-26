import os
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial

from tqdm import tqdm

from anticipation.config import *
from anticipation.vocab import *
from anticipation import ops
from alignment import align_tokens2


def _interleave_tokenize4_single(filegroup, skip_Nones=True, prefix_controls=33):
    """Worker: process a single (perf, score, perf_ann, score_ann) tuple.

    Returns a tuple: (seq_lines: List[str], stats: dict)
    Mirrors tokenize4 logic but within a single piece (no cross-piece concatenation).
    """
    file1, file2, file3, file4 = filegroup
    try:
        matched_tuples = align_tokens2(file1, file2, file3, file4, skip_Nones=skip_Nones)
    except Exception as e:
        return [], {"seq": 0, "discarded": 1, "err": str(e)}

    # Build interleaved stream: fixed-length control+pad prefix, then alternate score/control
    interleaved_tokens = []

    k = min(prefix_controls, len(matched_tuples))
    for t in matched_tuples[:k]:
        cc = t[0]
        interleaved_tokens.extend(cc)
        cc_time = cc[0] - CONTROL_OFFSET
        interleaved_tokens.extend([TIME_OFFSET + cc_time, DUR_OFFSET + 0, REST])

    for i, t in enumerate(matched_tuples):
        sc = t[2]
        if sc[0] is not None:
            interleaved_tokens.extend(sc)
        ii = i + k
        if ii < len(matched_tuples):
            interleaved_tokens.extend(matched_tuples[ii][0])

    # Prepend separators
    interleaved_tokens[0:0] = [SEPARATOR, SEPARATOR, SEPARATOR]

    # Chunk into sequences of 1023 body tokens and add global mode token
    concatenated_tokens = interleaved_tokens
    lines = []
    z = ANTICIPATE
    stats_discards = 0
    while len(concatenated_tokens) >= EVENT_SIZE * M:
        seq = concatenated_tokens[0:EVENT_SIZE * M]
        concatenated_tokens = concatenated_tokens[EVENT_SIZE * M:]
        seq = ops.translate(seq, -ops.min_time(seq, seconds=False), seconds=False)
        if ops.min_time(seq, seconds=False) != 0:
            # safety
            dt = -ops.min_time(seq, seconds=False)
            seq = ops.translate(seq, dt, seconds=False)
        if ops.max_time(seq, seconds=False) >= MAX_TIME:
            stats_discards += 1
            continue
        seq.insert(0, z)
        lines.append(' '.join(str(tok) for tok in seq))

    return lines, {"seq": len(lines), "discarded": stats_discards}


def _worker_split(payload):
    """Top-level wrapper to keep Windows spawn picklable.

    payload = (filegroup, split, skip_Nones, prefix_controls)
    Returns (split, lines, stats)
    """
    fg, split, skip_Nones, prefix_controls = payload
    lines, stats = _interleave_tokenize4_single(fg, skip_Nones=skip_Nones, prefix_controls=prefix_controls)
    return split, lines, stats

def main():
    ap = argparse.ArgumentParser(description='Parallel ASAP tokenization with producer-consumer and dataset split by score')
    ap.add_argument('--asap-root', default='./asap-dataset-master', help='Path to ASAP dataset root')
    ap.add_argument('--workers', type=int, default=max(os.cpu_count() or 1, 1), help='Number of parallel workers')
    ap.add_argument('--test-frac', type=float, default=0.2, help='Fraction of unique scores to reserve for test split')
    ap.add_argument('--prefix-controls', type=int, default=33, help='Fixed number of control tokens for t4 prefix')
    ap.add_argument('--skip-nones', action='store_true', help='Drop unmatched performance notes')
    ap.add_argument('--seed', type=int, default=0, help='Random seed for split reproducibility')
    ap.add_argument('--out-train', default='./data/train_output.txt')
    ap.add_argument('--out-test', default='./data/test_output.txt')
    args = ap.parse_args()

    print('Tokenization parameters:')
    print(f'  anticipation interval = {DELTA}s')
    print(f'  max track length = {MAX_TRACK_TIME_IN_SECONDS}s')
    print(f'  min track length = {MIN_TRACK_TIME_IN_SECONDS}s')
    print(f'  min track events = {MIN_TRACK_EVENTS}')
    print(f'  workers = {args.workers}')

    meta_csv = os.path.join(args.asap_root, 'metadata.csv')
    df = pd.read_csv(meta_csv)

    # Build file tuples and track score IDs for split
    datafiles = []
    score_keys = []  # use midi_score path as the key
    for _, row in df.iterrows():
        file1 = os.path.join(args.asap_root, row['midi_performance'])
        file2 = os.path.join(args.asap_root, row['midi_score'])
        file3 = os.path.join(args.asap_root, row['performance_annotations'])
        file4 = os.path.join(args.asap_root, row['midi_score_annotations'])
        datafiles.append((file1, file2, file3, file4))
        score_keys.append(file2)

    # Split by unique score: enforce that a score never appears in both splits
    rng = np.random.default_rng(args.seed)
    unique_scores = list(sorted(set(score_keys)))
    rng.shuffle(unique_scores)
    n_test = int(np.ceil(args.test_frac * len(unique_scores)))
    test_scores = set(unique_scores[:n_test])
    train_scores = set(unique_scores[n_test:])

    tasks_train = []
    tasks_test = []
    for fg, score in zip(datafiles, score_keys):
        if score in test_scores:
            tasks_test.append(fg)
        else:
            tasks_train.append(fg)

    os.makedirs(os.path.dirname(args.out_train), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_test), exist_ok=True)

    total_tasks = len(tasks_train) + len(tasks_test)
    print(f'Total pieces: {total_tasks} (train: {len(tasks_train)}, test: {len(tasks_test)})')
    print(f'Writing to: {args.out_train} and {args.out_test}')

    seq_train = seq_test = 0
    disc_train = disc_test = 0

    with open(args.out_train, 'w') as f_train, open(args.out_test, 'w') as f_test:
        with Pool(processes=args.workers) as pool:
            # Chain test and train with split tags
            payloads = [(fg, 'test', args.skip_nones, args.prefix_controls) for fg in tasks_test] + \
                       [(fg, 'train', args.skip_nones, args.prefix_controls) for fg in tasks_train]

            # Submit work and consume results with a giant progress bar
            with tqdm(total=len(payloads), desc='Tokenizing pieces', unit='piece') as pbar:
                for split, lines, stats in pool.imap_unordered(_worker_split, payloads):
                    if split == 'test':
                        for line in lines:
                            f_test.write(line + '\n')
                        seq_test += stats.get('seq', 0)
                        disc_test += stats.get('discarded', 0)
                    else:
                        for line in lines:
                            f_train.write(line + '\n')
                        seq_train += stats.get('seq', 0)
                        disc_train += stats.get('discarded', 0)
                    pbar.update(1)

    print('Tokenization complete.')
    print(f'  Train sequences written: {seq_train} (discarded slices: {disc_train})')
    print(f'  Test sequences written : {seq_test} (discarded slices: {disc_test})')
    print('Remember to shuffle the training split!')

if __name__ == '__main__':
    main()
