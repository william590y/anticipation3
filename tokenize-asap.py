import os
import numpy as np
import pandas as pd
from multiprocessing import Pool, RLock
from glob import glob

from tqdm import tqdm

from anticipation.config import *
from anticipation.tokenize import tokenize2

def main():
    print('Tokenization parameters:')
    print(f'  anticipation interval = {DELTA}s')
    print(f'  max track length = {MAX_TRACK_TIME_IN_SECONDS}s')
    print(f'  min track length = {MIN_TRACK_TIME_IN_SECONDS}s')
    print(f'  min track events = {MIN_TRACK_EVENTS}')

    BASE = "./asap-dataset-master/"
    df = pd.read_csv('asap-dataset-master/metadata.csv')

    datafiles = []

    for i, row in df.iterrows():
        file1 = BASE + row['midi_performance']
        file2 = BASE + row['midi_score']
        file3 = BASE + row['performance_annotations']
        file4 = BASE + row['midi_score_annotations']

        datafiles.append((file1, file2, file3, file4))

    print('Tokenizing data; will be written to output.txt')

    seq_count, rest_count, too_short, too_long, too_manyinstr, discarded_seqs, truncations \
            = tokenize2(datafiles, output='./data/output.txt')
    rest_ratio = round(100*float(rest_count)/(seq_count*M),2)

    trunc_type = 'duration' #'interarrival' if args.interarrival else 'duration'
    trunc_ratio = round(100*float(truncations)/(seq_count*M),2)

    print('Tokenization complete.')
    print(f'  => Processed {seq_count} training sequences')
    print(f'  => Inserted {rest_count} REST tokens ({rest_ratio}% of events)')
    print(f'  => Discarded {too_short+too_long} event sequences')
    print(f'      - {too_short} too short')
    print(f'      - {too_long} too long')
    print(f'      - {too_manyinstr} too many instruments')
    print(f'  => Discarded {discarded_seqs} training sequences')
    print(f'  => Truncated {truncations} {trunc_type} times ({trunc_ratio}% of {trunc_type}s)')

    print('Remember to shuffle the training split!')

if __name__ == '__main__':
    main()
