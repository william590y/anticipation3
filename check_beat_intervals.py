import os
import pandas as pd
from alignment import load_annotation_file
import numpy as np

df = pd.read_csv(os.path.join('asap-dataset-master', 'metadata.csv'))
print('Checking beat intervals for first 10 pieces:')
print()

for idx in range(min(10, len(df))):
    row = df.iloc[idx]
    file4 = os.path.join('asap-dataset-master', row['midi_score_annotations'])
    
    if os.path.exists(file4):
        annotations = load_annotation_file(file4)
        beat_times = [a[0] for a in annotations]
        
        if len(beat_times) >= 5:
            intervals = [beat_times[i+1] - beat_times[i] for i in range(min(10, len(beat_times)-1))]
            
            print(f'Piece {idx}: {row["title"][:40]}')
            print(f'  First beat: {beat_times[0]:.3f}s')
            print(f'  First 10 intervals: {[f"{x:.3f}" for x in intervals]}')
            print(f'  Min interval: {min(intervals):.3f}s, Max interval: {max(intervals):.3f}s')
            print(f'  All uniform (within 1ms): {all(abs(x - intervals[0]) < 0.001 for x in intervals)}')
            print()
