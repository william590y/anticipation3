import os
import pandas as pd
from alignment import load_annotation_file
import numpy as np

df = pd.read_csv(os.path.join('asap-dataset-master', 'metadata.csv'))
print(f'Checking beat intervals across ALL {len(df)} pieces in ASAP dataset')
print()

results = []

for idx, row in df.iterrows():
    file4 = os.path.join('asap-dataset-master', row['midi_score_annotations'])
    
    if os.path.exists(file4):
        try:
            annotations = load_annotation_file(file4)
            beat_times = [a[0] for a in annotations]
            
            if len(beat_times) >= 2:
                intervals = [beat_times[i+1] - beat_times[i] for i in range(len(beat_times)-1)]
                
                results.append({
                    'piece': idx,
                    'title': row['title'],
                    'min': min(intervals),
                    'max': max(intervals),
                    'mean': np.mean(intervals),
                    'std': np.std(intervals),
                    'median': np.median(intervals)
                })
        except:
            pass

print(f'Successfully analyzed: {len(results)} pieces')
print()

# Check uniformity
uniform_pieces = [r for r in results if r['std'] < 0.001]  # Within 1ms
nearly_uniform_05 = [r for r in results if abs(r['mean'] - 0.5) < 0.01 and r['std'] < 0.01]
non_uniform = [r for r in results if r['std'] >= 0.01]

print(f'Perfectly uniform (std < 0.001s): {len(uniform_pieces)}/{len(results)}')
print(f'Nearly uniform at 0.5s (mean ≈ 0.5s, std < 0.01s): {len(nearly_uniform_05)}/{len(results)}')
print(f'Non-uniform (std >= 0.01s): {len(non_uniform)}/{len(results)}')
print()

if non_uniform:
    print('NON-UNIFORM pieces found:')
    for r in non_uniform[:20]:
        print(f'  Piece {r["piece"]}: {r["title"][:40]}')
        print(f'    Min: {r["min"]:.3f}s, Max: {r["max"]:.3f}s, Mean: {r["mean"]:.3f}s, Std: {r["std"]:.4f}s')
else:
    print('ALL PIECES ARE UNIFORM!')

print()
print('Distribution of mean intervals:')
means = [r['mean'] for r in results]
print(f'  Minimum mean: {min(means):.3f}s')
print(f'  Maximum mean: {max(means):.3f}s')
print(f'  Overall mean: {np.mean(means):.3f}s')
print(f'  Median: {np.median(means):.3f}s')
print(f'  Std dev: {np.std(means):.4f}s')
print()
print(f'All pieces have mean ≈ 0.5s: {all(abs(m - 0.5) < 0.01 for m in means)}')
