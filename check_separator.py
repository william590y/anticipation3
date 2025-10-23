"""Check that SEPARATOR only appears at the beginning of each piece/augmentation"""
from verify_tokenization import _interleave_tokenize4_single
from anticipation.vocab import SEPARATOR
import pandas as pd
import os

df = pd.read_csv('./asap-dataset-master/metadata.csv')
row = df.iloc[0]

fg = (
    os.path.join('./asap-dataset-master', row['midi_performance']),
    os.path.join('./asap-dataset-master', row['midi_score']),
    os.path.join('./asap-dataset-master', row['performance_annotations']),
    os.path.join('./asap-dataset-master', row['midi_score_annotations'])
)

lines, _ = _interleave_tokenize4_single(
    fg, skip_Nones=True, prefix_controls=33,
    perturb_std_ms=50.0, mask_prob=0.5, num_augmentations=5
)

print(f"Total sequences: {len(lines)}")
print(f"Sequences per augmentation: {len(lines)//5}")
print()
print("SEPARATOR locations:")
print("-" * 60)

seqs_per_aug = len(lines) // 5
for i, line in enumerate(lines):
    count = line.split().count(str(SEPARATOR))
    if count > 0:
        aug_num = i // seqs_per_aug
        chunk_num = i % seqs_per_aug
        print(f"  Sequence {i:2d}: Aug {aug_num}, Chunk {chunk_num} -> {count} SEPARATORs")

print()
print("Expected: SEPARATOR should only appear in Chunk 0 of each augmentation")
print("          (i.e., at indices 0, 4, 8, 12, 16 for 5 augmentations)")
