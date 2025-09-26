import argparse
import os
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

import mido


def count_notes_first_seconds(midi_path: str, seconds: float) -> int:
	"""Return the number of note-on events (velocity > 0) that start within the first `seconds` seconds.

	Uses mido to iterate over messages in playback order and sums delta times to compute absolute time.
	"""
	try:
		mid = mido.MidiFile(midi_path)
	except Exception:
		# Unreadable or corrupt file
		return 0

	t = 0.0
	count = 0
	# Iterate through all messages merged in playback order
	for msg in mid:
		t += float(msg.time)
		if t > seconds:
			break
		if msg.type == "note_on" and getattr(msg, "velocity", 0) > 0:
			count += 1
	return count


def main():
	parser = argparse.ArgumentParser(
		description="Compute average number of notes in the first N seconds across the ASAP dataset"
	)
	parser.add_argument(
		"--asap-root",
		default="./asap-dataset-master",
		help="Path to the ASAP dataset root folder (containing metadata.csv)",
	)
	parser.add_argument(
		"--seconds",
		type=float,
		default=5.0,
		help="Time window in seconds for counting notes (default: 5)",
	)
	parser.add_argument(
		"--which",
		choices=["performance", "score"],
		default="performance",
		help="Choose which MIDI to analyze from metadata.csv",
	)
	parser.add_argument(
		"--per-file",
		action="store_true",
		help="Print per-file counts in addition to summary statistics",
	)
	args = parser.parse_args()

	metadata_csv = os.path.join(args.asap_root, "metadata.csv")
	if not os.path.isfile(metadata_csv):
		raise FileNotFoundError(f"metadata.csv not found at {metadata_csv}")

	df = pd.read_csv(metadata_csv)

	column = "midi_performance" if args.which == "performance" else "midi_score"
	if column not in df.columns:
		raise KeyError(
			f"Column '{column}' not found in metadata.csv. Available columns: {list(df.columns)}"
		)

	midi_paths: List[str] = [os.path.join(args.asap_root, p) for p in df[column].tolist()]

	counts: List[int] = []
	missing: List[str] = []

	for path in tqdm(midi_paths, desc=f"Counting first {args.seconds:.1f}s notes ({args.which})"):
		if not os.path.isfile(path):
			missing.append(path)
			counts.append(0)
			continue
		c = count_notes_first_seconds(path, args.seconds)
		counts.append(c)

	counts_np = np.array(counts, dtype=float)
	total_files = len(midi_paths)
	analyzed_files = total_files - len(missing)

	print("\nSummary")
	print("-------")
	print(f"Dataset root: {os.path.abspath(args.asap_root)}")
	print(f"Analyzed: {analyzed_files}/{total_files} files (missing: {len(missing)})")
	print(f"Window: first {args.seconds:.2f} seconds")
	print(f"Source: {args.which} MIDI ({column})")

	if analyzed_files > 0:
		avg = float(np.mean(counts_np))
		med = float(np.median(counts_np))
		std = float(np.std(counts_np))
		p10, p90 = float(np.percentile(counts_np, 10)), float(np.percentile(counts_np, 90))
		mn, mx = int(np.min(counts_np)), int(np.max(counts_np))
		print(f"Average notes in window: {avg:.2f}")
		print(f"Median: {med:.2f}  Std: {std:.2f}")
		print(f"Percentiles: p10={p10:.1f}, p90={p90:.1f}")
		print(f"Min/Max: {mn}/{mx}")
	else:
		print("No files analyzed.")

	if args.per_file:
		print("\nPer-file counts:")
		for rel, c in zip(df[column].tolist(), counts):
			print(f"{rel}\t{c}")


if __name__ == "__main__":
	main()

