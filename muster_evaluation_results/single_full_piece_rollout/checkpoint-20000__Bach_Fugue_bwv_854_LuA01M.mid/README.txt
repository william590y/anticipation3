Full-piece evaluate_muster_asap.py rollout (all performance notes,
KV window resets as in training eval).

piece_index: 10
perf_path: Bach/Fugue/bwv_854/LuA01M.mid
checkpoint: checkpoint-20000
gt_score_source: midi
ground_truth_score_notes_to_feed: 0
temperature: 0.0

Per-piece outputs: subdirectory named after this piece (safe_name).
MIDIs: ground_truth_score.mid, output_score.mid; MUSTER: muster_metrics.json, XMLs.