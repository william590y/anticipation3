ASAP-native full piece: same conditioning stack as
export_dual_conditioning_midis.py → 2_asap_native_opening_first_window/,
but autoregressive_generate_from_controls with no max_notes (whole piece).
Sibling folder 2_asap_native_opening_first_window/ is the truncated run.

input_performance.mid: all performance control triplets from preprocess_asap_piece.
ground_truth_score.mid: full normalized score (REST rows dropped for MIDI export).
output_score.mid: model rollout over the whole performance.

piece_index: 10
gt_score_source: midi
ground_truth_score_notes_to_feed: 0
temperature: 0.0
num_controls_used: 739
num_window_resets: 5