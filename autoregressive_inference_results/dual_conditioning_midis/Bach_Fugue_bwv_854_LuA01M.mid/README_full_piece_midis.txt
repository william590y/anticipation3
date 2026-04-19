Two branches (compare generators side by side):

1) 1_packed_inference_opening_first_window/
   Same packed line as export_dual_conditioning_midis.py → 1_packed_inference_opening/
   (tokenize-asap-sliding start_idx=0 only; one window, not full-piece packed).
   inference.autoregressive_generate_score; GT is window-aligned score from the pack.
   If the model predicts only REST score tokens, output_score.mid is omitted (MIDI
   export would otherwise be empty after unpad).

2) 2_asap_native_full_piece_rollout/
   Same ASAP stack as dual branch 2, but full piece (no max_notes).
   input_performance.mid = all controls; GT = full score (REST omitted in MIDI).

GT MIDI differs between (1) and (2) on purpose (packed window vs full score).
Temporal span of (1) is the opening context only; (2) is the whole performance.

piece_index: 10
opening_body_slots: 137
gt_score_source: midi
ground_truth_score_tokens_to_feed: 1
ground_truth_score_notes_to_feed: 0
temperature: 0.0
num_controls_used: 739
num_window_resets: 5