Two conditioning / generation branches for the same piece.

1) 1_packed_inference_opening/
   Input + GT score: extracted from the opening PACKED token window
   (same object as tokenize-asap-sliding.py start_idx==0).
   Output: inference.autoregressive_generate_score (inference.py path).

2) 2_asap_native_opening_first_window/
   Input: first N performance control triplets from preprocess_asap_piece,
   N = min(len(controls), 170) (opening window control span).
   GT score: full normalized score from MIDI (REST notes removed for export),
   as used by evaluate_muster_asap.py / MUSTER-style evaluation.
   Output: autoregressive_generate_from_controls, same as evaluate_muster_asap,
   stopped after max_notes=137 (same body length as opening packed window).

GT MIDI intentionally differs: (1) aligned window score vs (2) full score grid.