# CLAUDE.md

Guidance for working in this repository. This is a research codebase for training and
evaluating an **anticipatory music transformer** that infills a symbolic *score* from an
expressive MIDI *performance* (ASAP dataset). It is a fork/extension of the Stanford
`anticipation` library, specialized to the packed score/performance "alternating" format.

> The repo is messy by design (many one-off experiment dirs and checkpoints). The four
> scripts below are the supported entry points; most other top-level files and the many
> `results_*`, `finale*`, `checkpoint-*`, `.marchsmoke*` directories are experiment
> artifacts and can be ignored.

## The four key scripts

| Script | Purpose |
| --- | --- |
| `tokenize-asap-sliding.py` | Build the packed token dataset from the ASAP MIDI corpus via a sliding window (90/10 train/test split by unique score). Writes `data/train_normalized.txt` / `data/test_normalized.txt`. |
| `train.py` | Fine-tune a causal LM on the packed sequences. Logs all metrics to **Weights & Biases**. |
| `inference.py` | Autoregressively decode scores from packed test windows, save MIDIs, report accuracy, and optionally run MUSTER. |
| `evaluate_muster.py` | Shared MUSTER (symbolic transcription error) helpers: model loading, MusicXML export, running the MUSTER C++ pipeline. Imported by `inference.py`. |

## Token / sequence format (read this first)

Everything revolves around the **packed alternating format** defined in
`anticipation/vocab.py`, `anticipation/config.py`, and `anticipation/packed_sequence.py`.

- A sequence is a flat list of integer tokens, **triplet-aligned**: every group of 3
  tokens is one event. **Token role = `index % 3`:**
  - `index % 3 == 0` → **onset / time** (`TIME_OFFSET` for score, `ATIME_OFFSET` for control)
  - `index % 3 == 1` → **duration** (`DUR_OFFSET` / `ADUR_OFFSET`)
  - `index % 3 == 2` → **pitch / note** (`NOTE_OFFSET` / `ANOTE_OFFSET`), or `REST`
- Times/durations are in **10 ms bins** (`TIME_RESOLUTION = 100` bins/sec).
- Two triplet kinds:
  - **Score triplet** (tokens `< CONTROL_OFFSET`): the symbolic score note to predict.
    A score triplet whose pitch is `REST` is a *dummy* placeholder, not a real note —
    these occur **only in the prefix**. Performance notes with no aligned score note
    are dropped at tokenization time, so every body score slot holds a real note
    (decoding constraints exclude `REST` accordingly).
  - **Control triplet** (tokens `>= CONTROL_OFFSET`, `!= SEPARATOR`): the conditioning
    performance note.
- Layout: a **prefix** of `PREFIX_CONTROLS = 32` (control, dummy-rest) pairs, then from
  `ALTERNATING_START = 32*2*3 = 192` the body **strictly alternates** score/control
  triplets (a score slot every 6 tokens). Packed length is `CONTEXT_SIZE - 4 = 1020`
  tokens (340 triplets, 138 score slots).
- Score onsets are normalized to a fixed **0.5 s beat grid**; performance/control times
  keep original tempo but are shifted to start at 0.

Use the helpers in `anticipation/packed_sequence.py` rather than re-deriving offsets:
`iter_score_slot_positions`, `is_real_score_triplet`, `is_score_triplet`,
`extract_packed_components`, `ALTERNATING_START`. Slot-decoding constraints live in
`anticipation/score_constraints.py` (`constrain_score_token_logits`).

## Data flow

```
ASAP MIDI (asap-dataset-master/)
   │  tokenize-asap-sliding.py  (anticipation.asap_aligned_stream + packed_sequence)
   ▼
data/train_normalized.txt, data/test_normalized.txt   # one packed sequence per line: "tok tok ... | "
   │  train.py
   ▼
<output_dir>/checkpoint-*/ , <output_dir>/final/       # HF model checkpoints; metrics -> wandb
   │  inference.py / evaluate_muster.py
   ▼
autoregressive_inference_results/ , muster_evaluation_results/   # MIDIs, stats.json, MUSTER scores
```

The token files are huge (GBs) and gitignored (`data/`). Each line is space-separated
tokens followed by ` | ` (an optional metadata separator; everything after `|` is ignored
when reading).

## Running

```bash
# Tokenize (writes data/*_normalized.txt). Heavy: 128 worker processes by default.
python tokenize-asap-sliding.py

# Train (single GPU). Metrics stream to wandb project "anticipation-asap".
python train.py --data_file data/train_normalized.txt --val_file data/test_normalized.txt \
    --output_dir ./run1 --wandb_run_name run1

# Multi-GPU
accelerate launch train.py --output_dir ./run1

# Disable wandb (e.g. offline box)
python train.py --wandb_mode disabled         # or: offline

# Autoregressive evaluation of a checkpoint (+ optional MUSTER)
python inference.py --checkpoint ./run1/checkpoint-2500 --num-examples 25 --compute-muster
```

Key training knobs (see `train.py` argparse for the full list): `--batch_size`,
`--gradient_accumulation_steps`, `--learning_rate`, `--max_steps`, `--eval_steps`,
`--save_steps`, augmentation (`--onset_jitter_std`, `--dur_jitter_range`, `--mask_prob`,
`--transpose_range_semitones`, `--tempo_scale_range`), and `--original_weight_l2` (L2
anchor to the pretrained weights).

## What `train.py` logs to wandb

All training/validation reporting goes through wandb (the old matplotlib `.png` and
`losses.npz` artifacts were removed; train.py no longer depends on matplotlib at all).

- **Line metrics**
  - `train/loss`, `train/learning_rate`, `train/anchor_l2`, `train/anchor_term`
  - `val/loss` and per-token-type validation losses `val/loss_onset`,
    `val/loss_duration`, `val/loss_pitch` (teacher-forced cross-entropy bucketed by
    `index % 3`)
  - `val/teacher_forced_pitch_accuracy`
  - **Autoregressive accuracy** for all three token types:
    `val/ar_pitch_accuracy`, `val/ar_onset_accuracy`, `val/ar_duration_accuracy`
- **Heatmaps** (`heatmaps/*`): native W&B heatmap charts (the built-in
  `wandb/heatmap/v0` Vega preset, logged via `wandb.plot_table`) with **training step
  (y)** vs **token / note index in the sequence (x)**, accumulated across validations:
  - `heatmaps/pitch_error_freq` — heat = *frequency* of autoregressive pitch errors at
    each score slot
  - `heatmaps/onset_mae` — heat = *mean absolute error* (in 10 ms bins) of the
    autoregressive onset at each slot
  - `heatmaps/duration_mae` — heat = MAE of the autoregressive duration at each slot
  - The x-axis index is the score-slot ordinal (the k-th predicted note); only real
    score notes contribute, so empty slots are masked (grey).

Validation has two phases (`evaluate_model`): a **teacher-forced** pass (loss + per-type
loss + pitch accuracy) and an **autoregressive** pass that greedily decodes score triplets
with ground-truth controls teacher-forced after each one (matching
`inference.autoregressive_generate_score`), then scores onset/duration/pitch against the
ground truth. Both phases reduce their statistics across ranks for multi-GPU correctness.

## The `anticipation/` package

- `config.py` — global constants (`CONTEXT_SIZE`, `TIME_RESOLUTION`, `MAX_*`, ...).
- `vocab.py` — token offsets (`TIME_OFFSET`, `DUR_OFFSET`, `NOTE_OFFSET`, `CONTROL_OFFSET`,
  `REST`, `SEPARATOR`, `VOCAB_SIZE = 55028`).
- `packed_sequence.py` — packed-format predicates/iterators (the canonical helpers).
- `score_constraints.py` — masks logits to the legal vocab range for a given slot.
- `asap_aligned_stream.py` — builds aligned performance/score triplet streams from ASAP
  MIDI + beat annotations (cached under `data/asap_aligned_stream_cache/`).
- `convert.py` — events ↔ MIDI/compound conversions (`events_to_midi`, `midi_to_events`).

## MUSTER

`evaluate_muster.py` shells out to C++ programs under `MUSTER/Programs/` (auto-compiled
from `MUSTER/Code/` on Linux via `g++`). It exports score triplets to MusicXML on the exact
token grid (splitting notes across barlines with ties), runs the score-to-performance
matcher + error detection, and parses error rates (PER/MNR/ENR/OTER/OFTER/MER/VER).
Requires `g++` available on Linux; on Windows the programs must be pre-compiled.

## Conventions / gotchas

- Prefer the `anticipation/packed_sequence.py` helpers; do not hand-roll offset math.
- The model vocab is resized to `VOCAB_SIZE` (55028) on load; the base checkpoint is
  `stanford-crfm/music-medium-800k`.
- `train.py` uses `accelerate` (bf16 on GPU). Don't manually `.to(device)` the model —
  let `accelerator.prepare` place it (critical for multi-GPU).
- Token files use ` | ` as a trailing separator; readers split on the first `|`.
- Times and durations everywhere are in 10 ms bins, not seconds.
