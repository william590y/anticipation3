# CLAUDE.md

Guidance for working in this repository. This is a research codebase for training and
evaluating an **anticipatory music transformer** that infills a symbolic *score* from an
expressive MIDI *performance* (ASAP dataset). It is a fork/extension of the Stanford
`anticipation` library, specialized to the packed score/performance "alternating" format.

> The repo is messy by design (many one-off experiment dirs and checkpoints). The scripts
> below are the supported entry points; most other top-level files and the many
> `results_*`, `finale*`, `checkpoint-*`, `.marchsmoke*` directories are experiment
> artifacts and can be ignored.

## The key scripts

| Script | Purpose |
| --- | --- |
| `tokenize-asap-sliding.py` | Build the packed token dataset from the ASAP MIDI corpus via a sliding window (90/10 train/test split by unique score). Writes `data/train_normalized.txt` / `data/test_normalized.txt`. Pass `--split-input <manifest> --val-output <path>` to use an external **three-way** train/validation/test split instead of the built-in seeded one (see "Paper split" below). |
| `make_paper_split.py` | Reproduce the external ASAP split shared by both reference papers; writes `data/paper_split.txt`. |
| `train.py` | Fine-tune a causal LM on the packed sequences. Logs all metrics to **Weights & Biases**. |
| `train_lora.py` | Same training loop, but wraps the resized base model with a PEFT LoRA adapter (`peft.get_peft_model`) instead of full fine-tuning. Checkpoints save only the adapter (`adapter_config.json` + `adapter_model.safetensors`); load with `PeftModel.from_pretrained(base_model, checkpoint_dir)` then `.merge_and_unload()`. `train_lora_highrank.sbatch` is the r=512 run (`run_nodummy_lora_r512`). |
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
- A window therefore holds **170 controls but only 138 score notes**. The 32 extra are
  the prefix's *lookahead*: **score slot `k` pairs with control `k`**, so it is the
  **trailing** 32 controls that have no score note, not the leading ones. (Verified on
  the real data: the ground-truth pitch sequence equals the *first* 138 control pitches,
  never the last 138.) Anything that lines a per-note prediction up against `gt_score`
  must take `perf_notes[:len(gt_score)]`.
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

## Paper split (external train/validation/test)

Besides the built-in seeded 90/10 split, the repo can train on the ASAP split used by
the two reference papers:

- Beyer & Dai, *End-to-end Piano Performance-MIDI to Score Conversion with Transformers*
  (ISMIR 2024) — `github.com/TimFelixBeyer/MIDI2ScoreTransformer`
- Zeng+, *Bridging Piano Transcription and Rendering via Disentangled Score Content and
  Style* (ICLR 2026) — `github.com/wei-zeng98/joint-apt-epr`

**These two splits are byte-identical** — joint-apt-epr copied Beyer's `constants.py`
verbatim ("From Beyer's paper"): same `TEST_PIECE_IDS`, `TO_IGNORE_INDICES`, `SKIP`, and
the same `_load_metadata` filtering pipeline. There is only one split to reproduce, so
there is no point training separate "paper 1" and "paper 2" models.

The split is defined over **ACPAS** metadata (`cheriell/ACPAS-dataset`, vendored to
`data/acpas/metadata_{R,S}.csv`) plus ASAP's `asap_annotations.json`, keyed by ACPAS
`piece_id`: test = `piece_id in TEST_PIECE_IDS` (first piece per composer), validation =
`piece_id % 10 == 0`, train = the rest. `make_paper_split.py` reproduces it into
`data/paper_split.txt`: **822 train / 86 validation / 59 test** performances
(176 / 16 / 14 unique pieces). ~100 of our performances fall outside it — the papers
drop those deliberately, so they are excluded from all three splits.

```bash
python make_paper_split.py                     # -> data/paper_split.txt
sbatch tokenize_asap_paper.sbatch              # -> data/{train,val,test}_paper.txt
sbatch train_asap_paper.sbatch                 # full FT  -> run_paper_split_v2
sbatch train_lora_highrank_paper.sbatch        # LoRA r512 -> run_paper_split_lora_r512
```

Both training jobs validate on the **validation** split; the test split is held out.
They run 2 GPUs on `thickstun` with `batch_size=4 x grad_accum=8 x 2` = the same
effective batch 64 as every other run — note that a 3-GPU layout **cannot** preserve it
(64 = 2^6 is not divisible by 3).

## MUSTER

`evaluate_muster.py` shells out to C++ programs under `MUSTER/Programs/` (auto-compiled
from `MUSTER/Code/` on Linux via `g++`). It exports score triplets to MusicXML on the exact
token grid (splitting notes across barlines with ties), runs the score-to-performance
matcher + error detection, and parses error rates (PER/MNR/ENR/OTER/OFTER/MER/VER).
Requires `g++` available on Linux; on Windows the programs must be pre-compiled.

## Visualizer

`visualizer/` is a self-contained (no server needed) HTML tool for inspecting a checkpoint's
autoregressive rollout against a handful of sampled validation windows.

- `visualizer/precompute_visualizer.py` runs real AR rollouts (greedy, with per-slot
  candidate/perplexity capture via `constrain_score_token_logits`) over `--num-examples`
  sampled test windows and writes a single `visualizer/data.js`
  (`window.VISUALIZER_DATA = {...}`) that `visualizer.html` loads directly from `file://`.
  Current payload is **format 4**: for both the filtered (model input) and raw/unfiltered
  (recovered from `data/asap_aligned_stream_cache/`, mistakes included) performance streams,
  it stores a plain AR rollout and a GT-seeded variant (slot 0 force-fed the true score note).
  Pass `--lora-checkpoint <adapter_dir>` to additionally compute all four rollouts through a
  PEFT LoRA adapter merged onto the base pretrained model (not onto `--checkpoint`), stored
  under `rollouts_lora`. Pass `--append-to <data.js>` to add `--num-examples` *new* windows
  to an existing file without touching or duplicating the windows already there; the
  resulting `data.js` carries an explicit `example_order` array (new windows always after
  old ones) since JS would otherwise silently reorder the plain-integer example keys
  ascending regardless of insertion order.
- `visualizer/precompute.sbatch` is the SLURM wrapper (GPU job; override `CHECKPOINT`,
  `OUTPUT`, `NUM_EXAMPLES`, `EXTRA_FLAGS` via `--export`).
- **Paper-split visualizer + external model comparison.** `visualizer/build_paper_viz.sbatch`
  rebuilds the whole visualizer on the paper split and is meant to be chained after the two
  training jobs (`sbatch --dependency=afterok:<ft>:<lora> ...`). It runs:
  `select_paper_windows.py` (12 validation + 12 test windows, de-duplicated by *musical work*
  — the splits hold only 16/14 unique scores but many performances each, so deduping by
  `.mid` path alone yields N pianists playing one sonata) → `precompute_visualizer.py` on both
  token files → `merge_paper_viz.py` (re-keys to stable `val-01`…`test-12` string keys; plain
  integer keys would be reordered by JS) → `run_paper_models.py` → `compute_f1.py`.
- `visualizer/run_paper_models.py` adds the two reference papers' own transcriptions
  (`rollouts_paper1` / `rollouts_paper2`) from their released weights in `external/weights/`.
  Both models are **note-aligned** (one score note per input performance note), so a window is
  sliced out of a full-piece transcription by matching our window's performance notes onto the
  model's input notes. It MUST run in the separate `paperpipe` conda env
  (`external/setup_env.sh`): paper 2 needs TimFelixBeyer's **music21 fork**, which would
  otherwise silently change `precompute_scores_xml.py`'s engraving behaviour in `base`.
  Decoding their score representation has four traps, all of which silently produce
  plausible-looking-but-wrong output rather than an error:
  1. **Both papers share one representation** (paper 1 copied paper 2's tokenizer, same
     `PARAMS`): measure-relative, `offset` = position within the measure, `downbeat` =
     the length of the measure that just ended (sentinel `-1/24` = "same measure").
     Absolute onset accumulates: `measure_start += downbeat` on each new measure.
  2. **Every stream is a bucket index, including `pitch`** — leaving it raw makes every
     note a semitone sharp, which drops F1 from ~27% to ~0.4% while still *looking* like
     a real transcription.
  3. They differ **only** in output shape and bucket origin: paper 1 returns indices
     `(B, T)` where index 0 is `<PAD>` (unbucket with `-1`), paper 2 returns per-class
     scores `(B, T, vocab)` needing an `argmax` and no shift. `_stream_indices` handles
     both.
  4. A model may emit `<PAD>` for an input note; those decode to pitch `-1` and are
     dropped, as their own decoders do via the pad mask.
  `visualizer/test_paper_decode.py` guards 1–3 by round-tripping a real
  `xml_score.musicxml` through paper 1's own `parse_mxl` → `bucket_mxl` → our decoder and
  checking we recover music21's absolute offsets (~1e-5 quarters), durations and pitches
  exactly — model quality plays no part, so any failure is our decoding.
  Output is then converted from quarters to our 50-bins-per-annotated-beat grid — the
  annotated beat is **not** always a quarter (6/8 is annotated at the dotted quarter) —
  and a window is *skipped* rather than displayed misaligned when the converted span
  disagrees with the ground truth's.
- `visualizer/compute_f1.py` **replaces `compute_muster.py`** in the paper-split visualizer.
  It scores every rollout's `pred_score` against the window's `gt_score` under three
  criteria — `onset_pitch`, `onset_pitch_dur`, `onset_pitch_tol1` (±1 bin) — with one-to-one
  matching so duplicate predictions cannot inflate the score, and writes `<rollout>.f1`.
  The UI's F1 panel shows one row per model (ours — following the LoRA toggle — plus paper 1
  and paper 2). `compute_muster.py` still exists for the older non-paper-split datasets.
- `visualizer/compute_muster.py` is a CPU-only post-process (no model/GPU needed — run after
  the GPU job) that reads an existing `data.js`, runs the real MUSTER pipeline
  (`evaluate_muster.py`'s `triplets_to_musicxml`/`run_muster_evaluation`) comparing each
  rollout's `pred_score` directly against the window's `gt_score`, and writes the result back
  into `data.js` as `rollouts*.<variant>.muster`. Deliberately does **not** reuse
  `evaluate_muster.py`/`inference.py`'s `normalize_triplet_times`: that helper independently
  re-anchors GT and predicted triplets to each side's own earliest onset, which is built for
  multi-window/full-piece evaluation but is unnecessary (and silently masks a wrong first-note
  prediction) for a single packed window: GT and pred already share one fixed, non-negative
  time origin fixed once per window at tokenization time (`min_score_time_units` in
  `tokenize-asap-sliding.py`) — the model predicts tokens directly on that same axis, it never
  establishes its own.
- `visualizer/precompute_scores_xml.py` is another CPU-only post-process: for each window it
  locates the window in its source piece (same pitch-bytes probe as the raw-stream recovery),
  converts the window's piece-level beat span to a measure range via the piece's
  `midi_score_annotations.txt` downbeats (the score beat grid is 50 bins per annotated beat),
  slices those measures out of the REAL `asap-dataset-master/**/xml_score.musicxml` with
  music21, and writes `visualizer/scores_xml.js`. Handles pickup measures, unannotated final
  measures, and real measures split across two XML measure elements; refuses (per piece) when
  the downbeat↔measure reconciliation fails rather than displaying misaligned bars. Each
  window also gets a `meter` block (notated time signature, annotated beats per measure from
  downbeat spacing, the window's phase offset within its first measure, key fifths, first
  measure number) — the annotated beat is NOT always a 4/4 quarter (6/8 pieces are annotated
  at the dotted quarter, so a 16th is 1/6 beat ≈ 8 bins). Real excerpts are sanitized of
  empty-`<words/>` directions (music21 artifact that crashes OSMD's createMetronomeMark).
- `visualizer.html` toggles: filtered/unfiltered stream, GT-seed-first-note, base/LoRA, a
  minimap of the whole packed sequence, a top-p + rank slider over candidate logits, audio
  playback, a live onset/duration/pitch accuracy panel, and a MUSTER panel (PER/OTER/OFTER/MER)
  for whichever rollout is currently selected. The "sheet music (engraved)" view uses vendored
  OpenSheetMusicDisplay (`visualizer/osmd.min.js`) to typeset the real MusicXML excerpt from
  `scores_xml.js` plus the selected rollout converted by `visualizer/sheet_xml.js` (a JS port
  of `evaluate_muster.triplets_to_musicxml`). Given the window's `meter` block the port snaps
  notes to a 1/12-beat grid and typesets them in the piece's real meter/key with barlines on
  the real downbeat phase; without one it falls back to the exact-grid 4/4 export, which is
  the mode verified byte-identical to the Python (the MUSTER evaluation path is untouched).
- The script only writes `data.js` (no `.json` sibling) — a stale duplicate JSON file caused
  real confusion once (branch-key offsets from an older script version), so don't reintroduce
  a second output file for this data.

## Conventions / gotchas

- Prefer the `anticipation/packed_sequence.py` helpers; do not hand-roll offset math.
- The model vocab is resized to `VOCAB_SIZE` (55028) on load; the base checkpoint is
  `stanford-crfm/music-medium-800k`.
- `train.py` uses `accelerate` (bf16 on GPU). Don't manually `.to(device)` the model —
  let `accelerator.prepare` place it (critical for multi-GPU).
- Token files use ` | ` as a trailing separator; readers split on the first `|`.
- Times and durations everywhere are in 10 ms bins, not seconds.
