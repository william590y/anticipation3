# Score Infilling Visualizer

Interactive piano-roll comparison of predicted vs ground-truth scores, with
per-slot logit inspection from the model's own autoregressive rollout.

## Files

- `visualizer.html` — UI (loads `data.js`)
- `precompute_visualizer.py` — generates format-4 `data.js` (GPU; runs the AR rollouts)
- `compute_muster.py` — CPU-only post-process; adds MUSTER scores to an existing `data.js`
- `precompute_scores_xml.py` — CPU-only post-process; writes `scores_xml.js` (each window's
  measure span sliced from the piece's real ASAP `xml_score.musicxml`)
- `sheet_xml.js` — JS port of `evaluate_muster.triplets_to_musicxml` (predicted-rollout
  engraving); without a meter argument it is verified byte-identical (modulo whitespace)
  to the Python output, and with the per-window `meter` block from `scores_xml.js` it
  instead typesets the rollout in the piece's real meter/key (see UI feature 12)
- `osmd.min.js` — vendored OpenSheetMusicDisplay 1.8.9 (engraves MusicXML in the browser)
- `precompute.sbatch` — SLURM wrapper for `precompute_visualizer.py`

## Usage

```bash
python visualizer/precompute_visualizer.py \
    --checkpoint run_nodummy_v2/checkpoint-15000 \
    --test-file data/test_normalized.txt \
    --num-examples 24 --output visualizer/data.js \
    --lora-checkpoint run_nodummy_lora_r512/checkpoint-10000

python visualizer/compute_muster.py --data visualizer/data.js
python visualizer/precompute_scores_xml.py --data visualizer/data.js   # enables engraved view
python visualizer/compute_sequence_ppl.py --data visualizer/data.js    # gen vs GT sequence PPL
```

`compute_sequence_ppl.py` is a GPU post-process: for each of our rollouts it reports
sequence perplexity of the **generated** score stream vs the **ground-truth** score
stream under the model (teacher-forced controls; exp(mean NLL) over score
onset/duration/pitch tokens), plus per-slot Shannon **entropy** / **log-entropy** of
the constrained predictive distributions along the generated walk (used by the
"color pred by log-entropy" checkbox). Results land on
`rollouts*.<variant>.sequence_perplexity`, `.entropy`, and `.log_entropy`.
Or via SLURM:

```bash
sbatch visualizer/compute_sequence_ppl.sbatch
```


`--lora-checkpoint` is optional; omit it to skip the LoRA rollouts entirely (faster). The
adapter is merged onto the base pretrained model (`stanford-crfm/music-medium-800k`, the same
starting point `train_lora.py` uses), not onto `--checkpoint`'s fine-tuned weights.
`run_nodummy_lora_r512/checkpoint-10000` is the highest-AR-pitch-accuracy checkpoint (93.78%,
per wandb run `run_nodummy_lora_r512`) from the r=512 LoRA run.

To add more windows without disturbing the ones already in a `data.js` (and without
duplicating any of them), pass `--append-to`:

```bash
python visualizer/precompute_visualizer.py \
    --checkpoint run_nodummy_v2/checkpoint-15000 \
    --lora-checkpoint run_nodummy_lora_r512/checkpoint-10000 \
    --num-examples 16 --append-to visualizer/data.js --output visualizer/data.js
```

`--num-examples` in `--append-to` mode means "how many *new* windows to add" — the windows
already in the target file are excluded from resampling (so no duplicates) and are kept
exactly as they are. New windows are always appended *after* the existing ones in the
dropdown: since JS reorders plain-integer object keys ascending regardless of insertion
order, `data.js` also carries an explicit `example_order` array that `visualizer.html`
reads for display order (falling back to `Object.keys(examples)` for older files that
predate this field).

`compute_muster.py` runs after the GPU job finishes — it needs no model, just `gt_score`/
`pred_score`, so it's a fast (~30s for 24 windows x 8 rollout variants) CPU pass that reads
and rewrites `data.js` in place, adding `rollouts*.<variant>.muster`. It intentionally does
**not** reuse `evaluate_muster.py`/`inference.py`'s `normalize_triplet_times` (built for
multi-window/full-piece evaluation, where GT and predicted triplets get re-anchored
independently to each side's own earliest onset — harmless when both sides' earliest onset
already agrees, but it silently masks a genuinely wrong first-note prediction and smears a
phantom shift across every other note when they don't). For a single packed window, GT and
predicted score notes already share one fixed, non-negative time origin from tokenization
(`tokenize-asap-sliding.py`'s per-window `min_score_time_units`), so `compute_muster.py` just
exports both directly with no re-anchoring at all.

## Paper split, external model comparison, and F1

`build_paper_viz.sbatch` rebuilds this visualizer on the external paper split
(`make_paper_split.py`), showing 12 validation + 12 test windows from 24 distinct
musical works, and adds the two reference papers' own transcriptions from their
released weights alongside ours. MUSTER has been replaced by note-level **F1**
(`compute_f1.py`, three matching criteria). Per-window scores are stored in `data.js`;
the visualizer table reports their piece-macro mean over the common 24-piece set,
using unfiltered performance input with no GT seeding for all three models.

The 0.48 s/beat beat-grid toggle was removed: there are no 0.48 s models on the paper
split, so the dropdown had nothing to switch to.

```bash
python make_paper_split.py
sbatch tokenize_asap_paper.sbatch
sbatch --dependency=afterok:<ft_job>:<lora_job> visualizer/build_paper_viz.sbatch
```


## Format 4

Each example stores:

- `perf_notes` — filtered performance (model training input)
- `raw_notes` — unfiltered stream from aligned-stream cache (mistakes included)
- `gt_score` — ground-truth score on beat grid
- `rollouts.filtered` — AR rollout + per-slot candidates conditioned on **filtered** controls
- `rollouts.raw` — AR rollout + candidates conditioned on **raw** controls (mistakes shift the rollout)
- `rollouts.filtered_seeded` / `rollouts.raw_seeded` — same, but the ground-truth score note
  for slot 0 (the window's first note) is dropped in as context instead of the model's own
  guess, so the rest of the rollout runs conditioned on the true first note
- `rollouts.*.perplexity` — per score-slot onset/duration/pitch perplexity from the greedy AR path
- `rollouts.*.entropy` / `rollouts.*.log_entropy` — Shannon entropy (nats) and
  `log(H+eps)` of the constrained predictive distribution at each score token along
  the generated walk; `log_entropy.triplet = log(H_t+H_d+H_p+eps)` drives the heatmap
- `rollouts.*.sequence_perplexity` — sequence PPL of the generated score stream vs the
  ground-truth score stream under the model (from `compute_sequence_ppl.py`)
- `rollouts_lora.*` — the same four rollouts, but from the LoRA-adapted model (absent if
  `--lora-checkpoint` wasn't given)

The **stream** dropdown switches performance display **and** orange predictions/logits.
Defaults to unfiltered (raw). The **seed first note (GT)** checkbox switches between the
plain and GT-seeded rollout for whichever stream is selected. The **LoRA** checkbox switches
between the base and LoRA-adapted rollouts.

## UI features

1. AR-conditioned logits (not teacher-forced GT walk)
2. Filtered / unfiltered stream toggle (dual rollouts)
3. Click a perf note → its quantized score slot (not next-triplet)
4. Minimap triplet bar (hover to locate)
5. Top-p slider + rank slider with hover highlight
6. Audio playback
7. Per-slot perplexity sparklines (onset / duration / pitch vs score-slot index)
8. Seed-first-note toggle (drop in the GT note for slot 0, see how the rollout recovers)
9. Rollout accuracy panel (onset / duration / pitch % over the whole window, top right)
10. Base / LoRA toggle
11. Sequence PPL panel (generated vs ground-truth under the model) + optional
    log-entropy heatmap coloring of predicted triplets
12. Mean F1 table (note-level F1 vs GT across models)
13. Engraved sheet view (view → "sheet music (engraved)"): OSMD-typeset REAL sheet music —
    the window's measure span sliced out of the piece's actual ASAP `xml_score.musicxml`
    by `precompute_scores_xml.py` — plus the selected AR rollout typeset via
    `sheet_xml.js`. The window→measure mapping goes through the piece's
    `midi_score_annotations.txt` downbeats (the aligned stream's beat grid is 50 bins per
    annotated beat), with split XML measures merged by duration and pickup/trailing
    unannotated measures handled; pieces where the mapping cannot be validated are skipped
    (the pane says so) rather than shown misaligned. Note the real score legitimately shows
    FEWER notes than the GT slots where trills/tremolos are notated as ornament signs but
    rendered as separate notes in the score MIDI the stream was aligned against.

    The rollout pane engraves in the window's REAL meter, not a hardcoded 4/4:
    `precompute_scores_xml.py` also emits a per-window `meter` block (notated time
    signature active over the window's measures, annotated beats per measure from the
    downbeat spacing, the window's phase offset within its first measure — windows anchor
    to their first matched score note, not to a barline — plus key fifths and the first
    measure number). This matters because the annotated beat is not always a 4/4 quarter:
    6/8 and 12/8 pieces are annotated at the dotted quarter (a 16th is 1/6 beat ≈ 8 bins),
    Chopin Ballade No. 1's 6/4 at the dotted half — under an assumed 4/4 every duration in
    such pieces becomes a nonstandard note value. With a meter block `sheet_xml.js` snaps
    onsets/durations to a 1/12-beat grid (which represents both duple and triple beat
    subdivisions exactly, healing the ±1-bin quantization jitter of e.g. 12/13-bin 16ths)
    and emits real barlines/measure numbers aligned with the real-score pane above it.
    Windows without a validated meter (non-uniform downbeat spacing or time signature
    inside the window) fall back to the old exact-grid 4/4 export, which is also the mode
    verified byte-identical to `evaluate_muster.triplets_to_musicxml` (the MUSTER export
    path itself is untouched — evaluation numbers are unaffected by all of this).
    `precompute_scores_xml.py` additionally strips `<direction>` elements whose only
    content is an empty `<words/>` from the real excerpts (music21 export artifact for
    bare tempo `<sound>` carriers) — OSMD's createMetronomeMark crashes on them.
