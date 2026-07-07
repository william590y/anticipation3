# Score Infilling Visualizer

Interactive piano-roll comparison of predicted vs ground-truth scores, with
per-slot logit inspection from the model's own autoregressive rollout.

## Files

- `visualizer.html` — UI (loads `data.js`)
- `precompute_visualizer.py` — generates format-3 `data.js`
- `precompute.sbatch` — SLURM wrapper

## Usage

```bash
python visualizer/precompute_visualizer.py \
    --checkpoint run_nodummy_v2/checkpoint-15000 \
    --test-file data/test_normalized.txt \
    --num-examples 8 --output visualizer/data.js
```

## Format 3

Each example stores:

- `perf_notes` — filtered performance (model training input)
- `raw_notes` — unfiltered stream from aligned-stream cache (mistakes included)
- `gt_score` — ground-truth score on beat grid
- `rollouts.filtered` — AR rollout + per-slot candidates conditioned on **filtered** controls
- `rollouts.raw` — AR rollout + candidates conditioned on **raw** controls (mistakes shift the rollout)
- `rollouts.*.perplexity` — per score-slot onset/duration/pitch perplexity from the greedy AR path

The **stream** dropdown switches performance display **and** orange predictions/logits.
Defaults to unfiltered (raw).

## UI features

1. AR-conditioned logits (not teacher-forced GT walk)
2. Filtered / unfiltered stream toggle (dual rollouts in format 3)
3. Click a perf note → its quantized score slot (not next-triplet)
4. Minimap triplet bar (hover to locate)
5. Top-p slider + rank slider with hover highlight
6. Audio playback
7. Per-slot perplexity sparklines (onset / duration / pitch vs score-slot index)
