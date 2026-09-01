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
| `inference.py` | Autoregressively decode scores from packed test windows, save MIDIs, report accuracy, and optionally run MUSTER. Also exports `batched_autoregressive_generate_score` (KV-cached, whole-batch greedy rollout — the canonical AR decode used by `train.py` validation and the eval scripts). (`evaluate_test_combined_autoregressive.py` is just a backward-compatible shim that calls `inference.main`.) |
| `evaluate_muster.py` | Shared MUSTER (symbolic transcription error) helpers: model loading, MusicXML export, running the MUSTER C++ pipeline. Imported by `inference.py`. |
| `evaluate_muster_asap.py` | Full-piece "fair" MUSTER evaluation on ASAP with performance-only conditioning (multiprocessing + plots), as opposed to `inference.py`'s per-window eval. |
| `train_imitation.py` | Scheduled-sampling / DAgger-style variant of `train.py` (imports it as `base_train`): every `--il_rollin_interval` optimizer steps it rolls in the model's own (constrained) score predictions for a fraction of each batch (`--il_sequence_fraction`), mixing in expert tokens with probability decaying from `--il_teacher_prob_start` to `--il_teacher_prob_end`, and trains on those states. Default output `./imitation_learning`. |
| `packed_dataset.py` | Transformers-free **copy** of `train.py`'s `TokenizedDataset`/`iter_sequence_triplets` for the plan/LTLM family (`train_plan_lm.py`, `train_plan_vq.py`, plan probes, `test_ltlm.py`). `train.py` does **not** import it — its own in-file copy is canonical for training and has already diverged (extra `offsets=`/`sequence_length=` ctor params). Port augmentation fixes to both. |
| `eval_base_score_ppl.py` | GT-score vs generated-score perplexity under the **untuned** base model (see "Base-model score perplexity" below). `submit_base_score_ppl.sbatch` shards it 8-ways over thickstun. |
| `posttrain_common.py` + `train_{grpo,crpo,ppo,ppo_f1,onpolicy_distill}.py` | RL / post-training arms (see section below). |
| `train_ltlm.py` | Latent Thought Model arm (see section below). |
| `train_plan_vq.py` / `train_plan_lm.py` | Two-stage discrete-plan arm (see section below). |
| `nbest/` | N-best pool generation + the selection-method family (rerankers, duel knockout, ListT5-FiD, GenRM) and the draft-and-verify decode experiments (see "N-best selection" and "Draft-and-verify decode" below). |

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
  must take `perf_notes[:len(gt_score)]`. **Scope: this pairing holds only for the
  tokenizer's filtered data.** The `data/{train,val}_paper_unfiltered.txt` files
  (see "Data variants") deliberately repack windows with the raw, mistakes-included
  performance stream, so there slot `k`'s aligned control sits at a shifted,
  content-dependent index.
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
when reading). The built-in seeded split also writes its piece membership to
`data/normalized_split.txt` (1067 pieces: 943 train / 124 test).
(`data/test_normalized_prev.txt` is a stale pre-nodummy orphan with no consumers.)

### Data variants

- **Paper split** (`data/{train,val,test}_paper.txt`) — see "Paper split" below.
- **b048** (`tokenize_asap_b048.sbatch`): retokenized at `--beat-interval 0.48` with a
  **dedicated** `--cache-dir data/asap_aligned_stream_cache_b048`, writing
  `data/{train,test}_normalized_b048.txt` + `normalized_split_b048.txt` (same 90/10
  membership as 0.5s — the split is a pure function of seed 42). The aligned-stream
  cache filename is keyed on the perf-MIDI path only (beat interval lives in the
  fingerprint, not the name), so sharing one cache dir across intervals silently
  overwrites entries; `tokenize-asap-sliding.py` refuses a non-default beat interval
  with the default cache dir. The cache fingerprint includes `mtime_ns` of the inputs —
  touching ASAP files forces hours of realignment instead of the fast cache-hit pass.
- **Unfiltered controls** (`build_unfiltered_windows.py` via
  `build_unfiltered_data.sbatch`): paper-split windows repacked with the RAW performance
  stream (mistakes included) → `data/{train,val}_paper_unfiltered.txt`, subsampled
  (stride 20/5, capped 150k/8k). Breaks the slot-`k`↔control-`k` pairing (see above).
  Consumed by the PPO-F1 arm. Imports window-location helpers from
  `visualizer/precompute_visualizer.py`, so renames there break this build.

The alignment layer feeding tokenization: top-level `alignment.py` (`align_tokens2`:
pitch-exact greedy match of each performance note to a score note within 0.1s of the
beat-interpolated time) is imported by `anticipation/asap_aligned_stream.py` as
`from alignment import ...` — the repo root must be on `sys.path` (run from the repo
root). If you change alignment logic, bump `_ALIGN_TOKENS2_CACHE_VERSION` (in-process)
AND `STREAM_PREPROCESS_VERSION` (on-disk cache) or stale alignments persist.

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

### Supervised sbatch → run-dir map

| sbatch | data | run dir | note |
| --- | --- | --- | --- |
| `train_asap.sbatch` | 90/10 normalized | `run_nodummy` | original run |
| `train_asap_v2.sbatch` | 90/10 normalized | `run_nodummy_v2` | clean rerun (v1 collided with another job's output_dir) |
| `train_asap_v2_b048.sbatch` | b048 variant | `run_nodummy_v2_b048` | |
| `train_lora_highrank.sbatch` | 90/10 normalized | `run_nodummy_lora_r512` | LoRA r=512 |
| `train_lora_highrank_b048.sbatch` | b048 variant | `run_nodummy_lora_r512_b048` | |
| `train_asap_paper.sbatch` | paper split | `run_paper_split_v2` | init checkpoint for all RL arms |
| `train_lora_highrank_paper.sbatch` | paper split | `run_paper_split_lora_r512` | |
| `train_asap_paper_masked.sbatch` | paper split | `run_paper_split_v2_masked` | exactly one diff from `run_paper_split_v2`: `--loss_mask_performance_tokens` (score-only loss; the LTLM-comparable baseline) |
| `train_asap_paper_masked_40k.sbatch` | paper split | `run_paper_split_v2_masked_40k` | 40k-step version of the above |

Key training knobs (see `train.py` argparse for the full list): `--batch_size`,
`--gradient_accumulation_steps`, `--learning_rate`, `--max_steps`, `--eval_steps`,
`--save_steps`, augmentation (`--onset_jitter_std`, `--dur_jitter_range`, `--mask_prob`,
`--transpose_range_semitones`, `--tempo_scale_range`), `--original_weight_l2` (L2
anchor to the pretrained weights), `--loss_mask_performance_tokens` (score-only loss),
`--compile` / `--compile_mode`, and `--attn_implementation` (default sdpa).

## What `train.py` logs to wandb

All training/validation reporting goes through wandb.

- **Line metrics**
  - `train/loss`, `train/learning_rate`, `train/anchor_l2`, `train/anchor_term`
  - `val/loss` and per-token-type validation losses `val/loss_onset`,
    `val/loss_duration`, `val/loss_pitch` (teacher-forced cross-entropy bucketed by
    `index % 3`)
  - `val/teacher_forced_pitch_accuracy`
  - **Autoregressive accuracy** for all three token types:
    `val/ar_pitch_accuracy`, `val/ar_onset_accuracy`, `val/ar_duration_accuracy`
- **Heatmaps** (`heatmaps/*`): rendered by `build_error_heatmap_chart` (train.py) with
  **matplotlib** (Agg) — the native `wandb/heatmap/v0` Vega preset it briefly used did
  not render reliably. PNGs are saved to `<output_dir>/heatmaps/<metric>.png` and the
  same figures logged as `wandb.Image`. **Training step (y)** vs **score-slot index
  (x)**, accumulated across validations:
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

## Base-model score perplexity (`eval_base_score_ppl.py`)

Answers: under the **untuned** base model (`stanford-crfm/music-medium-800k`, vocab
already 55028 — the script refuses to run if a resize would be needed), which has
higher perplexity, the ground-truth scores or the scores the base model itself
generates from the performance?

Protocol: greedy packed-format conditional rollout per window (the canonical
`batched_autoregressive_generate_score`, `ground_truth_score_tokens_to_feed=0`), then
teacher-forced NLL of each side's **score tokens alone** — flattened triplets primed
with the `AUTOREGRESS` mode flag, which is exactly how the base model's pretraining
marks control-free sequences (`anticipation/tokenize.py`). Reports slot-constrained and
unconstrained variants, plus a secondary in-context (packed) NLL of the same tokens.
Shard over GPUs with `submit_base_score_ppl.sbatch` (array 0-7 on thickstun; ~0.33
s/window/GPU at batch 96), merge with `--merge 'base_score_ppl_results/*_shard*.json'`.

**Result (2026-08-18, full val+test paper split, 448k windows):** the GENERATED scores
have far **higher** score-only perplexity on both splits — val GT 43.95 vs gen 1692.9,
test GT 34.95 vs gen 1660.2 (constrained; unconstrained agrees: 297.5/231.0 vs ~8270) —
GT higher in ≈0% of windows. The **in-context** comparison flips (GT ~2000 vs gen ~270
constrained): the base model prefers its own greedy continuation in context, but as
standalone scores its generations are wildly out-of-distribution (onset PPL ~11k vs
~100-270 for real scores). `visualizer/compute_sequence_ppl.py` is the related
teacher-forced-controls comparison under a *fine-tuned* checkpoint.

## RL / post-training arms

On-policy post-training of the best supervised paper-split checkpoint to close the
measured exposure-bias gap (teacher-forced score CE 0.266 vs 10.76 along the model's own
greedy rollout). **Every arm starts from `run_paper_split_v2/checkpoint-2500`**
(`posttrain_common.DEFAULT_INIT_CHECKPOINT`) and shares `posttrain_common.py`
(`PostTrainer`: bf16 accelerate, cosine LR with floor, L2 anchor default 1e5, fixed
96-window AR validation, `ar_val_loss.csv`/`.png`) + `onpolicy_rollout.py`
(`rollout_score_slots`: prefix+controls teacher-forced, only the 138 score triplets
sampled; 414 KV-cached forwards per window, kernel-launch bound → batch rollouts big,
chunk only the update pass).

| arm | reward | sbatch → run dir | status |
| --- | --- | --- | --- |
| `train_onpolicy_distill.py` | dense GT-CE on own rollout | `train_onpolicy_distill.sbatch` → `run_onpolicy_distill` | KILLED — collapses at every lr (CE is tail-dominated; model flattens its distribution) |
| `train_grpo.py` | seq accuracy-sum ∈ [0,3] (`--reward accuracy`) | `train_grpo.sbatch` → `run_grpo_acc_reward` | reward climbs, TF loss intact; best saved `checkpoint-250` (1.584) |
| `train_crpo.py` + `crpo.py` | GRPO + contrastive InfoNCE vs GT-prefix teacher (λ/τ rescaled 2.0/10.0 — see sbatch header) | `train_crpo.sbatch` → `run_crpo_acc_reward` | as GRPO |
| `train_ppo.py` + `ppo.py` | token-level ±1 + ValueHead/GAE | `train_ppo.sbatch` → `run_ppo`; `train_ppo_production.sbatch` (env `PPO_RUN_DIR`) → `run_ppo_corrected_*` | best: `run_ppo_corrected_20260814_020654_2364547/best-val-reward` (1.620; actual hparams γ=0 λ=0 lr 3e-7 in its `selected_best_val.json` — read the manifest, not the sbatch) |
| `train_ppo_f1.py` + `f1_reward.py` | Δ-F1 per triplet (onset_pitch_tol1) − KL-in-reward; trains on `*_paper_unfiltered.txt` | `train_ppo_f1.sbatch` (env `RUN_DIR`) → `run_ppo_f1_triplet_*` | val F1 39.9% → 45.6% (step 4900, `best-val-f1/`) |

Key facts:
- `val/REWARD` (ALL-CAPS = wandb-findability convention) means the [0,3] accuracy-sum
  for GRPO/CRPO/token-PPO but a [0,1] F1 for PPO-F1. Training rollouts are sampled
  (T=1.0 GRPO/CRPO, T=0.7 PPO); validation is always greedy — the two REWARD series are
  not comparable across that divide.
- Headline negative result: the exact-match accuracy reward does **not** improve
  note-level F1 (GRPO 17.2/10.4/21.0 vs base 18.9/11.1/22.7 on the visualizer table —
  regenerate via `visualizer/rl_f1_table.py` + `render_f1_table.py`); PPO-F1 exists to
  optimize the table's own criterion, and does lift it.
- Refactor invariants (each guards a measured failure): validation shards whole
  *batches* across ranks, never items (batched-matmul reduction order flips greedy
  argmaxes; item-sharding made 3- vs 4-GPU runs disagree); the model stays in `eval()`
  during training so the first-epoch importance ratio is exactly 1
  (`--dropout_in_update` is the only opt-in exception; `probe_dropout_ratio.py` is the
  PPO diagnostic for it, despite its plan-probe-like name); during PPO critic warmup
  the DDP-wrapped policy must never be invoked; `train_ppo_f1.py` must keep setting
  `trainer.last_validated_step`; checkpoints save every 250 steps but validation runs
  every 25 — only the PPO arms keep rolling `best-val-*` dirs.
- Dead-arm post-mortems (neg_ce divergence, distill collapse, CRPO λ/τ rescale
  rationale) live in the **sbatch headers** — read them before re-deriving.
- `plot_ppo_f1_reward.py` parses the SLURM log + `val_f1.csv` for reward-trend stats;
  `submit_ppo_viz_chain.sh` chains production PPO → best-checkpoint selection →
  visualizer publish; `ppo_ellis_probes.sbatch` is the 150-step hparam probe array
  (→ `debug/ppo_ellis/`).

## Latent Thought Model (LTLM) arm

GPT-2 decoder conditioned on continuous latent thoughts `z` (B, 24 layers × 4
thoughts, 1024-d) injected via per-layer residual cross-attention
(`ThoughtInjectedBlock`). Objective `L = E_q[-log p_θ(S|P,z)] + β·D(q(z|P,S), p_φ(z|P))`
with a per-window AdamVI fast loop (16 AdamW steps on μ_q/log σ_q, decoder frozen)
and a slow decoder+planner step; `p_φ` is `--planner diffusion` (default; DDPM bound,
fp32, DDIM init) or `gaussian` (analytic KL). ~17 GPT-2-medium passes per micro-batch
(`--mcmc_steps` is the cost knob).

- **Current implementation** = `anticipation/ltlm_{model,posterior,objective,diffusion,eval}.py`
  driven by `train_ltlm.py`. Top-level `ltlm.py`, `ltlm_diffusion.py`, `test_ltlm.py`
  are the **superseded untracked first generation** (old trainer in
  `.cornell_ltlm_backup/`; produced `run_ltlm`, `run_ltlm_kl{20,30,96}`) — don't extend
  them; `test_ltlm.py` no longer even runs to completion (imports names the current
  trainer renamed). Use `pytest tests/test_ltlm_*.py`.
- Validation arms: **oracle** (AdamVI on full labels; `val/oracle/loss` aliased to
  `val/loss`), **prefix** (AdamVI observing only the 170 control tokens — q(z|P); the
  name is historical, it is NOT the 192-token packed prefix), **planner** (z from
  50-step DDIM, no MCMC).
- **sbatch jobs do not run the working tree**: `train_ltlm.sbatch` (diffusion) and
  `train_ltlm_planner.sbatch` (gaussian; must pass `--planner gaussian`) clone LOCAL
  `main` into `~/anticipation3-diffusion` / `~/anticipation3-planner` (data/
  symlinked). Uncommitted work never reaches a job; never clone origin (its main still
  has the DDP desync). Current-gen outputs live in those external checkouts
  (`run_ltlm_diffusion`, `run_ltlm_planner`; wandb ri7dgjs2, backfilled by
  `backfill_ltlm_wandb.py` — wandb silently drops `wandb.log(step=N)` behind a live
  run's `_step`, hence its `checkpoint_step` step-metric + socket-attach upload).
- DDP/compile rules (asserted by `tests/test_ltlm_ddp.py`, each a real dead job): the
  slow step must call the DDP-wrapped `model(...)` (forward dispatches to `elbo` when
  `mu_q` is passed) — unwrapped `.elbo` skips gradient all-reduce (job 69260); use
  `train_ltlm.unwrap_ddp`/`compiled_root`, not `accelerator.unwrap_model` (KeyError on
  DDP(OptimizedModule), job 81476); compile only `model.base_model` (job 81490);
  AdamVI inside `@no_grad` validation needs its `torch.enable_grad()` wrapper.
- Checkpoint format is NOT plain HF: `checkpoint-N/model.safetensors` holds the
  wrapped trunk + cross-attn, `ltlm_extra.pt` the planner/args/step. Load only via
  `anticipation.ltlm_eval.load_ltlm_checkpoint`. No `final/` is ever written.
- Piano-roll rendering delegates to `plan_vq.split_packed_batch` and
  `plan_vq_viz.piano_roll_figure` — refactoring plan_vq*.py breaks LTLM validation
  (lazy imports).

## Plan-conditioning arms (plan_vq / plan_lm)

Two-stage test of whether a short discrete "plan" (K=8 codes from a 512-entry VQ
codebook, ~72 bits — in practice score *timing*; pitch is a near-free copy of the
aligned control) helps the LM infill the score.

- **Stage 1** `plan_vq.py`/`train_plan_vq.py`: ~34M VQ autoencoder — frozen pretrained
  PianoBART (`pianobart_encoder.py`, vendored `external/pianobart/`) encodes the score,
  learned queries pool to `--num_codes` EMA-quantized latents, a performance-conditioned
  decoder rebuilds the score. Selects `plan_vq_best.pt` on `val/onset_plan_gain`
  (own-plan minus neighbour's-plan onset accuracy; the 20260814_192509 run: +42 pts).
  100% pitch reconstruction is expected, not success.
- **Stage 2** `plan_lm.py`/`train_plan_lm.py`: `run_paper_split_v2` recipe + plan
  prefix; codes computed **online** per batch (augmentation changes the window — never
  precompute). Plan tokens live above the vocab (`PLAN_CODE_OFFSET=55028`) with their
  own `wpe` rows past 1024 while the window keeps positions 0..1019.
- Invariants (guarded by `test_plan_pipeline.py`): never recompute positions
  contiguously; an explicit all-ones `attention_mask` is load-bearing (with `None`,
  transformers reads the out-of-band position_ids as a packed-sequence document
  boundary and silently severs the plan — loss looks normal); vocab widening must use
  `resize_token_embeddings(..., mean_resizing=False)` + manual init (default
  mean-resizing makes plan rows ~265× too small) and rebuild each attention block's
  cached causal-bias buffer after extending wpe.
- Findings: `--plan_placement front` (the spec) makes the plan unpredictable at
  inference (LM sees no performance at position 0 → learns the marginal prior;
  `val/plan_code_accuracy` ~1%, ceiling via `probe_plan_prior_ceiling.py`), and
  oracle-plan vs own-plan AR accuracy was near-identical (onset 16.47% vs 16.39%) —
  the plan added almost nothing. `after_prefix` conditions on the 32 lookahead
  controls instead. Stage-2 `val/loss` includes plan tokens — compare `val/loss_packed`
  against `run_paper_split_v2`, and `val/plan_onset_gap` is the metric the scheme
  lives or dies by.
- Run dirs: `run_plan_vq_20260814_185416` = crashed (no checkpoints);
  `run_plan_vq_20260814_192509` = completed stage 1 (use its `plan_vq_best.pt`);
  `run_plan_lm_20260814_192509` = partial stage 2 (cancelled ~step 3419/20000, only
  `checkpoint-2500`). `submit_plan_chain.sh` chains both stages (env-driven).
- `probe_plan_conditioning.py` (GPU) tests whether the trained LM actually conditions
  on the plan; `probe_plan_influence{,2}.py` are CPU toy plumbing checks that run at
  import.

## MUSTER

`evaluate_muster.py` shells out to C++ programs under `MUSTER/Programs/` (auto-compiled
from `MUSTER/Code/` on Linux via `g++`). It exports score triplets to MusicXML on the exact
token grid (splitting notes across barlines with ties), runs the score-to-performance
matcher + error detection, and parses error rates (PER/MNR/ENR/OTER/OFTER/MER/VER).
Requires `g++` available on Linux; on Windows the programs must be pre-compiled.

## Visualizer

`visualizer/` is a self-contained (no server needed) HTML tool for inspecting a checkpoint's
autoregressive rollout against a handful of sampled validation windows.
**`visualizer/README.md` is the authoritative, up-to-date doc for this directory** — the
section below covers the core pipeline; the directory has since grown many publish/compare
pipelines not detailed here: `compute_sequence_ppl.py` (+`.sbatch`; gen-vs-GT sequence PPL
and per-slot entropy under a fine-tuned checkpoint, feeds the "color pred by log-entropy"
toggle), `compare_ckpt_f1.py`, `rl_f1_table.py`/`render_f1_table.py` (the RL F1 results
table), `select_ppo_best.py`/`precompute_ppo`/`publish_ppo` (the PPO publish chain),
seed/b048/masked-40k pipelines (`submit_*_pipeline.sh`, `finish_*.sbatch`), and
`fast_rollout.py`. Various `*_shards/` dirs and `data*.js.bak*` files are artifacts.

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

## Fast inference decode (`anticipation/fast_decode.py` + `bench/`)

`rollout_score_slots` / `batched_autoregressive_generate_score` run 414 sequential
KV-cached forwards per window and are the hot path for every RL arm, `nbest/`, eval, and
`train.py` validation. `anticipation/fast_decode.py` is a **bit-identical** faster
implementation of the same decode, opt-in everywhere:

```python
rollout_score_slots(model, ids, temperature=0.0, fast={"cuda_graph": True, "buckets": 8})
batched_autoregressive_generate_score(..., ground_truth_score_tokens_to_feed=0, fast=True)
```

Default is OFF in both, so untouched callers keep the exact path they were measured on.
`fast_decode.StaticKVDecoder` is also usable directly as a single-step decoder
(`prefill` / `step` / `logits` / `rewind`) for propose-then-verify style decoding.

What it changes, all four verified bit-identical on greedy decode over 208 windows by
`bench/check_identical.py`:

- LM head applied only where its logits are read (the baseline runs it over all 192
  prefill positions and all 4 chunk positions and discards the rest — 8.4 GiB of fp32
  logits at batch 198 for one useful row);
- **sliced head**: each slot role's legal vocab is one contiguous range
  (`(0,10000)/(10000,11000)/(11000,27512)`, derived from `constrain_score_token_logits`
  and asserted contiguous), so greedy needs 1000–16512 columns, not 55028;
- **`StaticCache`** instead of `DynamicCache` — the latter `torch.cat`s the entire cache
  every step, which is both the dominant memory-traffic term at large batch and the
  reason fp32 batch 198 OOMs on a 48 GB card;
- **CUDA graph per score slot** (3 cached forwards captured together, cache positions
  advanced in-graph by 6), which removes the per-step Python; greedy only, because a
  graph-private philox stream cannot reproduce a seeded `torch.multinomial`;
- **`buckets=N`** (`_WindowedStaticLayer`): a static cache otherwise attends over all
  1020 slots every step while the real cache averages 606. Bucketing rounds the visible
  length up to one of N values (one captured graph each) and gives that bandwidth back.

Two traps encoded there:

- with `StaticCache`, transformers rebuilds the causal mask every step through nested
  `torch.vmap`, which is milliseconds of **CPU** per step and made the static path slower
  than the baseline it replaced. `prebuilt_mask=True` (default) hands it a row slice of
  one 1020×1020 triangular table instead. This is a 4-D mask and takes
  `_preprocess_mask_arguments`' early exit — it is *not* the 2-D chunk-shaped all-ones
  mask that silently truncates the cache (see "KV-cache trap" above).
- with `collect_gt_ce=True` the returned cross-entropies move by up to ~3e-4 absolute
  (the emitted tokens do not): a `(batch, 1)` head GEMM and the baseline's `(batch, 4)` /
  `(batch, 192)` one accumulate differently in cuBLAS. `exact_chunk_logits=True` restores
  the baseline's GEMM shapes and makes `gt_ce` bit-identical, at the baseline's cost.

`bench/` holds the measurement harness — `bench_decode.py` (throughput table + a
`--probe` mode that separates launch-bound from KV-bandwidth-bound, and `--sync-debug`),
`check_identical.py` (the correctness gate: bit-exact tokens, then accuracy/F1 parity),
`bench_tensorrt.py`, `attn_report.py`, `collect_results.py`, JSON under `bench/results/`.
Pass arguments **on the sbatch command line**, never via `--export`: SLURM splits an
`--export` value on commas, so `BENCH_ARGS="--batch-sizes 8,32,96"` silently delivers
`--batch-sizes 8` (that produced one bogus baseline table, job 462074).

Measured facts worth not re-deriving (RTX 6000 Ada, fp32, `run_paper_split_v2/checkpoint-2500`):

- The model runs **fp32 + sdpa** at inference (`evaluate_muster.load_model` sets neither).
- `attn_implementation` is **not** a free choice: this config has
  `scale_attn_by_inverse_layer_idx=True`, and transformers' GPT-2 never forwards that
  per-layer scale to the sdpa or flash paths (it passes no `scaling`), only to `eager`.
  sdpa vs eager on this checkpoint differs by 1.6× the max logit and agrees on 24% of
  greedy tokens. The model was trained under sdpa, so sdpa is the correct path.
- fp32 also rules the fast attention kernels out: SDPA's flash and cuDNN backends reject
  fp32 outright, so an fp32 decode is on math/mem-efficient no matter what. `flash_attn`
  is not installed, and would not help fp32 either.
- **bf16 changes the output materially, but the F1 effect is unresolved — do not quote a
  single number.** Two measurements disagree in sign:
  - a bf16 **weight cast** (`bench/bench_common.py:80`, `model.to(dtype)`) over 208
    `data/val_paper.txt` windows: 76.6% token agreement, only 31/208 windows identical,
    `f1_reward.final_f1` tol1 0.3533 → 0.3208 (`bench/results/gate_bf16_208.json`);
  - a bf16 **autocast** (what every decode path in the repo actually uses) over the 24
    visualizer windows: 63.3% token agreement, `compute_f1` tol1 **22.71 → 24.94**, i.e.
    the opposite sign (`logs/dtype_ab_463802.out`).

  The two differ in mechanism, window set and metric, and the 24-window gap is not
  significant (paired sign-flip permutation p = 0.65). Per-window values were never stored
  for the 208-window run (`check_identical.py:230` writes only `_mean_dicts`), so its
  significance cannot be recovered from anything in the repo. **Treat dtype as a confound
  to remove, not as a measured quality effect** — that is exactly the framing in
  `nbest/generate_fp32.slurm:11-21`. (`visualizer/dtype_ab_viz.py:4`'s "bf16 costing ~9%
  relative F1" conflates the weight cast with autocast and should be reworded.)
  fp16 is ~2× faster and holds F1 (0.3533 → 0.3539, 96.1% token agreement) but is not
  bit-identical; both are opt-in only.
- TensorRT is installed and usable (`tensorrt-cu12==10.15.1.29` +
  `torch-tensorrt==2.11.0+cu128`; the PyPI `torch-tensorrt` is a CUDA-13 build and will
  not load against our cu128 torch). It compiles the **prefill** 2.5–3.2× faster but not
  bit-identically, and that is ~1% of a rollout. It cannot compile the **decode step** at
  all: `index_put_converter` rejects the KV-cache write
  (`Cannot broadcast (B,16,1,64) to (1,B,16,64)`). Not worth adopting.

### Draft-and-verify decode: measured, and it does not pay here

Three separate propose-then-verify schemes were built, gated for exactness, and
benchmarked. **None of them beats `fast_decode` at the batch sizes this repo actually
decodes at.** Do not re-run them hoping for a different answer; re-read this instead.
Caveat on the comparison: the speculative numbers were taken on an **RTX A6000** and the
fast-decode numbers on an **RTX 6000 Ada**, so only the *speedup factors over each
harness's own in-process baseline* are comparable, not the absolute win/s.

| file | what |
| --- | --- |
| `nbest/speculative.py` (+ `test_speculative.py`) | Exact speculative sampling adapted to the packed format. Its two structural facts are worth keeping even if the technique is not: the control triplets are **known in advance**, so a draft can be marched through them exactly as the target would be; and because a verification block ends on a control triplet, the usual "bonus token" falls out for free. `ModelProposer` (neural draft), `NgramProposer` (`nbest/draft_ngram.py`, launch-free table draft), `StagedProposer` (staged/cascade — D1's proposals produced by speculating D1 against a cheaper D2; exactness composes). |
| `train_draft.py` / `train_draft.sbatch` | Trains the shallow drafts. `run_draft_L4/final` = 4 layers/107.8M, `run_draft_L2/final` = 2 layers/82.6M — **shallow copies of the target at full width**, because incremental decode here is kernel-launch bound, so cost tracks layer count far more closely than FLOPs. |
| `nbest/diffdraft.py`, `diffdraft_decode.py`, `train_diffdraft.py`, `gen_diffdraft_kd.py`, `bench/bench_diffdraft.py` | Masked-diffusion-LM drafter (drafts a whole block of score slots in parallel). Runs in `run_diffdraft_{3l,6l,6l_r2}`. |

**Speculative** (job 463991, RTX A6000, `run_paper_split_v2/checkpoint-2500`, 256
`val_paper` windows; baseline = plain `rollout_score_slots`, *not* the fast path):

| batch | T=0 baseline win/s | best speculative | T=1.0 baseline | best speculative |
| --- | --- | --- | --- | --- |
| 1 | 0.372 | 0.748 (L4 γ=4) **2.01×** | 0.369 | 0.639 (L4 γ=2) 1.73× |
| 4 | 1.110 | 1.976 (L4 γ=2) 1.78× | 1.188 | 1.766 (L4 γ=2) 1.49× |
| 16 | 2.009 | 2.814 (L2 γ=1) 1.40× | 1.998 | 2.456 (L4 γ=1) 1.23× |
| 64 | 2.280 | 2.925 (L2 γ=1) **1.28×** | 2.471 | 2.660 (L2 γ=1) 1.08× |

Acceptance is *high* (L4 0.95 greedy / 0.89 sampled, L2 0.88 / 0.79 — but the **untrained**
truncation `U4` accepts only **0.12 greedy / 0.03 sampled**, so the draft has to be trained)
and exactness holds
(100% token-identical to baseline for every draft at T=0, and the sampled path's TV shift
sits inside the two-independent-samples noise floor). The problem is purely economic: the
speedup **decays with batch** — the draft's cost is a fixed fraction of the target's per
step, but the batched KV cache can only roll back to the *minimum* accepted position across
rows, so the batch advances at the pace of its unluckiest row. Meanwhile `fast_decode`'s
speedup **grows** with batch (1.05× at 8 → 1.30× at 96 fp32, and 1.75× with `buckets=8`,
RTX 6000 Ada) and is bit-identical and free. Every shard generator here runs at ~200
sequences in flight.

- **Staged ("speculative speculative") is never the best row** — at every batch and both
  temperatures `staged:L4` is at or below plain `L4` (batch 64, T=0: 2.672 vs 2.843) while
  matching its acceptance exactly at T=0 (0.949 both) and to within noise at T=1.0. The
  extra level buys nothing once the outer draft is already cheap relative to the target.
- **The n-gram (zero-forward) draft never earns its place, and loses outright at batch ≥ 16
  greedy / ≥ 4 sampled** (T=0: batch 64 2.025 vs 2.280 baseline, batch 16 1.922 vs 2.009;
  T=1.0: batch 4 1.159 vs 1.188). Its one win is batch 4 at T=0 — 1.146 vs 1.110, a 1.03×
  that is not worth a table. Its acceptance at T=0 is ~0.46–0.55, dominated by pitch (0.82)
  with onset at 0.49 — the table cannot predict onsets.
- **The diffusion drafter loses at every batch** (job 470497/470496, RTX 3090, 6-layer
  75.6M drafter trained 6000 steps): 0.98× at batch 8 and **0.85 / 0.67 / 0.46×** at batch
  32 for block sizes 2/4/8. It *is* exact (0 of 86,112 score tokens differ over 208
  windows) and it does accept ~2 tokens per target forward — but a denoising step costs
  about what a target step costs, so the arithmetic never closes. Its standalone top-1
  agreement with the teacher degrades sharply with block size (B=2 86%, B=16 69%), driven
  entirely by onset (79% → 32%; pitch stays ~100%).

`bench/bench_speculative.py`, `bench/bench_diffdraft.py` and `bench/summarize_speculative.py`
(rendering kept out of the GPU job on purpose) are the harnesses; `nbest/speculative.py`
also exports `predicted_speedup` / `crossover_cost_ratio` so the verdict can be re-derived
against a cheaper target step without re-running any decode experiment.

## N-best selection (`nbest/`, `nbest_data/`, `run_nbest_reranker/`)

Everything above decodes **one** score per window. This family decodes **many** and picks
one. Two stages, deliberately separated:

1. **Generate a pool.** `nbest/generate_nbest.py` rolls `N` sampled (T=1.0, constrained)
   candidates + 1 greedy candidate per packed window and stores, per candidate, the 414
   score tokens plus three numbers: `logp_ft` (constrained interleaved score-token log-prob
   under the FT model), `logp_base` (MINUS the constrained score-only NLL under the
   *untuned* base AMT — `eval_base_score_ppl`'s `so_c` convention), and `f1`
   (`f1_reward.final_f1`, onset±1) against that window's own GT slots.
2. **Fit a selector on those pools, offline.** Nothing in stage 2 touches the FT model
   itself except GenRM, which *is* the FT model.

The split exists because stage 1 is the expensive part (414 sequential forwards per
candidate) and is reusable: every selector below is trained and compared on the same frozen
`.pt` shards.

### The scripts

| File | What it is |
| --- | --- |
| `nbest/generate_nbest.py` | Pool generator. Flags that matter: `--n-sampled`, `--augment` (train.py-style transpose ±12 / tempo ±20%, seeded per *source line index* at `:179-188`, so augmentation is reproducible independent of sharding), `--fp32`, `--fast-decode`, `--save-every`/`--resume`. |
| `nbest/add_token_features.py` | Second offline pass over an existing shard: one teacher-forced forward per candidate under each model, writing `cand_tok_logp_ft` / `cand_tok_logp_base` as `(C,414) float16`. The scalars were reductions of exactly these, so this is a strict refinement — token identities alone cannot express *where* each model was surprised. Prints a `\|sum(per-token) − stored\|` self-check; read it (see traps). |
| `nbest/reranker.py` | The ~48M `Reranker` `q_phi(x, y)`: full 1020-token window with the candidate substituted into the score slots → 6-layer pre-norm encoder → mean-pool → MLP + sigmoid. `RerankerConfig.token_features > 0` adds the per-token features at the score positions via `index_add` (`reranker.py:65-71`). `build_reranker_from_ckpt` rebuilds from the checkpoint's own `model_cfg`, so older checkpoints missing `token_features`/`feat_clamp` load on dataclass defaults. |
| `nbest/train_reranker.py` | **Pointwise MSE** `q_phi ≈ F1`. Superseded; its only live consumer is the fitted α/β/γ objective, whose `reranker_checkpoint` is `run_nbest_reranker/unfiltered_0819_0955/final.pt`. Emits `holdout_mse`/`pairwise_acc`/`spearman` and **no** `sel_f1`, and truncates its holdout by *candidates* (`max_pairs=6000`, `train_reranker.py:85,:87`), not windows — it is not on the same axis as anything below. |
| `nbest/train_reranker_pairwise.py` | **BRIO/SLiC margin-weighted RankNet**: `L_ij = \|Δ_ij\|·softplus(−sgn(Δ_ij)(q_i−q_j))` over in-window pairs, skipping `\|Δ\| < --min_gap` (0.01). Rewards only *within-pool order*, which is the signal selection actually uses. Also the home of `load_shards`, `split_windows`, `cosine_lr` — every other trainer imports from here. |
| `nbest/train_reranker_listwise.py` | Same machinery, **listwise soft-CE**: match `softmax(q/τ_q)` to `softmax(F1/τ_R)` (`τ_R=0.05`, `τ_q=1.0`). Inference is still `argmax q`. |
| `nbest/duel.py` + `nbest/train_duel.py` | **PairJudge RM knockout** (arXiv:2501.13007). `DuelComparator` takes *both* candidates in one forward (window-with-A ⊕ B's 414 tokens, learned segment embedding) → `logit(A beats B)`; `knockout()` runs the paper's unseeded, reshuffled-every-round single-elimination bracket. Three deliberate deviations, documented at `duel.py:15-37`: **relative** rather than two absolute correctness labels (F1 is graded, and their code eliminates *both* on "both incorrect", which can empty a pool); **order-symmetrised** `s(A,B) = (logit(A,B) − logit(B,A))/2` (the paper has no position-bias handling and its code always advances slot A on ties); **no team grouping** (distinct sampled transcriptions are essentially never token-identical). |
| `nbest/listt5.py` + `nbest/train_listt5.py` | **ListT5-style Fusion-in-Decoder** (arXiv:2402.15838). `m = 5` candidates, one encoder pass each, encoder outputs concatenated along the sequence axis, a 2-layer decoder cross-attends over the fused memory and emits the full permutation **worst-first**. Candidates are told apart by a learned **index** embedding on every token of that pass, not by position. `tournament_sort` handles pools > m (10 passes for 33 candidates). **Collapsed — see Results.** |
| `nbest/train_genrm.py` | **Generative verifier** (arXiv:2408.15240). The FT AMT itself, vocab extended by `ASK`/`YES`/`NO` = 55028/55029/55030 (`NEW_VOCAB = 55031`, resized with `mean_resizing=False`); `score(x,y) = log p(YES \| window-with-candidate, ASK)`; joint loss `L_verify + λ·L_generate` with `λ=1/3`. Two forced adaptations at `train_genrm.py:23-40`: labels binarised at the **within-window median** F1 (balanced 50/50 by construction, and a *within-pool* judgement — a global threshold would mostly encode window difficulty, the exact failure the fitted α/β objective has); no CoT variant. Saves no `model_cfg` — the loader rebuilds the architecture from `--checkpoint` first. |
| `nbest/fit_weights.py` | Grid + coordinate refinement for `α·z(logp_ft) + β·z(logp_base) + γ·z(q_phi)` on a **val** shard → `nbest_data/decode_weights_unfiltered.json` (α=0.96, β=0.47, γ=0.56, per-feature mean/std, and the reranker checkpoint γ refers to). |
| `nbest/relabel_f1.py` | Re-scores a shard's `cand_f1` with the **table's** F1 semantics, keeping the old labels as `cand_f1_emission_order`. See "Two F1 implementations". |
| `nbest/gt_pool_experiment.py` | Diagnostic: computes `B_GT`/`F_GT` for each window's ground truth so the GT can be put in the pool and asked how often the base model ranks it above every generated candidate. Outputs `nbest_data/gtpool_val_s*.pt`. |
| `nbest/generate_unfiltered.sbatch`, `scripts/nb32_gen_train_shards.slurm`, `nbest/add_token_features.slurm`, `nbest/generate_fp32.slurm`, `nbest/generate_bigpool.slurm` | The shard launchers. **Always a file, never `sbatch --wrap`** — see traps. |
| `nbest/fp32_sweeper.sh` | A **file-keyed** resubmit loop around `generate_fp32.slurm`: resubmits only the array indices whose output `.pt` is still missing, up to `MAX_PASSES`. Exists because three independent things break single array submissions here — shared-partition preemption, dead GPU nodes, and untrustworthy `sacct` — and a loop that keys off files on disk (`[ -f "$(outfile "$t")" ]`, polling `squeue`) is immune to all three. |
| `nbest/speculative.py`, `diffdraft*.py`, `draft_ngram.py` | **Not** selection — draft-and-verify decode-speed work that happens to live here. See "Draft-and-verify decode" above. |

### The data shards (`nbest_data/`)

All are `torch.save` dicts with one schema: `window_line_idx (W,)`, `window_tokens
(W,1020) int32`, and per candidate `cand_line_idx`, `cand_tokens (C,414) int16`,
`cand_logp_ft`, `cand_logp_base`, `cand_f1`, `cand_kind` (0 = greedy, 1 = sampled).
**Greedy is always candidate 0 of its window** (`generate_nbest.py:274`,
`cand_flat = torch.cat([greedy_flat, sampled_flat])`) — several eval paths rely on that.

| Prefix | Source | Pool | Windows | Augmented? | dtype |
| --- | --- | --- | --- | --- | --- |
| `unf_{train,val}_shard*` | `data/{train,val}_paper_unfiltered.txt`, stride 3 | **9** (8+greedy) | 35,908 train / 2,667 val | **no** | bf16 autocast |
| `unf32_{train,val}_shard*` | same | **33** (32+greedy) | 35,908 / 2,667 | **train yes, val no** | bf16 autocast |
| `tokfeat32_*` | `unf32_*` + `add_token_features` | 33 | identical | identical | tokens as above; features fp16 |
| `*_tblf1` | `unf*` + `relabel_f1` | — | identical | identical | — |
| `fp32_9_*`, `fp32_32_*` | `generate_fp32.slurm` | 9 / 33 | identical windows | matches its counterpart | **fp32 + `--fast-decode`** |
| `bp64_*`, `bp128_*` | `generate_bigpool.slurm` | 65 / 129 | same 35,908, train only | yes | fp32 |
| `gtpool_val_s*` | `gt_pool_experiment.py` | — | — | — | — |

`tokfeat32_*` is verified to be **bit-identical to `unf32_*`** on all eight shared tensors,
for all seven shards, plus two extra `(C, 414) float16` feature tensors per shard
(`C` = 197,505 on train shards 00–03, 197,472 on 04/05, 88,011 on val) — so a
`tokfeat32`-trained model and an `unf32`-trained one are on the same pools and are directly
comparable.

`unf32` train shards pass `--augment` (`scripts/nb32_gen_train_shards.slurm:25`); `unf_*`
does not (`nbest/generate_unfiltered.sbatch:26-29`). **The 9-candidate and 33-candidate
shards therefore differ on two axes, not one** — greedy F1 on the same 400-window holdout
prefix is 0.31301 unaugmented vs 0.28848 augmented.

`bp64`/`bp128` are a pool-depth ablation held **constant in windows** and therefore
deliberately *not* compute-matched (`generate_bigpool.slurm:10-25`): the question is
whether a deeper oracle ceiling is reachable, not whether depth is worth its FLOPs. Pairs
per window grow 528 → 2080 → 8256, so the pairwise loss also gets far more signal.

### Two F1 implementations exist, and they are not the same function

This is the single most confusing thing in the family. Both are correct; they answer
different questions.

| | `f1_reward.final_f1` | `visualizer/compute_f1.score_notes` |
| --- | --- | --- |
| matching order | **emission order** (online, one-to-one, a consumed GT note is never returned) | **onset order** (`sorted(pred, key=t)`, greedy nearest-onset per pitch) |
| rests / out-of-range slots | counted in `n_pred` | dropped before scoring |
| used by | `generate_nbest.py` to **label** candidates, `bench/check_identical.py`, PPO-F1's reward | the **results table** (`<rollout>.f1`) |
| so it is | what every reranker was trained to rank by, and what every holdout `sel_f1` reports | the number every result is finally judged on |

`nbest/relabel_f1.py` re-scores a shard under the table's semantics with no GPU and no
regeneration. **Measured: the two matchers are effectively the same function on identical
inputs.** On `unf32_val_shard00` (88,011 candidates) mean 0.31135 → 0.31135, mean delta
+0.00000, **max \|delta\| 0.00361**, 0.02% of rows changed; on `unf_val_shard00` (24,003)
max \|delta\| 0.00103, 0.01% changed. **Relabelling the training labels is a no-op** — do
not spend a job on it.

Note that `relabel_f1.py` changes *two* things at once — the matching order **and** the
handling of REST/out-of-range slots (it drops them, shrinking `n_pred`) — and the combined
effect is still ~0. The relabelled shards `unf{,32}_val_shard00_tblf1.pt` exist and carry
`cand_f1_emission_order` for audit.

`relabel_f1.py`'s own docstring attributes the +0.63 pt train/eval gap to the matcher. That
is **wrong**, and the measurement above is why: the real gap is the **GT input** on the
visualizer side — see the pool-oracle bullet under "Comparability rules".

### How a selector becomes a row in the F1 table

```
visualizer/rerank_sample_viz.py            # decodes the 9-candidate pool for the 24 viz
   --qsel-reranker <ckpt> --output-qsel …  # windows; writes ONE shard json per selector
   ▼
visualizer/merge_rerank_rollouts.py --group rollouts_<name> --shards …
   ▼                                        # attaches <group>.<variant>.pred_score +
visualizer/data.js                          # rerank_meta, and checkpoint_<group>
   ▼
visualizer/compute_f1.py --data visualizer/data.js   # writes <group>.<variant>.f1
   ▼
visualizer/split_visualizer_payload.py      # -> data_slim.js (+ data_ex/*.js)
```

`rerank_sample_viz.py` builds one pool per window and emits **several** shards from that
*same* pool, one per selection rule: the fitted objective (`--output`), pool oracle
(`--output-oracle`, ties keep greedy), first sampled candidate (`--output-sample1`),
unweighted MBR consensus (`--output-mbr`), a second reranker's bare argmax
(`--qsel-reranker`/`--output-qsel`), duel knockout (`--duel-ckpt`/`--output-tournament`),
ListT5 tournament sort (`--listt5-ckpt`/`--output-listt5`), and GenRM argmax `log p(YES)`
(`--genrm-ckpt`/`--output-genrm`). All selection polarities were audited against their
trainers' own holdout expressions — there is no sign error anywhere in this path.

**There is no committed generator for the N-best table.** `visualizer/rl_f1_table.py` is
the *old RL* table: it macro-averages by piece and its `ROWS` still name a
`rollouts_valloss` group that no longer exists in `data.js`. The numbers below are plain
means over the 24 windows (which equals macro-by-piece here, because each of the 24 windows
is a distinct `piece`). **If that table is ever published, write the generator** — it is
the standing deliverable for this whole family.

### Results — the 24 visualizer windows (`onset_pitch_tol1` %, `variant=raw`)

All pool rows are `run_paper_split_v2/checkpoint-2500`, bf16 autocast, and share a pool
verified **bit-identical** across the seven groups (9,072 stored field values compared,
0 mismatches).

| row | tol1 | vs pool greedy (23 w) |
| --- | --- | --- |
| pool oracle (of the pool's own diagnostic F1) | 43.36 | +18.71 |
| **pairwise32** | **30.22** | +5.32 |
| pairwise (9-cand training) | 28.74 | +3.47 |
| fitted α/β/γ | 25.12 | +0.00 |
| *pool greedy = candidate 0* | *25.80 (23 w)* | — |
| MBR consensus | 19.11 | −6.21 |
| listwise32 | 18.90 | −6.81 |
| T=1 sample | 18.15 | −7.15 |

Non-pool rows for context: greedy "ours" 23.58 raw / 21.23 filtered, paper 2 28.76,
paper 1 17.40, 8-wide beam rerank 18.63, α/β beam sweep 4.02 → 24.73.

Honest reading:

- **`pairwise32` is the best non-oracle row, and its win is real but thin.** Against
  candidate 0 on the 23 recoverable windows it wins 6, loses 8, ties 9, and the +5.32 mean
  does **not** survive a paired sign-flip permutation test (p = 0.31; pairwise p = 0.54,
  MBR p = 0.16, listwise32 p = 0.43, T=1 p = 0.26). **Only the pool oracle separates from
  greedy** (+18.71, 17 wins / 0 losses / 6 ties, p < 1e-4). At n=24 this table can rank
  methods; it cannot certify any non-oracle gap.
- **The fitted α/β/γ objective does essentially nothing.** It differs from greedy on 3 of
  23 windows and they cancel to exactly +0.00 — coincidence, not a bug. Its own val fit
  says the same: 0.38753 selected vs 0.37016 greedy on 2,667 windows against a 0.52674
  oracle (`nbest_data/decode_weights_unfiltered.json`).
- **MBR, listwise32 and T=1 sampling all anti-select** — 6–7 pt *below* just taking the
  pool's greedy candidate. For T=1 that is expected (it is the honest "no selection"
  control). For MBR and listwise32 it is a negative result and should be reported as one.
- **listwise32 is the clean negative.** Against `pairwise32` it is the one fully controlled
  head-to-head in the set: same `unf32` shards, same 33-candidate pools, same 400-window
  holdout, same greedy baseline 0.28848, same `--windows_per_batch 2` and step budget —
  only the loss differs. Pairwise wins by **+0.0362** best-vs-best on holdout (0.33618 vs
  0.30002) and by 11.3 pt on the viz table. Soft-CE over a pool where most candidates are
  near-ties spends its gradient on the ordering of noise.

### Results — trainer holdouts (read the comparability rules before quoting any of these)

Exact launch commands are recoverable from `wandb/run-*/files/wandb-metadata.json`.

| run | shards | pool | holdout N | greedy | oracle | best `sel_f1` | final | launch args beyond `--shards`/`--run_dir` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `pairwise_0821` | `unf_*` (unaug) | 9 | 400 | 0.31301 | 0.50947 | 0.39482 @14000 | 0.38868 | `--steps 20000` (default `--windows_per_batch 8`) |
| `pairwise32_0821` | `unf32_*` (aug) | 33 | 400 | 0.28848 | 0.63937 | 0.33618 @15000 | 0.32290 | `--steps 20000 --windows_per_batch 2` |
| **`pairwise32feat_0822`** | `tokfeat32_*` | 33 | 400 | 0.28848 | 0.63937 | **0.38116 @16000** | 0.37411 | identical to `pairwise32_0821` |
| `listwise32_0821` | `unf32_*` | 33 | 400 | 0.28848 | 0.63937 | 0.30002 @11000 | 0.29207 | `--steps 20000 --windows_per_batch 2` |
| `duel32_0822` | `tokfeat32_*` | 33 | **120** | 0.36643 | 0.69173 | 0.43297 @19000 | **0.43907** | `--steps 20000 --windows_per_batch 2 --pairs_per_window 8` |
| `listt5_0822` | `unf32_*` | 33 | **150** | 0.34710 | 0.68826 | 0.34710 @1000 *(= greedy exactly)* | 0.29961 | `--steps 20000 --windows_per_batch 4 --m 5` |
| `genrm_0822` | `unf32_*` | 33 | **150** | 0.34710 | 0.68826 | 0.38498 @7000 | **0.40758** | `--steps 8000 --cands_per_step 4` |
| `unfiltered_0819_0955` | `unf_*` | 9 | candidate-truncated, not window-truncated | n/a | n/a | emits no `sel_f1` | — | `--steps 20000` |

- **Per-token generative features are the biggest measured win in this family — and they
  look like a dud for the first 3,000 steps, so do not kill the run early.**
  `pairwise32feat_0822` (`token_features=2`, i.e. per-token `log p_FT` and `log p_base` at
  each of the 414 score positions) *starts* worse than the scalar model — 0.29467 @1000,
  0.27368 @2000 against the same 0.28848 greedy — then overtakes it from step 4000 and ends
  at **0.38116 vs pairwise32's 0.33618** on a bit-identical holdout with an identical
  optimisation budget: **+0.045 sel_f1, i.e. 13.6% → 26.4% of the (oracle − greedy)
  headroom.**
  The features are a strict refinement of the two scalars, so this says the *summed*
  log-probs were throwing away most of their signal. **This model has no row in the F1
  table and cannot get one until the decode-time plumbing exists — see traps.**
- **The duel comparator works.** After the `@torch.no_grad()` fix (below) it trains
  cleanly: `sel_f1` 0.2490 → 0.43907 against greedy 0.36643 and oracle 0.69173, with duel
  accuracy on gapped pairs rising 0.536 → 0.616. `matches_per_window` is **32.0** at every
  eval, i.e. exactly N−1 for a 33-candidate pool. The knockout invariants are verified
  independently (500 random pools over N ∈ {2,3,5,8,9,17,33,65,129}): matches == N−1 and
  rounds ≤ ceil(log2 N) in all 500, and with a perfect comparator the true best is crowned
  **500/500**. But its holdout is 120 windows, so it is **not** comparable to the 400-window
  rows above.
- **GenRM is strong but unfinished**: `final.pt` (step 8000) `sel_f1 = 0.40758` vs greedy
  0.34710 and oracle 0.68826 on 150 windows, verdict accuracy 0.667. Not comparable to
  `pairwise32feat`'s 0.38116 (150 vs 400 windows), and it has no row in `data.js` at all.
- **ListT5-FiD collapsed to a constant permutation and is not a method.** Its decoder emits
  the *same* permutation regardless of input: `(1,2,4,3,0)` on **all 120** diagnostic
  groups, 1 distinct permutation, `crowned slot histogram {0: 120}` against a true-best
  histogram of `{0:42, 1:19, 2:20, 3:18, 4:21}` (`logs/diag_listt5_463981.out`).
  **Cause: teacher-forced permutation CE is nearly free at later positions.** The loss is a
  plain CE over all `m` decoder steps (`train_listt5.py:189-190`); at step *t* the decoder
  has already been told which *t* slots were emitted, so it only has to choose among `m−t`,
  and the last step is worth exactly 0 nats. A model that learns nothing but "don't repeat
  a slot" scores `(ln5+ln4+ln3+ln2+ln1)/5 = 0.9575`; the logged `perm_ce` then sat **on that
  baseline and never left it** — mean **0.9575** over all 400 logged points (0.9485 after
  step 1000), i.e. equal to the no-information score to four decimals, with only per-batch
  noise around it (range 0.7193–1.6784 overall, 0.7193–1.1869 after step 1000; last logged
  0.9078). The loss fell without quality ever entering it.
  Consequence: `sel_f1` takes only **three** values across all 20 evals (0.29961 ×17,
  0.30488 ×1, 0.34710 ×2) — `best.pt` is **step 1000**, the first of the two evals where the
  constant permutation happened to land on index 0, which is
  the greedy candidate, so its `sel_f1` is **bit-equal** to `greedy_f1`
  (0.34710144992917774 both); `final.pt` lands elsewhere and scores 0.29961, 4.7 pt *below*
  greedy. Shipping either would silently duplicate or undercut the greedy row.

### Comparability rules for this table (read before adding a row)

The most expensive mistake in this family is putting two numbers in one column that were
not measured on the same thing. All of the following are verified against the repo:

- **The greedy "ours" row is a DIFFERENT MODEL from every selection row.** `data.js`
  `checkpoint = run_paper_split_v2/checkpoint-7500`, while every
  `checkpoint_rollouts_{rerank*,mbr,sample_t1}` is `run_paper_split_v2/checkpoint-2500`
  (`rerank_sample_viz.sbatch:30`). They diverge wildly per window (ckpt-7500 vs ckpt-2500
  tol1: val-08 100.00 vs 13.77, val-09 83.33 vs 2.90, test-10 6.52 vs 96.38 — the sign of
  the gap is not even stable). **Every reported "gain over greedy" that uses the
  `rollouts` row is a gain over a different model.** There is no ckpt-2500 greedy row in
  `data.js`; the correct baseline exists only as candidate 0 inside `rerank_meta`,
  recoverable on **23/24** windows (val-06 has no group that selected index 0).
- **Every selection row is `variant = "raw"`; both external-paper rows are
  `variant = "filtered"`.** On the same checkpoint the conditioning alone is worth
  **2.35 pt** (`rollouts.filtered` 21.23 vs `rollouts.raw` 23.58). Note that
  `compute_f1.py:185`'s own "like-for-like" table deliberately restricts to
  `variant == 'filtered'`, which **excludes every N-best row** — so whatever assembles the
  deliverable table is not that code path.
- **"pairwise32"/"listwise32" do not select from 33 candidates.** `--n-sampled` defaults to
  8 (`rerank_sample_viz.py:70`) and `rerank_sample_viz.sbatch` never passes it, so both
  shards carry `"n_sampled": 8`. The "32" is the **reranker's training** pool depth. The
  correct label is *"reranker trained on 33-candidate augmented pools, evaluated on the
  same 9-candidate pool"*.
- **`rollouts_rerank` and `rollouts_rerank_ab*` are not selection rows.** They are 8-wide
  and 5-wide **beam** decodes with no per-candidate F1 and no pool. Their pruning score
  *is* the objective, so the rescoring argmax is index 0 in 24/24 windows for all five
  `ab*` rows — their 4.02 → 24.73 spread is an α/β **decoding** sweep. Separate table.
- **The oracle row *now in `data.js`* is an oracle of the wrong metric — but the script has
  since been fixed, so do not "re-fix" it.** The merged rows were produced by a version that
  maximised the *pool* F1: `f1_reward.final_f1` matched against `gt_notes_for_variant(ex,
  "raw", n_slots)`, a raw-aligned **subset** of the ground truth — **2,998 of 3,312** GT
  notes over the 24 windows, as low as **94/138** on val-07 (both counts are also recorded
  in the fix's own comment at `rerank_sample_viz.py:256-267`). The table's
  `compute_f1.score_notes` matches against the **full** `gt_score`. Since the two F1
  *functions* agree to ≤0.004 on identical inputs (previous section), this gap is purely the
  GT input, and it is real: over all 168 (window, method) pairs the stored pool number is
  higher by a mean of **0.63 pt**, up to **7.15 pt** on a single pair (val-07 / sample_t1,
  tied with val-07 / oracle). It is large enough to let an ordinary row beat the "oracle"
  under the table metric — test-07: `rollouts_rerank_pairwise` 19.57 vs oracle 18.12, the
  one such case in 144 comparisons.
  **Current code**: `rerank_sample_viz.py:433-434` selects the oracle by
  `rows[i]["f1"]`, which is now `score_notes(pred_notes, gt_full)` — the table's own metric
  against the full GT — and keeps the old value as the `f1_emission_order` diagnostic. You
  can tell the two generations apart on disk: pre-fix candidate records carry `f1` but **no**
  `f1_emission_order`, which is the case for every group currently in `data.js`.
  **So: label the existing row *oracle of the pool diagnostic F1*, never "the ceiling of the
  pool" — and regenerate it before quoting it as a ceiling.**
- **Trainer holdouts are not comparable across runs and are not generalization numbers.**
  Four independent breaks: (i) `hold_rows[:max_windows]` is a **prefix** of the token file,
  not a sample, and the head of the file is much easier — the same shards give greedy
  0.36643 / 0.34710 / 0.28848 / 0.28069 at N = 120 / 150 / 400 / 1796, so GenRM's headline
  +0.0605 is smaller than the +0.059 artefact of scoring on 150 windows instead of 400;
  (ii) 9-candidate shards are unaugmented and 33-candidate ones are not; (iii) `best.pt` is
  max-over-evals on the same holdout it is reported from, and every run gets 20 such evals
  except GenRM's 8; (iv) optimisation budget varies ~4× in `windows_per_batch × steps`
  (40k for the `*32*` runs vs 160k for `pairwise_0821`), before any multi-GPU factor. Also, the
  "holdout" shares a sliding window at stride 3 with training data, and it is a slice of the
  paper-split **train** file, while the deliverable table is 12 val + 12 test windows.
  The only defensible cross-row protocol is: **one shard family** (`tokfeat32_*` covers
  everyone — feature-free models just ignore the extra tensor), the **full 1,796** holdout,
  and **Δ over greedy** / **% of (oracle − greedy) headroom** rather than raw `sel_f1`.
  Common baselines for that: greedy **0.28069**, oracle **0.61382**.

### Traps (each of these cost real jobs)

- **`sbatch --wrap` expands `$VARIABLES` in the OUTER shell.** A wrap string containing
  `$SLURM_ARRAY_TASK_ID` is interpolated at submit time with the *submitting* shell's
  (empty) value, so every task takes the same branch. Arrays **369449** and **370733** both
  ran the **val** branch for every task instead of their own train shard. That is why
  `scripts/nb32_gen_train_shards.slurm:12-14` exists as a file at all, and why
  `generate_fp32.slurm:30-32` and `generate_bigpool.slurm:29-30` repeat the warning.
  **Always submit via a script file.**
- **`sacct` on this cluster serves records for REUSED job IDs.** Still true today:
  `sacct -j 470308` returns a `pysr_eval` job submitted **2026-01-28**, not our `nb-fp32`
  array from 2026-08-23. Job state from `sacct` is not evidence of anything. Only `squeue`
  (while queued/running) and the **output files on disk** are trustworthy — which is exactly
  why `fp32_sweeper.sh` keys off files.
- **A bare `--gres=gpu:1` schedules onto 11 GB cards, and they do not merely OOM.** Array
  **470107** lost all 11 tasks: every log prints `GPU has 11GiB -> window_batch 2 (32-cand)
  / 7 (9-cand)` and then dies with `CUDA error: no kernel image is available for execution
  on the device` — the card's compute capability predates this torch build's cubins, so no
  batch size would have helped. Submit with `--constraint="a6000|6000ada|a100|a40"` (as
  `fp32_sweeper.sh:21` does) and keep `generate_fp32.slurm`'s refuse-below-20 GiB branch.
- **`generate_nbest.py` used to write its shard only at completion** — a cancellation at 7 h
  lost a whole shard. It now writes a resumable `<output>.partial` every `--save-every`
  batches via write-to-temp-then-`rename` and continues with `--resume` (`:231-245`,
  `:294-307`). Two follow-ons: **(a) FIXED** — the checkpoint path was untested when it
  landed: `_pack` was defined *after* the loop, so the first checkpoint write raised
  `UnboundLocalError: cannot access local variable '_pack'` and killed array **470308**
  tasks 0 and 9 at 756/11970 windows. `_pack` is now defined at `:215`, before the loop at
  `:247`. **(b) STILL TRUE** — `--resume` does **not** restore the RNG
  (`torch.manual_seed` runs once at `:171`), so a resumed shard's *sampled* candidates are
  not the ones an uninterrupted run would have produced. `--augment` is unaffected (seeded
  per source line index).
- **The dtype split in the F1 table.** The greedy "ours" row comes from
  `precompute_visualizer.py`, which has **no autocast at all** (fp32); every
  sample-then-select row draws its pool from `rerank_sample_viz.py`, which defaulted to
  **bf16 autocast** before `--fp32` existed (no shard under `visualizer/rerank_*shards*/`
  carries a `decode_dtype` key at all — the string appears in none of them — i.e. they all
  predate the stamp); the two
  paper rows run their own `infer()`, fp32 on chunk 0 and **fp16 autocast** thereafter.
  Also, within a single row `logp_ft` is computed **under** autocast while `logp_base` is
  computed **outside** it (`generate_nbest.py:267-278`) — **if the pools are regenerated in
  fp32, α/β/γ and `feature_stats` must be refit or the α/β/γ row is invalid.**
- **`generate_nbest.py`'s `DEFAULT_CKPT` is a landmine.** `:54` still defaults to
  `run_paper_split_v2/checkpoint-7500` and the docstring at `:23-25` still claims 7500 is
  "the model of record". **Every real shard on disk is checkpoint-2500**, and every launcher
  passes `--checkpoint` explicitly. Omitting the flag silently produces a pool from a
  different model than everything it would be compared against — and it has already happened
  once: `nbest_data/smoke_shard.pt` is the only shard stamped checkpoint-7500, because it was
  made without the flag.
- **`token_features > 0` checkpoints are silently mis-scored at decode time. This is the
  live blocker on the best model in the family.** `rerank_sample_viz.py` has no per-token
  feature plumbing at all: it calls `reranker2(substitute_candidates(...))` with no
  `feats`, and both `Reranker.embed_tokens` (`reranker.py:65-71`, reached from `forward` at
  `:73`, which itself defaults `feats=None`) and `DuelComparator.forward`
  (`duel.py:105-113`) default `feats=None` and just **skip** the `index_add`. No exception.
  Measured on `pairwise32feat_0822/final.pt` over 64 candidates of `tokfeat32_val_shard00`:
  dropping `feats` moves the score by a factor of **0.001× to 73×** (median 0.07×), and
  **83%** of candidates move by more than 3.2× in one direction or the other — it is not a
  uniform shift, it is arbitrary re-ranking deep in the sigmoid tail. The
  features come from a separate offline pass; the viz only ever has the summed scalars.
  Both `pairwise32feat_0822` and `duel32_0822` are `token_features=2`. **Either plumb the
  features through `rerank_sample_viz.py` (they exist in `tokfeat32_*`) or add a guard at
  the two load sites that refuses a checkpoint whose `model_cfg["token_features"] > 0`.**
- **`train_duel.py`'s `@torch.no_grad()` bug — FIXED, and the fix is load-bearing.** The
  decorator sat on `duel_batch`, which is correct for the two eval call sites but fatal for
  the *training* forward that goes through the same function: every step produced a
  graph-less loss and `loss.backward()` raised `element 0 of tensors does not require grad`
  (job **461863**, zero steps trained). `duel_batch` is now undecorated with the reason in
  its docstring (`train_duel.py:124-131`) and `selection_metrics` carries the `no_grad`
  itself (`:148`). Job **474782** then trained all 20k steps. Do not "tidy up" that
  decorator back.
- **`best.pt` is not the best checkpoint for GenRM, ListT5 or the duel.** Those trainers
  save `final.pt` and print final metrics but never compare them to `best_sel`
  (`train_genrm.py:260-263`, `train_listt5.py:219-222`), unlike
  `train_reranker_pairwise.py:345-347`. GenRM's `final.pt` (0.40758) beats its `best.pt`
  (0.38498); the duel's final (0.43907) beats its best (0.43297); ListT5's `best.pt` is
  frozen at the first eval.
- **`logp_base` is reported not to be reproducible off-GPU — UNVERIFIED, not re-measured
  in this pass.** An earlier audit found that on 12 random candidates from
  `unf32_val_shard00.pt`, a CPU rerun of the same `nll_at_positions` on the same tokens
  differed from the stored GPU value by mean **+38.4** nats (sd 48.0, range [−35.7,
  +151.9]), while two *independent GPU* computations of the same quantity (generate-time
  and `add_token_features`) agreed to ≤0.03 on those rows. That magnitude is large enough
  that it may be a bug in the CPU rerun rather than a numerics fact — it was not
  reproduced here. Until someone does reproduce it: do not refit α/β or recompute
  `logp_base` on CPU, and do not use CPU to arbitrate a GPU-vs-GPU discrepancy (including
  the `add_token_features` one below).
- **Read `add_token_features.py`'s consistency print, and know which half matters.** The
  **ft** side is *expected* to disagree (mean 1.27, max ~12) — sampled candidates' stored
  `logp_ft` came from the rollout's own logprobs, not from a teacher-forced rescore. The
  **base** side must match, and mostly does (mean 0.015–0.019). But on `tokfeat32` shards
  **04, 05 and val** the base max is **62.1 / 130.4 / 126.4** nats, against ≤0.11 on shards
  00–03. It is a handful of rows — 4 / 4 / 7 rows over |Δ|>0.5, out of 197,472 / 197,472 /
  88,011 — and cannot move any aggregate (the means stay 0.016 / 0.017 / 0.019), but the
  cause is unexplained.
- **The fast path has never been gated for *sampled* decode.** `generate_fp32.slurm:57` and
  `generate_bigpool.slurm:48` pass `--fast-decode`, and `generate_nbest.py:210` gives the
  **T=1.0** rollouts `fast={"buckets": 8}`. Every recorded gate in `bench/results/gate_*`
  ran at `temperature=0.0`; the bit-identity claim in "Fast inference decode" covers greedy
  only. Separately, **do not run `--fast-decode` without `--fp32`**: `fast_decode.py:182`
  sets the static-cache/graph buffer dtype from `next(model.parameters()).dtype`, which
  under autocast is fp32 while the forward emits bf16 hidden states
  (`_SlotGraph.hidden`, `:358`). Untested combination.
- **Superseded pool generations are still on disk under names that look current.** Verified
  by comparing each shard's stored candidate F1 vectors against the merged `rerank_meta`:
  `all_v3.json`, `all_v4_check.json`, `all_oracle_v3.json`, `all_sample1_v3.json`,
  `all_mbr.json`, `all_pairwise.json` match **24/24** windows; `all.json`, `all_fixed.json`,
  `all_rerun.json`, `all_rerun2.json`, `all_oracle.json`, `all_oracle2.json`,
  `all_oracle_fixed.json`, `all_sample1.json`, `all_sample1_fixed.json` match **0/24**.
  Do not re-merge one of the latter by name.
- **The viz pool seed used to depend on how the run was sharded — FIXED, and the fix is
  backward-compatible.** `rerank_sample_viz.py` seeded with `args.seed + 1000*ki + sum(map(
  ord, key))` where `ki` was the index *within this process's key list*, while
  `rerank_sample_viz.sbatch:39` shards with `order[TASK_ID::4]`. It now keys on
  `order.index(key)`, the window's global position in `example_order` (`:219-221`), so the
  pool is a function of the window and not of the job layout. For the single-process
  24-key runs that produced every merged row, `ki == order.index(key)`, so those pools are
  still reproducible. Do not revert this.
- Cosmetic, no effect on results: `r["logp_yes"]` is attached to the candidate records
  *after* the MBR / tournament / ListT5 candidate dicts were snapshotted, so those three
  shards' records lack it. `train_listt5.build_group`'s docstring claims it shuffles slots;
  it does not — the shuffle is in the training loop (`train_listt5.py:182`) and there is
  none at eval, which is what let a constant permutation land on a constant pool index.
  `visualizer/compare_reward_criterion.py:49-50` documents `reward_criterion_f1` as "pitch+dur
  exact, onset ±1" but calls `IncrementalF1` with the default `require_duration=False`; the
  code is right, the comment is wrong.

### Unfinished, as of 2026-08-23

| what | job | state |
| --- | --- | --- |
| `bp64_*` / `bp128_*` pool-depth array (64/128 candidates, 12 tasks, `%2` throttle on `thickstun`) | **473909** | running; tasks 0–1 in flight ~8 h, 2–11 pending. Only `bp64_train_shard0{0,1}.pt.partial` exist. |
| fp32 shard regeneration (11 tasks: 4 nine-candidate + 7 thirty-three-candidate) | **475511**, resubmitted by sweeper **470509** | running; 3 of 11 complete (`fp32_9_train_shard0{0,1}.pt`, `fp32_9_val_shard00.pt`), 6 in progress as `.partial`, `fp32_32_train_shard05` / `fp32_32_val_shard00` not started. Do **not** hand-resubmit — the sweeper re-derives the missing set from disk. |
| **The full F1 table across all methods** | — | the standing deliverable. `pairwise32feat`, `duel32`, `genrm` and `listt5` have holdout numbers but **no row in `data.js`**, and the first two are blocked on the `token_features` plumbing above. |

## Other directories and stray files

- `bench/` — the inference decode benchmark, correctness gate, TensorRT/attention probes,
  and the speculative / diffusion-drafter harnesses (see "Fast inference decode" above).
  `bench/results/SUMMARY.txt` is the rendered collection of every decode measurement and is
  the fastest way to answer "was that already measured?".
- `nbest/` — N-best pool generation and the whole selection-method family (rerankers,
  duel/knockout, ListT5-FiD, GenRM), plus the draft-and-verify decode experiments. See
  "N-best selection" above. Shards live in `nbest_data/`, trained selectors in
  `run_nbest_reranker/`, drafts in `run_draft_{L2,L4,ngram}` and `run_diffdraft_*`.
- `scripts/` — one-off analysis/export utilities (dataset stats, MIDI exports for
  listening, alignment benchmarks, `check_packed_prefix_length.py`, staff/interleave
  visualization). Useful references, not part of any pipeline.
- `tests/` — mix of the upstream anticipation library's utilities
  (`benchmark.py`, `check-integrity.py`, `print-tokens.py`, `sonify-tokens.py`, …) and
  this project's pytest suites: `test_ltlm_{ddp,diffusion,eval,objective}.py`,
  `test_ppo.py`, `test_seed_pipeline.py`, `test_paper_seed_pipeline.py`,
  `test_train_speed.py`. Run project tests with `pytest tests/test_*.py`; top-level
  `test_f1_reward.py`, `test_plan_pipeline.py`, `test_pianobart_encoder.py` are plain
  `python <file>` CPU checks.
- `analysis/` — duration-histogram study (dataset vs model) artifacts + the sbatches
  that made them.
- `humaneval/` — the **upstream** anticipatory-music-transformer human-evaluation
  tooling for Lakh MIDI clips (vendored; unrelated to the ASAP score-infilling work).
- `jae_README.md` / `m2s_README.md` — vendored copies of the two reference papers'
  READMEs (Zeng+ `joint-apt-epr` and Beyer `MIDI2ScoreTransformer` respectively); the
  actual integration lives in `external/` (see Visualizer section).
- `asap-dataset-master/`, `ATEPP/`, `MUSTER/`, `data/` — datasets / external tools.
- Everything else at top level (`results_*`, `finale*`, `checkpoint-*`, `april_output`,
  `aug_labels*`, `bach_example`, `debug/`, `exposure_model_smoke`, `greedy_analysis_*`,
  `heatmaps_run_genai*`, `jitter`, `masked_40k`, `media`, `model_*`, `opening_examples`,
  `recovery`, `scratch_heatmaps`, `test_examples`, `test_outputs`, `tmp_*`,
  `triplet_beam`, `highreg_90`, `hf-ckpt-3500`, `asap_only`, `experimental_greedy_outputs`,
  `autoregressive_inference_results`, `muster_evaluation_results`, `logs/`, `wandb/`)
  is experiment output — see "Live vs dead run dirs" below before deleting any `run_*`.

## Conventions / gotchas

- Prefer the `anticipation/packed_sequence.py` helpers; do not hand-roll offset math.
- The model vocab is resized to `VOCAB_SIZE` (55028) on load; the base checkpoint is
  `stanford-crfm/music-medium-800k` (whose vocab is already 55028, so the resize is a
  no-op — every row is pretrained).
- `train.py` uses `accelerate` (bf16 on GPU). Don't manually `.to(device)` the model —
  let `accelerator.prepare` place it (critical for multi-GPU).
- Token files use ` | ` as a trailing separator; readers split on the first `|`.
- Times and durations everywhere are in 10 ms bins, not seconds.
- **KV-cache trap** (silent): during cached AR decode, never pass a chunk-shaped
  all-ones `attention_mask` — transformers reads it as a padding mask over
  past+current keys and truncates the cache to one token while the rollout still looks
  structurally valid. Pass no mask once `past_key_values` exists (as
  `inference.batched_autoregressive_generate_score` and
  `ltlm_eval.ltlm_autoregressive_generate_score` do).
- **SLURM**: `NCCL_P2P_DISABLE=1` is mandatory on the `ellis` partition (broken P2P on
  ellis-compute-02 hangs the first collective while GPUs sit at 100% util looking like
  slow startup); multi-GPU ellis jobs pin the GPU TYPE (`gpu:nvidia_rtx_a6000:4`)
  because ellis mixes a 5×3090 node with an 8×A6000 node. See the section below for the
  cluster-wide traps.
- **Live vs dead run dirs**: `run_grpo`, `run_crpo`, `run_onpolicy_distill`, `run_ppo`,
  `run_ltlm*`, `run_plan_vq_20260814_185416`, `debug/ppo_ellis`,
  `run_nbest_reranker/smoke` are dead/superseded artifacts, but `run_paper_split_v2` (init
  for all RL arms), `run_grpo_acc_reward`, `run_crpo_acc_reward`,
  `run_ppo_corrected_20260814_020654_2364547`, `run_ppo_f1_triplet_20260814_135022`,
  `run_plan_vq_20260814_192509`, **all of `run_nbest_reranker/{unfiltered_0819_0955,
  pairwise_0821, pairwise32_0821, listwise32_0821, pairwise32feat_0822, duel32_0822,
  listt5_0822, genrm_0822}`**, **`nbest_data/`** (~5.9 GB, and days of GPU time — the
  `.partial` files belong to jobs still running), and `run_draft_{L2,L4,ngram}` /
  `run_diffdraft_6l_r2` (the drafts every speculative number was measured against) hold
  referenced checkpoints — don't delete them.

## SLURM on this cluster (Cornell CS) — read before debugging a scheduling problem

These are properties of the cluster, not of this repo; they apply identically in
`~/anticipation3`, `~/worldmodel` and `~/voiceclone`. Each one has cost real jobs.

- **You cannot see the cluster.** `PrivateData = accounts,jobs,usage,users`, so `squeue`
  shows **only your own jobs**. A partition can look completely empty while being full.
  Node-level `AllocTRES` from `scontrol show node <node>` is the **only** trustworthy view
  of occupancy (`grep -E "CfgTRES|AllocTRES"`); `sinfo`'s GRES column is capacity, not
  availability.
- **`Reason=ReqNodeNotAvail,UnavailableNodes:unicorn-compute-04` is a generic red herring.**
  Slurm attaches it to *any* pending job when some node in the partition is down.
  unicorn-compute-04 has a40 GPUs and cannot satisfy an a6000/6000ada constraint anyway.
  Do not chase it.
- **`ma-compute-02`, `kuleshov-compute-01` and `badfellow` each have a node-wide-dead GPU.** NVML fails
  for the WHOLE node, so `torch.cuda.device_count()` returns 0 even though the other cards
  are idle, and HF then raises the misleading `ValueError: Your setup doesn't support
  bf16/gpu`. **Always add `--exclude=ma-compute-02,kuleshov-compute-01`, and always
  preflight with a real bf16 matmul on every card**, not just `device_count()`. Confirmed by
  probe job **475272** (`probe-ma02`, ma-compute-02); the jobs it explains are the killed
  **474996** (`persona-fullft`, ma-compute-02) and **475341** (`persona-ft-3090`,
  kuleshov-compute-01). **`badfellow` joined the list 2026-08-24**: array
  **548999** tasks 4/5/6 died there in <10s with `RuntimeError: CUDA unknown
  error ... Failed to get device handle for GPU 0`. The runtime preflight
  (`mem_get_info` + a real bf16 matmul, with `set -euo pipefail`) caught it in
  seconds instead of burning the slot — keep that guard in every GPU sbatch.
  SECOND-ORDER DAMAGE, which is the expensive part: a partially-failed ARRAY
  leaves every `--dependency=afterok:<array>` job in `DependencyNeverSatisfied`
  forever, so one dead node silently killed the other three stages of the fp32
  pipeline. After any array failure, check `squeue` for
  `DependencyNeverSatisfied` and resubmit the dependents, or gate stages on
  files-on-disk the way `nbest/fp32_sweeper.sh` does.
- **Hostnames are not Slurm node names.** `ma-compute-02` resolves as
  `en-cc-unicorn-compute-134` (both 128.84.97.228). A traceback naming an unfamiliar host
  may be a node you already know.
- **`sacct` returns REUSED job IDs**, and has returned another user's job for an ID we
  submitted (`sacct -j 470308` → a `pysr_eval` job from 2026-01-28). Use `squeue` plus the
  job's own output file as truth. Scripts should key off **files on disk**, as
  `nbest/fp32_sweeper.sh` does.
- **`sbatch --wrap` expands `$VARS` in the OUTER shell** — `$SLURM_ARRAY_TASK_ID` becomes
  empty at submit time and every array task takes the same branch. Use script files.
- **`--constraint=6000ada` selects the NODE, not the GPU**; a node can hand you a different
  card. Add a runtime guard asserting the device name. **Bit again 2026-08-24**: job
  **572394** was submitted with `--constraint="a6000|6000ada|a100|a40"` and was handed an
  **RTX 2080 Ti (10.6 GiB)**. The reliable mechanism is a TYPED GRES request --
  `--gres=gpu:nvidia_rtx_a6000:1` -- which asks for the card, not a node label; the
  constraint string is at best a hint. Keep the runtime guard anyway: it turned this into
  a 4-second failure instead of an OOM (or, on an 11 GiB card, a
  `no kernel image is available` death) hours in.
- **Backfill economics.** A 12 h walltime cannot fit into a backfill gap, and a large `-c`
  ask disqualifies most nodes that have free GPUs. Shorter walltime + fewer cores often
  starts hours sooner. **CPU, not GPU, is frequently the binding constraint.**
- **torch 2.11.0+cu128 includes sm_120**, so the RTX PRO 6000 Blackwell Max-Q (96 GB) cards
  on `jjs533-compute-03` are usable (verified bf16 matmul, probe 475342).
  `torch.cuda.get_arch_list()` returns `[]` on a login node — it proves nothing there.

## Archived to /share/ellis (2026-08-31)

The NFS home export hit 99% full. Everything in this repo **untouched since
2026-08-01** (a 30-day rule) was migrated to `/share/ellis/wjl86/anticipation3/`.
**Nothing was deleted.** Every directory was copied, verified, and only then removed
from home.

**The old paths still work.** Each migrated directory was replaced by a symlink at its
original location, so scripts, configs and generated manifests resolve unchanged:
`run_nodummy/final/model.safetensors` still opens from `anticipation3/run_nodummy/...`.

### What moved — 184.6 GiB, 19 directories

| directory | GiB | files |
|---|---|---|
| `run_genai` | 22.81 | 206 |
| `finale2` | 22.78 | 85 |
| `.marchsmoke3` | 22.78 | 17 |
| `.marchsmoke4` | 22.78 | 17 |
| `run_nodummy_v2_b048` | 12.09 | 196 |
| `run_nodummy_v2` | 12.08 | 122 |
| `.marchsmoke` | 12.06 | 9 |
| `run_nodummy` | 10.07 | 166 |
| `finale` | 9.38 | 35 |
| `run_nodummy_lora_r512_b048` | 8.45 | 201 |
| `run_nodummy_lora_r512` | 8.43 | 113 |
| `run_lora` | 7.45 | 215 |
| `aug_labels_v2` | 4.02 | 6 |
| `jitter` | 2.68 | 10 |
| `checkpoint-750` | 1.34 | 5 |
| `highreg_90` | 1.34 | 5 |
| `aug_labels` | 1.34 | 2 |
| `checkpoint-1000` | 1.34 | 1 |
| `checkpoint-1750` | 1.34 | 1 |

### Verification method

Per directory: `rsync -a` -> compare **apparent bytes** (`du -sb`) **and** file count
-> only on exact match `rm -rf` the source and `ln -s` the target. A mismatch aborted
that directory with its source left intact. No directory failed.

### Gotchas

- **`du -sh` on the share reads larger than the source** (33G vs 23G for `run_genai`).
  That is allocated-vs-apparent size — `/share/ellis` uses a larger block size.
  Apparent bytes match exactly; the data is fine.
- **`du -sh ~` no longer counts this data**, since symlinks are not followed.
  Use `du -shL` to include archived sizes.
- **Do not `rm` a symlink target** assuming it is a stale duplicate. It is the only copy.
- **SLURM captures a job script at submit time**, so editing an `.sbatch` does not
  affect already-queued jobs. Nothing referenced by a running or pending job was moved.
  Excluded on those grounds: `run_paper_split_v2` (the shared base checkpoint that all
  four active chains load as `--model_name`), `run_paper_split_v2_maskft_muon40k_v2`,
  `run_soup_seed*`, `run_bayes_lora_r512`, `run_paper_split_v2_fisher_sam`,
  `run_paper_split_lora_r512`.
- **`.marchsmoke3` and `.marchsmoke4` have identical byte totals and file lists.**
  They are NOT duplicates — same architecture and checkpoint schedule produce
  identically-sized files. mtimes differ (2026-03-15 vs 03-16); the weights differ.

## SLURM: prefer the `gpu` partition over `default_partition` (2026-08-31)

`gpu` and `default_partition` contain the **same nodes**, but `gpu` is
`PriorityTier=15` against `default_partition`'s `10`. Same hardware, strictly
better scheduling: it queues ahead of default_partition work and cannot be
preempted by it. (`ellis` and `thickstun` are tier 20 but hold few GPUs, and ours
are usually busy with training.)

Always submit shared-pool work as:

    #SBATCH --partition=gpu,default_partition

listing both so Slurm takes whichever frees first. Same for the `-interactive`
pair. `gpu` sets `DenyAccounts=digitalgreen`, which does not affect account
`thickstun`.

Measured: an 8-task OCR array plus a TTS job sat `PENDING(Priority)` on
default_partition with 14 free 3090s cluster-wide; resubmitted to
`gpu,default_partition` they all started within 45 s.

**Corollary for smoke tests:** after `sbatch`, check `squeue -h -j <id> -o %T`
within ~30-60 s. `PENDING(Priority)` or `PENDING(ReqNodeNotAvail)` is a
scheduling problem to fix by resubmitting elsewhere, not a result to wait out.
