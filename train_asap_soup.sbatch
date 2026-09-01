#!/bin/bash
#SBATCH --job-name=soup-ladder
#SBATCH --partition=ellis
#SBATCH --account=thickstun
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=UNLIMITED
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/wjl86/anticipation3/logs/soup_%j.out
#
# EXPERIMENT 2 -- MODEL SOUPS (Wortsman et al., ICML 2022), 2026-08-30.
#
# Ten fine-tunes of run_paper_split_v2/checkpoint-2500 that differ ONLY in
# --seed (data order, augmentation draws, dropout), then average the WEIGHTS
# of the ten best-validation checkpoints. The paper's point is that models
# fine-tuned from one shared initialization stay in one linearly-connected
# basin, so their weight average is a real model -- and that the GREEDY soup
# (sort by val score, fold a model in only if it does not hurt val) is the
# variant that reliably matches or beats the best individual member, where
# the uniform soup can be dragged down by one bad ingredient. soup.py builds
# both; the greedy one is what gets evaluated.
#
# ---------------------------------------------------------------------------
# HONEST COST. This is the expensive part of the experiment and it is serial.
#
#   These are MEASURED, not guessed: job 937514 ran 30 steps of this exact
#   recipe on one RTX 3090 (campbell-compute-01) before this was submitted.
#
#   one run  : 5000 optimizer steps at effective batch 64 on ONE 3090
#              (batch 8 x grad_accum 8 -- the repo-wide eff-64 invariant).
#              MEASURED 5.55 s/step steady state (29 steps in 2:40 after
#              compile; tqdm settled at 5.56 s/it) => 7.7 h of pure training.
#              First step costs 2:15 of torch.compile, and the first
#              validation costs another ~50 s of eval-shape recompile.
#              11 validations fire per run (eval_steps 500 at 500..4500,
#              checkpoint validations at 1000..5000, plus the final one), all
#              measured on the probe: teacher-forced 500 windows at 3.6 it/s
#              = ~18 s, autoregressive 100 windows at 7.7 s per 8-window batch
#              = ~100 s, i.e. ~2.0 min per validation => ~22 min/run.
#              One-time compiles per run: 135 s (train step 1) + ~50 s (eval
#              shape) + 257 s (the constrained AR decode path) + 46 s (its
#              ragged tail-batch shape) = ~8 min.
#              Five 1.4 GB checkpoint saves and one 12.3 GB offset scan of
#              train_paper.txt add ~5 min.
#              => ~8.3 h per seed.
#
#   ten runs : ~83 h SERIAL = ~3.5 DAYS of wall clock, and that is pure
#              compute -- queue wait on top.
#   souping  : + ~15-20 min (soup.py: 20 checkpoint loads + ~20 scored soups).
#   TOTAL    : ~3.5-4 days. There is no partial-credit answer before then;
#              the greedy soup needs all ten members to exist.
#
#   DISK: 10 runs x (5 checkpoints + final) x 1.44 GB = ~86 GB. /home had
#   565 GB free at submit time. Do not delete run_soup_seed*/ before the
#   soup has been built and evaluated.
#
# The loop is RESUMABLE: a seed whose output dir already holds final/ is
# skipped, so a requeue or a manual resubmit continues the ladder instead of
# redoing finished seeds. Each run's stdout is teed to
# run_soup_seed$SEED/train.log so the best-checkpoint pick survives a requeue
# (the SLURM log would not: the picks are parsed per run, from its own log).
#
# ONE 3090 via TYPED gres, not `gpu:1`: a bare gpu:1 request is a node label,
# and this cluster has handed such jobs an RTX 2080 Ti (job 572394). The
# runtime preflight below re-checks the card by name anyway, plus a real bf16
# matmul -- ma-compute-02 / kuleshov-compute-01 / badfellow style node-wide
# NVML failures make device_count() lie, and the matmul turns that into a
# 4-second failure instead of a burned slot.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh; conda activate base
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

echo "=== soup ladder job ${SLURM_JOB_ID:-local} on $(hostname) at $(date) ==="

# --- runtime preflight: real bf16 matmul + assert we got a 3090 -------------
python -c "
import torch
n = torch.cuda.device_count()
assert n == 1, f'expected 1 GPU, got {n}'
name = torch.cuda.get_device_name(0)
free, total = torch.cuda.mem_get_info()
a = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16)
val = float((a @ a).float().sum())
assert val == val, 'bf16 matmul produced NaN'
assert '3090' in name, f'refusing {name}: this recipe is sized for a 3090'
print(f'preflight OK: {name}, {total/2**30:.1f} GiB total, bf16 matmul live')
"

# --- guard the souping code BEFORE burning four days on the ladder ---------
python -u soup.py --selftest

SEEDS=$(seq 1 10)
RUN_DIRS=()

for SEED in $SEEDS; do
    OUT="./run_soup_seed${SEED}"
    RUN_DIRS+=("$OUT")

    # Resume: a run that already wrote final/ is finished, skip it.
    if [ -f "$OUT/final/model.safetensors" ]; then
        echo "=== [seed $SEED] final/ already present -> SKIP (resume) ==="
        continue
    fi

    echo "=== [seed $SEED] starting at $(date) ==="
    mkdir -p "$OUT"
    python -u train.py \
        --data_file data/train_paper.txt \
        --val_file data/val_paper.txt \
        --model_name run_paper_split_v2/checkpoint-2500 \
        --seed "$SEED" \
        --max_steps 5000 \
        --batch_size 8 \
        --gradient_accumulation_steps 8 \
        --val_batch_size 8 \
        --save_steps 1000 \
        --eval_steps 500 \
        --output_dir "$OUT" \
        --wandb_run_name "run_soup_seed${SEED}" \
        --wandb_mode online 2>&1 | tee "$OUT/train.log"
    echo "=== [seed $SEED] finished at $(date) ==="
done

echo "=== all ten runs present; selecting each run's best checkpoint ==="
# soup.py's pick_best_checkpoint() parses each run's OWN train.log, pairing
# every "Running validation at checkpoint step N" with the "Validation Loss:"
# that follows it, and takes the argmin over the checkpoints that exist on
# disk. The LAST checkpoint is only a warned-about fallback, never an
# assumption.
#
# Caveat, stated because it bounds this stage: train.py resamples its 500
# validation windows on every eval (see "seed=..." in the eval banner), so a
# logged val/loss carries sampling noise and the within-run argmin is only
# approximately the best checkpoint. That noise does NOT leak into the soup
# decisions: soup.py re-scores every selected checkpoint itself, on one FIXED
# subset, so the greedy ordering and the accept/reject tests are exactly
# paired.
python -u soup.py --run_dirs "${RUN_DIRS[@]}" --print_best

echo "=== souping at $(date) ==="
python -u soup.py \
    --run_dirs "${RUN_DIRS[@]}" \
    --val_file data/val_paper.txt \
    --val_windows 384 \
    --val_batch_size 4 \
    --device cuda \
    --out_dir ./run_soup_greedy \
    --uniform_out_dir ./run_soup_uniform \
    --results_file results/model_soup.txt

echo "SOUP_DONE $(date)"
