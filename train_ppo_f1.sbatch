#!/bin/bash
#SBATCH --job-name=ppo-f1-unfiltered
#SBATCH --partition=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/ppo_f1_%j.out
#SBATCH --error=logs/ppo_f1_%j.out

set -euo pipefail

cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

: "${RUN_DIR:?RUN_DIR must name a fresh output directory}"

# lr 3e-7 is the value the previous PPO arm ran stably at; the shared parser's
# 1e-5 default collapsed F1 within three updates in the single-GPU smoke test.
LR="${LR:-3e-7}"
MAX_STEPS="${MAX_STEPS:-6000}"
KL_COEF="${KL_COEF:-0.05}"
# 0/0, matching ppo_corrected. An ACTION here is a whole triplet, so its F1 delta
# is an immediate reward and the advantage is r_t - V(s_t) with no bootstrap --
# the critic gets back the one-step regression target that reached EVgae ~0.32.
GAMMA="${GAMMA:-0.0}"
GAE_LAMBDA="${GAE_LAMBDA:-0.0}"

echo "PPO-F1 job ${SLURM_JOB_ID} on $(hostname); start $(date --iso-8601=seconds)"
echo "run_dir=${RUN_DIR} lr=${LR} kl_coef=${KL_COEF} gamma=${GAMMA} lambda=${GAE_LAMBDA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u -m torch.distributed.run --standalone --nproc_per_node=4 \
    train_ppo_f1.py \
    --data_file data/train_paper_unfiltered.txt \
    --val_file data/val_paper_unfiltered.txt \
    --init_checkpoint run_paper_split_v2/checkpoint-2500 \
    --output_dir "${RUN_DIR}" \
    --wandb_run_name "ppo_f1_unfiltered_${SLURM_JOB_ID}" \
    --wandb_mode online \
    --learning_rate "${LR}" \
    --max_steps "${MAX_STEPS}" \
    --prompts_per_step 8 \
    --micro_batch 4 \
    --rollout_temperature 0.7 \
    --ppo_epochs 1 \
    --gamma "${GAMMA}" \
    --gae_lambda "${GAE_LAMBDA}" \
    --clip_eps 0.2 \
    --vf_coef 0.5 \
    --value_lr 3e-5 \
    --value_warmup_steps 10 \
    --target_kl 0.02 \
    --advantage_clip 5.0 \
    --log_ratio_clip 5.0 \
    --kl_coef "${KL_COEF}" \
    --reward_scale 100.0 \
    --onset_jitter_std 0.05 \
    --dur_jitter_range 0.05 \
    --transpose_range_semitones 12 \
    --tempo_scale_range 0.2 \
    --max_grad_norm 1.0 \
    --eval_steps 25 \
    --save_steps 250 \
    --eval_windows 96 \
    --val_batch_size 8

echo "PPO-F1 job ${SLURM_JOB_ID} end $(date --iso-8601=seconds)"
