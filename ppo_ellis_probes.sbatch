#!/bin/bash
#SBATCH --job-name=ppo-probe
#SBATCH --partition=ellis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/ppo_ellis_probe_%A_%a.out
#SBATCH --error=logs/ppo_ellis_probe_%A_%a.out

set -euo pipefail

cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

probe_name=""
probe_lr=""
probe_epochs=""
probe_gamma=""
probe_lambda=""
probe_target_kl=""

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        probe_name="core_v2"
        probe_lr="1e-6"
        probe_epochs="4"
        probe_gamma="1.0"
        probe_lambda="0.95"
        probe_target_kl="0.05"
        ;;
    1)
        probe_name="stable_v2"
        probe_lr="3e-7"
        probe_epochs="2"
        probe_gamma="1.0"
        probe_lambda="0.95"
        probe_target_kl="0.02"
        ;;
    2)
        probe_name="immediate_v2"
        probe_lr="3e-7"
        probe_epochs="2"
        probe_gamma="0.0"
        probe_lambda="0.0"
        probe_target_kl="0.02"
        ;;
    *)
        echo "Unexpected array index ${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
        ;;
esac

echo "Probe ${probe_name} on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-unset}, start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u train_ppo.py \
    --data_file data/train_paper.txt \
    --val_file data/val_paper.txt \
    --init_checkpoint run_paper_split_v2/checkpoint-2500 \
    --output_dir "debug/ppo_ellis/${probe_name}" \
    --wandb_run_name "ppo_ellis_${probe_name}" \
    --wandb_mode online \
    --learning_rate "${probe_lr}" \
    --max_steps 150 \
    --prompts_per_step 24 \
    --micro_batch 4 \
    --rollout_temperature 0.7 \
    --ppo_epochs "${probe_epochs}" \
    --gamma "${probe_gamma}" \
    --gae_lambda "${probe_lambda}" \
    --clip_eps 0.2 \
    --vf_coef 0.5 \
    --value_lr 3e-5 \
    --value_warmup_steps 10 \
    --target_kl "${probe_target_kl}" \
    --advantage_clip 5.0 \
    --max_grad_norm 1.0 \
    --eval_steps 25 \
    --save_steps 10000 \
    --eval_windows 96 \
    --val_batch_size 8 \
    --no_save_best_val \
    --no_save_final

echo "Probe ${probe_name} end $(date)"
