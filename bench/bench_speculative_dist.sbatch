#!/bin/bash
#SBATCH --job-name=bench-spec-dist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=ellis
#SBATCH --nodelist=ellis-compute-01
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/bench_spec_dist_%j.out
#SBATCH --error=logs/bench_spec_dist_%j.out
# Distributional exactness only. Deliberately NOT pinned to the timing node:
# this mode measures a *law*, not a wall clock, so the GPU model is irrelevant
# and a 3090 is fine.
#
# Runs on `ellis` (our own partition) rather than the shared default partition
# because the default partition PREEMPTED this exact mode twice: job 463991 on
# zabih-compute-01 and job 466426 on nikola-compute-18, the latter after only
# 8 minutes.  Those nodes are owned by other groups and guest jobs lose.  On
# ellis-compute-01 we are not a guest.  ellis-compute-02 is the 8xA6000 node
# and is fully allocated to other jobs; ellis-compute-01 is the 5x3090 node,
# so the GPU TYPE must be pinned (ellis mixes two node types).
#
# Args go on the COMMAND LINE, not through --export: SLURM splits an --export
# value on commas, which silently truncates any comma-separated list (this is
# the documented trap that produced one bogus baseline table, job 462074).
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name --format=csv,noheader
python -m bench.bench_speculative "$@"
