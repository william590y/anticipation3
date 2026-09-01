#!/bin/bash
#SBATCH -J papertest
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -c 8
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH -A thickstun
#SBATCH -o /home/wjl86/anticipation3/logs/papertest_%j.out
# Beyer & Dai and Zeng+ on the SAME 1,181 test windows as our selectors.
# run_paper_models.py caches transcriptions per PIECE, so 1,181 windows across
# 14 works costs 14 transcriptions plus cheap slices.
#
# paperpipe env is mandatory: paper 2 needs TimFelixBeyer's music21 FORK, which
# in `base` would silently change precompute_scores_xml.py's engraving.
# The decode path is reused UNMODIFIED -- its four documented traps (shared
# representation, bucketed pitch, per-repo output shape, <PAD>) all live there
# and each fails silently into plausible output rather than erroring.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate paperpipe
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python3 -c "import torch;t=torch.cuda.mem_get_info()[1]/2**30;n=torch.cuda.get_device_name(0);print(f'{n} {t:.1f}GiB');assert t>=20,f'refusing {n}'"
python3 -u visualizer/run_paper_models.py --data visualizer/data_testset.js
echo "=== scoring both paper rows ==="
conda activate base
python3 -u visualizer/compute_f1.py --data visualizer/data_testset.js
echo PAPERTEST_OK
