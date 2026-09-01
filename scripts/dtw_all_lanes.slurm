#!/bin/bash
#SBATCH -J dtw-lanes
#SBATCH -p thickstun
#SBATCH -A thickstun
#SBATCH -c 128
#SBATCH --mem=64G
#SBATCH -t 3:00:00
#SBATCH -o /home/wjl86/anticipation3/logs/dtw_lanes_%j.out
set -uo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh; conda activate base
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
run() { python nbest/dtw_drift_analysis.py --dir="$1" --out="nbest_data/dtw_lane_$2.json" ${3:+--paper=$3} & }
G=fullsong_rollouts/test/slide
run fullsong_rollouts/test/reset             base_rst
run fullsong_rollouts/test/slide_gtctx       base_gtc
run fullsong_rollouts/test/slide_pforce      base_pf
run fullsong_rollouts_maskft25/test/slide    m25_s69
run fullsong_rollouts_maskft25/test/reset    m25_rst
run fullsong_rollouts_maskft25/test/slide_pforce m25_pf
run fullsong_rollouts_maskft50/test/slide    m50_s69
run fullsong_slide1_mask50/test/slide        m50_s1
run fullsong_rollouts_maskft50/test/reset    m50_rst
run fullsong_rollouts_maskft50/test/slide_pforce m50_pf
run fullsong_rollouts_maskft/test/slide      m75_s69
run fullsong_rollouts_maskft/test/reset      m75_rst
run fullsong_rollouts_maskft/test/slide_gtctx m75_gtc
run fullsong_rollouts_maskft/test/slide_pforce m75_pf
run $G beyer fullsong_papers/test/paper2
run $G zeng  fullsong_papers/test/paper1
wait
echo DTW_LANES_DONE
