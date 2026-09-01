#!/bin/bash
#SBATCH -J fp32table
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -c 8
#SBATCH -t 12:00:00
#SBATCH -o /home/wjl86/anticipation3/logs/fp32table_%j.out
# FINAL STAGE: decode the 24-window pool in FULL precision and select with the
# fp32-trained selectors, then merge under *_fp32 group names so the bf16 rows
# survive alongside for comparison.
#
# --fp32 now reaches all three places it has to: the FT decode autocast, the
# BASE model's weight dtype (it was cast to bf16 regardless before), and the
# per-token feature storage. The weights json is the refitted fp32 one; the
# bf16 alpha/beta/gamma are fitted to different feature scales.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh; conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python3 -c "import torch;t=torch.cuda.mem_get_info()[1]/2**30;n=torch.cuda.get_device_name(0);print(f'{n} {t:.1f}GiB');assert t>=20,f'refusing {n}'"

R=run_nbest_reranker
# EVERY checkpoint this script touches, including the three used only by the
# per-arm qsel loop at the bottom -- listwise32_fp32 and pairwise_fp32 were
# missing here, so a gap in them would have failed an hour into the run
# instead of in the first second.
for f in $R/pairwise32_fp32/final.pt $R/pairwise32feat_fp32/final.pt \
         $R/duel32_fp32/final.pt $R/genrm_fp32/final.pt \
         $R/listwise32_fp32/final.pt $R/pairwise_fp32/final.pt \
         $R/pointwise_fp32/final.pt nbest_data/decode_weights_fp32.json; do
  test -f "$f" || { echo "MISSING: $f"; exit 1; }
done

OUT=visualizer/rerank_fp32_shards; mkdir -p $OUT
python3 -u visualizer/rerank_sample_viz.py \
    --checkpoint run_paper_split_v2/checkpoint-2500 \
    --variant raw --n-sampled 8 --seed 0 --fp32 \
    --weights nbest_data/decode_weights_fp32.json \
    --output            $OUT/fitted.json \
    --output-oracle     $OUT/oracle.json \
    --output-sample1    $OUT/sample1.json \
    --output-mbr        $OUT/mbr.json \
    --qsel-reranker     $R/pairwise32feat_fp32/final.pt --output-qsel       $OUT/pw32feat.json \
    --duel-ckpt         $R/duel32_fp32/final.pt         --output-tournament $OUT/duel32.json \
    --genrm-ckpt        $R/genrm_fp32/final.pt          --output-genrm      $OUT/genrm.json

# pairwise32 / listwise32 / pairwise need their own qsel passes (one reranker
# per --qsel-reranker), sharing the SAME pool via the same seed.
for arm in pairwise32 listwise32 pairwise; do
  python3 -u visualizer/rerank_sample_viz.py \
      --checkpoint run_paper_split_v2/checkpoint-2500 \
      --variant raw --n-sampled 8 --seed 0 --fp32 \
      --weights nbest_data/decode_weights_fp32.json \
      --output $OUT/fitted_${arm}_dup.json \
      --qsel-reranker $R/${arm}_fp32/final.pt --output-qsel $OUT/${arm}.json
done

echo "=== merging under *_fp32 group names ==="
M="python3 visualizer/merge_rerank_rollouts.py --data visualizer/data.js --shards"
$M $OUT/fitted.json      --group rollouts_fitted_fp32
$M $OUT/oracle.json      --group rollouts_oracle_fp32
$M $OUT/sample1.json     --group rollouts_sample_t1_fp32
$M $OUT/mbr.json         --group rollouts_mbr_fp32
$M $OUT/pw32feat.json    --group rollouts_rerank_pw32feat_fp32
$M $OUT/duel32.json      --group rollouts_duel32_fp32
$M $OUT/genrm.json       --group rollouts_genrm_fp32
$M $OUT/pairwise32.json  --group rollouts_rerank_pairwise32_fp32
$M $OUT/listwise32.json  --group rollouts_rerank_listwise32_fp32
$M $OUT/pairwise.json    --group rollouts_rerank_pairwise_fp32

python3 visualizer/compute_f1.py --data visualizer/data.js
python3 visualizer/split_visualizer_payload.py || true
python3 visualizer/nbest_f1_table.py
echo FP32TABLE_OK
