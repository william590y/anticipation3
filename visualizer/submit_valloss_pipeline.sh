#!/bin/bash
# End-to-end best-val-loss viz pipeline on thickstun (7 GPUs).
#   bash visualizer/submit_valloss_pipeline.sh
set -euo pipefail
cd /home/wjl86/anticipation3
mkdir -p logs visualizer/valloss_shards visualizer/valloss_beam_shards visualizer/ppl_shards

ROLL=$(sbatch --parsable visualizer/precompute_valloss_rollouts.sbatch)
echo "rollouts array: $ROLL"

MERGE1=$(sbatch --parsable --dependency=afterok:${ROLL} visualizer/merge_valloss_rollouts.sbatch)
echo "merge+F1:       $MERGE1"

BEAMS=$(sbatch --parsable --dependency=afterok:${MERGE1} visualizer/precompute_valloss_beams.sbatch)
echo "beams array:    $BEAMS"

MERGE2=$(sbatch --parsable --dependency=afterok:${BEAMS} visualizer/merge_valloss_beams.sbatch)
echo "beam merge:     $MERGE2"

PPL=$(sbatch --parsable --dependency=afterok:${MERGE2} visualizer/compute_sequence_ppl.sbatch)
echo "seq-PPL array:  $PPL"

FINAL=$(sbatch --parsable --dependency=afterok:${PPL} visualizer/merge_valloss_final.sbatch)
echo "final merge:    $FINAL"

echo "$ROLL $MERGE1 $BEAMS $MERGE2 $PPL $FINAL" | tee visualizer/valloss_shards/PIPELINE_JOBS.txt
echo "Pipeline submitted."
