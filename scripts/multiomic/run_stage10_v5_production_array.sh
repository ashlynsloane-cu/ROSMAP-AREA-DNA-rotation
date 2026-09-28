#!/bin/bash
#SBATCH --job-name=stage10_v5
#SBATCH --array=1-25%5
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/home/assl5508/rosmap_multiomic_slurm/results/stage10_multimodal_simulations/slurm_logs/stage10_%A_%a.out
#SBATCH --error=/home/assl5508/rosmap_multiomic_slurm/results/stage10_multimodal_simulations/slurm_logs/stage10_%A_%a.err

set -euo pipefail

cd /home/assl5508/rosmap_multiomic_slurm

module purge
module load python/3.11.15
source /home/assl5508/venvs/rosmap-monocyte/bin/activate

echo "============================================================"
echo "Stage 10 v5 production"
echo "Replicate: ${SLURM_ARRAY_TASK_ID}"
echo "Job: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Python: $(which python)"
echo "============================================================"

python3 -u \
  scripts/multiomic/build_stage10_multimodal_simulations.py \
  --replicate-id "${SLURM_ARRAY_TASK_ID}" \
  --modality all

echo "============================================================"
echo "Replicate ${SLURM_ARRAY_TASK_ID} complete"
echo "Finished: $(date)"
echo "============================================================"
