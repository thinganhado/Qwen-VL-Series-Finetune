#!/bin/bash
#SBATCH --account=OD-237777
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=09:59:00
#SBATCH --output=/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/logs/%x_%j.out
#SBATCH --error=/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/logs/%x_%j.err

set -eo pipefail

REPO=/scratch3/che489/Ha/interspeech/training/Qwen-VL-Series-Finetune
RUN_ROOT=/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT

mkdir -p "${RUN_ROOT}/logs"
cd "${REPO}"

source /scratch3/che489/Ha/.conda/etc/profile.d/conda.sh
conda activate deepfake
source scripts/env_baseline_sft.sh

if [ -z "${MODEL_ID:-}" ]; then
  echo "MODEL_ID is required"
  exit 1
fi

MODEL_TAG="$(basename "${MODEL_ID%/}")"

echo "Starting model: ${MODEL_ID}"
bash scripts/run_stage1_sft_from_config.sh "${MODEL_ID}" \
  2>&1 | tee "${RUN_ROOT}/logs/train_${MODEL_TAG}_${SLURM_JOB_ID}.log"
