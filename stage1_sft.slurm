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

# --- Conda init (robust) ---
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found. Set correct path in stage1_sft.slurm."
  exit 1
fi

conda activate deepfake || {
  echo "ERROR: failed to activate env 'deepfake'"
  conda env list
  exit 1
}

source scripts/env_baseline_sft.sh

if [ -z "${MODEL_ID:-}" ]; then
  echo "MODEL_ID is required"
  exit 1
fi

MODEL_TAG="$(basename "${MODEL_ID%/}")"

echo "Starting model: ${MODEL_ID}"
bash scripts/run_stage1_sft_from_config.sh "${MODEL_ID}" \
  2>&1 | tee "${RUN_ROOT}/logs/train_${MODEL_TAG}_${SLURM_JOB_ID}.log"
