#!/bin/bash
#SBATCH --account=OD-237777
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/logs/%x_%j.out
#SBATCH --error=/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/logs/%x_%j.err

set -eo pipefail

REPO=${REPO:-/scratch3/che489/Ha/interspeech/training/Qwen-VL-Series-Finetune}
RUN_ROOT=${RUN_ROOT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT}

mkdir -p "${RUN_ROOT}/logs"
cd "${REPO}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found."
  exit 1
fi

conda activate deepfake || {
  echo "ERROR: failed to activate env 'deepfake'"
  conda env list
  exit 1
}

source scripts/env_baseline_sft.sh

if [ -z "${MODEL_PATH:-}" ] || [ -z "${MODEL_BASE:-}" ] || [ -z "${SAVE_MODEL_PATH:-}" ]; then
  echo "Required env vars: MODEL_PATH, MODEL_BASE, SAVE_MODEL_PATH"
  echo "Example:"
  echo "  sbatch --export=MODEL_PATH=/path/to/sft_lora,MODEL_BASE=Qwen/Qwen2.5-VL-3B-Instruct,SAVE_MODEL_PATH=/path/to/sft_merged stage2_merge_sft_lora.slurm"
  exit 1
fi

python src/merge_lora_weights.py \
  --model-path "${MODEL_PATH}" \
  --model-base "${MODEL_BASE}" \
  --save-model-path "${SAVE_MODEL_PATH}" \
  --safe-serialization

