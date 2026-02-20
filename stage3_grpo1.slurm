#!/bin/bash
#SBATCH --account=OD-237777
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/logs/%x_%j.out
#SBATCH --error=/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/logs/%x_%j.err

set -eo pipefail

REPO=${REPO:-/scratch3/che489/Ha/interspeech/training/Qwen-VL-Series-Finetune}
RUN_ROOT=${RUN_ROOT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT}
CONFIG_JSON=${CONFIG_JSON:-configs/grpo1_prompt1_config.json}

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

if [ -z "${MODEL_ID:-}" ]; then
  echo "MODEL_ID is required (should be merged SFT model path)"
  echo "Example:"
  echo "  sbatch --export=MODEL_ID=/path/to/sft_merged stage3_grpo1.slurm"
  exit 1
fi

bash scripts/run_grpo_from_config.sh "${MODEL_ID}" "${CONFIG_JSON}" \
  2>&1 | tee "${RUN_ROOT}/logs/grpo1_${SLURM_JOB_ID}.log"
