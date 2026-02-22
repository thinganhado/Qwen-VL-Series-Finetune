#!/bin/bash
#SBATCH --account=OD-237777
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --time=16:00:00
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

if [ -z "${MODEL_ID:-}" ] || [ -z "${IMAGE_FOLDER:-}" ] || [ -z "${GT_DATA_PATH:-}" ] || [ -z "${PRED_DATA_PATH:-}" ] || [ -z "${OUTPUT_DIR:-}" ]; then
  echo "Required env vars: MODEL_ID IMAGE_FOLDER GT_DATA_PATH PRED_DATA_PATH OUTPUT_DIR"
  echo "Optional env vars: WARMUP_EPOCHS (default 1), TOTAL_EPOCHS (default 3), REWARD_MODULE (default src.train.reward_funcs_prompt2)"
  echo "Optional speed/LoRA vars: WARMUP_MAX_STEPS (default 120), TOTAL_MAX_STEPS (default 360), LORA_ENABLE (default True), LORA_RANK (default 8), LORA_ALPHA (default 32)"
  echo "Example:"
  echo "  sbatch --export=MODEL_ID=/path/to/grpo1_A_merged,IMAGE_FOLDER=/path/to/images,GT_DATA_PATH=/path/to/grpo2_gt.json,PRED_DATA_PATH=/path/to/grpo2_pred.json,OUTPUT_DIR=/path/to/grpo2_lora,WARMUP_EPOCHS=1,TOTAL_EPOCHS=3,WARMUP_MAX_STEPS=120,TOTAL_MAX_STEPS=360 stage5_grpo2_curriculum.slurm"
  exit 1
fi

WARMUP_EPOCHS=${WARMUP_EPOCHS:-1}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
REWARD_MODULE=${REWARD_MODULE:-src.train.reward_funcs_prompt2}

bash scripts/run_grpo2_curriculum.sh \
  "${MODEL_ID}" \
  "${IMAGE_FOLDER}" \
  "${GT_DATA_PATH}" \
  "${PRED_DATA_PATH}" \
  "${OUTPUT_DIR}" \
  "${WARMUP_EPOCHS}" \
  "${TOTAL_EPOCHS}" \
  "${REWARD_MODULE}" \
  2>&1 | tee "${RUN_ROOT}/logs/grpo2_curriculum_${SLURM_JOB_ID}.log"
