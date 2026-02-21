#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 7 ]; then
  echo "Usage:"
  echo "  bash scripts/run_grpo2_curriculum.sh <model_id_or_checkpoint_A> <image_folder> <grpo2_gt_json> <grpo2_pred_json> <output_dir> <warmup_epochs> <total_epochs> [reward_module]"
  echo
  echo "Example:"
  echo "  bash scripts/run_grpo2_curriculum.sh \\"
  echo "    /path/to/checkpoint_A_merged \\"
  echo "    /path/to/images \\"
  echo "    /path/to/grpo2_gt.json \\"
  echo "    /path/to/grpo2_pred.json \\"
  echo "    /path/to/output/grpo2_run \\"
  echo "    1 3 src.train.reward_funcs_prompt2"
  exit 1
fi

MODEL_ID="$1"
IMAGE_FOLDER="$2"
GT_DATA_PATH="$3"
PRED_DATA_PATH="$4"
OUTPUT_DIR="$5"
WARMUP_EPOCHS="$6"
TOTAL_EPOCHS="$7"
REWARD_MODULE="${8:-src.train.reward_funcs}"

# Training defaults (override via environment variables if needed).
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero2.json}"
USE_LIGER_LOSS="${USE_LIGER_LOSS:-True}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-True}"
FREEZE_LLM="${FREEZE_LLM:-True}"
FREEZE_MERGER="${FREEZE_MERGER:-False}"
BF16="${BF16:-False}"
FP16="${FP16:-True}"
DISABLE_FLASH_ATTN2="${DISABLE_FLASH_ATTN2:-False}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-768}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
IMAGE_MIN_PIXELS="${IMAGE_MIN_PIXELS:-100352}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-200704}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-False}"
REPORT_TO="${REPORT_TO:-tensorboard}"
LAZY_PREPROCESS="${LAZY_PREPROCESS:-True}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
REMOVE_UNUSED_COLUMNS="${REMOVE_UNUSED_COLUMNS:-False}"

if [ "${WARMUP_EPOCHS}" -gt "${TOTAL_EPOCHS}" ]; then
  echo "Error: warmup_epochs (${WARMUP_EPOCHS}) cannot be greater than total_epochs (${TOTAL_EPOCHS})."
  exit 1
fi

export PYTHONPATH=src:${PYTHONPATH:-}

if [ -f scripts/env_baseline_sft.sh ]; then
  # Reuse existing cache/env setup if available.
  source scripts/env_baseline_sft.sh
fi

if command -v deepspeed >/dev/null 2>&1; then
  DS_LAUNCHER="deepspeed"
elif python -c "import deepspeed" >/dev/null 2>&1; then
  DS_LAUNCHER="python -m deepspeed"
else
  echo "Error: DeepSpeed is not available in this environment."
  exit 1
fi

COMMON_ARGS=(
  --deepspeed "${DEEPSPEED_CONFIG}"
  --use_liger_loss "${USE_LIGER_LOSS}"
  --model_id "${MODEL_ID}"
  --image_folder "${IMAGE_FOLDER}"
  --freeze_vision_tower "${FREEZE_VISION_TOWER}"
  --freeze_llm "${FREEZE_LLM}"
  --freeze_merger "${FREEZE_MERGER}"
  --bf16 "${BF16}"
  --fp16 "${FP16}"
  --disable_flash_attn2 "${DISABLE_FLASH_ATTN2}"
  --output_dir "${OUTPUT_DIR}"
  --num_generations "${NUM_GENERATIONS}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --max_completion_length "${MAX_COMPLETION_LENGTH}"
  --max_prompt_length "${MAX_PROMPT_LENGTH}"
  --image_min_pixels "${IMAGE_MIN_PIXELS}"
  --image_max_pixels "${IMAGE_MAX_PIXELS}"
  --learning_rate "${LEARNING_RATE}"
  --remove_unused_columns "${REMOVE_UNUSED_COLUMNS}"
  --weight_decay "${WEIGHT_DECAY}"
  --warmup_ratio "${WARMUP_RATIO}"
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
  --logging_steps "${LOGGING_STEPS}"
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
  --report_to "${REPORT_TO}"
  --lazy_preprocess "${LAZY_PREPROCESS}"
  --save_strategy "${SAVE_STRATEGY}"
  --save_steps "${SAVE_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --reward_module "${REWARD_MODULE}"
)

echo "[Phase 1/2] GRPO-2 warmup on GT-conditioned regions"
echo "data_path=${GT_DATA_PATH}"
echo "num_train_epochs=${WARMUP_EPOCHS}"

${DS_LAUNCHER} src/train/train_grpo.py \
  "${COMMON_ARGS[@]}" \
  --data_path "${GT_DATA_PATH}" \
  --num_train_epochs "${WARMUP_EPOCHS}"

if [ "${WARMUP_EPOCHS}" -eq "${TOTAL_EPOCHS}" ]; then
  echo "Curriculum complete in phase 1 (warmup_epochs == total_epochs)."
  exit 0
fi

echo "[Phase 2/2] GRPO-2 continuation on GRPO-1 predicted regions"
echo "data_path=${PRED_DATA_PATH}"
echo "num_train_epochs=${TOTAL_EPOCHS} (resume from checkpoints in ${OUTPUT_DIR})"

${DS_LAUNCHER} src/train/train_grpo.py \
  "${COMMON_ARGS[@]}" \
  --data_path "${PRED_DATA_PATH}" \
  --num_train_epochs "${TOTAL_EPOCHS}"
