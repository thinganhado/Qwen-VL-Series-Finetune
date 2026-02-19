#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/run_stage1_query2_sft.sh <model_id> [config_json]"
  exit 1
fi

MODEL_ID="$1"
CONFIG_JSON="${2:-configs/stage1_sft_query2_lora_config.json}"

MULTITURN_TRAIN_JSON="${MULTITURN_TRAIN_JSON:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/SFT_2turn/stage1_multiturn_train.json}"
MULTITURN_VAL_JSON="${MULTITURN_VAL_JSON:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/SFT_2turn/stage1_multiturn_val.json}"
QUERY2_TRAIN_JSON="${QUERY2_TRAIN_JSON:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/SFT_2turn/query2_only_train.json}"
QUERY2_VAL_JSON="${QUERY2_VAL_JSON:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/SFT_2turn/query2_only_val.json}"

python build_stage1_query2_sft_json.py \
  --input-json "$MULTITURN_TRAIN_JSON" \
  --output-json "$QUERY2_TRAIN_JSON"

python build_stage1_query2_sft_json.py \
  --input-json "$MULTITURN_VAL_JSON" \
  --output-json "$QUERY2_VAL_JSON"

bash scripts/run_stage1_sft_from_config.sh "$MODEL_ID" "$CONFIG_JSON"

