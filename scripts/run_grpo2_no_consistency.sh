#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/run_grpo2_no_consistency.sh <model_id> [config_json]"
  echo "Example: bash scripts/run_grpo2_no_consistency.sh /path/to/checkpoint_A_merged"
  exit 1
fi

MODEL_ID="$1"
CONFIG_JSON="${2:-configs/grpo2_prompt2_no_consistency_config.json}"

bash scripts/run_grpo_from_config.sh "${MODEL_ID}" "${CONFIG_JSON}"

