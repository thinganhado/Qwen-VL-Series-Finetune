#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/run_stage1_sft_from_config.sh <model_id> [config_json]"
  echo "Example: bash scripts/run_stage1_sft_from_config.sh Qwen/Qwen2.5-VL-3B-Instruct"
  exit 1
fi

MODEL_ID="$1"
CONFIG_JSON="${2:-configs/stage1_sft_lora_config.json}"

export PYTHONPATH=src:${PYTHONPATH:-}

EXTRA_ARGS="$(
python - "$CONFIG_JSON" <<'PY'
import json
import shlex
import sys

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

parts = []
for k, v in cfg.items():
    flag = f"--{k}"
    if v is None:
        continue
    if isinstance(v, bool):
        parts.extend([flag, "True" if v else "False"])
    else:
        parts.extend([flag, str(v)])

print(" ".join(shlex.quote(x) for x in parts))
PY
)"

CMD="deepspeed src/train/train_sft.py --model_id \"$MODEL_ID\" $EXTRA_ARGS"
echo "$CMD"
eval "$CMD"

