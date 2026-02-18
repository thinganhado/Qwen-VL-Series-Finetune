#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/run_stage1_sft_from_config.sh <model_id> [config_json]"
  echo "Example: bash scripts/run_stage1_sft_from_config.sh Qwen/Qwen2.5-VL-3B-Instruct"
  exit 1
fi

MODEL_ID="$1"
CONFIG_JSON="${2:-configs/stage1_sft_lora_config.json}"
MODEL_TAG="$(basename "${MODEL_ID%/}")"

export PYTHONPATH=src:${PYTHONPATH:-}

BASE_OUTPUT_DIR="$(
python - "$CONFIG_JSON" <<'PY'
import json
import sys

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)
print(cfg.get("output_dir", "output/stage1_lora"))
PY
)"

RUN_OUTPUT_DIR="${BASE_OUTPUT_DIR}_${MODEL_TAG}"

if command -v deepspeed >/dev/null 2>&1; then
  DS_LAUNCHER="deepspeed"
elif python -c "import deepspeed" >/dev/null 2>&1; then
  DS_LAUNCHER="python -m deepspeed"
else
  echo "Error: DeepSpeed is not available in this environment."
  echo "Install with: pip install deepspeed"
  exit 1
fi

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
    if k == "output_dir":
        continue
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

CMD="$DS_LAUNCHER src/train/train_sft.py --model_id \"$MODEL_ID\" --output_dir \"$RUN_OUTPUT_DIR\" $EXTRA_ARGS"
echo "$CMD"
eval "$CMD"
