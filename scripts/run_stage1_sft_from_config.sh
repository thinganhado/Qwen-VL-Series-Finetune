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
RUN_ROOT="/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT"

export PYTHONPATH=src:${PYTHONPATH:-}
mkdir -p "${RUN_ROOT}"

# Force all caches under RUN_ROOT to avoid home quota issues.
export XDG_CACHE_HOME="${RUN_ROOT}/cache/xdg"
export TRITON_CACHE_DIR="${RUN_ROOT}/cache/triton"
export HF_HOME="${RUN_ROOT}/cache/huggingface"
export TORCH_HOME="${RUN_ROOT}/cache/torch"
export TRANSFORMERS_CACHE="${RUN_ROOT}/cache/huggingface/transformers"
mkdir -p "${TRITON_CACHE_DIR}"
mkdir -p "${HF_HOME}"
mkdir -p "${TORCH_HOME}"
mkdir -p "${TRANSFORMERS_CACHE}"
mkdir -p "${XDG_CACHE_HOME}"

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
RUN_OUTPUT_DIR="${RUN_ROOT}/$(basename "${RUN_OUTPUT_DIR}")"

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
