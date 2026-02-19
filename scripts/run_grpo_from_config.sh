#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/run_grpo_from_config.sh <model_id> [config_json]"
  echo "Example: bash scripts/run_grpo_from_config.sh /path/to/sft_or_grpo_checkpoint configs/grpo1_prompt1_config.json"
  exit 1
fi

MODEL_ID="$1"
CONFIG_JSON="${2:-configs/grpo1_prompt1_config.json}"

export PYTHONPATH=src:${PYTHONPATH:-}

if [ -f scripts/env_baseline_sft.sh ]; then
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

BASE_OUTPUT_DIR="$(
python - "$CONFIG_JSON" <<'PY'
import json
import sys

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)
print(cfg.get("output_dir", "output/grpo_run"))
PY
)"

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

CMD="$DS_LAUNCHER src/train/train_grpo.py --model_id \"$MODEL_ID\" --output_dir \"$BASE_OUTPUT_DIR\" $EXTRA_ARGS"
echo "$CMD"
eval "$CMD"

