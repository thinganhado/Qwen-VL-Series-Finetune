#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT"

export TRITON_CACHE_DIR="${RUN_ROOT}/cache/triton"
export HF_HOME="${RUN_ROOT}/cache/huggingface"
export TORCH_HOME="${RUN_ROOT}/cache/torch"
export XDG_CACHE_HOME="${RUN_ROOT}/cache/xdg"
export TRANSFORMERS_CACHE="${RUN_ROOT}/cache/huggingface/transformers"
export PIP_CACHE_DIR="${RUN_ROOT}/cache/pip"

mkdir -p \
  "${RUN_ROOT}" \
  "${TRITON_CACHE_DIR}" \
  "${HF_HOME}" \
  "${TORCH_HOME}" \
  "${XDG_CACHE_HOME}" \
  "${TRANSFORMERS_CACHE}" \
  "${PIP_CACHE_DIR}"

