#!/usr/bin/env bash

# Activate Jiali Wang's persistent Qwen3.5 environment in Cybertron notebooks.
export ORENA_USER_ROOT="/storage/main/users/jialiwang"
export ORENA_QWEN35_ENV="$ORENA_USER_ROOT/envs/orena-qwen35"

export HOME="$ORENA_USER_ROOT/home"
export USER="jialiwang"
export LOGNAME="jialiwang"

export XDG_CACHE_HOME="$ORENA_USER_ROOT/cache"
export HF_HOME="$ORENA_USER_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PIP_CACHE_DIR="$ORENA_USER_ROOT/cache/pip"
export TORCHINDUCTOR_CACHE_DIR="$ORENA_USER_ROOT/cache/torchinductor"
export TORCH_EXTENSIONS_DIR="$ORENA_USER_ROOT/cache/torch-extensions"
export TMPDIR="$ORENA_USER_ROOT/tmp"

export CUDA_HOME="$ORENA_QWEN35_ENV"
export PATH="$ORENA_QWEN35_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ORENA_QWEN35_ENV/lib:$ORENA_QWEN35_ENV/lib64:$ORENA_QWEN35_ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p \
  "$HOME" \
  "$HF_DATASETS_CACHE" \
  "$PIP_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TMPDIR"

if [[ ! -x "$ORENA_QWEN35_ENV/bin/python" ]]; then
  echo "Missing Cybertron Qwen3.5 environment: $ORENA_QWEN35_ENV" >&2
  return 1 2>/dev/null || exit 1
fi
