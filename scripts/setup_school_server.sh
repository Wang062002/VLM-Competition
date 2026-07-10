#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/workspace}"
OFFICIAL_REPO_DIR="${OFFICIAL_REPO_DIR:-$WORKSPACE_DIR/orena-focus}"
VENV_DIR="${VENV_DIR:-$WORKSPACE_DIR/.venvs/orena-focus}"
FOCUS_ROOT_DIR="${FOCUS_ROOT_DIR:-$HOME/data/focus}"
RUNS_DIR="${RUNS_DIR:-$WORKSPACE_DIR/focus-runs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OFFICIAL_REPO_URL="${OFFICIAL_REPO_URL:-https://github.com/IMSY-DKFZ/orena-focus.git}"

echo "== ORena FOCUS remote setup =="
echo "WORKSPACE_DIR=$WORKSPACE_DIR"
echo "OFFICIAL_REPO_DIR=$OFFICIAL_REPO_DIR"
echo "VENV_DIR=$VENV_DIR"
echo "FOCUS_ROOT_DIR=$FOCUS_ROOT_DIR"
echo "RUNS_DIR=$RUNS_DIR"

mkdir -p "$WORKSPACE_DIR" "$FOCUS_ROOT_DIR" "$RUNS_DIR" "$(dirname "$VENV_DIR")"

echo
echo "== System check =="
hostname || true
whoami || true
df -h "$HOME" "$FOCUS_ROOT_DIR" || true
nvidia-smi || echo "WARNING: nvidia-smi failed or no NVIDIA GPU is visible."
"$PYTHON_BIN" --version

echo
echo "== Official repository =="
if [ ! -d "$OFFICIAL_REPO_DIR/.git" ]; then
  git clone "$OFFICIAL_REPO_URL" "$OFFICIAL_REPO_DIR"
else
  echo "Repository already exists: $OFFICIAL_REPO_DIR"
fi
git -C "$OFFICIAL_REPO_DIR" rev-parse HEAD | tee "$RUNS_DIR/orena-focus-commit.txt"

echo
echo "== Python environment =="
if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

if [ -n "${TORCH_INSTALL_CMD:-}" ]; then
  echo "Running custom TORCH_INSTALL_CMD=$TORCH_INSTALL_CMD"
  eval "$TORCH_INSTALL_CMD"
else
  echo "Installing default torch/torchvision from pip. Set TORCH_INSTALL_CMD for a server-specific CUDA wheel if needed."
  python -m pip install torch torchvision
fi

python -m pip install -e "$OFFICIAL_REPO_DIR"
python -m pip install qwen-vl-utils transformers accelerate huggingface_hub

echo
echo "== Python GPU check =="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count:", torch.cuda.device_count())
    print("device_0:", torch.cuda.get_device_name(0))
PY

echo
echo "Done. Activate with:"
echo "source \"$VENV_DIR/bin/activate\""
echo "export FOCUS_ROOT_DIR=\"$FOCUS_ROOT_DIR\""
