# ORena FOCUS Baseline Runner

This repository contains a small runbook and scripts for getting the official
ORena FOCUS SEGMENT baseline running on a school GPU server through VS Code
Remote-SSH.

The intended first milestone is modest: connect to the remote server, verify
GPU/Python, install the official `orena-focus` package, prepare the `heico`
data, and run Qwen3-VL on a tiny validation smoke test before scaling up.

## Files

- `knowledge_base/START_HERE.md`: compact project memory entry point for future
  context recovery.
- `docs/remote_ssh_school_server.md`: step-by-step VS Code Remote-SSH guide.
- `scripts/setup_school_server.sh`: creates remote directories, clones the
  official repo, creates a venv, and installs dependencies.
- `scripts/prepare_heico_data.py`: downloads HeiCo data and optionally creates
  timestamp overlay videos / extracted frames.
- `scripts/check_focus_dataset.py`: verifies QA annotations and video clip
  generation.
- `scripts/run_segment_baseline.py`: configurable SEGMENT inference and
  evaluation runner based on the official example.

## Remote Quick Start

After connecting to the school server with VS Code Remote-SSH:

```bash
git clone https://github.com/<your-user>/VLM-Competition.git ~/workspace/VLM-Competition
cd ~/workspace/VLM-Competition

export FOCUS_ROOT_DIR="$HOME/data/focus"
bash scripts/setup_school_server.sh
source ~/workspace/.venvs/orena-focus/bin/activate

python scripts/prepare_heico_data.py --root-dir "$FOCUS_ROOT_DIR" --skip-frames
python scripts/check_focus_dataset.py --root-dir "$FOCUS_ROOT_DIR" --make-video-sample --no-overlay
python scripts/run_segment_baseline.py --root-dir "$FOCUS_ROOT_DIR" --num-eval 3 --no-overlay
```

Once the 3-sample smoke test works, run `--num-eval 100`, then run the full
split with `--num-eval none`.
