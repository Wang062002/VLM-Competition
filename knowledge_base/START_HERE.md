# START HERE: ORena FOCUS Project Memory

Read this file first after context compression. It points to the minimum set of
project memory needed to continue smoothly.

## Current Stage

The project has moved from baseline reproduction to LoRA-SFT method development.

Completed:

- official Qwen3-VL SEGMENT baseline reproduction on HeiCo TEST
- raw vs timestamp-overlay full 4000 comparison
- official TRAIN/TEST split audit
- deterministic internal train/validation split from official TRAIN
- Qwen3-VL LoRA-SFT 32-sample smoke test
- Qwen3-VL LoRA-SFT 512-sample filtered medium test
- Qwen3-VL full clip-valid LoRA-SFT run

Current next step:

- run full 4000-sample held-out TEST evaluation for the trained adapter
- compare against the reproduced `official-overlay-full-4000` baseline

## Must-Preserve Rules

- Official TEST split is held out. Do not train on it.
- Official TRAIN split is the source for LoRA-SFT.
- Baseline comparison must use the reproduced TEST metrics already logged.
- Timestamp overlay should be generated with official
  `VideoTimestampOverlayPreprocessor` defaults.
- Overlay integrity must be checked by duration coverage, not file count only.
- Do not store passwords or Hugging Face tokens in project files.
- After every substantive project progress, update the memory library, paper
  notes, workflow docs, structured result files, and GitHub commit history.
  See `knowledge_base/maintenance_protocol.md`.

## Key Files

- `knowledge_base/project_state.md`: project facts, paths, remote state
- `knowledge_base/experiments.md`: baseline and training experiment results
- `knowledge_base/workflows.md`: repeatable command workflows
- `knowledge_base/training_stage.md`: LoRA-SFT plan and current status
- `knowledge_base/paper_notes.md`: thesis/paper observations and claims
- `knowledge_base/maintenance_protocol.md`: rules for keeping memory, paper
  notes, workflows, and GitHub versions updated after each progress step
- `knowledge_base/sync_commands.md`: local-to-remote upload commands
- `docs/research_log.md`: longer research log
- `docs/lora_sft_training_plan.md`: detailed LoRA-SFT plan
- `docs/script_workflow_explained.md`: explanation of each script and what it
  did in the project workflow
- `codex.md`: compact operational memory

## Critical Numbers

Dataset:

- HeiCo source videos: `30`
- SEGMENT official TRAIN: `8000`
- SEGMENT official TEST: `4000`
- Internal train from official TRAIN: `7198`
- Internal val from official TRAIN: `802`
- Clip-valid internal train: `5959`
- Clip-valid internal val: `663`
- Split seed: `20260707`

Baselines on official TEST:

- raw full 4000 overall: `0.194250`
- raw full 4000 pre-eval: `0.364083`
- overlay full 4000 overall: `0.207500`
- overlay full 4000 pre-eval: `0.372647`

LoRA-SFT smoke:

- train samples: `32`
- val samples: `8`
- optimizer steps: `8`
- eval loss: `1.0017873756587505`
- adapter:
  `/home/Jiali_Wang/workspace/focus-runs/lora-sft/qwen3vl-4b-smoke-32/adapter-final`

LoRA-SFT medium:

- requested train samples: `512`
- requested val samples: `128`
- valid train samples: `512`
- valid val samples: `99`
- invalid val clip rows: `29`
- optimizer steps: `128`
- eval loss: `0.35957938603553546`
- adapter:
  `/home/Jiali_Wang/workspace/focus-runs/lora-sft/qwen3vl-4b-smoke-512-filtered/adapter-final`

LoRA-SFT full clip-valid:

- train samples: `5959`
- val samples: `663`
- optimizer steps: `1490`
- eval loss: `0.42800752680308324`
- adapter:
  `/home/Jiali_Wang/workspace/focus-runs/lora-sft/qwen3vl-4b-sft-valid5959-e1/adapter-final`

LoRA-SFT TEST-100:

- official TEST samples: `100`
- overall MEAN: `0.350000`
- pre-evaluation SCORE: `0.328905`
- compared with `official-overlay-100` overall `0.210000`, delta `+0.140000`

## Current Recommended Remote Command

Run full held-out TEST evaluation for the LoRA adapter:

```bash
source ~/tools/miniconda3/etc/profile.d/conda.sh
conda activate orena-focus
cd ~/workspace/VLM-Competition

python scripts/run_segment_baseline.py \
  --root-dir /home/Jiali_Wang/data/focus \
  --dataset heico \
  --split test \
  --model-id Qwen/Qwen3-VL-4B-Instruct \
  --adapter-dir ~/workspace/focus-runs/lora-sft/qwen3vl-4b-sft-valid5959-e1/adapter-final \
  --device cuda:0 \
  --num-eval none \
  --video-stride 25 \
  --width 640 \
  --height 360 \
  --output-dir ~/workspace/focus-runs/lora-sft-eval/qwen3vl-4b-sft-valid5959-e1-overlay-test-full
```
