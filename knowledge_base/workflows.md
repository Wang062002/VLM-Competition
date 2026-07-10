# Workflows

## New Remote Terminal

```bash
source ~/tools/miniconda3/etc/profile.d/conda.sh
conda activate orena-focus
```

For official repo work:

```bash
cd ~/workspace/orena-focus
```

For project scripts:

```bash
cd ~/workspace/VLM-Competition
```

## After Every Substantive Progress

When a new experiment result, script change, data finding, or paper-relevant
conclusion appears, update the project memory before moving on:

1. Update memory recovery files:
   - `knowledge_base/START_HERE.md`
   - `knowledge_base/experiments.md`
   - `knowledge_base/training_stage.md`
   - `knowledge_base/workflows.md`
   - `knowledge_base/maintenance_protocol.md` if the maintenance rule changes
2. Update paper-supporting files:
   - `knowledge_base/paper_notes.md`
   - `docs/research_log.md`
   - relevant `results/*.csv`
3. Update workflow documentation if commands or scripts changed:
   - `docs/script_workflow_explained.md`
   - relevant `scripts/*.py`
4. Commit and push to GitHub:

```bash
git status
git add .
git commit -m "<short progress summary>"
git push
```

Do not commit data, model weights, passwords, tokens, or `focus-runs/`.

## Data Audit And Train/Val Split

Already executed successfully, but rerun if the dataset version changes:

```bash
cd ~/workspace/VLM-Competition

python scripts/audit_and_split_segment_train.py \
  --root-dir /home/Jiali_Wang/data/focus \
  --output-dir ~/workspace/focus-runs/data-audit/segment-trainval-seed20260707
```

Expected summary:

- official TRAIN: `8000`
- official TEST: `4000`
- internal train: `7198`
- internal val: `802`

## Check SFT Manifest Paths

```bash
python - <<'PY'
import json
from pathlib import Path

base = Path("/home/Jiali_Wang/workspace/focus-runs/data-audit/segment-trainval-seed20260707")

for name in ["sft_train_overlay.jsonl", "sft_val_overlay.jsonl"]:
    bad = []
    total = 0
    with (base / name).open() as f:
        for line in f:
            total += 1
            row = json.loads(line)
            p = Path(row["overlay_video_path"])
            if not p.exists():
                bad.append((row["qID"], row["videoID"], str(p)))

    print(name, "total=", total, "bad_paths=", len(bad))
    for item in bad[:10]:
        print(item)
PY
```

Known result:

- `sft_train_overlay.jsonl total=7198 bad_paths=0`
- `sft_val_overlay.jsonl total=802 bad_paths=0`

## Audit SFT Clip Windows

Run this before full LoRA-SFT training. It verifies that each QA time window can
be cut from the referenced overlay video and writes clean/invalid manifests.

```bash
cd ~/workspace/VLM-Competition

python scripts/audit_sft_clip_windows.py \
  --input-jsonl ~/workspace/focus-runs/data-audit/segment-trainval-seed20260707/sft_train_overlay.jsonl \
  --input-jsonl ~/workspace/focus-runs/data-audit/segment-trainval-seed20260707/sft_val_overlay.jsonl \
  --output-dir ~/workspace/focus-runs/data-audit/clip-window-audit-seed20260707
```

Expected outputs:

- `clip_window_audit_summary.json`
- `sft_train_overlay.clip_valid.jsonl`
- `sft_train_overlay.invalid_clips.jsonl`
- `sft_val_overlay.clip_valid.jsonl`
- `sft_val_overlay.invalid_clips.jsonl`

## LoRA-SFT 32-Sample Smoke

Already completed:

```bash
cd ~/workspace/VLM-Competition

python scripts/train_qwen3vl_lora_sft_smoke.py \
  --train-jsonl ~/workspace/focus-runs/data-audit/segment-trainval-seed20260707/sft_train_overlay.jsonl \
  --val-jsonl ~/workspace/focus-runs/data-audit/segment-trainval-seed20260707/sft_val_overlay.jsonl \
  --output-dir ~/workspace/focus-runs/lora-sft/qwen3vl-4b-smoke-32 \
  --max-train-samples 32 \
  --max-val-samples 8 \
  --epochs 1 \
  --gradient-accumulation-steps 4
```

## LoRA-SFT 512-Sample Medium Test

Completed filtered run:

```bash
cd ~/workspace/VLM-Competition

python scripts/train_qwen3vl_lora_sft_smoke.py \
  --train-jsonl ~/workspace/focus-runs/data-audit/segment-trainval-seed20260707/sft_train_overlay.jsonl \
  --val-jsonl ~/workspace/focus-runs/data-audit/segment-trainval-seed20260707/sft_val_overlay.jsonl \
  --output-dir ~/workspace/focus-runs/lora-sft/qwen3vl-4b-smoke-512-filtered \
  --max-train-samples 512 \
  --max-val-samples 128 \
  --epochs 1 \
  --gradient-accumulation-steps 4
```

Known result:

- requested: `512 train / 128 val`
- effective: `512 train / 99 val`
- invalid train clip rows: `0`
- invalid val clip rows: `29`
- optimizer steps: `128`
- eval loss: `0.35957938603553546`
- adapter saved to:
  `/home/Jiali_Wang/workspace/focus-runs/lora-sft/qwen3vl-4b-smoke-512-filtered/adapter-final`

Monitor GPU:

```bash
nvidia-smi
```

Watch outputs:

```bash
tail -f ~/workspace/focus-runs/lora-sft/qwen3vl-4b-smoke-512/train_history.jsonl
```

## Evaluate Full LoRA Adapter On TEST

Run a 100-sample held-out TEST check first:

```bash
cd ~/workspace/VLM-Competition

python scripts/run_segment_baseline.py \
  --root-dir /home/Jiali_Wang/data/focus \
  --dataset heico \
  --split test \
  --model-id Qwen/Qwen3-VL-4B-Instruct \
  --adapter-dir ~/workspace/focus-runs/lora-sft/qwen3vl-4b-sft-valid5959-e1/adapter-final \
  --device cuda:0 \
  --num-eval 100 \
  --video-stride 25 \
  --width 640 \
  --height 360 \
  --output-dir ~/workspace/focus-runs/lora-sft-eval/qwen3vl-4b-sft-valid5959-e1-overlay-test-100
```

If the 100-sample run is healthy, run the full TEST evaluation:

```bash
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

Compare against `official-overlay-full-4000`:

- overall MEAN: `0.207500`
- pre-evaluation SCORE: `0.372647`

## If Script Changed Locally

Always provide both:

- local PowerShell upload command
- remote extraction or run command

See `knowledge_base/sync_commands.md`.
