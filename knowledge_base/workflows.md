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

Full TEST evaluation has completed:

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

Known full TEST LoRA result:

- overall MEAN: `0.279000`
- pre-evaluation SCORE: `0.402794`
- overall delta vs overlay baseline: `+0.071500`
- pre-evaluation delta vs overlay baseline: `+0.030147`

## Analyze Full LoRA TEST Result

Recommended next analyses:

- inspect `results.csv` and `responses.jsonl` from:
  `/home/Jiali_Wang/workspace/focus-runs/lora-sft-eval/qwen3vl-4b-sft-valid5959-e1-overlay-test-full`
- compare wrong/correct cases between baseline and LoRA
- focus on:
  - temporal grounding still being weak
  - multiple-choice accuracy dropping slightly
  - object identification and `fo_class` improving strongly

Current all-data comparison entry points:

- human-readable Chinese overview:
  `docs/comprehensive_data_comparison.md`
- machine-readable all-data table:
  `results/all_data_comparison_table.csv`
- paper-ready main table:
  `results/main_result_summary.csv`
- LoRA-vs-overlay full TEST delta table:
  `results/lora_full_test_vs_overlay_baseline.csv`
- evaluator-style full TEST summaries:
  `results/evaluator_style_full_4000_summaries.csv`
- terminal-style evaluator output:
  `results/evaluator_style_full_4000_summaries.txt`

Print evaluator-style tables:

```bash
python scripts/print_evaluator_style_summary.py
```

Only print the LoRA full TEST result:

```bash
python scripts/print_evaluator_style_summary.py \
  --experiment qwen3vl_lora_full_4000
```

## Download Open VLM Candidates

Candidate list:

```text
configs/vlm_candidate_models.csv
```

Remote download command:

```bash
source ~/tools/miniconda3/etc/profile.d/conda.sh
conda activate orena-focus
cd ~/workspace/VLM-Competition

python -m py_compile scripts/download_vlm_candidates.py

python scripts/download_vlm_candidates.py \
  --config configs/vlm_candidate_models.csv \
  --output-dir ~/workspace/vlm-models \
  --continue-on-error
```

Dry-run:

```bash
python scripts/download_vlm_candidates.py --dry-run
```

Download one model first:

```bash
python scripts/download_vlm_candidates.py \
  --model minicpm_v_4_5 \
  --output-dir ~/workspace/vlm-models
```

Important:

- Gemma and MedGemma may require Hugging Face license acceptance before download.
- By default the script verifies/resumes existing target directories instead of
  skipping them. Use `--skip-existing` only when you intentionally want to skip
  non-empty directories.
- If Gemma fails after license acceptance with
  `Unable to parse string as hex hash value`, rerun with `--disable-xet`.
- The download manifest is written to:
  `~/workspace/vlm-models/download_manifest.json`
- After downloading, run smoke tests before TEST-100.

Check downloaded snapshots:

```bash
python scripts/check_vlm_downloads.py \
  --model-dir ~/workspace/vlm-models \
  --json-output ~/workspace/focus-runs/open-vlm-download-check.json
```

Run first common multi-frame smoke test:

```bash
pip install -r requirements/open_vlm_smoke.txt

python scripts/run_open_vlm_smoke.py \
  --model minicpm_v_4_5 \
  --model llava_onevision_7b \
  --model internvl3_5_8b \
  --model gemma3_12b \
  --model medgemma_4b \
  --model-dir ~/workspace/vlm-models \
  --root-dir /home/Jiali_Wang/data/focus \
  --num-eval 3 \
  --frames-per-clip 4 \
  --output-dir ~/workspace/focus-runs/open-vlm-smoke/test3 \
  --continue-on-error
```

Run class-constrained fo_class prompt smoke test:

```bash
python scripts/run_open_vlm_smoke.py \
  --model minicpm_v_4_5 \
  --model llava_onevision_7b \
  --model internvl3_5_8b \
  --model gemma3_12b \
  --model medgemma_4b \
  --model-dir ~/workspace/vlm-models \
  --root-dir /home/Jiali_Wang/data/focus \
  --num-eval 3 \
  --frames-per-clip 4 \
  --prompt-mode class_constrained \
  --normalize-answer \
  --output-dir ~/workspace/focus-runs/open-vlm-smoke/test3-class-prompt \
  --continue-on-error
```

Run 30-sample prompt ablation before TEST-100:

```bash
python scripts/run_open_vlm_smoke.py \
  --model minicpm_v_4_5 \
  --model llava_onevision_7b \
  --model internvl3_5_8b \
  --model gemma3_12b \
  --model medgemma_4b \
  --model-dir ~/workspace/vlm-models \
  --root-dir /home/Jiali_Wang/data/focus \
  --num-eval 30 \
  --frames-per-clip 4 \
  --output-dir ~/workspace/focus-runs/open-vlm-smoke/test30-default \
  --continue-on-error

python scripts/run_open_vlm_smoke.py \
  --model minicpm_v_4_5 \
  --model llava_onevision_7b \
  --model internvl3_5_8b \
  --model gemma3_12b \
  --model medgemma_4b \
  --model-dir ~/workspace/vlm-models \
  --root-dir /home/Jiali_Wang/data/focus \
  --num-eval 30 \
  --frames-per-clip 4 \
  --prompt-mode class_constrained \
  --normalize-answer \
  --output-dir ~/workspace/focus-runs/open-vlm-smoke/test30-class-prompt \
  --continue-on-error
```

Run TEST-100 for selected open VLM candidates:

```bash
python scripts/run_open_vlm_smoke.py \
  --model llava_onevision_7b \
  --model medgemma_4b \
  --model minicpm_v_4_5 \
  --model internvl3_5_8b \
  --model-dir ~/workspace/vlm-models \
  --root-dir /home/Jiali_Wang/data/focus \
  --num-eval 100 \
  --frames-per-clip 4 \
  --prompt-mode class_constrained \
  --normalize-answer \
  --output-dir ~/workspace/focus-runs/open-vlm-smoke/test100-class-prompt-selected \
  --continue-on-error
```

Current selection rationale:

- LLaVA-OneVision and MedGemma tie for best TEST-30 class-prompt overall
  (`0.366667`).
- MiniCPM remains relevant because it is video-capable and improved under
  class-prompt on 30 samples.
- InternVL improved but is lower priority.
- Gemma is currently deprioritized.

Run TEST-100 with more frames for the two strongest open VLM candidates:

```bash
python scripts/run_open_vlm_smoke.py \
  --model llava_onevision_7b \
  --model medgemma_4b \
  --model-dir ~/workspace/vlm-models \
  --root-dir /home/Jiali_Wang/data/focus \
  --num-eval 100 \
  --frames-per-clip 8 \
  --prompt-mode class_constrained \
  --normalize-answer \
  --output-dir ~/workspace/focus-runs/open-vlm-smoke/test100-class-prompt-selected-8frames \
  --continue-on-error
```

Rationale:

- MedGemma and LLaVA are the top two TEST-100 prompt-only open VLMs with
  4-frame input.
- An 8-frame comparison checks whether the 4-frame sampling bottleneck is
  limiting recognition before deciding on full TEST-4000.

If evaluator memory becomes an issue, first validate generation only:

```bash
python scripts/run_open_vlm_smoke.py \
  --model minicpm_v_4_5 \
  --model-dir ~/workspace/vlm-models \
  --root-dir /home/Jiali_Wang/data/focus \
  --num-eval 1 \
  --skip-evaluator \
  --output-dir ~/workspace/focus-runs/open-vlm-smoke/minicpm-generate-only
```

Known smoke issue:

- MiniCPM-V-4.5 can fail on some Transformers versions with
  `MiniCPMV object has no attribute all_tied_weights_keys`. The smoke runner
  patches this compatibility gap before loading MiniCPM.
- MiniCPM-V-4.5 can also fail during processor import with
  `str object has no attribute __module__` from `AutoImageProcessor.register`.
  The smoke runner skips that string-based registration side effect.
- InternVL3.5 needs extra Python packages beyond the base Qwen environment:
  `einops` and `timm`. Install them with
  `pip install -r requirements/open_vlm_smoke.txt`.
- InternVL3.5 can also hit the `all_tied_weights_keys` compatibility issue.
  The same generic tied-weights patch used for MiniCPM is applied before
  InternVL loading.

## If Script Changed Locally

Always provide both:

- local PowerShell upload command
- remote extraction or run command

See `knowledge_base/sync_commands.md`.
