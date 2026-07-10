# LoRA + SFT Training Plan For ORena FOCUS

This document outlines the next-stage plan after reproducing the official
Qwen3-VL baseline on ORena FOCUS SEGMENT.

## Core Principle

LoRA and SFT are not competing methods:

- SFT is the training objective and data format: train the model to produce the
  reference answer given the video clip and question.
- LoRA is the parameter-efficient adaptation technique: instead of updating all
  model weights, insert small trainable low-rank matrices into selected layers.

Therefore the intended method is:

```text
Qwen3-VL-4B-Instruct + video-QA supervised fine-tuning + LoRA adapters
```

## Data-Safety Rule

Before training, verify the competition rules and dataset splits.

Do not train on a split that must be treated as held-out evaluation data for
leaderboard submission. The current reproduced baselines used
`DatasetSplit.TEST` for local evaluation because the official package provides
answers locally, but using those labels for training may create leakage if the
same split is used for reported evaluation or submission.

Official dataset structure checked on 2026-07-07:

- `data/frame/train.parquet`
- `data/frame/test.parquet`
- `data/segment/train.parquet`
- `data/segment/test.parquet`
- `data/procedure/train.parquet`
- `data/procedure/test.parquet`

The official Python API maps:

- `DatasetSplit.TRAIN` to HuggingFace split `"train"`
- `DatasetSplit.TEST` to HuggingFace split `"test"`
- `DatasetSplit.ALL` to `"train+test"`

For the next training step, use the SEGMENT train split:

```python
FocusDataset("heico", DatasetSplit.TRAIN, Track.SEGMENT)
```

Recommended safe practice:

1. Identify official train/validation/test splits for `heico` and SEGMENT.
2. Train only on allowed training data.
3. Keep one held-out validation split for method selection.
4. Report final local evaluation on a split that was never used to train or tune
   prompts/hyperparameters.

## Why Fine-Tune

The reproduced baseline suggests two limitations:

- Generic Qwen3-VL-4B lacks systematic surgical/medical domain knowledge.
- Timestamp overlay improves temporal grounding only modestly; the model still
  struggles with time answers and procedure-specific reasoning.

Training should target:

- foreign-object class recognition
- surgical object/action vocabulary
- temporal grounding with visible timestamps
- answer-format control
- concise challenge-style answers

## Phase 1: Data Audit And Split Construction

Goals:

- determine available train/test counts for SEGMENT
- inspect answer formats and class distribution
- prevent train/eval leakage
- create reproducible train/validation files

Tasks:

- summarize sample counts by split, primary category, and answer format
- decide whether to train on overlay clips, raw clips, or both
- save a fixed random seed for any internal validation split
- preserve `qID`, `videoID`, `start_time`, `end_time`, question, and reference

Recommended initial setup:

- train input: timestamp overlay clips
- validation input: timestamp overlay clips
- held-out comparison: raw and overlay evaluation baselines

## Phase 2: Data Formatting For SFT

Each training sample should represent one conversational VQA example:

```text
user: <video clip> + question
assistant: reference answer
```

Recommended target style:

- short answer only
- no explanation unless the task explicitly asks for reasoning
- preserve official answer format:
  - `time`: standardized time string
  - `fo_class`: one valid foreign-object class or `none`
  - `binary`: yes/no or official binary format
  - `multiple_choice`: selected option only
  - `number` / `percentage`: normalized numeric output

For paper writing, record all formatting decisions because they affect reported
performance.

## Phase 3: First LoRA-SFT Experiment

Start conservative.

Model:

- base: `Qwen/Qwen3-VL-4B-Instruct`
- trainable method: LoRA adapters
- likely target modules: attention projection layers and MLP projection layers
- freeze vision backbone initially

Initial LoRA hyperparameters to test:

- rank `r`: `8` or `16`
- alpha: `16` or `32`
- dropout: `0.05`
- learning rate: `1e-4` to `2e-4`
- epochs: `1` to `3`
- effective batch size: as large as memory allows via gradient accumulation
- precision: bf16 if stable
- gradient checkpointing: enabled if needed

Hardware expectation:

- 2 x RTX A5000 24GB should be enough for Qwen3-VL-4B LoRA-SFT with careful
  batch size and gradient accumulation
- full fine-tuning is not recommended initially

## Phase 4: Evaluation Protocol

Evaluate every trained checkpoint with the same official evaluator used in the
baseline reproduction.

Primary metrics:

- `pre_evaluation SCORE`
- `overall MEAN`
- category-level accuracy
- answer-format-level accuracy

Always compare against:

- raw full baseline
- overlay full baseline
- any prompt-only baseline

Important categories:

- `temporal_grounding`
- `temporal_localization`
- `time`
- `object_recognition`
- `object_identification`
- `fo_class`

## Phase 5: Ablations

Suggested ablations:

- overlay vs raw training input
- prompt-only improvement vs LoRA-SFT
- LoRA rank `8` vs `16`
- language-only LoRA vs projector/vision-adjacent LoRA
- full mixed-category training vs category-focused training
- answer-format normalization on/off

## Phase 6: Paper Narrative

Potential paper story:

1. Reproduce official Qwen3-VL baseline.
2. Show that timestamp overlay modestly improves temporal grounding but remains
   insufficient.
3. Identify domain and answer-format gaps through error analysis.
4. Apply parameter-efficient supervised adaptation with LoRA.
5. Evaluate whether domain-specific SFT improves foreign-object recognition,
   temporal QA, and format compliance.

## Immediate Next Steps

1. Run the split audit script on the remote server:

   ```bash
   cd ~/workspace/VLM-Competition
   python scripts/audit_and_split_segment_train.py \
     --root-dir /home/Jiali_Wang/data/focus \
     --output-dir ~/workspace/focus-runs/data-audit/segment-trainval-seed20260707
   ```

2. Inspect the generated files:

   - `split_counts.csv`
   - `distribution_by_official_split_primary.csv`
   - `distribution_by_official_split_answer_format.csv`
   - `train_internal_split_manifest.csv`
   - `sft_train_overlay.jsonl`
   - `sft_val_overlay.jsonl`

3. Use `sft_train_overlay.jsonl` for the first LoRA-SFT smoke test.
4. Use `sft_val_overlay.jsonl` for internal validation and checkpoint choice.
5. Keep the official SEGMENT TEST split untouched for final local evaluation.

Execution result from 2026-07-09:

- Official SEGMENT TRAIN: `8000`
- Official SEGMENT TEST: `4000`
- Internal train: `7198`
- Internal validation: `802`
- Output directory:
  `/home/Jiali_Wang/workspace/focus-runs/data-audit/segment-trainval-seed20260707`

## Smoke-Test Training Command

After installing `peft`, run a tiny LoRA-SFT smoke test before full training:

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

This run is only a plumbing test. It should produce:

- `run_config.json`
- `train_history.jsonl`
- `smoke_summary.json`
- `adapter-final/`

If bf16 LoRA exceeds memory, retry with `--load-in-4bit` after installing
`bitsandbytes`.

Smoke test result from 2026-07-09:

- Status: completed
- Train samples: `32`
- Validation samples: `8`
- Optimizer steps: `8`
- Validation loss: `1.0017873756587505`
- Adapter directory:
  `/home/Jiali_Wang/workspace/focus-runs/lora-sft/qwen3vl-4b-smoke-32/adapter-final`

Next scaling step:

- run a medium smoke experiment on `512` train samples and `128` validation
  samples
- inspect loss trend and adapter loading before launching all `7198` internal
  train samples
