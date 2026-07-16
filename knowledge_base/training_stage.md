# Training Stage

## Concept

The method is:

```text
Qwen3-VL-4B-Instruct + video-QA SFT + LoRA adapters
```

SFT:

- supervised fine-tuning objective
- trains the model to answer the reference answer for a given video clip and
  question

LoRA:

- parameter-efficient adaptation method
- trains small low-rank adapter matrices instead of full model weights

## Data Policy

Use:

- official SEGMENT TRAIN for training and internal validation

Do not use for training:

- official SEGMENT TEST

Current split:

- official TRAIN: `8000`
- internal train: `7198`
- internal val: `802`
- official TEST: `4000`, held out

## Current Training Script

Script:

```text
scripts/train_qwen3vl_lora_sft_smoke.py
```

Capabilities:

- reads `sft_train_overlay.jsonl` and `sft_val_overlay.jsonl`
- cuts overlay video segments using `start_time` and `end_time`
- formats Qwen3-VL video-chat samples
- masks prompt tokens so loss is focused on the assistant answer
- injects LoRA adapters via `peft`
- trains tiny/medium subsets
- saves adapter and processor
- computes validation loss

Default LoRA settings:

- `r=8`
- `alpha=16`
- `dropout=0.05`
- target modules:
  `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- bf16 base loading by default
- optional `--load-in-4bit` if memory is tight

## Completed Smoke Test

- train samples: `32`
- val samples: `8`
- optimizer steps: `8`
- eval loss: `1.0017873756587505`
- adapter saved successfully

Meaning:

- training infrastructure works
- not a meaningful model

## Next Medium Test

Completed run:

- `512 train`
- `128 val` requested, `99 val` valid after clip-window filtering
- `1 epoch`
- gradient accumulation `4`

Result:

- optimizer steps: `128`
- eval loss: `0.35957938603553546`
- invalid train clip rows: `0`
- invalid val clip rows: `29`
- adapter saved successfully

Interpretation:

- medium-scale LoRA-SFT plumbing is validated
- full training should wait until the complete internal TRAIN/VAL manifests are
  audited for clip-window validity

## Next Data Integrity Step

Completed full clip-window audit:

- internal train rows: `7198`
- clip-valid train rows: `5959`
- invalid train rows: `1239`
- internal val rows: `802`
- clip-valid val rows: `663`
- invalid val rows: `139`

Clean manifests:

- train:
  `/home/Jiali_Wang/workspace/focus-runs/data-audit/clip-window-audit-seed20260707/sft_train_overlay.clip_valid.jsonl`
- val:
  `/home/Jiali_Wang/workspace/focus-runs/data-audit/clip-window-audit-seed20260707/sft_val_overlay.clip_valid.jsonl`

## Full Clip-Valid Training

Completed run:

- train samples: `5959`
- val samples: `663`
- epochs: `1`
- learning rate: `1e-4`
- LoRA rank: `8`
- alpha: `16`
- dropout: `0.05`
- gradient accumulation: `4`
- optimizer steps: `1490`
- eval loss: `0.42800752680308324`
- last 30 optimizer steps mean loss: `0.38498429132790385`
- last recorded train loss: `0.37840183079242706`
- elapsed training time: about `10.27` hours
- adapter:
  `/home/Jiali_Wang/workspace/focus-runs/lora-sft/qwen3vl-4b-sft-valid5959-e1/adapter-final`

Do not tune on official TEST.

## Next Evaluation Step

Completed:

- inspect `train_history.jsonl` for loss trend and instability: done, no
  obvious tail divergence
- run adapter inference on a small held-out TEST subset first: done
- run full official TEST evaluation using the same overlay settings as the
  reproduced baseline: done
- compare against `official-overlay-full-4000`:
  - overall MEAN: `0.207500`
  - pre-evaluation SCORE: `0.372647`

Full TEST result:

- LoRA adapter overall MEAN: `0.279000`
- LoRA adapter pre-evaluation SCORE: `0.402794`
- Overall delta: `+0.071500`
- Pre-evaluation delta: `+0.030147`
- Main gains: object recognition, object identification, and foreign-object
  class answers
- Remaining bottleneck: temporal grounding/time answers

## Next Method Step

Two method directions are now active:

1. Analyze Qwen3-VL LoRA full TEST errors and plan a second Qwen iteration.
2. Train the selected open-VLM base model, MedGemma-4B, using the same
   official-TRAIN-derived clip-valid split.

Qwen error-analysis tasks:

- inspect temporal grounding failures
- inspect multiple-choice drop
- compare baseline wrong / LoRA correct examples
- compare baseline correct / LoRA wrong examples
- consider temporal-specific prompting, data balancing, or second-stage
  fine-tuning

MedGemma training setup:

- selected base: `google/medgemma-4b-it`
- pre-training full TEST baseline:
  `open-vlm-medgemma-8frames-full-4000`
- pre-training overall MEAN: `0.188250`
- pre-training pre-evaluation SCORE: `0.281741`
- input setting: timestamp overlay, `8` sampled frames per clip,
  class-constrained prompt, normalized answers
- train JSONL:
  `/home/Jiali_Wang/workspace/focus-runs/data-audit/clip-window-audit-seed20260707/sft_train_overlay.clip_valid.jsonl`
- val JSONL:
  `/home/Jiali_Wang/workspace/focus-runs/data-audit/clip-window-audit-seed20260707/sft_val_overlay.clip_valid.jsonl`

Training goal:

- establish whether MedGemma's medical prior plus LoRA/SFT can outperform its
  prompt-only baseline and approach or exceed the Qwen3-VL LoRA reference.

Secondary candidate:

- LLaVA-OneVision-7B full prompt-only baseline:
  - overall MEAN: `0.155500`
  - pre-evaluation SCORE: `0.249757`
  - object_identification: `0.279323`
  - fo_class: `0.230758`
- LLaVA is weaker than MedGemma overall, but stronger on class/object
  identification metrics.
- Keep LLaVA as a second training candidate or specialist route if MedGemma
  training fails to improve `fo_class` sufficiently.
