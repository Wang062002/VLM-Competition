# Experiments And Results

## Baseline Model

- Model: `Qwen/Qwen3-VL-4B-Instruct`
- Evaluator: official `Evaluator`
- Judge model: `Qwen/Qwen3.5-4B`
- Track: `SEGMENT`
- Dataset: `heico`

Runtime compatibility patch used for inference:

```python
self.model = Qwen3VLForConditionalGeneration.from_pretrained(
    self.model_id,
    torch_dtype=torch.bfloat16,
).to(self.device).eval()
```

This was needed because `device_map="auto"` placed the model on CPU in the
remote environment.

## Completed Baselines

### official-smoke-3

- Split: official TEST
- Samples: `3`
- Input: raw video, `use_overlay=False`
- Result: pipeline worked
- Score: not meaningful

### official-smoke-100

- Split: official TEST
- Samples: `100`
- Input: raw video, `use_overlay=False`
- Overall MEAN: `0.200000`
- Pre-evaluation SCORE: `0.186795`

### official-overlay-100

- Split: official TEST
- Samples: `100`
- Input: timestamp overlay, `use_overlay=True`
- Overall MEAN: `0.210000`
- Pre-evaluation SCORE: `0.200886`

### official-raw-full-4000

- Split: official TEST
- Samples: `4000`
- Input: raw video, `use_overlay=False`
- Overall MEAN: `0.194250`
- Pre-evaluation SCORE: `0.364083`
- `temporal_grounding`: `0.007741`
- `temporal_localization`: `0.000000`
- `time`: `0.003199`
- `object_recognition`: `0.302873`
- `event_understanding`: `0.580415`
- `complex_reasoning`: `0.600999`

### official-overlay-full-4000

- Split: official TEST
- Samples: `4000`
- Input: timestamp overlay, `use_overlay=True`
- Overall MEAN: `0.207500`
- Pre-evaluation SCORE: `0.372647`
- `temporal_grounding`: `0.033822`
- `temporal_localization`: `0.027867`
- `time`: `0.029623`
- `object_recognition`: `0.308308`
- `object_identification`: `0.149298`
- `fo_class`: `0.175904`

## Raw vs Overlay Full Comparison

- Overall MEAN: `0.194250 -> 0.207500`, delta `+0.013250`
- Pre-evaluation SCORE: `0.364083 -> 0.372647`, delta `+0.008564`
- `temporal_grounding`: `0.007741 -> 0.033822`, delta `+0.026081`
- `temporal_localization`: `0.000000 -> 0.027867`, delta `+0.027867`
- `time`: `0.003199 -> 0.029623`, delta `+0.026424`
- `object_recognition`: `0.302873 -> 0.308308`, delta `+0.005435`
- `object_identification`: `0.153572 -> 0.149298`, delta `-0.004274`
- `fo_class`: `0.179436 -> 0.175904`, delta `-0.003532`

Interpretation:

- Overlay helps temporal grounding in relative terms.
- Overlay does not solve temporal QA.
- Overall improvement is modest.
- Main bottlenecks remain domain knowledge, foreign-object recognition, visual
  grounding, temporal reasoning, and answer-format control.

## LoRA-SFT Smoke Test

Run: `qwen3vl-4b-smoke-32`

- Date: `2026-07-09`
- Base model: `Qwen/Qwen3-VL-4B-Instruct`
- Train samples: `32`
- Val samples: `8`
- Optimizer steps: `8`
- Eval loss: `1.0017873756587505`
- Adapter:
  `/home/Jiali_Wang/workspace/focus-runs/lora-sft/qwen3vl-4b-smoke-32/adapter-final`
- History:
  `/home/Jiali_Wang/workspace/focus-runs/lora-sft/qwen3vl-4b-smoke-32/train_history.jsonl`

Interpretation:

- End-to-end LoRA-SFT plumbing is validated.
- This model is not intended to be useful yet.
- Next scaling step is `512 train / 128 val`.

## LoRA-SFT Medium Test Attempt

Run: `qwen3vl-4b-smoke-512`

- Date: `2026-07-09`
- Intended scale: `512 train / 128 val`
- Training phase reached completion, then evaluation failed at `85/128`.
- Failing sample: `qID=118214`
- Video: `/home/Jiali_Wang/data/focus/heico/overlayed/0001 - Heico - Prokto - 2_overlay.mp4`
- Error: `start_frame=349700` while video has only `135600` frames.

Interpretation:

- The failure is a TRAIN-derived validation data issue, not a LoRA optimizer
  failure.
- Path-existence checks were insufficient; QA time-window coverage must also be
  validated against source video duration.
- `scripts/train_qwen3vl_lora_sft_smoke.py` was updated to pre-filter invalid
  clip rows and write `invalid_clips_train.jsonl` / `invalid_clips_val.jsonl`
  in the run output directory.

## LoRA-SFT Medium Test Completed

Run: `qwen3vl-4b-smoke-512-filtered`

- Date: `2026-07-09`
- Base model: `Qwen/Qwen3-VL-4B-Instruct`
- Intended scale: `512 train / 128 val`
- Effective train samples: `512`
- Effective val samples: `99`
- Invalid train clip rows: `0`
- Invalid val clip rows: `29`
- Optimizer steps: `128`
- Eval loss: `0.35957938603553546`
- Adapter:
  `/home/Jiali_Wang/workspace/focus-runs/lora-sft/qwen3vl-4b-smoke-512-filtered/adapter-final`
- History:
  `/home/Jiali_Wang/workspace/focus-runs/lora-sft/qwen3vl-4b-smoke-512-filtered/train_history.jsonl`

Interpretation:

- The LoRA-SFT medium-scale training and adapter saving path is validated.
- The lower eval loss compared with the 32-sample smoke test suggests the SFT
  objective is learnable at this scale.
- The invalid-val rate in the first 128 validation examples is high enough that
  a full TRAIN/VAL clip-window audit should be run before full training.

## Full SFT Clip-Window Audit

Run: `clip-window-audit-seed20260707`

- Date: `2026-07-09`
- Input train manifest:
  `/home/Jiali_Wang/workspace/focus-runs/data-audit/segment-trainval-seed20260707/sft_train_overlay.jsonl`
- Input val manifest:
  `/home/Jiali_Wang/workspace/focus-runs/data-audit/segment-trainval-seed20260707/sft_val_overlay.jsonl`
- Internal train rows: `7198`
- Clip-valid train rows: `5959`
- Invalid train rows: `1239`
- Internal val rows: `802`
- Clip-valid val rows: `663`
- Invalid val rows: `139`
- Clean train manifest:
  `/home/Jiali_Wang/workspace/focus-runs/data-audit/clip-window-audit-seed20260707/sft_train_overlay.clip_valid.jsonl`
- Clean val manifest:
  `/home/Jiali_Wang/workspace/focus-runs/data-audit/clip-window-audit-seed20260707/sft_val_overlay.clip_valid.jsonl`

Interpretation:

- About `17.2%` of both internal train and val rows cannot be cut from the
  referenced overlay videos and should be excluded from video-SFT runs.
- Full LoRA-SFT should use the clip-valid manifests, not the original manifests.
- This is a data integrity rule, not a model selection decision, because the
  original QA time windows are impossible to realize as video clips.

## Full Clip-Valid LoRA-SFT Completed

Run: `qwen3vl-4b-sft-valid5959-e1`

- Date: `2026-07-10`
- Base model: `Qwen/Qwen3-VL-4B-Instruct`
- Training data: clip-valid internal TRAIN from official SEGMENT TRAIN
- Validation data: clip-valid internal VAL from official SEGMENT TRAIN
- Train samples: `5959`
- Val samples: `663`
- Invalid train clip rows during run: `0`
- Invalid val clip rows during run: `0`
- Epochs: `1`
- Gradient accumulation steps: `4`
- Optimizer steps: `1490`
- Eval loss: `0.42800752680308324`
- Final train-history tail:
  - last 30 optimizer steps mean loss: `0.38498429132790385`
  - last recorded train loss: `0.37840183079242706`
  - tail min/max loss: `0.08090472221374512` / `0.8931623101234436`
  - elapsed training time: `36967.608` seconds, about `10.27` hours
- Adapter:
  `/home/Jiali_Wang/workspace/focus-runs/lora-sft/qwen3vl-4b-sft-valid5959-e1/adapter-final`
- History:
  `/home/Jiali_Wang/workspace/focus-runs/lora-sft/qwen3vl-4b-sft-valid5959-e1/train_history.jsonl`

Interpretation:

- This is the first full clip-valid Qwen3-VL LoRA-SFT adapter for the project.
- Training is complete, but task improvement is not yet established until the
  adapter is evaluated on the held-out official TEST split.
- The eval loss is higher than the 512-sample filtered run, so the next step is
  to inspect the loss curve and run official evaluator comparisons rather than
  assuming improvement from loss alone.
- The training tail does not show obvious divergence; held-out TEST evaluation
  is the next decision point.

## LoRA-SFT Adapter TEST-100 Evaluation

Run: `qwen3vl-4b-sft-valid5959-e1-overlay-test-100`

- Date: `2026-07-10`
- Split: official TEST
- Samples: `100`
- Input: timestamp overlay
- Base model: `Qwen/Qwen3-VL-4B-Instruct`
- Adapter:
  `/home/Jiali_Wang/workspace/focus-runs/lora-sft/qwen3vl-4b-sft-valid5959-e1/adapter-final`
- Overall MEAN: `0.350000`
- Pre-evaluation SCORE: `0.328905`
- `object_recognition`: `0.592593`
- `temporal_grounding`: `0.065217`
- `object_identification`: `0.576923`
- `fo_class`: `0.521739`
- `time`: `0.065217`

Comparison to `official-overlay-100`:

- Overall MEAN: `0.210000 -> 0.350000`, delta `+0.140000`
- Pre-evaluation SCORE: `0.200886 -> 0.328905`, delta `+0.128019`
- This is a promising small-sample held-out TEST result, especially for
  object-recognition and foreign-object class answers.
- Full TEST-4000 evaluation is still required before making the final
  performance claim.

## Structured Result Files

- `results/experiment_log.csv`
- `results/experiment_events.csv`
- `results/full_raw_vs_overlay_summary.csv`
- `results/dataset_status.csv`
