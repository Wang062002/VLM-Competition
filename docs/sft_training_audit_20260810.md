# SFT Training Audit: Qwen3-VL LoRA-SFT Official Pre-Eval Gap

Date: 2026-08-10

## Why This Audit Exists

The first successful official SEGMENT pre-evaluation submission scored:

```text
pre_evaluation_score = 0.32331911598560603
leaderboard_position = 27
questions_forfeited = 0
questions_unanswered = 0
```

The container is healthy: no timeout forfeits and no missing answers. The gap to
the reported stronger baseline around 0.5 should therefore be investigated at
the method/training/data level rather than as a Docker/runtime issue.

## What The First SFT Run Actually Did

Training script:

```text
scripts/train_qwen3vl_lora_sft_smoke.py
```

Data construction:

- Source: official `heico` SEGMENT `DatasetSplit.TRAIN`.
- Initial split: 8000 official TRAIN QA pairs split into internal train/val.
- After clip-window filtering:
  - train: 5959 valid rows
  - val: 663 valid rows
  - invalid/dropped rows: 1378 rows across train+val
- Training input: timestamp-overlay videos.
- Each training sample was cut from a full overlay video using `start_time` and
  `end_time`.
- Clip sampling: `video_stride=25`, roughly 1 fps for 25 fps source videos.
- Resolution: 640x360.

SFT format:

```text
system: surgical assistant prompt + FO definitions
user: <video clip> + original question
assistant: raw reference answer
```

Loss masking:

- The script tokenizes a full conversation including the assistant answer.
- It tokenizes a prompt-only conversation separately.
- All prompt tokens are masked with `-100`; loss is applied mainly to assistant
  answer tokens.

LoRA settings:

- Base: `Qwen/Qwen3-VL-4B-Instruct`
- LoRA rank: 8
- LoRA alpha: 16
- LoRA dropout: 0.05
- LR: 1e-4
- Epochs: 1
- Gradient accumulation: 4
- Target modules:
  `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- Finished run:
  `qwen3vl-4b-sft-valid5959-e1`
- Eval loss: 0.42800752680308324

## High-Risk Issues Found

### 1. Training/inference frame-budget mismatch

The SFT script trains on clips generated at about 1 fps from the full
start/end window, without an explicit maximum frame cap in `process_vision_info`.
For a 5-minute segment this can expose up to about 300 sampled frames.

The official submission container, however, was constrained for runtime:

```text
VIDEO_FPS = 1.0
VIDEO_MIN_FRAMES = 4
VIDEO_MAX_FRAMES = 64
MAX_NEW_TOKENS = 64
```

This means the model was trained and locally evaluated under a richer visual
context than the official container may actually use. This mismatch is a
plausible reason why local TEST-4000 pre-evaluation-style score was 0.402794,
but official hidden pre-eval was only 0.323319.

### 2. Local evaluation was not identical to official submission inference

`scripts/run_segment_baseline.py` uses `FocusVideoDataset` and a simpler Qwen
video-processing path:

- no explicit `max_frames`
- no `return_video_metadata=True`
- default `max_new_tokens=128`

The submission container uses:

- official platform-provided pre-cut `/input/<qID>.mp4`
- `max_frames=64`
- `return_video_metadata=True`
- `do_resize=False`
- `max_new_tokens=64`

Therefore local held-out TEST scores are directionally useful, but they are not
a faithful replica of the official runtime input pipeline.

### 3. Data coverage is probably insufficient

The first SFT run used only HeiCo-FOCUS TRAIN. The public official data ecosystem
now also includes LapChole-FOCUS-VQA, described as the second ORena FOCUS data
batch with:

- 100 labeled laparoscopic cholecystectomies
- 70 unlabeled laparoscopic cholecystectomies
- 20,000 VQA pairs
- the same frame/segment/procedure track structure

If stronger teams/baselines use the second batch or pretrain/adapt on it, our
single-HeiCo SFT run is undertrained and under-diverse. Before using it in the
next training run, we must explicitly verify that the current challenge rules
allow using this public batch for the SEGMENT submission.

Sources:

- https://huggingface.co/datasets/orena-dkfz/lapchole-focus-vqa
- https://huggingface.co/datasets/orena-dkfz/heico-focus-vqa

### 4. We dropped a non-trivial fraction of official TRAIN

The clip-window audit removed 1378 of 8000 official TRAIN rows from the first
training pipeline. These invalid rows were removed because their start/end
windows did not fit the generated overlay video files. This avoided runtime
crashes, but also lost supervision, possibly in long or difficult temporal
cases.

Next step should determine whether these rows can be recovered by:

- regenerating overlay videos correctly,
- training from raw videos for rows with invalid overlay windows,
- using official pre-cut clips if available,
- clamping rather than dropping edge windows only when semantically safe.

### 5. Prompt and answer-format supervision are too generic

The first SFT used the original question and raw reference answer only. It did
not add structured answer-format instructions beyond what the question already
contains.

This is risky because ORena scoring is bucketed by answer format and capability.
The model should probably see explicit format contracts during training and
inference, for example:

- binary: answer exactly `yes` or `no`
- fo_class: answer one of the known FO classes or `none`
- time: answer exactly `hh:mm:ss`
- number/percentage: answer a normalized numeric value only
- multiple_choice: answer the option only

### 6. The training objective is unbalanced

The first SFT used one epoch over the clip-valid training rows with no category
or answer-format balancing. Official pre-eval weakness is concentrated in some
buckets:

| Bucket | Official Pre-Eval Accuracy |
|---|---:|
| aggregation ID | 0.170732 |
| aggregation OOD | 0.300000 |
| event understanding OOD | 0.157895 |
| temporal grounding ID | 0.174081 |
| temporal grounding OOD | 0.242678 |

The next SFT should use balanced sampling or loss weighting across capability
group and answer format rather than simply shuffling all available rows.

### 7. LoRA target scope may be too language-only

The current LoRA targets standard attention and MLP modules. This may mainly
adapt the language model while leaving video/vision projection behavior weak.
For surgical video QA, the next audit should inspect actual trainable module
names and consider adding projector or vision-adjacent modules if Qwen3-VL
exposes them safely under memory limits.

## Immediate Diagnosis Commands To Run On The Server

1. Check whether the latest `orena-focus` package can see the LapChole dataset:

```bash
source ~/tools/miniconda3/etc/profile.d/conda.sh && conda activate orena-focus && cd ~/workspace/VLM-Competition && python - <<'PY'
from focus import FocusConfig, set_config
from focus.data.base_dataset import FocusDataset
from focus.enums import DatasetSplit, Track

set_config(FocusConfig(root_dir="/home/Jiali_Wang/data/focus"))
for dataset in ["heico", "lapchole"]:
    for split in [DatasetSplit.TRAIN, DatasetSplit.TEST]:
        try:
            ds = FocusDataset(dataset, split, Track.SEGMENT)
            print(dataset, split.value, len(ds))
        except Exception as exc:
            print(dataset, split.value, type(exc).__name__, exc)
PY
```

2. Inspect trainable LoRA module names from the current training setup:

```bash
source ~/tools/miniconda3/etc/profile.d/conda.sh && conda activate orena-focus && cd ~/workspace/VLM-Competition && python - <<'PY'
from transformers import Qwen3VLForConditionalGeneration

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    torch_dtype="auto",
)
for name, module in model.named_modules():
    if any(key in name for key in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "proj", "vision"]):
        print(name, module.__class__.__name__)
PY
```

3. Re-evaluate the trained LoRA locally using the same 64-frame constraints as
the official submission container, not the older local evaluation pipeline.

## Recommended Next Training Iteration

Priority should be:

1. Build an official-runtime-matched local evaluator:
   - use pre-cut qID clips or mimic platform clip layout,
   - `max_frames=64`,
   - `max_new_tokens=64`,
   - same Qwen `process_vision_info` path as submission.
2. Re-score base Qwen and the current LoRA under this matched evaluator.
3. If LoRA still underperforms the base under matched constraints, stop using
   this adapter as the mainline.
4. Verify whether LapChole TRAIN is accessible and allowed, then add it if it
   is compliant with the challenge rules.
5. Build answer-format-aware SFT JSONL.
6. Train a small but controlled ablation matrix:
   - base Qwen prompt-only official-style,
   - LoRA rank 8 vs 16,
   - 32 vs 64 max frames,
   - balanced sampler vs raw sampler,
   - language-only LoRA vs language+projector LoRA.

## Current Working Hypothesis

The main issue is not that SFT itself is invalid. The first SFT run likely
learned useful object-recognition behavior on local TEST, but it was trained and
validated under a different visual budget and a narrower dataset than the
official hidden pre-evaluation setting. The next iteration should align the SFT
input pipeline with the official container first, then expand/ rebalance data.
