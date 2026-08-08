# Method Description: Qwen3-VL-4B + LoRA-SFT for ORena FOCUS SEGMENT

## Model

- Base: `Qwen/Qwen3-VL-4B-Instruct` (vision-language model, ~4B parameters)
- Adaptation: LoRA (Low-Rank Adaptation) supervised fine-tuning

## LoRA Configuration

- rank `r=8`, `alpha=16`, `dropout=0.05`
- target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- base model loaded in `bfloat16`

## Training Data

- Official ORena FOCUS SEGMENT TRAIN split (HeiCo-FOCUS VQA)
- Deterministic internal train/val split (seed `20260707`): 7198 train / 802 val
- Clip-window audit excluded QA pairs whose time windows cannot be cut from the
  overlay videos (1239 invalid train / 139 invalid val)
- Final clean training set: **5959 train / 663 val** (clip-valid only)
- Official TEST split was held out and never used for training or tuning

## Training

- 1 epoch, learning rate `1e-4`, gradient accumulation `4`
- 1490 optimizer steps, ~10.27 hours on a single RTX A5000 (24GB)
- Final validation loss: `0.428`
- Prompt tokens masked so loss is focused on the assistant answer only

## Input Preprocessing

- Timestamp overlay videos generated with the official
  `VideoTimestampOverlayPreprocessor` (overlays the running timestamp onto each
  frame to help temporal grounding)
- Inference samples video at `VIDEO_FPS=1.0`

## Inference

- Qwen3-VL base + LoRA adapter loaded fully offline (all weights baked into the
  Docker image under `resources/`)
- `bfloat16` on GPU, `max_new_tokens=128`
- Single GPU, fully automated, no internet access at runtime

## Results (local held-out TEST, HeiCo SEGMENT, 4000 QA)

| Metric | Overlay baseline | LoRA-SFT | Delta |
|---|---:|---:|---:|
| Overall MEAN | 0.2075 | **0.2790** | +0.0715 |
| Pre-evaluation SCORE | 0.3726 | **0.4028** | +0.0301 |
| object_recognition | 0.3083 | 0.4722 | +0.1639 |
| object_identification | 0.1493 | 0.4692 | +0.3199 |
| fo_class | 0.1759 | 0.4380 | +0.2621 |
| temporal_grounding | 0.0338 | 0.0717 | +0.0379 |

Largest gains come from foreign-object recognition, identification, and class
answers. Temporal grounding improves in relative terms but remains the main
bottleneck and the target of the next iteration.

## Reproducibility

- Code: `github.com/Wang062002/VLM-Competition`
- LoRA adapter baked into `resources/qwen3vl-lora` (~74M)
- Base model baked into `resources/qwen3vl-4b` (~8.3G, real files not symlinks)
- Training script: `scripts/train_qwen3vl_lora_sft_smoke.py`
- Clip-valid manifests produced by `scripts/audit_sft_clip_windows.py`
