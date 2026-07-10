# Project State

## Objective

The research goal is to improve ORena FOCUS SEGMENT performance for surgical
video QA, starting from official Qwen3-VL baseline reproduction and moving to
parameter-efficient supervised fine-tuning.

## Competition / Dataset

- Challenge/toolkit: ORena FOCUS, Foreign Object Contextual Understanding for
  Surgery
- Current dataset: `heico`
- Current track: `SEGMENT`
- Other known dataset: `lapchole`, not downloaded or used
- Source videos in HeiCo: `30`
- SEGMENT official TRAIN: `8000` QA samples
- SEGMENT official TEST: `4000` QA samples

## Remote Server

- Host/IP: `10.176.61.126`
- User: `Jiali_Wang`
- Hostname: `UNNC-CVIP-03`
- GPUs: `2 x NVIDIA RTX A5000`, 24GB each
- Driver observed: `470.256.02`
- Driver-reported CUDA: `11.4`

Do not store or repeat the server password.

## Remote Paths

- Official repo: `/home/Jiali_Wang/workspace/orena-focus`
- Local project copy on remote: `/home/Jiali_Wang/workspace/VLM-Competition`
- Conda install: `/home/Jiali_Wang/tools/miniconda3`
- Conda env: `orena-focus`
- Data root: `/home/Jiali_Wang/data/focus`
- Raw videos: `/home/Jiali_Wang/data/focus/heico/videos`
- Overlay videos: `/home/Jiali_Wang/data/focus/heico/overlayed`
- Experiment outputs: `/home/Jiali_Wang/workspace/focus-runs`

## Environment

- Python env: `orena-focus`
- Python version: `3.10.20`
- PyTorch: `2.7.1+cu118`
- CUDA available: `True`
- GPU count: `2`

Activate every new remote terminal with:

```bash
source ~/tools/miniconda3/etc/profile.d/conda.sh
conda activate orena-focus
```

## Data Integrity Notes

- Timestamp overlay files should be exactly `30`.
- File count alone is not enough. Four Sigma overlay files were initially
  truncated and caused full overlay inference to crash.
- Bad overlays repaired:
  - `0020 - Heico - Sigma - 1.avi`
  - `0021 - Heico - Sigma - 2.avi`
  - `0027 - Heico - Sigma - 8.avi`
  - `0028 - Heico - Sigma - 9.avi`
- Lesson: validate overlay duration coverage against QA metadata.

