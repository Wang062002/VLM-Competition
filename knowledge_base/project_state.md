# Project State

## Objective

The research goal is to improve ORena FOCUS SEGMENT performance for surgical
video QA, starting from official Qwen3-VL baseline reproduction and moving to
parameter-efficient supervised fine-tuning.

## Competition / Dataset

- Challenge/toolkit: ORena FOCUS, Foreign Object Contextual Understanding for
  Surgery
- Current accessible dataset: `heico`
- Current track: `SEGMENT`
- Other known dataset: `lapchole`, gated access approved as of `2026-08-16`,
  ready for download and preprocessing
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
- Preferred large-storage root: `/mnt/data/jiali_wang`
- Current data root: `/home/Jiali_Wang/data/focus`
  -> symlink to `/mnt/data/jiali_wang/focus`
- Raw videos: `/home/Jiali_Wang/data/focus/heico/videos`
- Overlay videos: `/home/Jiali_Wang/data/focus/heico/overlayed`
- Experiment outputs: `/home/Jiali_Wang/workspace/focus-runs`

Storage policy:

- `/mnt/data` has a newly mounted disk. Create and use
  `/mnt/data/jiali_wang` before storing large files.
- FOCUS data migration completed on `2026-08-12`; root disk free space improved
  from about `25G` to `261G`.
- Avoid placing large model snapshots, Hugging Face cache, or future run
  artifacts on the main system disk.
- Open-VLM candidate snapshots under `~/workspace/vlm-models` may be removed
  after their baseline metrics have been recorded, because Qwen remains the
  stronger mainline model.

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

## Docker Environment (installed 2026-08-07)

- Docker Engine 28.1.1 (official docker-ce) + containerd 1.7.27 + buildx + compose
- NVIDIA Container Toolkit 1.19.1; nvidia runtime in `/etc/docker/daemon.json`
- Jiali_Wang in docker group
- `/etc/docker/daemon.json` also has `registry-mirrors`
  (docker.m.daocloud.io / docker.1panel.live / hub.rat.dev) — Docker Hub large
  pulls get `connection reset by peer` without mirrors
- VS Code Remote-SSH terminals do not refresh group membership; run
  `newgrp docker` first, or use a real `ssh Jiali_Wang@10.176.61.126` login shell
- Verified: `hello-world` OK; `--gpus all nvidia/cuda:12.4.1-base nvidia-smi`
  shows 2x A5000 (driver 470.256.02). Note: PyTorch CUDA 12.4 runtime does NOT
  work with driver 470 (CUDA 11.4); `inference.py` falls back to CPU locally.
  Official eval machine has a newer driver and runs on GPU.
- `./do_test_run.sh` PASSED (3 samples, 0 failures, CPU fallback);
  `./do_save.sh` produced tarball
  `segment-algorithm_2026-08-07T15-57-33.08855081+08-00.tar.gz`
- Previous disk risk was reduced by migrating FOCUS data to `/mnt/data`;
  continue keeping new large files on `/mnt/data/jiali_wang`.

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

## Cybertron Cloud Notebook (2026-08-28)

- Platform: University Cybertron online-development environment.
- Notebook name: `orena-fucus-segment` (platform spelling).
- Project: `ORena-SAVE-FOCUS-Challenge`.
- Runtime image: `orena-env-v2@v2`.
- Requested resources: `24` CPU cores, `128 GB` RAM, `4 x NVIDIA L20`.
- Current status when recorded: queued; no terminal-level hardware audit yet.
- The platform page reports `GPU total memory: 5600 MB`. This is inconsistent
  with a full L20 allocation and must be checked with `nvidia-smi` before model
  setup or training.
- The current `train_qwen3vl_lora_sft_smoke.py` is single-process/single-GPU:
  it moves the model to one `args.device` and has no DDP initialization or data
  sharding. Allocating four GPUs does not accelerate it without a DDP update.
- Historical dual-dataset run on one RTX A5000 measured about `0.08 Hz`
  (`13.5 s/sample`) and estimated `164 h` for four epochs. A direct single-GPU
  three-epoch rerun is therefore roughly `123 h` before final validation.
- Provisional target after implementing four-GPU DDP: about `35-45 h` total for
  three epochs plus validation. Replace this estimate with a measured 128-256
  sample smoke-test projection after the notebook becomes available.
