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
- A minimum one-GPU notebook was started for platform and storage discovery;
  this is not the formal four-GPU training allocation.
- Terminal mount audit confirmed two writable persistent NFS mounts:
  - shared project area: `/storage/main/projects/orenafocus-prj`
  - personal area: `/storage/main/users/jialiwang`
- The notebook root is an overlay filesystem with about `159 GB` free. Keep
  datasets, model weights, manifests, and checkpoints on the NFS mounts.
- The collaborator copied the raw datasets to the shared project mount:
  - HeiCo: `/storage/main/projects/orenafocus-prj/heico-focus-vqa` (`150 GB`)
  - LapChole: `/storage/main/projects/orenafocus-prj/lapchole-focus-vqa` (`91 GB`)
- Both datasets were then copied into Jiali Wang's persistent personal root as
  `/storage/main/users/jialiwang/data/focus/{heico,lapchole}`. The Cybertron
  config now trains from this personal copy. The inspected shared sources
  contain annotations and raw `videos`, but no timestamp `overlayed` directory;
  overlays still need to be generated in the personal copy.
- The platform page temporarily reported `GPU total memory: 5600 MB` because a
  collaborating FRAME-track participant was occupying all four GPUs. Recheck
  the full allocation with `nvidia-smi` after those jobs release the GPUs.
- The historical `train_qwen3vl_lora_sft_smoke.py` remains single-GPU for v1
  reproducibility. A new `train_qwen35_lora_sft_ddp.py` now implements
  single-node NCCL DDP for Qwen3.5 on four L20 GPUs, including rank-local data
  shards, `no_sync()` accumulation, distributed validation, and rank-zero-only
  logging/checkpoint writes.
- Historical dual-dataset run on one RTX A5000 measured about `0.08 Hz`
  (`13.5 s/sample`) and estimated `164 h` for four epochs. A direct single-GPU
  three-epoch rerun is therefore roughly `123 h` before final validation.
- Provisional target after implementing four-GPU DDP: about `35-45 h` total for
  three epochs plus validation. Replace this estimate with a measured 128-256
  sample smoke-test projection after the notebook becomes available.
- Updated model decision: use `Qwen3.5-4B` rather than Qwen3-VL-4B for the next
  dual-dataset, three-epoch run. Qwen3.5 support from historical commit
  `458925e` has been reworked into the new DDP entry point instead of replacing
  the v1 script. The implementation is syntax-checked but still requires a real
  four-L20 smoke test before the formal run.
- Cybertron configuration:
  `configs/qwen35_lora_sft_cybertron_l20x4.json`.
- Detailed runbook: `docs/qwen35_l20x4_training_20260829.md`.
- The formal Cybertron notebook now exposes all four L20 GPUs, each with
  `44.4 GiB` in PyTorch. The personal runtime is Python `3.11.16`, PyTorch
  `2.11.0+cu128`, Transformers `5.13.0`, PEFT `0.20.0`, and FLA `0.5.2` under
  `/storage/main/users/jialiwang/envs/orena-qwen35`.
- Source `scripts/activate_cybertron_qwen35.sh` in every Cybertron terminal.
  This is especially important because the notebook UID has no passwd entry
  and the platform default remains Python 3.8.
- The official Qwen3.5-4B weights are downloaded under the personal model
  directory. Real weight loading and a short generation passed at `8.81 GiB`
  peak GPU allocation; all configured LoRA target suffixes exist.
- Gated access checks passed for HeiCo and LapChole train/test SEGMENT splits.
  Timestamp overlays completed for all 30 metadata-referenced HeiCo videos and
  100 metadata-referenced LapChole videos. The combined official TRAIN pool is
  13,746 questions, split into 12,372 internal-train and 1,374 validation rows;
  every clip window passed the audit. The 6,254 official TEST rows are held out.
- The first four-L20 smoke validated NCCL, four model replicas, all LoRA
  targets, and 13,959,168 trainable parameters, then stopped before step 1 when
  Transformers tried the removed `torchvision.io.read_video` fallback. The
  trainer now passes Decord-sampled RGB arrays directly to the processor,
  removing the temporary-MP4/TorchCodec dependency and redundant decode. It
  also renders the chat template directly through the tokenizer with
  `enable_thinking=False`, because a follow-up probe showed Transformers 5.13
  did not expand processor-level `template_kwargs`. The 128-row smoke must be
  rerun before formal training.
- `causal-conv1d 1.7.0` has no PyTorch 2.11 wheel and failed to compile locally.
  It is not a launch blocker; use the four-GPU smoke to measure fallback cost
  before revisiting this optional extension.
- A collaborator in the FRAME track already downloaded HeiCo and LapChole to a
  personal path and will move/share them through a public area. Treat the
  shared datasets as read-only; keep manifests, clips, checkpoints, caches, and
  experiment outputs under Jiali Wang's own persistent directory.
- Runtime image `orena-env-v2` is collaborator-owned. Do not modify or overwrite
  the shared image. Use it only as the base for a separate notebook/environment
  and save a new project-owned image after dependency validation.
- The Cybertron platform does not expose Docker inside the notebook, but it can
  save the configured runtime as a new image. Stop training and leave CPU/RAM
  headroom before image saving; keep datasets/model weights out of the image.
