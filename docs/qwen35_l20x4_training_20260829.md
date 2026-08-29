# Qwen3.5 LoRA-SFT on Cybertron 4 x L20

Date: 2026-08-29

## Objective

Train `Qwen3.5-4B` with LoRA-SFT for the ORena FOCUS SEGMENT track using the
combined HeiCo and LapChole training data for five epochs. The formal job is
designed for one Cybertron node with four NVIDIA L20 GPUs.

## Storage Layout

The notebook mount audit confirmed two persistent NFS mounts:

- shared project storage: `/storage/main/projects/orenafocus-prj`
- Jiali Wang's personal storage: `/storage/main/users/jialiwang`

The container root filesystem has only about `159 GB` free and is an overlay
filesystem. Do not store datasets, model weights, or training checkpoints under
`/workspace`, `/root`, or `/tmp` except for short-lived temporary clips.

Confirmed layout after the personal copy:

```text
/storage/main/users/jialiwang/
  data/focus/heico/
  data/focus/lapchole/
  models/Qwen3.5-4B/
  workspace/VLM-Competition/
  focus-runs/data-audit/
  focus-runs/lora-sft/
```

The original shared datasets remain read-only at
`/storage/main/projects/orenafocus-prj/{heico,lapchole}-focus-vqa`. Training
uses the personal copies under `/storage/main/users/jialiwang/data/focus` so
overlay generation and local integrity repair cannot affect the collaborator's
files. Generated manifests, checkpoints, logs, and Hugging Face caches also
belong in the personal mount.

## Training Implementation

The entry point is:

```text
scripts/train_qwen35_lora_sft_ddp.py
```

It provides:

- single-node NCCL DDP, with one model replica per L20;
- deterministic equal-size train shards for all ranks;
- rank-local GPU binding from `LOCAL_RANK`;
- DDP `no_sync()` during gradient accumulation;
- rank-zero-only logs and adapter writes;
- distributed final validation-loss reduction;
- an explicit `64`-frame hard cap before Qwen3.5 processing;
- one-pass Decord sampling into in-memory RGB frames, avoiding temporary MP4
  encoding and a second TorchCodec/torchvision decode;
- explicit non-thinking chat-template formatting for short FOCUS answers;
- BF16, gradient checkpointing, SDPA, TF32, and fused AdamW defaults;
- language-backbone LoRA targets for Qwen3.5 full-attention, Gated DeltaNet,
  and MLP projections;
- explicit freezing of any accidentally injected visual LoRA parameters;
- adapter checkpoints at every completed epoch.

DDP does not combine the four GPUs into one `192 GB` device. Each GPU holds a
full model replica. This is appropriate because Qwen3.5-4B fits on one L20;
DDP accelerates throughput by processing four independent video-QA samples at
once and synchronizing only trainable LoRA gradients.

With micro-batch size `1` per rank and gradient accumulation `1`, the global
effective batch size is `4`, matching the earlier single-GPU effective batch
size (`1 x accumulation 4`) while exposing four-way data parallelism.

## Configuration

Cybertron paths and hyperparameters are recorded in:

```text
configs/qwen35_lora_sft_cybertron_l20x4.json
```

The configured dataset root is the confirmed personal copy:
`/storage/main/users/jialiwang/data/focus`.

Source the persistent environment helper in every new terminal:

```bash
source /storage/main/users/jialiwang/workspace/VLM-Competition/scripts/activate_cybertron_qwen35.sh
```

The generated smoke and training commands source this helper automatically. It
selects the personal Python environment and redirects Hugging Face, pip,
TorchInductor, extension-build, and temporary caches away from the container
root filesystem. It also supplies `USER` and `LOGNAME`, which are required
because Cybertron runs the notebook as UID `20083` without an `/etc/passwd`
entry.

Do not prepend the Conda environment's generic library directories to the
global `LD_LIBRARY_PATH`. Doing so can make system utilities such as `watch`
load incompatible Conda terminal libraries and crash. Set additional CUDA
library paths only for an individual extension-build command when required.

## Verified Cybertron Environment

The formal notebook was validated with:

- `4 x NVIDIA L20`, each reporting `44.4 GiB` to PyTorch;
- NVIDIA driver `570.86.15`;
- Python `3.11.16` in
  `/storage/main/users/jialiwang/envs/orena-qwen35`;
- PyTorch `2.11.0+cu128`, CUDA runtime `12.8`, and Triton `3.6.0`;
- Transformers `5.13.0`, PEFT `0.20.0`, and Accelerate `1.14.0`;
- editable `orena-focus 0.3.5` from official commit `7b7e5c5`;
- Decord `0.6.0` and `opencv-python-headless 4.12.0.88`.

The GUI OpenCV wheel cannot load in this headless image because `libxcb.so.1`
is absent. Do not reinstall `opencv-python`; keep the pinned headless wheel.

`Qwen/Qwen3.5-4B` is stored at
`/storage/main/users/jialiwang/models/Qwen3.5-4B`. A real one-GPU weight-load
and short generation smoke passed with `8.81 GiB` peak allocated memory. A
meta-device architecture audit also confirmed every configured LoRA target:
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `out_proj`, `gate_proj`,
`up_proj`, and `down_proj`.

`flash-linear-attention 0.5.2` is installed. `causal-conv1d 1.7.0` has no
published wheel for PyTorch 2.11 and its local CUDA build failed. This package
remains optional; measure the actual fallback throughput in the four-GPU smoke
before spending more time on extension compilation.

The personal data copy contains 30 HeiCo videos and 171 LapChole source files.
The current official metadata references 30 HeiCo videos and 100 LapChole
videos. Timestamp-overlay generation completed for all metadata-referenced
videos: 30 HeiCo overlays (`99 GB`) and 100 LapChole overlays (`118 GB`).

The combined official SEGMENT TRAIN split contains `13,746` questions. The
seeded internal split contains `12,372` training rows and `1,374` validation
rows; all `13,746` clip windows passed the duration/path audit. The `6,254`
official TEST questions remain held out.

## Required Validation Order

Do not launch the full job immediately. Use this order:

1. Confirm all four L20 GPUs and full `48 GB` memory with `nvidia-smi`.
2. Confirm the active runtime has compatible PyTorch, Transformers, PEFT,
   OpenCV, Decord, and Qwen3.5 model classes.
   Record whether `causal_conv1d`, `fla`, `flash_attn`, or `kernels` are
   available. They are optional in the preflight because the shared runtime
   must not be modified in place, but missing Qwen3.5 linear-attention kernels
   can materially reduce throughput and increase memory use.
3. Confirm shared HeiCo/LapChole data and overlay videos are readable.
4. Run the combined-data split and clip-window audit.
5. Run the 128-train/32-val four-GPU smoke test.
6. Derive measured samples/second and projected full-run time.
7. Run the full job only if loss, GPU memory, and throughput are
   healthy.

The read-only environment check for the current minimum notebook is:

```bash
cd /storage/main/users/jialiwang/workspace/VLM-Competition && \
python scripts/check_qwen35_training_env.py --require-gpus 1
```

Repeat it with `--require-gpus 4`, model path, and manifest paths inside the
formal four-GPU notebook before starting the smoke test.

Print the smoke command:

```bash
python scripts/print_qwen_lora_sft_v2_commands.py \
  --config configs/qwen35_lora_sft_cybertron_l20x4.json \
  --stage smoke
```

Print the full command:

```bash
python scripts/print_qwen_lora_sft_v2_commands.py \
  --config configs/qwen35_lora_sft_cybertron_l20x4.json \
  --stage train
```

## Current Status

The four-L20 allocation, personal Python environment, both gated datasets,
Qwen3.5 weights, overlays, combined split, clip-window audit, and all LoRA
target suffixes are validated.

The first 128-row distributed smoke reached NCCL initialization, loaded all
four model replicas, injected `13,959,168` trainable LoRA parameters
(`0.3066%`), and constructed the optimizer. It then failed before step 1
because Transformers fell back to the removed
`torchvision.io.read_video` API while reopening a temporary MP4. The trainer
now sends the Decord-sampled NumPy video directly to the processor with
`do_sample_frames=False`, so TorchCodec is not a training dependency and each
sample is decoded only once. A preprocessing probe showed that Transformers
5.13 also failed to expand processor-level `template_kwargs`. The trainer now
renders the Qwen3.5 template through `processor.tokenizer` with the direct
`enable_thinking=False` argument, then sends the rendered text and in-memory
video to the multimodal processor. The supplied video metadata includes the
sampled-frame indices required by Qwen3.5's timestamp-token calculation.

The corrected 128-train/32-validation smoke completed successfully on four
L20s. It processed 128 training samples in `298.492 s` at `0.42882` global
samples/s, completed 32 optimizer steps, and reported validation loss
`0.326603`. Total runtime including validation and adapter saves was
`328.242 s`.

The 31 visible per-step losses averaged `0.5603`; the first and last 15-step
means were `0.5994` and `0.5269`, respectively. The curve is noisy at global
batch size four but finite and non-divergent. Both the epoch and final adapters
were saved successfully.

At the aggregate smoke throughput, one full 12,372-row epoch projects to
about `8.0 h`; steady-state steps excluding the roughly one-minute compilation
warm-up project closer to `6.5 h` per epoch. The final decision is five epochs:
`61,860` processed training samples and `15,465` optimizer steps. Budget
`33-41 h` for training plus approximately `18 min` for the 1,374-row final
validation set and small checkpoint overheads.

Adapters are saved after every epoch, allowing downstream evaluation to choose
epoch 3, 4, or 5 if the final checkpoint overfits. Formal five-epoch training
is approved on the validated configuration.
