# Qwen3.5 LoRA-SFT on Cybertron 4 x L20

Date: 2026-08-29

## Objective

Train `Qwen3.5-4B` with LoRA-SFT for the ORena FOCUS SEGMENT track using the
combined HeiCo and LapChole training data for three epochs. The formal job is
designed for one Cybertron node with four NVIDIA L20 GPUs.

## Storage Layout

The notebook mount audit confirmed two persistent NFS mounts:

- shared project storage: `/storage/main/projects/orenafocus-prj`
- Jiali Wang's personal storage: `/storage/main/users/jialiwang`

The container root filesystem has only about `159 GB` free and is an overlay
filesystem. Do not store datasets, model weights, or training checkpoints under
`/workspace`, `/root`, or `/tmp` except for short-lived temporary clips.

Planned layout:

```text
/storage/main/projects/orenafocus-prj/shared/focus-data/
  heico/
  lapchole/

/storage/main/users/jialiwang/
  models/Qwen3.5-4B/
  workspace/VLM-Competition/
  focus-runs/data-audit/
  focus-runs/lora-sft/
```

The shared dataset must be treated as read-only. Generated manifests,
checkpoints, logs, and Hugging Face caches belong in the personal mount.

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

The shared dataset path remains provisional until the collaborator copies the
data and confirms the exact directory structure. Update only the config when
the final path is known.

## Required Validation Order

Do not launch the three-epoch job immediately. Use this order:

1. Confirm all four L20 GPUs and full `48 GB` memory with `nvidia-smi`.
2. Confirm the active runtime has compatible PyTorch, Transformers, PEFT,
   OpenCV, Decord, and Qwen3.5 model classes.
3. Confirm shared HeiCo/LapChole data and overlay videos are readable.
4. Run the combined-data split and clip-window audit.
5. Run the 128-train/32-val four-GPU smoke test.
6. Derive measured samples/second and projected three-epoch time.
7. Run the full three-epoch job only if loss, GPU memory, and throughput are
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

The DDP code, Cybertron configuration, and generated commands pass local syntax
and formatting checks. They have not yet run against the real Qwen3.5 weights,
the collaborator's shared datasets, or a four-L20 allocation. The smoke test is
therefore a required compatibility and performance gate, not an optional demo.
