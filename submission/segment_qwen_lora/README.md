# SEGMENT Qwen3-VL + LoRA Submission Assets

This folder contains the files to copy into the official ORena FOCUS submission
template for the SEGMENT track.

Official template target:

```bash
~/workspace/orena-focus-submission-template/segment-algorithm
```

Required resource layout in the template:

```text
segment-algorithm/
  inference.py
  requirements.txt
  resources/
    qwen3vl-4b/
    qwen3vl-lora/
```

The model directories are intentionally not tracked in Git.

## Remote Preparation

Copy Qwen base model and LoRA adapter into the official template:

```bash
mkdir -p ~/workspace/orena-focus-submission-template/segment-algorithm/resources/qwen3vl-4b
mkdir -p ~/workspace/orena-focus-submission-template/segment-algorithm/resources/qwen3vl-lora

cp -L ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17/* \
  ~/workspace/orena-focus-submission-template/segment-algorithm/resources/qwen3vl-4b/

cp -L ~/workspace/focus-runs/lora-sft/qwen3vl-4b-sft-valid5959-e1/adapter-final/* \
  ~/workspace/orena-focus-submission-template/segment-algorithm/resources/qwen3vl-lora/
```

Copy submission files from this repository into the official template:

```bash
cp ~/workspace/VLM-Competition/submission/segment_qwen_lora/inference.py \
  ~/workspace/orena-focus-submission-template/segment-algorithm/inference.py

cp ~/workspace/VLM-Competition/submission/segment_qwen_lora/requirements.txt \
  ~/workspace/orena-focus-submission-template/segment-algorithm/requirements.txt
```

## Dry Run Without Docker

The school server currently has no Docker/Podman/Singularity/Apptainer. The
submission script supports `ORENA_INPUT_PATH` and `ORENA_OUTPUT_PATH` so it can
be tested without Docker:

```bash
cd ~/workspace/orena-focus-submission-template/segment-algorithm
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ORENA_INPUT_PATH="$PWD/test/input/interface_1"
export ORENA_OUTPUT_PATH="$PWD/test/output/interface_1"
rm -rf "$ORENA_OUTPUT_PATH"
mkdir -p "$ORENA_OUTPUT_PATH"

python inference.py
```

The preferred full test is still `./do_test_run.sh` on a machine with Docker and
NVIDIA runtime, because that reproduces the official no-network container
environment.

## Docker Build Requirement

Official submission requires a Docker image tarball produced by:

```bash
cd ~/workspace/orena-focus-submission-template/segment-algorithm
./do_test_run.sh
./do_save.sh
```

The current school server has no container runtime, so build/save must happen on
another machine with Docker. Runtime internet is unavailable on the official
platform, so both `resources/qwen3vl-4b` and `resources/qwen3vl-lora` must be in
the image.
