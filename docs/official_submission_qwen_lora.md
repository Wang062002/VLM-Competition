# Official SEGMENT Submission: Qwen3-VL + LoRA

Date: `2026-07-30`

Goal: submit the already trained Qwen3-VL LoRA adapter through the official
ORena FOCUS SEGMENT channel.

## Current Status

- Official submission template cloned on the remote server:
  `~/workspace/orena-focus-submission-template`
- Target track directory:
  `~/workspace/orena-focus-submission-template/segment-algorithm`
- School server status:
  - no `docker`
  - no `podman`
  - no `singularity`
  - no `apptainer`
- Therefore, the school server can prepare resources and run direct Python
  dry-runs, but cannot produce the final official Docker tarball.

## Model Resources

Required template resource layout:

```text
segment-algorithm/resources/qwen3vl-4b/
segment-algorithm/resources/qwen3vl-lora/
```

Observed remote resources:

- Qwen base model snapshot:
  `/home/Jiali_Wang/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17`
- Copied base model size: about `8.3G`
- LoRA adapter:
  `/home/Jiali_Wang/workspace/focus-runs/lora-sft/qwen3vl-4b-sft-valid5959-e1/adapter-final`
- Copied adapter size: about `74M`

Important: Hugging Face snapshots are symlink-based, so copy with `cp -L` or
`rsync -L`.

## Files Added In This Repo

- `submission/segment_qwen_lora/inference.py`
- `submission/segment_qwen_lora/requirements.txt`
- `submission/segment_qwen_lora/README.md`

These files are copied into the official `segment-algorithm` template.

## Remote Update Via Git Pull

```bash
source ~/tools/miniconda3/etc/profile.d/conda.sh
conda activate orena-focus
cd ~/workspace/VLM-Competition
git pull origin main
```

Copy submission files into the official template:

```bash
cp ~/workspace/VLM-Competition/submission/segment_qwen_lora/inference.py \
  ~/workspace/orena-focus-submission-template/segment-algorithm/inference.py

cp ~/workspace/VLM-Competition/submission/segment_qwen_lora/requirements.txt \
  ~/workspace/orena-focus-submission-template/segment-algorithm/requirements.txt
```

## Direct Python Dry Run On School Server

Since Docker is unavailable, test the script directly:

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
cat "$ORENA_OUTPUT_PATH/answer.json"
```

## Final Official Build

Final submission still requires a Docker-capable machine:

```bash
cd ~/workspace/orena-focus-submission-template/segment-algorithm
./do_test_run.sh
./do_save.sh
```

The official platform runs without internet, so the Docker image must contain
both `resources/qwen3vl-4b` and `resources/qwen3vl-lora`.
