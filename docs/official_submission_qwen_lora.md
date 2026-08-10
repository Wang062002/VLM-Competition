# Official SEGMENT Submission: Qwen3-VL + LoRA

Date: `2026-07-30`

Goal: submit the already trained Qwen3-VL LoRA adapter through the official
ORena FOCUS SEGMENT channel.

## Current Status

- Official submission template cloned on the remote server:
  `~/workspace/orena-focus-submission-template`
- Target track directory:
  `~/workspace/orena-focus-submission-template/segment-algorithm`
- School server Docker environment installed 2026-08-07 (user self-installed
  after receiving sudo):
  - Docker Engine 28.1.1 (official docker-ce) + containerd 1.7.27 + buildx + compose
  - NVIDIA Container Toolkit 1.19.1, nvidia runtime configured in
    `/etc/docker/daemon.json`
  - Jiali_Wang in docker group; registry-mirrors configured (Docker Hub large
    files get `connection reset by peer` without mirrors)
- `./do_test_run.sh` PASSED: image built in 818s, 3 test samples ran with 0
  failures, output written to `test/output/interface_1/answer.json`.
  Note: PyTorch fell back to CPU because driver 470 (CUDA 11.4) is too old for
  the CUDA 12.4 base image (`UserWarning: NVIDIA driver ... too old`); the
  official evaluation machine has a newer driver and will run on GPU.
- `./do_save.sh` produced the submission tarball:
  `segment-algorithm_2026-08-07T15-57-33.08855081+08-00.tar.gz`

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

## Official Pre-Evaluation Result

Date: `2026-08-10`

The CUDA 12.8 / flexible input-path image was uploaded and marked active:

```text
Image version: 16860a54-5d41-40c9-a925-34d5ec0aecb9
Comment: Qwen3-VL-4B LoRA-SFT, CUDA12.8, flexible batch-video input paths
```

Try-out confirmed the official platform mounts uploaded batch videos directly
under `/input/<qID>.mp4`, not `/input/overlayed/<qID>.mp4`. The updated
`inference.py` resolved this by recursively locating qID-named video clips.

Official SEGMENT pre-evaluation then completed successfully:

| Metric | Value |
|---|---:|
| Status | Succeeded |
| Leaderboard position | 27 |
| Hidden questions | 2000 |
| Batches | 100 |
| Pre-evaluation score | 0.32331911598560603 |
| Questions forfeited | 0 |
| Questions unanswered | 0 |
| Mean batch duration | 63.47976909 s |

Detailed metrics are stored in
`results/official_preeval_20260810_qwen_lora_metrics.csv`.
