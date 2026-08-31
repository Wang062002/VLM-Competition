# SEGMENT Qwen3.5-4B + LoRA Submission

This directory contains the inference and dependency files for the second
ORena FOCUS SEGMENT submission. It is intentionally separate from
`submission/segment_qwen_lora`, which preserves the successful Qwen3-VL v1
submission.

## Required Layout

Copy these files into a fresh official `segment-algorithm` template with this
resource layout:

```text
segment-algorithm/
  answer_utils.py
  inference.py
  requirements.txt
  resources/
    qwen35-4b/
    qwen35-lora/
```

The model resources are not tracked in Git. The official runtime has no
internet access, so both directories must contain real files rather than
symlinks.

## What Must Move Between Servers

Do not copy the HeiCo or LapChole datasets to the Docker server. They are not
part of inference. Use these routes instead:

- source code: `git pull` on the old school server;
- base model: download `Qwen/Qwen3.5-4B` again on the old server's large disk;
- trained artifact: transfer only `adapter-final` (and its SHA256 file) from
  Cybertron to the old server.

### 1. Package the adapter on Cybertron

Run this only after training reports `status: completed` and `adapter-final`
exists. The packaging helper validates the run, includes the summary and
history, and writes SHA256 metadata:

```bash
RUN_ROOT=/storage/main/users/jialiwang/focus-runs/lora-sft/qwen35-4b-heico-lapchole-e5-l20x4-rerun1
TRANSFER_ROOT=/storage/main/users/jialiwang/transfers/qwen35-e5-rerun1
python scripts/package_qwen35_training_artifacts.py \
  --run-root "$RUN_ROOT" \
  --output-dir "$TRANSFER_ROOT"
ls -lh "$TRANSFER_ROOT"
```

First test whether Cybertron can reach the old server directly:

```bash
ssh -o ConnectTimeout=8 Jiali_Wang@10.176.61.126 hostname
```

If it succeeds, create the destination and copy the two small transfer files.
Enter the server password only at the SSH prompt; never paste it into logs or
chat messages.

```bash
ssh Jiali_Wang@10.176.61.126 'mkdir -p /mnt/data/jiali_wang/transfers/qwen35-e5-rerun1'
scp "$TRANSFER_ROOT/qwen35-lora-e5-final.tar.gz" \
  "$TRANSFER_ROOT/qwen35-lora-e5-final.tar.gz.sha256" \
  Jiali_Wang@10.176.61.126:/mnt/data/jiali_wang/transfers/qwen35-e5-rerun1/
```

If direct SSH is blocked, download these two files through the Cybertron file
manager to the laptop and upload them to the same old-server directory with
WinSCP or `scp` after reconnecting to the school VPN.

### 2. Verify and unpack on the old school server

```bash
TRANSFER_ROOT=/mnt/data/jiali_wang/transfers/qwen35-e5-rerun1
ADAPTER_ROOT=/mnt/data/jiali_wang/models/qwen35-lora-e5-final
cd "$TRANSFER_ROOT"
sha256sum -c qwen35-lora-e5-final.tar.gz.sha256
mkdir -p "$ADAPTER_ROOT"
tar -xzf qwen35-lora-e5-final.tar.gz -C "$ADAPTER_ROOT"
du -sh "$ADAPTER_ROOT"
```

### 3. Download the public base model on the old server

Keep it on the large disk:

```bash
source /home/Jiali_Wang/tools/miniconda3/etc/profile.d/conda.sh
conda activate orena-focus
hf download Qwen/Qwen3.5-4B \
  --local-dir /mnt/data/jiali_wang/models/Qwen3.5-4B
du -sh /mnt/data/jiali_wang/models/Qwen3.5-4B
```

### 4. Create a separate Qwen3.5 template checkout

Preserve the known-good Qwen3-VL template. Discover its origin and clone a new
copy on the large disk:

```bash
OLD_TEMPLATE=/home/Jiali_Wang/workspace/orena-focus-submission-template
QWEN35_TEMPLATE=/mnt/data/jiali_wang/workspace/orena-focus-submission-template-qwen35
mkdir -p /mnt/data/jiali_wang/workspace
git clone "$(git -C "$OLD_TEMPLATE" remote get-url origin)" "$QWEN35_TEMPLATE"
```

Pull the prepared inference code and stage it:

```bash
cd /home/Jiali_Wang/workspace/VLM-Competition
git pull --ff-only origin main
ALGORITHM_ROOT=/mnt/data/jiali_wang/workspace/orena-focus-submission-template-qwen35/segment-algorithm
cp submission/segment_qwen35_lora/inference.py "$ALGORITHM_ROOT/inference.py"
cp submission/segment_qwen35_lora/answer_utils.py "$ALGORITHM_ROOT/answer_utils.py"
cp submission/segment_qwen35_lora/requirements.txt "$ALGORITHM_ROOT/requirements.txt"
mkdir -p "$ALGORITHM_ROOT/resources/qwen35-4b"
mkdir -p "$ALGORITHM_ROOT/resources/qwen35-lora"
rsync -aL /mnt/data/jiali_wang/models/Qwen3.5-4B/ \
  "$ALGORITHM_ROOT/resources/qwen35-4b/"
rsync -aL /mnt/data/jiali_wang/models/qwen35-lora-e5-final/ \
  "$ALGORITHM_ROOT/resources/qwen35-lora/"
du -sh "$ALGORITHM_ROOT/resources/qwen35-4b" \
  "$ALGORITHM_ROOT/resources/qwen35-lora"
```

Audit the offline resource bundle before Docker consumes tens of gigabytes:

```bash
python /home/Jiali_Wang/workspace/VLM-Competition/scripts/check_qwen35_submission_resources.py \
  --algorithm-root "$ALGORITHM_ROOT"
```

Continue only when it prints `"status": "ok"` and `"symlink_count": 0`.

Keep the template Docker base at the previously successful Blackwell-compatible
image, `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`, unless the official
template has since changed to a newer CUDA 12.8 image.

## Required Validation

The old server has driver 470, so a CUDA 12.8 container may fall back to CPU
during the local official test. That is acceptable for interface validation,
although Qwen3.5 can be slow on CPU. Run:

```bash
cd /mnt/data/jiali_wang/workspace/orena-focus-submission-template-qwen35/segment-algorithm
newgrp docker
./do_test_run.sh
cat test/output/interface_1/answer.json
```

Do not save or upload if the log reports any failed or empty responses. After a
clean three-sample test, inspect Docker disk usage and create the tarball:

```bash
df -h /var/lib/docker
docker system df
./do_save.sh
ls -lh segment-algorithm_*.tar.gz
sha256sum segment-algorithm_*.tar.gz
```

Upload the new tarball as a new Algorithm Image. Wait until it is both import
completed and `Active`, run a three-sample Try-out, and only then spend a
pre-evaluation submission.
