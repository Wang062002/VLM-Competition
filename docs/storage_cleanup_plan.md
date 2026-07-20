# Storage Migration And Model Cleanup Plan

Date: `2026-07-20`

The remote server has a new mounted disk at `/mnt/data`. Large project files
should use a user-owned directory:

```bash
/mnt/data/jiali_wang
```

## Rationale

- The main system disk should not keep growing with datasets, model snapshots,
  Hugging Face cache, and long-running experiment artifacts.
- Full evaluations showed that the open VLM candidates did not outperform the
  Qwen3-VL baseline/mainline.
- Candidate model snapshots are large and can be removed after their metrics
  have been recorded in this repository.

## Keep

- Code repositories:
  - `~/workspace/VLM-Competition`
  - `~/workspace/orena-focus`
- Conda environment:
  - `~/tools/miniconda3`
- Qwen-related Hugging Face cache, unless it is intentionally re-downloadable.
- Experiment results and CSV summaries:
  - `~/workspace/focus-runs`
- LoRA adapters and logs:
  - `~/workspace/focus-runs/lora-sft`

## Move To `/mnt/data/jiali_wang`

Recommended large-storage layout:

```bash
/mnt/data/jiali_wang/focus
/mnt/data/jiali_wang/focus-runs
/mnt/data/jiali_wang/hf-cache
/mnt/data/jiali_wang/tmp
```

The safest migration is copy-first, verify, then replace the old path with a
symlink.

## Remote Commands

Check disk state:

```bash
df -h
du -sh ~/data/focus ~/workspace/vlm-models ~/workspace/focus-runs ~/.cache/huggingface 2>/dev/null
```

Create directories:

```bash
mkdir -p /mnt/data/jiali_wang/focus
mkdir -p /mnt/data/jiali_wang/focus-runs
mkdir -p /mnt/data/jiali_wang/hf-cache
mkdir -p /mnt/data/jiali_wang/tmp
```

Copy the FOCUS dataset to the new disk:

```bash
rsync -aH --info=progress2 ~/data/focus/ /mnt/data/jiali_wang/focus/
```

Verify before deleting or replacing anything:

```bash
du -sh ~/data/focus /mnt/data/jiali_wang/focus
find ~/data/focus/heico/overlayed -type f -name '*_overlay.mp4' | wc -l
find /mnt/data/jiali_wang/focus/heico/overlayed -type f -name '*_overlay.mp4' | wc -l
```

Switch the old path to a symlink after verification:

```bash
mv ~/data/focus ~/data/focus.before_mnt_data_migration
ln -s /mnt/data/jiali_wang/focus ~/data/focus
```

Keep the backup until one dataset load and one small inference smoke test pass.
Then remove the backup if space is needed:

```bash
rm -rf ~/data/focus.before_mnt_data_migration
```

Clean open-VLM candidate snapshots after metrics are recorded:

```bash
du -sh ~/workspace/vlm-models
find ~/workspace/vlm-models -maxdepth 1 -mindepth 1 -type d -print
```

If the listed directories are only the open-VLM candidates, remove them:

```bash
rm -rf ~/workspace/vlm-models
```

Do not remove `~/workspace/focus-runs`, `~/workspace/VLM-Competition`,
`~/workspace/orena-focus`, or Qwen/LoRA result directories.

## Future Environment Variables

For future large downloads, prefer:

```bash
export HF_HOME=/mnt/data/jiali_wang/hf-cache
export TRANSFORMERS_CACHE=/mnt/data/jiali_wang/hf-cache
export TMPDIR=/mnt/data/jiali_wang/tmp
```

These can be added to a project run script or exported manually before large
download/training jobs.
