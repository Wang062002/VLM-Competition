# Codex Working Memory: ORena FOCUS Baseline

This file compresses the useful state from the ORena FOCUS setup/debugging
session. It intentionally omits passwords, Hugging Face tokens, and verbose SSH
troubleshooting.

## Current Goal

Run the official ORena FOCUS `SEGMENT` baseline on the official local test split,
then compare raw-video vs timestamp-overlay video input.

Official baseline alignment should use the official `examples/inference.py`
defaults wherever possible:

- `dataset_name`: `heico`
- `track`: `Track.SEGMENT`
- `split`: `DatasetSplit.TEST`
- `model_id`: `Qwen/Qwen3-VL-4B-Instruct`
- `use_overlay`: `True`
- `video_stride`: `25`
- `video_resolution`: `(640, 360)`
- `num_eval`: `100` for quick baseline, then optionally full `4000`

Only environment/path/runtime compatibility changes should differ from official:

- `root_dir`: `/home/Jiali_Wang/data/focus`
- `output_dir`: under `/home/Jiali_Wang/workspace/focus-runs`
- model load patched if needed so it actually runs on GPU
- official evaluator import bug patched if needed

## Remote Server State

Remote server:

- Host: `10.176.61.126`
- User: `Jiali_Wang`
- Hostname observed: `UNNC-CVIP-03`
- GPUs: `2 x NVIDIA RTX A5000`, 24GB each
- Driver observed: `470.256.02`, CUDA shown by driver: `11.4`

Do not store or repeat the server password.

Important remote paths:

- Official repo: `/home/Jiali_Wang/workspace/orena-focus`
- Conda install: `/home/Jiali_Wang/tools/miniconda3`
- Conda env: `orena-focus`
- Data root: `/home/Jiali_Wang/data/focus`
- Run outputs: `/home/Jiali_Wang/workspace/focus-runs`

Activate environment on every new terminal:

```bash
source ~/tools/miniconda3/etc/profile.d/conda.sh
conda activate orena-focus
cd ~/workspace/orena-focus
```

VS Code Remote-SSH is working. The local VS Code window title should include:

```text
orena-focus [SSH: 10.176.61.126]
```

VS Code may show Python 3.8 in the status bar. That is not relevant when running
from a terminal whose prompt is `(orena-focus)`.

## Environment State

Working Python env:

- Python: `3.10.20`
- PyTorch: `2.7.1+cu118`
- `torch.cuda.is_available()`: `True`
- `torch.cuda.device_count()`: `2`

The system Python is `3.8.10` and should not be used for this project.

Hugging Face auth is configured on the remote server. Do not store the token.

## Data State

Downloaded/available:

- `heico` videos: `/home/Jiali_Wang/data/focus/heico/videos`
- raw video size observed: `150G`
- timestamp overlays: `/home/Jiali_Wang/data/focus/heico/overlayed`
- overlay count observed: `30`
- overlay size observed: `81G`

Dataset verified:

```python
FocusDataset("heico", DatasetSplit.TEST, Track.SEGMENT)
```

Observed:

- split size: `4000`
- first sample example:
  - `qID`: `1956569`
  - `videoID`: `0029 - Heico - Sigma - 10.avi`
  - answer: `Clip`

Not downloaded/used yet:

- `lapchole`

Local CSV records:

- `results/experiment_log.csv`
- `results/official_smoke_100_breakdown.csv`
- `results/dataset_status.csv`

## Completed Runs

### Raw-video smoke test

Run:

- name: `official-smoke-3`
- samples: `3`
- input: raw video, `use_overlay=False`
- result: pipeline ran end-to-end; score was `0.0`, not meaningful due to tiny sample

### Raw-video 100-sample baseline

Run:

- name: `official-smoke-100`
- samples: `100`
- input: raw video, `use_overlay=False`
- model: `Qwen/Qwen3-VL-4B-Instruct`
- evaluator judge: `Qwen/Qwen3.5-4B`

Result:

- `overall MEAN`: `0.200000`
- `pre_evaluation SCORE`: `0.186795`
- `object_recognition`: `0.351852`
- `temporal_grounding`: `0.021739`
- `time`: `0.021739`

Interpretation:

- Raw-video baseline is valid as an ablation/sanity baseline.
- It is not the most official-aligned baseline because official inference uses
  `use_overlay=True`.
- Main weakness is temporal/time questions, likely because raw video lacks visible
  timestamps.

### Timestamp-overlay 100-sample baseline

Run:

- name: `official-overlay-100`
- samples: `100`
- input: timestamp overlay videos, `use_overlay=True`
- model: `Qwen/Qwen3-VL-4B-Instruct`
- evaluator judge: `Qwen/Qwen3.5-4B`

Result:

- `overall MEAN`: `0.210000`
- `pre_evaluation SCORE`: `0.200886`

Interpretation:

- This is closer to the official example configuration than the raw-video run.
- Temporal/time performance improved compared with raw 100, while some
  object-identification metrics decreased.

### Raw-video full baseline

Run:

- name: `official-raw-full-4000`
- samples: `4000`
- input: raw video, `use_overlay=False`

Result:

- `overall MEAN`: `0.194250`
- `pre_evaluation SCORE`: `0.364083`
- `temporal_grounding`: `0.007741`
- `temporal_localization`: `0.000000`
- `time`: `0.003199`

Interpretation:

- The raw full baseline shows that Qwen3-VL-4B is nearly unable to solve
  temporal grounding without visible timestamps.
- The higher pre-evaluation score relative to overall accuracy likely comes
  from official grouping/bucket averaging rather than simple sample averaging.

### Timestamp-overlay full baseline

Run:

- name: `official-overlay-full-4000`
- samples: `4000`
- input: timestamp overlay videos, `use_overlay=True`
- status: completed as of 2026-07-07 after overlay repair

Result:

- `overall MEAN`: `0.207500`
- `pre_evaluation SCORE`: `0.372647`
- `temporal_grounding`: `0.033822`
- `temporal_localization`: `0.027867`
- `time`: `0.029623`
- `object_recognition`: `0.308308`
- `object_identification`: `0.149298`
- `fo_class`: `0.175904`

Important preprocessing repair:

- Initial overlay count was `30`, but 4 overlay files were truncated.
- Bad videos:
  - `0020 - Heico - Sigma - 1.avi`
  - `0021 - Heico - Sigma - 2.avi`
  - `0027 - Heico - Sigma - 8.avi`
  - `0028 - Heico - Sigma - 9.avi`
- These were regenerated before restarting full overlay inference.

Paper note:

- Overlay correctness should be validated by duration coverage against QA
  metadata, not merely by counting overlay files.
- Overlay full improved raw full modestly: overall `0.194250 -> 0.207500`,
  pre-evaluation `0.364083 -> 0.372647`.
- Temporal categories improved in relative terms but remained weak:
  `temporal_grounding 0.007741 -> 0.033822`, `time 0.003199 -> 0.029623`.

## LoRA-SFT Stage

Official split status:

- `heico` SEGMENT TRAIN: `8000` samples
- `heico` SEGMENT TEST: `4000` samples
- The previous raw/overlay full baselines were run on official TEST and must
  remain held-out evaluation data.

Training policy:

- Use official TRAIN only for LoRA-SFT.
- Create an internal train/validation split from official TRAIN.
- Keep official TEST untouched for final local evaluation and paper reporting.

Local execution artifact:

- `scripts/audit_and_split_segment_train.py`

Remote command:

```bash
cd ~/workspace/VLM-Competition
python scripts/audit_and_split_segment_train.py \
  --root-dir /home/Jiali_Wang/data/focus \
  --output-dir ~/workspace/focus-runs/data-audit/segment-trainval-seed20260707
```

Expected outputs:

- `split_counts.csv`
- `distribution_by_official_split_primary.csv`
- `distribution_by_official_split_answer_format.csv`
- `distribution_by_official_split_primary_answer_format.csv`
- `train_internal_split_manifest.csv`
- `sft_train_overlay.jsonl`
- `sft_val_overlay.jsonl`
- `audit_summary.json`

Execution result on 2026-07-09:

- Output directory:
  `/home/Jiali_Wang/workspace/focus-runs/data-audit/segment-trainval-seed20260707`
- Official TRAIN: `8000`
- Official TEST: `4000`
- Internal train: `7198`
- Internal val: `802`
- Seed: `20260707`

LoRA-SFT smoke result on 2026-07-09:

- Script: `scripts/train_qwen3vl_lora_sft_smoke.py`
- Base model: `Qwen/Qwen3-VL-4B-Instruct`
- Train samples: `32`
- Val samples: `8`
- Optimizer steps: `8`
- Eval loss: `1.0017873756587505`
- Adapter:
  `/home/Jiali_Wang/workspace/focus-runs/lora-sft/qwen3vl-4b-smoke-32/adapter-final`
- Meaning: end-to-end LoRA-SFT plumbing is validated, but this is not a useful
  trained model yet.

## Important Debugging Notes

### Model accidentally loaded on CPU

The official script used `device_map="auto"` and the model landed on CPU in this
environment. Symptom:

```text
model is on cpu
input_ids is on cuda
```

GPU memory was only about `800MiB`.

Runtime patch used in experiment scripts:

```python
self.model = Qwen3VLForConditionalGeneration.from_pretrained(
    self.model_id,
    torch_dtype=torch.bfloat16,
).to(self.device).eval()
```

After patch, Python GPU memory rose to about `9359MiB`, confirming model on GPU.

### Official evaluator import bug

The evaluator reached `focus/evaluation/judges.py` and failed with:

```text
NameError: name 'AutoTokenizer' is not defined
```

The file uses `AutoTokenizer` and `AutoModelForCausalLM`, but the required
transformers import was missing in the checked-out version.

Patch should be inserted at top-level after `import requests`, not inside a
function:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
```

Important: a previous bad patch inserted this import inside `judge()` and caused:

```text
IndentationError: unexpected indent (judges.py, line 200)
```

Recovery/fix command:

```bash
cd ~/workspace/orena-focus
git checkout -- src/focus/evaluation/judges.py

python - <<'PY'
from pathlib import Path

p = Path("src/focus/evaluation/judges.py")
lines = p.read_text().splitlines()

lines = [
    line for line in lines
    if line.strip() not in {
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "from transformers import AutoTokenizer",
        "from transformers import AutoModelForCausalLM",
    }
]

for i, line in enumerate(lines):
    if line.strip() == "import requests":
        lines.insert(i + 1, "from transformers import AutoModelForCausalLM, AutoTokenizer")
        break
else:
    raise RuntimeError("Could not find `import requests`")

p.write_text("\n".join(lines) + "\n")
print("fixed judges.py import")
PY

python -m py_compile src/focus/evaluation/judges.py
```

Only rerun inference/evaluation after `py_compile` succeeds without output.

## Recommended Next Step

Run the official-style overlay 100-sample baseline.

Preparation:

```bash
source ~/tools/miniconda3/etc/profile.d/conda.sh
conda activate orena-focus
cd ~/workspace/orena-focus
python -m py_compile src/focus/evaluation/judges.py
```

Create a fresh official-aligned script from official `examples/inference.py`:

```bash
cp examples/inference.py examples/inference_official_overlay_100.py

python - <<'PY'
from pathlib import Path

p = Path("examples/inference_official_overlay_100.py")
s = p.read_text()

s = s.replace('"root_dir": "/data/focus"', '"root_dir": "/home/Jiali_Wang/data/focus"')
s = s.replace('"output_dir": None', '"output_dir": "/home/Jiali_Wang/workspace/focus-runs/official-overlay-100"')

s = s.replace(
'''self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto" if self.device == "cuda" else None,
        ).eval()''',
'''self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        ).to(self.device).eval()'''
)

p.write_text(s)
print("wrote", p)
PY
```

Verify official-aligned parameters:

```bash
grep -n '"use_overlay"\|"video_stride"\|"video_resolution"\|"num_eval"\|"root_dir"\|"output_dir"' examples/inference_official_overlay_100.py
grep -nA4 -B2 "from_pretrained" examples/inference_official_overlay_100.py
```

Expected:

- `use_overlay=True`
- `num_eval=100`
- `video_stride=25`
- `video_resolution=(640, 360)`
- model load uses `.to(self.device).eval()`

Run:

```bash
mkdir -p ~/workspace/focus-runs/official-overlay-100
python examples/inference_official_overlay_100.py
```

After completion, append results to:

- `results/experiment_log.csv`
- optionally create `results/official_overlay_100_breakdown.csv`

## Compressed / Low-Value Context

These details were useful during setup but are no longer central:

- VPN conflict: local proxy VPN and school VPN conflicted. For SSH, school VPN
  must route `10.176.61.126`; Codex may need local proxy VPN separately.
- VS Code Remote-SSH setup succeeded after password correction.
- Miniconda was installed manually under `~/tools/miniconda3` because server had
  no conda/mamba and default Python was 3.8.
- Anaconda ToS had to be accepted before creating the env.
- `git` was missing system-wide and installed via conda.
- Hugging Face dataset download initially hit a `403` through Xet. It worked with:
  `export HF_HUB_DISABLE_XET=1`.
- HeiCo download appeared to grow from ~80G to >100G because Hugging Face showed
  `Downloading (incomplete total...)`.
- Timestamp overlay generation took hours because it re-encoded 30 large videos;
  this was not a loop.
- Computer-use can inspect VS Code UI, but terminal copy/paste remains more
  reliable than automating terminal input through the GUI.

## Current Open VLM Baseline State

As of 2026-07-15, five open VLM candidates were downloaded and smoke-tested:

- MiniCPM-V-4_5
- LLaVA-OneVision-Qwen2-7B
- InternVL3.5-8B
- Gemma-3-12B-IT
- MedGemma-4B-IT

Key TEST-100 class-constrained results:

- Qwen3-VL LoRA TEST-100 remains strongest overall: `0.350000`.
- MedGemma-4B, 4 frames: overall `0.270000`.
- MedGemma-4B, 8 frames: overall `0.290000`; temporal/time improved from
  `0.021739` to `0.130435`.
- LLaVA-OneVision, 4 frames: overall `0.260000`.
- LLaVA-OneVision, 8 frames: overall `0.240000`; `fo_class` improved from
  `0.391304` to `0.434783`.

Current next recommended remote run:

```bash
source ~/tools/miniconda3/etc/profile.d/conda.sh
conda activate orena-focus
cd ~/workspace/VLM-Competition

python scripts/run_open_vlm_smoke.py \
  --model medgemma_4b \
  --model-dir ~/workspace/vlm-models \
  --root-dir /home/Jiali_Wang/data/focus \
  --num-eval none \
  --frames-per-clip 8 \
  --prompt-mode class_constrained \
  --normalize-answer \
  --output-dir ~/workspace/focus-runs/open-vlm-smoke/test4000-medgemma-8frames-class-prompt
```

That full run has now completed:

- Overall MEAN: `0.188250`
- Pre-evaluation SCORE: `0.281741`
- Processed: `4000`
- Failures: `0`
- Output:
  `/home/Jiali_Wang/workspace/focus-runs/open-vlm-smoke/test4000-medgemma-8frames-class-prompt/medgemma_4b`

Interpretation:

- MedGemma-4B is not the strongest full prompt-only baseline by overall score;
  it is below Qwen3-VL overlay full overall `0.207500`.
- It is still selected as the next training target because it is medically
  oriented and has promising object-recognition behavior.
- Next stage: implement MedGemma-4B LoRA/SFT training on the existing
  clip-valid official-TRAIN-derived JSONL files.
