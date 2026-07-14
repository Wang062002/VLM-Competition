# ORena FOCUS Research Log

This file records experiment context, decisions, anomalies, and paper-relevant
observations for the ORena FOCUS project. It is intended to preserve enough
detail to support later thesis/paper writing, not just command replay.

## Project Objective

Current milestone: reproduce the official ORena FOCUS SEGMENT inference
pipeline on the HeiCo-FOCUS local test split, then use the reproduced baseline
as the starting point for method design and model training.

Current task:

- Dataset: `heico`
- Track: `SEGMENT`
- Split: `DatasetSplit.TEST`
- Number of QA samples: `4000`
- Number of source videos: `30`
- Baseline model: `Qwen/Qwen3-VL-4B-Instruct`
- Official-style evaluator: official `Evaluator` with `Qwen/Qwen3.5-4B` judge

## Reproducibility Context

Remote server:

- Hostname observed: `UNNC-CVIP-03`
- User: `Jiali_Wang`
- GPUs: `2 x NVIDIA RTX A5000`, 24GB each
- Driver observed: `470.256.02`
- Driver-reported CUDA: `11.4`

Python environment:

- Conda environment: `orena-focus`
- Python: `3.10.20`
- PyTorch: `2.7.1+cu118`
- CUDA available in PyTorch: `True`
- GPU count in PyTorch: `2`

Important paths on the remote server:

- Official repo: `/home/Jiali_Wang/workspace/orena-focus`
- Data root: `/home/Jiali_Wang/data/focus`
- Raw videos: `/home/Jiali_Wang/data/focus/heico/videos`
- Timestamp overlays: `/home/Jiali_Wang/data/focus/heico/overlayed`
- Run outputs: `/home/Jiali_Wang/workspace/focus-runs`

## Official Baseline Alignment

The official inference example uses these key parameters:

- `model_id = "Qwen/Qwen3-VL-4B-Instruct"`
- `dataset = "heico"`
- `track = Track.SEGMENT`
- `split = DatasetSplit.TEST`
- `video_stride = 25`
- `video_resolution = (640, 360)`
- `use_overlay = True`
- `num_eval = 100` in the example; `None` evaluates the full split

Local changes considered environment/runtime compatibility only:

- `root_dir` changed to `/home/Jiali_Wang/data/focus`
- `output_dir` changed to a project run directory
- model loading patched from `device_map="auto"` to explicit
  `torch_dtype=torch.bfloat16` plus `.to(self.device).eval()` because the model
  otherwise landed on CPU in this environment
- `focus/evaluation/judges.py` patched to import
  `AutoModelForCausalLM, AutoTokenizer` at top level

## Dataset And Overlay State

Confirmed data shape:

- `heico` contains `30` complete surgical videos
- `SEGMENT TEST` contains `4000` QA samples
- One source video corresponds to many QA samples
- Each QA sample has a `videoID` plus `start_time` / `end_time`; inference cuts
  the relevant clip from the selected source video

Overlay method:

- Overlay is generated per full video, not per QA sample
- Expected overlay files: `30`
- Expected naming pattern: `<raw video stem>_overlay.mp4`
- Official class: `VideoTimestampOverlayPreprocessor`
- Default overlay text format: `HH:MM:SS`
- Default output folder: `heico/overlayed`

Observed overlay issue:

- Initial overlay file count was `30`, but count alone was insufficient
- Full overlay inference crashed at around `372/4000`
- Error: `IndexError: Out of bound indices: [139150]`
- Root cause: several overlay files existed but were truncated
- Bad videos detected:
  - `0020 - Heico - Sigma - 1.avi`
  - `0021 - Heico - Sigma - 2.avi`
  - `0027 - Heico - Sigma - 8.avi`
  - `0028 - Heico - Sigma - 9.avi`
- These were regenerated before restarting full overlay inference

Paper-relevant point: video preprocessing integrity needs to be verified by
duration/metadata consistency, not only by file count.

## Completed Experiments

### Raw-video smoke test

- Run name: `official-smoke-3`
- Samples: `3`
- Input: raw videos, `use_overlay=False`
- Result: end-to-end pipeline worked
- Score: `0.0`, not meaningful due to sample size

### Raw-video 100-sample baseline

- Run name: `official-smoke-100`
- Samples: `100`
- Input: raw videos, `use_overlay=False`
- Overall MEAN: `0.200000`
- Pre-evaluation SCORE: `0.186795`
- Main observation: temporal/time questions were weak

### Timestamp-overlay 100-sample baseline

- Run name: `official-overlay-100`
- Samples: `100`
- Input: timestamp overlay videos, `use_overlay=True`
- Overall MEAN: `0.210000`
- Pre-evaluation SCORE: `0.200886`
- Main observation: small overall improvement; temporal/time categories improved
  compared with raw 100, but object identification decreased

### Raw-video full baseline

- Run name: `official-raw-full-4000`
- Samples: `4000`
- Input: raw videos, `use_overlay=False`
- Overall MEAN: `0.194250`
- Pre-evaluation SCORE: `0.364083`
- Key category observations:
  - `temporal_grounding`: `0.007741`
  - `temporal_localization`: `0.000000`
  - `time`: `0.003199`
  - `object_recognition`: `0.302873`
  - `event_understanding`: `0.580415`
  - `complex_reasoning`: `0.600999`
- Interpretation: without timestamp overlay, the model is nearly unable to
  answer time-grounded questions, but it retains some ability on event,
  reasoning, and object-related categories.

### Timestamp-overlay full baseline

- Run name: `official-overlay-full-4000`
- Samples: `4000`
- Input: timestamp overlay videos, `use_overlay=True`
- Status: completed after truncated overlay files were regenerated
- Overall MEAN: `0.207500`
- Pre-evaluation SCORE: `0.372647`
- Key category observations:
  - `temporal_grounding`: `0.033822`
  - `temporal_localization`: `0.027867`
  - `time`: `0.029623`
  - `object_recognition`: `0.308308`
  - `event_understanding`: `0.572450`
  - `complex_reasoning`: `0.616896`
- Interpretation: overlay improved full-split performance only modestly. The
  clearest relative gains were on temporal/time categories, but absolute
  temporal performance remained very low.

## Raw Full vs Overlay Full Comparison

Primary comparison:

- Overall MEAN: `0.194250` -> `0.207500` (`+0.013250`)
- Pre-evaluation SCORE: `0.364083` -> `0.372647` (`+0.008564`)
- `temporal_grounding`: `0.007741` -> `0.033822` (`+0.026081`)
- `temporal_localization`: `0.000000` -> `0.027867` (`+0.027867`)
- `time`: `0.003199` -> `0.029623` (`+0.026424`)
- `object_recognition`: `0.302873` -> `0.308308` (`+0.005435`)
- `object_identification`: `0.153572` -> `0.149298` (`-0.004274`)
- `fo_class`: `0.179436` -> `0.175904` (`-0.003532`)

Paper-relevant conclusion:

- Timestamp overlay helps temporal grounding in relative terms, but it does not
  solve the temporal QA problem.
- The overall improvement is modest, suggesting that the bottleneck is not only
  the visibility of timestamps. Surgical domain knowledge, foreign-object
  recognition, visual grounding, and answer-format control remain major
  limitations.
- The overlay result should be reported as the official-style reproduced
  baseline, while the raw result serves as an ablation on explicit timestamp
  availability.

## Working Interpretation

Low accuracy is likely not only an engineering issue. The baseline model
`Qwen/Qwen3-VL-4B-Instruct` has limited systematic medical/surgical training,
which makes foreign-object recognition and surgical-context QA difficult. The
raw-video result also shows that time grounding depends strongly on explicit
timestamp information.

Hypotheses to test after baseline reproduction:

- Timestamp overlay primarily improves temporal grounding and time-format
  answers
- Medical/surgical domain knowledge limits object identification and
  foreign-object class answers
- Prompt constraints and answer-format normalization may improve some categories
  before any model training
- Model training or adaptation should target category knowledge and surgical
  visual grounding, not only generic VLM reasoning

## Next Analysis After Overlay Full Run

After full baseline reproduction:

1. Identify samples where raw is wrong and overlay is correct
2. Identify samples where raw is correct and overlay is wrong
3. Save representative failure cases with:
   - `qID`
   - `videoID`
   - question
   - reference answer
   - model response
   - task category
   - answer format

## Paper-Oriented Notes To Preserve

- Baseline reproduction required fixing an environment-specific model placement
  issue where automatic device mapping placed the model on CPU
- Official evaluator required a missing `transformers` import patch in the
  checked-out code
- Overlay preprocessing should be validated by duration coverage against QA
  metadata; a complete file count can still hide truncated videos
- The raw full result demonstrates a strong failure mode on temporal grounding
- The overlay full result demonstrates that explicit timestamps help temporal
  grounding but are insufficient by themselves
- The project should separate:
  - official reproduction
  - preprocessing validation
  - prompt/inference improvements
  - retrieval/rule-assisted improvements
  - model training or fine-tuning

## Planned Method Stage: LoRA-SFT

After reproducing the raw and overlay baselines, the next method stage is
parameter-efficient supervised fine-tuning:

```text
Qwen3-VL-4B-Instruct + ORena FOCUS video-QA SFT + LoRA adapters
```

Important distinction:

- SFT defines the supervised training objective and data format.
- LoRA defines how the model is adapted efficiently without updating all
  parameters.

Before training, the competition rules and official dataset splits must be
checked to avoid leakage. The `DatasetSplit.TEST` labels used for local baseline
evaluation should not automatically be treated as training data for any result
that will be reported as held-out performance.

Official split status checked on 2026-07-07:

- The official dataset card describes per-track annotation files under
  `data/frame/`, `data/segment/`, and `data/procedure/`, each with
  `train.parquet` and `test.parquet`.
- The official library exposes `DatasetSplit.TRAIN`, `DatasetSplit.TEST`, and
  `DatasetSplit.ALL`.
- For LoRA-SFT, the default safe route is to use
  `FocusDataset("heico", DatasetSplit.TRAIN, Track.SEGMENT)` for training and
  keep `DatasetSplit.TEST` as held-out local evaluation unless the competition
  rules state otherwise.

Detailed plan: `docs/lora_sft_training_plan.md`.

Execution artifact:

- `scripts/audit_and_split_segment_train.py` creates a deterministic 90/10
  internal train/validation split from official SEGMENT TRAIN only, preserving
  official SEGMENT TEST as held-out evaluation data.

Executed on 2026-07-09:

- Command output directory:
  `/home/Jiali_Wang/workspace/focus-runs/data-audit/segment-trainval-seed20260707`
- Random seed: `20260707`
- Validation fraction: `0.1`
- Official SEGMENT TRAIN: `8000`
- Official SEGMENT TEST: `4000`
- Internal train from official TRAIN: `7198`
- Internal validation from official TRAIN: `802`
- Policy confirmed: official TEST remains held out for final local evaluation.

LoRA-SFT smoke test executed on 2026-07-09:

- Script: `scripts/train_qwen3vl_lora_sft_smoke.py`
- Base model: `Qwen/Qwen3-VL-4B-Instruct`
- Training manifest:
  `/home/Jiali_Wang/workspace/focus-runs/data-audit/segment-trainval-seed20260707/sft_train_overlay.jsonl`
- Validation manifest:
  `/home/Jiali_Wang/workspace/focus-runs/data-audit/segment-trainval-seed20260707/sft_val_overlay.jsonl`
- Train samples: `32`
- Validation samples: `8`
- Optimizer steps: `8`
- Validation loss: `1.0017873756587505`
- Adapter output:
  `/home/Jiali_Wang/workspace/focus-runs/lora-sft/qwen3vl-4b-smoke-32/adapter-final`
- Training history:
  `/home/Jiali_Wang/workspace/focus-runs/lora-sft/qwen3vl-4b-smoke-32/train_history.jsonl`

Interpretation:

- The Qwen3-VL LoRA-SFT training plumbing works end to end.
- Verified components include JSONL manifest reading, overlay clip extraction,
  Qwen processor encoding, LoRA adapter injection, backward pass, optimizer
  step, validation loss computation, and adapter saving.
- This run is not intended to produce a useful model; it is a technical smoke
  test before scaling.

## Project Maintenance Rule Added On 2026-07-10

The project now has an explicit maintenance rule:

- Every substantive experiment, script change, data issue, evaluation result,
  or paper-relevant conclusion must be recorded in the project knowledge base.
- The recovery entry point is `knowledge_base/START_HERE.md`.
- The detailed protocol is `knowledge_base/maintenance_protocol.md`.
- Paper-supporting notes should be kept in `knowledge_base/paper_notes.md`,
  `docs/research_log.md`, and structured `results/*.csv` files.
- Workflow and script changes should be reflected in
  `knowledge_base/workflows.md` and `docs/script_workflow_explained.md`.
- Stable progress should be committed and pushed to GitHub.

This rule is meant to prevent memory loss after context compression and to keep
the project ready for later thesis/paper writing.

## Full LoRA-SFT TEST Evaluation On 2026-07-10

The first full clip-valid LoRA-SFT adapter was evaluated on the full 4000-sample
official TEST split using timestamp-overlay videos.

Run:

- `qwen3vl-4b-sft-valid5959-e1-overlay-test-full`

Result:

- Overall MEAN: `0.279000`
- Pre-evaluation SCORE: `0.402794`

Comparison to reproduced overlay baseline:

- Overlay baseline overall MEAN: `0.207500`
- LoRA overall MEAN: `0.279000`
- Overall delta: `+0.071500`
- Overlay baseline pre-evaluation SCORE: `0.372647`
- LoRA pre-evaluation SCORE: `0.402794`
- Pre-evaluation delta: `+0.030147`

Most important category-level changes:

- `object_identification`: `0.149298 -> 0.469223`, delta `+0.319925`
- `fo_class`: `0.175904 -> 0.437977`, delta `+0.262073`
- `object_recognition`: `0.308308 -> 0.472173`, delta `+0.163865`
- `temporal_grounding`: `0.033822 -> 0.071740`, delta `+0.037918`
- `time`: `0.029623 -> 0.064236`, delta `+0.034613`

Interpretation:

- Full held-out TEST confirms that the first clip-valid LoRA-SFT adapter improves
  over the reproduced overlay baseline.
- The improvement is strongest for object/foreign-object recognition.
- Temporal grounding improves but remains poor in absolute terms.
- Multiple-choice accuracy decreased slightly and should be inspected before the
  next method iteration.

Summary tables added:

- `docs/comprehensive_data_comparison.md`
- `results/all_data_comparison_table.csv`
- `results/main_result_summary.csv`
- `results/lora_full_test_vs_overlay_baseline.csv`

## Open VLM Baseline Search Started On 2026-07-14

After confirming that Qwen3-VL-4B + LoRA improves over the reproduced overlay
baseline, the project moved to open-source VLM baseline search. The goal is to
find stronger zero-shot or prompt-only models before investing in another
fine-tuning stage.

First-batch candidates:

- `openbmb/MiniCPM-V-4_5`
- `llava-hf/llava-onevision-qwen2-7b-ov-hf`
- `OpenGVLab/InternVL3_5-8B-Instruct`
- `google/gemma-3-12b-it`
- `google/medgemma-4b-it`

Artifacts added:

- `configs/vlm_candidate_models.csv`
- `scripts/download_vlm_candidates.py`
- `docs/open_vlm_baseline_plan.md`

Testing plan:

- Stage 1: smoke test, `num_eval=10` or `30`
- Stage 2: TEST-100 for models that can run
- Stage 3: full TEST-4000 only for the strongest 1-2 candidates

Download note:

- If a download is interrupted, a non-empty target directory may be incomplete.
  The downloader was updated to verify/resume existing directories by default
  instead of skipping them.
- `google/gemma-3-12b-it` can return Hugging Face `GatedRepoError` / `403
  Forbidden` until access is accepted with the same Hugging Face account used on
  the remote server.
- After access is accepted, Gemma may still fail in the `hf-xet` transfer layer
  with `Unable to parse string as hex hash value`. The downloader now supports
  `--disable-xet`, which sets `HF_HUB_DISABLE_XET=1` before importing
  `huggingface_hub`.

Next checkpoint:

- After all five candidate snapshots are downloaded, run
  `scripts/check_vlm_downloads.py` to confirm local directories, `config.json`,
  weight files, and manifest status before model loading.
