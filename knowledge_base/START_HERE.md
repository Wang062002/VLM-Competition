# START HERE: ORena FOCUS Project Memory

Read this file first after context compression. It points to the minimum set of
project memory needed to continue smoothly.

## Current Stage

The project has moved from baseline reproduction to LoRA-SFT method development.

Completed:

- official Qwen3-VL SEGMENT baseline reproduction on HeiCo TEST
- raw vs timestamp-overlay full 4000 comparison
- official TRAIN/TEST split audit
- deterministic internal train/validation split from official TRAIN
- Qwen3-VL LoRA-SFT 32-sample smoke test
- Qwen3-VL LoRA-SFT 512-sample filtered medium test
- Qwen3-VL full clip-valid LoRA-SFT run
- first-batch open-source VLM download and integrity check
- open-VLM TEST-3, TEST-30, TEST-100 prompt ablations
- LLaVA / MedGemma TEST-100 4-frame vs 8-frame comparison

Current next step:

- run full TEST-4000 for the strongest prompt-only open VLM setting so far:
  MedGemma-4B with class-constrained prompt, answer normalization, and 8 sampled
  frames per clip
- keep Qwen3-VL + LoRA as the current strongest trained baseline
- use open-VLM TEST-100 results as model-selection evidence, not final
  performance claims

## Must-Preserve Rules

- Official TEST split is held out. Do not train on it.
- Official TRAIN split is the source for LoRA-SFT.
- Baseline comparison must use the reproduced TEST metrics already logged.
- Timestamp overlay should be generated with official
  `VideoTimestampOverlayPreprocessor` defaults.
- Overlay integrity must be checked by duration coverage, not file count only.
- Do not store passwords or Hugging Face tokens in project files.
- After every substantive project progress, update the memory library, paper
  notes, workflow docs, structured result files, and GitHub commit history.
  See `knowledge_base/maintenance_protocol.md`.

## Key Files

- `knowledge_base/project_state.md`: project facts, paths, remote state
- `knowledge_base/experiments.md`: baseline and training experiment results
- `knowledge_base/workflows.md`: repeatable command workflows
- `knowledge_base/training_stage.md`: LoRA-SFT plan and current status
- `knowledge_base/paper_notes.md`: thesis/paper observations and claims
- `knowledge_base/maintenance_protocol.md`: rules for keeping memory, paper
  notes, workflows, and GitHub versions updated after each progress step
- `knowledge_base/sync_commands.md`: local-to-remote upload commands
- `docs/research_log.md`: longer research log
- `docs/lora_sft_training_plan.md`: detailed LoRA-SFT plan
- `docs/script_workflow_explained.md`: explanation of each script and what it
  did in the project workflow
- `docs/comprehensive_data_comparison.md`: Chinese summary tables covering
  datasets, splits, training runs, baseline results, LoRA results, and deltas
- `results/all_data_comparison_table.csv`: machine-readable all-data comparison
  table
- `results/main_result_summary.csv`: main paper-ready result table
- `results/open_vlm_test100_class_prompt_selected.csv`: TEST-100 class-prompt
  open-VLM comparison
- `results/open_vlm_test100_frame_ablation.csv`: TEST-100 4-frame vs 8-frame
  ablation for LLaVA and MedGemma
- `results/lora_full_test_vs_overlay_baseline.csv`: full TEST LoRA-vs-overlay
  delta table
- `results/evaluator_style_full_4000_summaries.csv`: evaluator-style long table
  with `experiment, level, name, accuracy, ci_low, ci_high, count`
- `results/evaluator_style_full_4000_summaries.txt`: terminal-style evaluator
  output matching official summary formatting
- `docs/evaluator_style_summary_tables.md`: explanation and print command for
  evaluator-style tables
- `report/README.md`: paper/report artifact requirements, including the
  screenshot-style evaluator breakdown table format
- `scripts/print_evaluator_style_summary.py`: script for printing evaluator-style
  tables from CSV
- `configs/vlm_candidate_models.csv`: first-batch open VLM candidate list
- `scripts/download_vlm_candidates.py`: remote Hugging Face snapshot downloader
  for candidate VLMs
- `scripts/check_vlm_downloads.py`: lightweight local snapshot check before
  smoke / batch evaluation
- `scripts/run_open_vlm_smoke.py`: common multi-frame smoke runner for the five
  downloaded VLM candidates; supports `--prompt-mode class_constrained` and
  `--normalize-answer`
- `docs/open_vlm_baseline_plan.md`: plan for downloading and batch-testing
  open-source VLM baselines
- `codex.md`: compact operational memory

## Critical Numbers

Dataset:

- HeiCo source videos: `30`
- SEGMENT official TRAIN: `8000`
- SEGMENT official TEST: `4000`
- Internal train from official TRAIN: `7198`
- Internal val from official TRAIN: `802`
- Clip-valid internal train: `5959`
- Clip-valid internal val: `663`
- Split seed: `20260707`

Baselines on official TEST:

- raw full 4000 overall: `0.194250`
- raw full 4000 pre-eval: `0.364083`
- overlay full 4000 overall: `0.207500`
- overlay full 4000 pre-eval: `0.372647`

LoRA-SFT smoke:

- train samples: `32`
- val samples: `8`
- optimizer steps: `8`
- eval loss: `1.0017873756587505`
- adapter:
  `/home/Jiali_Wang/workspace/focus-runs/lora-sft/qwen3vl-4b-smoke-32/adapter-final`

LoRA-SFT medium:

- requested train samples: `512`
- requested val samples: `128`
- valid train samples: `512`
- valid val samples: `99`
- invalid val clip rows: `29`
- optimizer steps: `128`
- eval loss: `0.35957938603553546`
- adapter:
  `/home/Jiali_Wang/workspace/focus-runs/lora-sft/qwen3vl-4b-smoke-512-filtered/adapter-final`

LoRA-SFT full clip-valid:

- train samples: `5959`
- val samples: `663`
- optimizer steps: `1490`
- eval loss: `0.42800752680308324`
- adapter:
  `/home/Jiali_Wang/workspace/focus-runs/lora-sft/qwen3vl-4b-sft-valid5959-e1/adapter-final`

LoRA-SFT TEST-100:

- official TEST samples: `100`
- overall MEAN: `0.350000`
- pre-evaluation SCORE: `0.328905`
- compared with `official-overlay-100` overall `0.210000`, delta `+0.140000`

LoRA-SFT full TEST:

- official TEST samples: `4000`
- overall MEAN: `0.279000`
- pre-evaluation SCORE: `0.402794`
- compared with `official-overlay-full-4000` overall `0.207500`, delta
  `+0.071500`
- largest gains:
  - `object_identification`: `+0.319925`
  - `fo_class`: `+0.262073`
  - `object_recognition`: `+0.163865`
- temporal grounding improved but remains weak:
  - `temporal_grounding`: `0.033822 -> 0.071740`
  - `time`: `0.029623 -> 0.064236`

Open VLM TEST-100 class-constrained, 4 frames:

- MedGemma-4B overall: `0.270000`
- LLaVA-OneVision-7B overall: `0.260000`
- MiniCPM-V-4_5 overall: `0.150000`
- InternVL3.5-8B overall: `0.150000`

Open VLM TEST-100 frame ablation:

- LLaVA-OneVision: `4` frames overall `0.260000`; `8` frames overall
  `0.240000`
- LLaVA `fo_class`: `0.391304 -> 0.434783` when increasing `4 -> 8` frames
- MedGemma: `4` frames overall `0.270000`; `8` frames overall `0.290000`
- MedGemma temporal/time: `0.021739 -> 0.130435` when increasing `4 -> 8`
  frames
- Current best prompt-only open VLM setting: MedGemma-4B, 8 frames,
  class-constrained prompt, answer normalization

## Current Recommended Remote Command

Run the current strongest prompt-only open VLM on full TEST-4000:

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
