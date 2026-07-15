# Paper Notes

## Maintenance Rule

After every substantive project progress, update this file with any
paper-relevant material:

- method changes
- dataset or preprocessing rules
- training settings
- evaluation settings
- supported conclusions
- unsupported hypotheses
- table-ready numbers
- limitations and failure modes

Small-sample results must be labelled as preliminary until full held-out TEST
evaluation confirms them.

## Reproducible Story So Far

1. Reproduced official Qwen3-VL inference pipeline on HeiCo SEGMENT.
2. Compared raw video vs official timestamp-overlay input on the full official
   TEST split.
3. Found that overlay improves temporal grounding relatively but only modestly
   overall.
4. Identified domain knowledge, foreign-object recognition, answer format, and
   temporal grounding as major bottlenecks.
5. Built leakage-safe LoRA-SFT train/val split from official TRAIN.
6. Verified that Qwen3-VL LoRA-SFT training plumbing works.
7. Found that TRAIN-derived SFT manifests require clip-duration validation, not
   only file-existence validation.
8. Completed the first full clip-valid Qwen3-VL LoRA-SFT run.
9. Evaluated the trained LoRA adapter on the full 4000-sample held-out TEST
   split.

## Claims Supported By Current Evidence

Supported:

- Timestamp overlay is useful but insufficient.
- Raw videos lead to near-zero temporal localization/time performance.
- Qwen3-VL-4B baseline struggles with surgical domain-specific QA.
- Preprocessing integrity matters; overlay file count alone is insufficient.
- Training data integrity also requires checking that each QA time window is
  inside the referenced video duration.
- Official TEST should remain held out when training on official TRAIN.
- Clip-valid LoRA-SFT improves full held-out TEST performance over the
  reproduced overlay baseline.
- The largest LoRA-SFT gains are in object recognition, object identification,
  and foreign-object class answers.

Not yet supported:

- Larger training improves temporal grounding.
- Any specific LoRA hyperparameter is optimal.
- Prompt-only improvements are weaker/stronger than LoRA-SFT.

## Useful Wording

Baseline reproduction:

> We reproduced the official Qwen3-VL-4B inference pipeline for the HeiCo-FOCUS
> SEGMENT track and evaluated both raw-video and timestamp-overlay variants on
> the official local TEST split.

Overlay ablation:

> Timestamp overlays improved temporal metrics in relative terms, but the
> absolute temporal grounding accuracy remained low, indicating that explicit
> time visibility alone is insufficient for robust surgical temporal QA.

Training plan:

> We use the official SEGMENT TRAIN split for supervised adaptation and keep the
> official SEGMENT TEST split strictly held out for final local evaluation.

Full LoRA-SFT status:

> We trained a Qwen3-VL-4B LoRA adapter for one epoch on the 5,959 clip-valid
> training samples and evaluated validation loss on 663 clip-valid validation
> samples from the official TRAIN split. This establishes a trained adapter for
> subsequent held-out TEST evaluation but does not by itself establish task-level
> improvement.

TEST-100 adapter result:

> On a 100-sample held-out TEST subset using timestamp-overlay videos, the
> clip-valid LoRA-SFT adapter improved overall accuracy from 0.210 to 0.350
> compared with the reproduced overlay baseline. This motivates full TEST-set
> evaluation but should be treated as preliminary because of the small sample
> size.

Full TEST adapter result:

> On the full 4,000-sample held-out TEST split, the clip-valid LoRA-SFT adapter
> improved overall accuracy from 0.2075 to 0.2790 over the reproduced
> timestamp-overlay baseline. The largest gains occurred in object
> identification and foreign-object class answers, while temporal grounding
> improved but remained low in absolute terms.

Data integrity:

> Before supervised adaptation, each video-QA sample was validated by confirming
> that its annotated time window could be cut from the referenced timestamp
> overlay video. Invalid samples were excluded from training/evaluation subsets
> and logged for reproducibility.

Clip-window audit result:

> The internal SFT split contained 7,198 training and 802 validation samples
> before video-window filtering. After validating that each annotated time window
> could be cut from its timestamp-overlay video, 5,959 training samples and 663
> validation samples remained. The excluded samples correspond to impossible
> video windows rather than model errors.

## Baseline Table Values

For a fuller table, see:

- `docs/comprehensive_data_comparison.md`
- `results/all_data_comparison_table.csv`
- `results/main_result_summary.csv`
- `results/lora_full_test_vs_overlay_baseline.csv`
- `results/evaluator_style_full_4000_summaries.csv`
- `results/evaluator_style_full_4000_summaries.txt`

| Setting | Overall | Pre-eval | temporal_grounding | time |
|---|---:|---:|---:|---:|
| Raw full 4000 | 0.194250 | 0.364083 | 0.007741 | 0.003199 |
| Overlay full 4000 | 0.207500 | 0.372647 | 0.033822 | 0.029623 |
| LoRA overlay 100 | 0.350000 | 0.328905 | 0.065217 | 0.065217 |
| LoRA overlay full 4000 | 0.279000 | 0.402794 | 0.071740 | 0.064236 |

## Full TEST LoRA Delta Values

| Metric | Overlay baseline | LoRA adapter | Delta |
|---|---:|---:|---:|
| Overall MEAN | 0.207500 | 0.279000 | +0.071500 |
| Pre-evaluation SCORE | 0.372647 | 0.402794 | +0.030147 |
| object_recognition | 0.308308 | 0.472173 | +0.163865 |
| object_identification | 0.149298 | 0.469223 | +0.319925 |
| fo_class | 0.175904 | 0.437977 | +0.262073 |
| temporal_grounding | 0.033822 | 0.071740 | +0.037918 |
| time | 0.029623 | 0.064236 | +0.034613 |

## SFT Data Integrity Table Values

| Split | Total | Clip-valid | Invalid | Invalid rate |
|---|---:|---:|---:|---:|
| Internal train | 7198 | 5959 | 1239 | 0.172131 |
| Internal val | 802 | 663 | 139 | 0.173317 |

## Next Paper-Relevant Analyses

- raw wrong / overlay correct examples
- overlay wrong / raw correct examples
- LoRA-SFT validation loss trend
- adapter full TEST error analysis
- category-level gains after training
- prompt-only vs LoRA-SFT ablation
- open-source VLM baseline comparison:
  MiniCPM-V, LLaVA-OneVision, InternVL, Gemma, MedGemma
- prompt and output-normalization ablation for fo_class questions:
  default open-ended prompting vs class-constrained prompting, preserving raw
  model outputs and normalized class predictions separately
- preliminary open-VLM smoke observation:
  class-constrained prompting improves output validity but not uniformly
  accuracy; prompt mode should be model-specific and validated on larger samples
  before reporting final baseline comparisons
- TEST-30 open VLM prompt ablation:
  class-constrained prompting improved overall score for every candidate on the
  first 30 TEST samples; LLaVA-OneVision and MedGemma reached `0.366667`,
  compared with best default score `0.200000` from MedGemma. This supports
  reporting prompt design as a controlled experimental variable.
- TEST-100 selected open VLM results:
  MedGemma reached overall `0.270000` and LLaVA-OneVision reached `0.260000`
  with class-constrained prompt and 4-frame input, exceeding the reproduced
  Qwen3-VL overlay-100 baseline `0.210000` but remaining below Qwen3-VL LoRA
  TEST-100 `0.350000`.
- TEST-100 frame ablation:
  MedGemma improves from `0.270000` to `0.290000` when increasing sampled frames
  from 4 to 8, mainly because temporal/time accuracy improves from `0.021739`
  to `0.130435`. LLaVA decreases overall from `0.260000` to `0.240000`, but
  `fo_class` improves from `0.391304` to `0.434783`.
- report table requirement:
  formal evaluation runs should include evaluator-style breakdown tables with
  `level`, `name`, `accuracy`, `ci_low`, `ci_high`, and `count`, matching the
  official evaluator terminal output style. See `report/README.md`.
