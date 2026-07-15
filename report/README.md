# Report Assets

This directory records tables and artifacts that should be prepared for the
paper/report, separate from runnable scripts and raw experiment logs.

## Required Result Table Style

For every formal evaluation run that may be discussed in the paper, prepare an
evaluator-style breakdown table with these columns:

```text
level  name  accuracy  ci_low  ci_high  count
```

This is the same style as the official evaluator terminal output, for example:

```text
       level                         name  accuracy   ci_low  ci_high  count
        leaf causal_consequence_reasoning  0.825079 0.659952 0.960000     33
        leaf          duration_estimation  0.108336 0.064553 0.163303    251
        leaf            event_aggregation  0.595000 0.433333 0.750083    121
       group                  aggregation  0.551013 0.459412 0.640335    329
answer_format                       binary 0.386725 0.299501 0.483935    388
     overall                         MEAN  0.324500 0.290000 0.369513   2000
```

## Usage In This Project

Current helper:

```bash
python scripts/print_evaluator_style_summary.py
```

For one experiment:

```bash
python scripts/print_evaluator_style_summary.py \
  --experiment qwen3vl_lora_full_4000
```

For future model comparison runs, keep both:

- `summary.csv`: machine-readable evaluator output
- terminal-style table text or screenshot-ready printed table

## Paper Notes

When writing the paper, use this table style for:

- official raw baseline
- official overlay baseline
- Qwen3-VL LoRA-SFT result
- open VLM prompt ablations once sample size is large enough
- final selected model full TEST result

Small smoke tests such as `num_eval=3` should be labeled as engineering smoke
tests, not performance claims.

Prompt ablations such as the open-VLM `num_eval=30` comparison can be used as
method-selection evidence, but final model performance claims should use larger
evaluations such as TEST-100 or full TEST.

Current open-VLM TEST-100 comparison table:

```text
results/open_vlm_test100_class_prompt_selected.csv
results/open_vlm_test100_frame_ablation.csv
```

Use it as a small-scale model-selection table. Full-paper performance claims
still require either larger TEST subsets or full TEST-4000.
