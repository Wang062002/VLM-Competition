# Evaluator 风格结果表

这个文件说明 `results/evaluator_style_full_4000_summaries.csv` 的用途。

该 CSV 按照官方 evaluator 输出截图中的字段整理：

```text
level, name, accuracy, ci_low, ci_high, count
```

为了同时容纳多个实验，额外增加了第一列：

```text
experiment
```

## 覆盖的实验

| experiment | 说明 |
|---|---|
| `official_raw_full_4000` | raw video baseline，官方 TEST 4000 条 |
| `official_overlay_full_4000` | timestamp overlay baseline，官方 TEST 4000 条 |
| `qwen3vl_lora_full_4000` | Qwen3-VL + LoRA adapter，timestamp overlay，官方 TEST 4000 条 |

## 文件路径

```text
results/evaluator_style_full_4000_summaries.csv
```

## 字段解释

| 字段 | 含义 |
|---|---|
| `experiment` | 实验名称 |
| `level` | 指标层级，例如 `leaf`、`group`、`answer_format`、`overall`、`pre_evaluation` |
| `name` | 具体类别或指标名称 |
| `accuracy` | 正确率 |
| `ci_low` | 置信区间下界 |
| `ci_high` | 置信区间上界 |
| `count` | 该类别样本数 |

`pre_evaluation SCORE` 没有置信区间，因此 `ci_low` 和 `ci_high` 留空。

## 打印成截图样式

在本地或远端项目目录下运行：

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("results/evaluator_style_full_4000_summaries.csv")

for experiment, part in df.groupby("experiment", sort=False):
    print()
    print("=" * 100)
    print(experiment)
    print("=" * 100)
    print(
        part[["level", "name", "accuracy", "ci_low", "ci_high", "count"]]
        .to_string(index=False)
    )
PY
```

输出格式会接近官方 evaluator 的终端表格：

```text
       level                         name  accuracy   ci_low  ci_high  count
        leaf causal_consequence_reasoning  0.877218 0.755624 0.966699     62
        ...
     overall                         MEAN  0.279000 0.252731 0.306012   4000
pre_evaluation                        SCORE 0.402794      NaN      NaN      5
```

## 当前最重要对比

full TEST 4000 的主结果：

| Setting | Overall | Pre-eval |
|---|---:|---:|
| Raw baseline | 0.194250 | 0.364083 |
| Overlay baseline | 0.207500 | 0.372647 |
| LoRA-SFT | 0.279000 | 0.402794 |

LoRA-SFT 相比 overlay baseline：

| Metric | Overlay | LoRA | Delta |
|---|---:|---:|---:|
| Overall | 0.207500 | 0.279000 | +0.071500 |
| Pre-eval | 0.372647 | 0.402794 | +0.030147 |
| object_identification | 0.149298 | 0.469223 | +0.319925 |
| fo_class | 0.175904 | 0.437977 | +0.262073 |
| object_recognition | 0.308308 | 0.472173 | +0.163865 |
| temporal_grounding | 0.033822 | 0.071740 | +0.037918 |
| time | 0.029623 | 0.064236 | +0.034613 |
