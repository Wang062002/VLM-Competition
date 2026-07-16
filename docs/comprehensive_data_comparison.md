# 项目数据与实验结果总览对比

本文档汇总 ORena FOCUS 项目当前已经拥有和产生的主要数据，包括数据集规模、数据清洗结果、训练运行、baseline 评估、LoRA-SFT 评估以及关键指标对比。

如果需要官方 evaluator 截图风格的 `level / name / accuracy / ci_low / ci_high / count` 表格，见：

- `results/evaluator_style_full_4000_summaries.csv`
- `results/evaluator_style_full_4000_summaries.txt`
- `docs/evaluator_style_summary_tables.md`

## 1. 数据资源总览

| 数据项 | 状态 | 本地/远端路径 | 规模或数量 | 用途 | 备注 |
|---|---|---|---:|---|---|
| HeiCo 原始视频 | 已下载 | `/home/Jiali_Wang/data/focus/heico/videos` | 约 150G，30 个视频 | raw-video baseline、overlay 生成 | 当前主要数据源 |
| HeiCo timestamp overlay | 已生成 | `/home/Jiali_Wang/data/focus/heico/overlayed` | 约 81G，30 个视频 | overlay baseline、LoRA-SFT 输入 | 使用官方 `VideoTimestampOverlayPreprocessor` |
| LapChole-FOCUS | 未下载 | 无 | 无 | 暂未使用 | 官方支持但当前未纳入实验 |
| SEGMENT 官方 TRAIN | 已加载 | Hugging Face / FOCUS API | 8000 QA | LoRA-SFT 数据来源 | 不直接用作最终 TEST |
| SEGMENT 官方 TEST | 已加载 | Hugging Face / FOCUS API | 4000 QA | held-out 评估 | 严格不用于训练 |

## 2. SFT 数据切分与有效性

| 数据阶段 | split | 原始样本数 | clip-valid 样本数 | 无效样本数 | 无效率 | 用途 |
|---|---|---:|---:|---:|---:|---|
| 官方数据 | TRAIN | 8000 | 未直接使用 | 未直接使用 | 未直接使用 | 原始训练池 |
| 官方数据 | TEST | 4000 | 不适用 | 不适用 | 不适用 | held-out 评估 |
| 内部切分 | train | 7198 | 5959 | 1239 | 0.172131 | LoRA-SFT 训练 |
| 内部切分 | val | 802 | 663 | 139 | 0.173317 | LoRA-SFT 验证 |
| 内部 clean SFT | train + val | 8000 | 6622 | 1378 | 0.172250 | 当前可用于 video-SFT 的总数据 |

说明：

- 内部 train/val 从官方 TRAIN 切分，随机种子为 `20260707`。
- 官方 TEST 始终保持 held-out。
- `clip-valid` 表示该 QA 样本的时间窗可以从对应 timestamp-overlay 视频中真实切出。
- 无效样本是时间窗超出视频长度等数据完整性问题，不是模型错误。

## 3. LoRA-SFT 训练运行对比

| run | 日期 | 训练样本 | 验证样本 | epoch | 梯度累积 | optimizer steps | eval loss | 作用 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `qwen3vl-4b-smoke-32` | 2026-07-09 | 32 | 8 | 1 | 4 | 8 | 1.001787 | 训练链路 smoke test |
| `qwen3vl-4b-smoke-512-filtered` | 2026-07-09 | 512 | 99 | 1 | 4 | 128 | 0.359579 | 中等规模 sanity check |
| `qwen3vl-4b-sft-valid5959-e1` | 2026-07-10 | 5959 | 663 | 1 | 4 | 1490 | 0.428008 | 第一版 full clip-valid LoRA adapter |

full LoRA 训练补充信息：

- base model：`Qwen/Qwen3-VL-4B-Instruct`
- LoRA：`r=8`，`alpha=16`，`dropout=0.05`
- target modules：`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- 训练耗时约 `10.27` 小时
- adapter 路径：`/home/Jiali_Wang/workspace/focus-runs/lora-sft/qwen3vl-4b-sft-valid5959-e1/adapter-final`

## 4. 全部评估运行总览

| experiment | split | 样本数 | 输入 | 模型 | overall | pre-eval | 结论 |
|---|---|---:|---|---|---:|---:|---|
| `official-smoke-3` | TEST | 3 | raw video | Qwen3-VL | 0.000000 | 0.000000 | 仅验证推理链路 |
| `official-smoke-100` | TEST | 100 | raw video | Qwen3-VL | 0.200000 | 0.186795 | 第一版小样本 raw baseline |
| `official-overlay-100` | TEST | 100 | timestamp overlay | Qwen3-VL | 0.210000 | 0.200886 | 小样本 overlay baseline |
| `official-raw-full-4000` | TEST | 4000 | raw video | Qwen3-VL | 0.194250 | 0.364083 | full raw baseline |
| `official-overlay-full-4000` | TEST | 4000 | timestamp overlay | Qwen3-VL | 0.207500 | 0.372647 | full overlay baseline |
| `open-vlm-llava-onevision-4frames-full-4000` | TEST | 4000 | timestamp overlay / 4 frames / class prompt | LLaVA-OneVision-7B | 0.155500 | 0.249757 | LLaVA 未训练 full baseline，0 failures |
| `open-vlm-medgemma-8frames-full-4000` | TEST | 4000 | timestamp overlay / 8 frames / class prompt | MedGemma-4B | 0.188250 | 0.281741 | MedGemma 未训练 full baseline，0 failures |
| `qwen3vl-4b-sft-valid5959-e1-overlay-test-100` | TEST | 100 | timestamp overlay | Qwen3-VL + LoRA | 0.350000 | 0.328905 | LoRA 小样本 TEST 明显提升 |
| `qwen3vl-4b-sft-valid5959-e1-overlay-test-full` | TEST | 4000 | timestamp overlay | Qwen3-VL + LoRA | 0.279000 | 0.402794 | LoRA full TEST 确认提升 |

## 5. full 4000 主结果对比

| setting | 输入 | 模型 | overall | pre-eval | temporal_grounding | time | object_recognition | object_identification | fo_class |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Raw baseline | raw video | Qwen3-VL | 0.194250 | 0.364083 | 0.007741 | 0.003199 | 0.302873 | 0.153572 | 0.179436 |
| Overlay baseline | timestamp overlay | Qwen3-VL | 0.207500 | 0.372647 | 0.033822 | 0.029623 | 0.308308 | 0.149298 | 0.175904 |
| LLaVA prompt-only baseline | timestamp overlay / 4 frames / class prompt | LLaVA-OneVision-7B | 0.155500 | 0.249757 | 0.015283 | 0.006235 | 0.331625 | 0.279323 | 0.230758 |
| MedGemma prompt-only baseline | timestamp overlay / 8 frames / class prompt | MedGemma-4B | 0.188250 | 0.281741 | 0.051561 | 0.049098 | 0.359666 | 0.227830 | 0.186662 |
| LoRA-SFT | timestamp overlay | Qwen3-VL + LoRA | 0.279000 | 0.402794 | 0.071740 | 0.064236 | 0.472173 | 0.469223 | 0.437977 |

## 6. LoRA-SFT 相对 overlay baseline 的 full TEST 提升

| 指标 | Overlay baseline | LoRA-SFT | Delta |
|---|---:|---:|---:|
| overall MEAN | 0.207500 | 0.279000 | +0.071500 |
| pre-evaluation SCORE | 0.372647 | 0.402794 | +0.030147 |
| object_recognition | 0.308308 | 0.472173 | +0.163865 |
| object_identification | 0.149298 | 0.469223 | +0.319925 |
| fo_class | 0.175904 | 0.437977 | +0.262073 |
| temporal_grounding | 0.033822 | 0.071740 | +0.037918 |
| time | 0.029623 | 0.064236 | +0.034613 |
| binary | 0.551885 | 0.588102 | +0.036217 |
| multiple_choice | 0.416370 | 0.381838 | -0.034532 |
| open_ended | 0.768312 | 0.766402 | -0.001910 |
| number | 0.251833 | 0.276006 | +0.024173 |
| temporal_localization | 0.027867 | 0.065457 | +0.037590 |

## 7. 主要结论

1. Timestamp overlay 相比 raw video 有小幅提升，尤其让 temporal/time 类问题从接近零提升到非零，但绝对值仍然低。
2. 第一版 full clip-valid LoRA-SFT 在 full held-out TEST 上确认优于 overlay baseline。
3. LoRA-SFT 最大收益来自：
   - `object_identification`
   - `fo_class`
   - `object_recognition`
4. temporal grounding 虽然提升，但仍然是最主要瓶颈。
5. `multiple_choice` 在 LoRA-SFT 后略有下降，需要后续做误差分析。
6. MedGemma-4B 的 full prompt-only baseline 低于 Qwen3-VL overlay baseline，
   说明 TEST-100 结果存在样本偏差；但 MedGemma 在 `object_recognition`、
   `object_identification`、`fo_class` 上强于 Qwen overlay，因此仍适合作为下一
   个 LoRA/SFT 训练底座。
7. LLaVA-OneVision 的 full prompt-only baseline overall `0.155500` 低于
   MedGemma，但它在 `object_identification` 与 `fo_class` 上更强，因此可以作为
   第二训练候选或类别识别专项路线。
8. 当前最值得推进的下一步不是继续盲目训练，而是把 MedGemma 的训练前 baseline
   与后续 LoRA/SFT 结果进行严格 before/after 对比，同时保留 full TEST error
   analysis：
   - baseline wrong / LoRA correct
   - baseline correct / LoRA wrong
   - temporal grounding failure cases
   - answer-format-specific failure cases
