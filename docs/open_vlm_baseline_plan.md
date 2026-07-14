# 开源 VLM 横向 Baseline 计划

目标：在当前 `Qwen3-VL-4B + LoRA` 已经有效的基础上，寻找更强的开源 VLM baseline，为后续比赛提交和第二轮方法设计提供模型选择依据。

## 当前已验证基线

| 设置 | TEST 样本 | 输入 | overall | pre-eval | 备注 |
|---|---:|---|---:|---:|---|
| Qwen3-VL-4B raw baseline | 4000 | raw video | 0.194250 | 0.364083 | 原始视频 baseline |
| Qwen3-VL-4B overlay baseline | 4000 | timestamp overlay | 0.207500 | 0.372647 | 当前主要 zero-shot baseline |
| Qwen3-VL-4B + LoRA | 4000 | timestamp overlay | 0.279000 | 0.402794 | 第一版 full clip-valid LoRA-SFT |

## 第一批候选模型

| key | repo_id | 优先级 | 类型 | 预期价值 | 风险 |
|---|---|---|---|---|---|
| `minicpm_v_4_5` | `openbmb/MiniCPM-V-4_5` | A | video VLM | 高效视频理解，可能适合 SEGMENT | 需要适配 MiniCPM chat 接口 |
| `llava_onevision_7b` | `llava-hf/llava-onevision-qwen2-7b-ov-hf` | A | video VLM | 成熟 video/image VLM | processor 与 video 输入格式需单独适配 |
| `internvl3_5_8b` | `OpenGVLab/InternVL3_5-8B-Instruct` | A | image/video MLLM | 强开源 MLLM | 视频输入可能需采帧 |
| `gemma3_12b` | `google/gemma-3-12b-it` | B | image-text VLM | 强通用 VLM，可测多帧图像输入 | 不是原生视频；可能需 license acceptance |
| `medgemma_4b` | `google/medgemma-4b-it` | B | medical image-text VLM | 医学先验，可能提升手术对象识别 | 非原生视频；可能需 license acceptance |

候选清单文件：

```text
configs/vlm_candidate_models.csv
```

## 下载流程

远端服务器：

```bash
source ~/tools/miniconda3/etc/profile.d/conda.sh
conda activate orena-focus
cd ~/workspace/VLM-Competition

python scripts/download_vlm_candidates.py \
  --config configs/vlm_candidate_models.csv \
  --output-dir ~/workspace/vlm-models \
  --continue-on-error
```

只下载某一个：

```bash
python scripts/download_vlm_candidates.py \
  --model minicpm_v_4_5 \
  --output-dir ~/workspace/vlm-models
```

说明：

- 默认会校验并续传已经存在的目录。
- 只有明确想跳过非空目录时才使用 `--skip-existing`。
- 如果 Gemma / MedGemma 因 gated repo 报 403，可加
  `--continue-on-error` 让脚本继续下载后续模型。

先 dry-run：

```bash
python scripts/download_vlm_candidates.py --dry-run
```

## 分批测试策略

### Stage 1: smoke test

- 每个模型先跑 `num_eval=10` 或 `num_eval=30`。
- 目标是确认：
  - 模型能加载。
  - 视频或多帧输入能走通。
  - 输出不是空。
  - evaluator 能评分。

### Stage 2: TEST-100

- 每个 smoke 通过的模型跑 100 条官方 TEST。
- 对比：
  - `official-overlay-100`: overall `0.210000`
  - `Qwen3-VL + LoRA TEST-100`: overall `0.350000`

建议筛选规则：

- `< 0.25`：暂不继续。
- `0.25 - 0.35`：考虑 prompt/router 优化。
- `> 0.35`：优先跑 full 4000。

### Stage 3: full TEST-4000

- 只给最有希望的 1-2 个模型跑全量。
- 对比当前主结果：
  - overlay baseline overall `0.207500`
  - Qwen3-VL + LoRA overall `0.279000`

## 后续脚本方向

下一步需要逐个实现 model adapter：

- `qwen3vl`
- `minicpmv`
- `llava_onevision`
- `internvl`
- `gemma3`
- `medgemma`

目标是让所有模型最终都能输出：

```text
responses.jsonl
results.csv
summary.csv
run_config.json
```

并统一由官方 `Evaluator` 打分。
