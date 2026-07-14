# 脚本工作流说明

本文档用于说明 ORena FOCUS 项目至今我们写过、跑过、或在工作流中使用过的脚本。结构按项目推进顺序整理：每一步为什么做、脚本在干什么、输入输出是什么、我们实际得到过什么结果。

维护规则：后续只要新增脚本、修改脚本参数、改变实验运行方式，或跑出新的关键结果，都需要同步更新本文档，避免后续上下文压缩后丢失工作流细节。

## 0. 远端服务器与环境准备

### `scripts/setup_school_server.sh`

作用：

- 准备远端 Linux 工作目录。
- 克隆官方仓库 `IMSY-DKFZ/orena-focus`。
- 创建 Python 环境。
- 安装官方包和基础模型依赖。
- 检查 GPU 是否可见。

主要输入：

- `WORKSPACE_DIR`：远端工作目录。
- `OFFICIAL_REPO_DIR`：官方仓库目录。
- `FOCUS_ROOT_DIR`：FOCUS 数据根目录。
- `TORCH_INSTALL_CMD`：可选，如果需要指定 CUDA 版 PyTorch 安装命令。

主要输出：

- 远端官方仓库目录。
- Python 环境。
- 官方仓库 commit 记录。

本项目实际情况：

- 最终我们主要使用 conda 环境 `orena-focus`，而不是这个脚本默认的 venv。
- 学校服务器已验证为 `2 x RTX A5000`。

## 1. 数据下载与预处理

### `scripts/prepare_heico_data.py`

作用：

- 设置 FOCUS 数据根目录。
- 通过官方 `download("heico")` 下载 HeiCo 数据。
- 可选生成 timestamp overlay 视频。
- 可选抽取视频帧。

主要输入：

- `--root-dir`
- `--dataset heico`
- `--skip-overlay`
- `--skip-frames`
- `--overlay-frames`
- `--max-workers`

主要输出：

- 原始视频：`/home/Jiali_Wang/data/focus/heico/videos`
- overlay 视频：`/home/Jiali_Wang/data/focus/heico/overlayed`
- 可选的抽帧目录。

本项目实际情况：

- SEGMENT 推理和 LoRA-SFT 主要使用视频切片，不优先使用完整抽帧。
- timestamp overlay 使用官方 `VideoTimestampOverlayPreprocessor` 生成。
- overlay 不能只检查文件数量，还必须检查视频时长覆盖。之前虽然有 30 个 overlay 文件，但其中 4 个 Sigma 视频 overlay 被截断，后来重新生成。

## 2. 数据集可用性检查

### `scripts/check_focus_dataset.py`

作用：

- 检查官方 FOCUS 数据集能否正常加载。
- 打印样本数量和第一个样本的元信息。
- 可选通过 `FocusVideoDataset` 生成一个临时视频切片，确认视频切片流程可用。

主要输入：

- `--root-dir`
- `--dataset heico`
- `--split train|test|all`
- `--track segment`
- `--make-video-sample`
- `--no-overlay`
- `--video-stride`
- `--width`
- `--height`

主要输出：

- 数据集样本数。
- 第一个样本的 `qID`、`videoID`、时间窗、问题、答案格式、参考答案。
- 可选生成一个临时 `.mp4`，检查完后删除。

本项目结果：

- HeiCo SEGMENT 官方 TRAIN：`8000` 条 QA。
- HeiCo SEGMENT 官方 TEST：`4000` 条 QA。
- 原始视频数量：`30` 个。

## 3. 官方 baseline 推理与评估

### `scripts/run_segment_baseline.py`

作用：

- 在 ORena FOCUS SEGMENT track 上运行 Qwen3-VL 推理。
- 对每个 QA 样本使用 `FocusVideoDataset` 切出对应视频片段。
- 调用 `Qwen/Qwen3-VL-4B-Instruct` 生成回答。
- 保存模型回答。
- 调用官方 `Evaluator` 打分。
- 保存 `results.csv`、`summary.csv`、`responses.jsonl`、`run_config.json`。

主要输入：

- `--root-dir /home/Jiali_Wang/data/focus`
- `--dataset heico`
- `--split test`
- `--model-id Qwen/Qwen3-VL-4B-Instruct`
- `--adapter-dir`：可选，用于加载 LoRA adapter。
- `--device cuda:0`
- `--num-eval`：数字或 `none`。
- `--video-stride 25`
- `--width 640`
- `--height 360`
- `--no-overlay`：使用原始视频；不加这个参数则使用 overlay 视频。
- `--output-dir`

主要输出：

- `run_config.json`
- `responses.jsonl`
- `results.csv`
- `summary.csv`

几种使用方式：

- 加 `--no-overlay`：评估原始视频版本。
- 不加 `--no-overlay`：评估 timestamp overlay 视频版本。
- 不加 `--adapter-dir`：评估 base Qwen3-VL。
- 加 `--adapter-dir`：评估 base Qwen3-VL + LoRA adapter。

本项目结果：

- `official-smoke-3`：raw TEST，3 条，仅验证流程。
- `official-smoke-100`：raw TEST，100 条，overall `0.200000`。
- `official-overlay-100`：overlay TEST，100 条，overall `0.210000`。
- `official-raw-full-4000`：raw TEST，4000 条，overall `0.194250`。
- `official-overlay-full-4000`：overlay TEST，4000 条，overall `0.207500`。
- `qwen3vl-4b-sft-valid5959-e1-overlay-test-100`：LoRA adapter，overlay TEST，100 条，overall `0.350000`。

为什么重要：

- 这是我们最核心的评估脚本。
- baseline 和 LoRA adapter 都通过同一个官方 evaluator 打分，因此结果可对比。

## 4. 官方 TRAIN 审计与内部 train/val 切分

### `scripts/audit_and_split_segment_train.py`

作用：

- 加载官方 HeiCo SEGMENT TRAIN 和 TEST。
- 统计 primary category、answer format 等分布。
- 从官方 TRAIN 中确定性切分内部 train/val。
- 避免使用官方 TEST 训练，防止数据泄漏。
- 生成 LoRA-SFT 所需的 JSONL 数据。

主要输入：

- `--root-dir /home/Jiali_Wang/data/focus`
- `--dataset heico`
- `--track segment`
- `--val-fraction 0.1`
- `--seed 20260707`
- `--output-dir`

主要输出：

- `audit_summary.json`
- `split_counts.csv`
- `distribution_by_official_split_primary.csv`
- `distribution_by_official_split_answer_format.csv`
- `distribution_by_official_split_primary_answer_format.csv`
- `train_internal_split_manifest.csv`
- `sft_train_overlay.jsonl`
- `sft_val_overlay.jsonl`

本项目结果：

- 官方 TRAIN：`8000` 条。
- 官方 TEST：`4000` 条。
- 内部 train：`7198` 条。
- 内部 val：`802` 条。
- 随机种子：`20260707`。

为什么重要：

- 这是防止数据泄漏的关键步骤。
- 官方 TEST 只用于最终评估，不进入训练。

## 5. SFT 视频时间窗有效性审计

### `scripts/audit_sft_clip_windows.py`

作用：

- 检查每条 SFT 样本的 `start_time` / `end_time` 是否真的能从对应 overlay 视频中切出来。
- 输出 clean manifest 和 invalid manifest。
- 避免训练或验证中途因为时间窗越界崩溃。
- 保证训练数据清洗过程可复现。

主要输入：

- 一个或多个 `--input-jsonl`
- `--output-dir`

主要输出：

- `clip_window_audit_summary.json`
- `sft_train_overlay.clip_valid.jsonl`
- `sft_train_overlay.invalid_clips.jsonl`
- `sft_val_overlay.clip_valid.jsonl`
- `sft_val_overlay.invalid_clips.jsonl`

本项目结果：

- 内部 train：`7198` total，`5959` valid，`1239` invalid。
- 内部 val：`802` total，`663` valid，`139` invalid。
- clip-valid SFT 数据总数：`6622` 条。

为什么重要：

- 之前只检查路径存在是不够的。
- 有些 QA 的时间窗超过了视频真实长度。
- 后续训练只使用 clip-valid 样本。
- 这是数据完整性规则，不是调参或挑数据。

## 6. LoRA-SFT 训练

### `scripts/train_qwen3vl_lora_sft_smoke.py`

作用：

- 对 Qwen3-VL 进行 PEFT LoRA 监督微调。
- 根据每条样本的时间窗切出 overlay 视频片段。
- 把视频、问题、答案组织成 Qwen chat-style SFT 样本。
- mask prompt token，只让 assistant answer 参与主要 loss。
- 注入 LoRA 模块。
- 保存 adapter 和 processor。
- 计算 validation loss。

主要输入：

- `--model-id Qwen/Qwen3-VL-4B-Instruct`
- `--train-jsonl`
- `--val-jsonl`
- `--output-dir`
- `--max-train-samples`
- `--max-val-samples`
- `--epochs`
- `--learning-rate`
- `--gradient-accumulation-steps`
- `--video-stride`
- `--width`
- `--height`
- `--lora-r`
- `--lora-alpha`
- `--lora-dropout`
- `--target-modules`
- `--load-in-4bit`
- `--invalid-clip-policy skip|error`

主要输出：

- `run_config.json`
- `train_history.jsonl`
- `smoke_summary.json`
- `adapter-final/`
- 可选 `invalid_clips_train.jsonl`
- 可选 `invalid_clips_val.jsonl`

LoRA 设置：

- `r=8`
- `alpha=16`
- `dropout=0.05`
- target modules：
  `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- CUDA 上使用 bf16 base model。

本项目训练记录：

1. `qwen3vl-4b-smoke-32`
   - 32 train，8 val。
   - 8 optimizer steps。
   - eval loss `1.0017873756587505`。
   - 目的：只验证训练链路是否跑通。

2. `qwen3vl-4b-smoke-512-filtered`
   - 请求 512 train，128 val。
   - 实际 512 train，99 val。
   - 29 条 val 样本时间窗无效，被过滤。
   - 128 optimizer steps。
   - eval loss `0.35957938603553546`。
   - 目的：中等规模 sanity check。

3. `qwen3vl-4b-sft-valid5959-e1`
   - 5959 train，663 val。
   - 1 epoch。
   - 1490 optimizer steps。
   - eval loss `0.42800752680308324`。
   - 训练耗时约 10.27 小时。
   - 目的：第一个 full clip-valid LoRA adapter。

为什么重要：

- 这是我们当前方法开发的核心训练脚本。
- 它把官方 TRAIN split 转换为 supervised adaptation 实验。
- 训练出来的 adapter 后续通过 `run_segment_baseline.py --adapter-dir` 接入官方 TEST evaluator。

## 6.5 Evaluator 风格结果打印

### `scripts/print_evaluator_style_summary.py`

作用：

- 读取 `results/evaluator_style_full_4000_summaries.csv`。
- 按官方 evaluator 终端输出的风格打印：
  `level / name / accuracy / ci_low / ci_high / count`。
- 支持一次打印全部实验，也支持只打印某一个实验。
- 支持输出到 `.txt` 文件。

主要输入：

- `--input`
- `--experiment`
- `--precision`
- `--output`
- `--no-separators`

示例：

```bash
python scripts/print_evaluator_style_summary.py \
  --experiment qwen3vl_lora_full_4000
```

输出文件示例：

```bash
python scripts/print_evaluator_style_summary.py \
  --experiment qwen3vl_lora_full_4000 \
  --output results/qwen3vl_lora_full_4000_summary.txt
```

为什么重要：

- 它把我们整理好的 CSV 结果变成和官方 evaluator 截图接近的表格。
- 后续写论文或做展示时，可以快速生成某个实验的完整类别结果表。

## 7. 调试阶段用过的手动命令

有些关键步骤不是单独脚本，而是用临时 shell / Python 片段完成。

### 环境检查

作用：

- 确认远端服务器身份、GPU、Python、磁盘。

代表命令：

- `hostname`
- `whoami`
- `pwd`
- `nvidia-smi`
- `python3 --version`
- `df -h`
- `torch.cuda.is_available()`

### conda 环境准备

作用：

- 安装 Miniconda。
- 创建 `orena-focus`。
- 在每个新 terminal 激活环境。

关键命令：

```bash
source ~/tools/miniconda3/etc/profile.d/conda.sh
conda activate orena-focus
```

### Hugging Face 登录与数据下载

作用：

- 登录 Hugging Face。
- 下载 HeiCo 数据集。

代表命令：

- `hf auth login`
- 通过官方 FOCUS API 执行 `download("heico")`

### overlay 完整性排查

作用：

- 解释为什么 full overlay inference 在约 372/4000 附近崩溃。
- 找出被截断的 overlay 视频。
- 重新生成坏的 overlay。

重要结果：

发现并重新生成了 4 个坏 overlay：

- `0020 - Heico - Sigma - 1.avi`
- `0021 - Heico - Sigma - 2.avi`
- `0027 - Heico - Sigma - 8.avi`
- `0028 - Heico - Sigma - 9.avi`

## 8. 当前脚本依赖关系

```text
setup_school_server.sh
        |
prepare_heico_data.py
        |
check_focus_dataset.py
        |
run_segment_baseline.py  -> baseline TEST 结果
        |
audit_and_split_segment_train.py
        |
audit_sft_clip_windows.py
        |
train_qwen3vl_lora_sft_smoke.py
        |
run_segment_baseline.py --adapter-dir  -> LoRA TEST 结果
        |
download_vlm_candidates.py -> 开源 VLM 候选模型下载
        |
check_vlm_downloads.py -> 候选模型本地快照完整性检查
        |
run_open_vlm_smoke.py -> 五个开源 VLM 的多帧 smoke / 小批量测试
```

## 8.5 开源 VLM 候选模型下载

### `scripts/download_vlm_candidates.py`

作用：

- 读取 `configs/vlm_candidate_models.csv` 中的候选模型。
- 使用 Hugging Face `snapshot_download` 下载模型到远端服务器。
- 为每个模型写入下载 manifest。
- 支持 dry-run、只下载单个模型、强制刷新。

主要输入：

- `--config`
- `--output-dir`
- `--model`
- `--revision`
- `--dry-run`
- `--skip-existing`
- `--continue-on-error`
- `--disable-xet`
- `--manifest`

当前第一批候选：

- `openbmb/MiniCPM-V-4_5`
- `llava-hf/llava-onevision-qwen2-7b-ov-hf`
- `OpenGVLab/InternVL3_5-8B-Instruct`
- `google/gemma-3-12b-it`
- `google/medgemma-4b-it`

注意：

- Gemma / MedGemma 可能需要先在 Hugging Face 页面接受 license。
- 如果下载被中断，默认重新运行脚本会校验/续传已有目录；不要因为目录非空就直接认定完成。
- `--continue-on-error` 可用于跳过 gated repo 报错，继续下载后续候选。
- `--disable-xet` 可用于绕过 `hf-xet` 层的下载异常，例如 Gemma 下载时出现
  `Unable to parse string as hex hash value`。
- 下载完成后还不能直接代表可评估，需要继续写或接入对应 model adapter。

### `scripts/check_vlm_downloads.py`

作用：

- 检查 `~/workspace/vlm-models` 下每个候选模型目录是否存在。
- 统计目录大小、文件数、权重文件数。
- 检查 `config.json` 是否存在。
- 读取 `download_manifest.json` 中的下载状态。
- 不加载模型权重，因此可作为批量测试前的低风险检查。

远端命令：

```bash
python scripts/check_vlm_downloads.py \
  --model-dir ~/workspace/vlm-models \
  --json-output ~/workspace/focus-runs/open-vlm-download-check.json
```

如果全部为 `ok`，再进入模型加载 / inference smoke test。

### `scripts/run_open_vlm_smoke.py`

作用：

- 从官方 FOCUS SEGMENT TEST 构造视频片段。
- 每个片段抽取少量 RGB 帧，作为跨模型通用输入。
- 对候选模型逐个加载并生成答案。
- 保存 `responses.jsonl`、`run_config.json`、`status.json`。
- 默认调用官方 `Evaluator` 生成 `results.csv` 和 `summary.csv`。
- 每个模型推理完后释放模型显存，再加载 evaluator，降低 OOM 风险。

第一版支持的 engine：

- `minicpmv`
- `llava_onevision`
- `internvl`
- `gemma3`
- `medgemma`

远端 smoke 命令：

```bash
python scripts/run_open_vlm_smoke.py \
  --model minicpm_v_4_5 \
  --model llava_onevision_7b \
  --model internvl3_5_8b \
  --model gemma3_12b \
  --model medgemma_4b \
  --model-dir ~/workspace/vlm-models \
  --root-dir /home/Jiali_Wang/data/focus \
  --num-eval 3 \
  --frames-per-clip 4 \
  --output-dir ~/workspace/focus-runs/open-vlm-smoke/test3 \
  --continue-on-error
```

注意：

- 这是共同多帧输入的第一版 smoke test，不是最终公平最优适配。
- MiniCPM / InternVL 后续可以继续升级为更贴近官方说明的视频接口。
- Gemma / MedGemma 本身是 image-text 路线，因此多帧抽样是合理的第一版。

## 9. 当前下一步

当前建议下载第一批开源 VLM 候选模型：

```bash
source ~/tools/miniconda3/etc/profile.d/conda.sh
conda activate orena-focus
cd ~/workspace/VLM-Competition

python scripts/download_vlm_candidates.py \
  --config configs/vlm_candidate_models.csv \
  --output-dir ~/workspace/vlm-models \
  --continue-on-error
```

下载完成后进入 smoke / TEST-100 批量测试阶段。
