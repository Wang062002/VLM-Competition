# 辅助 Agent 工作交接文档：ORena FOCUS SEGMENT 项目

更新日期：2026-08-06

本文档用于让新的辅助 agent 快速接手当前项目。请先完整阅读本文，再读取
`knowledge_base/START_HERE.md`、`knowledge_base/project_state.md`、
`knowledge_base/experiments.md`、`knowledge_base/workflows.md` 和
`docs/official_submission_qwen_lora.md`。

## 1. 项目一句话概述

本项目参加 ORena FOCUS surgical video QA challenge，目前聚焦 `SEGMENT`
track。核心任务是在官方 HeiCo 数据集上回答关于腹腔镜手术视频中 foreign
objects 的视觉问答问题。当前主线是以 `Qwen/Qwen3-VL-4B-Instruct` 为基础模型，
通过官方训练集做 LoRA-SFT，提高官方 TEST split 上的 evaluator 分数，并准备
符合官方 Docker submission template 的可提交方案。

## 2. 用户目标和协作偏好

用户的阶段性目标：

1. 跑通官方代码和官方数据验证流程。
2. 复现官方 baseline，并记录 raw / overlay 等设置对结果的影响。
3. 用官方 TRAIN split 做 LoRA-SFT 微调，提升官方 TEST split 表现。
4. 对比若干开源 VLM baseline，判断是否值得转向其他模型。
5. 将最佳模型打包成官方 submission，提交到官方通道测试分数。
6. 将项目过程沉淀为论文材料，包括实验设计、数据表格、踩坑记录和结论。

用户偏好的工作方式：

- 回复使用中文。
- 远端服务器操作时，尽量每次只给一条命令；用户执行后会贴回输出，再分析并给下一条。
- 用户希望边推进实验边学习方法，因此需要解释每一步“在做什么、为什么做、怎么看结果”。
- 每次有实质性进展，应该更新知识库、论文材料、workflow 和 GitHub 版本。
- 不要在项目文件里记录密码、Hugging Face token 或其他密钥。
- 用户允许推送到 GitHub `main` 分支。

## 3. 代码仓库与远端路径

本地 Windows 项目仓库：

```text
C:\Users\28101\Documents\VLM-Competition
```

GitHub 仓库：

```text
https://github.com/Wang062002/VLM-Competition
branch: main
```

学校服务器：

```text
Host/IP: 10.176.61.126
Username: Jiali_Wang
Hostname: UNNC-CVIP-03
OS: Ubuntu 20.04.5 LTS
```

重要远端路径：

```text
/home/Jiali_Wang/workspace/VLM-Competition
/home/Jiali_Wang/workspace/orena-focus
/home/Jiali_Wang/workspace/orena-focus-submission-template
/home/Jiali_Wang/workspace/orena-focus-submission-template/segment-algorithm
/home/Jiali_Wang/workspace/focus-runs
/home/Jiali_Wang/data/focus
/mnt/data/jiali_wang
```

解释：

- `~/workspace/VLM-Competition` 是本项目在服务器上的同步副本，通过 `git pull origin main` 更新。
- `~/workspace/orena-focus` 是官方 ORena FOCUS 仓库。
- `~/workspace/orena-focus-submission-template` 是官方 submission template。
- `~/workspace/focus-runs` 保存实验输出、训练 adapter、summary 等大文件，不应提交到 GitHub。
- `~/data/focus` 是历史数据根目录，包含 HeiCo videos 和 overlayed videos。
- `/mnt/data/jiali_wang` 是新挂载大盘上用户自己的目录，后续大数据、cache、模型快照优先放这里。

## 4. 远端环境

每次打开远端 terminal，先激活 conda 环境：

```bash
source ~/tools/miniconda3/etc/profile.d/conda.sh
conda activate orena-focus
```

已知环境：

```text
Conda: /home/Jiali_Wang/tools/miniconda3
Env: orena-focus
Python: 3.10.20
PyTorch: 2.7.1+cu118
transformers: 5.13.0
peft: 0.19.1
qwen-vl-utils: 0.0.14
orena-focus: 0.3.2
decord: 0.6.0
accelerate: 1.14.0
safetensors: 0.8.0
```

GPU 信息：

```text
2 x NVIDIA RTX A5000, 24GB each
Driver observed: 470.256.02
Driver-reported CUDA: 11.4
```

使用 GPU 的规则：

- 服务器有两张卡，但用户需要保证同时只使用一张卡。
- 当前约定使用：

```bash
export CUDA_VISIBLE_DEVICES=0
```

验证：

```bash
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
print("visible_gpu_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("visible_gpu_name:", torch.cuda.get_device_name(0))
PY
```

如果没有激活 `orena-focus`，会出现 `ModuleNotFoundError: No module named 'torch'`，
这通常不是 PyTorch 丢失，而是当前 terminal 还在 `base` 环境。

## 5. 数据集状态

当前使用：

```text
Dataset: heico
Track: SEGMENT
Official TRAIN: 8000 QA
Official TEST: 4000 QA
Source videos: 30
```

重要原则：

- 官方 TEST split 是 held-out 测试集，不能用于训练。
- 官方 TRAIN split 是 LoRA-SFT 数据来源。
- 当前论文和实验比较均以官方 TEST split 的 evaluator 结果为准。

数据位置：

```text
/home/Jiali_Wang/data/focus/heico/videos
/home/Jiali_Wang/data/focus/heico/overlayed
```

overlay 状态：

- 已经生成 30 个 timestamp overlay 视频。
- 曾经有 4 个 Sigma overlay 文件截断，导致 full overlay inference 崩溃。
- 已修复的问题视频：
  - `0020 - Heico - Sigma - 1.avi`
  - `0021 - Heico - Sigma - 2.avi`
  - `0027 - Heico - Sigma - 8.avi`
  - `0028 - Heico - Sigma - 9.avi`

重要踩坑：

- 不能只检查 overlay 文件数量是否为 30。
- 必须检查 QA 的 `start_time/end_time` 是否能被对应 overlay 视频覆盖。
- 后续训练前应使用 clip-window audit 生成 clip-valid jsonl。

## 6. 已完成的主要阶段

### 6.1 官方代码和数据跑通

完成内容：

- VS Code Remote-SSH 连接学校服务器。
- 安装 Miniconda 和 `orena-focus` 环境。
- 克隆官方仓库 `IMSY-DKFZ/orena-focus`。
- 下载 HeiCo 数据。
- 登录 Hugging Face 并下载 Qwen 模型。
- 跑通官方 example inference 和 official evaluator。

关键经验：

- 学校服务器需要学校 VPN。
- 用户本地 Codex 需要本地代理/VPN，两者可能冲突。
- 如果 SSH timeout，先检查 VPN 和网络路由。
- VS Code Remote-SSH 第一次安装 server 会比较慢，等待即可。

### 6.2 复现官方 baseline

基础模型：

```text
Qwen/Qwen3-VL-4B-Instruct
```

评估器：

```text
Official Evaluator
Judge model: Qwen/Qwen3.5-4B
```

关键修复：

官方示例或初始脚本中 `device_map="auto"` 在学校服务器上可能把模型放到 CPU，
导致 GPU 显存占用异常小、推理极慢。已改为：

```python
self.model = Qwen3VLForConditionalGeneration.from_pretrained(
    self.model_id,
    torch_dtype=torch.bfloat16,
).to(self.device).eval()
```

已完成 baseline：

| Run | Split | Samples | Overlay | Overall MEAN | Pre-eval SCORE |
|---|---:|---:|---|---:|---:|
| official-smoke-100 | TEST | 100 | no | 0.200000 | 0.186795 |
| official-overlay-100 | TEST | 100 | yes | 0.210000 | 0.200886 |
| official-raw-full-4000 | TEST | 4000 | no | 0.194250 | 0.364083 |
| official-overlay-full-4000 | TEST | 4000 | yes | 0.207500 | 0.372647 |

结论：

- Overlay 对 temporal grounding 有帮助，但整体提升有限。
- 原始 Qwen 对医学外物类别和时间定位仍然弱。
- 后续 SFT 的主要收益空间是 object recognition / object identification / fo_class 和 answer-format control。

### 6.3 训练数据审计与内部 train/val split

从官方 TRAIN 8000 QA 中按固定 seed 生成内部训练/验证：

```text
Seed: 20260707
Internal train: 7198
Internal val: 802
```

路径：

```text
/home/Jiali_Wang/workspace/focus-runs/data-audit/segment-trainval-seed20260707/sft_train_overlay.jsonl
/home/Jiali_Wang/workspace/focus-runs/data-audit/segment-trainval-seed20260707/sft_val_overlay.jsonl
```

进一步做 clip-window audit 后：

```text
Clip-valid train: 5959
Clip-valid val: 663
Invalid train: 1239
Invalid val: 139
```

可用于训练的 clean manifest：

```text
/home/Jiali_Wang/workspace/focus-runs/data-audit/clip-window-audit-seed20260707/sft_train_overlay.clip_valid.jsonl
/home/Jiali_Wang/workspace/focus-runs/data-audit/clip-window-audit-seed20260707/sft_val_overlay.clip_valid.jsonl
```

重要结论：

- 最早理解成“可用数据只有 6969 条”是不准确的简单相加口误。
- clip-valid 总数是 `5959 + 663 = 6622`。
- 原始 internal 总数是 `7198 + 802 = 8000`。
- invalid rows 不能直接用于 video-SFT，因为对应时间窗无法从视频里截出有效 clip。

### 6.4 Qwen LoRA-SFT 训练

训练脚本主入口：

```text
scripts/train_qwen3vl_lora_sft_smoke.py
```

32-sample smoke：

```text
Run: qwen3vl-4b-smoke-32
Train: 32
Val: 8
Optimizer steps: 8
Eval loss: 1.0017873756587505
```

512-sample filtered：

```text
Run: qwen3vl-4b-smoke-512-filtered
Train: 512
Val effective: 99
Invalid val rows: 29
Optimizer steps: 128
Eval loss: 0.35957938603553546
```

Full clip-valid LoRA-SFT：

```text
Run: qwen3vl-4b-sft-valid5959-e1
Train: 5959
Val: 663
Epochs: 1
Gradient accumulation: 4
Optimizer steps: 1490
Eval loss: 0.42800752680308324
Training time: about 10.27 hours
Adapter: /home/Jiali_Wang/workspace/focus-runs/lora-sft/qwen3vl-4b-sft-valid5959-e1/adapter-final
```

Full LoRA adapter on official TEST：

```text
Run: qwen3vl-4b-sft-valid5959-e1-overlay-test-full
Split: official TEST
Samples: 4000
Input: timestamp overlay
Overall MEAN: 0.279000
Pre-eval SCORE: 0.402794
```

与 Qwen overlay baseline 对比：

| Metric | Qwen overlay baseline | Qwen LoRA | Delta |
|---|---:|---:|---:|
| overall MEAN | 0.207500 | 0.279000 | +0.071500 |
| pre-evaluation SCORE | 0.372647 | 0.402794 | +0.030147 |
| object_recognition | 0.308308 | 0.472173 | +0.163865 |
| object_identification | 0.149298 | 0.469223 | +0.319925 |
| fo_class | 0.175904 | 0.437977 | +0.262073 |
| temporal_grounding | 0.033822 | 0.071740 | +0.037918 |
| time | 0.029623 | 0.064236 | +0.034613 |

结论：

- Full TEST 证明 LoRA-SFT 有真实收益。
- 最大收益来自 foreign object 识别和类别回答。
- temporal grounding 有提升但仍然很弱，是下一阶段优化重点。
- multiple-choice 曾出现轻微下降，需要后续做分类错误分析。

### 6.5 开源 VLM baseline 对比

曾下载并测试：

```text
MiniCPM-V-4_5
LLaVA-OneVision-7B
InternVL3.5-8B
Gemma-3-12B
MedGemma-4B
```

下载目录曾为：

```text
/home/Jiali_Wang/workspace/vlm-models
```

由于这些模型整体不如 Qwen 主线，且占用约 80G，已删除 `~/workspace/vlm-models`。
实验结果已保存到 `results/*.csv` 和 `docs/research_log.md`。

TEST-100 class-constrained, 4 frames：

| Model | Overall | Pre-eval |
|---|---:|---:|
| MedGemma-4B | 0.270000 | 0.251610 |
| LLaVA-OneVision-7B | 0.260000 | 0.242351 |
| MiniCPM-V-4_5 | 0.150000 | 0.138889 |
| InternVL3.5-8B | 0.150000 | 0.140499 |

Frame ablation：

| Model | Frames | Overall |
|---|---:|---:|
| LLaVA-OneVision-7B | 4 | 0.260000 |
| LLaVA-OneVision-7B | 8 | 0.240000 |
| MedGemma-4B | 4 | 0.270000 |
| MedGemma-4B | 8 | 0.290000 |

Full TEST-4000 prompt-only：

| Model | Setting | Overall | Pre-eval |
|---|---|---:|---:|
| MedGemma-4B | overlay, 8 frames, class prompt | 0.188250 | 0.281741 |
| LLaVA-OneVision-7B | overlay, 4 frames, class prompt | 0.155500 | 0.249757 |

结论：

- MedGemma 和 LLaVA 在小样本上看起来接近或强于 Qwen baseline，但 full TEST 后整体不如 Qwen。
- MedGemma 医学取向有研究价值，但不是当前最强主线。
- LLaVA 在 `object_identification` / `fo_class` 某些项上有特色，可作为 specialist 或 ensemble 思路参考。
- 当前主线仍是 Qwen3-VL LoRA/SFT。

## 7. 官方 submission 当前状态

目标：

将已经训练好的 Qwen3-VL LoRA adapter 通过官方 SEGMENT Docker submission template
提交到官方评测通道。

官方 template 路径：

```text
/home/Jiali_Wang/workspace/orena-focus-submission-template/segment-algorithm
```

本项目中准备好的 submission 文件：

```text
submission/segment_qwen_lora/inference.py
submission/segment_qwen_lora/requirements.txt
submission/segment_qwen_lora/README.md
docs/official_submission_qwen_lora.md
```

已复制到官方 template 的模型资源：

```text
segment-algorithm/resources/qwen3vl-4b       # about 8.3G
segment-algorithm/resources/qwen3vl-lora     # about 74M
```

Qwen base model 原始 cache：

```text
/home/Jiali_Wang/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17
```

LoRA adapter 原始路径：

```text
/home/Jiali_Wang/workspace/focus-runs/lora-sft/qwen3vl-4b-sft-valid5959-e1/adapter-final
```

注意：

- Hugging Face snapshot 是 symlink-based，复制进 Docker resources 时要用 `cp -L` 或 `rsync -L`。
- 当前 `resources/qwen3vl-4b` 已确认是实体文件，不是 symlink。

### 7.1 Direct Python dry run 已通过

在学校服务器无 Docker 时，已经通过直接 Python 方式模拟官方输入输出：

```bash
cd ~/workspace/orena-focus-submission-template/segment-algorithm
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ORENA_INPUT_PATH="$PWD/test/input/interface_1"
export ORENA_OUTPUT_PATH="$PWD/test/output/interface_1"
rm -rf "$ORENA_OUTPUT_PATH"
mkdir -p "$ORENA_OUTPUT_PATH"
python inference.py
```

结果：

```text
Device: cuda:0
GPU: NVIDIA RTX A5000
Batch size: 3
Model setup completed in about 7 s
Wrote 3 responses with 0 failures
```

输出格式：

```json
[
  {"qID": "q001", "content": "Yes.", "latency": 3.8453794103115797},
  {"qID": "q002", "content": "1", "latency": 1.4248216934502125},
  {"qID": "q003", "content": "No.", "latency": 1.4184772800654173}
]
```

解释：

- 这说明离线资源加载、Qwen+LoRA 推理、官方 request/response 格式链路均正常。
- 但这不是最终官方 Docker 验收，因为它没有通过 `do_test_run.sh` 在容器中运行。

### 7.2 Docker 状态（2026-08-07 装好，卡点已解决）

管理员已授予 sudo 权限，用户自行安装完成：

- **Docker Engine 28.1.1**（官方 docker-ce 源，非 docker.io/snap）+ containerd 1.7.27 + buildx 0.23.0 + compose 2.35.1。
- **NVIDIA Container Toolkit 1.19.1**，nvidia runtime 已配进 `/etc/docker/daemon.json`。
- **Jiali_Wang 在 docker 组**。
- **/etc/docker/daemon.json** 同时含 `runtimes.nvidia` 和 `registry-mirrors`（docker.m.daocloud.io / docker.1panel.live / hub.rat.dev）。Docker Hub 大文件直拉会 connection reset，必须走加速器。

用 docker 的项目约定（重要）：

- **VS Code Remote-SSH 终端不刷新组关系**：用 docker 前先 `newgrp docker`，或用普通终端 `ssh Jiali_Wang@10.176.61.126` 真 login shell（组正常加载）。
- 每次新终端先激活 conda：`source ~/tools/miniconda3/etc/profile.d/conda.sh && conda activate orena-focus`。
- 约定单卡：`export CUDA_VISIBLE_DEVICES=0`。

已验证：

- `docker run --rm hello-world` ✓
- `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` ✓ → 2×RTX A5000, Driver 470.256.02。
- **注意**：nvidia-smi 能跑 ≠ PyTorch CUDA runtime 能跑。驱动 470（CUDA 11.4）跑不了官方 base image 的 CUDA 12.4 PyTorch，inference.py 自动回退 CPU。本地 do_test_run 用 CPU 跑通即算通过；官方评测机有新驱动能 GPU 跑。
- `./do_test_run.sh` ✓（build 818s，3 条 0 failures，CPU 回退）。
- `./do_save.sh` ✓ → tarball `segment-algorithm_2026-08-07T15-57-33.08855081+08-00.tar.gz` 生成，可提交。

服务器磁盘隐患：`/` 盘 98% 满（剩约 40G），后续需迁移数据到 `/mnt/data/jiali_wang` 或清理 docker 镜像。

如果通过，再进入：

```bash
cd ~/workspace/orena-focus-submission-template/segment-algorithm
./do_test_run.sh
./do_save.sh
```

## 8. 常见踩坑与解决方法

### 8.1 Conda 没生效

症状：

```text
conda: command not found
python: command not found
ModuleNotFoundError: No module named 'torch'
```

解决：

```bash
source ~/tools/miniconda3/etc/profile.d/conda.sh
conda activate orena-focus
```

如果是新 terminal，必须重新执行。

### 8.2 Git 没找到

症状：

```text
Command 'git' not found
```

常见原因：

- 当前没激活 `orena-focus`。
- 系统无全局 git，但 conda env 内有 git。

解决：

```bash
source ~/tools/miniconda3/etc/profile.d/conda.sh
conda activate orena-focus
git --version
```

### 8.3 Hugging Face 403 / Gemma license / Xet 问题

曾遇到：

- 数据下载 403 Forbidden。
- Gemma/MedGemma 需要在 Hugging Face 页面接受 license。
- Gemma 下载可能报 `Unable to parse string as hex hash value`。

解决思路：

- 确认 `hf auth login` 已登录。
- 在浏览器里接受模型 license。
- 必要时用下载脚本的 `--disable-xet`。

### 8.4 `device_map="auto"` 导致模型在 CPU

症状：

- `nvidia-smi` 中 Python 只占用几百 MB。
- 推理卡住或特别慢。
- 日志里出现 model on cpu / input_ids on cuda 的 warning。

解决：

显式用 `.to(self.device)`，不要让 `device_map="auto"` 在当前环境自行判断。

### 8.5 Overlay 生成很慢

原因：

- `VideoTimestampOverlayPreprocessor` 要对 30 条长视频逐帧叠 timestamp。
- overlay 最终约 81G。

处理：

- 这是一次性预处理，不是 4000 QA 各自重复生成。
- 不要中途误以为是无限循环。
- 观察 `find ~/data/focus/heico/overlayed -type f -name '*_overlay.mp4' | wc -l` 和近期修改文件即可。

### 8.6 Overlay 文件数正确但视频截断

症状：

Full overlay inference 到某一条时：

```text
IndexError: Out of bound indices
```

解决：

- 对比 QA time window 和视频总帧数/时长。
- 修复截断 overlay。
- 后续训练使用 clip-window audit 后的 `.clip_valid.jsonl`。

### 8.7 Open-VLM 模型占用空间过大

曾经 `~/workspace/vlm-models` 约 80G，包含 MiniCPM、LLaVA、InternVL、Gemma、MedGemma。
因为结果已记录且 Qwen 主线更强，已删除该目录。

原则：

- 不要重新下载这些候选模型，除非明确开启新的 open-VLM 实验。
- 大模型和 cache 后续放 `/mnt/data/jiali_wang`。

### 8.8 Docker 不能由当前账号安装

症状：

```text
sudo -v
对不起，用户 Jiali_Wang 不能在 UNNC-CVIP-03 上运行 sudo。
```

解决：

- 让管理员安装 Docker Engine + NVIDIA Container Toolkit。
- 或让管理员把用户加入 docker 组。
- 没有 Docker 时只能 direct Python dry run，不能生成官方 tarball。

### 8.9 Qwen video metadata warning

在 direct Python dry run 中仍可能看到：

```text
Qwen3VL requires frame timestamps to construct prompts...
Defaulting to fps=24...
```

当前判断：

- 该 warning 不影响 `0 failures` 和输出文件生成。
- 曾尝试传 `return_video_kwargs=True`，但需要将 `fps: [x]` 压平为 `fps: x`。
- 当前 `submission/segment_qwen_lora/inference.py` 已包含兼容逻辑。
- 为了稳定，不建议在正式 Docker 前为了消除 warning 大幅改动核心推理路径。

## 9. 当前最推荐的下一步

> ⚠️ **2026-08-08 更新**：Docker 环境已装好（见 7.2 节）、submission tarball
> 已生成、知识库已更新 commit `6740346`。**首次官方提交 Evaluations: Failed
> （2026-08-08 21:44，pre-evaluation phase，剩余 9/10 额度）**。当前首要任务
> 是诊断 Failed 原因（优先怀疑超时，降 `VIDEO_FPS`）→ 修复 → 重新提交。
> **完整交接状态见 `knowledge_base/HANDOFF_CODEX_2026-08-08.md`**。
>
> 下方"优先级 1/2/3"为历史记录（均已完成），保留备查。

优先级 1：解决 Docker 环境。

1. 管理员安装 Docker Engine 和 NVIDIA Container Toolkit。
2. 验证：

```bash
docker --version && docker run --rm hello-world && docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

3. 更新 project repo：

```bash
source ~/tools/miniconda3/etc/profile.d/conda.sh
conda activate orena-focus
cd ~/workspace/VLM-Competition
git pull origin main
```

4. 覆盖 official template：

```bash
cp ~/workspace/VLM-Competition/submission/segment_qwen_lora/inference.py \
  ~/workspace/orena-focus-submission-template/segment-algorithm/inference.py

cp ~/workspace/VLM-Competition/submission/segment_qwen_lora/requirements.txt \
  ~/workspace/orena-focus-submission-template/segment-algorithm/requirements.txt
```

5. 运行官方 Docker 测试：

```bash
cd ~/workspace/orena-focus-submission-template/segment-algorithm
export CUDA_VISIBLE_DEVICES=0
./do_test_run.sh
```

6. 如果通过，生成 submission tarball：

```bash
./do_save.sh
```

优先级 2：如果 Docker build 报依赖问题。

可能需要检查：

- `requirements.txt` 与官方 base image `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`
  是否兼容。
- `transformers>=5.13.0` 和 `peft>=0.19.1` 是否能在 Docker build 中顺利安装。
- Docker build 是否有网络访问 PyPI。
- 镜像层是否包含 8.3G Qwen base + 74M LoRA adapter。

优先级 3：如果 official Docker test 通过。

立刻更新：

- `docs/official_submission_qwen_lora.md`
- `knowledge_base/START_HERE.md`
- `knowledge_base/project_state.md`
- `knowledge_base/workflows.md`
- `docs/research_log.md`
- `results/experiment_events.csv`
- commit + push 到 GitHub。

## 10. 论文/报告材料现状

重要文件：

```text
docs/comprehensive_data_comparison.md
docs/evaluator_style_summary_tables.md
docs/research_log.md
docs/script_workflow_explained.md
report/README.md
results/main_result_summary.csv
results/evaluator_style_full_4000_summaries.csv
results/evaluator_style_full_4000_summaries.txt
results/lora_full_test_vs_overlay_baseline.csv
```

用户需要的表格风格：

- 类似官方 evaluator terminal output。
- 列包含：

```text
level, name, accuracy, ci_low, ci_high, count
```

已有脚本：

```bash
python scripts/print_evaluator_style_summary.py
```

只打印 LoRA full TEST：

```bash
python scripts/print_evaluator_style_summary.py \
  --experiment qwen3vl_lora_full_4000
```

论文可用主要结论：

- Qwen overlay baseline full TEST overall `0.207500`。
- Qwen LoRA-SFT full TEST overall `0.279000`，绝对提升 `+0.071500`。
- 最大提升来自 object identification 和 fo_class。
- timestamp overlay 对 temporal grounding 有帮助，但不足以解决时间定位。
- 开源医学模型 MedGemma prompt-only full TEST overall `0.188250`，不如 Qwen 主线，但在部分 object recognition 指标上有参考价值。

## 11. 协作协议：辅助 agent 应该怎么工作

建议新 agent 遵守：

1. 不要一上来重构项目。
2. 先读：
   - `knowledge_base/AGENT_HANDOFF_ZH.md`
   - `knowledge_base/START_HERE.md`
   - `knowledge_base/project_state.md`
   - `knowledge_base/experiments.md`
   - `knowledge_base/workflows.md`
   - `docs/official_submission_qwen_lora.md`
3. 远端操作一次只给一条命令。
4. 每次分析用户贴回来的 terminal 输出后，再给下一条命令。
5. 不要要求用户暴露密码或 token。
6. 任何改动脚本的操作都要：
   - 本地修改
   - 本地检查
   - commit + push
   - 服务器 `git pull`
   - 复制到 official template 或运行对应命令
7. 遇到实验结果，立刻结构化记录到 `results/*.csv` 或 docs。
8. 遇到工程状态变化，更新 knowledge base。
9. 遇到模型/数据大文件，不要提交 GitHub。
10. Docker 相关操作要注意：
    - 当前用户原本无 sudo。
    - 管理员安装后才继续。
    - 官方最终验收必须走 `./do_test_run.sh` 和 `./do_save.sh`。

## 12. 最短恢复路径

如果上下文全部丢失，新 agent 只需要做：

```bash
cd C:\Users\28101\Documents\VLM-Competition
```

然后阅读：

```text
knowledge_base/AGENT_HANDOFF_ZH.md
knowledge_base/START_HERE.md
docs/official_submission_qwen_lora.md
```

如果要在远端继续：

```bash
source ~/tools/miniconda3/etc/profile.d/conda.sh
conda activate orena-focus
cd ~/workspace/VLM-Competition
git pull origin main
```

如果 Docker 已安装，则立即验证：

```bash
docker --version && docker run --rm hello-world && docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

如果验证通过，继续官方 template：

```bash
cd ~/workspace/orena-focus-submission-template/segment-algorithm
export CUDA_VISIBLE_DEVICES=0
./do_test_run.sh
```

