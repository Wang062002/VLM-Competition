# Codex 交接手册：ORena FOCUS SEGMENT 项目

交接日期：2026-08-08 22:00
交接人：小巴 → Codex
状态：Docker 环境装好 + submission tarball 生成 + 官方首次提交 **Failed**（待诊断）

> 本文档聚焦"当前状态 + Codex 要接手的首要任务"。完整项目背景见
> `knowledge_base/AGENT_HANDOFF_ZH.md`，先读它再读本文。

## 1. 项目一句话

参加 ORena FOCUS 手术视频 QA challenge，SEGMENT track，HeiCo 数据集。
主线：`Qwen3-VL-4B-Instruct + LoRA-SFT`，本地 full TEST 4000 overall **0.279**。

## 2. 当前状态（截至 2026-08-08 22:00）

### ✅ 已完成
- **Docker 环境**（2026-08-07 装好）：Docker Engine 28.1.1 + NVIDIA Container
  Toolkit 1.19.1 + 国内镜像加速器（daemon.json）。详见
  `knowledge_base/AGENT_HANDOFF_ZH.md` 第 7.2 节。
- **官方 submission 本地自测**：`./do_test_run.sh` 通过（3 条 0 failures，
  CPU 回退——驱动 470 跑不了 CUDA 12.4 PyTorch，但链路通）；
  `./do_save.sh` 生成 tarball
  `segment-algorithm_2026-08-07T15-57-33.08855081+08-00.tar.gz`（10.8G）。
- **知识库已更新**：commit `6740346` 已 push 到 GitHub main（2026-08-07），
  服务器副本待 `git pull`。
- **官方平台注册 + Join 通过**：用户名 `Wjiali`，已加入 SEGMENT challenge。
- **首次官方提交**：2026-08-08 21:44 提交到 pre-evaluation phase，
  Algorithm "Qwen3 VL 4B LoRA SFT SEGMENT"（GPU 选 RTX PRO 6000 Blackwell 96GB）。

### ❌ 当前卡点：首次提交 Evaluations: Failed
- 提交后状态 Queued → **Failed**（红色）。
- **失败原因未诊断**——Codex 首要任务就是查这个。
- 提交额度剩余 **9/10**（10 次预评估额度，已用 1 次失败）。

### 📊 本地参考分数（官方结果出来前的预期）
- Qwen overlay baseline full TEST 4000：overall 0.2075
- Qwen LoRA-SFT full TEST 4000：**overall 0.279**，pre-eval 0.4028
- 最大提升：object_identification +0.32，fo_class +0.26
- temporal_grounding 仍弱（0.072）

## 3. Codex 首要任务：诊断官方提交 Failed

### 2026-08-08 Codex 官方要求复核结论

已重新读取官方 `IMSY-DKFZ/orena-focus-submission-template` README 和
`segment-algorithm/inference.py`，确认当前修复方向应优先对齐两点：

- SEGMENT 按批处理，预算为 `120 s + B x 15 s`；超预算会 forfeiture 题目，
  超过 20% 会 forfeiture 整批。
- 官方 SEGMENT dummy 的视频预处理使用 `TARGET_FPS=1.0` 且 `MAX_FRAMES=64`
  的有界采样；我们首次提交版只有 `VIDEO_FPS=1.0`，没有显式最大帧数上限，
  长 clip 可能采到数百帧，是官方失败的高优先级嫌疑。

已在本地 `submission/segment_qwen_lora/inference.py` 对齐：

- `MAX_NEW_TOKENS` 默认从 `128` 降到 `64`。
- 新增 `VIDEO_MIN_FRAMES=4`、`VIDEO_MAX_FRAMES=64`。
- 视频消息传入 `min_frames/max_frames`。
- Qwen3-VL 预处理改为 `image_patch_size=16`、
  `return_video_metadata=True`，并向 processor 传 `video_metadata` 和
  `do_resize=False`，对齐 Qwen3-VL 官方 `qwen-vl-utils` 用法。

下一步：commit/push 后，服务器 `git pull`，复制到官方 template，重新
`./do_test_run.sh` 和 `./do_save.sh`，上传新 tarball 再提交。

### 步骤 1：拿失败日志
登录 `https://segment.orena-focus-challenge.org/` → Submissions 页 →
点那条 Failed 记录的 **Failed** 按钮 → 看错误详情/logs。把日志贴回来分析。

### 步骤 2：候选原因 + 修复（按可能性排序）

| 候选 | 症状关键词 | 修复方向 |
|---|---|---|
| **超时**（最可能）| `MAX_TIME_LIMIT` / `time limit exceeded` / 某题超 15s | 降 `VIDEO_FPS`：inference.py 默认 `1.0`，改 `0.5` 或 `0.3`（环境变量，do_test_run 时 `export VIDEO_FPS=0.5`）；或减 `MAX_NEW_TOKENS`（默认 128）|
| 找不到输入 | `FileNotFoundError` `/input/overlayed/` 或 `FO_definitions.json` | inference.py 里 `VIDEO_DIR = INPUT_PATH / "overlayed"`，官方 input 结构可能不同——查官方 test/input 实际目录 |
| 输出格式错 | answer.json schema 不符 | 对比 inference.py 输出与官方 schema（`save_items` + `Response`） |
| 显存/驱动 | `CUDA out of memory` / driver 报错 | 96GB 不该 OOM；Blackwell + CUDA 12.4 镜像兼容问题要看 logs |
| inference 抛异常 | Python traceback | 看具体错 |

### 步骤 3：修复后重新提交
1. 本地改 `submission/segment_qwen_lora/inference.py`（如降 VIDEO_FPS）
2. commit + push main
3. 服务器：`cd ~/workspace/VLM-Competition && git pull origin main`
4. 复制到官方 template：
   `cp submission/segment_qwen_lora/inference.py ~/workspace/orena-focus-submission-template/segment-algorithm/inference.py`
5. `cd ~/workspace/orena-focus-submission-template/segment-algorithm && ./do_test_run.sh`（验证）
6. `./do_save.sh`（生成新 tarball）
7. 服务器 scp 新 tarball 到本地，或本地已有则用新的
8. 官方平台：Algorithms → 该 algorithm → Upload a Container → 传新 tarball
   → 等 Image Can Be Used=True → Submit → pre-evaluation
9. 等 leaderboard（不要重复提交）

## 4. 关键约束 / 坑（务必遵守）

### Docker 用法
- **VS Code Remote-SSH 终端不刷新组**：用 docker 前 `newgrp docker`，或用
  普通 `ssh Jiali_Wang@10.176.61.126` 真 login shell。
- 每次新终端：`source ~/tools/miniconda3/etc/profile.d/conda.sh && conda activate orena-focus`。
- 约定单卡：`export CUDA_VISIBLE_DEVICES=0`。
- **Docker Hub 大文件直拉 connection reset**：daemon.json 已配加速器
  （docker.m.daocloud.io / docker.1panel.live / hub.rat.dev），别删。

### 驱动 470 的限制（重要）
- 驱动 470（CUDA 11.4）**跑不了**官方 base image 的 CUDA 12.4 PyTorch。
- inference.py 自动回退 CPU（`Device: cpu`），本地 do_test_run 用 CPU 跑通
  即算通过——**不要为了本地测试升级驱动**（风险大，官方评测机有新驱动）。
- 官方评测用 80GB+ GPU（你选的 RTX PRO 6000 Blackwell 96GB），能 GPU 跑。

### VIDEO_FPS 超时风险
- inference.py 默认 `VIDEO_FPS=1.0`，5 分钟视频采 ~300 帧。
- 官方 SEGMENT 每题 **15 秒预算**，300 帧 + LoRA 推理 + 128 token 生成
  **可能超时**——这是首次提交 Failed 的头号嫌疑。
- 降 VIDEO_FPS 是环境变量，不用改代码，do_test_run 时 `export VIDEO_FPS=0.5`。

### 磁盘 98% 满
- 服务器 `/` 盘 98% 满，剩 ~40G。do_save 新 tarball 前先 `docker builder prune -f`
  清 build cache（别 `docker system prune`，会删 segment-algorithm 镜像）。
- 后续大数据迁 `/mnt/data/jiali_wang`。

## 5. 官方提交平台信息

- 主平台：`https://orena-focus-challenge.org/`
- SEGMENT 子域名：`https://segment.orena-focus-challenge.org/`
- 官方 template 仓库：`https://github.com/IMSY-DKFZ/orena-focus-submission-template`
- 提交方式：纯网页表单上传 tarball（无 CLI/presigned URL）。
- 提交流程：登录 → 顶部 Algorithms → 创建 algorithm（填标题+描述）→
  Containers → Upload a Container → 等 Image Can Be Used=True →
  顶部 Submit → pre-evaluation phase → 选 algorithm → Save 提交。
- 两个官方 baseline：① frontier closed-source VLM（GPT/Gemini 级 zero-shot）；
  ② fine-tuned open-source VLM。**预评估要超两个才进决赛**。
- 官方未公开 baseline 具体分数。

## 6. 关键路径速查

- 本地仓库：`C:\Users\28101\Documents\VLM-Competition`（GitHub: Wang062002/VLM-Competition, main）
- 远端项目副本：`~/workspace/VLM-Competition`
- 官方 template：`~/workspace/orena-focus-submission-template/segment-algorithm`
- submission 文件：本地 `submission/segment_qwen_lora/{inference.py,requirements.txt,README.md,method_description.md}`
- 已复制进 template 的资源：`segment-algorithm/resources/qwen3vl-4b`（8.3G）+ `qwen3vl-lora`（74M）
- LoRA adapter 源：`~/workspace/focus-runs/lora-sft/qwen3vl-4b-sft-valid5959-e1/adapter-final`
- 本地 tarball（已下载）：`C:\Users\28101\Downloads\segment-algorithm_2026-08-07T15-57-33.08855081+08-00.tar.gz`（10.8G）

## 7. 协作约定（用户偏好）

- 回复中文，用户称呼"先生"，风格"干脆利落"。
- 远端操作一次只给一条命令，他贴回输出再给下一条。
- 边推进边解释每步"在做什么、为什么、怎么看结果"。
- 有实质进展就更新知识库/论文材料/workflow/GitHub main。
- 不在项目文件存密码、HF token。
- 用户允许推送到 GitHub main 分支。

## 8. 必读文件（接手顺序）

1. `knowledge_base/AGENT_HANDOFF_ZH.md`（完整项目背景）
2. `knowledge_base/START_HERE.md`（关键数字 + 当前阶段）
3. `knowledge_base/project_state.md`（含 Docker 环境段）
4. `knowledge_base/workflows.md`（含 Official Docker Submission workflow）
5. `docs/official_submission_qwen_lora.md`（submission 状态）
6. 本文（当前交接状态）

## 9. 最短恢复路径（如果上下文丢失）

1. 读本文 + `AGENT_HANDOFF_ZH.md`。
2. 登录官方平台看 Submissions 页 Failed 详情。
3. 诊断失败原因（优先怀疑超时，降 VIDEO_FPS）。
4. 改 inference.py → commit push → 服务器 git pull → 复制到 template →
   do_test_run 验证 → do_save → 上传 → 重新提交。
5. 等 leaderboard，对比本地 0.279。
