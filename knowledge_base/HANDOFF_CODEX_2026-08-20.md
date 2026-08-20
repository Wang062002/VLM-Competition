# Codex 交接手册 v2：ORena FOCUS SEGMENT 项目

交接日期：2026-08-20
交接人：小巴 → Codex
状态：v2 训练在跑（4 epochs，ETA 8-26，超 8-22 停电）→ **先生定 B：等新算力从头跑 3 epochs，当前 v2 进度放弃** + 8-22 机房停电 + 申请新算力

> 本文档聚焦"当前状态 + Codex 接手的首要任务"。完整背景见 `knowledge_base/AGENT_HANDOFF_ZH.md` + 上版 `knowledge_base/HANDOFF_CODEX_2026-08-08.md`。

## 1. 项目一句话

ORena FOCUS 手术视频 QA challenge，SEGMENT track。主线 Qwen3-VL-4B + LoRA-SFT。v1 官方 pre-eva 0.3233（leaderboard 27），baseline ~0.50。v2 训练中（HeiCo+LapChole + 4 epochs）。

## 2. 当前状态（截至 2026-08-20）

### ✅ 已完成
- **v1 完整链路**：Qwen3-VL-4B + LoRA-SFT（HeiCo 5959，1 epoch，eval loss 0.428）→ 官方 pre-eva 0.3233
- **LapChole 数据集**：gated 权限已获，下载完成（`/mnt/data/jiali_wang/focus/lapchole`）
- **联合 split**：HeiCo 8000 + LapChole 5746 = 13746 → clip-valid 11124 train / 1244 val（manifest: `~/workspace/focus-runs/data-audit/clip-window-audit-heico-lapchole-seed20260707/`）
- **OOM 根因找到 + 修复**：Codex 4ebfe56 加的 `do_resize=False` 导致 OOM，回退 v1 简单 encode_sample 修复（commit `6d88508`）
- **v2 训练启动**：2026-08-19 03:00，nohup PID 2856655，Qwen3-VL + HeiCo+LapChole + 4 epochs + stride 25

### ❌ 当前卡点
- **v2 训练 ETA 164h（6.95 天）**，8-26 完成，**超 8-22 停电 4 天**
- **8-22 学校机房停电维修**，训练会被迫中断
- **当前脚本不支持中途存 checkpoint**（只在最后存 adapter-final），中断 = 丢失进度
- **新算力申请可能等停电后才批下来**

### 📊 关键数字
- v1 官方 pre-eva：0.3233（leaderboard 27）
- v1 本地 full TEST：0.4028（pre-eval style）
- v1 eval loss：0.428
- 官方 baseline：~0.50
- v2 训练量：11124 × 4 = 44496 micro steps
- v2 实测 rate：0.08 Hz（13.5 秒/样本）

## 3. OOM 根因 + 修复（重要教训）

**根因**：Codex 4ebfe56 为"对齐提交容器帧预算"给 `encode_sample` 的 `process_vision_info` 加了 `image_patch_size=16, return_video_kwargs=True, return_video_metadata=True` + 给 processor 加了 `do_resize=False`。**`do_resize=False` 最可能让视频帧保持原分辨率不被压缩到 640×360，显存暴涨 OOM**。

**修复**：encode_sample 回退到 v1 简单调用 `process_vision_info(messages)` + `processor(text, images, videos, padding, return_tensors)`。commit `6d88508`。

**教训**：改脚本前先验证不破坏 v1 基线。Codex 加的参数本身没测过。

## 4. 备份策略（8-22 停电前）

### 代码（GitHub 已同步，不用迁移）
- 本地 `C:\Users\28101\Documents\VLM-Competition` ↔ GitHub `Wang062002/VLM-Competition`
- 新机器 `git clone https://github.com/Wang062002/VLM-Competition` 即可

### adapter（训练产物，必须备份）
- **v1 adapter**（74M）：`~/workspace/focus-runs/lora-sft/qwen3vl-4b-sft-valid5959-e1/adapter-final`
  - scp 到本地备份：`scp -r Jiali_Wang@10.176.61.126:~/workspace/focus-runs/lora-sft/qwen3vl-4b-sft-valid5959-e1/adapter-final ~/Downloads/v1-adapter-backup`
- **v2 checkpoint**：当前训练没 save_every，22 号停 = 没 checkpoint。如果要保 v2 进度，需改脚本加 `--save-every-steps`（见第 5 节）

### 数据集 + 模型（不备份，新机器重下）
- HeiCo：`hf download orena-dkfz/heico-focus-vqa`（gated 已获权限，237G）
- LapChole：`hf download orena-dkfz/lapchole-focus-vqa`（gated 已获权限）
- Qwen3-VL-4B：`hf download Qwen/Qwen3-VL-4B-Instruct`
- Qwen3.5-4B（备用）：`hf download Qwen/Qwen3.5-4B`（已下过，8.8G）
- 走 `HF_ENDPOINT=https://hf-mirror.com` 加速

## 5. Codex 接手首要任务

### 选项 A：改脚本支持 save/resume（推荐）
1. 给 `train_qwen3vl_lora_sft_smoke.py` 加：
   - `--save-every-steps N`：每 N optimizer steps 存 adapter + optimizer state + step number
   - `--resume-from <checkpoint_dir>`：load adapter + optimizer + step，跳过已训练样本继续
2. 现在 Ctrl+C 停训练（浪费 19h+），重启加 `--save-every-steps 500`
3. 22 号停电前停，scp 最后一个 checkpoint 到本地
4. 新机器 `--resume-from <checkpoint>` 继续

### 选项 B：等新算力从头跑（**先生已定**）
- 不改脚本，22 号停电训练中断（丢失进度，先生接受）
- 等新 80GB GPU 到位，**从头跑 3 epochs**（先生从 4 改 3 减时间）
- 80GB GPU 估计 ~12-15h 完成（3 epochs，比 4 epochs 快）
- v1 adapter 先备份到本地，停电期间不会断档

### 选项 C：stride 50 重启（赶 22 号）
- Ctrl+C 停，改 `--video-stride 50`（帧数减半 300→150，时间减半 ~82h）
- inference.py `VIDEO_FPS` 改 `0.5` 对齐
- 22 号前能完成 4 epochs
- 代价：150 帧比 300 帧信息少，temporal_grounding 可能更差

## 6. v2 训练完成后流程

1. 收 adapter：`~/workspace/focus-runs/lora-sft/qwen3vl-4b-sft-v2-heico-lapchole-e4/adapter-final`
2. 替换 submission：`cp -r <v2-adapter> ~/workspace/orena-focus-submission-template/segment-algorithm/resources/qwen3vl-lora/`
3. do_test_run 验证：`cd ~/workspace/orena-focus-submission-template/segment-algorithm && ./do_test_run.sh`
4. do_save 生成 tarball：`./do_save.sh`
5. 上传官方平台 + 提交 pre-eva
6. 对比 v1 的 0.3233

## 7. 关键路径速查

- 本地仓库：`C:\Users\28101\Documents\VLM-Competition`（GitHub: Wang062002/VLM-Competition, main）
- 远端项目副本：`~/workspace/VLM-Competition`
- 官方 template：`~/workspace/orena-focus-submission-template/segment-algorithm`
- 数据：`/mnt/data/jiali_wang/focus/{heico,lapchole}`（237G+）
- 模型：`/mnt/data/jiali_wang/models/Qwen3.5-4B`（8.8G）+ Qwen3-VL-4B（HF cache）
- v1 adapter：`~/workspace/focus-runs/lora-sft/qwen3vl-4b-sft-valid5959-e1/adapter-final`（74M）
- v2 训练中：`~/workspace/focus-runs/lora-sft/qwen3vl-4b-sft-v2-heico-lapchole-e4/`
- 联合 manifest：`~/workspace/focus-runs/data-audit/clip-window-audit-heico-lapchole-seed20260707/sft_{train,val}_overlay.clip_valid.jsonl`

## 8. 环境信息

- 服务器：10.176.61.126，user `Jiali_Wang`，2×RTX A5000 24GB，driver 470（CUDA 11.4）
- conda env：`orena-focus`（Python 3.10，torch 2.7.1+cu118，transformers 5.13.0，peft 0.19.1，qwen-vl-utils 0.0.14）
- 激活：`source ~/tools/miniconda3/etc/profile.d/conda.sh && conda activate orena-focus`
- 约定单卡：`CUDA_VISIBLE_DEVICES=0`
- driver 470 限制：跑不了 Triton / CUDA 12.4 PyTorch / flash-linear-attention（Qwen3.5 fallback torch 慢）
- 新算力期望：≥80GB VRAM + driver≥550

## 9. 协作约定

- 中文回复，用户称呼"先生"，风格"干脆利落"
- 远端操作一次一条命令，贴回输出再给下一条
- 边推进边解释"在做什么、为什么、怎么看结果"
- 脚本改完直接 commit + push main（先生授权，不用问）——但小巴的 Bash 不走先生梯子，push 要先生本地或 Codex 做
- 不在项目文件存密码、HF token

## 10. 必读文件（接手顺序）

1. `knowledge_base/AGENT_HANDOFF_ZH.md`（完整项目背景）
2. `knowledge_base/HANDOFF_CODEX_2026-08-08.md`（上版交接，Docker/提交细节）
3. `docs/sft_training_audit_20260810.md`（v1 差距审计 7 风险）
4. `docs/qwen_lora_sft_v2_lapchole_plan_20260811.md`（v2 训练计划）
5. `docs/project_brief_for_compute_application_20260819.md`（算力申请简介）
6. 本文（当前交接状态）

## 11. 最短恢复路径

1. 读本文 + `AGENT_HANDOFF_ZH.md`
2. SSH 服务器看训练状态：`ps aux | grep train_qwen3vl && tail ~/workspace/focus-runs/lora-sft/qwen3vl-4b-sft-v2-heico-lapchole-e4/train.log`
3. 决定 A/B/C（改脚本 resume / 等新算力 / stride 50 重启）
4. 备份 v1 adapter 到本地（scp 74M）
5. 8-22 停电前完成备份 + 决策
6. 新算力到位 → git clone 代码 + hf download 数据/模型 + resume/从头训练
