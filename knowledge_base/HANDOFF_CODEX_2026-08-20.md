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

## 12. 完整改动时间线（2026-08-18 ~ 08-20，小巴接手期间）

### git commit 序列（main 分支）
- `6740346`（Codex，8-7）：Docker 环境装好 + submission 知识库更新
- `0f66677`（Codex，8-16）：LapChole access approved
- `458925e`（小巴，8-18）：Adapt train + inference scripts to Qwen3.5-4B（后因 OOM 推迟）
- `2d7cd94`（小巴，8-18）：Revert Qwen3.5 adaptation, back to Qwen3-VL
- `fac25f1`（小巴，8-19）：Align MAX_NEW_TOKENS 64→128（对齐官方示例）
- `5af51dc`（小巴，8-19）：Fix 4bit quantization BitsAndBytesConfig（load_in_4bit deprecated）
- `ce02dc2`（小巴，8-19）：Remove min_frames/max_frames from video item（restore v1 behavior）
- `6d88508`（小巴，8-19）：**Fix OOM root cause** — revert encode_sample to v1 simple process_vision_info（去掉 do_resize=False 等）
- `87eebd4`（小巴，8-20）：Add Codex handoff v2 + compute application brief
- `bf2f9a6`（小巴，8-20）：Update handoff confirm option B + 3 epochs

### 关键变故
1. **Qwen3.5 换基模失败**（8-18）：Qwen3.5-4B vocab 248k（比 Qwen3 的 151k 大 64%）+ fps 采样 300 帧 → logits 11.68GB OOM。driver 470 跑不了 Triton（Qwen3.5 GatedDeltaNet fallback torch 慢）。回退 Qwen3-VL。
2. **OOM 反复**（8-19 凌晨）：stride 25/50/100、4bit 都 OOM。根因是 Codex 加的 `do_resize=False`，不是帧数或 4bit。v1 原始脚本（69c3bb5）能跑，Codex 改的（4ebfe56）不能跑。
3. **ETA 误判**（8-19→8-20）：昨天报 42h（第一个 step ETA 不准），实际 164h（6.95 天），超 8-22 停电 4 天。先生定 B（等新算力从头跑 3 epochs）。
4. **8-22 停电 + 新算力可能延后**：v2 训练进度放弃，备份 v1 adapter 兜底。

## 13. 对 Codex 的质疑（小巴视角）

### 4ebfe56 加 do_resize=False 没测过
Codex 为"对齐提交容器帧预算"给 `encode_sample` 加了 `image_patch_size=16, return_video_kwargs=True, return_video_metadata=True, do_resize=False`。**但这些参数没在 v1 基线上验证过**，导致 OOM。改脚本前应先验证不破坏 v1。

### "官方推理 max_frames=64" 判断不准
Codex 审计文档 `sft_training_audit_20260810.md` 说 v1 mismatch 是训练 300 帧 vs 推理 max_frames=64。**实际官方 examples/inference.py 用 stride 25（1fps，~300 帧）**，inference.py 的 VIDEO_MAX_FRAMES=64 是配置但 Qwen3-VL process_vision_info 可能也忽略 max_frames 用 fps。v1 训练推理帧率本来就一致（都 stride 25），不需要加 max_frames。

### min_frames/max_frames 加到 video item 改变 qwen_vl_utils 行为
Codex 4ebfe56 给 video item 加 min_frames/max_frames 字段，v1 没有这俩字段。qwen_vl_utils 收到这俩参数后采样行为变了（即使 max_frames 没限制住仍采 300 帧，但参数本身让 qwen_vl_utils 走不同代码路径）。

### ETA 估算不可靠
训练启动第一个 step 的 ETA（42h）严重不准（含启动开销 + 前 128 条短样本 rate 估高）。应等稳定 rate（smoke 128 实测 7.69s/样本 → 95h）再报 ETA。

### 4bit 量化没真生效
bitsandbytes 装上 + quantization_config 传进 model.config，但 Qwen3-VL 模型实际没被量化（`has bnb module: False`）。transformers 5.13.0 对 Qwen3-VL 的量化路径可能有兼容问题。

## 14. 小巴的工具限制

- **小巴的 Bash 工具不走先生本地梯子**，`git push github` schannel 直连失败（退出码 1 无输出）。commit 本地能做，push 要先生本地终端或 Codex（本地 CLI）做。
- 先生授权：脚本改完直接 commit + push main，不用问。但 push 受小巴网络限制，实际要先生手动 push。

## 15. 先生偏好（补充）

- 先生是 Docker 零基础，涉及 Docker/容器操作要同步教学（先讲概念再给命令）。
- 先生风格"干脆利落"，少废话直奔结果。
- 远端操作一次一条命令，贴回输出再给下一条。
- 称呼"先生"。
- 边推进边解释每步"在做什么、为什么、怎么看结果"。
