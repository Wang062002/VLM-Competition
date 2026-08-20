# ORena FOCUS SEGMENT 项目简介（算力申请用）

## 项目目标

参加 **ORena FOCUS 手术视频问答挑战赛**（SEGMENT track），用视觉语言模型（VLM）回答手术视频中的问题（如器械识别、手术阶段、时间定位、FO 分类等）。目标在官方 leaderboard 上超过 fine-tuned open-source baseline（~0.50），进入决赛。

当前成绩：首次官方预评估 pre-evaluation score = **0.3233**（leaderboard 第 27 名），与 baseline 0.50 有差距，正在迭代提升。

## 方法

**主线方法：Qwen3-VL-4B-Instruct + LoRA-SFT**

- **基模**：Qwen3-VL-4B-Instruct（4B 参数视觉语言模型，Apache-2.0）
- **适配方法**：LoRA（rank=8, alpha=16, dropout=0.05），目标模块 q/k/v/o/gate/up/down_proj，**视觉部分冻结**（手术 VLM 常规做法，省显存 + 防视觉特征漂移）
- **训练数据**：官方 HeiCo-FOCUS + LapChole-FOCUS 两个数据集的 SEGMENT track TRAIN，联合 13746 样本（内部 split 后 clip-valid 11124 train / 1244 val）
- **训练配置**：4 epochs，lr 1e-4，gradient accumulation 4，video stride 25（1fps，5 分钟视频 ~300 帧，和官方推理一致），bf16，gradient checkpointing
- **推理**：Docker 容器提交（pytorch:2.7.1-cuda12.8），官方评测机 96GB GPU 跑

**v2 改进**（正在训练）：
1. 数据从单 HeiCo（5959）扩展到 HeiCo + LapChole（11124，+87%）
2. 训练从 1 epoch 提升到 4 epochs（v1 欠拟合，eval loss 0.428）
3. 对齐官方推理帧率（stride 25 / VIDEO_FPS=1.0）
4. MAX_NEW_TOKENS 64→128（对齐官方示例）

## 数据集

| 数据集 | 视频数 | TRAIN QA | TEST QA | 说明 |
|---|---|---|---|---|
| HeiCo-FOCUS | 30 | 8000 | 4000 | 结直肠手术，第一批数据 |
| LapChole-FOCUS | 100（labeled）+ 70（unlabeled）| 5746 | 2254 | 腹腔镜胆囊切除术，第二批数据 |
| **联合** | 130 | **13746** | **6254** | v2 训练用联合 TRAIN |

- 数据来源：HuggingFace `orena-dkfz/heico-focus-vqa` + `orena-dkfz/lapchole-focus-vqa`（Apache-2.0 / gated，已获权限）
- 视频预处理：timestamp overlay（官方 `VideoTimestampOverlayPreprocessor`），视频 + QA 配对
- 联合 split：`~/workspace/focus-runs/data-audit/clip-window-audit-heico-lapchole-seed20260707/sft_{train,val}_overlay.clip_valid.jsonl`

## 硬件需求

- **当前环境**：学校服务器 2×RTX A5000 24GB，driver 470（CUDA 11.4），单卡训练
- **瓶颈**：driver 470 跑不了 Triton（Qwen3.5 GatedDeltaNet）、CUDA 12.4 PyTorch；Qwen3-VL bf16 训练 4B 模型 24GB 够但边界紧（300 帧视频 logits + 模型 + 激活 ~16GB）
- **新算力期望**：
  - GPU ≥ 80GB VRAM（A100 80GB / H100 / 或更新）→ 能跑 Qwen3.5（更强基模）+ 高帧训练 + Triton fast path
  - driver ≥ 550（CUDA 12.4+）→ 解锁 CUDA 12 fast path + flash-linear-attention + Triton
  - 存储 ≥ 500GB（HeiCo 237G + LapChole 视频 + Qwen3-VL/Qwen3.5 模型 + 训练 checkpoint）
  - 持续训练时间预算：单次 4 epochs ~42 小时（当前 A5000），80GB GPU 估计 ~15-20 小时

## 迁移清单

- 数据：`/mnt/data/jiali_wang/focus/{heico,lapchole}`（视频 + overlay，~300G+）
- 模型：`/mnt/data/jiali_wang/models/{Qwen3.5-4B}`（8.8G）+ Qwen3-VL-4B（HF cache）
- 训练产物：`~/workspace/focus-runs/lora-sft/`（v1 adapter 74M + v2 训练中 + manifest）
- 代码：`~/workspace/VLM-Competition`（git 仓库，直接 clone）
- 官方 template：`~/workspace/orena-focus-submission-template`（Docker submission）
- conda env：`orena-focus`（python 3.10 + torch 2.7.1+cu118 + transformers 5.13.0 + peft 0.19.1 + qwen-vl-utils 0.0.14）

## 时间节点

- 8-19（今）：v2 训练启动，ETA ~42 小时（周六晚/周日完成）
- 8-22：学校机房停电维修开始 → **需在此前完成迁移**或暂停训练
- v2 训练完成后：收 adapter → 本地评测 → 重新提交官方 pre-eva → 看 leaderboard
