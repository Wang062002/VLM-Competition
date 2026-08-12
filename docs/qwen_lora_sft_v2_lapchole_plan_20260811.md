# Qwen3-VL LoRA-SFT v2：HeiCo + LapChole 再训练计划

日期：2026-08-11

## 背景

当前官方 SEGMENT pre-evaluation 已成功跑通，但分数为：

```text
pre_evaluation_score = 0.32331911598560603
leaderboard_position = 27
questions_forfeited = 0
questions_unanswered = 0
```

这说明容器、输入输出格式、GPU 兼容性和提交流程已经基本正确。现在的主要矛盾不是工程提交失败，而是模型能力和训练策略不足。

师兄建议：

- 训练数据量可能不够；
- 应该把 LapChole 也加入训练；
- 只训练 1 轮可能太少，建议至少 5 轮左右；
- prompt 构造大体没问题；
- LoRA 只加在线性层、视觉部分冻结是合理做法。

我对照官方资料后的判断：这个建议方向成立。下一轮应优先提升数据量和训练轮次，同时把训练时的视频帧预算对齐官方提交容器。

## 官方数据源核对

官方 `orena-focus` README 写明 FOCUS 基于两个数据集：

- HeiCo-FOCUS：30 个视频；
- LapChole-FOCUS：170 个视频。

官方 Hugging Face 数据卡显示：

- HeiCo-FOCUS-VQA：包含 `frame`、`segment`、`procedure` 三个 track，每个 track 有 `train.parquet` 和 `test.parquet`。
- LapChole-FOCUS-VQA：第二批 ORena FOCUS 数据，包含 100 个 labeled laparoscopic cholecystectomy 视频、70 个 unlabeled 视频、20,000 个 VQA pairs，并同样提供 `frame`、`segment`、`procedure` 三个 track 的 train/test parquet。
- LapChole 是 gated dataset，需要 ORena FOCUS Challenge 参与者、Hugging Face 登录、同邮箱注册和数据使用协议批准。

官方来源：

- https://github.com/IMSY-DKFZ/orena-focus
- https://huggingface.co/datasets/orena-dkfz/heico-focus-vqa
- https://huggingface.co/datasets/orena-dkfz/lapchole-focus-vqa

结论：SEGMENT 赛道可用的官方训练数据源应按两个数据集考虑：`heico` 和 `lapchole`。我们上一轮只用了 `heico`，确实少用了一批官方数据。

## 已更新的工程脚本

### `scripts/audit_and_split_segment_train.py`

已从单 HeiCo hardcode 改为支持多数据集：

```bash
--dataset heico --dataset lapchole
```

输出的 SFT JSONL 会保留每条样本所属的 dataset，并自动生成对应路径：

```text
/home/Jiali_Wang/data/focus/<dataset>/overlayed/<video_stem>_overlay.mp4
```

### `scripts/train_qwen3vl_lora_sft_smoke.py`

已补充官方一致帧预算参数：

```text
--video-min-frames 4
--video-max-frames 64
```

并让训练端使用和 submission container 更接近的视频处理方式：

- `return_video_kwargs=True`
- `return_video_metadata=True`
- `do_resize=False`
- `max_frames=64`

同时修正了 `--max-train-samples 0` / `--max-val-samples 0` 表示使用全部样本，避免全量训练时误跑成 32 条 smoke。

## v2 训练路线

### Step 1：确认 LapChole 是否可访问

先不下载大视频，只检查官方 loader 是否能看到 split。现在项目里已经有通用访问检查脚本：

```bash
source ~/tools/miniconda3/etc/profile.d/conda.sh && conda activate orena-focus && cd ~/workspace/VLM-Competition && python scripts/check_focus_dataset_access.py --root-dir /mnt/data/jiali_wang/focus --dataset heico --dataset lapchole --track segment --json-output ~/workspace/focus-runs/data-audit/qwen-lora-sft-v2-access-check.json
```

如果 LapChole 报 gated/access error，则先在 Hugging Face 页面申请权限。

2026-08-12 最新状态：使用 `JialiWang620` 账号检查后，HeiCo SEGMENT 可访问，
LapChole SEGMENT train/test 仍返回 Hugging Face gated dataset error。等待权限通过前，
不要执行 LapChole 下载。

### Step 2：下载并预处理 LapChole

权限通过后，下载到主数据目录或 `/mnt/data/jiali_wang/focus`。考虑主盘紧张，优先使用大盘：

```bash
source ~/tools/miniconda3/etc/profile.d/conda.sh && conda activate orena-focus && cd ~/workspace/VLM-Competition && python scripts/prepare_focus_data.py --root-dir /mnt/data/jiali_wang/focus --dataset lapchole --skip-frames
```

注意：`prepare_focus_data.py` 是通用脚本，后续可以用于 `heico`、`lapchole` 或官方新增数据集。

所有阶段命令都可以由配置文件统一打印，避免手动拼错参数：

```bash
python scripts/print_qwen_lora_sft_v2_commands.py --stage all
```

### Step 3：生成 HeiCo + LapChole 联合 SFT split

如果 HeiCo 也迁移到 `/mnt/data/jiali_wang/focus`，则统一 root-dir 用大盘；否则需要先完成数据迁移或保持 root-dir 在 `/home/Jiali_Wang/data/focus`。

```bash
source ~/tools/miniconda3/etc/profile.d/conda.sh && conda activate orena-focus && cd ~/workspace/VLM-Competition && python scripts/audit_and_split_segment_train.py \
  --root-dir /mnt/data/jiali_wang/focus \
  --dataset heico \
  --dataset lapchole \
  --output-dir ~/workspace/focus-runs/data-audit/segment-trainval-heico-lapchole-seed20260707
```

### Step 4：clip-window audit

```bash
source ~/tools/miniconda3/etc/profile.d/conda.sh && conda activate orena-focus && cd ~/workspace/VLM-Competition && python scripts/audit_sft_clip_windows.py \
  --input-jsonl ~/workspace/focus-runs/data-audit/segment-trainval-heico-lapchole-seed20260707/sft_train_overlay.jsonl \
  --input-jsonl ~/workspace/focus-runs/data-audit/segment-trainval-heico-lapchole-seed20260707/sft_val_overlay.jsonl \
  --output-dir ~/workspace/focus-runs/data-audit/clip-window-audit-heico-lapchole-seed20260707
```

训练应优先使用 `.clip_valid.jsonl`，避免再次因为坏 clip 中断。

### Step 5：先跑 128 条 smoke

```bash
source ~/tools/miniconda3/etc/profile.d/conda.sh && conda activate orena-focus && cd ~/workspace/VLM-Competition && CUDA_VISIBLE_DEVICES=0 python scripts/train_qwen3vl_lora_sft_smoke.py \
  --train-jsonl ~/workspace/focus-runs/data-audit/clip-window-audit-heico-lapchole-seed20260707/sft_train_overlay.clip_valid.jsonl \
  --val-jsonl ~/workspace/focus-runs/data-audit/clip-window-audit-heico-lapchole-seed20260707/sft_val_overlay.clip_valid.jsonl \
  --output-dir ~/workspace/focus-runs/lora-sft/qwen3vl-4b-sft-v2-heico-lapchole-smoke128 \
  --max-train-samples 128 \
  --max-val-samples 32 \
  --epochs 1 \
  --video-min-frames 4 \
  --video-max-frames 64 \
  --gradient-accumulation-steps 4
```

### Step 6：正式训练 5 epochs

```bash
source ~/tools/miniconda3/etc/profile.d/conda.sh && conda activate orena-focus && cd ~/workspace/VLM-Competition && CUDA_VISIBLE_DEVICES=0 python scripts/train_qwen3vl_lora_sft_smoke.py \
  --train-jsonl ~/workspace/focus-runs/data-audit/clip-window-audit-heico-lapchole-seed20260707/sft_train_overlay.clip_valid.jsonl \
  --val-jsonl ~/workspace/focus-runs/data-audit/clip-window-audit-heico-lapchole-seed20260707/sft_val_overlay.clip_valid.jsonl \
  --output-dir ~/workspace/focus-runs/lora-sft/qwen3vl-4b-sft-v2-heico-lapchole-e5-max64 \
  --max-train-samples 0 \
  --max-val-samples 0 \
  --epochs 5 \
  --learning-rate 1e-4 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --video-min-frames 4 \
  --video-max-frames 64 \
  --gradient-accumulation-steps 4
```

## 预期与风险

预期收益：

- 数据从单 HeiCo 扩展到 HeiCo + LapChole；
- 训练从 1 epoch 提升到 5 epochs；
- 官方 runtime 与训练端帧预算对齐；
- 进一步提高 object recognition、temporal grounding、aggregation 等弱项。

主要风险：

- LapChole 需要 gated access，权限未通过前不能下载；
- 5 epochs 训练时间可能显著超过上一轮 10 小时；
- 如果 LapChole 视频量很大，overlay 预处理和存储压力会明显增加；
- 如果官方 hidden pre-eval 分布更偏 OOD，仍可能需要 answer-format-aware prompt/SFT 或 balanced sampler。

## 当前建议

先执行 Step 1，确认 `lapchole` 的 SEGMENT split 是否可读；如果可读，再下载和预处理。不要直接开 5 epochs，因为目前最大的不确定性是 LapChole 权限、实际样本数和视频路径。
