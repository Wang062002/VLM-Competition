# 项目维护协议

这个文件记录后续协作的默认规则：只要项目有实质进展，就必须同步更新项目记忆库、论文材料、workflow 和 GitHub 版本。

## 触发条件

出现以下任一情况，都视为需要更新记录：

- 跑出新的实验结果。
- 完成新的 baseline、训练、验证或 TEST 评估。
- 修改或新增脚本。
- 发现数据问题、环境问题、官方代码问题或评估流程问题。
- 形成可以写进论文的方法、实验设置、结果或结论。
- 改变下一步计划、训练策略、评估策略或数据使用策略。

## 必须更新的内容

### 1. 上下文恢复记忆库

优先更新：

- `knowledge_base/START_HERE.md`
- `knowledge_base/project_state.md`
- `knowledge_base/experiments.md`
- `knowledge_base/training_stage.md`
- `knowledge_base/workflows.md`
- `knowledge_base/maintenance_protocol.md`

目的：

- 防止上下文压缩后丢失项目状态。
- 让后续读取 `START_HERE.md` 后能快速恢复当前阶段、关键数字、下一步命令。

### 2. 论文材料

优先更新：

- `knowledge_base/paper_notes.md`
- `docs/research_log.md`
- `docs/lora_sft_training_plan.md`
- `results/*.csv`

目的：

- 保存论文中可能使用的方法描述、实验表格、数据清洗规则、分析结论。
- 区分已经被结果支持的结论和还没有被支持的假设。

### 3. 工作流与脚本说明

优先更新：

- `knowledge_base/workflows.md`
- `docs/script_workflow_explained.md`
- 对应的 `scripts/*.py`

目的：

- 保证每个脚本为什么存在、怎么运行、输入输出是什么，都能被复现。
- 新增脚本后必须解释它在整体流程中的位置。

### 4. GitHub 版本

每次完成一组稳定更新后，执行：

```bash
git status
git add .
git commit -m "<简短说明本次进展>"
git push
```

提交信息应尽量描述实际进展，例如：

- `Record full LoRA-SFT training result`
- `Add adapter TEST evaluation workflow`
- `Document script workflow in Chinese`
- `Update paper notes after full TEST evaluation`

## 记录原则

- 官方 TEST 永远保持 held-out，不记录任何将 TEST 用于训练的流程。
- 密码、token、私钥、Hugging Face token 不进入仓库。
- 大文件、数据集、模型权重、`focus-runs/` 目录不进入仓库。
- 记录结论时要注明依据：样本数、split、模型、adapter、overlay/raw、evaluator。
- 小样本结果只能写成 preliminary，不能写成最终结论。
- 论文结论必须区分：
  - 已经由实验支持。
  - 只是推测。
  - 仍需 full TEST 或 ablation 验证。

## 当前默认下一步

当前项目阶段是：

- 已完成 full clip-valid LoRA-SFT。
- 已完成 LoRA adapter 的 TEST-100 初步评估。
- 下一步是跑 full 4000-sample held-out TEST evaluation。

后续拿到 full TEST 结果后，必须更新：

- `knowledge_base/START_HERE.md`
- `knowledge_base/experiments.md`
- `knowledge_base/training_stage.md`
- `knowledge_base/paper_notes.md`
- `docs/research_log.md`
- `results/experiment_log.csv`
- `results/experiment_events.csv`
- 如有新脚本或新命令，也更新 `knowledge_base/workflows.md` 和 `docs/script_workflow_explained.md`
