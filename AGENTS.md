# Agent notes — multichoice (Yunbo-max/multichoice)

## 环境约束（务必遵守）

### 1. QZ 作业只能用单卡（gpus=1）
- DDP / torchrun 多卡在这个集群上不可靠：4×H100 DDP 在长时间训练接近结束时出现
  NCCL `ALLREDUCE` watchdog 超时（600s，`SIGABRT`），导致 job 失败。
- **以后所有 QZ CreateJob 都用 `gpus=1`，spec_id `79fe954a-be92-4772-ac0b-94ad8a79b7bb`。**
- `treejev/train_visual_action.py` 里的 DDP 分支保留但不再使用（单卡时走 `distributed=False`）。

### 2. 单 job 运行时长上限约 180 分钟
- 实测：单卡 job 跑到约 3 小时（180 分钟）会被系统强制 `job_stopped`。
- 即使 `max_running_time_ms` 设得更大，也不要超过约 170 分钟。
- 训练耗时估算要留足余量（建议预算 ≤ 175 分钟）。

### 3. 共享盘 checkout 会被正在运行的 job 读到
- QZ job 命令里 `cd $R` 后直接跑共享盘代码。提交 job 后**不要切换 git 分支**，
  否则 queuing/running 的 job 会读到错误版本的代码。
- 同时跑多个分支的 job 必须用独立代码目录（拷贝），不能共用同一个 checkout。

### 4. Valen 视觉 SFT 单卡吞吐
- `treejev/train_visual_action.py` 是单卡逐图 forward，约 0.81s/图（~6.5-7.2s/step，
  grad_accum=8）。
- 单卡 3 小时上限内最多约 1200-1300 steps（约 9.6k-10.4k 图，占 30k 的 33-35%）。
- 要跑满 30k 需 batch 化（目前未实现）——不同尺寸 Valen 图像的 batch 拼接较复杂。

## 关键路径

- 仓库：`/inspire/hdd/global_user/zhangjiaquan-253108540222/jev`
- venv：`/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/decision_compare/.venv-sii`
- 模型：`.../models/Qwen3.5-4B`（多模态 qwen3_5）
- QZ CLI：`qz train CreateJob --data "$(cat <payload>.json)"`，先 `--dry-run` 再提交。

## 两个论文分支

- Paper 1 Static SetJev：`research/static-setjev-paper-20261027`（PR #9）
- Paper 2 Episodic Tree：`research/episodic-tree-sokoban-20261027`（PR #10）
