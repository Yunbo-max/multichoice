# 8×H100 复现顺序（SII-Decision / StartLux-Decision）

> 2026-10-19. 所有路径都在共享盘，QZ 任务和登录节点看到的是同一份。
> 当前 `qz` CLI 只在用户自己的终端里；在 Claude 容器里 `command -v qz` 为空。

## 0. 提交前

```bash
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/decision_compare
export R=$PWD
```

- 8 卡 spec：`7166bd2e-6cbe-4bd9-be38-762d11003e7f`（8×H100 80GB，160 核，1800 GiB）
- 4 卡 spec：`8a53ac21-299a-4dee-85e9-9c04a544cf8d`（4×H100，900 GiB）
- 每个 payload 都是独立 JSON；先 `--dry-run`，确认后再去掉 `--dry-run`。

通用提交形式：

```bash
qz train CreateJob --dry-run --data "$(cat <payload>.json)"
qz train CreateJob --data "$(cat <payload>.json)"
```

## 1. SII 4B 利用率 benchmark（先跑这个）

覆盖 8×1 b4/b8/b16 + 4×2 b8；输出 `runs/qz/sii_4b_bench/profile.json`（items/s、tokens/s、padding 效率、nvidia-smi 利用率）。

```bash
qz train CreateJob --dry-run --data "$(cat qz/sii_4b_benchmark.json)"
qz train CreateJob --data "$(cat qz/sii_4b_benchmark.json)"
```

预计 15–30 min。跑完后看 `profile.json`，据此决定 4B eval 的 `BATCH_SIZE`/`SHARDS`。

## 2. SII 4B JEval（8 shards）

```bash
qz train CreateJob --dry-run --data "$(cat qz/sii_4b_eval.json)"
qz train CreateJob --data "$(cat qz/sii_4b_eval.json)"
```

输出：`runs/qz/sii_4b_jeval/{predictions.jsonl,summary.json,score.txt,profile.json}`。
目标论文值：ACC 71.92 / ECE 0.0843 / Brier 0.383（10,938 choice+noul）。
预计 3–8 min（8×H100，含模型加载）。

## 3. StartLux 4B JEval（8 replicas）

```bash
qz train CreateJob --dry-run --data "$(cat scripts/qz/startlux_4b_jeval_createjob.json)"
qz train CreateJob --data "$(cat scripts/qz/startlux_4b_jeval_createjob.json)"
```

输出：`logs/qz/startlux_4b/{predictions.jsonl,metrics.json,gpu_util.csv,score_summary.json}`。
预计 20–60 min（含 thought 生成；AutoThink gate 默认按发布设置）。

> 如果 4B benchmark 显示 GPU 利用率不够，先跑 `harness/startlux_tune.py` 的并发 sweep（`--launch-mode direct --max-num-seqs 64/128`），再把 `CONCURRENCY`/`MAX_NUM_SEQS` 写进 QZ 命令。

## 4. SII 27B benchmark + JEval（8 cards）

```bash
qz train CreateJob --dry-run --data "$(cat qz/sii_27b_benchmark.json)"
qz train CreateJob --data "$(cat qz/sii_27b_benchmark.json)"

qz train CreateJob --dry-run --data "$(cat qz/sii_27b_eval.json)"
qz train CreateJob --data "$(cat qz/sii_27b_eval.json)"
```

输出：`runs/qz/sii_27b_bench/profile.json`、`runs/qz/sii_27b_jeval/`。
目标：ACC 79.24 / ECE 0.0149 / Brier 0.279。benchmark 20–50 min，eval 10–30 min。
27B 最长 82,749 token；harness 会把超长行单独跑，OOM 会记录该行而不是丢 batch。

## 5. StartLux 27B JEval（等权重下完）

权重下载：`bash scripts/qz/download_startlux_27b.sh`（可续传；当前约 12/55 GB）。
下完后：

```bash
qz train CreateJob --dry-run --data "$(cat scripts/qz/startlux_27b_jeval_createjob.json)"
qz train CreateJob --data "$(cat scripts/qz/startlux_27b_jeval_createjob.json)"
```

`MAX_MODEL_LEN=98304`，8 个 vLLM replica；预计 0.7–2.5 h。

## 6. SII 训练（推理复现之后）

4B LoRA（官方 4 卡配置，建议用 4 卡 spec `8a53ac21-…`）：

```bash
qz train CreateJob --dry-run --data "$(cat qz/sii_4b_lora_train.json)"
qz train CreateJob --data "$(cat qz/sii_4b_lora_train.json)"
```

27B 全参（官方 8 卡 FSDP 配置）：

```bash
qz train CreateJob --dry-run --data "$(cat qz/sii_27b_full_train.json)"
qz train CreateJob --data "$(cat qz/sii_27b_full_train.json)"
```

输出：`runs/qz/sii_4b_lora/v1/`、`runs/qz/sii_27b_full/v1/`。
4B LoRA 30–90 min；27B full 2–5 h + 6 个 checkpoint（每个约 55 GB），注意磁盘。

## 7. 监控与结果

```bash
qz train GetJob --job-id <job-id>
# 或（按需）
qz train ListJobs --keyword sii --page-size 10 --page-number 1 \
  --workspace-id ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6
```

共享盘结果：

- SII：`runs/qz/<run>/{predictions.jsonl,summary.json,score.txt,profile.json}`；日志 `logs/qz/*.log`
- StartLux：`logs/qz/startlux_<tag>/{predictions.jsonl,metrics.json,gpu_util.csv,score_summary.json}`；日志 `logs/qz/startlux-*.log`

统一打分器：`harness/score_jeval.py`（SII/StartLux 的 predictions 格式都接受；已用 perfect/uniform 双基线验证）。

## 8. 已知边界

- StartLux **训练不可复现**（官方只发推理代码/权重）；只能做推理评测和部署对比。
- SII 论文的 4B/27B 数字在 10,938 条 choice+noul 上；score 319 条单独报告 NMAE。
- JEVal 有 1,213 条（10.78%）超过 SII 默认 6K 上限；两边都按 100K 上限跑全量。
- SII 训练集/验证集与 JEVal 有 18+17 条完全重叠，复现报告里要注明。
- vLLM 0.31 需要 CUDA 13/torch 2.13，当前驱动不支持；StartLux 用 0.19.1（官方发布测量版本）。
