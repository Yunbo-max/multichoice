#!/usr/bin/env python3
"""Generate QZ CreateJob payloads for Paper 2 TreeJev Stage 1 visual SFT.

Stage 1 = Qwen3.5-4B multimodal LoRA action SFT on the Valen Sokoban tied-optimal
action distributions (G1 smoke, then G2 full 30k x 3 seeds). Each job is single-
GPU (the trainer is single-card: ``model.to("cuda")``); submit G1 first, inspect
its log, then submit G2 only after G1 passes the nonzero-LoRA-gradient check.

Run from the repo root::

    python3 qz/make_tree_payloads.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "qz_tree"

R = "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/multichoice_publish"
V = "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/decision_compare/.venv-sii"
M = "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/decision_compare/models/Qwen3.5-4B"

TRAIN = f"{R}/runs/tree/valen_split/train_visual.jsonl"
DEV = f"{R}/runs/tree/valen_split/dev_visual.jsonl"
MODULES = f"{R}/innerjev/configs/4B/modules.json"

WORKSPACE = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
PROJECT = "project-160ccb20-98ab-4538-a847-01d1f83d5b0f"
LCG = "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"
SPEC_1GPU = "79fe954a-be92-4772-ac0b-94ad8a79b7bb"
SPEC_4GPU = "8a53ac21-299a-4dee-85e9-9c04a544cf8d"


def _base(name, description, command, max_running_time_ms="3600000", priority=4,
          spec_id=SPEC_1GPU, shm_gi=100):
    return {
        "workspace_id": WORKSPACE,
        "project_id": PROJECT,
        "logic_compute_group_id": LCG,
        "name": name,
        "description": description,
        "framework": "pytorch",
        "command": command,
        "max_running_time_ms": max_running_time_ms,
        "task_priority": priority,
        "framework_config": [
            {
                "image": "docker.sii.shaipower.online/inspire-studio/relay2:v2",
                "image_type": "SOURCE_PRIVATE",
                "instance_count": 1,
                "spec_id": spec_id,
                "shm_gi": shm_gi,
            }
        ],
    }


def _env():
    return (
        "set -uo pipefail\n"
        f"R={R}\nV={V}\nM={M}\n"
        "export HF_HUB_OFFLINE=1\nexport TRANSFORMERS_OFFLINE=1\n"
        "export PYTHONUNBUFFERED=1\nexport TOKENIZERS_PARALLELISM=false\n"
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
        f'export PYTHONPATH="{R}"\n'
    )


def _stage1(name, desc, seed, limit, dev_limit, max_steps, out,
            max_running_time_ms="3600000", gpus=1, batch_size=8):
    if gpus > 1:
        runner = f'"$V/bin/torchrun" --standalone --nnodes 1 --nproc_per_node {gpus}'
        script = f'"{R}/treejev/train_visual_action.py"'
        shm = 900
    else:
        runner = f'"$V/bin/python"'
        script = "-m treejev.train_visual_action"
        shm = 100
    command = (
        _env()
        + f'rm -rf "{out}"\n'
        + f'mkdir -p "{R}/logs/qz"\n'
        + f'cd "{R}"\n'
        + f'{runner} {script} '
        + f'--train "{TRAIN}" --dev "{DEV}" '
        + f'--model "{M}" --modules "{MODULES}" '
        + f'--output "{out}" --seed {seed} '
        + f'--limit {limit} --dev-limit {dev_limit} --max-steps {max_steps} '
        + f'--batch-size {batch_size} '
        + f'--local-files-only '
        + f'> "{R}/logs/qz/{name}.log" 2>&1\n'
        + 'status=$?; echo "JOB EXIT=$status"; exit $status\n'
    )
    spec = SPEC_4GPU if gpus > 1 else SPEC_1GPU
    return _base(name, desc, command, max_running_time_ms=max_running_time_ms,
                 spec_id=spec, shm_gi=shm)


def main():
    payloads = [
        _stage1(
            "tree-stage1-g1-smoke-s43",
            "Paper 2 Stage 1 G1: Valen 128-image LoRA smoke, nonzero-gradient check, seed 43, 1xH100",
            43, 128, 16, 8,
            f"{R}/runs/tree/stage1_g1_smoke_s43",
        ),
    ]
    for seed in (43, 44, 45):
        payloads.append(
            _stage1(
                f"tree-stage1-g2-s{seed}",
                f"Paper 2 Stage 1 G2: Valen batched SFT (batch 8, ~21.6k images), seed {seed}, 1xH100",
                seed, 0, 64, 2700,
                f"{R}/runs/tree/stage1_g2_s{seed}",
                max_running_time_ms="10800000",
                gpus=1,
                batch_size=8,
            )
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for payload in payloads:
        path = OUT_DIR / (payload["name"] + ".json")
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {path.name}")
    print(f"\n{len(payloads)} payloads in {OUT_DIR}/")
    print("Submit G1 first; only submit G2 after G1 passes the nonzero-gradient check.")


if __name__ == "__main__":
    main()
