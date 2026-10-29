#!/usr/bin/env python3
"""Generate qz payloads for the C experiment (on-policy LoRA Qwen episodic training)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "qz_lora"

R = "/inspire/hdd/global_user/zhangjiaquan-253108540222/jev"
V = "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/decision_compare/.venv-sii"
MODEL = f"{R}/runs/qz/grouped_bce_4gpu_s44/v1/final"
DATA = f"{R}/runs/episodic_data_shift_v5"
MODULES = f"{R}/innerjev/configs/4B/modules.json"

WORKSPACE = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
PROJECT = "project-160ccb20-98ab-4538-a847-01d1f83d5b0f"
LCG = "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"
SPEC_ID = "8a53ac21-299a-4dee-85e9-9c04a544cf8d"


def make(seed):
    out = f"{R}/runs/episodic_lora_v1_s{seed}"
    command = (
        "set -uo pipefail\n"
        f"R={R}\nV={V}\nMODEL={MODEL}\n"
        "export HF_HUB_OFFLINE=1\nexport TRANSFORMERS_OFFLINE=1\n"
        "export PYTHONUNBUFFERED=1\nexport TOKENIZERS_PARALLELISM=false\n"
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
        f'mkdir -p "{R}/logs/qz"\ncd "{R}"\n'
        f'test -f "{DATA}/train.jsonl" && test -f "{DATA}/dev.jsonl" || '
        '{ echo "missing raw shift episodes"; exit 3; }\n'
        f'"$V/bin/python" harness/train_episodic_lora.py '
        f'--train-jsonl "{DATA}/train.jsonl" --dev-jsonl "{DATA}/dev.jsonl" '
        f'--qwen-model "$MODEL" --modules "{MODULES}" '
        f'--output "{out}" --lora-r 16 --lora-alpha 32 --inner-steps 4 --latent-dim 128 '
        f"--epochs 2 --limit 200 --lr 1e-4 --accumulate-episodes 1 --seed {seed} --device cuda "
        f'> "{R}/logs/qz/episodic_lora_train_s{seed}.log" 2>&1\n'
        'status=$?; echo "JOB EXIT=$status"; exit $status\n'
    )
    return {
        "workspace_id": WORKSPACE,
        "project_id": PROJECT,
        "logic_compute_group_id": LCG,
        "name": f"episodic-lora-train-s{seed}",
        "description": "C experiment: on-policy LoRA Qwen + memory head on shift episodes",
        "framework": "pytorch",
        "command": command,
        "max_running_time_ms": "28800000",
        "task_priority": 4,
        "framework_config": [
            {
                "image": "docker.sii.shaipower.online/inspire-studio/relay2:v2",
                "image_type": "SOURCE_PRIVATE",
                "instance_count": 1,
                "spec_id": SPEC_ID,
                "shm_gi": 900,
            }
        ],
    }


def main():
    seeds = [int(s) for s in sys.argv[1:]] or [43, 44, 45]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        payload = make(seed)
        path = OUT_DIR / f"episodic-lora-train-s{seed}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
