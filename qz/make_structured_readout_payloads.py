#!/usr/bin/env python3
"""Generate qz payloads for the P0 structured-readout eval (3 seeds)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "qz_structured"

R = "/inspire/hdd/global_user/zhangjiaquan-253108540222/jev"
V = "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/decision_compare/.venv-sii"
MODEL = f"{R}/runs/qz/grouped_bce_4gpu_s44/v1/final"
DATA = f"{R}/runs/episodic_data_shift_v5"
HEAD = f"{R}/runs/episodic_shift_qwen_v6_s"

WORKSPACE = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
PROJECT = "project-160ccb20-98ab-4538-a847-01d1f83d5b0f"
LCG = "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"
SPEC_ID = "8a53ac21-299a-4dee-85e9-9c04a544cf8d"


def make(seed):
    out = f"{HEAD}{seed}/structured_eval.json"
    command = (
        "set -uo pipefail\n"
        f"R={R}\nV={V}\nMODEL={MODEL}\n"
        "export HF_HUB_OFFLINE=1\nexport TRANSFORMERS_OFFLINE=1\n"
        "export PYTHONUNBUFFERED=1\nexport TOKENIZERS_PARALLELISM=false\n"
        f'mkdir -p "{R}/logs/qz"\ncd "{R}"\n'
        f'test -f "{HEAD}{seed}/best.pt" && test -f "{DATA}/test.jsonl" || '
        '{ echo "missing head or test episodes"; exit 3; }\n'
        f'"$V/bin/python" harness/eval_structured_readout.py '
        f'--episodes "{DATA}/test.jsonl" --head "{HEAD}{seed}/best.pt" '
        f'--qwen-model "$MODEL" --model-identity BCE_S44_FINAL --tau 1.0 --device cuda '
        f'--output "{out}" > "{R}/logs/qz/structured_readout_s{seed}.log" 2>&1\n'
        'status=$?; echo "JOB EXIT=$status"; exit $status\n'
    )
    return {
        "workspace_id": WORKSPACE,
        "project_id": PROJECT,
        "logic_compute_group_id": LCG,
        "name": f"episodic-structured-readout-s{seed}",
        "description": "P0 structured readout: raw 21-subset set-MAP vs 6-successor softmax on frozen v4 checkpoint",
        "framework": "pytorch",
        "command": command,
        "max_running_time_ms": "3600000",
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
        path = OUT_DIR / f"episodic-structured-readout-s{seed}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
