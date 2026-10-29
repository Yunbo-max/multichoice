#!/usr/bin/env python3
"""Generate qz payloads for the DAgger round (s43): collect policy cache, merge, retrain."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "qz_dagger"

R = "/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/jev"
V = "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/decision_compare/.venv-sii"
MODEL = f"{R}/runs/qz/grouped_bce_4gpu_s44/v1/final"
DATA = f"{R}/runs/episodic_data_shift_v5"
HEAD = f"{R}/runs/episodic_shift_qwen_v6_s43/best.pt"
OUT = f"{R}/runs/episodic_dagger_s43"

WORKSPACE = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
PROJECT = "project-160ccb20-98ab-4538-a847-01d1f83d5b0f"
LCG = "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"
SPEC_ID = "8a53ac21-299a-4dee-85e9-9c04a544cf8d"


def _base(name, description, command, max_running_time_ms="7200000"):
    return {
        "workspace_id": WORKSPACE,
        "project_id": PROJECT,
        "logic_compute_group_id": LCG,
        "name": name,
        "description": description,
        "framework": "pytorch",
        "command": command,
        "max_running_time_ms": max_running_time_ms,
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


def _env():
    return (
        "set -uo pipefail\n"
        f"R={R}\nV={V}\nMODEL={MODEL}\n"
        "export HF_HUB_OFFLINE=1\nexport TRANSFORMERS_OFFLINE=1\n"
        "export PYTHONUNBUFFERED=1\nexport TOKENIZERS_PARALLELISM=false\n"
    )


def collect_job():
    command = (
        _env()
        + f'mkdir -p "{R}/logs/qz" "{OUT}"\ncd "{R}"\n'
        + f'test -f "{HEAD}" && test -f "{DATA}/train.jsonl" || '
        + '{ echo "missing head or train episodes"; exit 3; }\n'
        + f'"$V/bin/python" harness/collect_policy_cache.py '
        + f'--episodes "{DATA}/train.jsonl" --head "{HEAD}" '
        + f'--qwen-model "$MODEL" --model-identity BCE_S44_FINAL --device cuda '
        + f'--output "{OUT}/policy_cache.pt" '
        + f'> "{R}/logs/qz/dagger_collect_s43.log" 2>&1\n'
        + 'status=$?; echo "JOB EXIT=$status"; exit $status\n'
    )
    return _base(
        "episodic-dagger-collect-s43",
        "DAgger step 1: collect the v6-s43 model's on-policy rollout as a training cache",
        command,
        max_running_time_ms="10800000",
    )


def train_job():
    command = (
        _env()
        + f'mkdir -p "{R}/logs/qz"\ncd "{R}"\n'
        + f'test -f "{OUT}/policy_cache.pt" && test -f "{DATA}/train.pt" && test -f "{DATA}/dev.pt" || '
        + '{ echo "missing policy cache; run dagger-collect first"; exit 3; }\n'
        + f'"$V/bin/python" harness/merge_caches.py '
        + f'--expert "{DATA}/train.pt" --policy "{OUT}/policy_cache.pt" '
        + f'--output "{OUT}/merged_train.pt"\n'
        + f'"$V/bin/python" harness/train_episodic_features.py '
        + f'--train-cache "{OUT}/merged_train.pt" --dev-cache "{DATA}/dev.pt" '
        + f'--output "{OUT}/checkpoint" --device cuda --seed 43 '
        + f"--epochs 5 --inner-steps 4 --latent-dim 128 --rule-weight 0.2 "
        + f'> "{R}/logs/qz/dagger_train_s43.log" 2>&1\n'
        + 'status=$?; echo "JOB EXIT=$status"; exit $status\n'
    )
    return _base(
        "episodic-dagger-train-s43",
        "DAgger step 2: merge expert + policy cache and retrain the memory head",
        command,
        max_running_time_ms="10800000",
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for payload in (collect_job(), train_job()):
        path = OUT_DIR / (payload["name"] + ".json")
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
