#!/usr/bin/env python3
"""Generate qz payloads for the episodic Qwen pipeline (gen -> cache -> train).

Stage 1 ``episodic-cache``: generate textual rule-retention/rule-change episodes
and encode them into frozen-Qwen chronological feature caches (train + dev).

Stage 2 ``episodic-train-s{seed}``: train the persistent-memory head on the cached
features. The carry-vs-reset comparison uses the SAME trained weights (identical
compute/decoder budget), so a single run yields both numbers.

Run:  python3 qz/make_episodic_payloads.py [43 44 45]
Writes JSON payloads into qz_episodic/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "qz_episodic"

R = "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/multichoice_publish"
V = "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/decision_compare/.venv-sii"
MODEL = f"{R}/runs/qz/grouped_bce_4gpu_s44/v1/final"
DATA = f"{R}/runs/episodic_data"

WORKSPACE = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
PROJECT = "project-160ccb20-98ab-4538-a847-01d1f83d5b0f"
LCG = "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"
SPEC_ID = "8a53ac21-299a-4dee-85e9-9c04a544cf8d"


def _base(name, description, command, max_running_time_ms="7200000", priority=4):
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
                "spec_id": SPEC_ID,
                "shm_gi": 900,
            }
        ],
    }


def _env():
    return (
        "set -euo pipefail\n"
        f"R={R}\nV={V}\nMODEL={MODEL}\n"
        "export HF_HUB_OFFLINE=1\nexport TRANSFORMERS_OFFLINE=1\n"
        "export PYTHONUNBUFFERED=1\nexport TOKENIZERS_PARALLELISM=false\n"
    )


def cache_job():
    command = (
        _env()
        + f'mkdir -p "{DATA}" "{R}/logs/qz"\ncd "{R}"\n'
        + 'grep -q "oracle_rule_before_decision" harness/gen_text_episodes.py || '
        + '{ echo "STALE CODE: causal generator fix missing; abort"; exit 9; }\n'
        + f'"$V/bin/python" harness/gen_text_episodes.py '
        + f'--output "{DATA}/train.jsonl" --episodes 1024 --horizon 6 '
        + f"--rule-switch-prob 0.25 --seed 43\n"
        + f'"$V/bin/python" harness/gen_text_episodes.py '
        + f'--output "{DATA}/dev.jsonl" --episodes 256 --horizon 6 '
        + f"--rule-switch-prob 0.25 --seed 3043\n"
        + f'"$V/bin/python" harness/build_episodic_features.py '
        + f'--input "{DATA}/train.jsonl" --model "$MODEL" '
        + f'--model-identity BCE_S44_FINAL --output "{DATA}/train.pt" --device cuda '
        + f'> "{R}/logs/qz/episodic_cache_train.log" 2>&1 || '
        + f'{{ echo "train cache failed"; tail -40 "{R}/logs/qz/episodic_cache_train.log"; exit 3; }}\n'
        + f'"$V/bin/python" harness/build_episodic_features.py '
        + f'--input "{DATA}/dev.jsonl" --model "$MODEL" '
        + f'--model-identity BCE_S44_FINAL --output "{DATA}/dev.pt" --device cuda '
        + f'> "{R}/logs/qz/episodic_cache_dev.log" 2>&1 || '
        + f'{{ echo "dev cache failed"; tail -40 "{R}/logs/qz/episodic_cache_dev.log"; exit 3; }}\n'
        + 'status=$?; echo "JOB EXIT=$status"; exit $status\n'
    )
    return _base(
        "episodic-cache-bce-s44",
        "Generate textual rule-retention/rule-change episodes and encode frozen-Qwen chronological feature caches (train+dev)",
        command,
        max_running_time_ms="10800000",
    )


def train_job(seed):
    out = f"{R}/runs/episodic_qwen_s{seed}"
    command = (
        _env()
        + f'mkdir -p "{R}/logs/qz"\ncd "{R}"\n'
        + f'test -f "{DATA}/train.pt" && test -f "{DATA}/dev.pt" || '
        + f'{{ echo "cache missing; run episodic-cache first"; exit 3; }}\n'
        + f'"$V/bin/python" harness/train_episodic_features.py '
        + f'--train-cache "{DATA}/train.pt" --dev-cache "{DATA}/dev.pt" '
        + f'--output "{out}" --device cuda --seed {seed} '
        + f"--epochs 5 --inner-steps 4 --latent-dim 128 "
        + f'> "{R}/logs/qz/episodic_train_s{seed}.log" 2>&1\n'
        + 'status=$?; echo "JOB EXIT=$status"; exit $status\n'
    )
    return _base(
        f"episodic-train-s{seed}",
        "Train persistent-memory head on cached Qwen episodes; carry vs reset eval uses identical weights",
        command,
        max_running_time_ms="3600000",
    )


def main():
    seeds = [int(seed) for seed in sys.argv[1:]] or [43, 44, 45]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payloads = [cache_job()] + [train_job(seed) for seed in seeds]
    for payload in payloads:
        path = OUT_DIR / (payload["name"] + ".json")
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {path.name}")
    print(f"\n{len(payloads)} payloads in {OUT_DIR}/")


if __name__ == "__main__":
    main()
