#!/usr/bin/env python3
"""Generate qz payloads for the shift-by-k (v4) episodic pipeline.

Harder than v3: 7 cells, cyclic shift by k (6 rules, memory stores log2 6 bits
not 1 bit), C(7,2)=21 candidate pairs, horizon 8 (train/dev) and 12 (test).

Stage 1 ``episodic-shift-cache``: generate episodes + frozen-Qwen feature caches.
Stage 2 ``episodic-shift-train-s{seed}``: train the persistent-memory head.
Stage 3 ``episodic-shift-online-s{seed}``: on-policy carry vs reset eval.

Run: python3 qz/make_episodic_shift_payloads.py [43 44 45]  -> qz_episodic_shift/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "qz_episodic_shift"

R = "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/multichoice_publish"
V = "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/decision_compare/.venv-sii"
MODEL = f"{R}/runs/qz/grouped_bce_4gpu_s44/v1/final"
DATA = f"{R}/runs/episodic_data_shift_v5"

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
        "set -uo pipefail\n"
        f"R={R}\nV={V}\nMODEL={MODEL}\n"
        "export HF_HUB_OFFLINE=1\nexport TRANSFORMERS_OFFLINE=1\n"
        "export PYTHONUNBUFFERED=1\nexport TOKENIZERS_PARALLELISM=false\n"
    )


def cache_job():
    command = (
        _env()
        + f'mkdir -p "{DATA}" "{R}/logs/qz"\ncd "{R}"\n'
        + 'grep -q "cyclic7_shift2_v4" harness/gen_text_episodes_shift.py || '
        + '{ echo "STALE CODE: shift generator missing"; exit 9; }\n'
        + f'"$V/bin/python" harness/gen_text_episodes_shift.py '
        + f'--output "{DATA}/train.jsonl" --episodes 1024 --horizon 8 '
        + f"--rule-switch-prob 0.25 --seed 43\n"
        + f'"$V/bin/python" harness/gen_text_episodes_shift.py '
        + f'--output "{DATA}/dev.jsonl" --episodes 256 --horizon 8 '
        + f"--rule-switch-prob 0.25 --seed 3043\n"
        + f'"$V/bin/python" harness/gen_text_episodes_shift.py '
        + f'--output "{DATA}/test.jsonl" --episodes 256 --horizon 12 '
        + f"--rule-switch-prob 0.25 --seed 9043\n"
        + f'"$V/bin/python" harness/build_episodic_features.py '
        + f'--input "{DATA}/train.jsonl" --model "$MODEL" '
        + f'--model-identity BCE_S44_FINAL --output "{DATA}/train.pt" --device cuda '
        + f'> "{R}/logs/qz/episodic_shift_cache_train.log" 2>&1 || '
        + f'{{ echo "train cache failed"; tail -40 "{R}/logs/qz/episodic_shift_cache_train.log"; exit 3; }}\n'
        + f'"$V/bin/python" harness/build_episodic_features.py '
        + f'--input "{DATA}/dev.jsonl" --model "$MODEL" '
        + f'--model-identity BCE_S44_FINAL --output "{DATA}/dev.pt" --device cuda '
        + f'> "{R}/logs/qz/episodic_shift_cache_dev.log" 2>&1 || '
        + f'{{ echo "dev cache failed"; tail -40 "{R}/logs/qz/episodic_shift_cache_dev.log"; exit 3; }}\n'
        + 'status=$?; echo "JOB EXIT=$status"; exit $status\n'
    )
    return _base(
        "episodic-shift-cache-bce-s44",
        "Generate shift-by-k episodes (7 cells, 6 rules, horizon 8/12) and frozen-Qwen feature caches",
        command,
        max_running_time_ms="10800000",
    )


def train_job(seed):
    out = f"{R}/runs/episodic_shift_qwen_v5_s{seed}"
    command = (
        _env()
        + f'mkdir -p "{R}/logs/qz"\ncd "{R}"\n'
        + f'test -f "{DATA}/train.pt" && test -f "{DATA}/dev.pt" || '
        + '{ echo "cache missing; run episodic-shift-cache first"; exit 3; }\n'
        + f'"$V/bin/python" harness/train_episodic_features.py '
        + f'--train-cache "{DATA}/train.pt" --dev-cache "{DATA}/dev.pt" '
        + f'--output "{out}" --device cuda --seed {seed} '
        + f"--epochs 5 --inner-steps 4 --latent-dim 128 --rule-weight 1.0 "
        + f'> "{R}/logs/qz/episodic_shift_train_s{seed}.log" 2>&1\n'
        + 'status=$?; echo "JOB EXIT=$status"; exit $status\n'
    )
    return _base(
        f"episodic-shift-train-s{seed}",
        "Train persistent-memory head on shift-by-k cached Qwen episodes (carry vs reset uses identical weights)",
        command,
        max_running_time_ms="3600000",
    )


def online_job(seed):
    out = f"{R}/runs/episodic_shift_qwen_v5_s{seed}"
    command = (
        _env()
        + f'mkdir -p "{R}/logs/qz"\ncd "{R}"\n'
        + f'test -f "{out}/best.pt" && test -f "{DATA}/test.jsonl" || '
        + '{ echo "missing head or held-out test episodes"; exit 3; }\n'
        + f'"$V/bin/python" harness/eval_episodic_text_online.py '
        + f'--episodes "{DATA}/test.jsonl" --head "{out}/best.pt" '
        + f'--qwen-model "$MODEL" --model-identity BCE_S44_FINAL --device cuda '
        + f'--output "{out}/online_eval.json" '
        + f'> "{R}/logs/qz/episodic_shift_online_s{seed}.log" 2>&1\n'
        + 'status=$?; echo "JOB EXIT=$status"; exit $status\n'
    )
    return _base(
        f"episodic-shift-online-s{seed}",
        "On-policy carry vs reset eval on shift-by-k held-out test episodes (horizon 12)",
        command,
        max_running_time_ms="3600000",
    )


def main():
    seeds = [int(seed) for seed in sys.argv[1:]] or [43, 44, 45]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payloads = (
        [cache_job()] + [train_job(s) for s in seeds] + [online_job(s) for s in seeds]
    )
    for payload in payloads:
        path = OUT_DIR / (payload["name"] + ".json")
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {path.name}")
    print(f"\n{len(payloads)} payloads in {OUT_DIR}/")


if __name__ == "__main__":
    main()
