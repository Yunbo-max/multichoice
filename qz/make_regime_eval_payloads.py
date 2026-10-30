#!/usr/bin/env python3
"""Generate QZ payloads to infer the 12 regime continuations on controlled DEV.

For each of the 4 arms x 3 seeds, run harness/eval_multiselect on the controlled
DEV student data, producing saved predictions that regime_eval.py then scores
against the oracle distributions. DEV only; never TEST.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "qz_regime_4b_eval"

R = "/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/jev"
V = "/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/jev/.venv-sii"
DEV = f"{R}/runs/paper1_regime_4b/data/student/dev.jsonl"

ARMS = ("bce_continue", "set_ce_independent", "set_ce_anchor", "set_ce_anchor_detach")
SEEDS = (43, 44, 45)

WORKSPACE = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
PROJECT = "project-160ccb20-98ab-4538-a847-01d1f83d5b0f"
LCG = "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"
SPEC_1GPU = "79fe954a-be92-4772-ac0b-94ad8a79b7bb"


def _env():
    return (
        "set -euo pipefail\n"
        f"R={R}\nV={V}\n"
        "export HF_HUB_OFFLINE=1\nexport TRANSFORMERS_OFFLINE=1\n"
        "export PYTHONUNBUFFERED=1\nexport TOKENIZERS_PARALLELISM=false\n"
        f'export PYTHONPATH="{R}"\n'
    )


def _base(name, description, command):
    return {
        "workspace_id": WORKSPACE, "project_id": PROJECT,
        "logic_compute_group_id": LCG, "name": name, "description": description,
        "framework": "pytorch", "command": command,
        "max_running_time_ms": "5400000", "task_priority": 4,
        "framework_config": [{
            "image": "docker.sii.shaipower.online/inspire-studio/relay2:v2",
            "image_type": "SOURCE_PRIVATE", "instance_count": 1,
            "spec_id": SPEC_1GPU, "shm_gi": 100,
        }],
    }


def main():
    payloads = []
    for seed in SEEDS:
        for arm in ARMS:
            ckpt = f"{R}/runs/paper1_regime_4b/matched/seed_{seed}/{arm}/final"
            out = f"{R}/runs/paper1_regime_4b/eval/seed_{seed}/{arm}"
            name = f"regime-eval-{arm}-s{seed}"
            command = (
                _env()
                + f'test ! -e "{out}"\n'
                + f'mkdir -p "{R}/logs/qz" "{R}/runs/paper1_regime_4b/eval"\n'
                + f'cd "{R}"\n'
                + f'"{V}/bin/python" -m harness.eval_multiselect '
                + f'--input "{DEV}" --model "{ckpt}" '
                + f'--expected-split dev --threshold 0.5 --decoder auto '
                + f'--output "{out}" '
                + f'--checkpoint-id "regime|{arm}|seed_{seed}" '
                + f'> "{R}/logs/qz/{name}.log" 2>&1\n'
                + 'status=$?; echo "JOB EXIT=$status"; exit $status\n'
            )
            payloads.append(_base(
                name,
                f"Paper 1 E2 controlled-DEV inference: {arm} seed {seed}, threshold 0.5",
                command,
            ))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for payload in payloads:
        path = OUT_DIR / (payload["name"] + ".json")
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(f"wrote {path.name}")
    print(f"\n{len(payloads)} payloads in {OUT_DIR}/")


if __name__ == "__main__":
    main()
