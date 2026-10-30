#!/usr/bin/env python3
"""Generate QZ payloads for the Paper 1 E3 learned-pair LoRA continuations.

Continues from the E2 warm BCE checkpoint with objective=pair on the full I/C/R
controlled data, learning count + dense pair heads. Single-GPU, 3 seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "qz_regime_4b_pair"

R = "/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/jev"
V = "/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/jev/.venv-sii"
TRAIN = f"{R}/runs/paper1_regime_4b/data/student/train.jsonl"
DEV = f"{R}/runs/paper1_regime_4b/data/student/dev.jsonl"
MODULES = f"{R}/innerjev/configs/4B/modules.json"
WARM = f"{R}/runs/paper1_regime_4b/warm_bce_shared/final"

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


def _base(name, description, command, max_running_time_ms):
    return {
        "workspace_id": WORKSPACE, "project_id": PROJECT,
        "logic_compute_group_id": LCG, "name": name, "description": description,
        "framework": "pytorch", "command": command,
        "max_running_time_ms": max_running_time_ms, "task_priority": 4,
        "framework_config": [{
            "image": "docker.sii.shaipower.online/inspire-studio/relay2:v2",
            "image_type": "SOURCE_PRIVATE", "instance_count": 1,
            "spec_id": SPEC_1GPU, "shm_gi": 100,
        }],
    }


def _stage1(seed, max_steps, name, out, desc):
    command = (
        _env()
        + f'test ! -e "{out}"\n'
        + f'mkdir -p "{R}/logs/qz"\n'
        + f'cd "{R}"\n'
        + f'"{V}/bin/torchrun" --standalone --nnodes 1 --nproc_per_node 1 '
        + f'"{R}/harness/train_multiselect.py" '
        + f'--model "{WARM}" --train "{TRAIN}" --dev "{DEV}" '
        + f'--modules "{MODULES}" --target gold --objective pair --pair-pairs 28 '
        + f'--set-loss set_ce --count-anchor --global-batch 64 --epochs 1 '
        + f'--eval-every-frac 1.0 --lr 2e-05 --lora --lora-r 16 --lora-alpha 32 '
        + f'--seed {seed} --max-steps {max_steps} --output "{out}" '
        + f'--max-input-tokens 8192 --max-batch-tokens 8192 '
        + f'--expect-trainable 0 '
        + f'> "{R}/logs/qz/{name}.log" 2>&1\n'
        + 'status=$?; echo "JOB EXIT=$status"; exit $status\n'
    )
    return _base(name, desc, command, "5400000")


def main():
    payloads = [
        _stage1(43, 10, "regime-pair-smoke-s43",
                f"{R}/runs/paper1_regime_4b/pair_smoke_s43",
                "Paper 1 E3 pair smoke: 10 steps, seed 43"),
    ]
    for seed in (43, 44, 45):
        payloads.append(_stage1(
            seed, 100, f"regime-pair-s{seed}",
            f"{R}/runs/paper1_regime_4b/pair/seed_{seed}",
            f"Paper 1 E3 learned-pair continuation: 100 steps, seed {seed}"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for payload in payloads:
        path = OUT_DIR / (payload["name"] + ".json")
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(f"wrote {path.name}")
    print(f"\n{len(payloads)} payloads in {OUT_DIR}/")
    print("Submit smoke FIRST; then the 3 seed jobs after smoke succeeds.")


if __name__ == "__main__":
    main()
