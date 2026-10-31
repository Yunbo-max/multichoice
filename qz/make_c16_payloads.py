#!/usr/bin/env python3
"""Generate QZ payloads for the E4 K=16 C-regime anchor-vs-BCE comparison."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "qz_c16"

R = "/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/jev"
V = "/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/jev/.venv-sii"
TRAIN = f"{R}/runs/paper1_c16/data/student/train.jsonl"
DEV = f"{R}/runs/paper1_c16/data/student/dev.jsonl"
MODULES = f"{R}/innerjev/configs/4B/modules.json"
BASE = "/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B"
WARM = f"{R}/runs/paper1_c16/warm_bce/final"

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


def _base(name, description, command, budget="7200000"):
    return {
        "workspace_id": WORKSPACE, "project_id": PROJECT,
        "logic_compute_group_id": LCG, "name": name, "description": description,
        "framework": "pytorch", "command": command,
        "max_running_time_ms": budget, "task_priority": 4,
        "framework_config": [{
            "image": "docker.sii.shaipower.online/inspire-studio/relay2:v2",
            "image_type": "SOURCE_PRIVATE", "instance_count": 1,
            "spec_id": SPEC_1GPU, "shm_gi": 100,
        }],
    }


def _train(seed, max_steps, name, out, desc, model, objective, extra):
    command = (
        _env()
        + f'test ! -e "{out}"\n'
        + f'mkdir -p "{R}/logs/qz"\n'
        + f'cd "{R}"\n'
        + f'"{V}/bin/torchrun" --standalone --nnodes 1 --nproc_per_node 1 '
        + f'"{R}/harness/train_multiselect.py" '
        + f'--model "{model}" --train "{TRAIN}" --dev "{DEV}" '
        + f'--modules "{MODULES}" --target gold --objective {objective} '
        + f'--global-batch 32 --epochs 1 --eval-every-frac 1.0 --lr 2e-05 '
        + f'--lora --lora-r 16 --lora-alpha 32 --seed {seed} '
        + f'--max-steps {max_steps} --output "{out}" '
        + f'--max-input-tokens 8192 --max-batch-tokens 16384 --expect-trainable 0 '
        + f'{extra}'
        + f'> "{R}/logs/qz/{name}.log" 2>&1\n'
        + 'status=$?; echo "JOB EXIT=$status"; exit $status\n'
    )
    return _base(name, desc, command)


def main():
    payloads = [
        _train(44, 100, "c16-warm-bce", f"{R}/runs/paper1_c16/warm_bce",
               "E4 K=16 warm BCE 100 steps", BASE, "bce", ""),
    ]
    for seed in (43, 44, 45):
        payloads.append(_train(seed, 100, f"c16-anchor-s{seed}",
                               f"{R}/runs/paper1_c16/anchor/seed_{seed}",
                               f"E4 K=16 count anchor seed {seed}", WARM, "set",
                               "--set-loss set_ce --count-anchor"))
        payloads.append(_train(seed, 100, f"c16-bce-s{seed}",
                               f"{R}/runs/paper1_c16/bce/seed_{seed}",
                               f"E4 K=16 BCE continue seed {seed}", WARM, "bce", ""))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for payload in payloads:
        path = OUT_DIR / (payload["name"] + ".json")
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(f"wrote {path.name}")
    print(f"\n{len(payloads)} payloads in {OUT_DIR}/ (1 warm + 3 anchor + 3 bce)")


if __name__ == "__main__":
    main()
