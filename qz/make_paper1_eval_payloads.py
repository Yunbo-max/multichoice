#!/usr/bin/env python3
"""Generate QZ CreateJob payloads for the frozen Paper 1 Test evaluation.

Evaluates each of the 24 pre-registered terminal checkpoints once on
``data/multiselect_grouped_multi/test.jsonl`` at the frozen threshold 0.5.
One single-GPU job per arm x seed. No selection is performed on Test.

Run from the repo root::

    python3 qz/make_paper1_eval_payloads.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "qz_static_eval"
MANIFEST = ROOT / "paper_static" / "runs_4b.jsonl"

R = "/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/jev"
V = "/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/jev/.venv-sii"
TEST = f"{R}/data/multiselect_grouped_multi/test.jsonl"

WORKSPACE = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
PROJECT = "project-160ccb20-98ab-4538-a847-01d1f83d5b0f"
LCG = "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"
SPEC_1GPU = "79fe954a-be92-4772-ac0b-94ad8a79b7bb"


def _env():
    return (
        "set -uo pipefail\n"
        f"R={R}\nV={V}\n"
        "export HF_HUB_OFFLINE=1\nexport TRANSFORMERS_OFFLINE=1\n"
        "export PYTHONUNBUFFERED=1\nexport TOKENIZERS_PARALLELISM=false\n"
        f'export PYTHONPATH="{R}"\n'
    )


def _base(name, description, command, max_running_time_ms="3600000"):
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
                "spec_id": SPEC_1GPU,
                "shm_gi": 100,
            }
        ],
    }


def main():
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    payloads = []
    for run in rows:
        ckpt = f'{R}/{run["output"]}/final'
        out = f'{R}/runs/paper_static_eval/{run["model_tag"]}/seed_{run["seed"]}/{run["arm"]}'
        name = f"paper1-eval-{run['arm']}-s{run['seed']}"
        command = (
            _env()
            + f'rm -rf "{out}"\n'
            + f'mkdir -p "{R}/logs/qz" "{R}/runs/paper_static_eval"\n'
            + f'cd "{R}"\n'
            + f'"{V}/bin/python" -m harness.eval_multiselect '
            + f'--input "{TEST}" --model "{ckpt}" '
            + f'--expected-split test --threshold 0.5 --decoder auto '
            + f'--output "{out}" '
            + f'--checkpoint-id "paper1|{run["arm"]}|seed_{run["seed"]}" '
            + f'> "{R}/logs/qz/{name}.log" 2>&1\n'
            + 'status=$?; echo "JOB EXIT=$status"; exit $status\n'
        )
        payloads.append(_base(
            name,
            f"Paper 1 frozen Test eval: {run['arm']}, seed {run['seed']}, threshold 0.5, 1xH100",
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
