#!/usr/bin/env python3
"""Generate QZ CreateJob payloads from the Paper 1 E2 regime-pilot plan.

Reads ``runs/paper1_regime_4b/plan.json`` (1 warm BCE + 12 matched LoRA
continuations) and emits one single-GPU payload per row under ``qz_regime_4b/``.

The warm BCE job must succeed first; the 12 continuations depend on its
``warm_bce_shared/final`` checkpoint. No job is submitted here.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "qz_regime_4b"
PLAN = ROOT / "runs" / "paper1_regime_4b" / "plan.json"

R = "/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/jev"
V = "/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/jev/.venv-sii"

WORKSPACE = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
PROJECT = "project-160ccb20-98ab-4538-a847-01d1f83d5b0f"
LCG = "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"
SPEC_1GPU = "79fe954a-be92-4772-ac0b-94ad8a79b7bb"

PATH_ARGS = {"--train", "--dev", "--modules", "--output"}


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


def _abs(p):
    if p.startswith("/"):
        return p
    return f"{R}/{p}"


def _rebuild(argv):
    tokens = []
    i = 0
    n = len(argv)
    while i < n:
        tok = argv[i]
        if tok == "torchrun":
            tokens.append(f'"{V}/bin/torchrun"')
        elif tok == "harness/train_multiselect.py":
            tokens.append(f'"{R}/harness/train_multiselect.py"')
        elif tok in ("--train", "--dev", "--modules", "--output", "--model") and i + 1 < n:
            tokens.append(tok)
            tokens.append(f'"{_abs(argv[i + 1])}"')
            i += 1
        else:
            tokens.append(tok)
        i += 1
    return " ".join(tokens)


def main():
    if not PLAN.is_file():
        raise SystemExit(f"missing plan: {PLAN}")
    plan = json.loads(PLAN.read_text())
    rows = plan["rows"]
    payloads = []
    for row in rows:
        is_warm = row["phase"] == "prerequisite"
        name = f"regime-{row['id']}"
        out = _abs(row["argv"][row["argv"].index("--output") + 1])
        command = (
            _env()
            + f'test ! -e "{out}"\n'
            + f'mkdir -p "{R}/logs/qz"\n'
            + f'cd "{R}"\n'
            + _rebuild(row["argv"])
            + f' > "{R}/logs/qz/{name}.log" 2>&1\n'
            + 'status=$?; echo "JOB EXIT=$status"; exit $status\n'
        )
        budget = "10800000" if is_warm else "5400000"
        payloads.append(_base(
            name,
            f"Paper 1 E2 {'warm BCE' if is_warm else 'continuation ' + row['arm']}"
            f" seed {row['seed']}",
            command,
            budget,
        ))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for payload in payloads:
        path = OUT_DIR / (payload["name"] + ".json")
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(f"wrote {path.name}")
    print(f"\n{len(payloads)} payloads in {OUT_DIR}/")
    print("Submit regime-warm-bce-s44 FIRST; then the 12 continuations after it succeeds.")


if __name__ == "__main__":
    main()
