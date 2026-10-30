#!/usr/bin/env python3
"""Generate QZ CreateJob payloads for the Paper 1 static SetJev 4B matrix.

Reads the pre-registered manifest ``paper_static/runs_4b.jsonl`` (24 rows =
8 arms x 3 seeds) and emits one QZ payload per row under ``qz_static_paper/``.

The command for each row is rebuilt from the manifest ``argv`` with absolute
paths (shared-disk ``R`` root) and the shared venv ``V/bin/torchrun``. No job is
submitted here; dry-run first, then live with ``CONFIRM_SUBMIT=1``.

Run from the repo root::

    python3 qz/make_static_paper_payloads.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "qz_static_paper"
MANIFEST = ROOT / "paper_static" / "runs_4b.jsonl"

R = "/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/jev"
V = "/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/jev/.venv-sii"
M = "/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B"

WORKSPACE = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
PROJECT = "project-160ccb20-98ab-4538-a847-01d1f83d5b0f"
LCG = "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"
SPEC_ID = "8a53ac21-299a-4dee-85e9-9c04a544cf8d"  # 4 x H100

PATH_ARGS = {"--train", "--dev", "--model", "--modules", "--output"}
QUOTE_ARGS = {"--init-checkpoint-id"}


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
        f"R={R}\nV={V}\nM={M}\n"
        "export HF_HUB_OFFLINE=1\nexport TRANSFORMERS_OFFLINE=1\n"
        "export PYTHONUNBUFFERED=1\nexport TOKENIZERS_PARALLELISM=false\n"
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
    )


def _rebuild_argv(argv):
    """Convert the manifest argv (relative paths, bare torchrun) into an
    absolute-path shell command using the shared venv."""
    tokens = []
    i = 0
    n = len(argv)
    while i < n:
        tok = argv[i]
        if tok == "torchrun":
            tokens.append(f'"{V}/bin/torchrun"')
        elif tok == "harness/train_multiselect.py":
            tokens.append(f'"{R}/harness/train_multiselect.py"')
        elif tok in PATH_ARGS and i + 1 < n:
            tokens.append(tok)
            tokens.append(f'"{R}/{argv[i + 1]}"')
            i += 1
        elif tok in QUOTE_ARGS and i + 1 < n:
            tokens.append(tok)
            tokens.append(f'"{argv[i + 1]}"')
            i += 1
        else:
            tokens.append(tok)
        i += 1
    return " ".join(tokens)


def _slug(run):
    return run["id"].replace("/", "-")


def main():
    if not MANIFEST.is_file():
        raise SystemExit(f"missing manifest: {MANIFEST}")
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit("empty manifest")
    payloads = []
    for run in rows:
        out = f'{R}/{run["output"]}'
        log = f'{R}/logs/qz/paper1_{_slug(run)}.log'
        command = (
            _env()
            + f'mkdir -p "{R}/logs/qz" "{out}"\n'
            + f'cd "{R}"\n'
            + _rebuild_argv(run["argv"])
            + f' > "{log}" 2>&1\n'
            + 'status=$?; echo "JOB EXIT=$status"; exit $status\n'
        )
        name = f"paper1-{run['arm']}-s{run['seed']}"
        description = (
            f"Paper 1 static SetJev: {run['arm']}, seed {run['seed']}, "
            f"4xH100, {run['update_budget']} steps, terminal-only DEV"
        )
        payloads.append(_base(name, description, command))

    if len({p["name"] for p in payloads}) != len(payloads):
        raise SystemExit("duplicate payload names")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for payload in payloads:
        path = OUT_DIR / (payload["name"] + ".json")
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {path.name}")
    print(f"\n{len(payloads)} payloads in {OUT_DIR}/")


if __name__ == "__main__":
    main()
