#!/usr/bin/env python3
"""Generate SetJev v2 Gate 1 + Gate 3 qz payloads.

Gate 1 (docs/setjev_v2_plan.md section 3): readout audit on a fixed Gold-BCE
checkpoint, comparing native option logits vs candidate Yes/No readout.

Gate 3 (section 4, modification 2): residual count-head ablation on gold targets,
same seed/data/recipe, with and without ``--count-anchor``, for ``set_ce`` and
``induced_marginal``.

Run from the repo root::

    python3 qz/make_setjev_v2_payloads.py

Writes JSON files into ``qz_setjev_v2/``. Submit them with
``qz/submit_setjev_v2.sh`` (dry-run by default; live requires CONFIRM_SUBMIT=1).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "qz_setjev_v2"

R = "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/multichoice_publish"
V = "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/decision_compare/.venv-sii"
M = "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/decision_compare/models/Qwen3.5-4B"
D5K = f"{R}/data/multiselect_grouped_multi_5k"
D_FULL = f"{R}/data/multiselect_grouped_multi"
CKPT = f"{R}/runs/qz/grouped_bce_4gpu_s44/v1/final"

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
        f"R={R}\nV={V}\nM={M}\n"
        "export HF_HUB_OFFLINE=1\nexport TRANSFORMERS_OFFLINE=1\n"
        "export PYTHONUNBUFFERED=1\nexport TOKENIZERS_PARALLELISM=false\n"
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
    )


def gate1_readout_audit():
    out = f"{R}/runs/qz/readout_audit_gate1_s44"
    command = (
        _env()
        + f'mkdir -p "{R}/logs/qz" "{out}"\n'
        + f'cd "{R}"\n'
        + f'"$V/bin/python" -I "{R}/harness/readout_audit.py" '
        + f'--input "{D_FULL}/dev.jsonl" --model "{CKPT}" --output "{out}" '
        + "--limit 300 --limit-mode stratified --sample-seed 43 --device cuda --batch-size 8 "
        + f'--max-input-tokens 8192 > "{R}/logs/qz/readout_audit_gate1_s44.log" 2>&1\n'
        + 'status=$?; echo "JOB EXIT=$status"; exit $status\n'
    )
    return _base(
        "readout-audit-gate1-bce-s44",
        "Gate 1 readout audit: native option logits vs candidate Yes/No on grouped_bce_4gpu_s44/final, stratified dev 300",
        command,
        max_running_time_ms="3600000",
    )


def _train(
    name, desc, set_loss, count_anchor, seed=43, data=D5K, gpus=4, eval_frac=0.2
):
    out = f"{R}/runs/qz/{name}/v1"
    anchor_flag = " --count-anchor" if count_anchor else ""
    anchor_slug = "anchor" if count_anchor else "noanchor"
    command = (
        _env()
        + f'D={data}\nDEV="{D_FULL}/dev.jsonl"\nOUT={out}\nLOG={R}/logs/qz/{name}.log\n'
        + f'mkdir -p "{R}/logs/qz" "$OUT"\ncd "{R}"\n'
        + f'"$V/bin/python" -I "{R}/harness/train_multiselect.py" --validate-only '
        + f'--train "$D/train.jsonl" --dev "$DEV" --model "$M" '
        + f'--modules "{R}/innerjev/configs/4B/modules.json" '
        + f"--target gold --objective set --set-loss {set_loss} --count-weight 1 "
        + f"--world-size {gpus} --global-batch 64 --max-input-tokens 8192 --max-batch-tokens 8192 "
        + f'{anchor_flag} --validate-output "{R}/logs/qz/{name}_validate.json" '
        + f'> "{R}/logs/qz/{name}_validate.log" 2>&1 '
        + f'|| {{ echo "validation failed"; tail -60 "{R}/logs/qz/{name}_validate.log"; exit 3; }}\n'
        + f'"$V/bin/torchrun" --standalone --nnodes 1 --nproc_per_node {gpus} '
        + f'"{R}/harness/train_multiselect.py" '
        + f'--train "$D/train.jsonl" --dev "$DEV" --model "$M" '
        + f'--modules "{R}/innerjev/configs/4B/modules.json" '
        + f'--target gold --objective set --set-loss {set_loss} --count-weight 1 --output "$OUT" '
        + f"--lr 2e-6 --global-batch 64 --epochs 1 --max-input-tokens 8192 --max-batch-tokens 8192 "
        + f"--eval-every-frac {eval_frac} --lambda-g 0 --graph none --seed {seed} "
        + f'{anchor_flag} --expect-trainable 3313500160 > "$LOG" 2>&1\n'
        + 'status=$?; echo "JOB EXIT=$status"; exit $status\n'
    )
    return _base(name, desc, command)


def main():
    payloads = [
        gate1_readout_audit(),
        _train(
            "gate3-set_ce-noanchor-5k-s43",
            "Gate 3 control: gold set_ce, 5k, seed 43, full count head (no anchor), 4xH100",
            "set_ce",
            False,
        ),
        _train(
            "gate3-set_ce-anchor-5k-s43",
            "Gate 3: gold set_ce + raw-logit count anchor (base start, not BCE warm start), 5k, seed 43, 4xH100",
            "set_ce",
            True,
        ),
        _train(
            "gate3-induced-noanchor-5k-s43",
            "Gate 3 control: gold induced_marginal, 5k, seed 43, full count head (no anchor), 4xH100",
            "induced_marginal",
            False,
        ),
        _train(
            "gate3-induced-anchor-5k-s43",
            "Gate 3: gold induced_marginal + raw-logit count anchor (base start, not BCE warm start), 5k, seed 43, 4xH100",
            "induced_marginal",
            True,
        ),
    ]
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
