"""Measure StartLux throughput and GPU utilization at several client concurrency levels.

The server is launched once (official or direct mode), then the same deterministic probe sample
runs at each concurrency level.  For every round the script records items/s, prefill tokens/s
(from ``usage.input_tokens``), thought tokens/s (``usage.output_tokens``), latency percentiles and
``nvidia-smi`` utilisation.  It then recommends the level with the highest prefill throughput.

This is the harness for task 3 of the reproduction.  It needs a GPU node; the CPU-only checks in
``run_cpu_checks.sh`` do not cover it.

Typical sweeps
  # 4B: one replica per GPU, 4 GPUs, release-ish flags
  python -I startlux_tune.py --model ../models/StartLux-4B-AutoThink --gpus 0,1,2,3 \
      --max-num-seqs 256 --concurrency-levels 32,64,128,256 --request-limit 200
  # 4B on all 8 GPUs (8 replicas)
  python -I startlux_tune.py --model ../models/StartLux-4B-AutoThink --gpus 0,1,2,3,4,5,6,7 ...
  # 27B: 8 replicas if one fits per H100, else tensor parallel 2 (4 servers)
  python -I startlux_tune.py --model ../models/StartLux-27B-AutoThink --gpus 0,...,7 \
      --tp-size 2 --max-num-seqs 128 ...

Hypothesis from the release (to be confirmed on the node):
  * The read-out is prefill-only, so throughput scales with the number of replicas and with
    batching long prompts; 4B on 8 GPUs should be near 2x 4B on 4 GPUs.
  * ``--gpu-memory-utilization 0.95`` and a client concurrency at or above ``--max-num-seqs`` keep
    each server's queue non-empty; below that the GPUs idle between waves of chunked thoughts.
  * vLLM 0.19.1 runs thoughts in waves, so concurrency should be a multiple of the wave size
    (the default wave max is 128 per server).  Levels 128/256 per server are the candidates.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import startlux_eval as E  # noqa: E402


def probe_sample(entries, limit):
    if limit <= 0 or limit >= len(entries):
        return entries
    stride = max(1, len(entries) // limit)
    return entries[::stride][:limit]


def parse_levels(text):
    levels = [int(v) for v in text.split(",") if v.strip()]
    if not levels or any(v < 1 for v in levels):
        raise argparse.ArgumentTypeError("concurrency levels must be positive integers")
    return levels


def eval_args(args, level, round_index):
    ns = argparse.Namespace(
        url=None, concurrency=level, timeout=args.timeout, retries=args.retries,
        retry_delay=args.retry_delay, monitor_gpu=True, gpu_interval=args.gpu_interval,
        gpu_log=args.out_dir / f"gpu_c{level}_r{round_index}.csv",
        controls_json=args.controls_json, think_mode=args.think_mode, gate=args.gate,
        mix=args.mix, think_budget=args.think_budget,
        thought_generation=args.thought_generation, none_check=args.none_check,
        allow_none=args.allow_none, return_details=args.return_details)
    return ns


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requests", type=Path, default=E.DEFAULT_REQUESTS)
    parser.add_argument("--sidecar", type=Path, default=E.DEFAULT_SIDECAR)
    parser.add_argument("--out-dir", type=Path, default=E.ROOT / "logs/qz/startlux_tune")
    parser.add_argument("--url", help="an already running server (skips launching; mock tests)")
    parser.add_argument("--model", default=str(E.ROOT / "models/StartLux-4B-AutoThink"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--launch-mode", choices=("official", "direct"), default="direct")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--vllm-extra-arg", action="append", default=[])
    parser.add_argument("--quantization", choices=("fp8",))
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8090)
    parser.add_argument("--vllm-base-port", type=int, default=8001)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-think-budget", type=int, default=4096)
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--request-limit", type=int, default=200)
    parser.add_argument("--concurrency-levels", type=parse_levels, default=[32, 64, 128, 256])
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--gpu-interval", type=float, default=2.0)
    parser.add_argument("--think-mode", choices=("auto", "never", "always", "hybrid", "plain", "think"))
    parser.add_argument("--gate", type=float)
    parser.add_argument("--mix", type=float)
    parser.add_argument("--think-budget", type=int)
    parser.add_argument("--thought-generation", choices=("auto", "stream", "chunked"))
    parser.add_argument("--none-check", dest="none_check", action="store_true", default=None)
    parser.add_argument("--no-none-check", dest="none_check", action="store_false")
    parser.add_argument("--allow-none", dest="allow_none", action="store_true", default=None)
    parser.add_argument("--return-details", action="store_true")
    parser.add_argument("--controls-json")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.vllm_log_dir = args.out_dir / "vllm"
    args.server_log = args.out_dir / "server.log"
    if args.name is None:
        args.name = Path(args.model).name

    entries = E.read_jsonl(args.requests)
    sidecar = json.loads(args.sidecar.read_text(encoding="utf-8"))
    sample = probe_sample(entries, args.request_limit)
    missing = [e["id"] for e in sample if e["id"] not in sidecar]
    if missing:
        raise SystemExit(f"sidecar misses {len(missing)} ids, e.g. {missing[:3]}")

    server = E.ServerProcess()
    results = []
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def run_all():
        nonlocal results
        if args.url is None:
            await server.launch(args)
            url = server.url
        else:
            url = args.url
        E.log(f"server ready at {url}; {len(sample)} probe items, "
              f"levels {args.concurrency_levels}, {args.rounds} round(s)")
        for round_index in range(args.rounds):
            for level in args.concurrency_levels:
                stem = args.out_dir / f"c{level}_r{round_index}"
                pred_path = stem.with_suffix(".predictions.jsonl")
                answers_path = stem.with_suffix(".answers.jsonl")
                round_args = eval_args(args, level, round_index)
                round_args.url = url
                with pred_path.open("w", encoding="utf-8") as pred, \
                        answers_path.open("w", encoding="utf-8") as answers:
                    metrics = await E.run(round_args, sample, sidecar, set(), answers, pred,
                                          asyncio.Lock())
                metrics.update({"concurrency": level, "round": round_index,
                                "probe_items": len(sample),
                                "settings": {
                                    "model": args.model, "gpus": args.gpus,
                                    "tp_size": args.tp_size, "max_num_seqs": args.max_num_seqs,
                                    "gpu_memory_utilization": args.gpu_memory_utilization,
                                    "max_model_len": args.max_model_len,
                                    "vllm_extra_arg": args.vllm_extra_arg}})
                results.append(metrics)
                gpu = metrics.get("gpu") or {}
                util = ", ".join(f"g{g}:{v['util_mean_pct']:.0f}%" for g, v in gpu.items())
                E.log(f"c={level:<4} ok={metrics['ok']:<4} failed={metrics['failed']:<3} "
                      f"items/s={metrics['items_per_s']:.2f} "
                      f"in_tok/s={metrics['input_tokens_per_s']:.0f} "
                      f"out_tok/s={metrics['output_tokens_per_s']:.1f} "
                      f"p50={metrics['latency_p50_s']} p95={metrics['latency_p95_s']} gpu[{util}]")

    try:
        loop.run_until_complete(run_all())
    finally:
        if server.procs:
            E.log("stopping servers")
            server.stop()
        loop.close()

    best = max(results, key=lambda r: r["input_tokens_per_s"]) if results else None
    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "probe_items": len(sample),
        "results": results,
        "recommendation": None if best is None else {
            "concurrency": best["concurrency"],
            "input_tokens_per_s": best["input_tokens_per_s"],
            "items_per_s": best["items_per_s"],
            "gpu_mean_utilization_pct": {g: v["util_mean_pct"]
                                         for g, v in (best.get("gpu") or {}).items()},
            "note": "choose --concurrency at or above this level; keep --max-num-seqs >= it so "
                    "each server queue stays non-empty",
        },
    }
    path = args.out_dir / "tune_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    E.log(f"wrote {path}")
    if best:
        E.log(f"best prefill throughput at concurrency {best['concurrency']}: "
              f"{best['input_tokens_per_s']:.0f} input tokens/s")
    return 0 if results and all(r["failed"] == 0 for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
