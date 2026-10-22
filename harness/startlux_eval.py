"""Evaluate StartLux-Decision on JEVal through the official ``/v1/systemone`` server.

The harness starts the model the way the release documents it (``startlux-decide serve`` with one
vLLM server per GPU on vLLM 0.19.1), sends the converted requests concurrently over HTTP, and
writes predictions in the format ``score_jeval.py`` accepts.

Two launch modes
  official  ``startlux-decide serve --launch-vllm`` with the release flags.  One vLLM server per
            GPU, no tensor parallel -- the deployment the release measured.
  direct    the same ``server_command`` from ``startlux_decide.launch``, but this script owns the
            processes, so it can add ``--max-num-seqs`` / ``--tensor-parallel-size`` and extra vLLM
            flags.  Used for GPU-utilisation tuning; the results are equivalent when the extra
            flags are left at their defaults.

Outputs
  --out          predictions.jsonl  {"id","type","probabilities":{...}}
  --answers      answers.jsonl      the full response per item (usage, timing, attempts)
  --metrics      metrics.json       latency, tokens/s, retries, GPU-utilisation summary
  --gpu-log      gpu_util.csv       nvidia-smi samples while the run is active

Resume: rerunning with the same --out/--answers skips the ids that already have a prediction.
Sharding: --shard i --shards N selects rows with index % N == i (one job per shard).

Usage
  # smoke against an already running server or the mock server
  python -I startlux_eval.py --requests ../data/jeval/startlux/startlux_requests.jsonl \
      --sidecar ../data/jeval/startlux/startlux_sidecar.json \
      --url http://127.0.0.1:8090 --limit 20 --out /tmp/preds.jsonl --answers /tmp/ans.jsonl
  # full run, launching 4B on GPUs 0-3
  python -I startlux_eval.py --model ../models/StartLux-4B-AutoThink --gpus 0,1,2,3 \
      --out ../logs/qz/startlux-4b/predictions.jsonl --answers ../logs/qz/startlux-4b/answers.jsonl \
      --monitor-gpu --score
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import aiohttp

QUESTION_KEY = "q"
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUESTS = ROOT / "data/jeval/startlux/startlux_requests.jsonl"
DEFAULT_SIDECAR = ROOT / "data/jeval/startlux/startlux_sidecar.json"


# ----------------------------------------------------------------------------- io helpers

def log(message):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {message}", flush=True)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # a process killed mid-write leaves a partial last line
                    log(f"ignoring a partial line in {path}")
    return rows


def sanitize_jsonl(path):
    """Drop a partial last line so appends start on a line boundary."""
    path = Path(path)
    if not path.exists():
        return
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        cut = data.rfind(b"\n")
        path.write_bytes(data[: cut + 1] if cut >= 0 else b"")
        log(f"truncated a partial last line in {path}")


def append_jsonl(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()


def is_true_key(key):
    return str(key).strip().lower() in ("true", "yes")


def prediction_for(item, answer):
    """Turn one /v1/systemone answer into a score_jeval prediction line."""
    qtype = item["type"]
    options = item["options"]
    if qtype == "choice":
        probs = answer.get("probabilities")
        if not isinstance(probs, dict):
            raise ValueError("choice answer has no probabilities object")
        missing = [o["key"] for o in options if o["key"] not in probs]
        if missing:
            raise ValueError(f"choice answer misses {missing[:3]} of {len(options)} option ids")
        out = {"id": item["id"], "type": "choice",
               "probabilities": {o["key"]: float(probs[o["key"]]) for o in options},
               "choice": answer.get("choice"), "confidence": answer.get("confidence")}
        if "none_probability" in answer:
            out["none_probability"] = answer["none_probability"]
        return out
    if qtype == "noul":
        p_true = answer.get("noul", answer.get("p_true"))
        if p_true is None:
            raise ValueError("noul answer has no noul field")
        p_true = float(p_true)
        return {"id": item["id"], "type": "noul", "noul": p_true,
                "probabilities": {o["key"]: (p_true if is_true_key(o["key"]) else 1.0 - p_true)
                                  for o in options}}
    if qtype == "score":
        probs = answer.get("probabilities")
        if not isinstance(probs, dict):
            raise ValueError("score answer has no probabilities object")
        mapped = {}
        for index, option in enumerate(options):
            key = option["key"]
            if str(index) in probs:
                mapped[key] = float(probs[str(index)])
            elif key in probs:
                mapped[key] = float(probs[key])
            else:
                raise ValueError(f"score answer misses level {index}")
        out = {"id": item["id"], "type": "score", "probabilities": mapped,
               "score": answer.get("score"), "confidence": answer.get("confidence")}
        return out
    raise ValueError(f"unknown type {qtype!r}")


def rebuilt_predictions(answers_path, sidecar):
    """Predictions recoverable from an answers file (for resume after a crash)."""
    out = {}
    for record in read_jsonl(answers_path):
        ident = record.get("id")
        if ident in sidecar and record.get("answer") is not None:
            with contextlib.suppress(Exception):
                out[ident] = prediction_for(sidecar[ident], record["answer"])
    return out


# ----------------------------------------------------------------------------- launcher

class ServerProcess:
    """The serve process and, in direct mode, the vLLM processes it talks to."""

    def __init__(self):
        self.procs = []
        self.url = None
        self.log_path = None

    def _spawn(self, cmd, env, log_path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("ab")
        proc = subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                start_new_session=True, cwd=str(ROOT))
        handle.close()
        self.procs.append(proc)
        return proc

    @staticmethod
    def _env(extra=None):
        env = dict(os.environ)
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.setdefault("VLLM_NO_USAGE_STATS", "1")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        repo = str(ROOT / "startlux-decision")
        env["PYTHONPATH"] = repo + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        if extra:
            env.update(extra)
        return env

    @staticmethod
    def _tail(path, lines=40):
        try:
            return "".join(Path(path).read_text(errors="replace").splitlines(True)[-lines:])
        except OSError:
            return "(no log)"

    async def health(self, url, timeout):
        deadline = time.monotonic() + timeout
        last = None
        async with aiohttp.ClientSession() as session:
            while time.monotonic() < deadline:
                for proc in self.procs:
                    if proc.poll() is not None:
                        raise RuntimeError(f"a server process exited with code {proc.returncode}:"
                                           f"\n{self._tail(self.log_path)}")
                try:
                    async with session.get(url + "/health",
                                           timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            return
                        last = f"HTTP {response.status}"
                except Exception as error:  # noqa: BLE001
                    last = str(error)
                await asyncio.sleep(5)
        raise RuntimeError(f"{url}/health not ready after {timeout:.0f}s (last: {last}):"
                           f"\n{self._tail(self.log_path)}")

    @staticmethod
    def _gpu_groups(gpus, tp_size):
        groups = [gpus[i:i + tp_size] for i in range(0, len(gpus), tp_size)]
        if any(len(g) != tp_size for g in groups):
            raise SystemExit(f"--gpus {','.join(gpus)} is not divisible by --tp-size {tp_size}")
        return groups

    async def launch(self, args):
        python = str(Path(args.python).resolve())
        server_log = Path(args.server_log)
        gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
        base_cmd = [python, "-m", "startlux_decide.cli", "serve",
                    "--model", str(Path(args.model).resolve()), "--backend", "vllm",
                    "--host", args.host, "--port", str(args.server_port), "--name", args.name,
                    "--max-model-len", str(args.max_model_len),
                    "--max-think-budget", str(args.max_think_budget), "--no-images"]

        if args.launch_mode == "official":
            cmd = base_cmd + ["--launch-vllm", "--gpus", ",".join(gpus),
                              "--vllm-base-port", str(args.vllm_base_port),
                              "--vllm-log-dir", str(Path(args.vllm_log_dir)),
                              "--vllm-python", python,
                              "--gpu-memory-utilization", str(args.gpu_memory_utilization)]
            if args.quantization:
                cmd += ["--quantization", args.quantization]
            if args.no_mtp:
                cmd += ["--no-mtp"]
            self.log_path = server_log
            log("launching (official): " + " ".join(cmd))
            self._spawn(cmd, self._env(), server_log)
            self.url = f"http://{args.host}:{args.server_port}"
            await self.health(self.url, args.startup_timeout)
            return

        # direct: vLLM servers from the release's server_command, plus our own extra flags
        from startlux_decide.launch import server_command
        groups = self._gpu_groups(gpus, args.tp_size)
        urls = []
        for index, group in enumerate(groups):
            port = args.vllm_base_port + index
            cmd = server_command(python, str(Path(args.model).resolve()), port, args.host, "m",
                                 args.max_model_len, args.gpu_memory_utilization, args.quantization,
                                 prefix_caching=False, mtp_tokens=0, images=0)
            if args.tp_size > 1:
                cmd += ["--tensor-parallel-size", str(args.tp_size)]
            if args.max_num_seqs:
                cmd += ["--max-num-seqs", str(args.max_num_seqs)]
            cmd += args.vllm_extra_arg
            env = self._env({"CUDA_VISIBLE_DEVICES": ",".join(group)})
            self._spawn(cmd, env, Path(args.vllm_log_dir) / f"vllm_{index}_gpu{'-'.join(group)}.log")
            log(f"launching vLLM gpu={','.join(group)} port={port}: " + " ".join(cmd))
            urls.append(f"http://{args.host}:{port}")
        self.log_path = server_log
        for url in urls:
            await self.health(url, args.startup_timeout)
        cmd = base_cmd + ["--vllm-url", ",".join(urls)]
        log("launching (direct serve): " + " ".join(cmd))
        self._spawn(cmd, self._env(), server_log)
        self.url = f"http://{args.host}:{args.server_port}"
        await self.health(self.url, args.startup_timeout)

    def stop(self):
        for proc in self.procs:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
        deadline = time.time() + 60
        for proc in self.procs:
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(max(0.1, deadline - time.time()))
        for proc in self.procs:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=10)
        self.procs.clear()


# ----------------------------------------------------------------------------- gpu monitor

async def gpu_monitor(stop, path, interval):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("nvidia-smi") is None:
        log("nvidia-smi not found: GPU utilisation is not recorded")
        return
    with path.open("w", encoding="utf-8") as out:
        out.write("timestamp,index,utilization_gpu_pct,memory_used_mib,memory_total_mib,power_w\n")
        out.flush()
        while not stop.is_set():
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits", stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL)
            text, _ = await proc.communicate()
            stamp = time.time()
            for line in text.decode(errors="replace").splitlines():
                cells = [c.strip() for c in line.split(",")]
                if len(cells) == 5:
                    out.write(f"{stamp:.0f}," + ",".join(cells) + "\n")
            out.flush()
            await asyncio.sleep(interval)


def summarize_gpu(path):
    path = Path(path)
    if not path.exists():
        return None
    by_gpu = {}
    for line in path.read_text().splitlines()[1:]:
        cells = line.split(",")
        if len(cells) != 6:
            continue
        try:
            gpu, util, used, total, power = int(cells[1]), float(cells[2]), float(cells[3]), \
                float(cells[4]), float(cells[5])
        except ValueError:
            continue
        row = by_gpu.setdefault(gpu, {"samples": 0, "util_sum": 0.0, "util_max": 0.0,
                                      "mem_used_max_mib": 0.0, "mem_total_mib": total,
                                      "power_sum": 0.0})
        row["samples"] += 1
        row["util_sum"] += util
        row["util_max"] = max(row["util_max"], util)
        row["mem_used_max_mib"] = max(row["mem_used_max_mib"], used)
        row["power_sum"] += power
    return {gpu: {"samples": r["samples"],
                  "util_mean_pct": r["util_sum"] / r["samples"],
                  "util_max_pct": r["util_max"],
                  "memory_used_max_mib": r["mem_used_max_mib"],
                  "memory_total_mib": r["mem_total_mib"],
                  "power_mean_w": r["power_sum"] / r["samples"]}
            for gpu, r in sorted(by_gpu.items())}


# ----------------------------------------------------------------------------- run

def controls_from_args(args):
    controls = {}
    if args.think_mode is not None:
        controls["think_mode"] = args.think_mode
    if args.gate is not None:
        controls["gate"] = args.gate
    if args.mix is not None:
        controls["mix"] = args.mix
    if args.think_budget is not None:
        controls["think_budget"] = args.think_budget
    if args.thought_generation is not None:
        controls["thought_generation"] = args.thought_generation
    if args.none_check is not None:
        controls["none_check"] = args.none_check
    if args.allow_none is not None:
        controls["allow_none"] = args.allow_none
    if args.return_details:
        controls["return_details"] = True
    if args.controls_json:
        controls.update(json.loads(args.controls_json))
    return controls


async def run(args, requests, sidecar, done, answers_stream, pred_stream, done_lock):
    controls = controls_from_args(args)
    endpoint = args.url.rstrip("/") + "/v1/systemone"
    metrics = {"started": time.time(), "items_total": len(requests), "items_done_before": len(done),
               "ok": 0, "failed": 0, "retries": 0, "attempt_errors": Counter(),
               "input_tokens": 0, "output_tokens": 0, "latency_s": [],
               "types": Counter(), "errors": []}
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency, limit_per_host=args.concurrency)
    stop_monitor = asyncio.Event()
    monitor_task = None
    if args.monitor_gpu:
        monitor_task = asyncio.ensure_future(gpu_monitor(stop_monitor, args.gpu_log, args.gpu_interval))

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async def one(entry):
            item = sidecar[entry["id"]]
            body = dict(entry["request"])
            body.update(controls)
            started = time.perf_counter()
            attempt = 0
            while True:
                attempt += 1
                try:
                    async with semaphore:
                        async with session.post(endpoint, json=body) as response:
                            text = await response.text()
                            status = response.status
                    if status == 200:
                        data = json.loads(text)
                        answer = (data.get("answers") or {}).get(QUESTION_KEY)
                        if answer is None:
                            raise ValueError(f"response has no answers[{QUESTION_KEY!r}]")
                        prediction = prediction_for(item, answer)
                        break
                    if status in RETRY_STATUS and attempt <= args.retries:
                        metrics["retries"] += 1
                        metrics["attempt_errors"][f"http_{status}"] += 1
                        await asyncio.sleep(min(60, args.retry_delay * (2 ** (attempt - 1))))
                        continue
                    raise RuntimeError(f"HTTP {status}: {text[:500]}")
                except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                    if attempt <= args.retries:
                        metrics["retries"] += 1
                        metrics["attempt_errors"][type(error).__name__] += 1
                        await asyncio.sleep(min(60, args.retry_delay * (2 ** (attempt - 1))))
                        continue
                    raise
            elapsed = time.perf_counter() - started
            usage = data.get("usage") or {}
            record = {"id": item["id"], "type": item["type"], "answer": answer,
                      "usage": usage, "model": data.get("model"),
                      "timing_ms": data.get("timing_ms"), "latency_s": elapsed,
                      "attempts": attempt, "started": started}
            async with done_lock:
                metrics["ok"] += 1
                metrics["types"][item["type"]] += 1
                metrics["input_tokens"] += int(usage.get("input_tokens") or 0)
                metrics["output_tokens"] += int(usage.get("output_tokens") or 0)
                metrics["latency_s"].append(elapsed)
                append_jsonl(answers_stream, record)
                append_jsonl(pred_stream, prediction)
                done.add(item["id"])
            return None

        async def guarded(entry):
            try:
                return await one(entry)
            except Exception as error:  # noqa: BLE001
                metrics["failed"] += 1
                metrics["errors"].append({"id": entry["id"], "error": str(error)[:1000]})
                log(f"FAILED {entry['id']}: {error}")
                return {"id": entry["id"], "error": str(error)}

        try:
            await asyncio.gather(*(guarded(entry) for entry in requests))
        finally:
            stop_monitor.set()
            if monitor_task is not None:
                with contextlib.suppress(Exception):
                    await monitor_task

    elapsed = time.time() - metrics["started"]
    latencies = sorted(metrics["latency_s"])

    def pct(p):
        if not latencies:
            return None
        return latencies[min(len(latencies) - 1, int(round(p * (len(latencies) - 1))))]

    metrics.update({
        "finished": time.time(), "elapsed_s": elapsed,
        "items_per_s": metrics["ok"] / elapsed if elapsed else 0.0,
        "input_tokens_per_s": metrics["input_tokens"] / elapsed if elapsed else 0.0,
        "output_tokens_per_s": metrics["output_tokens"] / elapsed if elapsed else 0.0,
        "latency_p50_s": pct(0.50), "latency_p95_s": pct(0.95), "latency_max_s": pct(1.0),
        "attempt_errors": dict(metrics["attempt_errors"]),
        "types": dict(metrics["types"]),
        "errors": metrics["errors"][:20],
        "gpu": summarize_gpu(args.gpu_log) if args.monitor_gpu else None,
        "controls": controls,
        "concurrency": args.concurrency,
    })
    return metrics


def tail(path, lines=30):
    try:
        return "".join(Path(path).read_text(errors="replace").splitlines(True)[-lines:])
    except OSError:
        return "(no log)"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--out", type=Path, required=False,
                        help="predictions JSONL (score_jeval format)")
    parser.add_argument("--answers", type=Path, help="raw answers JSONL (default: <out>.answers.jsonl)")
    parser.add_argument("--metrics", type=Path, help="metrics JSON (default: <out>.metrics.json)")
    parser.add_argument("--gpu-log", type=Path, default=ROOT / "logs/qz/startlux_gpu_util.csv")

    parser.add_argument("--url", help="an already running StartLux server (skips launching)")
    parser.add_argument("--model", default=str(ROOT / "models/StartLux-4B-AutoThink"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--launch-mode", choices=("official", "direct"), default="official")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--tp-size", type=int, default=1, help="direct mode: tensor parallel per server")
    parser.add_argument("--max-num-seqs", type=int, help="direct mode: vLLM --max-num-seqs")
    parser.add_argument("--vllm-extra-arg", action="append", default=[],
                        help="direct mode: extra vLLM flag (repeat; use --vllm-extra-arg=--flag=value)")
    parser.add_argument("--quantization", choices=("fp8",))
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8090)
    parser.add_argument("--vllm-base-port", type=int, default=8001)
    parser.add_argument("--max-model-len", type=int, default=131072,
                        help="context per server (131072 covers the longest JEVal prompt, 82,749 tokens)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-think-budget", type=int, default=4096)
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--server-log", type=Path, default=ROOT / "logs/qz/startlux_server.log")
    parser.add_argument("--vllm-log-dir", type=Path, default=ROOT / "logs/qz/vllm")

    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--limit", type=int, help="first N selected items (smoke test)")
    parser.add_argument("--ids", help="comma-separated ids to run (smoke test; overrides sharding)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)

    parser.add_argument("--think-mode", choices=("auto", "never", "always", "hybrid", "plain", "think"))
    parser.add_argument("--gate", type=float)
    parser.add_argument("--mix", type=float)
    parser.add_argument("--think-budget", type=int)
    parser.add_argument("--thought-generation", choices=("auto", "stream", "chunked"))
    parser.add_argument("--none-check", dest="none_check", action="store_true", default=None,
                        help="force none_check on for every request")
    parser.add_argument("--no-none-check", dest="none_check", action="store_false",
                        help="force none_check off for every request")
    parser.add_argument("--allow-none", dest="allow_none", action="store_true", default=None)
    parser.add_argument("--return-details", action="store_true")
    parser.add_argument("--controls-json", help="JSON object merged into every request")

    parser.add_argument("--monitor-gpu", action="store_true")
    parser.add_argument("--gpu-interval", type=float, default=2.0)
    parser.add_argument("--score", action="store_true", help="run score_jeval.py at the end")
    parser.add_argument("--sii-input", type=Path, default=ROOT / "data/jeval/sii_input.jsonl")
    parser.add_argument("--gold", type=Path, default=ROOT / "data/jeval/gold.json")
    parser.add_argument("--score-out", type=Path, help="summary JSON for score_jeval.py")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the launch plan and exit")
    parser.add_argument("--print-request", action="store_true",
                        help="print the first selected request body (with controls) and exit")
    args = parser.parse_args()

    if not 0 <= args.shard < args.shards:
        raise SystemExit("--shard must be in [0, --shards)")
    if args.out is None and not (args.dry_run or args.print_request):
        raise SystemExit("--out is required")
    if args.answers is None and args.out is not None:
        args.answers = args.out.with_suffix(".answers.jsonl")
    if args.metrics is None and args.out is not None:
        args.metrics = args.out.with_suffix(".metrics.json")
    if args.name is None:
        args.name = Path(args.model).name

    entries = read_jsonl(args.requests)
    sidecar = json.loads(Path(args.sidecar).read_text(encoding="utf-8"))
    selected = [e for index, e in enumerate(entries) if index % args.shards == args.shard]
    if args.ids:
        wanted = {ident.strip() for ident in args.ids.split(",") if ident.strip()}
        selected = [e for e in entries if e["id"] in wanted]
    if args.limit is not None:
        selected = selected[: args.limit]
    missing_sidecar = [e["id"] for e in selected if e["id"] not in sidecar]
    if missing_sidecar:
        raise SystemExit(f"sidecar misses {len(missing_sidecar)} ids, e.g. {missing_sidecar[:3]}")

    if args.print_request:
        if not selected:
            raise SystemExit("no rows selected")
        body = dict(selected[0]["request"])
        body.update(controls_from_args(args))
        print(json.dumps({"id": selected[0]["id"], "request": body}, ensure_ascii=False, indent=2)[:20000])
        return 0

    if args.dry_run:
        log(f"selected {len(selected)} / {len(entries)} items "
            f"(shard {args.shard}/{args.shards}, limit {args.limit})")
        if args.url is None:
            log(f"launch mode: {args.launch_mode}; model={args.model}; gpus={args.gpus}; "
                f"tp={args.tp_size}; max_model_len={args.max_model_len}; "
                f"gpu_mem={args.gpu_memory_utilization}; concurrency={args.concurrency}")
        else:
            log(f"using running server {args.url}")
        return 0

    for path in (args.out, args.answers):
        sanitize_jsonl(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    existing = {p["id"] for p in read_jsonl(args.out) if "id" in p}
    recovered = {k: v for k, v in rebuilt_predictions(args.answers, sidecar).items()
                 if k not in existing}
    done = set(existing) | set(recovered)
    answers_stream = args.answers.open("a", encoding="utf-8")
    pred_stream = args.out.open("a", encoding="utf-8")
    for ident, prediction in recovered.items():
        append_jsonl(pred_stream, prediction)
    if recovered:
        log(f"recovered {len(recovered)} predictions from {args.answers}")
    pending = [e for e in selected if e["id"] not in done]
    log(f"{len(selected)} selected; {len(done & {e['id'] for e in selected})} done; "
        f"{len(pending)} pending")

    server = ServerProcess()
    metrics = None
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def main_async():
        nonlocal metrics
        if not pending:
            metrics = {"ok": 0, "failed": 0, "retries": 0, "elapsed_s": 0.0,
                       "items_per_s": 0.0, "input_tokens_per_s": 0.0,
                       "output_tokens_per_s": 0.0, "latency_p50_s": None,
                       "latency_p95_s": None, "gpu": summarize_gpu(args.gpu_log)
                       if args.monitor_gpu else None, "controls": controls_from_args(args),
                       "concurrency": args.concurrency, "errors": [],
                       "input_tokens": 0, "output_tokens": 0, "types": {}}
            return metrics
        if args.url is None:
            await server.launch(args)
            log(f"server ready at {server.url}")
        url = args.url or server.url
        saved_url, args.url = args.url, url
        try:
            metrics = await run(args, pending, sidecar, done, answers_stream, pred_stream,
                                asyncio.Lock())
        finally:
            args.url = saved_url
        return metrics

    try:
        metrics = loop.run_until_complete(main_async())
    except KeyboardInterrupt:
        log("interrupted")
        raise SystemExit(130)
    finally:
        answers_stream.close()
        pred_stream.close()
        if server.procs:
            log("stopping servers")
            server.stop()
        loop.close()

    metrics["shard"] = args.shard
    metrics["shards"] = args.shards
    metrics["limit"] = args.limit
    metrics["server_log_tail"] = tail(args.server_log)
    args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(json.dumps({k: metrics[k] for k in
                    ("ok", "failed", "retries", "elapsed_s", "items_per_s",
                     "input_tokens_per_s", "output_tokens_per_s", "latency_p50_s",
                     "latency_p95_s", "gpu")}, ensure_ascii=False, indent=2))

    if args.score:
        summary_out = args.score_out or (args.out.parent / "score_summary.json")
        cmd = [sys.executable, "-I", str(ROOT / "harness/score_jeval.py"),
               "--input", str(args.sii_input), "--gold", str(args.gold),
               "--pred", str(args.out), "--out", str(summary_out)]
        log("scoring: " + " ".join(cmd))
        code = subprocess.run(cmd).returncode
        if code:
            log(f"score_jeval.py exited with {code}")
            return code

    return 0 if (args.allow_failures or metrics["failed"] == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
