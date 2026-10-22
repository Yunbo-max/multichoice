"""A deterministic stand-in for the StartLux ``/v1/systemone`` server (CPU-only tests).

It implements just enough of the API for ``startlux_eval.py``: ``GET /health``,
``GET /v1/models`` and ``POST /v1/systemone``.  Answers are synthetic (a fixed distribution per
question), so the point is to exercise the client, the prediction conversion, resume/retry and
scoring -- not model quality.

  python -I mock_startlux_server.py --port 18090 --fail-first 2 --fail-every 7

``--fail-first N`` answers the first N posts with 502, ``--fail-every K`` answers every Kth post
with 429; both exercise the retry path and are counted in ``GET /stats``.
"""
import argparse
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATS = {"posts": 0, "failures_502": 0, "failures_429": 0, "answered": 0}
LOCK = threading.Lock()
ARGS = None


def stable_index(text, n):
    digest = hashlib.sha256(text.encode("utf-8", "replace")).digest()
    return int.from_bytes(digest[:8], "big") % n


def answer_question(spec, state):
    qtype = spec.get("type", "choice")
    if qtype == "bool":
        qtype = "noul"
    instructions = spec.get("instructions") or spec.get("question") or ""
    if qtype == "choice":
        criteria = spec.get("criteria", spec.get("options"))
        if isinstance(criteria, (list, tuple)):
            keys = [str(c) for c in criteria]
        else:
            keys = [str(k) for k in criteria]
        n = len(keys)
        top = stable_index(instructions + str(len(state)), n)
        weights = [1.0 / (abs(i - top) + 1.0) ** 2 for i in range(n)]
        total = sum(weights)
        probs = {key: weights[i] / total for i, key in enumerate(keys)}
        return {"type": "choice", "choice": keys[top],
                "confidence": (probs[keys[top]] - 1.0 / n) / (1.0 - 1.0 / n) if n > 1 else 1.0,
                "probabilities": probs}
    if qtype == "noul":
        return {"type": "noul", "noul": 0.75}
    if qtype == "score":
        criteria = spec.get("criteria") or []
        n = len(criteria)
        top = stable_index(instructions + str(len(state)), n)
        weights = [1.0 / (abs(i - top) + 1.0) ** 2 for i in range(n)]
        total = sum(weights)
        probs = {str(i): weights[i] / total for i in range(n)}
        return {"type": "score", "score": sum(i * probs[str(i)] for i in range(n)),
                "confidence": 0.5, "legend": {str(i): criteria[i] for i in range(n)},
                "probabilities": probs}
    raise ValueError(f"unknown type {qtype!r}")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # quiet
        pass

    def _json(self, status, body, retry_after=None):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if retry_after is not None:
            self.send_header("Retry-After", str(retry_after))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "model": "MockStartLux", "kind": "auto_think",
                             "engine": "mock", "reads_images": False})
        elif self.path == "/v1/models":
            self._json(200, {"models": [{"name": "MockStartLux", "description": "mock"}]})
        elif self.path == "/stats":
            with LOCK:
                self._json(200, dict(STATS))
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/systemone":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        with LOCK:
            STATS["posts"] += 1
            post = STATS["posts"]
        if post <= ARGS.fail_first:
            with LOCK:
                STATS["failures_502"] += 1
            self._json(502, {"error": "mock engine failure"})
            return
        if ARGS.fail_every and post % ARGS.fail_every == 0:
            with LOCK:
                STATS["failures_429"] += 1
            self._json(429, {"error": "mock rate limit"}, retry_after=1)
            return
        try:
            body = json.loads(raw)
            state = body.get("state")
            state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
            answers = {key: answer_question(spec, state_text)
                       for key, spec in body["questions"].items()}
        except Exception as error:  # noqa: BLE001
            self._json(422, {"error": str(error)})
            return
        if ARGS.delay:
            time.sleep(ARGS.delay)
        with LOCK:
            STATS["answered"] += 1
        self._json(200, {"answers": answers,
                         "usage": {"input_tokens": max(1, len(state_text) // 4), "output_tokens": 0},
                         "model": "MockStartLux"})


def main():
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18090)
    parser.add_argument("--fail-first", type=int, default=0)
    parser.add_argument("--fail-every", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.0)
    ARGS = parser.parse_args()
    server = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    print(f"mock StartLux server on http://{ARGS.host}:{ARGS.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
