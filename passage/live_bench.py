"""Measure the live text box as a person typing into it experiences it.

    python -m passage.live_bench --label before --engine hosted --runs 5
    python -m passage.live_bench --label before --engine local  --runs 5

Two numbers, both taken from a real browser typing into the real page:

- **model calls per typed sentence**: requests that reached the model (a cache
  hit is not a call). For the hosted engine these are the paid calls.
- **time to first text**: from the moment a request leaves the browser to the
  first translated character painted in the output box, for every request
  that was allowed to finish.

The hosted engine is a fake OpenAI-compatible server (no key is needed, and
nothing is billed) with latency shaped like gpt-5.4-nano on this machine
(RESEARCH.md: 616 ms p50 for a short sentence). It counts every call, every
token it generated, and every token it generated for a client that had
already gone away. The local engine is the real Ollama.

Typing is simulated, deterministically per seed, from a keystroke model: a
log-normal inter-key interval (median 180 ms), a longer gap before each word,
occasional thinking pauses at word boundaries and occasional mid-word
hesitations. The same seeds are used before and after a change, so the only
thing that differs is the code.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import socket
import statistics
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "live_bench.jsonl"

SENTENCES = [
    "The meeting was moved to Thursday morning because the client is travelling.",
    "Please send me the signed contract before the end of the week.",
    "Our quarterly revenue grew by twelve percent, mostly from new customers.",
    "Could you check whether the train to Lyon leaves from platform four?",
    "The doctor said I should rest for three days and drink more water.",
]


# ── typist ───────────────────────────────────────────────────────────────

def keystrokes(sentence: str, seed: int) -> list[tuple[str, float]]:
    """(key, delay_ms_before_key). Key "Backspace" is a correction."""
    rng = random.Random(seed)
    out: list[tuple[str, float]] = []
    for i, ch in enumerate(sentence):
        delay = rng.lognormvariate(math.log(180), 0.45)
        at_word_start = i > 0 and sentence[i - 1] == " "
        if at_word_start:
            delay += rng.lognormvariate(math.log(120), 0.5)
            if rng.random() < 0.10:
                delay += rng.uniform(700, 2500)  # thinking about the next word
        elif ch != " " and rng.random() < 0.03:
            delay += rng.uniform(400, 900)  # hesitation inside a word
        if ch.isalpha() and rng.random() < 0.02:
            out.append((rng.choice("asdfghjkl"), delay))
            out.append(("Backspace", rng.lognormvariate(math.log(260), 0.3)))
            delay = rng.lognormvariate(math.log(160), 0.3)
        out.append((ch, delay))
    return out


# ── fake hosted engine ───────────────────────────────────────────────────

class FakeHosted:
    """OpenAI-compatible /v1/chat/completions, streaming and not, that
    keeps books. Latency: TTFT ~450 ms, then ~70 tokens/s, one word a token."""

    TTFT_S = 0.45
    TOKEN_S = 1 / 70

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                body = json.dumps({"object": "list", "data": [{"id": "fake-nano", "object": "model"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                prompt = " ".join(m["content"] for m in payload["messages"])
                text = payload["messages"][-1]["content"]
                # A real model answers with the translation only, so output
                # length follows the text between the markers, not the prompt.
                if "BEGIN TEXT\n" in text:
                    text = text.split("BEGIN TEXT\n", 1)[1].split("\nEND TEXT", 1)[0]
                words = [f"ES({w})" for w in text.split()]
                call = {"at": time.time(), "chars": len(text), "stream": bool(payload.get("stream")),
                        "prompt_words": len(prompt.split()),
                        "tokens": 0, "abandoned_tokens": 0, "completed": False}
                with fake.lock:
                    fake.calls.append(call)
                time.sleep(fake.TTFT_S)
                if not payload.get("stream"):
                    time.sleep(fake.TOKEN_S * len(words))
                    call["tokens"] = len(words)
                    body = json.dumps({
                        "id": "x", "object": "chat.completion", "created": 0, "model": "fake-nano",
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": " ".join(words)}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": len(words), "total_tokens": 1},
                    }).encode()
                    try:
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        call["completed"] = True
                    except OSError:
                        call["abandoned_tokens"] = len(words)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                gone = False
                for i, word in enumerate(words):
                    chunk = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "fake-nano",
                             "choices": [{"index": 0, "delta": {"content": (" " if i else "") + word},
                                          "finish_reason": None}]}
                    if gone:
                        call["abandoned_tokens"] += 1
                        continue
                    try:
                        self._chunk(f"data: {json.dumps(chunk)}\n\n".encode())
                        call["tokens"] += 1
                    except OSError:
                        gone = True
                        call["abandoned_tokens"] += len(words) - i
                        break
                    time.sleep(fake.TOKEN_S)
                if not gone:
                    try:
                        self._chunk(b"data: [DONE]\n\n")
                        self._chunk(b"")
                        call["completed"] = True
                    except OSError:
                        pass

            def _chunk(self, data: bytes) -> None:
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):
                pass  # a client that hung up is data (abandoned_tokens), not an error

        self.server = Server(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def reset(self) -> None:
        with self.lock:
            self.calls.clear()


# ── app under test ───────────────────────────────────────────────────────

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_app(engine: str, fake: FakeHosted | None, log: Path) -> tuple[subprocess.Popen, int]:
    port = _free_port()
    env = {**os.environ, "PORT": str(port), "PASSAGE_API_RATE_LIMIT": "100000",
           "PASSAGE_TRACE_DIR": str(log.parent / "bench-traces"), "PYTHONIOENCODING": "utf-8"}
    if engine == "hosted":
        env.update(PASSAGE_DEPLOYMENT="cloud", OPENAI_API_KEY="sk-bench",
                   OPENAI_BASE_URL=f"http://127.0.0.1:{fake.port}/v1")
    else:
        env.update(PASSAGE_DEPLOYMENT="local", OPENAI_API_KEY="",
                   PASSAGE_OLLAMA_BASE_URL="http://127.0.0.1:11434/v1")
    proc = subprocess.Popen([sys.executable, str(ROOT / "TranslationUI.py")], cwd=ROOT, env=env,
                            stdout=log.open("w"), stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
            return proc, port
        except OSError:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError(f"app did not start; see {log}")


#: Injected before the page's own scripts: timestamps every live request
#: and every change to the output box, so TTFT is measured where the user sees it.
_PROBE = """
window.__bench = {fetches: [], paints: []};
const _fetch = window.fetch;
window.fetch = function(url, opts) {
  const u = String(url);
  if (u.includes('/api/text_translate')) window.__bench.fetches.push({t: performance.now(), url: u});
  return _fetch.apply(this, arguments);
};
let __last = null;
new MutationObserver(() => {
  const out = document.getElementById('workspace_text_output');
  if (!out || out.textContent === __last) return;
  __last = out.textContent;
  window.__bench.paints.push({t: performance.now(), text: __last});
}).observe(document, {subtree: true, childList: true, characterData: true});
"""


def type_sentence(page, sentence: str, seed: int, settle_s: float) -> dict:
    page.evaluate("window.__bench.fetches = []; window.__bench.paints = [];")
    page.locator("#workspace_text_output").evaluate("el => el.textContent = ''")
    source = page.locator("#workspace_text_source")
    source.click()
    for key, delay in keystrokes(sentence, seed):
        page.wait_for_timeout(delay)
        page.keyboard.press(key) if key == "Backspace" else page.keyboard.type(key)
    typed_at = page.evaluate("performance.now()")
    page.wait_for_timeout(settle_s * 1000)
    bench = page.evaluate("window.__bench")
    final = page.locator("#workspace_text_output").text_content() or ""
    return {"bench": bench, "typed_at": typed_at, "final": final}


def ttfts(bench: dict) -> list[float]:
    """For each request, ms until the next non-empty paint that happened
    before the following request went out (an aborted or superseded request
    has no TTFT: nothing it produced was shown)."""
    fetches = [f["t"] for f in bench["fetches"]]
    paints = [(p["t"], p["text"]) for p in bench["paints"] if (p["text"] or "").strip()]
    out = []
    for i, start in enumerate(fetches):
        end = fetches[i + 1] if i + 1 < len(fetches) else float("inf")
        shown = [t for t, _ in paints if start < t < end]
        if shown:
            out.append(shown[0] - start)
    return out


def run(label: str, engine: str, runs: int, settle_s: float) -> list[dict]:
    from playwright.sync_api import sync_playwright

    fake = FakeHosted() if engine == "hosted" else None
    log = ROOT / "data" / f"live_bench_{engine}.log"
    log.parent.mkdir(exist_ok=True)
    proc, port = start_app(engine, fake, log)
    rows = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for n in range(runs):
                context = browser.new_context()
                context.add_init_script(_PROBE)
                page = context.new_page()
                page.goto(f"http://127.0.0.1:{port}/?mode=Text", wait_until="networkidle")
                page.wait_for_selector("#workspace_text_source")
                sentence = SENTENCES[n % len(SENTENCES)]
                if fake:
                    fake.reset()
                result = type_sentence(page, sentence, seed=1000 + n, settle_s=settle_s)
                t = ttfts(result["bench"])
                paints = [pt["t"] for pt in result["bench"]["paints"] if (pt["text"] or "").strip()]
                final_ms = (paints[-1] - result["typed_at"]) if paints and paints[-1] > result["typed_at"] else None
                row = {
                    "label": label, "engine": engine, "run": n, "seed": 1000 + n,
                    "final_ms": round(final_ms) if final_ms is not None else None,
                    "sentence_chars": len(sentence), "requests": len(result["bench"]["fetches"]),
                    "ttft_ms": [round(x) for x in t],
                    "ttft_median_ms": round(statistics.median(t)) if t else None,
                    "final_ok": bool(result["final"].strip()),
                    "at": time.time(),
                }
                if fake:
                    calls = list(fake.calls)
                    row.update(model_calls=len(calls),
                               prompt_words=sum(c["prompt_words"] for c in calls),
                               tokens=sum(c["tokens"] for c in calls),
                               abandoned_tokens=sum(c["abandoned_tokens"] for c in calls),
                               completed_calls=sum(c["completed"] for c in calls))
                else:
                    row.update(model_calls=_local_calls(log, row))
                rows.append(row)
                print(json.dumps(row), flush=True)
                context.close()
            browser.close()
    finally:
        proc.kill()
        if fake:
            fake.server.shutdown()
    with RESULTS.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return rows


def _local_calls(log: Path, row: dict) -> int | None:
    """Local calls are free; the count is not needed for cost, so it is not
    inferred from logs. Requests is reported instead."""
    return None


def summarize(rows: list[dict]) -> dict:
    calls = [r["model_calls"] for r in rows if r.get("model_calls") is not None]
    ttft = [r["ttft_median_ms"] for r in rows if r.get("ttft_median_ms") is not None]
    return {
        "runs": len(rows),
        "requests_median": statistics.median(r["requests"] for r in rows),
        "model_calls_median": statistics.median(calls) if calls else None,
        "ttft_median_ms": statistics.median(ttft) if ttft else None,
        "final_ms_median": statistics.median(f) if (f := [r["final_ms"] for r in rows if r.get("final_ms") is not None]) else None,
        "abandoned_tokens_median": statistics.median(r["abandoned_tokens"] for r in rows)
        if rows and "abandoned_tokens" in rows[0] else None,
        "final_ok": sum(r["final_ok"] for r in rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m passage.live_bench")
    parser.add_argument("--label", required=True)
    parser.add_argument("--engine", choices=["hosted", "local"], required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--settle", type=float, default=4.0, help="seconds to wait after the last key")
    args = parser.parse_args(argv)
    rows = run(args.label, args.engine, args.runs, args.settle)
    print(json.dumps({"label": args.label, "engine": args.engine, **summarize(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
