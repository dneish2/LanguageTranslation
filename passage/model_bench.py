"""Measure the local translation models, reproducibly.

David's ask was model diversity, and diversity is only useful if you can tell
which model to reach for. This is the harness that answers that: one fixed
suite of source sentences, every installed model, results written to JSONL so
a later run can be compared against an earlier one rather than re-argued.

**Why these sentences.** Engines agree trivially on "Where is the pharmacy?",
which makes every model look equally good and the exercise pointless. The
suite deliberately carries the things translation actually fails on: financial
idiom, a negation that flips meaning, a menu item that is a proper noun rather
than a description, and a sentence with a URL and a figure that must survive
untouched.

**Quality without references.** There is no gold translation to score against,
so quality is consensus: how closely a model's output matches the OTHER
models'. That rewards the mainstream reading and flags the outlier, which is
the honest signal available here — it is a triage tool for "which of these
should I look at", not a leaderboard. A model can be right and alone.

**Cold vs warm.** Every model is warmed before timing. Ollama loads weights on
first use, and a cold first call measured 2,957ms against 148ms warm on the
same model — timing that would rank models by load order, not speed.
"""
from __future__ import annotations

import difflib
import json
import re
import statistics
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable

#: Fixed suite. Changing these invalidates comparison with earlier runs, so
#: add rather than edit.
SUITE: tuple[tuple[str, str], ...] = (
    ("idiom", "The board pushed back on the buyback, arguing it would leave the "
              "balance sheet stretched heading into a soft quarter."),
    ("negation", "The filing does not suggest the merger will close this year, "
                 "and management declined to say otherwise."),
    ("menu", "Pimientos de Padrón a la plancha con aceite de oliva y sal gruesa."),
    ("preserve", "See https://example.com/q3-post-money-valuation for the 12.4% "
                 "figure and email ir@example.com with questions."),
    ("plain", "The train to the airport leaves from platform nine at half past six."),
)

DEFAULT_TARGET = "Spanish"


@dataclass
class ModelResult:
    model: str
    size_gb: float = 0.0
    latencies_ms: list[int] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    consensus: float | None = None
    preserved_spans: bool | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.outputs)

    @property
    def median_ms(self) -> int | None:
        return int(statistics.median(self.latencies_ms)) if self.latencies_ms else None


def _normalise(text: str) -> str:
    return re.sub(r"[^\w\s]", "", (text or "").lower()).strip()


_SPAN_RE = re.compile(r"https?://[^\s]+|[\w.+-]+@[\w-]+\.[\w.-]+|\d+[.,]\d+%?")


def check_preserved(source: str, output: str) -> bool:
    """Did URLs, emails and figures survive the trip?

    A separate axis from fluency on purpose: a model can produce beautiful
    Spanish and silently mangle the one URL in the sentence, and for a research
    document that is the worse failure. Measured, not assumed.
    """
    spans = _SPAN_RE.findall(source)
    if not spans:
        return True
    return all(span.rstrip(".,") in output for span in spans)


def score_consensus(results: list[ModelResult]) -> None:
    """Mean similarity to the other models, per case, averaged."""
    usable = [r for r in results if r.ok]
    if len(usable) < 2:
        for r in usable:
            r.consensus = None
        return
    for result in usable:
        per_case = []
        for case, _ in SUITE:
            mine = result.outputs.get(case)
            if not mine:
                continue
            others = [o.outputs.get(case) for o in usable if o is not result]
            scores = [
                difflib.SequenceMatcher(None, _normalise(mine), _normalise(other)).ratio()
                for other in others if other
            ]
            if scores:
                per_case.append(sum(scores) / len(scores))
        result.consensus = round(sum(per_case) / len(per_case), 3) if per_case else None


def run_bench(
    models: list[dict[str, Any]],
    translate: Callable[[str, str, str], str],
    target_language: str = DEFAULT_TARGET,
    repeats: int = 2,
) -> list[ModelResult]:
    """Run every model over the suite. `translate(model, text, language) -> str`.

    Serial by design: these share one GPU, and running them together measures
    VRAM contention rather than the models (six concurrent local models
    measured 12s-33s for models that answer in under a second alone).
    """
    results: list[ModelResult] = []
    for spec in models:
        name = spec["name"]
        result = ModelResult(model=name, size_gb=round(spec.get("size", 0) / 1e9, 2))
        try:
            translate(name, "warm up", target_language)  # load weights, untimed
            for case, source in SUITE:
                best_latency = None
                text = ""
                for _ in range(max(1, repeats)):
                    started = time.time()
                    text = translate(name, source, target_language)
                    elapsed = int((time.time() - started) * 1000)
                    best_latency = elapsed if best_latency is None else min(best_latency, elapsed)
                result.outputs[case] = text
                result.latencies_ms.append(best_latency or 0)
            result.preserved_spans = check_preserved(
                dict(SUITE)["preserve"], result.outputs.get("preserve", ""))
        except Exception as error:
            result.error = str(error)[:200]
        results.append(result)
    score_consensus(results)
    return results


def write_jsonl(results: list[ModelResult], path: Path, *, stamp: str) -> None:
    """Append one row per model so runs accumulate and can be diffed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for result in results:
            row = asdict(result)
            row["run_at"] = stamp
            row["median_ms"] = result.median_ms
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_table(results: list[ModelResult]) -> str:
    rows = ["model                     size    p50      consensus  spans",
            "------------------------- ------- -------- ---------- -----"]
    for r in sorted(results, key=lambda x: (not x.ok, -(x.consensus or 0))):
        if not r.ok:
            rows.append(f"{r.model:25s} {r.size_gb:5.1f}G  FAILED: {(r.error or '')[:40]}")
            continue
        rows.append(
            f"{r.model:25s} {r.size_gb:5.1f}G  {r.median_ms:5d}ms  "
            f"{(r.consensus if r.consensus is not None else 0):8.3f}   "
            f"{'ok' if r.preserved_spans else 'MANGLED'}")
    return "\n".join(rows)
