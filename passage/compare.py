"""Run one input across several engines and put the results side by side.

Two jobs. For a user: "I brought my own key / my own model — how does it
actually compare?" is unanswerable by staring at one translation at a time.
For the project: this is the substrate Phase 5's per-segment scoring and
LLM-as-judge need, because both start from the same shape — one source, N
candidate outputs, something measured about each.

**Measured, not asserted.** Latency and token counts are observed per run.
Cost is deliberately NOT invented: per-model prices change and are not
knowable from inside this process, so a rate table is read from the
environment (PASSAGE_RATE_<MODEL>=<usd per 1M output tokens>) and anything
unpriced reports None rather than a confident wrong number. Local runs are
genuinely free and say so.

**Agreement, not a winner.** There is no reference translation to score
against, so the honest measure is how much the engines agree with each other:
an output every engine converges on is more trustworthy than an outlier, and
an outlier is worth a human look. This ranks nothing and declares no winner —
it shows the spread and lets the person decide.
"""
from __future__ import annotations

import difflib
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

#: Cap on engines compared at once. Each is a real model call; a runaway list
#: would be slow and, for hosted engines, billable.
MAX_CANDIDATES = 6


@dataclass
class CandidateResult:
    label: str
    engine: str
    is_local: bool
    text: str = ""
    latency_ms: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    error: str | None = None
    agreement: float | None = field(default=None)

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text)


def rate_for(model: str) -> float | None:
    """USD per 1M output tokens, from the environment, or None if unpriced.

    Prices are not hardcoded on purpose — they go stale, they differ per
    account, and a plausible-looking wrong number in a comparison table is
    worse than an honest blank.
    """
    key = "PASSAGE_RATE_" + re.sub(r"[^A-Z0-9]+", "_", model.upper()).strip("_")
    raw = os.getenv(key)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def estimate_cost(model: str, output_tokens: int | None, *, is_local: bool) -> float | None:
    if is_local:
        return 0.0
    rate = rate_for(model)
    if rate is None or output_tokens is None:
        return None
    return rate * output_tokens / 1_000_000


def _normalise(text: str) -> str:
    """Compare meaningfully: casing and punctuation spacing shouldn't read as
    disagreement when the words are identical."""
    return re.sub(r"[^\w\s]", "", (text or "").lower()).strip()


def score_agreement(results: list[CandidateResult]) -> None:
    """Set each result's agreement: its mean similarity to the OTHER outputs.

    Mean-vs-others rather than vs-the-first: comparing everything to whichever
    engine happened to be listed first would make that engine's quirks the
    definition of correct.
    """
    usable = [r for r in results if r.ok]
    if len(usable) < 2:
        for r in usable:
            r.agreement = None
        return
    for result in usable:
        others = [o for o in usable if o is not result]
        scores = [
            difflib.SequenceMatcher(None, _normalise(result.text), _normalise(o.text)).ratio()
            for o in others
        ]
        result.agreement = round(sum(scores) / len(scores), 3)


def run_comparison(
    candidates: list[dict[str, Any]],
    translate: Callable[[dict[str, Any]], str],
    count_tokens: Callable[[str], int] | None = None,
) -> list[CandidateResult]:
    """Translate the same input with every candidate and score the results.

    Hosted candidates run concurrently — they are someone else's machines and
    do not contend. Local candidates run ONE AT A TIME, because they all share
    a single GPU: running six of them at once measured 12s, 23s, 26s and 33s
    for models that answer in a few hundred ms alone, since they cannot all be
    resident at once and thrash loading each other in and out. Those numbers
    are not just slow, they are WRONG — a comparison table whose latency column
    mostly measures VRAM contention would rank models by load order.

    A candidate that raises becomes a failed row: one broken endpoint must not
    void the whole comparison, because seeing which engine failed IS a result.
    """
    candidates = candidates[:MAX_CANDIDATES]

    def run_one(candidate: dict[str, Any]) -> CandidateResult:
        result = CandidateResult(
            label=candidate.get("label", "?"),
            engine=candidate.get("engine", "?"),
            is_local=bool(candidate.get("is_local")),
        )
        started = time.time()
        try:
            result.text = (translate(candidate) or "").strip()
            result.latency_ms = int((time.time() - started) * 1000)
            if not result.text:
                result.error = "returned no text"
            elif count_tokens is not None:
                result.output_tokens = count_tokens(result.text)
                result.cost_usd = estimate_cost(
                    candidate.get("model", ""), result.output_tokens, is_local=result.is_local)
        except Exception as error:
            result.latency_ms = int((time.time() - started) * 1000)
            result.error = str(error)[:200]
        return result

    if not candidates:
        return []
    remote = [(i, c) for i, c in enumerate(candidates) if not c.get("is_local")]
    local = [(i, c) for i, c in enumerate(candidates) if c.get("is_local")]
    slots: dict[int, CandidateResult] = {}

    with ThreadPoolExecutor(max_workers=max(1, len(remote))) as pool:
        # Fire the remote calls, then work the local queue one at a time while
        # they are in flight — the GPU is idle during that wait anyway.
        futures = {i: pool.submit(run_one, c) for i, c in remote}
        for i, candidate in local:
            slots[i] = run_one(candidate)
        for i, future in futures.items():
            slots[i] = future.result()

    results = [slots[i] for i in sorted(slots)]
    score_agreement(results)
    return results
