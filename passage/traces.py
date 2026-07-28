"""Durable record of what was translated, by which model, and what a human
changed afterwards.

The shape is Langfuse's, per the locked decision in PASSAGE_PLAN.md — a
**trace** is one document run, a **generation** is one segment's translation,
and a **score** is a human editing that translation. Borrowing an established
data model rather than inventing one means the eventual move to a real store
(or to Langfuse itself) is a change of writer, not a re-modelling.

Why this is worth having at all: the interesting product question here isn't
"was the translation good", it's "where did a person disagree with the model".
Those edits are the only ground truth this app ever gets — nobody writes a
reference translation, but they do fix the word that was wrong. Accumulated per
user, that is simultaneously the evaluation set, the per-user glossary, and the
answer to "is the local model actually good enough for my documents".

**Retention is not decided here.** Every write asks `passage.policy` whether
this surface may keep traces, so the rules live in one place. Documents may;
live typing, camera and voice may not, and calling `record_*` for them is a
no-op rather than an error — the caller shouldn't have to know the policy to
be safe.

JSONL on disk, because there is no database yet and waiting for one would mean
collecting nothing in the meantime. One file per day, append-only, so a run is
never rewritten and concurrent writers can't corrupt each other's rows.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from threading import Lock
from typing import Any

from passage import policy

TRACE_DIR = Path(os.getenv(
    "PASSAGE_TRACE_DIR",
    str(Path(__file__).resolve().parent.parent / "data" / "traces"),
))

#: Appends are small and infrequent (one per segment), so a process-wide lock
#: costs nothing and removes any chance of interleaved partial lines.
_write_lock = Lock()

#: Source text is truncated in traces. The full document already exists
#: wherever the user keeps it; a trace is for studying the DECISION, and
#: storing whole documents by accident is exactly the liability the retention
#: policy declines to take on.
MAX_TEXT_CHARS = 2000


@dataclass
class Trace:
    """One document run."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    target_language: str = ""
    engine: str = ""
    surface: str = policy.Surface.DOCUMENT.value
    started_at: float = field(default_factory=time.time)
    segment_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def _path_for(when: float | None = None) -> Path:
    stamp = time.strftime("%Y-%m-%d", time.localtime(when or time.time()))
    return TRACE_DIR / f"traces-{stamp}.jsonl"


def _append(row: dict[str, Any]) -> None:
    """Best effort. A trace that fails to write must never surface to the user
    or fail the translation it is describing — the record is a by-product, not
    the product."""
    try:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, ensure_ascii=False)
        with _write_lock:
            with _path_for().open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception as error:
        logging.info("[Traces] not written (%s)", error)


def _allowed(surface) -> bool:
    try:
        return policy.may_persist(surface, "durable_traces")
    except Exception:
        return False


def _clip(text: str | None) -> str:
    text = text or ""
    return text[:MAX_TEXT_CHARS]


def record_trace(trace: Trace) -> str | None:
    """Open a document run. Returns the trace id, or None if policy declines."""
    if not _allowed(trace.surface):
        return None
    row = asdict(trace)
    row["type"] = "trace"
    row["name"] = _clip(row.get("name"))
    _append(row)
    return trace.trace_id


def record_generation(
    *, trace_id: str | None, segment_id: str, source: str, output: str,
    engine: str, latency_ms: int | None = None,
    surface=policy.Surface.DOCUMENT,
) -> None:
    """One segment translated by a model."""
    if not trace_id or not _allowed(surface):
        return
    _append({
        "type": "generation",
        "trace_id": trace_id,
        "segment_id": segment_id,
        "source": _clip(source),
        "output": _clip(output),
        "engine": engine,
        "latency_ms": latency_ms,
        "at": time.time(),
    })


def record_edit(
    *, trace_id: str | None, segment_id: str, before: str, after: str,
    surface=policy.Surface.DOCUMENT,
) -> None:
    """A human changing a machine translation — the only ground truth here.

    `edit_ratio` is stored alongside the text so the common question ("which
    segments did people rewrite, not just tweak?") is answerable without
    re-reading every row and recomputing it.
    """
    if not trace_id or not _allowed(surface):
        return
    if (before or "") == (after or ""):
        return                      # not an edit; don't inflate the dataset
    import difflib

    ratio = difflib.SequenceMatcher(None, before or "", after or "").ratio()
    _append({
        "type": "score",
        "trace_id": trace_id,
        "segment_id": segment_id,
        "before": _clip(before),
        "after": _clip(after),
        "edit_ratio": round(ratio, 3),
        "rewritten": ratio < 0.6,
        "at": time.time(),
    })


def read_all(limit: int = 5000) -> list[dict[str, Any]]:
    """Every row across every day file, oldest first. Small enough to do
    naively — this is a study tool, not a query engine."""
    rows: list[dict[str, Any]] = []
    if not TRACE_DIR.is_dir():
        return rows
    for path in sorted(TRACE_DIR.glob("traces-*.jsonl")):
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        except (OSError, ValueError):
            continue
    return rows[-limit:]


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What the traces say, in the terms someone would ask.

    Edit rate is the headline: the share of machine translations a human
    changed. It is the closest thing to a quality signal this app can produce
    without a reference translation, and it is per-engine, so "is the local
    model good enough for my documents" becomes answerable from evidence
    instead of vibes.
    """
    generations = [r for r in rows if r.get("type") == "generation"]
    edits = [r for r in rows if r.get("type") == "score"]
    edited_segments = {r.get("segment_id") for r in edits}

    per_engine: dict[str, dict[str, Any]] = {}
    for gen in generations:
        entry = per_engine.setdefault(gen.get("engine", "?"), {"generations": 0, "edited": 0})
        entry["generations"] += 1
        if gen.get("segment_id") in edited_segments:
            entry["edited"] += 1
    for entry in per_engine.values():
        entry["edit_rate"] = (round(entry["edited"] / entry["generations"], 3)
                              if entry["generations"] else None)

    return {
        "traces": sum(1 for r in rows if r.get("type") == "trace"),
        "generations": len(generations),
        "edits": len(edits),
        "rewritten": sum(1 for r in edits if r.get("rewritten")),
        "edit_rate": round(len(edited_segments) / len(generations), 3) if generations else None,
        "per_engine": dict(sorted(per_engine.items(), key=lambda kv: -kv[1]["generations"])),
    }
