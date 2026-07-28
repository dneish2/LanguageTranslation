"""A running record of which engine served which translation, this session.

The comparison page answers "how do these models differ on one sentence".
This answers a different and more useful question: "what has actually been
happening to my text?" — how much ran on this machine versus somebody else's,
how fast each was, and what a session would have cost if none of it had run
locally.

Session-scoped by construction (see passage/policy.py): it lives in the same
per-visitor storage as Recent Threads, so it is the user's own picture of
their own session and disappears with it. Nothing here is durable, which is
exactly right for a transparency feature — a log of everywhere your text went
would be a strange thing to keep forever in the name of privacy.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable

#: Enough to see a pattern, small enough to keep in a session cookie store.
MAX_ENTRIES = 200


@dataclass
class LedgerEntry:
    surface: str
    engine: str
    is_local: bool
    latency_ms: int
    chars: int
    when: float

    @property
    def destination(self) -> str:
        return "this machine" if self.is_local else "sent out"


def record(store: list, *, surface: str, engine: str, is_local: bool,
           latency_ms: int, chars: int, when: float) -> None:
    """Append an entry, newest last, bounded."""
    store.append(asdict(LedgerEntry(
        surface=surface, engine=engine, is_local=bool(is_local),
        latency_ms=int(latency_ms), chars=int(chars), when=float(when),
    )))
    if len(store) > MAX_ENTRIES:
        del store[:len(store) - MAX_ENTRIES]


def summarise(entries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Local vs cloud, in the terms someone would actually ask about.

    Reports characters as well as counts: one document is one entry but a lot
    of text, so a count alone would say "mostly local" about a session that
    sent a whole report to a hosted model.
    """
    rows = list(entries)
    local = [r for r in rows if r.get("is_local")]
    remote = [r for r in rows if not r.get("is_local")]

    def stats(subset: list[dict[str, Any]]) -> dict[str, Any]:
        latencies = sorted(r.get("latency_ms", 0) for r in subset)
        return {
            "runs": len(subset),
            "chars": sum(r.get("chars", 0) for r in subset),
            "median_ms": latencies[len(latencies) // 2] if latencies else None,
        }

    engines: dict[str, int] = {}
    for r in rows:
        engines[r.get("engine", "?")] = engines.get(r.get("engine", "?"), 0) + 1

    total_chars = sum(r.get("chars", 0) for r in rows) or 1
    return {
        "total_runs": len(rows),
        "local": stats(local),
        "remote": stats(remote),
        "local_share_of_chars": round(len(local) and
                                      sum(r.get("chars", 0) for r in local) / total_chars or 0.0, 3),
        "engines": dict(sorted(engines.items(), key=lambda kv: -kv[1])),
    }
