"""Counting the work Passage actually pays for.

`policy.is_metered()` has existed since the BYO work and nothing read it, so
"bring your own key and it's free" was a promise the code made and never kept.
This is the counter that makes it true, and it is deliberately small: a count
of metered units per session, with the decision of what counts delegated
entirely to `policy`.

**Only metered work is counted at all.** Not counted-then-discounted, not
recorded with a zero — a local or BYO run doesn't reach the counter, so there
is no path by which a change to billing later starts charging for inference
somebody else paid for.

**Units are characters of source text, not requests.** Live typing fires ~8
requests for one sentence while a document is one request carrying thousands
of characters; counting requests would bill those the same way round. It also
means the number tracks something a user can predict from what they typed.

Session-scoped for now, matching where identity currently exists. When real
accounts land this becomes a read against a `passage_usage` table keyed by
uid, and nothing above it has to change — callers ask `record()` and
`summary()`, never where the number lives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Free allowance before a signed-out visitor is asked to sign in or bring a
#: key. Generous on purpose: the point is to make the paid path make sense,
#: not to interrupt someone evaluating the app. Nothing enforces this yet —
#: `over_free_allowance` reports it, and no caller blocks on it, because a
#: quota that starts refusing work the day it ships is a bad surprise.
FREE_CHARS = 50_000


@dataclass
class UsageSummary:
    metered_chars: int
    metered_runs: int
    free_chars: int = FREE_CHARS

    @property
    def remaining_chars(self) -> int:
        return max(0, self.free_chars - self.metered_chars)

    @property
    def over_free_allowance(self) -> bool:
        return self.metered_chars > self.free_chars

    @property
    def share_used(self) -> float:
        return min(1.0, self.metered_chars / self.free_chars) if self.free_chars else 0.0


def record(store: dict[str, Any], *, chars: int, metered: bool) -> None:
    """Count `chars` of source text, but only if Passage paid for it.

    `metered` comes from policy.is_metered(); this module deliberately does not
    re-derive it, so there is exactly one place that decides.
    """
    if not metered or chars <= 0:
        return
    store["metered_chars"] = int(store.get("metered_chars", 0)) + int(chars)
    store["metered_runs"] = int(store.get("metered_runs", 0)) + 1


def summary(store: dict[str, Any]) -> UsageSummary:
    return UsageSummary(
        metered_chars=int(store.get("metered_chars", 0)),
        metered_runs=int(store.get("metered_runs", 0)),
    )


def describe(summary_: UsageSummary) -> str:
    """A line for the UI that says what was actually charged for."""
    if summary_.metered_runs == 0:
        return "Nothing metered this session — everything ran locally or on your own key."
    if summary_.over_free_allowance:
        return (f"{summary_.metered_chars:,} characters metered — over the "
                f"{summary_.free_chars:,} free allowance.")
    return (f"{summary_.metered_chars:,} of {summary_.free_chars:,} free characters used "
            f"across {summary_.metered_runs} metered run"
            f"{'s' if summary_.metered_runs != 1 else ''}.")
