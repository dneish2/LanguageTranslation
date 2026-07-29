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


_UNSET = object()


@dataclass
class LedgerEntry:
    surface: str
    engine: str
    is_local: bool
    latency_ms: int
    chars: int
    when: float
    #: Tri-state, and the reason this dataclass grew a field. `is_local` was
    #: computed as `engine.startswith("local")`, so a cache hit — where no
    #: model ran and nothing was sent — was filed under "sent out", and an
    #: engine label nobody recognises was filed there too by luck rather than
    #: evidence. None means "not known", which is a thing the page has to be
    #: able to say.
    left_machine: bool | None = None
    #: Whether Passage paid for this run, decided by policy.classify_run from
    #: the engine that actually served it.
    metered: bool = False
    #: `policy.Ran` for this row, as a string. The `engine` column names WHO
    #: PRODUCED these bytes, which for a cache hit is the model that answered
    #: some earlier request — so `engine` alone can no longer tell a run from
    #: a re-read. Without this column a cache hit looked exactly like a fresh
    #: local run: five identical translations of one 44-character sentence
    #: rendered "100% of your text stayed on this machine · local: 5 runs ·
    #: 220 chars", and the hosted version of the same session claimed 80%
    #: local about text that had entirely left the machine.
    ran: str | None = None

    @property
    def destination(self) -> str:
        if self.left_machine is None:
            return "not known"
        return "sent out" if self.left_machine else "this machine"


def record(store: list, *, surface: str, engine: str, is_local: bool,
           latency_ms: int, chars: int, when: float,
           left_machine: Any = _UNSET, metered: bool = False,
           ran: str | None = None) -> None:
    """Append an entry, newest last, bounded."""
    store.append(asdict(LedgerEntry(
        surface=surface, engine=engine, is_local=bool(is_local),
        latency_ms=int(latency_ms), chars=int(chars), when=float(when),
        left_machine=(not bool(is_local)) if left_machine is _UNSET else left_machine,
        metered=bool(metered), ran=None if ran is None else str(ran),
    )))
    if len(store) > MAX_ENTRIES:
        del store[:len(store) - MAX_ENTRIES]


def record_run(store: list, *, surface: str, run, latency_ms: int, chars: int,
               when: float) -> None:
    """Record a completed request from its `policy.EngineRun` receipt.

    The preferred entry point, because it removes the caller's ability to
    disagree with policy about where the text went. The old call site derived
    `is_local = engine.startswith("local")` on its own, which filed cache hits
    under "sent out" while the same request's JSON said it was free.
    """
    record(store, surface=surface, engine=run.engine,
           is_local=(run.left_machine is False), latency_ms=latency_ms,
           chars=chars, when=when, left_machine=run.left_machine,
           metered=bool(run.metered),
           ran=getattr(run.ran, "value", run.ran))


def left_machine_of(row: dict[str, Any]) -> bool | None:
    """Tri-state destination for a stored row, tolerating rows written before
    `left_machine` existed (a live session's storage outlives a deploy)."""
    if "left_machine" in row:
        return row["left_machine"]
    return not row.get("is_local")


def voice_state(target_language: str | None = None) -> dict[str, Any]:
    """What /engines says about voice, from the RESOLVED tri-state.

    Not from the raw env var. Local voice is now on by default when the models
    are present (DECISIONS.md §2), so `PASSAGE_LOCAL_VOICE` being unset no
    longer means "hosted" — reading the variable would make the page claim
    metered hosted speech on a machine where every recording stays put. This
    page has made exactly that class of mistake before: it once printed
    "hosted — metered" directly above "100% stayed on this machine".

    The privacy sentence belongs in here rather than beside it. This helper was
    for a while a probe with no production caller — /engines assembled its own
    voice block from policy + local_voice — and the bug it existed to prevent
    promptly reappeared in the block it was not wired into. One probe, one page.

    `target_language` is optional and normally absent: /engines is rendered per
    page load and does not know the visitor's target. Absent means the privacy
    line describes capability instead of promising anything.
    """
    from passage import local_voice, policy
    state = local_voice.status()
    return {
        "mode": state["mode"],
        "engine": state["stt_engine"],
        "is_local": state["stt_ready"],
        "detail": local_voice.describe(),
        "forced_but_missing": state["forced_but_missing"],
        "voices": list(state["voices"]),
        "privacy": policy.describe_voice_privacy(target_language),
    }


def served_from_cache(row: dict[str, Any]) -> bool:
    """Whether this row is a re-read rather than a run.

    Reads the recorded `ran`, falling back to the pre-demotion convention
    where a cache hit's `engine` column literally said "cache" (a live
    session's storage outlives a deploy). A row with neither marker is taken
    to be a real run, which is what it meant when it was written.
    """
    ran = row.get("ran")
    if ran:
        return str(ran).lower() == "cache"
    return (row.get("engine") or "").strip().lower() == "cache"


def summarise(entries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Local vs cloud, in the terms someone would actually ask about.

    Reports characters as well as counts: one document is one entry but a lot
    of text, so a count alone would say "mostly local" about a session that
    sent a whole report to a hosted model.

    CACHE HITS ARE NOT NEW TEXT AND ARE NOT COUNTED AS ANY OF IT. A cache row
    carries the full character count of text that was already recorded once,
    so folding it into a chars-weighted privacy share re-counts the same
    sentence: five identical translations of one 44-character sentence, one of
    which was actually sent to a hosted model, rendered "80% of your text
    stayed on this machine · local: 4 runs · 176 chars". The user had 44
    characters and all of them left. Everything local-vs-sent-out here is
    therefore computed over rows where something actually ran, and re-reads
    are reported on their own line where they cannot inflate anything.
    """
    rows = list(entries)
    # Three kinds of row, not two. A row where NOTHING was asked of anything
    # (empty input: a picture-only DOCX, a blank box) is neither a run nor a
    # re-read, so it is evidence of no destination and of no cache. Filing it
    # with the runs printed "local: 1 runs · 0 chars" for a document that was
    # never translated at all.
    cache: list[dict[str, Any]] = []
    nothing: list[dict[str, Any]] = []
    ran_rows: list[dict[str, Any]] = []
    for r in rows:
        if served_from_cache(r):
            cache.append(r)
        elif (r.get("ran") or "").lower() == "nothing":
            nothing.append(r)
        else:
            ran_rows.append(r)
    local = [r for r in ran_rows if left_machine_of(r) is False]
    remote = [r for r in ran_rows if left_machine_of(r) is True]
    unknown = [r for r in ran_rows if left_machine_of(r) is None]

    def stats(subset: list[dict[str, Any]]) -> dict[str, Any]:
        latencies = sorted(r.get("latency_ms", 0) for r in subset)
        return {
            "runs": len(subset),
            "chars": sum(r.get("chars", 0) for r in subset),
            "median_ms": latencies[len(latencies) // 2] if latencies else None,
        }

    # HOW MANY TIMES A MODEL RAN — not how many rows name it. A cache row
    # names the engine that produced the bytes, which is the right thing on
    # that row's own receipt and the wrong thing in a run-count histogram:
    # one hosted call answering five identical requests printed
    # "hosted:gpt-5.4-nano — 5 runs" two lines under a claim that most of the
    # session stayed local.
    engines: dict[str, int] = {}
    for r in ran_rows:
        engines[r.get("engine", "?")] = engines.get(r.get("engine", "?"), 0) + 1

    new_chars = sum(r.get("chars", 0) for r in ran_rows)
    local_chars = sum(r.get("chars", 0) for r in local)
    return {
        "total_runs": len(rows),
        # Rows where a model (or nothing at all, for empty input) actually
        # served the request. `total_runs` still counts every row because the
        # page's "nothing translated yet" test is about the whole ledger.
        "runs_of_new_text": len(ran_rows),
        "local": stats(local),
        "remote": stats(remote),
        # Runs whose engine label nobody could classify. Reported separately
        # rather than folded into either side: counting them as local would
        # inflate a privacy claim, counting them as remote would invent a
        # disclosure that may not have happened.
        "unknown": stats(unknown),
        # Re-reads: the same words, already counted above on the run that
        # produced them. Never local evidence, never remote evidence.
        "cache": stats(cache),
        "new_chars": new_chars,
        # None when there is no new text to describe — a session of nothing
        # but cache hits has no privacy share, and printing 0% or 100% for it
        # would be inventing one. The page has to be able to say "no answer".
        "local_share_of_chars": (round(local_chars / new_chars, 3)
                                 if new_chars else None),
        "metered_runs": sum(1 for r in rows if r.get("metered")),
        "metered_chars": sum(r.get("chars", 0) for r in rows if r.get("metered")),
        "engines": dict(sorted(engines.items(), key=lambda kv: -kv[1])),
    }
