"""Where work may run, what may be kept, and what gets billed.

These are David's three decisions from 2026-07-27, put in one place so they are
enforced rather than remembered:

1. **Bring your own key or your own machine and it isn't metered.** If Passage
   didn't pay for the inference, it doesn't bill for it. Revenue comes from the
   things that stay Passage's problem — durable storage, traces, history.
2. **Documents may persist segments and traces; original files may not.** That
   keeps the decision-review and per-user-preference value while declining to
   become a document store, which is a liability rather than a feature. Live
   typing and camera frames persist nothing at all: they're transient by
   nature and the camera case is somebody standing in a restaurant.
3. **Local-first stays the default when a local model is reachable**, and the
   engine label is what makes that honest instead of sneaky.

One module rather than conditionals spread across the app, because these are
the rules most likely to drift silently as features land — and a data-retention
rule that drifts is the kind of bug you find out about from someone else.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Surface(str, Enum):
    """Where a piece of work came from. Retention differs per surface, so the
    caller has to say which one it is."""

    LIVE_TEXT = "live_text"      # keystroke preview
    TEXT = "text"                # an explicit text translation
    DOCUMENT = "document"        # an uploaded file
    IMAGE = "image"              # camera / photo OCR
    VOICE = "voice"              # recorded speech


@dataclass(frozen=True)
class Retention:
    """What may be kept for a given surface, at two very different durations.

    The distinction matters more than it first looks. `session_history` is the
    Recent Threads drawer: server-side but keyed to one browser session,
    thrown away when it ends, and the reason the app is pleasant to use. It is
    NOT what "documents go to the cloud, camera stays local" was about.
    `durable_*` is the part that outlives the session, crosses devices, and
    carries a deletion obligation — the part actually worth deciding.

    Collapsing the two would have quietly deleted Recent Threads for text mode
    while claiming to implement a privacy decision.
    """

    session_history: bool
    durable_text: bool
    durable_segments: bool
    durable_traces: bool
    durable_original_file: bool
    reason: str

    @property
    def persists_durably(self) -> bool:
        return any((self.durable_text, self.durable_segments,
                    self.durable_traces, self.durable_original_file))


#: The original file is False everywhere on purpose. Segments and traces carry
#: the reviewable value; keeping the upload itself adds storage, exposure and
#: a deletion obligation without adding product.
_RETENTION = {
    Surface.DOCUMENT: Retention(
        session_history=True, durable_text=True, durable_segments=True,
        durable_traces=True, durable_original_file=False,
        reason="A document is a durable artifact people come back to, and its "
               "segments plus edit history are the reviewable value — the "
               "per-user preference dataset is a read over exactly those. The "
               "uploaded file itself is storage, exposure and a deletion "
               "obligation without adding product."),
    Surface.TEXT: Retention(
        session_history=True, durable_text=True, durable_segments=False,
        durable_traces=True, durable_original_file=False,
        reason="An explicit translation someone asked for is worth keeping; it "
               "has no segment structure to keep alongside it."),
    Surface.LIVE_TEXT: Retention(
        session_history=True, durable_text=False, durable_segments=False,
        durable_traces=False, durable_original_file=False,
        reason="Keystroke previews belong in the session drawer — that is what "
               "makes the app pleasant — but nothing half-typed should outlive "
               "the session or follow you to another device. Thread "
               "continuation already collapses one sentence into one entry."),
    Surface.IMAGE: Retention(
        session_history=True, durable_text=False, durable_segments=False,
        durable_traces=False, durable_original_file=False,
        reason="Somebody is pointing a camera at a menu. The result is useful "
               "for the next minute; a photograph of wherever someone happens "
               "to be standing is the last thing to keep."),
    Surface.VOICE: Retention(
        session_history=True, durable_text=False, durable_segments=False,
        durable_traces=False, durable_original_file=False,
        reason="Recorded speech is the most sensitive input here and the least "
               "useful to keep. Transcripts stay in the session only."),
}


def retention_for(surface: Surface) -> Retention:
    return _RETENTION[Surface(surface)]


def may_persist(surface: Surface, field: str) -> bool:
    """Whether `field` may be written down for this surface."""
    policy = retention_for(surface)
    if not hasattr(policy, field):
        raise ValueError(f"Unknown retention field: {field!r}")
    return bool(getattr(policy, field))


def is_metered(profile, *, local_first_model: str | None = None) -> bool:
    """Whether this run should count against a quota.

    A single question with a single answer: did Passage pay for the inference?
    Local models and a user's own key both mean no. Metering must read this
    rather than inspect provider details itself, so billing cannot drift away
    from routing.

    `local_first_model` is part of that answer, not a detail: with no explicit
    profile the live path still runs locally when a model is reachable, and
    billing for inference that happened on the user's own GPU would be
    charging for electricity someone else paid for.
    """
    if profile is None:
        return local_first_model is None
    return bool(getattr(profile, "is_metered", True))


def data_leaves_machine(profile, *, local_first_model: str | None = None) -> bool:
    """Whether text is sent off this machine at all.

    `local_first_model` matters and was originally missed: with no explicit
    profile the app still routes the live path to a local model when one is
    reachable, so answering purely from the profile claimed "hosted, metered"
    on a page that simultaneously reported 100% of the text had stayed on the
    machine. A privacy line that contradicts the ledger beside it is worse
    than none — this is the one claim that has to be right.
    """
    if profile is None:
        return local_first_model is None
    return not bool(getattr(profile, "uses_local_inference", False))


def voice_stays_local() -> bool:
    """Whether a recording is processed on this machine.

    Voice deserves its own answer rather than inheriting the text one. It is
    the most sensitive input the app takes — a recording of someone's actual
    voice — and until local speech existed, "nothing leaves your machine" was
    simply false for it no matter which text model was selected.
    """
    from passage import local_voice
    return local_voice.stt_available()


def describe_voice_privacy(target_language: str | None = None) -> str:
    from passage import local_voice
    if not local_voice.stt_available():
        return "Recordings are sent to Passage's hosted speech models."
    if local_voice.tts_available(target_language):
        return "Recording and playback both run on this machine."
    return ("Your recording is transcribed on this machine; only the translated "
            "text is sent out to be spoken.")


def describe_privacy(profile, *, local_first_model: str | None = None) -> str:
    """One line a user can act on, for the UI."""
    if not data_leaves_machine(profile, local_first_model=local_first_model):
        if profile is None:
            return (f"Runs on {local_first_model} on this machine — text isn't sent "
                    "anywhere unless that model is unavailable.")
        return "Runs on your machine — text isn't sent anywhere."
    if not is_metered(profile):
        return "Runs on your own endpoint — not metered by Passage."
    return "Runs on Passage's hosted models — metered."
