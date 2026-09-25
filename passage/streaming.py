"""Streaming a translation out of the one place model calls are made.

Every text model call leaves through a provider's `create_chat_completion`.
That is the seam the router, provenance and every test fake are built on, so
streaming does not add a second way out. A request that wants the answer as it
is written installs a `StreamSink` for its duration; a provider that finds one
asks the model to stream, hands each piece to the sink, and still returns the
whole completion to its caller. Code between the request and the provider
(translate_live, translate_text, chunking, the caches, the retry loop) is
unchanged, and a provider or fake that cannot stream simply delivers its
answer as one piece.

Cancellation travels the same way. A superseded keystroke sets `cancel`; the
provider closes the model's response at the next piece, which is what stops
generation (and, on a metered API, stops billing output tokens), and raises
`StreamCancelled` so nothing half-written is cached or shown as final.
"""
from __future__ import annotations

import re
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable

#: Masked URLs and emails ("[[PSG:0]]") are restored only once the whole
#: answer is in. Mid-stream, complete placeholders and a half-written one at
#: the end are hidden rather than flashed on screen.
_PLACEHOLDERS = re.compile(r"\[\[\s*PSG\s*:\s*\d+\s*\]\]|\[(?:\[[^\]]*)?\]?$")


class StreamCancelled(Exception):
    """The request that wanted this answer has gone away."""


class StreamSink:
    def __init__(self, on_text: Callable[[str], None], cancel: threading.Event | None = None):
        self.on_text = on_text
        self.cancel = cancel or threading.Event()
        self.raw = ""
        #: True once any model has produced text for this request. After
        #: that, falling back to another engine would splice two answers.
        self.started = False

    def begin_call(self) -> None:
        """A new model call is starting (the next chunk of a long text)."""
        self.check()
        if self.raw.strip():
            self.raw += " "

    def feed(self, piece: str) -> None:
        self.check()
        if not piece:
            return
        self.started = self.started or bool(piece.strip())
        self.raw += piece
        self.on_text(_PLACEHOLDERS.sub("", self.raw).strip())

    def reset(self) -> None:
        """Forget text from an engine that failed before saying anything."""
        self.raw = ""

    def check(self) -> None:
        if self.cancel.is_set():
            raise StreamCancelled()


_sink: ContextVar[StreamSink | None] = ContextVar("passage_stream_sink", default=None)


def current() -> StreamSink | None:
    return _sink.get()


@contextmanager
def streaming(sink: StreamSink):
    token = _sink.set(sink)
    try:
        yield sink
    finally:
        _sink.reset(token)
