"""A provider that speaks Ollama's native /api/chat instead of its
OpenAI-compatible shim.

The shim looked like the obvious choice — same surface as hosted, one code
path. Measured against real models, it silently loses output:

* `qwen3:30b` and `qwen3-vl:8b` return an EMPTY `content` through the shim.
  Their answer goes to a reasoning channel the shim doesn't expose, so both
  models look broken. Through /api/chat the same call returns
  `message.content` = "La junta directiva rechazó la recompra." with the
  working kept separately in `message.thinking`.
* `think: false` is not a general cure, and on the shim it makes things worse:
  the model writes 6,000 characters of deliberation INTO `content`. It is the
  right setting for EXTRACTION work — with it, qwen3-vl reads a whole menu
  photo in 2.6s — and the wrong one for translation. The provider therefore
  exposes it per call rather than picking a side.
* Images: /api/chat takes them as a plain base64 list on the message, which is
  what makes local vision (and therefore fully offline OCR) reachable at all.

So this exists to make locally-installed models usable as they actually
behave, not as the compatibility layer wishes they did. It deliberately mimics
the small slice of the OpenAI response shape the rest of the app reads
(`.choices[0].message.content`), so callers need no special-casing.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

DEFAULT_TIMEOUT_SECONDS = 600.0

#: Extra generation budget handed to thinking models. `num_predict` caps the
#: WHOLE generation, and a thinking model spends that budget on deliberation
#: BEFORE it writes any answer — asking qwen3:30b for a one-line translation
#: with num_predict=200 produced 754 characters of thinking and an empty
#: `content`, which looks exactly like a broken model. The reserve is what
#: makes these models usable rather than mysteriously silent.
THINKING_RESERVE_TOKENS = 2048

#: Model families that deliberate before answering.
THINKING_MODEL_PREFIXES = ("qwen3", "deepseek-r1", "magistral")


def is_thinking_model(model: str) -> bool:
    name = (model or "").lower()
    return any(name.startswith(p) for p in THINKING_MODEL_PREFIXES)


def suits_translation(model: str) -> bool:
    """Whether this model is a sensible choice for translation.

    Thinking models are not. Translation is transduction, not reasoning, and
    they price it as if it were: asked for a one-line Spanish translation,
    qwen3:30b produced 8,786 characters of deliberation and STILL no answer,
    even with a 2,048-token reserve on top of the request. There is no budget
    that reliably fixes that, because the deliberation is unbounded and the
    task never needed it.

    They stay available — a user can pick one deliberately, and the vision
    variant is genuinely useful for OCR with thinking turned off — but they
    are kept out of default rosters and comparisons, where they would burn
    time and look broken.
    """
    return not is_thinking_model(model)


@dataclass
class _Message:
    content: str
    #: The model's working, kept separate. Never treated as the answer: it is
    #: deliberation ("Got it, let's list out every line…"), and passing it off
    #: as output is exactly the fluent-but-wrong failure this codebase keeps
    #: designing against.
    thinking: str = ""

    @property
    def reasoning(self) -> str:  # what passage.compare looks for
        return self.thinking


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Completion:
    choices: list[_Choice]
    total_duration_ms: int | None = None


class NativeOllamaProvider:
    """Minimal client for Ollama's /api/chat.

    Not a BaseTranslationProvider subclass: it covers text and vision only.
    Voice stays on the hosted path, which is honest — there is no local STT/TTS
    here to route to, and pretending otherwise would fail deep inside a call.
    """

    is_openai_hosted = False

    def __init__(
        self, *, base_url: str, text_model: str, max_input_chars: int | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        # Accept either the OpenAI-style base ("…/v1") or the bare host, so a
        # profile written for the shim keeps working.
        self.host = base_url.rstrip("/").removesuffix("/v1")
        self.base_url = base_url
        self.text_model = text_model
        self.max_input_chars = max_input_chars
        self.timeout = timeout

    # ---- the slice of the OpenAI surface the rest of the app uses ---------

    def create_chat_completion(self, *, messages: list[dict[str, Any]], max_tokens: int = 1000,
                               images: list[str] | None = None, **_ignored: Any) -> _Completion:
        message = self.chat(messages, max_tokens=max_tokens, images=images)
        return _Completion(choices=[_Choice(message=message)])

    # ---- the native call --------------------------------------------------

    def chat(self, messages: list[dict[str, Any]], *, max_tokens: int = 1000,
             images: list[str] | None = None, fmt: str | None = None,
             think: bool | None = None) -> _Message:
        # Thinking is spent from the same budget as the answer, so a thinking
        # model needs headroom or it deliberates until it is cut off and
        # returns nothing (see THINKING_RESERVE_TOKENS).
        budget = max_tokens + (THINKING_RESERVE_TOKENS if is_thinking_model(self.text_model) else 0)
        payload: dict[str, Any] = {
            "model": self.text_model,
            "messages": [dict(m) for m in messages],
            "stream": False,
            "options": {"num_predict": budget},
        }
        if images:
            # Ollama attaches images to the last user message.
            payload["messages"][-1]["images"] = images
        if think is not None:
            # think=False is right for extraction work (OCR), where the answer
            # is transcription and deliberation is pure cost. It is WRONG as a
            # general fix: on a translation prompt it made qwen3 write its
            # reasoning into `content` instead of suppressing it.
            payload["think"] = think
        if fmt:
            # NOTE: format="json" plus a thinking model is a bad combination —
            # qwen3-vl spent 17,500 characters deliberating and emitted no
            # content at all. Callers that need structure from a thinking model
            # should ask for a simple line format in the prompt instead.
            payload["format"] = fmt

        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:200]
            raise RuntimeError(f"Ollama returned {error.code}: {detail}") from None
        except urllib.error.URLError as error:
            raise RuntimeError(f"Ollama unreachable at {self.host}: {error.reason}") from None

        message = body.get("message") or {}
        return _Message(
            content=(message.get("content") or "").strip(),
            thinking=(message.get("thinking") or ""),
        )

    def list_models(self) -> list[dict[str, Any]]:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5) as response:
                return json.loads(response.read().decode("utf-8")).get("models", [])
        except Exception as error:
            logging.info("[Ollama] tag list unavailable (%s)", error)
            return []

    # Voice is hosted-only; say so clearly rather than failing inside a call.
    def transcribe_audio(self, **_kwargs: Any):
        raise NotImplementedError(
            f"Voice transcription needs the hosted OpenAI provider; {self.host} has no speech model.")

    def synthesize_speech(self, **_kwargs: Any):
        raise NotImplementedError(
            f"Text-to-speech needs the hosted OpenAI provider; {self.host} has no speech model.")
