"""Bring-your-own-endpoint profiles, scoped to a browser session.

A profile is somewhere to send translation work: the app's own hosted key, a
local Ollama, or a user's own OpenAI-compatible endpoint and key. Three things
shape this module:

**Session-scoped, not process-scoped.** Provider selection was parked for a
long time because `TranslationBackend` is ONE instance shared by every
connected client, so a picker would have been a process-wide toggle affecting
everybody's session — new debt in the same shape as the cross-user bugs already
fixed here. Since `TranslationUI()` is constructed fresh per page load and
`app.storage.user` is keyed by session cookie, a profile can now belong to one
visitor without any database.

**The key never leaves the server.** `app.storage.user` is server-side storage
keyed by a signed cookie, so a user's own API key is stored there and only a
redacted form is ever rendered, returned by an endpoint, or logged. `redacted()`
is the only representation anything outside this module should show.

**BYO means unmetered.** If the work runs on the user's own key or their own
machine, the app is not paying for it, so it must not be billed for it either.
`is_metered` is the single place that decides, so metering (when it lands)
cannot drift from provider selection.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, asdict, field
from typing import Any

#: Profile kinds. "app" is Passage's own hosted key (the only metered one);
#: "local" is an endpoint on the user's machine; "byo" is the user's own
#: credentials against someone else's API.
KIND_APP = "app"
KIND_LOCAL = "local"
KIND_BYO = "byo"


@dataclass
class ProviderProfile:
    label: str
    kind: str = KIND_BYO
    base_url: str | None = None
    api_key: str = ""
    model: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def is_metered(self) -> bool:
        """Only Passage's own hosted key costs Passage money."""
        return self.kind == KIND_APP

    @property
    def uses_local_inference(self) -> bool:
        return self.kind == KIND_LOCAL

    def redacted(self) -> dict[str, Any]:
        """The only shape safe to render, return over HTTP, or log."""
        data = asdict(self)
        key = data.pop("api_key", "") or ""
        data["api_key_hint"] = f"…{key[-4:]}" if len(key) >= 4 else ("set" if key else "")
        data["is_metered"] = self.is_metered
        return data

    def describe(self) -> str:
        """Short human label for the status line, e.g. 'byo:gpt-4o-mini'."""
        return f"{self.kind}:{self.model or 'default'}"


def app_default_profile(model: str) -> ProviderProfile:
    return ProviderProfile(label="Passage (hosted)", kind=KIND_APP, model=model, id="app-default")


def local_profile(base_url: str, model: str) -> ProviderProfile:
    return ProviderProfile(label="Local", kind=KIND_LOCAL, base_url=base_url, model=model, id="local-default")


def validate(base_url: str, api_key: str, model: str) -> str | None:
    """Return a user-facing problem with these settings, or None if usable.

    Deliberately permissive about which host — the whole point is that someone
    can point this at their own server — but a base_url has to at least be an
    http(s) URL, and a remote endpoint needs a key.
    """
    base_url = (base_url or "").strip()
    if not model.strip():
        return "Pick a model name — the endpoint won't guess one for you."
    if not base_url:
        return "Enter the endpoint's base URL, ending in /v1."
    if not base_url.startswith(("http://", "https://")):
        return "The base URL must start with http:// or https://."
    is_loopback = any(h in base_url for h in ("localhost", "127.0.0.1", "[::1]"))
    if not is_loopback and not (api_key or "").strip():
        return "A remote endpoint needs an API key."
    return None


def from_stored(raw: dict[str, Any]) -> ProviderProfile:
    """Rebuild from app.storage.user, tolerating keys added in later versions."""
    allowed = {f for f in ProviderProfile.__dataclass_fields__}
    return ProviderProfile(**{k: v for k, v in raw.items() if k in allowed})
