"""Interim hardening for the public /api/* surface.

Until real accounts land (Supabase, Phase 4 in PASSAGE_PLAN.md), the API is
gated by a short-lived signed token that only pages served by this app embed.
This stops direct scripted use of the metered translation endpoints; it is
not authentication. Set PASSAGE_PUBLIC_API=1 to disable the gate entirely.
"""

import hashlib
import hmac
import os
import secrets
import threading
import time

TOKEN_TTL_SECONDS = 12 * 60 * 60
RATE_WINDOW_SECONDS = 60

MAX_TEXT_CHARS = int(os.getenv("PASSAGE_MAX_TEXT_CHARS", "8000"))
MAX_UPLOAD_BYTES = int(os.getenv("PASSAGE_MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))


class ApiGuard:
    """Boot-scoped token issuer/validator plus a per-IP sliding-window rate limit."""

    def __init__(self, max_requests_per_window: int | None = None):
        self._secret = secrets.token_bytes(32)
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        self.max_requests = max_requests_per_window or int(
            os.getenv("PASSAGE_API_RATE_LIMIT", "30")
        )

    def issue_token(self, now: float | None = None) -> str:
        timestamp = str(int(now if now is not None else time.time()))
        signature = hmac.new(self._secret, timestamp.encode(), hashlib.sha256).hexdigest()
        return f"{timestamp}.{signature}"

    def validate_token(self, token: str | None, now: float | None = None) -> bool:
        if not token or "." not in token:
            return False
        timestamp, signature = token.split(".", 1)
        if not timestamp.isdigit():
            return False
        expected = hmac.new(self._secret, timestamp.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return False
        age = (now if now is not None else time.time()) - int(timestamp)
        return 0 <= age < TOKEN_TTL_SECONDS

    def allow_request(self, client_key: str, now: float | None = None) -> bool:
        current = now if now is not None else time.time()
        with self._lock:
            recent = [
                t for t in self._hits.get(client_key, [])
                if current - t < RATE_WINDOW_SECONDS
            ]
            if len(recent) >= self.max_requests:
                self._hits[client_key] = recent
                return False
            recent.append(current)
            self._hits[client_key] = recent
            return True


def client_ip(request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    client = getattr(request, "client", None)
    return client.host if client else "unknown"


def gate_disabled() -> bool:
    return os.getenv("PASSAGE_PUBLIC_API") == "1"
