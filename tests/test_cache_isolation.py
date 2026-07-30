"""Cross-user isolation of the shared translation cache.

The backend is ONE process-wide object shared by every connected client
(``TranslationUI.shared_backend``), and its translation cache used to be keyed
only on (text, target, mode). That meant a BYO/private-endpoint user's output
was served verbatim to a different user, labelled as that second user's own
engine. These tests drive the real cache through the real backend methods --
no ``__new__`` bypasses, no hand-assigned attributes -- and assert on the
OUTPUT BYTES and the ENGINE LABEL, never merely that a call returned.

Every test here is written so it FAILS if the wrong engine served the request:
each fake endpoint stamps its own identity into the text it returns, and call
counts are recorded per endpoint.
"""

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import openai

import TranslationBackend as tb
from TranslationBackend import BoundedTranslationCache, TranslationBackend
from passage import provider_profiles as pp


# ─────────────────────────── endpoint doubles ─────────────────────────── #


class _Endpoints:
    """Registry of fake OpenAI-compatible endpoints, keyed by base_url.

    ``None`` is the hosted default. Each endpoint returns text that names
    itself, so an assertion on the returned string proves WHICH engine ran.
    """

    def __init__(self) -> None:
        self.replies: dict[object, str] = {}
        self.calls: dict[object, int] = {}

    def register(self, base_url, reply: str) -> None:
        self.replies[base_url] = reply
        self.calls.setdefault(base_url, 0)

    def count(self, base_url) -> int:
        return self.calls.get(base_url, 0)


class _FakeCompletions:
    def __init__(self, endpoints: _Endpoints, base_url) -> None:
        self._endpoints = endpoints
        self._base_url = base_url

    def create(self, **_kwargs):
        key = self._base_url
        if key not in self._endpoints.replies:
            raise AssertionError(f"unexpected call to unregistered endpoint {key!r}")
        self._endpoints.calls[key] = self._endpoints.calls.get(key, 0) + 1
        content = self._endpoints.replies[key]
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
        )


class _FakeOpenAI:
    """Stands in for ``openai.OpenAI`` so ChatCompletionsProvider is built by
    the production code path (provider_for_profile -> ChatCompletionsProvider
    -> openai.OpenAI) rather than assembled by the test."""

    endpoints: _Endpoints

    def __init__(self, **kwargs):
        self.api_key = kwargs.get("api_key")
        self.base_url = kwargs.get("base_url")
        self.chat = types.SimpleNamespace(
            completions=_FakeCompletions(type(self).endpoints, self.base_url)
        )


@pytest.fixture
def endpoints(monkeypatch):
    registry = _Endpoints()
    monkeypatch.setattr(_FakeOpenAI, "endpoints", registry, raising=False)
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    return registry


@pytest.fixture
def backend(monkeypatch, endpoints):
    monkeypatch.setenv("OPENAI_API_KEY", "hosted-key")
    monkeypatch.setenv("TRANSLATION_PROVIDER", "openai")
    endpoints.register(None, "<<HOSTED>> translation")
    return TranslationBackend()


def _byo(base_url: str, model: str = "translategemma:4b", api_key: str = "secret") -> pp.ProviderProfile:
    return pp.ProviderProfile(
        label="BYO", kind=pp.KIND_BYO, base_url=base_url, api_key=api_key, model=model
    )


SECRET = "El código de combinación del cofre es 4471, no lo compartas con nadie."


# ───────────────────────── the exact reproduction ─────────────────────── #


def test_byo_output_is_never_served_to_a_hosted_user(backend, endpoints):
    """The reported P0, verbatim.

    User A translates through their own private endpoint inside
    ``using_profile``. User B, a different session with no profile, asks for
    the same sentence and must get the HOSTED answer, produced by a real
    hosted call -- not A's bytes.
    """
    byo_url = "https://user-a.example.com/v1"
    endpoints.register(byo_url, "<<BYO-USER-A>> translation")

    with backend.cache_scope("session-A"), backend.using_profile(_byo(byo_url)):
        user_a_output = backend.translate_text(SECRET, "English")

    hosted_calls_before = endpoints.count(None)
    with backend.cache_scope("session-B"):
        user_b_output = backend.translate_text(SECRET, "English")
    hosted_calls_for_user_b = endpoints.count(None) - hosted_calls_before

    assert user_a_output == "<<BYO-USER-A>> translation"
    assert user_b_output == "<<HOSTED>> translation"
    assert user_b_output != user_a_output, "user B received user A's private-endpoint output"
    assert hosted_calls_for_user_b == 1, "user B was answered from cache, not by their own engine"


def test_private_content_does_not_cross_sessions_on_the_same_engine(backend, endpoints):
    """The leak is a privacy defect, not only a labelling one.

    Same hosted engine, two sessions: session B must still be answered by a
    real call. The guaranteed property is that cached user CONTENT never
    crosses a cache scope, whatever engine produced it.
    """
    endpoints.replies[None] = "<<HOSTED-A>> " + SECRET
    with backend.cache_scope("session-A"):
        a_out = backend.translate_text(SECRET, "English")

    endpoints.replies[None] = "<<HOSTED-B>> fresh"
    with backend.cache_scope("session-B"):
        b_out = backend.translate_text(SECRET, "English")

    assert a_out == "<<HOSTED-A>> " + SECRET
    assert b_out == "<<HOSTED-B>> fresh", "session B was served session A's cached content"
    assert SECRET not in b_out
    assert endpoints.count(None) == 2


# ───────────────────── engine identity is a key dimension ─────────────── #


def test_two_different_byo_endpoints_never_share_even_in_one_scope(backend, endpoints):
    url_a, url_b = "https://a.example.com/v1", "https://b.example.com/v1"
    endpoints.register(url_a, "<<ENDPOINT-A>>")
    endpoints.register(url_b, "<<ENDPOINT-B>>")

    with backend.cache_scope("same-scope"):
        with backend.using_profile(_byo(url_a)):
            out_a = backend.translate_text("hello", "Spanish")
        with backend.using_profile(_byo(url_b)):
            out_b = backend.translate_text("hello", "Spanish")

    assert out_a == "<<ENDPOINT-A>>"
    assert out_b == "<<ENDPOINT-B>>", "endpoint B's user was served endpoint A's output"
    assert endpoints.count(url_a) == 1 and endpoints.count(url_b) == 1


def test_same_endpoint_different_api_key_is_a_different_identity(backend, endpoints):
    """A shared base_url with a different credential is a different tenant."""
    url = "https://shared.example.com/v1"
    endpoints.register(url, "<<SHARED-1>>")

    with backend.cache_scope("scope"):
        with backend.using_profile(_byo(url, api_key="key-one")):
            first = backend.translate_text("hello", "Spanish")
        endpoints.replies[url] = "<<SHARED-2>>"
        with backend.using_profile(_byo(url, api_key="key-two")):
            second = backend.translate_text("hello", "Spanish")

    assert first == "<<SHARED-1>>"
    assert second == "<<SHARED-2>>"
    assert endpoints.count(url) == 2


def test_hosted_result_is_not_served_to_a_byo_request(backend, endpoints):
    """The reverse direction of the P0: hosted bytes under a BYO label."""
    url = "https://user.example.com/v1"
    endpoints.register(url, "<<BYO>>")

    with backend.cache_scope("scope"):
        hosted = backend.translate_text("hello", "Spanish")
        with backend.using_profile(_byo(url)):
            byo = backend.translate_text("hello", "Spanish")

    assert hosted == "<<HOSTED>> translation"
    assert byo == "<<BYO>>", "a BYO request was answered with the hosted cache entry"
    assert endpoints.count(url) == 1


def test_cache_still_works_within_one_session_and_engine(backend, endpoints):
    """Isolation must not have been bought by disabling the cache."""
    url = "https://user.example.com/v1"
    endpoints.register(url, "<<BYO>>")

    with backend.cache_scope("scope"), backend.using_profile(_byo(url)):
        first = backend.translate_text("hello", "Spanish")
        second = backend.translate_text("  hello ", " spanish ")

    assert first == second == "<<BYO>>"
    assert endpoints.count(url) == 1, "the cache stopped serving legitimate repeats"


def test_instruction_refinement_is_scoped_too(backend, endpoints):
    endpoints.replies[None] = "<<HOSTED-A>> refined"
    with backend.cache_scope("session-A"):
        a = backend.translate_text_with_instructions("draft", "French", "formal")
    endpoints.replies[None] = "<<HOSTED-B>> refined"
    with backend.cache_scope("session-B"):
        b = backend.translate_text_with_instructions("draft", "French", "formal")

    assert a == "<<HOSTED-A>> refined"
    assert b == "<<HOSTED-B>> refined", "the refine path leaked across sessions"


# ──────────────────────────── the live path ───────────────────────────── #


def _install_live_local(backend, endpoints, url="http://127.0.0.1:11434/v1", model="translategemma:4b"):
    """Make the live path's local branch take, without touching the cache.

    Only the reachability probe is short-circuited; the provider itself is a
    real ChatCompletionsProvider built the production way.
    """
    endpoints.register(url, f"<<LOCAL {model}>>")
    backend._live_provider = tb.ChatCompletionsProvider(
        api_key="ollama", base_url=url, text_model=model
    )
    backend._live_reachable = True
    backend._live_probe_at = tb.time.time()
    return url


def test_live_no_profile_sessions_do_not_share_the_auto_bucket(backend, endpoints, monkeypatch):
    """Every profile-less session used to share one ``live:auto`` key."""
    monkeypatch.setattr(tb, "LIVE_LOCAL_ENABLED", True, raising=False)
    url = _install_live_local(backend, endpoints)

    with backend.cache_scope("session-A"):
        a_text, a_engine = backend.translate_live(SECRET, "English")
    endpoints.replies[url] = "<<LOCAL fresh>>"
    with backend.cache_scope("session-B"):
        b_text, b_engine = backend.translate_live(SECRET, "English")

    assert a_text == "<<LOCAL translategemma:4b>>"
    assert b_text == "<<LOCAL fresh>>", "session B was served session A's live output"
    assert a_engine == "local:translategemma:4b"
    assert b_engine == "local:translategemma:4b"
    assert endpoints.count(url) == 2


def test_live_cache_hit_reports_the_engine_that_produced_the_bytes(backend, endpoints, monkeypatch):
    """The live path used to label every cache hit ``"cache"`` -- and in the
    live reproduction user B saw engine ``"cache"`` for user A's local output.
    A hit must name the engine that actually made the text."""
    monkeypatch.setattr(tb, "LIVE_LOCAL_ENABLED", True, raising=False)
    url = _install_live_local(backend, endpoints)

    with backend.cache_scope("session-A"):
        first_text, first_engine = backend.translate_live("hello", "Spanish")
        second_text, second_engine = backend.translate_live("hello", "Spanish")

    assert endpoints.count(url) == 1, "expected the second call to be a cache hit"
    assert second_text == first_text
    assert second_engine == first_engine == "local:translategemma:4b"
    assert second_engine != "cache"


def test_live_byo_output_is_not_served_to_a_local_session(backend, endpoints, monkeypatch):
    """The live half of the P0: A on a private endpoint, B on the local model.

    B must get local bytes with a local label, and A's endpoint must not be
    the thing that answered B.
    """
    monkeypatch.setattr(tb, "LIVE_LOCAL_ENABLED", True, raising=False)
    local_url = _install_live_local(backend, endpoints)
    byo_url = "https://user-a.example.com/v1"
    endpoints.register(byo_url, "<<BYO-USER-A>> translation")
    profile = _byo(byo_url)

    with backend.cache_scope("session-A"):
        a_text, a_engine = backend.translate_live(SECRET, "English", profile=profile)
    with backend.cache_scope("session-B"):
        b_text, b_engine = backend.translate_live(SECRET, "English")

    assert a_text == "<<BYO-USER-A>> translation"
    assert a_engine == profile.describe()
    assert b_text == "<<LOCAL translategemma:4b>>", "user B received user A's BYO output"
    assert b_engine == "local:translategemma:4b"
    assert endpoints.count(byo_url) == 1, "user B's request hit user A's private endpoint"


def test_live_same_endpoint_different_profile_ids_share_within_a_session(backend, endpoints, monkeypatch):
    """Identity is the endpoint, not the profile id: re-saving a profile used
    to invalidate a perfectly valid cache entry."""
    monkeypatch.setattr(tb, "LIVE_LOCAL_ENABLED", True, raising=False)
    url = "https://user.example.com/v1"
    endpoints.register(url, "<<BYO>>")
    first_profile = _byo(url)
    second_profile = _byo(url)
    assert first_profile.id != second_profile.id

    with backend.cache_scope("session-A"):
        one, _ = backend.translate_live("hello", "Spanish", profile=first_profile)
        two, _ = backend.translate_live("hello", "Spanish", profile=second_profile)

    assert one == two == "<<BYO>>"
    assert endpoints.count(url) == 1


# ──────────────────────────── memory bound ────────────────────────────── #


def test_cache_is_bounded_and_evicts_oldest_first():
    cache = BoundedTranslationCache(max_entries=3)
    for i in range(10):
        cache[("scope", "engine", f"text{i}", "es", "translate")] = i

    assert len(cache) == 3
    assert cache.get(("scope", "engine", "text0", "es", "translate")) is None
    assert cache.get(("scope", "engine", "text9", "es", "translate")) == 9


def test_cache_bound_holds_when_sessions_multiply(backend, endpoints):
    """Per-session partitioning must not turn into an unbounded leak."""
    backend.translation_cache.max_entries = 5
    for i in range(40):
        endpoints.replies[None] = f"<<HOSTED {i}>>"
        with backend.cache_scope(f"session-{i}"):
            backend.translate_text("hello", "Spanish")

    assert len(backend.translation_cache) == 5


# ─────────────────────── how the scope is resolved ───────────────────── #


def test_scope_falls_back_to_an_ambient_session_identity(monkeypatch):
    """No explicit scope: the ambient per-browser/per-client identity is used.

    This is the production wiring -- nothing in the UI has to remember to pass
    a scope for isolation to hold.
    """
    monkeypatch.setattr(tb, "_ambient_cache_scope", lambda: "browser:abc123")
    assert tb.current_cache_scope() == "browser:abc123"

    monkeypatch.setattr(tb, "_ambient_cache_scope", lambda: None)
    assert tb.current_cache_scope() == "shared"


def test_explicit_scope_wins_over_ambient(monkeypatch, backend):
    monkeypatch.setattr(tb, "_ambient_cache_scope", lambda: "browser:abc123")
    with backend.cache_scope("session-X"):
        assert tb.current_cache_scope() == "session-X"
    assert tb.current_cache_scope() == "browser:abc123"


def test_document_worker_thread_keeps_the_submitting_sessions_scope(monkeypatch, backend, endpoints):
    """ContextVars do not cross threads, so the document job captures the scope
    at submit time. Without that, every document job would collapse into the
    one shared partition -- which is the leak again."""
    from threading import Thread

    monkeypatch.setattr(tb, "_ambient_cache_scope", lambda: None)
    seen: list[str] = []

    with backend.cache_scope("session-A"):
        submit_scope = tb.current_cache_scope()

        def naive():
            seen.append(tb.current_cache_scope())

        def captured():
            with backend.cache_scope(submit_scope):
                seen.append(tb.current_cache_scope())

        for target in (naive, captured):
            thread = Thread(target=target)
            thread.start()
            thread.join()

    assert seen == ["shared", "session-A"]


def test_backend_cache_default_bound_is_finite(backend):
    assert backend.translation_cache.max_entries == tb.CACHE_MAX_ENTRIES
    assert 0 < tb.CACHE_MAX_ENTRIES < 100_000
