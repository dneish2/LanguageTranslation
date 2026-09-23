import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TranslationBackend import TranslationBackend


class _StubChatCreate:
    def __init__(self, responses=None, errors=None):
        self.calls = 0
        self._responses = list(responses or [])
        self._errors = list(errors or [])

    def __call__(self, **_kwargs):
        self.calls += 1
        if self._errors:
            err = self._errors.pop(0)
            if err is not None:
                raise err
        content = self._responses.pop(0) if self._responses else ""
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
        )


def _build_backend(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return TranslationBackend()


def test_translate_text_cache_hit_uses_normalized_key(monkeypatch):
    backend = _build_backend(monkeypatch)
    stub = _StubChatCreate(responses=["Hola mundo"])
    monkeypatch.setattr(backend.client.chat.completions, "create", stub)

    first = backend.translate_text("\t Hello   world  ", " Spanish ")
    second = backend.translate_text("Hello world", "spanish")

    assert first == "Hola mundo"
    assert second == "Hola mundo"
    assert stub.calls == 1


def test_translate_text_normalization_deduplicates_equivalent_inputs(monkeypatch):
    backend = _build_backend(monkeypatch)
    stub = _StubChatCreate(responses=["Bonjour monde"])
    monkeypatch.setattr(backend.client.chat.completions, "create", stub)

    first = backend.translate_text(" Hello\t world ", " French ")
    second = backend.translate_text("Hello   world", "french")

    assert first == "Bonjour monde"
    assert second == "Bonjour monde"
    assert stub.calls == 1


def test_translate_text_cache_metrics_not_double_counted(monkeypatch):
    backend = _build_backend(monkeypatch)
    stub = _StubChatCreate(responses=["Ciao mondo"])
    monkeypatch.setattr(backend.client.chat.completions, "create", stub)

    backend.translate_text("Hello world", "Italian")
    backend.translate_text("  Hello\tworld  ", " italian ")

    assert backend.metrics.cache_misses == 1
    assert backend.metrics.cache_hits == 1
    assert stub.calls == 1


def test_instruction_translation_cache_hit_uses_mode_and_instructions(monkeypatch):
    backend = _build_backend(monkeypatch)
    stub = _StubChatCreate(responses=["Veuillez reformuler"])
    monkeypatch.setattr(backend.client.chat.completions, "create", stub)

    first = backend.translate_text_with_instructions(
        "  Please rephrase this ", "French", " Use formal tone "
    )
    second = backend.translate_text_with_instructions(
        "Please rephrase this", " french ", "use   formal tone"
    )

    assert first == "Veuillez reformuler"
    assert second == "Veuillez reformuler"
    assert stub.calls == 1


def test_translate_text_retries_on_transient_openai_error(monkeypatch):
    backend = _build_backend(monkeypatch)
    backend.retry_base_delay = 0.01
    backend.retry_max_delay = 0.02

    transient = Exception("temporary outage")
    transient.status_code = 503
    stub = _StubChatCreate(errors=[transient, None], responses=["Recovered translation"])
    monkeypatch.setattr(backend.client.chat.completions, "create", stub)

    slept = []
    monkeypatch.setattr("TranslationBackend.time.sleep", lambda seconds: slept.append(seconds))

    translated = backend.translate_text("Resilient text", "German")

    assert translated == "Recovered translation"
    assert stub.calls == 2
    assert len(slept) == 1
    assert slept[0] >= backend.retry_base_delay


def test_translate_text_raises_after_max_attempts(monkeypatch):
    # A failed translation must surface as an error, never echo the source
    # text back as if it were translated.
    backend = _build_backend(monkeypatch)
    backend.max_openai_attempts = 3
    backend.retry_base_delay = 0
    backend.retry_max_delay = 0

    transient = Exception("still failing")
    transient.status_code = 503
    stub = _StubChatCreate(errors=[transient, transient, transient])
    monkeypatch.setattr(backend.client.chat.completions, "create", stub)
    monkeypatch.setattr("TranslationBackend.time.sleep", lambda _seconds: None)

    with pytest.raises(Exception, match="still failing"):
        backend.translate_text("Fallback text", "Italian")

    assert stub.calls == backend.max_openai_attempts


def test_finished_jobs_are_evicted_after_their_ttl_but_running_ones_are_not():
    import time as _time
    import TranslationBackend as tb

    backend = tb.TranslationBackend()
    old = _time.time() - backend.FINISHED_JOB_TTL_SECONDS - 1
    for job_id, state in [("done", tb.JOB_STATE_SUCCEEDED), ("busy", tb.JOB_STATE_RUNNING)]:
        backend._jobs[job_id] = tb.TranslationJob(job_id=job_id, state=state, updated_at=old)
        backend._run_states[job_id] = tb.TranslationRunState()
    backend._result_handle_to_job_id["h"] = "done"
    backend._job_results["h"] = {"output_stream": object()}

    with backend._jobs_lock:
        backend._prune_finished_jobs_locked(_time.time())

    assert "done" not in backend._jobs and "done" not in backend._run_states
    assert "h" not in backend._job_results and "h" not in backend._result_handle_to_job_id
    assert "busy" in backend._jobs and "busy" in backend._run_states


def test_cache_reads_survive_concurrent_eviction():
    """Hammer a tiny cache from several threads: a read must never raise."""
    import threading
    import TranslationBackend as tb

    import sys

    cache = tb.BoundedTranslationCache(max_entries=4)
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # switch threads between nearly every bytecode
    errors = []

    def churn(offset):
        try:
            for i in range(20000):
                key = (offset + i) % 9
                cache[key] = i
                cache.get((key + 1) % 9)
        except Exception as exc:  # pragma: no cover - the failure being guarded
            errors.append(exc)

    threads = [threading.Thread(target=churn, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    sys.setswitchinterval(previous)

    assert errors == []
    assert len(cache) <= 4
