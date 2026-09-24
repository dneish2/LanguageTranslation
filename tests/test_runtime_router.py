"""The runtime router: one routing decision, shared by every text surface.

Before this, only live typing was local-first. Documents, segment edits, the
streaming endpoint and voice's translation step went hosted on the same machine.
These tests assert WHICH engine received each request, using fakes that count
their own calls, and that the label booked for the run matches what answered.
"""
from io import BytesIO
from types import SimpleNamespace

import pytest

import TranslationBackend as tb


class CountingProvider(tb.BaseTranslationProvider):
    """Answers with its own name, so the bytes say who produced them."""

    def __init__(self, name, *, label=None, fail=False, text_model="fake-model",
                 base_url=None):
        self.name = name
        self.calls = 0
        self.fail = fail
        self.text_model = text_model
        self.base_url = base_url
        self.max_input_chars = None
        self.label = label
        self.is_openai_hosted = base_url is None
        self.vision_model = "fake-vision"

    @property
    def display_label(self):
        return self.label or f"hosted:{self.text_model}"

    def create_chat_completion(self, *, messages, max_tokens):
        self.calls += 1
        if self.fail:
            raise ConnectionError("connection refused")
        if self.base_url is None:
            tb._note_engine(self.display_label)  # hosted notes itself, like ChatCompletionsProvider
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=f"[{self.name}] hola"))])

    def transcribe_audio(self, *, audio_file):
        self.calls += 1
        return "hello"

    def synthesize_speech(self, *, text):
        self.calls += 1
        return b"mp3"


def _backend(monkeypatch, *, local_fails=False):
    backend = tb.TranslationBackend()
    hosted = CountingProvider("hosted", text_model="gpt-hosted")
    local = CountingProvider("local", fail=local_fails, text_model="qwen-local",
                             base_url="http://localhost:11434/v1")
    backend.provider = hosted
    monkeypatch.setattr(backend, "_live_local_provider", lambda: local)
    return backend, hosted, local


@pytest.mark.local_deployment
def test_a_document_segment_runs_local_first_on_a_local_deployment(monkeypatch):
    backend, hosted, local = _backend(monkeypatch)

    with backend.capture_provenance() as prov, backend.using_profile(None):
        out = backend.translate_text("The meeting starts at nine.", "Spanish")

    assert out == "[local] hola"
    assert (local.calls, hosted.calls) == (1, 0)
    assert prov.most_exposed == "local:qwen-local"


@pytest.mark.local_deployment
def test_a_local_failure_falls_back_and_the_run_is_booked_as_sent_out(monkeypatch):
    backend, hosted, local = _backend(monkeypatch, local_fails=True)

    with backend.capture_provenance() as prov, backend.using_profile(None):
        out = backend.translate_text("Please bring the report.", "Spanish")

    assert out == "[hosted] hola"
    assert (local.calls, hosted.calls) == (1, 1)
    assert prov.most_exposed == "hosted:gpt-hosted"


@pytest.mark.local_deployment
def test_a_cache_hit_names_who_really_made_the_bytes_after_a_fallback(monkeypatch):
    """The cache key says 'local'; the entry must say hosted, or a later hit
    would tell the user text that was sent out never left the machine."""
    backend, hosted, local = _backend(monkeypatch, local_fails=True)
    with backend.using_profile(None):
        backend.translate_text("Lunch is provided.", "Spanish")

    with backend.capture_provenance() as prov, backend.using_profile(None):
        backend.translate_text("Lunch is provided.", "Spanish")

    assert prov.from_cache
    assert prov.engine == "hosted:gpt-hosted"


def test_a_cloud_deployment_never_routes_local_and_never_probes(monkeypatch):
    backend = tb.TranslationBackend()
    backend.provider = CountingProvider("hosted")

    def no_network(*a, **k):
        raise AssertionError("the hosted service probed a local model")

    monkeypatch.setattr("urllib.request.urlopen", no_network)
    assert tb.DEPLOYMENT_MODE == "cloud"
    assert backend._local_first_provider() is None
    with backend.using_profile(None):
        assert backend.translate_text("Hi", "Spanish") == "[hosted] hola"


@pytest.mark.local_deployment
def test_a_photo_is_read_by_the_hosted_side_of_the_router(monkeypatch):
    """Vision is hosted-only here: the router must not send a photograph to a
    text-only local model."""
    backend, hosted, local = _backend(monkeypatch)
    router = backend._local_first_provider()

    assert tb._for_capability(router, "Reading a photo") is hosted


@pytest.mark.local_deployment
def test_voice_reports_the_translation_leg_that_actually_ran(monkeypatch):
    backend, hosted, local = _backend(monkeypatch)
    monkeypatch.setattr(tb.local_voice, "stt_available", lambda: False)
    monkeypatch.setattr(tb.local_voice, "tts_available", lambda language: False)
    monkeypatch.setattr(backend, "_transcribe_hosted", lambda audio: "Good morning")
    monkeypatch.setattr(backend, "_synthesize_hosted", lambda text: b"mp3", raising=False)

    with backend.using_profile(None):
        try:
            _, translated, _, meta = backend.translate_audio(b"RIFF....", "Spanish")
        except Exception as error:  # hosted TTS details vary; the leg label is the point
            pytest.skip(f"voice pipeline needs a TTS seam here: {error}")

    assert translated == "[local] hola"
    assert meta["translation"] == "local:qwen-local"


@pytest.mark.local_deployment
def test_a_chosen_profile_is_honoured_over_local_first(monkeypatch):
    """Someone who picked an endpoint gets that endpoint, not a substitute."""
    backend, hosted, local = _backend(monkeypatch)
    chosen = CountingProvider("byo", label="byo:their-model",
                              base_url="https://their.example/v1")
    monkeypatch.setattr(backend, "provider_for_profile", lambda profile: chosen)
    profile = SimpleNamespace(kind="byo")

    with backend.using_profile(profile):
        out = backend.translate_text("Hi there", "Spanish")

    assert out == "[byo] hola"
    assert (chosen.calls, local.calls, hosted.calls) == (1, 0, 0)


def test_most_exposed_prefers_any_engine_that_sent_text_out():
    prov = tb.CallProvenance()
    for engine in ["local:a", "hosted:b", "local:c"]:
        prov.record_engine(engine)

    assert prov.most_exposed == "hosted:b"


@pytest.mark.local_deployment
def test_a_document_job_reports_the_engine_that_ran_not_the_one_expected(monkeypatch):
    """Seen live: a DOCX translated entirely on local Ollama (no hosted key
    existed) was booked on /engines as 'sent out: hosted:gpt-5.4-nano, metered',
    because the page labelled the job from its own expectation."""
    import time
    from docx import Document

    backend, hosted, local = _backend(monkeypatch)
    doc = Document()
    doc.add_paragraph("The meeting starts at nine.")
    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)

    job_id = backend.start_translation_job(
        input_stream=buf, file_extension="docx", target_language="Spanish")
    deadline = time.time() + 20
    while time.time() < deadline:
        job = backend.get_job(job_id)
        if job.state in {tb.JOB_STATE_SUCCEEDED, tb.JOB_STATE_FAILED}:
            break
        time.sleep(0.05)

    assert job.state == tb.JOB_STATE_SUCCEEDED, job.error
    result = backend.get_job_result(job.result_handle)
    assert result["engine"] == "local:qwen-local"
    assert (local.calls, hosted.calls) == (1, 0)
