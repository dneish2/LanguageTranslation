"""Engine comparison: measurement honesty and failure isolation."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from passage import compare


def _candidates(*specs):
    return [{"label": l, "engine": l, "model": l, "is_local": local} for l, local in specs]


def test_a_failing_engine_does_not_void_the_comparison():
    """Seeing WHICH engine failed is itself a result."""
    def translate(candidate):
        if candidate["label"] == "broken":
            raise ConnectionError("connection refused")
        return "Hola mundo"

    results = compare.run_comparison(
        _candidates(("good", True), ("broken", True), ("other", True)), translate)

    assert len(results) == 3
    assert [r.ok for r in results] == [True, False, True]
    assert "refused" in results[1].error


def test_empty_output_is_reported_as_a_failure():
    """Two real local models (qwen3:30b, qwen3-vl:8b) returned nothing under a
    terse translation prompt. A blank row that looks successful would be worse
    than an honest failure."""
    results = compare.run_comparison(_candidates(("silent", True)), lambda c: "   ")

    assert results[0].ok is False
    assert results[0].error == "returned no text"


def test_local_engines_run_one_at_a_time():
    """They share a GPU. Running six at once measured 12s/23s/26s/33s for
    models that answer in a few hundred ms alone — a latency column that
    mostly measures VRAM contention would rank models by load order."""
    concurrent, peak = [0], [0]

    def translate(candidate):
        concurrent[0] += 1
        peak[0] = max(peak[0], concurrent[0])
        time.sleep(0.05)
        concurrent[0] -= 1
        return "ok"

    compare.run_comparison(_candidates(("a", True), ("b", True), ("c", True)), translate)

    assert peak[0] == 1, f"{peak[0]} local engines ran at once"


def test_remote_engines_still_run_concurrently():
    """Someone else's machines don't contend, so serialising them would only
    make the user wait longer."""
    concurrent, peak = [0], [0]

    def translate(candidate):
        concurrent[0] += 1
        peak[0] = max(peak[0], concurrent[0])
        time.sleep(0.15)
        concurrent[0] -= 1
        return "ok"

    compare.run_comparison(_candidates(("a", False), ("b", False), ("c", False)), translate)

    assert peak[0] > 1, "remote engines were serialised"


def test_results_keep_candidate_order():
    results = compare.run_comparison(
        _candidates(("first", False), ("second", True), ("third", False)), lambda c: "ok")

    assert [r.label for r in results] == ["first", "second", "third"]


def test_agreement_is_measured_against_the_others_not_the_first():
    """Scoring everything against whichever engine was listed first would make
    that engine's quirks the definition of correct."""
    outputs = {"a": "the cat sat on the mat", "b": "the cat sat on the mat",
               "c": "completely different words entirely"}
    results = compare.run_comparison(
        _candidates(("a", True), ("b", True), ("c", True)), lambda cd: outputs[cd["label"]])

    by_label = {r.label: r for r in results}
    assert by_label["a"].agreement > by_label["c"].agreement
    assert by_label["c"].agreement < 0.5


def test_agreement_is_none_when_there_is_nothing_to_compare_against():
    results = compare.run_comparison(_candidates(("only", True)), lambda c: "solo")
    assert results[0].agreement is None


def test_agreement_ignores_case_and_punctuation_but_not_accents():
    """Casing and punctuation spacing shouldn't read as disagreement. Accents
    deliberately DO: in Spanish "esta" and "está" are different words, so a
    model that drops them is genuinely worse and the score should say so."""
    same_but_for_punctuation = {"a": "¿Dónde está la farmacia?", "b": "Dónde está la farmacia"}
    results = compare.run_comparison(
        _candidates(("a", True), ("b", True)),
        lambda cd: same_but_for_punctuation[cd["label"]])
    assert results[0].agreement == 1.0

    accents_dropped = {"a": "¿Dónde está la farmacia?", "b": "donde esta la farmacia"}
    results = compare.run_comparison(
        _candidates(("a", True), ("b", True)), lambda cd: accents_dropped[cd["label"]])
    assert results[0].agreement < 1.0


def test_cost_is_blank_rather_than_invented_when_unpriced(monkeypatch):
    """Per-model prices go stale and differ per account. A plausible-looking
    wrong number in a comparison table is worse than an honest blank."""
    monkeypatch.delenv("PASSAGE_RATE_GPT_5_4_NANO", raising=False)

    assert compare.estimate_cost("gpt-5.4-nano", 100, is_local=False) is None
    assert compare.estimate_cost("qwen2.5:7b", 100, is_local=True) == 0.0

    monkeypatch.setenv("PASSAGE_RATE_GPT_5_4_NANO", "0.40")
    assert compare.estimate_cost("gpt-5.4-nano", 1_000_000, is_local=False) == 0.40


def test_comparison_is_capped():
    results = compare.run_comparison(
        _candidates(*[(f"m{i}", True) for i in range(20)]), lambda c: "ok")

    assert len(results) == compare.MAX_CANDIDATES


def test_empty_output_from_a_thinking_model_is_explained():
    """qwen3:30b and qwen3-vl:8b both showed as silent failures. Probed
    directly, qwen3-vl reads a menu photo perfectly — its answer just lands in
    a reasoning channel this endpoint doesn't expose. "returned no text" hides
    a diagnosable cause."""
    from types import SimpleNamespace

    thinking = SimpleNamespace(content="", reasoning="Got it, let's list every line…")
    results = compare.run_comparison(_candidates(("qwen3", True)), lambda c: thinking)

    assert results[0].ok is False
    assert "thinking model" in results[0].error


def test_reasoning_text_is_never_used_as_the_translation():
    """It is the model's working, not an answer. Passing it off as output is
    exactly the fluent-but-wrong failure this codebase keeps designing against."""
    from types import SimpleNamespace

    thinking = SimpleNamespace(content="", reasoning="First, the title is 'LA TABERNA DEL MAR'…")
    results = compare.run_comparison(_candidates(("qwen3", True)), lambda c: thinking)

    assert results[0].text == ""
    assert "TABERNA" not in (results[0].text or "")


def test_policy_keeps_session_history_separate_from_durable_storage():
    """The decision was "documents go to the cloud, camera and live typing stay
    local". Reading that as "live typing persists nothing" would have silently
    deleted the Recent Threads drawer, which is session-scoped and thrown away
    when the session ends — a different thing entirely."""
    from passage import policy

    live = policy.retention_for(policy.Surface.LIVE_TEXT)
    assert live.session_history is True, "would have removed Recent Threads"
    assert live.persists_durably is False

    document = policy.retention_for(policy.Surface.DOCUMENT)
    assert document.durable_segments is True and document.durable_traces is True
    assert document.durable_original_file is False, "storing uploads adds liability, not product"


def test_no_surface_may_durably_store_the_original_file():
    from passage import policy

    for surface in policy.Surface:
        assert policy.retention_for(surface).durable_original_file is False


def test_camera_and_voice_never_persist_durably():
    from passage import policy

    for surface in (policy.Surface.IMAGE, policy.Surface.VOICE):
        assert policy.retention_for(surface).persists_durably is False


def test_metering_is_one_question_with_one_answer():
    """Did Passage pay for the inference? Billing must read this rather than
    inspect provider details, so it can't drift away from routing."""
    from passage import policy, provider_profiles as pp

    assert policy.is_metered(None) is True                       # hosted default
    assert policy.is_metered(pp.local_profile("http://localhost:11434/v1", "x")) is False
    assert policy.is_metered(pp.ProviderProfile(label="mine", kind=pp.KIND_BYO)) is False


def test_privacy_line_distinguishes_local_from_byo_from_hosted():
    from passage import policy, provider_profiles as pp

    local = policy.describe_privacy(pp.local_profile("http://localhost:11434/v1", "x"))
    byo = policy.describe_privacy(pp.ProviderProfile(label="m", kind=pp.KIND_BYO))
    hosted = policy.describe_privacy(None)

    assert "your machine" in local and policy.data_leaves_machine(
        pp.local_profile("http://localhost:11434/v1", "x")) is False
    assert "not metered" in byo and policy.data_leaves_machine(
        pp.ProviderProfile(label="m", kind=pp.KIND_BYO)) is True
    assert "metered" in hosted
