"""The /voice engine line must describe the request that ACTUALLY just ran.

Why this file executes JavaScript instead of grepping for it: the defect these
tests exist for passed a source-string check. `assert "updateEngines(" in
source` is true in a build where updateEngines is called from exactly one place
(audio success), cleared from nowhere, and a recording's
"spoken by local:piper — this machine" therefore sits on screen above a later
transcript translation that produced no audio at all. A false privacy claim is
a behaviour, so it is tested as one: the real production JS strings are loaded
into Node against a stub DOM, the real handlers are driven, and the assertions
are about what the label ends up saying.

`test_the_harness_would_catch_the_original_defect` is the guard that keeps this
file honest — it removes the fix from the JS and proves the stale claim comes
back.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from passage.ui.voice_page import (  # noqa: E402
    ENGINE_LINE_AUDIO_FAILED,
    ENGINE_LINE_AUDIO_UNREPORTED,
    ENGINE_LINE_PENDING,
    ENGINE_LINE_TEXT_FAILED,
    ENGINE_LINE_TEXT_UNREPORTED,
    VOICE_PAGE_JS,
    VOICE_UX_JS,
    format_text_engine_line,
)

NODE = shutil.which("node")
requires_node = pytest.mark.skipif(NODE is None, reason="node is not installed")

LOCAL_AUDIO_SUMMARY = "heard by local:base · spoken by local:piper — this machine"
HOSTED_AUDIO_SUMMARY = "heard by hosted · spoken by hosted — sent out"


def _script_body(block: str) -> str:
    """The JS the browser actually gets, minus the <script> wrapper."""
    return block.split("<script>", 1)[1].rsplit("</script>", 1)[0]


HARNESS = r"""
const fs = require('fs');
const vm = require('vm');
const spec = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));

// ── stub DOM ────────────────────────────────────────────────────────────
const nodes = {};
function node(id) {
    if (!nodes[id]) {
        nodes[id] = {
            id, textContent: '', value: '', disabled: false, src: '',
            style: {}, classList: { remove() {}, add() {} },
            play() {}, closest() { return null; },
        };
    }
    return nodes[id];
}
const documentStub = {
    getElementById: (id) => (spec.missing || []).includes(id) ? null : node(id),
    querySelectorAll: () => [],
    addEventListener: () => {},
};

// ── stub Web Audio / mic ────────────────────────────────────────────────
class FakeAudioContext {
    constructor() { this.state = 'running'; this.sampleRate = 24000; this.destination = {}; }
    async resume() { this.state = 'running'; }
    close() {}
    createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
    createScriptProcessor() { return { connect() {}, disconnect() {}, onaudioprocess: null }; }
    createGain() { return { gain: { value: 1 }, connect() {}, disconnect() {} }; }
}
class FakeBlob {
    constructor(parts, opts) { this.size = 44; this.type = (opts || {}).type || ''; }
}
class FakeFormData { constructor() { this.parts = []; } append(k, v) { this.parts.push([k, v]); } }

// ── programmable fetch ──────────────────────────────────────────────────
const enginesText = () => node('desktop_voice_engines').textContent;
const observed = [];   // engine line as it stood at the moment each request left
let queue = [];

function makeResponse(r) {
    const headers = r.headers || {};
    const resp = {
        ok: r.ok !== false,
        headers: { get: (k) => (k in headers ? headers[k] : null) },
        text: async () => r.text || 'server error',
        json: async () => r.json || {},
        blob: async () => ({ size: r.audio_size === undefined ? 100 : r.audio_size }),
    };
    if (r.sse) {
        const payload = r.sse.join('');
        const bytes = new TextEncoder().encode(payload);
        let sent = false;
        resp.body = { getReader: () => ({
            read: async () => sent ? { done: true } : (sent = true, { done: false, value: bytes }),
        }) };
    }
    return resp;
}

async function fakeFetch(url, opts) {
    observed.push({ url, engines_at_request: enginesText() });
    const r = queue.shift();
    if (!r) throw new Error('no queued response for ' + url);
    if (r.network_error) throw new Error(r.network_error);
    return makeResponse(r);
}

// ── build the sandbox ───────────────────────────────────────────────────
const sandbox = {
    console, TextDecoder, TextEncoder, URLSearchParams, JSON, Math,
    Int16Array, ArrayBuffer, DataView, Uint8Array, Promise, Error,
    setTimeout, clearTimeout,
    document: documentStub,
    Blob: FakeBlob,
    FormData: FakeFormData,
    navigator: { mediaDevices: { getUserMedia: async () => ({ getTracks: () => [{ stop() {} }] }) } },
    fetch: fakeFetch,
    URL: { createObjectURL: () => 'blob:fake' },
};
sandbox.window = sandbox;
sandbox.window.AudioContext = FakeAudioContext;
sandbox.window.isSecureContext = true;
sandbox.window.location = { hostname: 'localhost', search: '' };
sandbox.window.addEventListener = () => {};
sandbox.window.PASSAGE_TOKEN = 'tok';
vm.createContext(sandbox);
vm.runInContext(spec.ux_js, sandbox, { filename: 'voice_ux.js' });
vm.runInContext(spec.page_js, sandbox, { filename: 'voice_page.js' });

// ── drive the real handlers ─────────────────────────────────────────────
(async () => {
    const timeline = [];
    for (const step of spec.steps) {
        queue = [step.response];
        if (step.kind === 'audio') {
            await sandbox.window.startRecording();
            await sandbox.window.stopRecording();
        } else {
            node('desktop_voice_transcript').value = step.transcript || 'hola mundo';
            node('language_select').value = step.language || 'Spanish';
            await sandbox.window.translateTranscriptFallback();
        }
        timeline.push({
            kind: step.kind,
            engines: enginesText(),
            status: node('desktop_voice_status').textContent,
            translated: node('desktop_voice_translated_text')
                ? node('translated_text').textContent : node('translated_text').textContent,
        });
    }
    process.stdout.write(JSON.stringify({ timeline, observed }));
})().catch((e) => { process.stderr.write(String(e && e.stack || e)); process.exit(3); });
"""


def _run(steps, tmp_path, page_js=None):
    """Drive the real /voice handlers in Node and return the observed timeline."""
    spec = {
        "ux_js": _script_body(VOICE_UX_JS),
        "page_js": _script_body(page_js if page_js is not None else VOICE_PAGE_JS),
        "steps": steps,
    }
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps(spec), encoding="utf-8")
    harness = tmp_path / "harness.js"
    harness.write_text(HARNESS, encoding="utf-8")
    proc = subprocess.run(
        [NODE, str(harness), str(spec_file)],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert proc.returncode == 0, f"harness failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


def _audio_ok(summary=LOCAL_AUDIO_SUMMARY):
    from urllib.parse import quote
    return {"kind": "audio", "response": {"ok": True, "headers": {
        "X-Engine-Summary": quote(summary, safe=""),
        "X-Original-Text": "hola",
        "X-Translated-Text": "hello",
    }}}


def _audio_failed():
    return {"kind": "audio", "response": {"ok": False, "text": "backend exploded"}}


def _text_ok(engine=None, sse=False):
    headers = {"Content-Type": "text/event-stream" if sse else "application/json"}
    if engine:
        headers["X-Text-Engine"] = engine
    response = {"ok": True, "headers": headers}
    if sse:
        response["sse"] = [
            'event: start\ndata: {"original_text": "hola mundo"}\n\n',
            'event: complete\ndata: {"translated_text": "hello world"}\n\n',
        ]
    else:
        response["json"] = {"original_text": "hola mundo", "translated_text": "hello world"}
    return {"kind": "text", "response": response}


def _text_failed():
    return {"kind": "text", "response": {
        "ok": False, "headers": {"Content-Type": "application/json"},
        "json": {"error": "text engine unavailable"},
    }}


# ───────────────────────── the reported defect ──────────────────────────── #

@requires_node
def test_transcript_after_a_recording_drops_the_audio_engine_claim(tmp_path):
    """The exact reported flow: record (local), then translate a transcript.

    The transcript ran the TEXT endpoint. No microphone opened, nothing was
    synthesised. A line still saying "spoken by local:piper — this machine" is
    a false privacy claim about a request that never touched a voice model.
    """
    out = _run([_audio_ok(), _text_ok()], tmp_path)
    after_audio, after_text = out["timeline"]

    assert after_audio["engines"] == LOCAL_AUDIO_SUMMARY  # the local run is named
    assert after_text["engines"] == ENGINE_LINE_TEXT_UNREPORTED
    for stale in ("local:piper", "local:base", "spoken by", "heard by", "this machine"):
        assert stale not in after_text["engines"]


@requires_node
def test_transcript_names_the_text_engine_the_server_reports(tmp_path):
    """When the text endpoint says which engine served it, say that — and only
    that. Still no audio engine, because no audio ran."""
    out = _run([_text_ok(engine="local:qwen2.5", sse=True)], tmp_path)
    line = out["timeline"][0]["engines"]

    assert "local:qwen2.5" in line and "this machine" in line
    assert line == format_text_engine_line("local:qwen2.5")
    assert "spoken by" not in line and "heard by" not in line


@requires_node
def test_transcript_on_a_hosted_text_engine_does_not_say_this_machine(tmp_path):
    out = _run([_audio_ok(), _text_ok(engine="openai:gpt-4o-mini")], tmp_path)
    line = out["timeline"][1]["engines"]

    assert "openai:gpt-4o-mini" in line and "sent out" in line
    assert "this machine" not in line


@requires_node
def test_a_failed_recording_does_not_keep_the_previous_privacy_claim(tmp_path):
    """Record successfully on local, then fail. Nothing was translated or
    spoken by the failed attempt, so nothing may be claimed for it."""
    out = _run([_audio_ok(), _audio_failed()], tmp_path)
    after_ok, after_fail = out["timeline"]

    assert "this machine" in after_ok["engines"]
    assert after_fail["engines"] == ENGINE_LINE_AUDIO_FAILED
    assert "local:piper" not in after_fail["engines"]
    assert "this machine" not in after_fail["engines"]
    assert after_fail["status"].startswith("Error:")


@requires_node
def test_a_failed_transcript_translation_makes_no_engine_claim(tmp_path):
    out = _run([_audio_ok(), _text_failed()], tmp_path)
    line = out["timeline"][1]["engines"]

    assert line == ENGINE_LINE_TEXT_FAILED
    assert "local:piper" not in line and "this machine" not in line


@requires_node
def test_the_line_is_cleared_before_each_request_leaves(tmp_path):
    """Not just corrected afterwards — cleared at the START. While a request is
    in flight the page must not still be describing the previous one."""
    steps = [_audio_ok(), _text_ok(), _audio_ok(HOSTED_AUDIO_SUMMARY)]
    out = _run(steps, tmp_path)

    seen = [entry["engines_at_request"] for entry in out["observed"]]
    assert len(seen) == 3
    assert seen == [ENGINE_LINE_PENDING] * 3


@requires_node
def test_back_to_back_recordings_show_the_second_recording_engines(tmp_path):
    """Local then hosted: the label must flip to hosted, not keep the local
    claim from the first recording."""
    out = _run([_audio_ok(), _audio_ok(HOSTED_AUDIO_SUMMARY)], tmp_path)

    assert out["timeline"][0]["engines"] == LOCAL_AUDIO_SUMMARY
    assert out["timeline"][1]["engines"] == HOSTED_AUDIO_SUMMARY
    assert "local:" not in out["timeline"][1]["engines"]
    assert "this machine" not in out["timeline"][1]["engines"]


@requires_node
def test_audio_success_without_a_summary_header_admits_it(tmp_path):
    """Silence from the backend must not be rendered as a blank line under a
    previous local claim, and must not be rendered as privacy either."""
    step = {"kind": "audio", "response": {"ok": True, "headers": {}}}
    out = _run([_audio_ok(), step], tmp_path)

    assert out["timeline"][1]["engines"] == ENGINE_LINE_AUDIO_UNREPORTED


# ───────────────── the guard: is this suite actually load-bearing? ───────── #

@requires_node
def test_the_harness_would_catch_the_original_defect(tmp_path):
    """Strip the transcript-path fix out of the real JS and the stale local
    claim must come back. Without this, every assertion above could be passing
    for reasons unrelated to the mechanism."""
    broken = VOICE_PAGE_JS \
        .replace("updateEngines(textEngineLine(textEngine));", "") \
        .replace("        beginEngineLine();\n        try {", "        try {")
    assert broken != VOICE_PAGE_JS

    out = _run([_audio_ok(), _text_ok()], tmp_path, page_js=broken)
    stale = out["timeline"][1]["engines"]

    # This is the defect, reproduced: a text-only request wearing the previous
    # recording's local-audio label.
    assert stale == LOCAL_AUDIO_SUMMARY
    assert "spoken by local:piper" in stale and "this machine" in stale


# ─────────────────── /voice prefetches the voice it needs ───────────────── #

def test_voice_page_language_selection_triggers_the_piper_prefetch(monkeypatch):
    """/voice is the one surface that actually needs a Piper voice, and its
    language box used to be a raw <input> wired to nothing — so the ~63MB voice
    for the chosen language was never fetched ahead of time."""
    from nicegui import ui  # noqa: F401
    from nicegui.client import Client
    from passage import local_voice
    from TranslationUI import TranslationUI

    asked: list[str | None] = []
    monkeypatch.setattr(local_voice, "prefetch_voice", lambda lang: asked.append(lang))

    app_ui = TranslationUI()
    with Client(lambda: None, request=None):
        app_ui.voice_translation_page()

    # Rendering the page fetches the voice for the language it opens on.
    assert asked == [app_ui.current_target_language]

    # And changing the language on /voice fetches that language's voice.
    app_ui.voice_language_input.value = "German"
    assert app_ui.current_target_language == "German"
    assert asked[-1] == "German"


def test_voice_page_goes_through_the_shared_prefetch_hook(monkeypatch):
    """/voice must route through the workspace's real
    `TranslationUI.request_voice_prefetch` rather than reaching past it into
    local_voice, so both surfaces share one policy — including the per-page
    dedupe. The hook is NOT stubbed into existence here: if the production
    method is renamed or removed again, this test fails instead of silently
    falling back to the direct call."""
    from nicegui.client import Client
    from passage import local_voice
    from TranslationUI import TranslationUI

    downloaded: list[str | None] = []
    monkeypatch.setattr(local_voice, "prefetch_voice", lambda lang: downloaded.append(lang))

    app_ui = TranslationUI()
    hook_calls: list[str | None] = []
    real_hook = app_ui.request_voice_prefetch

    def spy(language=None):
        hook_calls.append(language)
        return real_hook(language)

    monkeypatch.setattr(app_ui, "request_voice_prefetch", spy)

    with Client(lambda: None, request=None):
        app_ui.voice_translation_page()

    # The page's own render went through the shared hook, with the language
    # passed in explicitly (an argument-less hook call is DEFECT 1 itself).
    assert hook_calls == [app_ui.current_target_language]
    assert downloaded == [app_ui.current_target_language]

    # And the dedupe is genuinely SHARED: the workspace asking for the same
    # language afterwards does not start a second download.
    app_ui.request_voice_prefetch(app_ui.current_target_language)
    assert downloaded == [app_ui.current_target_language]


def test_a_failing_prefetch_cannot_take_the_voice_page_down(monkeypatch):
    """A 63MB download that fails must never reach the UI: the page still
    renders and translation falls back to hosted TTS, which the engine line
    then names."""
    from nicegui.client import Client
    from passage import local_voice
    from TranslationUI import TranslationUI

    def boom(_lang):
        raise RuntimeError("no network")

    monkeypatch.setattr(local_voice, "prefetch_voice", boom)
    app_ui = TranslationUI()
    with Client(lambda: None, request=None):
        app_ui.voice_translation_page()  # must not raise

    assert app_ui.voice_language_input is not None
