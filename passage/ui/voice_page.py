"""The /voice page: recorder UI, its head-injected JS, and the
/api/voice_translate route. Mixed into TranslationUI so it shares `self`
(backend, api_guard, _check_api_access, _inject_theme, _go_workspace, ...)
without redesigning the app's state model.
"""
import asyncio
import logging
import time
import uuid
from urllib.parse import quote

from nicegui import ui
from fastapi import Request, UploadFile, File, Form
from starlette.responses import Response

import theme
from api_security import MAX_UPLOAD_BYTES
from passage import local_voice, policy
from passage.ui.common import LANGUAGES, log_event


def voice_engine_label(meta: dict, translation_label: str) -> str:
    """The single engine label this recording is BOOKED against.

    A voice request is three steps that each choose their own engine, but the
    ledger row is one engine. Picking the wrong one of the three is how a
    receipt over-claims privacy, so the rule is deliberately asymmetric:

      * if ANY step ran on Passage's hosted key, the row says hosted — even
        when the words themselves went to the user's own endpoint. Audio did
        leave for Passage, and Passage paid for it; saying "your endpoint,
        not metered" over that would be the over-claim.
      * only when every step stayed on this machine is the row local.
      * otherwise the row names the endpoint the WORDS went to, which is the
        step a user cares about most.

    `translation_label` is derived from the profile that was actually applied
    (see TranslationUI._text_engine_label), not from meta: the backend fills
    meta["translation"] with a hosted literal because it has never been told
    about profiles.
    """
    steps = [meta.get("stt") or "hosted", translation_label, meta.get("tts") or "hosted"]
    hosted = [s for s in steps if _is_passage_hosted(s)]
    return hosted[0] if hosted else translation_label


def _is_passage_hosted(step: str) -> bool:
    """A step that ran on Passage's own key. An absent or unrecognised label
    counts as hosted: the honest default is the one that cannot over-claim."""
    return step == "hosted" or step.startswith(("hosted:", "app:"))


def _recording_stayed_local(meta: dict) -> bool:
    """True only when the AUDIO never left this machine — both the transcriber
    and the synthesiser ran locally."""
    return all((meta.get(step) or "hosted").startswith("local") for step in ("stt", "tts"))


def format_engine_line(meta: dict) -> str:
    """One line naming the engines that ACTUALLY served this request.

    Local voice now defaults on when the models are present (DECISIONS.md §2),
    and an automatic default is only honest if the page says where the audio
    went — the same argument that justified local-first for text. Built from
    the meta the backend returns, so it cannot drift from what ran: if local
    was tried and failed, this says "hosted", because hosted is what ran.
    """
    stt = meta.get("stt") or "hosted"
    tts = meta.get("tts") or "hosted"
    # THREE steps run, not two. The middle one carries the actual words, so a
    # sentence built from stt+tts alone claimed "this machine" over a
    # transcript that had just been sent to a hosted model. An absent label
    # counts as hosted: the honest default is the one that cannot over-claim.
    translation = meta.get("translation") or "hosted"
    steps = (stt, translation, tts)
    local_steps = [s for s in steps if s.startswith("local")]
    where = ("this machine" if len(local_steps) == len(steps)
             else "sent out" if not local_steps
             else "partly on this machine")
    line = f"heard by {stt} · translated by {translation} · spoken by {tts} — {where}"
    fell_back = [k for k in ("stt_fallback", "tts_fallback") if meta.get(k)]
    if fell_back:
        line += " (local was tried and failed)"
    return line


#: What the engine line says while a request is in flight. The line names the
#: engines that served ONE request, so it must never survive into the next one:
#: a stale "spoken by local:piper — this machine" sitting above a transcript
#: translation that produced no audio at all is a false privacy claim, not a
#: cosmetic bug. Every entry point clears to this first.
ENGINE_LINE_PENDING = "engines: waiting for this request to report what ran"

#: Audio finished but the backend sent no summary header.
ENGINE_LINE_AUDIO_UNREPORTED = (
    "engines: this recording finished but the server did not name what ran"
)

#: Audio failed. Nothing was translated, so nothing may be claimed.
ENGINE_LINE_AUDIO_FAILED = (
    "engines: unknown — this recording failed, nothing was translated or spoken"
)

#: The transcript path calls the TEXT endpoint. No microphone, no speech
#: synthesis: naming any audio engine here would be a lie by copy-paste.
ENGINE_LINE_TEXT_UNREPORTED = (
    "text only — no audio was heard or spoken · "
    "the server did not name the translation engine"
)
ENGINE_LINE_TEXT_FAILED = (
    "engines: unknown — this transcript translation failed · "
    "no audio was heard or spoken"
)


def format_text_engine_line(engine: str) -> str:
    """The engine line for the transcript path, which runs TEXT only."""
    if not engine:
        return ENGINE_LINE_TEXT_UNREPORTED
    where = "this machine" if engine.startswith("local") else "sent out"
    return f"translated by {engine} — {where} · text only, no audio was heard or spoken"


VOICE_UX_JS = """
<script>
window.voiceUx = window.voiceUx || (() => {
    const states = {
        READY: 'Ready: record audio or paste transcript',
        REQUESTING_AUDIO: 'Requesting microphone access…',
        RECORDING: 'Recording audio…',
        STOPPING: 'Stopping recording…',
        PROCESSING_AUDIO: 'Processing audio…',
        TRANSLATING_TEXT: 'Translating transcript…',
        COMPLETE: 'Complete: output ready',
    };

    const resolve = (scope, key) => document.getElementById(`${scope}_${key}`);

    function setStatus(scope, message) {
        const node = resolve(scope, 'status');
        if (node) node.textContent = message;
    }

    function setDebug(scope, message) {
        const node = resolve(scope, 'debug');
        if (node) node.textContent = message || '';
    }

    function setEngines(scope, message) {
        const node = resolve(scope, 'engines');
        if (node) node.textContent = message || '';
    }

    function setRecordingButtons(scope, recording) {
        const start = resolve(scope, 'start_recording');
        const stop = resolve(scope, 'stop_recording');
        if (start) {
            start.disabled = recording;
            start.style.opacity = recording ? '0.5' : '1';
        }
        if (stop) {
            stop.disabled = !recording;
            stop.style.opacity = recording ? '1' : '0.5';
        }
    }

    function init(scope) {
        setStatus(scope, states.READY);
        setDebug(scope, '');
        setEngines(scope, '');
        setRecordingButtons(scope, false);
    }

    // Developer readout: reveal the hidden Debug block only with ?debug=1.
    window.addEventListener('load', () => {
        if (new URLSearchParams(window.location.search).has('debug')) {
            document.querySelectorAll('.p-debug-block').forEach(el => el.classList.remove('hidden'));
        }
    });

    return { states, init, setStatus, setDebug, setEngines, setRecordingButtons };
})();
</script>
"""



_VOICE_PAGE_JS_TEMPLATE = r"""
<script>
    // PCM16 WAV capture via Web Audio (24 kHz): feeds the realtime
    // transcription models directly — no webm container, no transcoding.
    let audioCtx = null, sourceNode = null, procNode = null, silentGain = null;
    let stream = null, pcmChunks = [], isRecording = false;
    const TARGET_SAMPLE_RATE = 24000;
    const DESKTOP_SCOPE = 'desktop_voice';

    function updateStatus(msg) {
        window.voiceUx.setStatus(DESKTOP_SCOPE, msg);
    }
    function updateDebug(msg) {
        window.voiceUx.setDebug(DESKTOP_SCOPE, msg);
    }
    function updateEngines(msg) {
        window.voiceUx.setEngines(DESKTOP_SCOPE, msg);
    }
    // ── engine line lifecycle ────────────────────────────────────────────
    // The line names the engines that served ONE request. It used to be
    // written in exactly one place (audio success) and cleared nowhere, so a
    // recording's "spoken by local:piper — this machine" stayed on screen
    // above a later transcript translation that produced no audio at all, and
    // above failed recordings. Every entry point below now clears it FIRST and
    // writes an honest line on both success and failure.
    const ENGINE_PENDING = 'ENGINE_LINE_PENDING';
    const ENGINE_AUDIO_UNREPORTED = 'ENGINE_LINE_AUDIO_UNREPORTED';
    const ENGINE_AUDIO_FAILED = 'ENGINE_LINE_AUDIO_FAILED';
    const ENGINE_TEXT_UNREPORTED = 'ENGINE_LINE_TEXT_UNREPORTED';
    const ENGINE_TEXT_FAILED = 'ENGINE_LINE_TEXT_FAILED';

    function beginEngineLine() {
        updateEngines(ENGINE_PENDING);
    }
    // The transcript path runs the TEXT endpoint: no microphone was opened and
    // nothing was synthesised, so it must never name an STT or TTS engine.
    function textEngineLine(engine) {
        if (!engine) return ENGINE_TEXT_UNREPORTED;
        const where = engine.indexOf('local') === 0 ? 'this machine' : 'sent out';
        return `translated by ${engine} — ${where} · text only, no audio was heard or spoken`;
    }
    // The player holds a blob URL from the PREVIOUS recording. Leaving it in
    // place under a transcript-only translation (which synthesises nothing) or
    // under a failed recording offers the user audio of a different sentence
    // and reads as output of the request they just made.
    function clearAudioPlayer() {
        const player = document.getElementById('out_audio');
        if (!player) return;
        if (typeof player.pause === 'function') { try { player.pause(); } catch (_e) {} }
        const previous = player.src;
        player.src = 'data:audio/wav;base64,';
        if (typeof player.load === 'function') { try { player.load(); } catch (_e) {} }
        if (previous && previous.indexOf('blob:') === 0 &&
            typeof URL !== 'undefined' && URL.revokeObjectURL) {
            try { URL.revokeObjectURL(previous); } catch (_e) {}
        }
    }
    // A failed request translated nothing, so the previous request's panes
    // must not stay on screen as if they were its result.
    function clearResultPanes() {
        const orig = document.getElementById('original_text');
        const trans = document.getElementById('translated_text');
        if (orig) orig.textContent = '';
        if (trans) trans.textContent = '';
        clearAudioPlayer();
    }
    function updateButtons(recording) {
        window.voiceUx.setRecordingButtons(DESKTOP_SCOPE, recording);
        isRecording = recording;
    }
    function setRecordingControlsEnabled(enabled) {
        const start = document.getElementById('desktop_voice_start_recording');
        const stop = document.getElementById('desktop_voice_stop_recording');
        if (start) {
            start.disabled = !enabled;
            start.style.opacity = enabled ? '1' : '0.5';
        }
        if (stop) {
            stop.disabled = true;
            stop.style.opacity = '0.5';
        }
        if (!enabled) isRecording = false;
    }

    function encodeWavBlob(chunks, sampleRate) {
        let total = 0;
        chunks.forEach(c => { total += c.length; });
        const pcm = new Int16Array(total);
        let offset = 0;
        for (const c of chunks) { pcm.set(c, offset); offset += c.length; }
        const buf = new ArrayBuffer(44 + pcm.length * 2);
        const view = new DataView(buf);
        const writeStr = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
        writeStr(0, 'RIFF'); view.setUint32(4, 36 + pcm.length * 2, true); writeStr(8, 'WAVE');
        writeStr(12, 'fmt '); view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);   // PCM
        view.setUint16(22, 1, true);   // mono
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true); view.setUint16(34, 16, true);
        writeStr(36, 'data'); view.setUint32(40, pcm.length * 2, true);
        new Int16Array(buf, 44).set(pcm);
        return new Blob([buf], { type: 'audio/wav' });
    }

    function teardownAudioGraph() {
        if (procNode) { try { procNode.disconnect(); } catch (_e) {} procNode = null; }
        if (sourceNode) { try { sourceNode.disconnect(); } catch (_e) {} sourceNode = null; }
        if (silentGain) { try { silentGain.disconnect(); } catch (_e) {} silentGain = null; }
        if (audioCtx) { try { audioCtx.close(); } catch (_e) {} audioCtx = null; }
        if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    }

    function mapRecordingError(err) {
        const name = err?.name || 'Error';
        if (name === 'NotAllowedError') {
            return 'Microphone permission denied. Allow mic access in browser/site settings and retry.';
        }
        if (name === 'NotFoundError') {
            return 'No microphone found. Connect/enable a mic and try again.';
        }
        if (name === 'NotSupportedError') {
            return 'Audio recording is not supported in this browser. Try a current Chrome/Edge/Safari.';
        }
        return err?.message || 'Unexpected recording error.';
    }

    async function startRecording() {
        const hasGetUserMedia = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
        const isLocalhost = ['localhost', '127.0.0.1', '::1'].includes(window.location.hostname);
        const secureOk = window.isSecureContext || isLocalhost;
        const AudioContextImpl = window.AudioContext || window.webkitAudioContext;
        if (!hasGetUserMedia || !secureOk || !AudioContextImpl) {
            setRecordingControlsEnabled(false);
            updateStatus("Recording unavailable. Use HTTPS or localhost in a supported browser.");
            updateDebug(`preflight getUserMedia=${hasGetUserMedia} secure=${secureOk} webAudio=${!!AudioContextImpl}`);
            return;
        }

        updateStatus("Requesting mic…");
        updateDebug(`pcm-wav secure=${secureOk} permission=requesting`);
        try {
            const constraints = {
                audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true}
            };
            stream = await navigator.mediaDevices.getUserMedia(constraints);
            try {
                audioCtx = new AudioContextImpl({ sampleRate: TARGET_SAMPLE_RATE });
            } catch (_e) {
                audioCtx = new AudioContextImpl();
            }
            if (audioCtx.state === 'suspended') await audioCtx.resume();
            sourceNode = audioCtx.createMediaStreamSource(stream);
            procNode = audioCtx.createScriptProcessor(4096, 1, 1);
            silentGain = audioCtx.createGain();
            silentGain.gain.value = 0;  // keep the graph alive without echoing the mic
            pcmChunks = [];
            procNode.onaudioprocess = (e) => {
                if (!isRecording) return;
                const f32 = e.inputBuffer.getChannelData(0);
                const i16 = new Int16Array(f32.length);
                for (let i = 0; i < f32.length; i++) {
                    const s = Math.max(-1, Math.min(1, f32[i]));
                    i16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }
                pcmChunks.push(i16);
                const seconds = pcmChunks.length * 4096 / audioCtx.sampleRate;
                updateDebug(`pcm-wav ${seconds.toFixed(1)}s @ ${audioCtx.sampleRate}Hz`);
            };
            sourceNode.connect(procNode);
            procNode.connect(silentGain);
            silentGain.connect(audioCtx.destination);
            updateStatus("🔴 Recording…");
            updateButtons(true);
        } catch (err) {
            const mapped = mapRecordingError(err);
            updateStatus("Error: " + mapped);
            updateDebug(`pcm-wav secure=${secureOk} permission=denied (${err?.name || 'unknown'})`);
            teardownAudioGraph();
        }
    }

    async function stopRecording() {
        if (!isRecording || !audioCtx) {
            window.voiceUx.setDebug(DESKTOP_SCOPE, 'No active recording session.');
            return;
        }
        window.voiceUx.setStatus(DESKTOP_SCOPE, window.voiceUx.states.STOPPING);
        // Clear before the request, not after: whatever is on screen describes
        // the PREVIOUS request and is already wrong.
        beginEngineLine();
        // Same argument as the engine line: the panes and the player describe
        // the PREVIOUS request and are already wrong.
        clearResultPanes();
        updateButtons(false);
        const sampleRate = audioCtx.sampleRate;
        const captured = pcmChunks;
        pcmChunks = [];
        teardownAudioGraph();
        window.voiceUx.setStatus(DESKTOP_SCOPE, window.voiceUx.states.PROCESSING_AUDIO);
        const blob = encodeWavBlob(captured, sampleRate);
        let lang = document.getElementById('language_select')?.value || 'es';
        let fd = new FormData();
        fd.append('file', blob, 'rec.wav');
        fd.append('language', lang);
        try {
            const resp = await fetch('/api/voice_translate', { method:'POST', body:fd, headers: { 'X-Passage-Token': window.PASSAGE_TOKEN || '' } });
            if(!resp.ok) throw new Error(await resp.text());
            const audio = await resp.blob();
            const origHeader = resp.headers.get('X-Original-Text') || '';
            const transHeader = resp.headers.get('X-Translated-Text') || '';
            const decodeHeader = (value) => {
                if (!value) return '';
                try { return decodeURIComponent(value); } catch (_err) { return value; }
            };
            // Name the engines that served THIS request, from the response
            // headers the backend derives from what actually ran.
            updateEngines(decodeHeader(resp.headers.get('X-Engine-Summary') || '')
                          || ENGINE_AUDIO_UNREPORTED);
            const orig = decodeHeader(origHeader);
            const trans = decodeHeader(transHeader);
            document.getElementById('original_text').textContent   = orig;
            document.getElementById('translated_text').textContent= trans;
            if(audio.size>0){
                let url = URL.createObjectURL(audio);
                let player = document.getElementById('out_audio');
                player.src = url; player.play();
                window.voiceUx.setStatus(DESKTOP_SCOPE, window.voiceUx.states.COMPLETE);
            } else {
                // Text came back but nothing was spoken: no player content.
                clearAudioPlayer();
                window.voiceUx.setStatus(DESKTOP_SCOPE, window.voiceUx.states.COMPLETE);
            }
        } catch(e) {
            window.voiceUx.setStatus(DESKTOP_SCOPE, "Error: " + e.message);
            window.voiceUx.setDebug(DESKTOP_SCOPE, e.message);
            // Nothing was translated, so nothing of the previous request may
            // stay on screen under this failure.
            clearResultPanes();
            // A failed recording must not leave the last successful
            // recording's "this machine" claim standing over an empty result.
            updateEngines(ENGINE_AUDIO_FAILED);
        }
    }

    async function translateTranscriptFallback() {
        const lang = document.getElementById('language_select')?.value || 'es';
        const transcript = document.getElementById('desktop_voice_transcript')?.value || '';
        const cleaned = transcript.trim();
        if (!cleaned) {
            window.voiceUx.setStatus(DESKTOP_SCOPE, 'Please provide transcript text before translating.');
            return;
        }
        window.voiceUx.setStatus(DESKTOP_SCOPE, window.voiceUx.states.TRANSLATING_TEXT);
        window.voiceUx.setDebug(DESKTOP_SCOPE, `chars=${cleaned.length}`);
        // Drop any audio engine line from a previous recording BEFORE this
        // text-only request starts.
        beginEngineLine();
        // This path synthesises NOTHING. A player still holding the last
        // recording's audio under a text-only result is the same false claim
        // the engine line was fixed for.
        clearAudioPlayer();
        try {
            const resp = await fetch('/api/text_translate_stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded', 'X-Passage-Token': window.PASSAGE_TOKEN || '' },
                body: new URLSearchParams({ text: cleaned, language: lang }),
            });
            const contentType = resp.headers.get('Content-Type') || '';
            // The text endpoint names its engine when it can. If it does not,
            // we say so — we do not borrow the audio path's label.
            let textEngine = resp.headers.get('X-Text-Engine') || '';
            if (contentType.includes('text/event-stream')) {
                const reader = resp.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                let streamed = false;
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    const events = buffer.split('\n\n');
                    buffer = events.pop() || '';
                    for (const evt of events) {
                        const eventLine = evt.split('\n').find(line => line.startsWith('event: '));
                        const dataLine = evt.split('\n').find(line => line.startsWith('data: '));
                        const eventType = eventLine ? eventLine.replace('event: ', '').trim() : '';
                        const payload = dataLine ? JSON.parse(dataLine.replace('data: ', '')) : {};
                        // The events carry the engine too, so a proxy that
                        // strips response headers cannot silence the label.
                        if (payload.engine) textEngine = textEngine || payload.engine;
                        if (eventType === 'start') {
                            document.getElementById('original_text').textContent = payload.original_text || cleaned;
                        } else if (eventType === 'partial') {
                            streamed = true;
                            document.getElementById('translated_text').textContent = payload.translated_text || '';
                        } else if (eventType === 'complete') {
                            document.getElementById('translated_text').textContent = payload.translated_text || '';
                        } else if (eventType === 'error') {
                            throw new Error(payload.error || 'Transcript streaming failed.');
                        }
                    }
                }
                if (!streamed) {
                    window.voiceUx.setDebug(DESKTOP_SCOPE, 'Streaming unavailable; translation returned without partial chunks.');
                }
            } else {
                const data = await resp.json();
                if (!resp.ok) throw new Error(data?.error || 'Transcript translation failed.');
                textEngine = textEngine || data.engine || '';
                document.getElementById('original_text').textContent = data.original_text || cleaned;
                document.getElementById('translated_text').textContent = data.translated_text || '';
            }
            updateEngines(textEngineLine(textEngine));
            window.voiceUx.setStatus(DESKTOP_SCOPE, window.voiceUx.states.COMPLETE);
        } catch (e) {
            window.voiceUx.setStatus(DESKTOP_SCOPE, "Error: " + e.message);
            window.voiceUx.setDebug(DESKTOP_SCOPE, e.message || 'unknown error');
            clearResultPanes();
            updateEngines(ENGINE_TEXT_FAILED);
        }
    }

    window.stopRecording = stopRecording;
    window.startRecording = startRecording;

    window.translateTranscriptFallback = translateTranscriptFallback;

    // Delegated bindings for the Record/Stop buttons. They are rendered by
    // ui.html(), and NiceGUI 3 pipes that through DOMPurify, which strips
    // inline handler attributes — so handlers must be attached from script
    // instead. Delegating on `document` also survives a re-render of the row.
    document.addEventListener('click', (e) => {
        const el = e.target && e.target.closest ? e.target.closest('button') : null;
        if (!el || el.disabled) return;
        if (el.id === 'desktop_voice_start_recording') { e.preventDefault(); startRecording(); }
        else if (el.id === 'desktop_voice_stop_recording') { e.preventDefault(); stopRecording(); }
    });

    window.addEventListener('load', () => {
        const hasGetUserMedia = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
        const isLocalhost = ['localhost', '127.0.0.1', '::1'].includes(window.location.hostname);
        const secureOk = window.isSecureContext || isLocalhost;
        const hasWebAudio = !!(window.AudioContext || window.webkitAudioContext);
        const canRecord = hasGetUserMedia && secureOk && hasWebAudio;
        setRecordingControlsEnabled(canRecord);
        if (!canRecord) {
            updateStatus("Recording unavailable. Use HTTPS or localhost in a supported browser.");
        } else {
            updateStatus("Ready to record");
        }
        updateDebug(`pcm-wav secure=${secureOk} getUserMedia=${hasGetUserMedia} webAudio=${hasWebAudio}`);
    });
</script>
"""

#: The placeholders in the template are filled from the Python constants
#: above, so what the page shows and what the tests assert cannot drift.
VOICE_PAGE_JS = (
    _VOICE_PAGE_JS_TEMPLATE
    .replace('ENGINE_LINE_PENDING', ENGINE_LINE_PENDING)
    .replace('ENGINE_LINE_AUDIO_UNREPORTED', ENGINE_LINE_AUDIO_UNREPORTED)
    .replace('ENGINE_LINE_AUDIO_FAILED', ENGINE_LINE_AUDIO_FAILED)
    .replace('ENGINE_LINE_TEXT_UNREPORTED', ENGINE_LINE_TEXT_UNREPORTED)
    .replace('ENGINE_LINE_TEXT_FAILED', ENGINE_LINE_TEXT_FAILED)
)


class VoicePageMixin:
    def _render_voice_status_block(self, scope: str) -> None:
        ui.label("Status").classes(f"{theme.DATA} mt-3")
        ui.label("")\
            .classes("text-base")\
            .props(f"id={scope}_status")
        # Which engine served THIS request. Always visible (not behind ?debug),
        # because local voice is now on by default when the models are present
        # and a silent default is only acceptable if the page names where the
        # recording actually went.
        ui.label("")\
            .classes(f"{theme.DATA} min-h-[20px]")\
            .props(f"id={scope}_engines")
        # Developer readout — hidden unless the page is opened with ?debug=1
        # (voiceUx.init reveals .p-debug-block); updateDebug still writes to it.
        ui.label("Debug").classes(f"{theme.DATA} mt-1 hidden p-debug-block")
        ui.label("")\
            .classes(f"{theme.DATA} min-h-[20px] hidden p-debug-block")\
            .props(f"id={scope}_debug")

    def _inject_voice_frontend_helpers(self) -> None:
        ui.add_head_html(VOICE_UX_JS)

    def _prefetch_voice_for(self, language: str | None) -> None:
        """Start the background Piper-voice download for `language`.

        Prefers the workspace's own hook (`TranslationUI.request_voice_prefetch`)
        so both surfaces share one policy — including its per-page dedupe — and
        so a language typed on /voice is not fetched twice when the user then
        walks back into the workspace. Falls back to the underlying
        `local_voice.prefetch_voice` when that hook is not present, because
        /voice must not be the surface that silently never prefetches. Never
        blocks and never raises: prefetch_voice spawns a daemon thread, no-ops
        when local voice is off / already installed / the language has no
        Piper voice, and a failed download must not reach the UI.

        The hook takes the language as an argument. It deliberately does NOT
        read `self.current_target_language`: the argument-less version of this
        call is the exact defect (DEFECT 1) that made every page load fetch the
        `__init__` default instead of the user's real target.
        """
        try:
            # bind_value fires on_change once at construction, so the initial
            # language would otherwise be requested twice per page load.
            if getattr(self, "_last_prefetched_voice_language", None) == language:
                return
            self._last_prefetched_voice_language = language
            hook = getattr(self, "request_voice_prefetch", None)
            if callable(hook):
                hook(language)
                return
            local_voice.prefetch_voice(language)
        except Exception as error:
            logging.info("[Voice] voice prefetch skipped (%s)", error)

    def _on_voice_language_change(self, value: str | None) -> None:
        """Target language changed on /voice → fetch that voice now."""
        if not value:
            return
        self.current_target_language = value
        self._prefetch_voice_for(value)

    def _go_workspace(self, mode: str) -> None:
        # main_page() runs on a FRESH TranslationUI() instance (see
        # start_ui) — setting self.input_mode here would be discarded, so
        # the mode travels via the URL instead.
        ui.navigate.to(f"/?mode={quote(mode)}")

    def voice_translation_page(self):
        self._inject_theme()
        self._inject_api_token()
        self._inject_voice_frontend_helpers()

        # Same header as the workspace; Voice is the active tab.
        with ui.header().classes(f"items-center {theme.HEADER} px-4 py-1"):
            with ui.row().classes("w-full items-center gap-3"):
                ui.html(f'<span class="{theme.WORDMARK}">Passage<b>.</b></span>')\
                    .on("click", lambda: ui.navigate.to("/"))
                ui.element("div").classes("p-header-sep")
                with ui.row().classes("items-center gap-0"):
                    for label, mode in (("Text", "Text"), ("Document", "Document"), ("Image", "Image/Camera")):
                        ui.button(label, on_click=lambda _, m=mode: self._go_workspace(m))\
                            .props("flat no-caps").classes("p-mode-tab")
                    ui.button("Voice").props("flat no-caps").classes("p-mode-tab p-mode-tab-active")

        with ui.column().classes("w-full items-center p-4"):
            with ui.column().classes("w-full max-w-3xl gap-3"):
                with ui.row().classes(f"w-full items-center gap-3 flex-wrap {theme.WELL} p-3"):
                    ui.label("To").classes(theme.DATA)
                    # A real bound input, not a raw <input>. The raw one was
                    # disconnected from `current_target_language`, so /voice —
                    # the one surface that actually needs a Piper voice — could
                    # never tell the app which voice to fetch, and the ~63MB
                    # download landed on the request path (or never happened).
                    self.voice_language_input = ui.input(
                        placeholder="Type a language…",
                        autocomplete=LANGUAGES,
                        on_change=lambda e: self._on_voice_language_change(e.value),
                    ).bind_value(self, "current_target_language")\
                     .props("for=language_select").classes("min-w-[180px]")
                    # Fetch the voice for the language the page opens on, in
                    # the background, before anyone presses Record.
                    self._prefetch_voice_for(self.current_target_language)
                    ui.element("div").classes("flex-grow")
                    # NO inline event-handler attributes here. NiceGUI 3 renders
                    # ui.html() through DOMPurify, which strips them, so the
                    # handler these two buttons used to carry never reached the
                    # DOM and BOTH were dead for every user on every browser --
                    # silently, with no console error, the status label simply
                    # never changing. They are wired by delegated listeners on
                    # `document` (see the script block below), the same pattern
                    # the workspace live-translate JS already uses.
                    ui.html('<button id="desktop_voice_start_recording" class="p-btn p-btn-ok px-4 py-2" '
                            '>● Record</button>')
                    ui.html('<button id="desktop_voice_stop_recording" class="p-btn p-btn-danger px-4 py-2" '
                            'disabled>■ Stop</button>')

                self._render_voice_status_block("desktop_voice")

                with ui.grid(columns=2).classes("w-full gap-3"):
                    with ui.column().classes(f"w-full p-4 gap-2 {theme.PANEL_SOURCE}"):
                        ui.label("Original").classes(theme.DATA)
                        ui.label("").classes("min-h-[60px]").props("id=original_text")
                    with ui.column().classes(f"w-full p-4 gap-2 {theme.PANEL_TARGET}"):
                        ui.label("Translation").classes(theme.DATA)
                        ui.label("").classes("min-h-[60px]").props("id=translated_text")

                ui.audio(src="data:audio/wav;base64,")\
                  .props("id=out_audio controls")\
                  .classes("w-full")

                with ui.expansion("No microphone? Paste a transcript instead").classes(f"w-full {theme.WELL}"):
                    ui.textarea(
                        placeholder="Paste text here if your browser cannot record audio.",
                    ).props("for=desktop_voice_transcript autogrow").classes("w-full")
                    ui.button("Translate transcript", on_click=lambda: ui.run_javascript("translateTranscriptFallback()"))\
                        .classes(f"{theme.BTN_PRIMARY} mt-2 mb-2")

        # Raw string: the SSE parser below needs literal \n in the JS —
        # a plain triple-quote turned it into a real newline, which was a
        # SyntaxError that silently killed this whole script block.
        ui.add_head_html(VOICE_PAGE_JS)

    def _run_voice_on_profile(self, recorder, profile, data: bytes, language: str):
        """Runs in a worker thread: the endpoint override is established HERE,
        because ContextVars are not set by the route on this thread's behalf.

        Returns the backend's (source, translation, audio, meta) tuple with
        meta["translation"] corrected to the engine that really answered, so
        the sentence the page prints and the row the ledger stores are built
        from the same fact.
        """
        translation_label = self._text_engine_label(profile)
        with self.backend.using_profile(profile), \
                self.backend.capture_provenance() as provenance:
            started = time.perf_counter()
            source_text, translated_text, audio_out, meta = \
                self.backend.translate_audio(data, language)
            latency_ms = int((time.perf_counter() - started) * 1000)

        meta = dict(meta or {})
        meta["translation"] = translation_label
        chars = len(source_text or "")
        label, origin = voice_engine_label(meta, translation_label), None
        if not chars:
            label = "none"
        elif provenance.from_cache and _recording_stayed_local(meta):
            # `from_cache` only ever describes the TRANSLATION step — it is
            # written by the backend's translation cache and knows nothing
            # about speech. Booking "cache" on the strength of it alone would
            # print "nothing was sent anywhere" over a recording that had just
            # been uploaded to a hosted transcriber, which is precisely the
            # over-claim this surface exists to prevent. The label is only
            # allowed when the audio itself never left.
            label, origin = "cache", label
        recorder(surface=policy.Surface.VOICE, engine=label, origin=origin,
                 latency_ms=latency_ms, chars=chars)
        return source_text, translated_text, audio_out, meta

    async def api_voice_translate(
        self,
        request: Request,
        file: UploadFile = File(...),
        language: str = Form(...)
    ) -> Response:
        correlation_id = str(uuid.uuid4())
        denied = self._check_api_access(request, correlation_id)
        if denied:
            return denied
        try:
            if not language or language.lower() in ('undefined', 'null', ''):
                language = 'es'
                logging.warning("[API] Empty language → default to Spanish")
            log_event("ui.voice_translate_requested", correlation_id=correlation_id, language=language)
            data = await file.read()
            if not data:
                return Response(content=b"", status_code=400, headers={"X-Error": "Empty audio data"})
            if len(data) > MAX_UPLOAD_BYTES:
                return Response(
                    content=b"Audio file is too large.",
                    status_code=413,
                    media_type="text/plain",
                    headers={"X-Correlation-Id": correlation_id},
                )
            # The spoken path is a translation like every other surface, and
            # until now it was the only one that ran on neither the session's
            # endpoint nor the session's ledger:
            #
            #  * no `using_profile`, so a BYO user's transcript went to
            #    Passage's key while the /voice paste-a-transcript fallback
            #    (api_text_translate_stream) honoured their endpoint — the same
            #    sentence spoken and pasted went to two different companies;
            #  * no ledger call, so after Surface.VOICE lost its last producer
            #    /engines said "Nothing translated yet" over a recording that
            #    had been transcribed, translated and billed.
            #
            # Both are resolved here, on the request context, and applied
            # inside the worker (asyncio.to_thread copies the context, but the
            # overrides have to exist in it before the copy is taken).
            profile = self.active_profile
            recorder = self._engine_recorder()
            with self.backend.cache_scope(self.session_cache_scope):
                original_text, translated_text, audio_bytes, meta = await asyncio.to_thread(
                    self._run_voice_on_profile, recorder, profile, data, language
                )
            safe_original = (original_text or "")[:400]
            safe_translated = (translated_text or "")[:400]
            header_orig = quote(safe_original, safe="")
            header_translated = quote(safe_translated, safe="")
            # Piper returns WAV; the hosted path returns MP3. Sending the wrong
            # media type leaves the browser guessing at the container.
            return Response(
                content=audio_bytes,
                media_type=meta.get("media_type", "audio/mpeg"),
                headers={
                    "X-Original-Text": header_orig,
                    "X-Translated-Text": header_translated,
                    "X-Target-Language": language,
                    "X-Correlation-Id": correlation_id,
                    # Voice is where a user most deserves to know whether their
                    # recording left the machine, so say so explicitly.
                    "X-Speech-Engine": meta.get("stt", "hosted"),
                    "X-Voice-Engine": meta.get("tts", "hosted"),
                    # Human-readable version of the two above, rendered on the
                    # page per request (quoted: it is non-ASCII).
                    "X-Engine-Summary": quote(format_engine_line(meta), safe=""),
                    "Content-Length": str(len(audio_bytes))
                }
            )
        except Exception as e:
            logging.error(f"[API] Voice error: {e}", exc_info=True)
            log_event("ui.voice_translate_failed", correlation_id=correlation_id, error=str(e))
            msg = f"Translation error: {e}"
            return Response(content=msg.encode(), status_code=500,
                            media_type="text/plain", headers={"X-Error": msg})
