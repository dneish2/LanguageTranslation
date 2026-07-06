"""The /voice page: recorder UI, its head-injected JS, and the
/api/voice_translate route. Mixed into TranslationUI so it shares `self`
(backend, api_guard, _check_api_access, _inject_theme, _go_workspace, ...)
without redesigning the app's state model.
"""
import asyncio
import logging
import uuid
from urllib.parse import quote

from nicegui import ui
from fastapi import Request, UploadFile, File, Form
from starlette.responses import Response

import theme
from api_security import MAX_UPLOAD_BYTES
from passage.ui.common import LANGUAGES, log_event


class VoicePageMixin:
    def _render_voice_status_block(self, scope: str) -> None:
        ui.label("Status").classes(f"{theme.DATA} mt-3")
        ui.label("")\
            .classes("text-base")\
            .props(f"id={scope}_status")
        # Developer readout — hidden unless the page is opened with ?debug=1
        # (voiceUx.init reveals .p-debug-block); updateDebug still writes to it.
        ui.label("Debug").classes(f"{theme.DATA} mt-1 hidden p-debug-block")
        ui.label("")\
            .classes(f"{theme.DATA} min-h-[20px] hidden p-debug-block")\
            .props(f"id={scope}_debug")

    def _inject_voice_frontend_helpers(self) -> None:
        ui.add_head_html("""
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
        setRecordingButtons(scope, false);
    }

    // Developer readout: reveal the hidden Debug block only with ?debug=1.
    window.addEventListener('load', () => {
        if (new URLSearchParams(window.location.search).has('debug')) {
            document.querySelectorAll('.p-debug-block').forEach(el => el.classList.remove('hidden'));
        }
    });

    return { states, init, setStatus, setDebug, setRecordingButtons };
})();
</script>
        """)

    def _go_workspace(self, mode: str) -> None:
        self.input_mode = mode
        self.mobile_input_mode = mode
        ui.navigate.to("/")

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
                    options = "".join(f'<option value="{lang}"></option>' for lang in LANGUAGES)
                    ui.html(
                        f'<input id="language_select" list="passage_languages" value="Spanish" '
                        f'placeholder="Type a language…" class="px-3 py-2 p-well">'
                        f'<datalist id="passage_languages">{options}</datalist>'
                    )
                    ui.element("div").classes("flex-grow")
                    ui.html('<button id="desktop_voice_start_recording" class="p-btn p-btn-ok px-4 py-2" '
                            'onclick="startRecording()">● Record</button>')
                    ui.html('<button id="desktop_voice_stop_recording" class="p-btn p-btn-danger px-4 py-2" '
                            'onclick="stopRecording()" disabled>■ Stop</button>')

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
        ui.add_head_html(r"""
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
            const orig = decodeHeader(origHeader);
            const trans = decodeHeader(transHeader);
            document.getElementById('original_text').textContent   = orig;
            document.getElementById('translated_text').textContent= trans;
            if(audio.size>0){
                let url = URL.createObjectURL(audio);
                let player = document.getElementById('out_audio');
                player.src = url; player.play();
                window.voiceUx.setStatus(DESKTOP_SCOPE, window.voiceUx.states.COMPLETE);
            }
        } catch(e) {
            window.voiceUx.setStatus(DESKTOP_SCOPE, "Error: " + e.message);
            window.voiceUx.setDebug(DESKTOP_SCOPE, e.message);
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
        try {
            const resp = await fetch('/api/text_translate_stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded', 'X-Passage-Token': window.PASSAGE_TOKEN || '' },
                body: new URLSearchParams({ text: cleaned, language: lang }),
            });
            const contentType = resp.headers.get('Content-Type') || '';
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
                document.getElementById('original_text').textContent = data.original_text || cleaned;
                document.getElementById('translated_text').textContent = data.translated_text || '';
            }
            window.voiceUx.setStatus(DESKTOP_SCOPE, window.voiceUx.states.COMPLETE);
        } catch (e) {
            window.voiceUx.setStatus(DESKTOP_SCOPE, "Error: " + e.message);
            window.voiceUx.setDebug(DESKTOP_SCOPE, e.message || 'unknown error');
        }
    }

    window.translateTranscriptFallback = translateTranscriptFallback;

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
        """)

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
            original_text, translated_text, mp3_bytes = await asyncio.to_thread(
                self.backend.translate_audio, data, language
            )
            safe_original = (original_text or "")[:400]
            safe_translated = (translated_text or "")[:400]
            header_orig = quote(safe_original, safe="")
            header_translated = quote(safe_translated, safe="")
            return Response(
                content=mp3_bytes,
                media_type="audio/mpeg",
                headers={
                    "X-Original-Text": header_orig,
                    "X-Translated-Text": header_translated,
                    "X-Target-Language": language,
                    "X-Correlation-Id": correlation_id,
                    "Content-Length": str(len(mp3_bytes))
                }
            )
        except Exception as e:
            logging.error(f"[API] Voice error: {e}", exc_info=True)
            log_event("ui.voice_translate_failed", correlation_id=correlation_id, error=str(e))
            msg = f"Translation error: {e}"
            return Response(content=msg.encode(), status_code=500,
                            media_type="text/plain", headers={"X-Error": msg})
