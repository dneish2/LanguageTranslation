"""Speech recognition and synthesis that never leave the machine.

Voice was the last modality with no local path, which made "nothing leaves your
machine" untrue for the most sensitive input the app takes. Recording someone's
voice and shipping it to a third party is a different order of exposure from
translating a sentence they typed, and it was the one place the privacy line
could not be honoured.

Two purpose-built models rather than one general one:

* **faster-whisper** for recognition. Measured on this machine, CPU int8: model
  loads in 0.3s and a 6.6s clip transcribes in **0.6s**. CUDA was tried and
  abandoned — ctranslate2 wants `cublas64_12.dll` and the CUDA runtime wheels
  are ~700MB for a task already fast enough on CPU. CPU is also the portable
  choice, since the deploy target has no GPU.
* **Piper** for synthesis: **0.08s** per sentence. It is a VITS speech model,
  not a language model, so it structurally cannot do what the hosted
  conversational audio model did — answer the text instead of reading it.
  That failure produced fluent, plausible speech that was not the user's
  translation, undetectable to someone who doesn't speak the language. A model
  with no assistant turn cannot make that mistake.

Both are **optional**. They are imported lazily and only when enabled, so the
app runs normally when they aren't installed, and neither is in the core
requirements — the deploy image shouldn't carry a speech stack it won't use.
Every failure falls back to the hosted path rather than breaking voice.
"""
from __future__ import annotations

import io
import logging
import os
import wave
from functools import lru_cache
from pathlib import Path

#: ``PASSAGE_LOCAL_VOICE`` is TRI-STATE (DECISIONS.md §2):
#:
#:   unset -> **auto**: on when the models are actually present. Local voice is
#:            both faster (1.51s vs 7.62s end to end) and more private, so
#:            defaulting off penalised the better option. The old argument for
#:            off was "don't silently move where audio is processed" — but
#:            installing a speech stack and downloading ~63MB of voice weights
#:            is already an explicit act. Nobody does that by accident.
#:   truthy -> force on. If the models are missing this reports that honestly
#:            instead of pretending; a forced flag cannot conjure a model.
#:   falsey -> force off, even when everything is installed.
#:
#: "Truthy"/"falsey" are the obvious spellings, case- and whitespace-insensitive
#: (see _TRUTHY/_FALSEY). This used to accept ONLY "1"/"0" and treat every other
#: value as auto, which meant PASSAGE_LOCAL_VOICE=false on a machine with the
#: models installed turned local voice ON — a config value silently doing the
#: opposite of what it says. Anything still unrecognised is a typo, not a mode:
#: it warns, it is surfaced by status()/describe() as not understood, and the
#: resolved behaviour falls back to auto rather than guessing an intent.
#:
#: Resolved by CALL, not frozen at import: /engines and the tests must be able
#: to re-probe after a voice is downloaded, and an import-time constant would
#: make the page's answer stale the moment the fetch-on-demand helper ran.
_AUTO, _ON, _OFF = "auto", "on", "off"

_TRUTHY = frozenset({"1", "true", "yes", "y", "on", "t"})
_FALSEY = frozenset({"0", "false", "no", "n", "off", "f"})

#: Values already warned about, so a probe called on every page render doesn't
#: reprint the same warning forever. Keyed by the raw value, so CHANGING the
#: variable to a second bad spelling still warns.
_warned_settings: set[str] = set()

#: "base" is the smallest model that transcribes cleanly in testing; tiny
#: garbles proper nouns badly enough to poison the translation downstream.
#:
#: THE ONE LINE. This is the only place a Whisper size is named anywhere in the
#: app — the STT path reads it here and the engine label reported to the user
#: (`meta["stt"] == f"local:{WHISPER_MODEL}"`) is derived from it, so setting
#: PASSAGE_WHISPER_MODEL=small (or editing this default) changes the model and
#: the label together, with nothing else to keep in sync.
#:
#: CAVEAT, and it is a real one: base-vs-small has NEVER been measured on real
#: speech. Every number behind this default came from a synthetic TTS fixture,
#: which is unrealistically clean — exactly the audio that flatters a small
#: model. See RESEARCH.md §4a for why, and for the half-hour experiment that
#: would settle it. Do not switch to "small" on a hunch; measure first.
WHISPER_MODEL = os.getenv("PASSAGE_WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("PASSAGE_WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.getenv("PASSAGE_WHISPER_COMPUTE", "int8")

VOICE_DIR = Path(os.getenv(
    "PASSAGE_PIPER_VOICE_DIR",
    str(Path(__file__).resolve().parent.parent / "models" / "piper"),
))

#: Language name -> Piper voice-file prefix. Piper voices are per-language, so
#: synthesis needs to know which one to load; anything not listed simply has no
#: local voice and falls back to hosted, which is better than mispronouncing a
#: language with the wrong phoneme set.
VOICE_PREFIXES = {
    "english": "en_US",
    "spanish": "es_ES",
    "french": "fr_FR",
    "german": "de_DE",
    "italian": "it_IT",
    "portuguese": "pt_BR",
    "dutch": "nl_NL",
    "polish": "pl_PL",
}


def _lang_key(language: str | None) -> str:
    return (language or "").strip().lower().split("(")[0].strip()


def iso_code(language: str | None) -> str | None:
    """ISO 639-1 code for a language NAME, or None if we don't know it.

    Derived from the voice table rather than guessed. Slicing the first two
    letters of the English name looks like it works and mostly doesn't:
    "Spanish" gives "sp" (it is "es"), "German" gives "ge" ("de"), "Dutch"
    gives "du" ("nl"). Whisper rejects the invalid ones outright, which is the
    good case — the bad case is a code that happens to be valid for a
    DIFFERENT language and quietly transcribes as the wrong one.
    """
    prefix = VOICE_PREFIXES.get(_lang_key(language))
    return prefix.split("_")[0] if prefix else None


def raw_setting() -> str:
    """The env var exactly as set, trimmed. "" when unset."""
    return (os.getenv("PASSAGE_LOCAL_VOICE") or "").strip()


def unrecognised_setting() -> str | None:
    """The raw value if it is set but means nothing, else None.

    A setting nobody can parse must not be swallowed. It is warned once per
    distinct value and reported by status()/describe(), because the failure
    mode it replaces was invisible: PASSAGE_LOCAL_VOICE=false read as auto,
    turned local voice on, and describe() said "on (detected automatically)".
    """
    raw = raw_setting()
    if not raw or raw.casefold() in _TRUTHY or raw.casefold() in _FALSEY:
        return None
    if raw not in _warned_settings:
        _warned_settings.add(raw)
        logging.warning(
            "[LocalVoice] PASSAGE_LOCAL_VOICE=%r is not understood; expected one of "
            "%s (on) or %s (off), or unset for automatic. Falling back to automatic.",
            raw, "/".join(sorted(_TRUTHY)), "/".join(sorted(_FALSEY)),
        )
    return raw


def mode() -> str:
    """The tri-state, read fresh each call: "auto", "on" or "off"."""
    raw = raw_setting().casefold()
    if raw in _TRUTHY:
        return _ON
    if raw in _FALSEY:
        return _OFF
    unrecognised_setting()   # warns once if it is set-but-meaningless
    return _AUTO


def whisper_installed() -> bool:
    """Is the recogniser importable at all? Independent of the env flag.

    Deliberately uncached: a cached "no" would survive an install for the life
    of the process, and this is the probe /engines uses to tell the user what
    the app can do right now. ``import`` is itself cached by ``sys.modules``.
    """
    try:
        import faster_whisper  # noqa: F401
    except Exception:
        return False
    return True


def whisper_repo_id() -> str:
    """The Hugging Face repo the configured size resolves to.

    faster-whisper maps a bare size name onto Systran's converted CTranslate2
    repos; anything containing a "/" is already a repo id and is passed through
    untouched. The size itself is never named here — it comes from the one line
    above, so the probe and the reported label can never disagree.
    """
    name = (WHISPER_MODEL or "").strip()
    return name if "/" in name else f"Systran/faster-whisper-{name}"


def _hf_cache_roots() -> list[Path]:
    """Where Hugging Face may have put the weights, most specific first.

    Read per call, not at import: the tests point HF_HUB_CACHE at an empty
    directory to simulate a fresh machine, and /engines must re-probe after a
    download rather than answer from a frozen constant.
    """
    roots: list[Path] = []
    hub = os.getenv("HF_HUB_CACHE")
    if hub:
        roots.append(Path(hub))
    home = os.getenv("HF_HOME")
    if home:
        roots.append(Path(home) / "hub")
    if not roots:
        roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return roots


def whisper_weights_present() -> bool:
    """Are the Whisper WEIGHTS actually on disk for the configured model?

    Importability is not readiness. `_whisper()` constructs `WhisperModel(...)`,
    which fetches ~140MB from Hugging Face the first time — on the request path,
    during somebody's first recording. Reporting "ready" on the strength of an
    `import faster_whisper` meant /engines and /diagnostics promised "recordings
    are transcribed on this machine" while the real first recording either
    stalled on a download or fell back to the hosted model. The user's voice
    left the machine after being told it would not; that is a privacy
    overclaim, not a performance nit.

    So this asks the same question `voice_file_for` asks of Piper: is the file
    there? A CTranslate2 model directory is identified by `model.bin` inside a
    snapshot, which is what `WhisperModel` loads. If the answer cannot be
    established, callers report the distinct "installed but not downloaded"
    state rather than guessing "ready".
    """
    direct = Path(WHISPER_MODEL) if WHISPER_MODEL else None
    if direct is not None and direct.is_dir():
        return (direct / "model.bin").is_file()

    folder = "models--" + whisper_repo_id().replace("/", "--")
    for root in _hf_cache_roots():
        snapshots = root / folder / "snapshots"
        if not snapshots.is_dir():
            continue
        for snapshot in snapshots.iterdir():
            if (snapshot / "model.bin").is_file():
                return True
    return False


#: The three honest answers about local recognition. "not_downloaded" exists
#: because collapsing it into either neighbour lies: into "ready" it promises
#: local transcription that will not happen, into "not_installed" it hides a
#: stack that is one prefetch away from working.
STT_OFF, STT_NOT_INSTALLED, STT_NOT_DOWNLOADED, STT_READY = (
    "off", "not_installed", "not_downloaded", "ready",
)

#: How each state reads on /engines. Labels live with the states so a new state
#: cannot be added without deciding what the user is told.
STT_LABELS = {
    STT_OFF: "off",
    STT_NOT_INSTALLED: "not installed",
    STT_NOT_DOWNLOADED: "installed, model not downloaded yet — recordings use the hosted model",
    STT_READY: "ready",
}


def stt_state() -> str:
    """Which of the three (plus off) states local recognition is really in."""
    if mode() == _OFF:
        return STT_OFF
    if not whisper_installed():
        return STT_NOT_INSTALLED
    if not whisper_weights_present():
        return STT_NOT_DOWNLOADED
    return STT_READY if enabled() else STT_OFF


def piper_installed() -> bool:
    try:
        import piper  # noqa: F401
    except Exception:
        return False
    return True


def _is_complete(onnx: Path) -> bool:
    """A voice is its WEIGHTS PLUS its config. One file is not a voice.

    Piper loads `<stem>.onnx` together with `<stem>.onnx.json` (phoneme map,
    sample rate) and cannot load without the second. So "installed" is defined
    here, once, and every probe below is derived from it.
    """
    return onnx.is_file() and Path(str(onnx) + ".json").is_file()


def installed_voices() -> list[str]:
    """Stems of the COMPLETE voices on disk.

    Incomplete .onnx files are ignored rather than listed: an interrupted
    `piper.download_voices` (the install path requirements-local-voice.txt
    documents) leaves weights with no config, and calling that a voice made
    /engines claim a capability that fails at synthesis time.
    """
    if not VOICE_DIR.is_dir():
        return []
    return sorted(p.stem for p in VOICE_DIR.glob("*.onnx") if _is_complete(p))


def enabled() -> bool:
    """Whether local voice should be USED, after resolving the tri-state.

    "on" here means "permitted and at least one local capability is really
    present" — never "the flag was set". A flag cannot make a model exist, and
    reporting otherwise is exactly the lie /engines is supposed to prevent.

    There is deliberately no ``ENABLED`` constant to override any more. The
    whole point of the tri-state is that enablement is re-resolved per call, so
    a stored boolean anyone can pin would reintroduce exactly the staleness
    this replaced — and a pinned "off" is indistinguishable from a genuine
    absence of models.
    """
    current = mode()
    if current == _OFF:
        return False
    if current == _ON:
        # Forced on: honour the intent for the capabilities that exist, and let
        # describe() say plainly when the answer is "none of them".
        return True
    # Weights, not imports: an importable faster_whisper with nothing on disk is
    # not a local capability, it is a 140MB download waiting to happen on the
    # request path. Same standard the Piper side has always been held to.
    return (whisper_installed() and whisper_weights_present()) or (
        piper_installed() and bool(installed_voices()))


def stt_available() -> bool:
    """Will a recording ACTUALLY be transcribed here? Nothing weaker.

    Everything user-visible about voice privacy hangs off this one answer —
    `policy.voice_stays_local`, `describe_voice_privacy`, the engine ledger,
    /diagnostics — so it must mean the transcription will happen locally, not
    that the module imported.
    """
    return stt_state() == STT_READY


def voice_file_for(language: str | None) -> Path | None:
    """A COMPLETE Piper voice for this language, or None if we don't have one.

    Selects the first complete pair, not the first sorted `.onnx`. That
    distinction is the whole bug: this used to return the alphabetically first
    match and `voice_installed` then asked whether THAT file had a config. A
    single stale `es_ES-carlfm-x_low.onnx` — left by a Ctrl-C'd
    `piper.download_voices`, the install path we document — sorts before
    `es_ES-davefx-medium` and therefore answered for it forever. The strict
    probe never became True even after a perfect 63MB download, so the
    pre-fetch re-downloaded on every page load, and the loose probe handed
    `synthesize()` the broken file, so local TTS failed on every request.

    There is deliberately no loose/strict split any more: that seam is what let
    two probes disagree about the same directory.
    """
    prefix = VOICE_PREFIXES.get(_lang_key(language))
    if not prefix or not VOICE_DIR.is_dir():
        return None
    for candidate in sorted(VOICE_DIR.glob(f"{prefix}-*.onnx")):
        if _is_complete(candidate):
            return candidate
    return None


def tts_available(language: str | None) -> bool:
    if not enabled():
        return False
    if not piper_installed():
        return False
    return voice_file_for(language) is not None


@lru_cache(maxsize=1)
def _whisper():
    from faster_whisper import WhisperModel
    logging.info("[LocalVoice] loading whisper %s on %s/%s",
                 WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE)
    return WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)


@lru_cache(maxsize=4)
def _piper(voice_path: str):
    from piper import PiperVoice
    logging.info("[LocalVoice] loading piper voice %s", Path(voice_path).name)
    return PiperVoice.load(voice_path)


def transcribe(audio_bytes: bytes, *, language: str | None = None) -> str:
    """Transcribe locally. Raises if unavailable — callers fall back to hosted.

    Whisper is given the audio as a file-like object rather than a path so
    nothing is written to disk: a temp file of somebody's recorded voice is
    exactly the artifact this module exists to avoid creating.
    """
    if not stt_available():
        raise RuntimeError("Local speech recognition is not available.")
    # No code means "let Whisper detect it", which is the right fallback:
    # a wrong hint is worse than none, and detection is reliable on clear audio.
    segments, _info = _whisper().transcribe(
        io.BytesIO(audio_bytes), beam_size=1, language=iso_code(language),
    )
    return " ".join(segment.text for segment in segments).strip()


def synthesize(text: str, *, language: str) -> bytes:
    """Synthesize locally, returning WAV bytes. Raises if unavailable."""
    voice_path = voice_file_for(language)
    if not tts_available(language) or voice_path is None:
        raise RuntimeError(f"No local voice installed for {language}.")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        _piper(str(voice_path)).synthesize_wav(text, wav_file)
    return buffer.getvalue()


def status() -> dict:
    """The RESOLVED state, structured, for /engines and for tests.

    /engines once printed "hosted — metered" directly above "100% stayed on
    this machine", because the two lines were derived from different things.
    Everything that describes voice now derives from this one probe, so the
    page cannot contradict what actually ran.
    """
    current = mode()
    state = stt_state()
    stt = state == STT_READY
    voices = installed_voices()
    return {
        "mode": current,                      # what was asked for
        "setting": raw_setting(),             # verbatim, so the UI can quote it
        # Set-but-meaningless. Never None-and-fine: a value we couldn't parse
        # has to reach the surface, or it configures the app by accident.
        "unrecognised_setting": unrecognised_setting(),
        "enabled": enabled(),                 # what was resolved
        "stt_ready": stt,
        # The three-state answer alongside the boolean, because "not ready"
        # covers two situations a user can act on differently.
        "stt_state": state,
        "stt_engine": f"local:{WHISPER_MODEL}" if stt else "hosted",
        "piper_ready": enabled() and piper_installed() and bool(voices),
        "voices": voices,
        # Forced on with nothing installed is the one case that must not be
        # allowed to read as success.
        "forced_but_missing": current == _ON and not (stt or (piper_installed() and voices)),
    }


def describe() -> str:
    """What's actually available, for the engines page."""
    state = status()
    if state["mode"] == _OFF:
        return (f"off (PASSAGE_LOCAL_VOICE={state['setting']}) — "
                "recordings use the hosted models")
    detail = " · ".join((
        # Three states, three sentences. "not installed" used to cover the
        # not-yet-downloaded case too, and "ready" covered it before that.
        f"speech recognition: {STT_LABELS.get(state['stt_state'], 'not installed')}",
        f"voices: {', '.join(state['voices']) if state['voices'] else 'none installed'}",
    ))
    if state["forced_but_missing"]:
        return (f"PASSAGE_LOCAL_VOICE={state['setting']} but nothing local is installed — "
                f"falling back to hosted ({detail})")
    # A value we could not parse is reported before anything else it affected:
    # it silently chose this behaviour, so the page must not present the result
    # as if it had been asked for.
    bad = state["unrecognised_setting"]
    prefix = (f"PASSAGE_LOCAL_VOICE={bad} is not understood (expected 1/0, true/false, "
              "yes/no or on/off) — using automatic detection · ") if bad else ""
    if not state["enabled"] or not (state["stt_ready"] or state["piper_ready"]):
        return f"{prefix}off — nothing local detected ({detail})"
    how = "forced on" if state["mode"] == _ON else "on (detected automatically)"
    return f"{prefix}{how} · {detail}"


# --------------------- VOICE DOWNLOAD (fetch on demand) ---------------------
#
# Each Piper voice is ~63MB. Bundling several bloats the image; pure on-demand
# means the first use of a language is slow AND needs the network - the worst
# possible failure for a feature whose entire selling point is not needing the
# network. So: fetch on demand, and pre-fetch the current target language in
# the background as soon as the real target language is known (see
# TranslationUI.request_voice_prefetch, called from the "To" input's change
# handler on both the workspace and /voice).
#
# Nothing here is ever on the critical path of a translation. `ensure_voice`
# raises nothing and returns None on any failure; the caller then uses hosted
# TTS and says so via meta["tts"].

#: Where voices are fetched from. Piper's published voices live in the
#: rhasspy/piper-voices repo on Hugging Face; overridable for mirrors, and for
#: tests, which must never pull 63MB.
VOICE_DOWNLOAD_BASE = os.getenv(
    "PASSAGE_PIPER_VOICE_URL",
    "https://huggingface.co/rhasspy/piper-voices/resolve/main",
)

#: Voice-file prefix -> (repo path under the base URL, voice file stem).
#: Explicit rather than derived: the repo layout is lang/locale/speaker/quality
#: and the speaker name cannot be computed from the locale.
VOICE_ASSETS = {
    "en_US": ("en/en_US/lessac/medium", "en_US-lessac-medium"),
    "es_ES": ("es/es_ES/davefx/medium", "es_ES-davefx-medium"),
    "fr_FR": ("fr/fr_FR/siwis/medium", "fr_FR-siwis-medium"),
    "de_DE": ("de/de_DE/thorsten/medium", "de_DE-thorsten-medium"),
    "it_IT": ("it/it_IT/riccardo/x_low", "it_IT-riccardo-x_low"),
    "pt_BR": ("pt/pt_BR/faber/medium", "pt_BR-faber-medium"),
    "nl_NL": ("nl/nl_NL/mls/medium", "nl_NL-mls-medium"),
    "pl_PL": ("pl/pl_PL/darkman/medium", "pl_PL-darkman-medium"),
}

#: A voice download is ~63MB; a stalled socket must not pin a thread forever.
DOWNLOAD_TIMEOUT = float(os.getenv("PASSAGE_PIPER_DOWNLOAD_TIMEOUT", "60"))

#: Guards two page loads racing to fetch the same voice. Per-process only - a
#: second process would redo the work, which is wasteful but still correct,
#: because publication is an atomic rename either way.
_download_locks: dict = {}


def voice_installed(language: str | None) -> bool:
    """Is a COMPLETE voice on disk for this language?

    Now exactly `voice_file_for(...) is not None` — the same question, asked
    once. It used to be the "strict" half of a two-probe split (glob for the
    .onnx here, check the .json there) and the two could disagree about the
    same directory; see `voice_file_for`. Kept as a name because it reads
    better at the call sites that are asking about state rather than a path.
    """
    return voice_file_for(language) is not None


def _fetch_url(url: str, destination: Path, *, timeout: float) -> None:
    """Stream one URL to a path. Isolated so tests can replace the network."""
    from urllib.request import urlopen
    with urlopen(url, timeout=timeout) as response:  # noqa: S310 - fixed base URL
        with open(destination, "wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)


def ensure_voice(language: str | None, *, timeout: float | None = None) -> Path | None:
    """Return a usable voice path for `language`, downloading it if missing.

    Never raises, and never partially shadows a good voice:

    * both files stream into dot-prefixed `.part` temporaries first, so a
      killed download leaves nothing that `voice_file_for`'s `*.onnx` glob can
      mistake for an installed voice;
    * the .onnx.json is renamed into place BEFORE the .onnx, because the .onnx
      is what the probe looks for - publishing it last means the voice becomes
      visible only once it is complete;
    * an already-complete voice short-circuits, so a re-fetch can never replace
      a working file with a truncated one.

    This is slow by nature (~63MB) and must only be called off the request
    path, from the background pre-fetch. A translation that finds no voice uses
    hosted TTS and reports it rather than waiting for this.
    """
    if voice_installed(language):
        return voice_file_for(language)

    prefix = VOICE_PREFIXES.get(_lang_key(language))
    asset = VOICE_ASSETS.get(prefix or "")
    if asset is None:
        logging.info("[LocalVoice] no downloadable voice for %s", language)
        return None
    repo_path, stem = asset

    import threading
    lock = _download_locks.setdefault(stem, threading.Lock())
    with lock:
        # Another thread may have finished while this one waited for the lock.
        if voice_installed(language):
            return voice_file_for(language)

        target_dir = VOICE_DIR
        onnx = target_dir / (stem + ".onnx")
        config = target_dir / (stem + ".onnx.json")
        # Dot-prefixed and NOT ending in .onnx: invisible to the install probe.
        temp_onnx = target_dir / ("." + stem + ".onnx.part")
        temp_config = target_dir / ("." + stem + ".onnx.json.part")
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            wait = DOWNLOAD_TIMEOUT if timeout is None else timeout
            logging.info("[LocalVoice] fetching voice %s (~63MB)", stem)
            _fetch_url(VOICE_DOWNLOAD_BASE + "/" + repo_path + "/" + stem + ".onnx.json",
                       temp_config, timeout=wait)
            _fetch_url(VOICE_DOWNLOAD_BASE + "/" + repo_path + "/" + stem + ".onnx",
                       temp_onnx, timeout=wait)
            if temp_onnx.stat().st_size == 0 or temp_config.stat().st_size == 0:
                raise RuntimeError("empty voice download")
            os.replace(temp_config, config)   # config first...
            os.replace(temp_onnx, onnx)       # ...weights last: now it is installed
            logging.info("[LocalVoice] voice ready: %s", stem)
            return onnx if voice_installed(language) else None
        except Exception as error:
            logging.info("[LocalVoice] voice fetch failed for %s (%s)", language, error)
            for leftover in (temp_onnx, temp_config):
                try:
                    leftover.unlink()
                except OSError:
                    pass
            return None


def prefetch_voice(language: str | None) -> None:
    """Kick off a voice download in the background. Fire-and-forget.

    Same shape as `TranslationBackend.prewarm_live`: off the render path, in a
    daemon thread, failures swallowed and logged, never awaited by the UI. If
    it finishes in time the next translation speaks locally; if it does not,
    that translation goes hosted and says so. Nothing blocks on this.
    """
    # Downloads follow the same switch synthesis does — `enabled()`, the
    # tri-state above — so we can never fetch 63MB for a feature that is off.
    # Recognition is language-independent, so it is fetched from the same hook
    # rather than waiting for a language the user may never pick. Its own guards
    # make this a no-op whenever the weights are already there.
    prefetch_whisper()
    if not enabled() or voice_installed(language):
        return
    if VOICE_PREFIXES.get(_lang_key(language)) is None:
        return

    def fetch() -> None:
        try:
            ensure_voice(language)
        except Exception as error:  # ensure_voice swallows already; belt and braces
            logging.info("[LocalVoice] background prefetch skipped (%s)", error)

    import threading
    threading.Thread(target=fetch, daemon=True).start()


#: Single-flight guard for the recogniser download, so a burst of page loads
#: starts one fetch rather than one per render. Per process, like the voice
#: locks: a second process redoing the work is wasteful, never incorrect.
_whisper_prefetch_lock = None
_whisper_prefetch_started = False


def prefetch_whisper() -> None:
    """Pull the Whisper weights in the background. Fire-and-forget.

    The asymmetry this closes: the 63MB Piper download was deliberately moved
    off the request path by `prefetch_voice`, while the ~140MB recogniser
    download stayed ON it, inside the first `transcribe()`. Same shape as
    `prefetch_voice` and `TranslationBackend.prewarm_live` — daemon thread,
    failures swallowed and logged, never awaited, nothing blocks on it.

    It cannot make the app claim more than it has: readiness is still decided by
    `whisper_weights_present()` reading the disk, so /engines only says "ready"
    once this has actually finished.
    """
    global _whisper_prefetch_lock, _whisper_prefetch_started
    if mode() == _OFF or not whisper_installed() or whisper_weights_present():
        return

    import threading
    if _whisper_prefetch_lock is None:
        _whisper_prefetch_lock = threading.Lock()
    with _whisper_prefetch_lock:
        if _whisper_prefetch_started:
            return
        _whisper_prefetch_started = True

    def fetch() -> None:
        try:
            # Constructing the model IS the download; doing it here means the
            # first recording finds a warm cache instead of a 140MB stall.
            _whisper()
        except Exception as error:
            logging.info("[LocalVoice] whisper prefetch skipped (%s)", error)

    threading.Thread(target=fetch, daemon=True).start()
