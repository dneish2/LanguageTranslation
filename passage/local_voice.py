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
#:   "1"   -> force on. If the models are missing this reports that honestly
#:            instead of pretending; a forced flag cannot conjure a model.
#:   "0"   -> force off, even when everything is installed.
#:
#: Resolved by CALL, not frozen at import: /engines and the tests must be able
#: to re-probe after a voice is downloaded, and an import-time constant would
#: make the page's answer stale the moment the fetch-on-demand helper ran.
_AUTO, _ON, _OFF = "auto", "on", "off"

#: "base" is the smallest model that transcribes cleanly in testing; tiny
#: garbles proper nouns badly enough to poison the translation downstream.
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


def mode() -> str:
    """The tri-state, read fresh each call: "auto", "on" or "off"."""
    raw = (os.getenv("PASSAGE_LOCAL_VOICE") or "").strip()
    if raw == "1":
        return _ON
    if raw == "0":
        return _OFF
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


def piper_installed() -> bool:
    try:
        import piper  # noqa: F401
    except Exception:
        return False
    return True


def installed_voices() -> list[str]:
    return sorted(p.stem for p in VOICE_DIR.glob("*.onnx")) if VOICE_DIR.is_dir() else []


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
    return whisper_installed() or (piper_installed() and bool(installed_voices()))


def stt_available() -> bool:
    if not enabled():
        return False
    return whisper_installed()


def voice_file_for(language: str | None) -> Path | None:
    """The Piper voice for this language, or None if we don't have one."""
    prefix = VOICE_PREFIXES.get(_lang_key(language))
    if not prefix or not VOICE_DIR.is_dir():
        return None
    for candidate in sorted(VOICE_DIR.glob(f"{prefix}-*.onnx")):
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
    stt = stt_available()
    voices = installed_voices()
    return {
        "mode": current,                      # what was asked for
        "enabled": enabled(),                 # what was resolved
        "stt_ready": stt,
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
        return "off (PASSAGE_LOCAL_VOICE=0) — recordings use the hosted models"
    detail = " · ".join((
        f"speech recognition: {'ready' if state['stt_ready'] else 'not installed'}",
        f"voices: {', '.join(state['voices']) if state['voices'] else 'none installed'}",
    ))
    if state["forced_but_missing"]:
        return ("PASSAGE_LOCAL_VOICE=1 but nothing local is installed — "
                f"falling back to hosted ({detail})")
    if not state["enabled"] or not (state["stt_ready"] or state["piper_ready"]):
        return f"off — nothing local detected ({detail})"
    how = "forced on" if state["mode"] == _ON else "on (detected automatically)"
    return f"{how} · {detail}"
