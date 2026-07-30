# Passage

A translation workspace that tells you which engine served every request, and can prove it.

Paste text, drop a document, photograph a menu, or speak. Passage routes the work to a model on
your own machine when one is reachable, to your own endpoint if you brought one, and to a hosted
model otherwise — and then names, per request, which of those actually answered. Everything below
about model behaviour was measured against the running app rather than read off a model card;
`passage/model_bench.py` reproduces it.

**Origin.** This started as an internal tool for people doing real translation work, and it was
paid for. It is no longer that, and this repository is the rebuild: same problem, rewritten around
a question the original never asked — *where did this text actually go?*

> **`TODO(david)`** — replace the sentence above with the specifics: what the internal tool was
> for, roughly who used it, and what it replaced. That detail is the difference between "side
> project" and "shipped system with users", and it is the one claim here nobody but you can write.

---

## Three things this project measured

Each of these changed the code. None of them is a benchmark table for its own sake.

### 1. A prompt is not a mechanism

Asked to translate a sentence containing URLs, email addresses and figures — with an explicit
instruction not to alter them — **six of seven models mangled them anyway.** That is a property of
the model class, not one bad model.

So URL and figure protection is not asked for in the prompt. It is done in code
(`TranslationBackend._mask_protected_spans`), deterministically, and the test asserts the
*mechanism* rather than the wording. The same rule now applies everywhere in this codebase that a
model invariant matters: mask, route or validate it — never request it.

### 2. Thinking models are the wrong tool for translation, and the right one for extraction

`qwen3:30b` spent **8,786 characters deliberating a one-line translation and produced no answer.**
Through Ollama's OpenAI-compatible shim the qwen3 family's output lands in a reasoning channel the
endpoint doesn't expose, so `content` comes back empty and they look like silent failures.

`think: false` is not the fix — on a translation prompt it makes the model write its reasoning
*into* the answer. But the same flag on the same model is exactly right for **extraction**, where
deliberation is pure cost: with it, `qwen3-vl:8b` reads a full menu photograph in **2.6 s**,
offline. Translation is transduction; there is no token budget that reliably bounds deliberation a
task never needed. `ollama_native.suits_translation()` keeps thinking models out of default rosters
while leaving them selectable.

### 3. A purpose-built 4B beat a general 27B

| model | size | p50 | agreement |
|---|---|---|---|
| **translategemma:4b** | 3.3 GB | 368 ms | **0.788** ← default |
| gemma3:12b | 8.2 GB | 582 ms | 0.787 |
| translategemma:27b | 17.4 GB | 679 ms | 0.734 |
| gemma3:1b | 0.8 GB | 334 ms | 0.668 |

The 4B outscores the 27B *from its own family* at a fifth the size. And the smallest model isn't
approximate, it's **wrong**: `gemma3:1b` rendered "the board pushed back on the buyback" as *"el
consejo reguló la compra de acciones"* — "the council regulated the share purchase". A different
claim, confidently stated.

Agreement is mean similarity to the other models on the same input. There is no reference
translation to score against, so this rewards the mainstream reading and flags the outlier — it is
triage, not a leaderboard. A model can be right and alone.

Full method, the models that didn't make the table, and the camera/overlay work:
**[RESEARCH.md](RESEARCH.md)**.

---

## One run is not a result

An overlay-alignment fix was first reported as **17.5 px → 9.2 px** — a halving. Across five runs
it was **23.5 → 20.5 px, better in 4 of 5**, because the vision model's boxes vary noticeably run
to run and one lucky run is not a result.

The standing rule, which the rest of this repo is held to: anything whose result depends on a
model's output is reported as the **median across at least five runs, together with how many runs
improved**. Deterministic changes — masking, routing, parsing — can be measured once. The
distinction is whether sampling is in the loop.

The same discipline is why the shipped Whisper default carries a written statement that **the
evidence under it doesn't count**: every speech measurement so far used a TTS-generated fixture,
which has no accent, no noise and no phone-mic band limiting, and therefore flatters a small model
by construction. The protocol that would settle it is written down and deliberately not yet run.

## The honesty layer, and the defect shape underneath it

`/engines` reports what has actually happened to your text this session: how much stayed on this
machine, how much was sent out, what would have been billed. It is session-scoped and disappears
with the session — a permanent log of everywhere your text went would be a strange thing to keep
in the name of privacy.

An adversarial review found a cross-user data leak and three privacy overclaims that 419 passing
tests had missed. It took **six rounds to converge**, and the reason is the interesting part:
every one of those defects had the same shape — *the app deriving a privacy claim from state read
at render time, instead of recording what happened when it happened*. Fixing them one surface at a
time is why it took six rounds.

- A cache hit inferred from a process-wide counter, so an unrelated session's keystroke could print
  "no model ran and nothing was sent anywhere" over a photograph that had just been base64'd to a
  hosted model. Provenance is now a per-call fact ([`capture_provenance`](TranslationBackend.py)).
- A privacy sentence derived from a page-render capability probe, so one response body could report
  `hosted:gpt-5.4-nano` and promise the text never left the machine. Every claim about a completed
  request now comes from [`policy.classify_run`](passage/policy.py).
- A photograph the vision model received and found no text in, which raised before the ledger was
  written — so `/engines` said "Nothing translated yet." about a photo that had demonstrably been
  sent. Fixed at the shape rather than the surface: the disclosure is now booked from the
  provenance captured at the outbound call, on the path every surface shares.

The line between a call that was **delivered and then failed** (a 429, a timeout, an empty reply —
your text is on someone else's computer, and you are told) and one that **never connected** (an
offline session, where saying "sent out" would be a lie to precisely the user who chose this app)
is drawn in [DECISIONS.md §9](DECISIONS.md) — along with the residual case it still gets wrong, and
why erring the other way would be worse.

## What this deliberately does not do

Named because they are the tempting next features, not because they were forgotten.

- **No accounts, no billing, no durable storage.** The auth verification is ported and live-tested;
  it is deliberately unwired. A signup wall in front of a demo mostly measures how many people
  bounce at it. Abuse prevention is handled where the actual risk is — a signed page token, a
  per-IP rate limit, and text/upload caps (see below).
- **No original file is ever durably stored, on any surface.** Segments and traces carry the
  reviewable value; the upload itself is storage, exposure and a deletion obligation without
  product. Enforced in [`passage/policy.py`](passage/policy.py) rather than remembered.
- **No cost table.** Per-model prices go stale and differ per account, so rates are read from the
  environment and anything unpriced reports nothing. A plausible-looking wrong number in a
  comparison table is worse than a blank.
- **No `--min-instances=1`.** It bills continuously for an app with no users. The two Cloud Run
  flags that cost nothing are set; that one isn't.
- **No third overlay heuristic.** Two were tried, one made it worse, a third measured no better and
  was reverted rather than kept for the idea's sake. The remaining error needs real OCR geometry,
  which is a dependency decision and not a tuning one.

## Where things run

| | translation | speech | metered? |
|---|---|---|---|
| **local** | Ollama, chosen from what's installed | faster-whisper + Piper, CPU | never |
| **your endpoint** | any OpenAI-compatible `base_url` | — | never |
| **hosted** | Passage's key | hosted STT/TTS | yes |

Local-first is the default whenever a local model is reachable, and metering asks exactly one
question: *did Passage pay for this inference?* A hosted fallback after a local failure is metered;
a local run, a cached answer and a BYO endpoint never are. Recognise / translate / speak each
choose independently, so a missing voice for one language doesn't force the recording off the
machine.

Local speech runs on CPU, not CUDA — `ctranslate2` wanted a 700 MB CUDA wheel for a task that
transcribes a 6.6 s clip in 0.6 s on CPU. That choice was made for convenience and turned out to be
the portable one: there is no CUDA on Apple Silicon at all. See [PORTABILITY.md](PORTABILITY.md).

---

## Run it locally

Requires Python 3.12. The project uses a `uv` virtual environment at `.venv`.

```bash
git clone https://github.com/dneish2/LanguageTranslation.git
cd LanguageTranslation

uv venv && uv pip install -r requirements.txt

# provide your OpenAI key (a .env file in the repo root also works)
# PowerShell:  $env:OPENAI_API_KEY = "sk-..."
# bash:        export OPENAI_API_KEY=sk-...

.venv/Scripts/python.exe TranslationUI.py   # Windows
# .venv/bin/python TranslationUI.py         # macOS/Linux
```

Open http://localhost:8080. Without an API key the app still boots and serves the UI; translation
calls fail with a clear error until a provider is configured. With
[Ollama](https://ollama.com) running and `translategemma:4b` pulled, text translation goes local
automatically and `/engines` will say so.

Optional, for fully local speech:

```bash
uv pip install -r requirements-local-voice.txt
```

## Pages

| Route | What it is |
|-------|------------|
| `/` | The workspace: Text / Document / Image modes, language bar, facing source/translation panels, segment review for documents, Recent Threads drawer |
| `/voice` | Voice translation: record (or paste a transcript), hear the translation spoken back |
| `/engines` | What actually served this session — local vs sent out, per surface, with what it would have cost |
| `/diagnostics` | What this host resolved: models found, probes run, fonts, capabilities |
| `/mobile` | Mobile layout (being folded into one responsive layout) |

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | Provider key for translation, vision, and hosted speech |
| `PORT` | `8080` | Listen port (Cloud Run injects this) |
| `PASSAGE_PUBLIC_API` | off | Set `1` to disable the API session-token gate (local dev) |
| `PASSAGE_API_RATE_LIMIT` | `30` | Requests per minute per IP on `/api/*` |
| `PASSAGE_MAX_TEXT_CHARS` | `8000` | Max characters per text translation |
| `PASSAGE_MAX_UPLOAD_BYTES` | `8388608` | Max upload size (8 MB) |
| `LIVE_TEXT_STREAMING` | `false` | Enable SSE streaming for long text translations |
| `PASSAGE_LOCAL_VOICE` | unset (auto) | Local speech recognition and synthesis — see below |
| `PASSAGE_WHISPER_MODEL` | `base` | faster-whisper size for local transcription |
| `PASSAGE_PIPER_VOICE_DIR` | `models/piper` | Where Piper voices are stored and fetched into |
| `PASSAGE_TEXT_MODEL` / `_VISION_` / `_TTS_MODEL` … | see `TranslationBackend.py` | Override any hosted model |
| `PASSAGE_RATE_<MODEL>` | unset | USD per 1M output tokens, for the comparison view. Unset means the cost column stays blank rather than guessing |

### Local voice (`PASSAGE_LOCAL_VOICE`)

Speech can run entirely on your machine (faster-whisper for recognition, Piper for synthesis)
instead of going to the hosted provider. Voices are fetched on demand in the background, so nothing
needs downloading up front — and a translation never waits on a voice download; if it isn't there
yet, hosted TTS answers and the response says so.

The variable is **tri-state**:

| Value | Meaning |
|-------|---------|
| unset or empty | **auto** — local voice is used when the models are actually present, hosted otherwise |
| `1`, `true`, `yes`, `y`, `on`, `t` | force on (case- and whitespace-insensitive) |
| `0`, `false`, `no`, `n`, `off`, `f` | force off, even with everything installed |

Anything else is treated as a typo, not a mode: logged once as a warning, reported as "not
understood" on `/engines`, and behaving as auto. The first cut treated everything that wasn't `1`
or `0` as auto, which meant `PASSAGE_LOCAL_VOICE=false` on a machine with the models installed
turned local voice **on** — a config value doing the opposite of what it says.

Forcing it on cannot conjure a model. With nothing installed, `/engines` says so and speech falls
back to hosted; whichever engine actually ran is named per request (`meta["stt"]` / `meta["tts"]`
and the `X-Engine-Summary` header), so a fallback is never mistaken for a private run.

## API

The `/api/*` endpoints (`text_translate`, `text_translate_stream`, `voice_translate`,
`image_translate`) are used by the app's own pages and gated by a short-lived session token those
pages embed (`X-Passage-Token`), plus a per-IP rate limit and text/upload caps. This is abuse
prevention, not authentication — they are not a public API.

## Tests

```bash
.venv/Scripts/python.exe -m pytest tests/
```

589 tests, green on `ubuntu-latest`, `windows-latest` and `macos-14`. The suite gates deployment:
pushes to `main` run tests in CI and, on green, build and deploy to Cloud Run. Secrets live in
GitHub Actions secrets, never in the repo.

The cross-OS matrix earned its keep on its first run, catching a genuine hidden host assumption:
`redact_path` reduced a path to its basename with host-specific separator semantics, so a
Windows-style path went unredacted on POSIX. Both non-Windows runners caught it immediately.

## Design

The visual identity ("Press": warm paper, letterpress ink, burgundy accent; Palatino display over
Georgia body; monospace reserved for data) is defined in `static/passage.css` and mirrored in
`theme.py`. All UI styling flows through those tokens — no ad-hoc colour classes. Motion is synced
to real data movement: progress reflects actual job and segment state, never a decorative spinner.

## The documents

| | |
|---|---|
| **[RESEARCH.md](RESEARCH.md)** | Every model measurement, how it was taken, and what was measured but *not* changed |
| **[DECISIONS.md](DECISIONS.md)** | Each open call, resolved, with the reasoning — so a later session can disagree with evidence rather than guess |
| **[PORTABILITY.md](PORTABILITY.md)** | Whether this runs on an M1, what's verified, and what is explicitly not |
| **[PASSAGE_PLAN.md](PASSAGE_PLAN.md)** | The plan of record: locked decisions, roadmap, current state |
