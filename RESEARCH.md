# Passage — model landscape, measured

Everything here was measured on David's machine (RTX 5090, 32 GB VRAM) against
the running app, not read off a model card. Numbers are reproducible:
`passage/model_bench.py` runs a fixed suite and appends to
`data/model_bench.jsonl`, so a later run can be diffed against these rather
than re-argued.

Last measured: 2026-07-28.

---

## 1. Local translation models

| model | size | p50 | agreement | notes |
|---|---|---|---|---|
| **translategemma:4b** | 3.3 GB | 368 ms | **0.788** | current local default |
| gemma3:12b | 8.2 GB | 582 ms | 0.787 | best general model |
| translategemma:12b | 8.1 GB | 495 ms | 0.779 | |
| qwen2.5:7b | 4.7 GB | 237 ms | 0.763 | fastest usable |
| gemma4:latest | 9.6 GB | 398 ms | 0.741 | |
| translategemma:27b | 17.4 GB | 679 ms | 0.734 | |
| gemma3:1b | 0.8 GB | 334 ms | 0.668 | **not usable** |

**Agreement** is mean similarity to the other models on the same input. There
is no reference translation to score against, so this rewards the mainstream
reading and flags the outlier. It is triage, not a leaderboard — a model can
be right and alone.

### What the numbers actually say

**A purpose-built 4B beats a general 27B.** `translategemma:4b` outscores
`translategemma:27b` from its own family while being five times smaller and
nearly twice as fast. Task-specific beats big, at least at this scale of task.
Before spending VRAM on a larger model, check whether a specialised smaller
one exists.

**The smallest model is wrong, not approximate.** `gemma3:1b` rendered "the
board pushed back on the buyback" as *"el consejo reguló la compra de
acciones"* — "the council regulated the share purchase". That is not a rougher
translation, it is a different claim. Reaching for the tiniest model to save
memory buys a confidently wrong answer.

**Local beats hosted on latency here.** `gpt-5.4-nano` measured 616 ms against
`qwen2.5:7b`'s 237 ms and `translategemma:4b`'s 368 ms, all local, all free.
That inverts the usual assumption that local trades speed for cost. It is a
property of this hardware — a 5090 with the model already resident — and
should be re-measured on any other machine before being relied on.

**6 of 7 models mangled URLs, emails and figures** when asked to translate a
sentence containing them, even though the prompt said not to. This is a
property of the model class, not one bad model, and it is why URL masking in
`TranslationBackend._mask_protected_spans` is deterministic rather than a
prompt instruction.

### Thinking models are the wrong tool for translation

`qwen3:30b` and `qwen3-vl:8b` return empty `content` through Ollama's
OpenAI-compatible shim — their answer goes to a reasoning channel it doesn't
expose. Through the native `/api/chat` (`passage/ollama_native.py`) the
channels stay separate and the answer arrives.

Even then, `qwen3:30b` spent **8,786 characters deliberating a one-line
translation and still produced no answer**, with a 2,048-token reserve on top
of the request. Translation is transduction, not reasoning; there is no budget
that reliably bounds deliberation the task never needed.
`ollama_native.suits_translation()` keeps them out of default rosters while
leaving them selectable.

`think: false` is not the fix. On a translation prompt it makes the model
write its reasoning *into* the answer. It is correct for **extraction**, where
deliberation is pure cost — with it, `qwen3-vl:8b` reads a full menu photo in
**2.6 s**.

---

## 2. Vision and the camera path

**Local OCR works.** `qwen3-vl:8b` reads a photographed Spanish menu in 2.6 s
with accents and currency intact, entirely offline.

**It cannot report where it read.** Asked for bounding boxes it returns
nothing usable; with `format: json` it deliberated for 17,500 characters and
emitted no content. Text and geometry have to come from different places.

**Geometry comes from pixels** (`passage/text_rows.py`): a horizontal
projection profile finds text rows directly, needing no OCR engine. Three
things were required to survive a photograph:

- a **local (blurred) background** rather than a global threshold, or
  lighter-coloured headings read as paper;
- a **page-relative row threshold**, because a fixed one turned textured paper
  into a single band;
- **global exclusion of columns inked down most of the page** — vertical rules
  and paper grain, which otherwise made every row span the full width.

**Correcting the overlay: what didn't work.** The hosted vision model's boxes
sat ~35 px below their text.

1. *Pair detected rows to OCR lines in order.* Unsafe — the detector finds 12
   bands where the model reads 24 lines (a dish and its price share a visual
   row), so captions slide further out of place down the page.
2. *Snap each box to its nearest row.* **Worse.** Line spacing (~43 px) is
   close to the bias (~35 px), so "nearest" often resolved to the line below
   and each box moved confidently to the wrong place.
3. *Fit one correction for the whole page.* Works. Every box moves together,
   so a bad estimate degrades to "slightly off" rather than "scrambled". Both
   **scale and offset** are needed — a pure shift fixed the top of the page
   and left the bottom wrong, because the error grows with distance down.

**Measured across 5 runs** (an earlier single-run figure of 17.5 → 9.2 px was
not representative — the vision model's boxes vary noticeably run to run, and
one lucky run is not a result):

    median across runs   BEFORE 23.5px   AFTER 20.5px
    improved in 4/5 runs; run 4 got worse (23.2 -> 28.2)

So the correction is a **modest and mostly-positive** improvement of roughly
13%, not the halving a single run suggested. It is worth keeping — it is cheap,
it degrades gracefully, and it helps most of the time — but the honest headline
is "somewhat better on average", and the residual error is large enough that
the overlay is still not precise.

A **per-channel ink map** was tried to catch the burgundy section headings the
greyscale threshold misses (the missed rows being where the residual error
concentrates). It found one more row but also a spurious band over blank paper,
and measured no better. Reverted rather than kept for the idea's sake. The
headings problem is real and still open; the fix is not this.

---

## 3. Where things run

`passage/policy.py` holds the retention and metering rules in one place.

- **Session vs durable is the important distinction.** Recent Threads is
  session-cookie-scoped and discarded when the session ends. Treating "camera
  and live typing stay local" as "they persist nothing" would delete a feature
  people like while claiming to protect them.
- **No surface may durably store an original file.** Segments and traces carry
  the reviewable value; the upload itself is storage, exposure and a deletion
  obligation without product.
- **Metering asks one question**: did Passage pay for the inference? Local and
  BYO both mean no — including local-first auto-routing, where billing would
  otherwise charge for electricity the user paid for.

---

## 4. What's next

Ordered by value, with what each depends on.

1. ~~**Per-profile document routing.**~~ **Done.** The session's endpoint now
   travels by `ContextVar` (`TranslationBackend.using_profile`) rather than
   through every signature — document translation runs through half a dozen
   layers, and one missed call site would have silently sent a BYO user's
   document to the app's own key. A ContextVar rather than an attribute
   because the backend is shared across clients. Verified by setting
   `provider = None` so any leak to hosted would raise, then translating a
   27-segment DOCX entirely through a local profile.
2. ~~**Real metering on the policy hook.**~~ **Done.** `passage/usage.py`
   counts characters (not requests — live typing fires ~8 requests per
   sentence while a document is one request carrying thousands). Only metered
   work reaches the counter, so no later billing change can start charging for
   inference somebody else paid for. The free allowance is reported, not
   enforced: a quota that starts refusing work the day it ships is a bad
   surprise. Metering reads where the work ACTUALLY ran, not what was
   configured — a hosted fallback after a local failure is metered.
3. **Better row detection for coloured headings.** The remaining overlay error
   is concentrated in section headings the detector misses. A per-channel ink
   map (not just greyscale) would catch burgundy-on-cream.
4. ~~**Local STT/TTS.**~~ **Done.** faster-whisper + Piper, both optional
   (`requirements-local-voice.txt`) and **on by default once the models are
   actually installed**. `PASSAGE_LOCAL_VOICE` is tri-state (DECISIONS.md §2):
   unset = auto, truthy (`1`/`true`/`yes`/`on`) forces on, falsey
   (`0`/`false`/`no`/`off`) forces off. An unrecognised value warns and falls
   back to auto rather than being silently swallowed.
   Measured on the same 6.6s clip, cache cleared, 3 runs each:

       LOCAL   median 1.51s   stt=local:base  tts=local:piper
       HOSTED  median 7.62s   stt=hosted      tts=hosted

   **5x faster and private.** CPU rather than CUDA: ctranslate2 wants
   `cublas64_12.dll` and the CUDA wheels are ~700MB for a task that already
   transcribes in 0.6s on CPU — which is also the portable choice, as the
   deploy target has no GPU. Piper is a VITS model, not a language model, so
   it structurally cannot answer the text instead of reading it; round-trip
   fidelity 0.950. The three steps (recognise / translate / speak) choose
   local or hosted independently, so a missing voice for one language doesn't
   force the recording off the machine.
5. **Trace persistence for documents.** Policy already permits it; nothing
   writes it. This is the substrate for per-segment scoring and the per-user
   preference dataset.
6. **Re-benchmark on a second machine.** Every latency number here is one
   GPU's opinion. The ranking is more portable than the absolute numbers, but
   neither has been checked elsewhere.

---

## 4a. Decisions waiting on David

Each needed a call rather than more code. All three have now been made and
implemented; they are kept here with their reasoning, struck through, because
the argument matters more than the outcome if any of them is revisited.

- ~~**Which Whisper size ships as the default?**~~ **Decided: keep `base`** —
  see "Whisper size: decided, but on evidence that doesn't count" below. The
  decision is cheap to revisit: `PASSAGE_WHISPER_MODEL` in
  `passage/local_voice.py` is the only place a size is named.
- ~~**Should local voice default ON when the models are present?**~~
  **Decided and shipped: yes** (DECISIONS.md §2). Local voice is both faster
  (1.51s vs 7.62s) and more private, so defaulting off penalised the better
  option; and installing a speech stack plus ~63MB of weights is already an
  explicit act, so "don't silently change where audio is processed" does not
  bite. `PASSAGE_LOCAL_VOICE` became tri-state — unset = auto (on when the
  models are really present), truthy = force on, falsey = force off — and
  every page reports the RESOLVED state, never the flag.
- ~~**Do we ship voices, or fetch them on demand?**~~ **Decided and shipped:
  fetch on demand, pre-fetched in the background** (DECISIONS.md §3). Pure
  on-demand would put a 63MB transfer on the request path, so
  `local_voice.prefetch_voice` starts the download for the target language as
  soon as that language is known and nothing ever waits on it: a translation
  that finds no voice uses hosted TTS and says so in `meta["tts"]`. Downloads
  publish through `.part` temporaries, config first and weights last, so a
  killed transfer can never look installed.

### Whisper size: decided, but on evidence that doesn't count

The shipped default is `base` (DECISIONS.md §1). The honest statement of what
that rests on: **base-vs-small has never been compared on real speech.**

Every measurement in this document — the 0.6s transcription of a 6.6s clip, the
"accurate on the test audio" claim, the 1.51s vs 7.62s local/hosted median — was
taken on a *synthetic fixture generated by a TTS model*. That audio has no
background noise, no accent, no clipping, no room reverb, no phone-mic band
limiting, no disfluencies, and near-perfect articulation with textbook prosody.
It is, structurally, the easiest possible input for a speech recogniser, and the
gap between whisper sizes is precisely the gap that only opens on hard audio.
A synthetic fixture therefore cannot distinguish "base is good enough" from
"base is fine on audio nobody actually produces" — it flatters the small model
by construction. `small` is known to be slower and generally better on accents
and noise; how much better *here* is unmeasured, so switching would be trading a
known latency cost for an unquantified accuracy gain. That is a hunch, not a
decision, which is why `base` stays.

What would settle it, concretely:

1. Record 5–10 real clips of 5–15s each, deliberately spanning the hard cases:
   at least two non-native or strongly accented speakers, at least two with
   background noise (café, traffic, a fan), and at least two captured on a phone
   mic rather than a headset — the phone-mic case matters most, because `/voice`
   is used on phones.
2. Write a reference transcript for each by hand. This is the part that costs
   real time and the part that cannot be skipped or model-generated.
3. Transcribe each clip with `PASSAGE_WHISPER_MODEL=base` and again with
   `=small`, recording word error rate and wall-clock per clip.
4. Compare median WER and median latency across the set, and report how many of
   the clips each size won — a single clip's difference means nothing.

That is roughly half an hour of work, most of it transcription, and it replaces
an assumption sitting under a shipped default with a fact. If `small` wins on
WER by a margin that matters and the latency cost is tolerable on the target
hardware, changing the default is a one-line edit at `WHISPER_MODEL` in
`passage/local_voice.py` — nothing else in the app names a size, and the engine
label the user sees is derived from the same constant. Note also that all
timings above are one machine's opinion (§4 item 6); a WER study travels between
machines far better than a latency study does.

### Reporting model-dependent numbers

Per DECISIONS.md §6: any result that depends on a model's output must be
reported as the **median across at least 5 runs**, together with **how many runs
improved**. One run is not a result. The overlay fix is the standing example —
a single run read 17.5px → 9.2px, which looked like a halving; five runs read
23.5 → 20.5px, better in 4/5, which is a modest real improvement and a very
different thing to claim. Deterministic changes (masking, routing, parsing) can
be measured once; the distinction is whether sampling is in the loop. The WER
study above is deterministic per clip given a fixed model and greedy decoding,
so N=1 per clip is fine there — what needs the N is the *set of clips*, not
repeated runs of the same one.

## 4b. What this round measured, as distinct from what it changed

Most of this round was changes, not measurements. Two things were actually measured, and they are
kept separate here so nobody later reads a change as evidence.

### `/api/health` blocks the event loop

Measured on the Windows dev machine (RTX 5090) by timing a request to `/api/health` while the
diagnostics probe was reaching for an unreachable local model:

| condition | wall clock | 50 ms heartbeat ticks observed |
|---|---|---|
| probe target is a **silent socket** (connect accepted, no response) | **15.31 s** | **0** |
| probe target is a **refused port** | **8.32 s** | — |

Zero heartbeat ticks over 15.31 s is the finding. The handler is not merely slow; it holds the
event loop for the whole duration, so nothing else on the server progresses — a silent socket
anywhere in the probe chain stalls every other request. The refused-port case is faster only
because the OS answers immediately, and 8.32 s is still far past anything a health endpoint should
cost. These are single measurements of a deterministic blocking condition, not model-dependent
sampling, so N=1 is appropriate; the numbers belong to this machine and these two socket
conditions and should not be quoted as portable.

That is the motivating evidence for the event-loop fix.

#### After the fix — re-measured at integration, N=5 per condition, medians

The fix was measured twice by different parties. The numbers below are the integration
re-measurement, taken at **production timeouts** (2.5 s probe + 3.0 s refusal check) against the
same two real socket conditions, 5 runs each, median reported with every run listed. The "before"
row is the pre-fix route shape reconstructed exactly: four sequential blocking probes from an
`async def`.

| condition | wall clock (median of 5) | 50 ms heartbeat ticks (median) | probes/request |
|---|---|---|---|
| silent socket, **before** | **10.07 s** `[10.07, 10.07, 10.04, 10.09, 10.09]` | **0** `[0,0,0,0,0]` | 4 |
| silent socket, **after** | **2.53 s** `[2.76, 2.52, 2.53, 2.53, 2.52]` | **41** `[45,41,41,41,41]` | 1 |
| refused port, **before** | **8.14 s** `[8.14, 8.14, 8.11, 8.15, 8.15]` | **0** `[0,0,0,0,0]` | 4 |
| refused port, **after** | **2.04 s** `[2.05, 2.04, 2.04, 2.05, 2.04]` | **34** `[34,33,34,34,34]` | 1 |

5/5 runs improved in both conditions. **The wall-clock drop (~4x) is the de-duplication of the four
probes into one; the number that matters is heartbeat ticks 0 → 41 and 0 → 34** — other clients are
served throughout, which is the denial-of-service shape actually being fixed. A second request
inside the 60 s TTL costs zero probes.

The silent-socket "before" figure here is 10.07 s against the 15.31 s recorded above. Same shape,
different socket timing on a different day; both are reported rather than the more flattering one
being chosen, and neither is portable off this machine.

### The CI matrix caught a real host assumption

Run 30395914573 was the first execution of this project outside Windows: `windows-latest` PASS,
`ubuntu-latest` FAIL, `macos-14` FAIL, nightly speech job skipped (it has not run). The failure was
`redact_path` reducing a path to its basename with host-specific separator semantics, so a
Windows-style path went unredacted on POSIX. Confirmed from runner output on that run: the
`macos-14` Python is arm64 (`/Users/runner/hostedtoolcache/Python/3.11.9/arm64`). Everything else
about Apple Silicon remains unverified — the runners have no Ollama and no GPU. See
`PORTABILITY.md`.

### What was changed but not measured

Local voice tri-state parsing, Piper fetch-on-demand plus background prefetch, and the single
complete-pair voice probe are all covered by tests that assert which path ran, but none of them
carries a before/after latency number. They are correctness changes, and claiming a performance
result for them would be inventing one.

## 5. Prompt for the next session

Paste this to continue. It assumes nothing beyond the repo.

```
Read RESEARCH.md in this repo first — it holds measured results you should not
re-derive.

Task: pick up the "What's next" list in RESEARCH.md §4, starting with item 1
(per-profile document routing) and item 2 (metering on the policy hook), which
belong together.

Ground rules learned the hard way in this codebase:
- Never enforce a model invariant with a prompt. URLs, verbatim reading and
  output format all failed that way; mask/route/validate in code instead and
  test the mechanism, not the wording.
- Browser-verify UI changes. NiceGUI 3 strips inline event handlers from
  ui.html() via DOMPurify, and unit tests miss it entirely.
- Local models share one GPU: benchmark them serially or you measure VRAM
  contention rather than the model.
- pytest IS correct here: .venv\Scripts\python.exe -m pytest tests/
- Re-run the benchmark before trusting model choices:
  python -c "from passage import model_bench" and see §1 for the harness.

Deliverables: incremental commits with reasoning in the messages, pytest green,
and RESEARCH.md updated with anything you measure.
```
