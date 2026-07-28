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

Measured, single controlled run: median alignment error **17.5 px → 9.2 px**.
Still partial — section headings in a lighter colour are missed by the
detector, so their nearest row is genuinely far away.

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

1. **Per-profile document routing.** Documents still run on the process
   default, so BYO doesn't cover the surface that costs the most. Needs the
   async job to carry the profile. *Do this alongside real metering, not
   before it.*
2. **Real metering on the policy hook.** `is_metered()` exists and nothing
   reads it yet. Charge for durable storage and traces, not inference.
3. **Better row detection for coloured headings.** The remaining overlay error
   is concentrated in section headings the detector misses. A per-channel ink
   map (not just greyscale) would catch burgundy-on-cream.
4. **Local STT/TTS.** Voice is the only modality with no local path, so the
   "nothing leaves your machine" claim has a hole in it. Whisper.cpp or a
   Piper voice would close it.
5. **Trace persistence for documents.** Policy already permits it; nothing
   writes it. This is the substrate for per-segment scoring and the per-user
   preference dataset.
6. **Re-benchmark on a second machine.** Every latency number here is one
   GPU's opinion. The ranking is more portable than the absolute numbers, but
   neither has been checked elsewhere.

---

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
