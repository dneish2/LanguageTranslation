# Running everywhere: the problem, where an agent will fail at it, and the shape of the work

Target, in David's priority order:

1. **Windows + GPU** — the development machine. Already works.
2. **Apple Silicon Mac** — no CUDA, unified memory, different fonts.
3. **Windows, no GPU** — the floor. Everything must still function.

Plus: work in any reasonable browser.

This document is deliberately *not* an implementation plan yet. It is the framing, the failure
analysis, and the argument for a particular working method — because the naive version of this
task has a specific way of going wrong that is worth naming before any code is written.

---

## 1. The reframe that matters

**This is not a porting problem. It is a capability-detection and honest-degradation problem.**

The code must never ask *"am I on a Mac?"*. It should ask:

- is a local model reachable, and which ones are installed?
- did a real font load, or did PIL fall back to a bitmap face?
- does this browser expose `getUserMedia`, and is this a secure context?
- did the speech stack import?

Every one of those is answerable **at runtime, on the user's machine** — which is the only place
the truth exists. Platform sniffing encodes a guess; capability probing reads a fact.

The existing code already leans this way (`choose_local_model()` picks the best *installed* model;
the live path probes reachability; voice preflights `getUserMedia`). The work is to finish that
job and, crucially, to make the results **visible**.

## 2. Where an AI agent will fail at this — the honest list

These are not hypothetical. Several already happened in this codebase.

### 2.1 Claiming verification it cannot perform
I am on Windows with a 5090. I *cannot* run this on an M1. Any statement about Apple Silicon
behaviour is inference from wheel metadata and code reading. `PORTABILITY.md` says so explicitly,
but the pull toward writing "✅ works on M1" is strong and must be resisted structurally, not by
good intentions.

**Already happened:** the overlay fix was reported as 17.5 px → 9.2 px from a single run; five runs
gave 23.5 → 20.5. A number that felt solid was not.

### 2.2 Silent fallback makes brokenness indistinguishable from absence — *the biggest trap*
The architecture falls back everywhere: local → hosted, real font → bitmap font, local voice →
hosted voice. That is right for users and **actively dangerous for testing**.

On a machine with no Ollama, every local path silently goes hosted and the app *looks perfect*. An
agent testing there would report "works on Windows without GPU!" having exercised **none** of the
local code. The success is real; the conclusion is worthless.

> Any cross-platform test must assert **which path executed**, never merely that the call
> succeeded. `translate_live()` already returns an engine label and `translate_audio()` returns
> `meta["stt"]/["tts"]` — those exist precisely so a test can tell. Use them.

### 2.3 Constants tuned to the one visible machine
Several magic numbers were fitted on this hardware:

| constant | value | risk elsewhere |
|---|---|---|
| `LIVE_PROBE_TIMEOUT_SECONDS` | 0.6 s | a slower or busier machine times out → local silently "unavailable" |
| `THINKING_RESERVE_TOKENS` | 2048 | unrelated to hardware, but fitted to one model family |
| row-detector thresholds | several | fitted to one synthetic image |
| `FREE_CHARS` | 50,000 | arbitrary, unenforced |

The 0.6 s probe is the dangerous one: on a loaded Mac it could produce a **false negative that
looks exactly like "no local model installed"** — and thanks to §2.2, nothing would report a
problem.

### 2.4 Browser claims asserted from reasoning
The 24 kHz `AudioContext` fix is justified in a comment with "Safari/iOS returns the hardware
rate". That is almost certainly true and **was never observed** — it was inferred from an error
message produced on Chrome. Cross-browser work invites a lot more of this.

### 2.5 Testing only the happy path
Each fallback needs to be *forced* deliberately: Ollama stopped, voice models absent, fonts
missing, `getUserMedia` denied, hosted key absent. Fallbacks that have never executed are
decorative.

### 2.6 Environment detection written from assumptions
The classic version: checking for a GPU with `torch.cuda.is_available()` when torch isn't a
dependency, or shelling out to `nvidia-smi` on a Mac. Any capability check must be written against
what the app *actually uses* — Ollama's `/api/tags`, an import, a font resolution — not a generic
notion of "has GPU".

## 3. The insight that makes most of this verifiable

**Two of the three target tiers are free in CI, and the tier that needs David is the one that
already works.**

| tier | where it can be verified |
|---|---|
| Windows + GPU | David's machine only — and it is the current dev environment |
| **Apple Silicon** | **GitHub `macos-14`+ runners are arm64** |
| **Windows, no GPU** | **GitHub `windows-latest` — no GPU at all** |

Current CI is `ubuntu-latest` only, so *none* of this is exercised today.

What CI can genuinely verify per tier:

- **Cheap (every PR):** the suite on a matrix of ubuntu / windows / macos-14; imports resolve;
  pure-Python geometry, policy, usage, traces behave identically; a capability probe prints what
  the machine resolved. This alone would have caught the font problem.
- **Medium (nightly or manual):** `faster-whisper` + `piper` genuinely installed and run on the
  macOS runner — **real Apple Silicon local speech**, not inference about it. Both are pip-
  installable with arm64 wheels, so this is a handful of lines.
- **Expensive (manual):** Ollama installed and a small model pulled (~3.3 GB) to exercise local
  LLM inference on arm64. Worth doing once, not per PR.

The honest limit: CI runners have no Ollama and no GPU, so **local LLM inference on Apple Silicon
stays unverified** until either the expensive job runs or someone runs the harness on a real Mac.
`passage/model_bench.py` already appends to `data/model_bench.jsonl`, so such a run is directly
comparable to the 5090 numbers.

## 4. Browsers: what is testable and what is not

Playwright drives **Chromium, Firefox and WebKit** on all three OSes. WebKit *is* Safari's engine,
so running it on the macOS runner is the closest thing to real Safari available without a device —
and it is exactly where the 24 kHz `AudioContext` assumption could finally be checked.

Not testable that way, and must be labelled as such:

- **real iOS Safari** (differs from desktop WebKit, especially audio and autoplay)
- **real Android Chrome**
- **OS-level permission dialogs** (Playwright grants permissions programmatically)
- **actual microphone and camera hardware**

For those, the answer is not "test harder" — it is the diagnostic in §5, so a real device reports
its own truth in one visit.

## 5. The approach: make the system self-describing

The agent's job is **not to predict** what happens on other hardware. It is to make the system say
what happened, so the machine answers the question.

**Phase A — remove assumptions (fully verifiable here).**
Audit every place the code assumes something about its host; convert each to a runtime capability
probe with an explicit, *recorded* fallback. Verifiable on Windows because it is about removing
assumptions, not confirming remote behaviour.

**Phase B — a diagnostic surface (fully verifiable here).**
Extend `/api/health` (and a small `/diagnostics` page) to report what *this* machine resolved:
OS/arch/Python, Ollama reachability and installed models, chosen local model, speech stack
availability and voice list, resolved overlay font (**and whether it fell back**), plus
browser-side: secure context, `getUserMedia`, `AudioContext` sample rate actually granted, `MediaRecorder`
MIME support. Turns "does it work on an M1?" into a question the M1 answers in one page visit.

**Phase C — CI matrix (verifiable, and the real payoff).**
ubuntu + windows + macos-14, with tests asserting *which engine served each request*, not just
success. Add the medium-tier macOS speech job.

**Phase D — forced-degradation tests (fully verifiable here).**
Deliberately break each capability and assert the app degrades honestly and *says so*: no Ollama,
no voice models, no font, no hosted key, non-secure context.

**Phase E — real-hardware confirmation (needs David or a device).**
Open `/diagnostics` on the M1 and on a phone; paste the output back. Minutes of work, and it is
the only step that produces truth about those environments.

## 6. Why a long autonomous loop is the wrong tool here

Previous phases of this project used long `/loop` runs well, because the bottleneck was *volume of
verifiable work* — drive the app, measure, fix, repeat, all on one machine.

Here the bottleneck is **verification on hardware the agent cannot reach**. A long autonomous loop
would keep producing confident, unverifiable cross-platform claims — precisely the failure in §2.1,
at scale and with no natural stopping point.

**The right shape is bounded phases with a verification gate between them:** A→B→D are genuinely
verifiable locally and can run as one focused session; C is verifiable the moment CI runs; E is a
short human step that unblocks the rest. Where a loop *does* fit is inside Phase C — iterating
until the matrix is green is a real feedback signal, and CI provides the ground truth an agent
otherwise lacks.

## 7. What "done" means

- No code branches on operating system; every environment question is a capability probe.
- Every fallback is exercised by a test that asserts it fired.
- `/diagnostics` reports the full resolved picture, and a real M1 and a real phone have each been
  opened once and their output recorded here.
- CI is green on ubuntu + windows + macos-14, with engine-path assertions.
- `RESEARCH.md` carries per-platform benchmark rows, or explicitly states which are unmeasured.
- Anything unverified is labelled unverified — including in this document.
