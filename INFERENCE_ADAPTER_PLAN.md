# Passage Deployment and Streaming Plan

**Revised:** 2026-08-26
**Status:** architecture decision and implementation plan. No new runtime,
distribution package, or streaming implementation has been added yet.

## Product Decision

Passage has two deployment modes, not one hosted application trying to borrow a
visitor's GPU.

| Mode | Where Passage runs | Where translation runs | User mental model |
|---|---|---|---|
| **Passage Cloud** | Hosted web service | Managed API models | "I use the website; Passage runs the service." |
| **Passage Local** | The user's Mac, Windows, or Linux machine | A runtime on that same machine, or a user-selected remote endpoint | "I install/run Passage; it uses my machine when I select Local." |

The hosted service must never try to call a visitor's localhost, GPU, or browser
runtime. That was an architecture mismatch, not a missing adapter. WebGPU/WASM,
browser-to-loopback bridges, and a remote site executing a user's local model are
out of scope for this plan.

The shared concept is simple:

    Deployment says where Passage runs.
    Runtime says which engine translated this request.
    Receipt says what actually happened.

The existing CallProvenance and /engines ledger remain authoritative for privacy
and metering. A runtime receipt adds model, runtime, stream state, latency, and
fallback facts that the ledger does not currently have.

## Current State

| Area | Verified state | Gap to close |
|---|---|---|
| Cloud text | ChatCompletionsProvider supports hosted and OpenAI-compatible BYO endpoints. | The current default model and API shape need a current capability review. |
| Local text | NativeOllamaProvider uses native Ollama /api/chat; non-streaming live text can prefer it. | Extract it into the shared runtime contract and use it consistently across surfaces. |
| Streaming | stream_translate_text obtains the full translation first, then slices it into artificial partial strings. | Implement real upstream token/delta streaming and cancellation. |
| Voice | Local STT/TTS are available; the translation step remains a separate text call. | Make cloud voice mode deliberately choose between a composed pipeline and direct speech translation. |
| Distribution | README documents a local Python/uv launch. There is no packaged desktop binary, installer, or first-run runtime selector. | Ship a reliable local launch path before deciding whether a native bundle is justified. |
| Health | /api/health, /diagnostics, and /engines already have good redaction, freshness, and provenance rules. | Extend them with deployment/runtime/stream facts; do not create a second health system. |

The focused portability tests pass: 77 passed. The isolated full suite is not
green: 562 passed, 27 failed. The immediate common cause is that tiktoken does
not recognize the configured gpt-5.4-nano, then attempts a blocked download of
o200k_base. That is a separate M0.1 repair before a runtime change can claim a
clean full suite.

## Target Architecture

Both modes run the same Passage application and use the same in-process router.
There is no device-side adapter category.

    +---------------------+
    | Passage UI and API  |
    +----------+----------+
               |
    +----------v----------+
    | RuntimeRouter        |
    | selection + policy   |
    +-----+-----------+----+
          |           |
    +-----v-----+ +---v----------------+
    | Managed   | | Local              |
    | OpenAI API| | Ollama or MLX      |
    | Cloud     | | Local only         |
    +-----------+ +--------------------+

The router is selected once per request and passed through every chunk, retry,
and surface. A document, stream, voice translation stage, and image-text
translation must not independently rediscover a different runtime.

### User-Facing Selection

The first UI has three choices only:

1. **Cloud**: Passage's managed API runtime.
2. **Local (recommended)**: Passage detects healthy local runtimes and selects
   a compatible installed model.
3. **Choose engine**: the user selects one discovered local model or enters a
   compatible remote endpoint.

Local (recommended) never downloads a multi-gigabyte model without explicit
confirmation. It may select an already installed model automatically. The UI
shows the runtime and model before the first request, then the actual receipt
after it runs.

## One Streaming Contract

Streaming is a runtime behavior, not a UI effect. Replace final-answer slicing
with one event shape that every text runtime emits:

    TranslationRuntime.translate(request) -> Iterator[TranslationEvent]

    Event kinds: started, delta, completed, failed
    completed contains the canonical final text and RuntimeReceipt

Non-streaming callers consume the iterator to completed. Streaming callers
forward delta events to SSE. There is no second fake-streaming method.

Each RuntimeReceipt includes runtime ID, deployment mode, model ID, stream
capability, first-delta latency, end-to-end latency, cache status, failure or
fallback reason, and best-effort local memory data. Unknown memory is null, not
zero. Source text is never retained in health metrics.

| Capability | Behavior |
|---|---|
| token_deltas | Real incremental output; UI renders it as it arrives. |
| final_only | UI waits for the canonical answer and does not pretend it is streaming. |
| unsupported | Router selects another compatible runtime or returns a clear unavailable state. |

Cancellation is part of the contract. When a user types again, navigates away,
or presses stop, Passage closes the upstream managed stream or local generation.
Continuing a superseded request wastes API spend or local compute.

## Current OpenAI API Review

This is a capability review, not a decision to hardcode a model name. Actual
availability and rate limits remain account-specific and need a startup
capability probe.

| Workload | Candidate | Why it fits | Architecture decision |
|---|---|---|---|
| Managed text translation | gpt-5.6-luna | Current OpenAI documentation identifies it as the cost-sensitive, high-volume GPT-5.6 variant; it supports text/image input, text output, Responses, Chat Completions, and streaming. | Evaluate as the cloud text default with reasoning disabled or minimal; preserve a configured override and benchmark translation quality. |
| Higher-quality managed text | gpt-5.6-terra | Current documented balanced GPT-5.6 variant with streaming and a much larger cost envelope than Luna. | Offer as an explicit quality profile after corpus evaluation, not as the keystroke default. |
| Live transcription | gpt-live-transcribe | Streaming speech-to-text with live transcript deltas, tunable latency, context, keywords, and language hints. | Use when the product needs a reviewable transcript before translation. |
| Live interpreting | gpt-realtime-translate | Dedicated streaming speech-to-speech translation endpoint; returns translated audio and transcript deltas while audio arrives. | Evaluate as a distinct opt-in Interpreter mode, not as a replacement for the text/receipt pipeline. |
| Speech output | Current gpt-4o-mini-tts path | Existing dedicated speech endpoint avoids the known failure where a conversational audio model answered rather than read the translated text. | Retain until a controlled voice-quality and exact-read regression test justifies a change. |

OpenAI's current catalog lists GPT-5.6 Luna as the high-volume choice and
GPT-5.6 Terra as the balanced choice, with current multimodal text/image support
and the Responses API. [Model catalog](https://developers.openai.com/api/docs/models)

The Responses API accepts stream: true and emits streaming events, so it is a
better managed-text implementation target than fabricating partial output after
a completed Chat Completions call. [Responses API reference](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)

For voice, OpenAI documents gpt-realtime-translate as a streaming
speech-to-speech translation model on a dedicated realtime translation endpoint.
It is billed by audio duration and has a different product shape from text
translation. [GPT-Realtime-Translate](https://developers.openai.com/api/docs/models/gpt-realtime-translate)

gpt-live-transcribe is the composable alternative: it supplies live transcript
deltas from incoming audio but not translated audio. [GPT-Live-Transcribe](https://developers.openai.com/api/docs/models/gpt-live-transcribe)

### Voice Product Rule

Keep these workflows separate in the interface and code:

| Mode | Pipeline | Why |
|---|---|---|
| **Translate and review** | STT -> text translation -> TTS | Gives source text, canonical translation, editability, history, a complete receipt, and local/cloud parity. This remains the default Passage voice workflow. |
| **Interpreter** | Direct realtime speech-to-speech translation | Optimizes for conversational latency. It has a different interaction model, measurement profile, and edit/review story. |

The current code comment says direct speech translation was visible but rejected
by this account's sessions in July. Re-test access before designing around it:
published documentation does not prove this project's account can use it.

## Local Distribution Plan

Passage Local starts as a good install-and-run experience, not a native binary
for every operating system on day one.

### Phase 1: Repository Distribution

The README becomes the local front door:

    clone -> one setup command -> passage doctor -> passage run

passage doctor reports, without changing anything:

- OS and architecture, available RAM, and best-effort GPU information;
- whether Ollama and an MLX-capable environment are present;
- installed compatible models, their reported sizes, and stream capability;
- recommended launch profile and why it was chosen;
- missing requirements and exact install instructions.

It does not download models, alter drivers, or make an unsupported hardware
claim. First-run setup may offer an explicit model install after showing its
size, license/provenance, and expected memory floor.

### Phase 2: Supported Local Runtimes

1. **Ollama first**: extract the existing native provider as OllamaRuntime. It
   already covers the user's Windows path and works as a local runtime on other
   supported platforms when installed there.
2. **MLX second**: an optional Mac-only MLXRuntime, behind isolated optional
   dependencies. Validate it on an actual Apple Silicon Mac before it appears
   as supported.
3. **No direct CUDA or ONNX runtime initially**: the user benefit does not yet
   justify another model format, tokenizer, device, and packaging stack.
   Revisit llama.cpp/GGUF only if Ollama cannot deliver required control or
   distribution.

### Phase 3: Packaging Decision

After repository setup and doctor work reliably on target platforms, measure
whether a bundled desktop app materially reduces setup failures. A future
distribution may use platform-specific launchers or installers, but it must use
the same doctor, runtime selection, receipt, and streaming contract. Packaging
is a delivery concern, not an inference architecture.

## Model and Health Policy

Maintain a small explicit local-model manifest, not an open-ended catalog. Each
supported model has a source/revision, license status, runtime/format, estimated
memory floor, tokenizer requirement, stream support, and evidence status.

Discovered models missing from the manifest can be shown as detected,
unverified and selected manually. Passage must not infer license, quantization,
or capability from a model tag, and it must not background-download weights
during health checks or request handling.

Extend the existing health surface additively:

    deployment_mode: cloud | local
    selected_runtime: openai-responses | ollama | mlx
    model: configured model ID
    streaming: token_deltas | final_only | unsupported
    last_request: first_delta_ms, total_ms, fallback_reason

In Local mode, the same shape identifies Ollama or MLX, the selected model, and
measured or unavailable memory fields. /diagnostics retains its deeper redacted
probe report; the workspace only needs a compact runtime/model/state indicator
that opens engine settings.

## Delivery Sequence

| Milestone | Scope | Exit evidence |
|---|---|---|
| M0.1: clean baseline | Make unknown-model token counting deterministic without requiring a network download. | Full suite passes in an isolated test directory. |
| M1: runtime core | Add TranslationRuntime, receipt/event types, cloud and existing Ollama adapters, plus router selection. Route live text only. | Tests prove the intended runtime actually received the request. |
| M2: real streaming | Replace artificial partials with Responses streaming for Cloud and verified incremental streaming for local runtimes. Add cancellation. | First-delta, final-answer, cancellation, and fallback tests pass. |
| M3: surface convergence | Route streaming text, documents, image-text translation, and voice's text stage through the router when capable. | No surface silently uses a different runtime than its receipt. |
| M4: model evaluation | Run a fixed multilingual/protected-span corpus against Luna, Terra, current default, and selected local models. | Five-run quality/latency report and explicit default decision. |
| M5: local bootstrap | Implement doctor, local auto/manual selection, manifest, and README workflow. | Clean local setup on Windows and real Mac evidence report. |
| M6: MLX evidence | Run optional MLX runtime on real Apple Silicon. | Pass/fail benchmark and health receipt; supported only on a pass. |
| M7: responsive UX | Compact health indicator and explicit responsive CSS; validate desktop, tablet, and phone browser flows. | Playwright screenshots and interaction tests at all three sizes. |

## Definition of Done

Passage Cloud streams real managed-model text output and names the actual model
and route. Passage Local launches from the documented workflow, detects
already-installed compatible runtimes, makes a clear automatic recommendation,
and lets the user choose a model. Both modes use the same request events,
cancellation behavior, final answer, and runtime receipt.

MLX is not required to declare local distribution complete; it becomes supported
only after real Mac evidence. Direct visitor-GPU access from the hosted site,
WebGPU/WASM, arbitrary daemon bridging, and native desktop bundles are out of
scope until the two-mode product works cleanly.

## First Implementation Ticket

Start with M0.1 and M1, not MLX packaging:

1. Fix the tokenizer fallback so tests do not rely on downloading an encoding.
2. Add event/receipt data types and a minimal router.
3. Wrap the existing managed text call and native Ollama text call without
   changing prompts, cache ownership, or policy classification.
4. Return an additive runtime receipt from /api/text_translate.
5. Add a health entry for deployment mode, runtime, model, and stream capability.
6. Add path-identity tests using fake managed and local endpoints.

The next ticket is M2: replace synthetic partials with real delta streaming. No
model changes should ship merely because they are current; the M4 corpus
determines the managed and local defaults.

