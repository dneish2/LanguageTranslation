# Passage — plan of record

**This file is the single source of truth for the LanguageTranslation → Passage rebuild.**
Any session (human or agent) resuming this work starts here: read "Current state", pick up the
first unchecked item, and update this file as work lands. Keep it committed to git.

Last updated: 2026-07-06.

## What Passage is

`LanguageTranslation` (NiceGUI app, Cloud Run service `translation-app`, GCP project
`translation-app-452812`) rebuilt as **Passage**: a reliable, well-designed translation
workspace with accounts, model optionality (local 5090 / BYO endpoint / cheap hosted), and a
flagship feature — layout-preserving PDF translation with segment-level traces of every
machine translation and human edit.

## Locked decisions (2026-07-05, with David)

- **Name: Passage.** Keep NiceGUI and polish it — no React rewrite now. Keep `/api/*` clean so a swap stays possible.
- **Auth/DB: share the finplatform Supabase project** — one login, one future bundled subscription.
  Port finplatform's `jwt_verify` + credits patterns. NOTE: finplatform tokens are now
  **ES256/JWKS** (HS256 is legacy) — port that version.
- **Design: editorial / paper-and-ink identity**, deliberately distinct from finplatform.
  David picks from 2–3 static HTML mockups before any real UI is built.
- **Models: provider-agnostic** via OpenAI-compatible `base_url` + `key` + `model` profiles.
  Local default = TranslateGemma via Ollama on David's 5090 — he wants to try **4b and smaller,
  not just 12b**; 27b only as a quality toggle. TranslateGemma has ~2K-token input context →
  segment chunking must cap segment size. Hosted tier upgraded 2026-07-06 (David):
  `gpt-5.4-nano` text, `gpt-5.4-mini` vision OCR, `gpt-4o-mini-transcribe` STT,
  `gpt-4o-mini-tts` TTS — all env-overridable (`PASSAGE_TEXT_MODEL` etc.).
- **Flagship: PDF-first** — a finplatform print-to-PDF report dropped into Passage → layout-preserved
  translated PDF via pdf2zh/BabelDOC as an async job. Live HTML/article translation deferred, seams kept ready.
- **Traces: Langfuse-shaped JSONL/tables** (trace = document run, generation = segment,
  score/event = human edit). No self-hosted Langfuse.

## Repo facts a fresh session needs

- Entry: `TranslationUI.py` → `ui.run(host="0.0.0.0", port=PORT)`. Backend: `TranslationBackend.py`
  with a provider seam (`build_translation_provider`) and a real segment model
  (`segment_map`, `_translate_segment_text`, `update_segment`).
- **pytest IS correct in this repo** (finplatform's no-pytest rule does not apply):
  `.venv\Scripts\python.exe -m pytest tests/` — uv venv, Python 3.12.
- Dead code CLEANED 2026-07-05: `job_queue.py` deleted (the backend's own `TranslationJob` system
  is the live pattern; Phase 5's pdf2zh job gets designed deliberately). `translation_metrics.py`
  is LIVE (backend imports `MetricsCollector`/`TranslationMetrics`) — the old "dead" note was wrong.
  Legacy PPTX `handle_upload` flow deleted.
- CI: `.github/workflows/deploy.yml` — tests gate the deploy; the deploy step reads
  `OPENAI_API_KEY` from a GitHub Actions secret.

## Current state

- **Phase 0 shipped** (commit `47d2851`, branch `fix/boot-crash-and-ci-gate`, PR #34):
  boot crash fixed (malformed `add_api_route` from the PR #33 merge, split into two calls;
  undefined `text_status_scope` defined), PORT env honored, backend boots without
  `OPENAI_API_KEY` (`_require_provider` guard), deploy.yml gates on pytest (44 tests).
- **Honest-error contract shipped** (this branch): `translate_text` now **raises** on provider
  failure instead of echoing the source text back disguised as a translation. All callers
  verified: UI call sites catch and surface via `show_error`/`ui.notify`; document jobs land in
  a failed state. Tests updated (`test_translate_text_raises_after_max_attempts`,
  `test_translate_raises_on_openai_exception`). Plus two mobile fixes: NiceGUI
  `on_value_change` for the mode selector, progress widgets created inside their container.
- **Suite green**: 61/61 passing locally.
- **Model roster upgraded + 2 crashes fixed (2026-07-06, on `design/press-tokens`)**:
  text `gpt-5.4-nano` + vision `gpt-5.4-mini` (both with `max_completion_tokens` +
  `reasoning_effort="none"` — GPT-5-family rejects `max_tokens`; "none" is the 5.4 spelling
  of "minimal"); OCR call also gained `response_format=json_object`. Voice moved to the current
  families per David: STT `gpt-realtime-whisper` over a transcription-intent websocket
  (`intent=transcription`, `turn_detection: None` — the model has no server VAD; PCM16 only,
  so the /voice recorder now captures **24 kHz mono PCM16 WAV via Web Audio** instead of
  MediaRecorder webm), TTS `gpt-audio-mini` via the chat-completions audio modality
  (voice `nova`; TTS-engine system prompt). Non-PCM payloads fall back to REST
  `gpt-4o-mini-transcribe` with magic-byte filename sniffing. All env-overridable
  (`PASSAGE_TEXT_MODEL`/`_VISION_`/`_TRANSCRIBE_`/`_TRANSCRIBE_REST_`/`_TTS_MODEL`/`_TTS_VOICE`).
  **`gpt-realtime-translate` (speech→translated speech, $0.034/min) is listed in /v1/models but
  unusable on this account as of 2026-07-06**: realtime sessions fail with
  `inference_not_found_error`, transcription sessions reject the model id — re-probe later; the
  provider seam is ready for it. Live-verified: text/vision/STT/TTS + full voice loop (~7s clip
  round-trip) + REST fallback + full PDF pipeline + booted `/api/text_translate`.
  Crashes fixed: NiceGUI 3.x upload events (`event.file` + async read, was
  `event.name`/`event.content`) broke ALL document/image uploads; live-translation JS
  re-injection on workspace re-render died with "parent slot deleted" (swap ⇄) — now injected
  once per page, bindings delegated on `document`.
- **Browser-verified pass (2026-07-06, Playwright vs the running app, 6/6 features, 0 console
  errors)** — found and fixed three more latent bugs:
  1. `.props("id=…")` on NiceGUI inputs/textareas never reaches the native control in NiceGUI 3
     (buttons/labels are fine) → the Text-mode live-translate JS had NO source/target elements
     and was silently dead; Quasar's **`for=`** prop is the correct way (also fixed the /voice
     transcript textarea).
  2. The /voice head script had `buffer.split('\n')` inside a non-raw Python triple-quote → a
     literal newline inside a JS string literal → SyntaxError that killed the ENTIRE script
     block (recorder, transcript fallback, streaming SSE parser). Now a raw string.
  3. `show_result` touched the already-deleted Cancel button (cleared with progress_container).
  Also: default workspace mode is now **Text** (was Document; Text is the first tab).
  UX debt observed while testing, not yet fixed: (a) the **global-state collision is vivid** —
  a second browser session inherits the first session's mode/languages/uploaded filename;
  Phase 4's per-user workspace is the real fix.
- **Phase 2 quality pass (2026-07-06)**, closing debt items (b)–(d) from the browser-verified
  pass above, all zero-API-cost (pytest + one no-translate Playwright pass, since this landed
  mid-loop after OpenAI credits ran low):
  - **(b) same-language guard**: From==To (case-insensitive) now blocks with "swap ⇄ or pick a
    different target" instead of silently running an identity translation — enforced both
    server-side (`start_mobile_translation`) and in the Text-mode live-translate JS.
  - **(c) Debug line hidden by default**: `/voice`'s developer readout only shows with
    `?debug=1` (`voiceUx.init` reveals `.p-debug-block` on load); `updateDebug` still writes to
    it so the flag is loss-free for real debugging.
  - **(d) Recent Threads dedupe**: `_record_thread` (generalizes `_record_chat_thread`) drops any
    existing entry with the same (kind, label, language) before prepending, so repeating a
    translation moves its thread to the top instead of stacking duplicates.
  - **Unified error surface**: `show_error` now classifies via `_is_technical_error` (long or
    provider-shaped messages) — short user-facing messages show directly in the banner; raw
    OpenAI/stack detail collapses into a "Technical detail" expansion instead of dumping on the
    page. Pure-function classifier is unit tested without touching NiceGUI rendering.
  - **Segment icon buttons**: approve/reject/delete switched from emoji glyphs to `ui.icon` +
    tooltips on the theme's OK/secondary/danger classes (still burgundy-only per David's taste
    call below, but no longer cramped).
  - **Language input binding**: From/To now `bind_value` to `current_source_language`/
    `current_target_language` instead of a one-shot `value=` — edits used to reset on every
    workspace re-render (mode-tab clicks); a stray `current_target_language = "Processed"`
    write (regenerate-without-retranslate path) was removed since it would have leaked into the
    bound To field.
  - **Reverted**: a CSS specificity fix (`.p-btn.p-btn-secondary` double-class selectors) that
    would have made secondary/danger/ok buttons render outlined/tinted instead of solid burgundy
    (Quasar's own `bg-primary` was winning the cascade on every `ui.button`). David saw the
    screenshots first and said he likes the current uniform burgundy — reverted rather than ship
    an unrequested visual change. The underlying specificity bug is still real if this is
    revisited later; don't re-fix without asking.
  - 71/71 pytest (7 new: same-language guard both modes, error classifier, thread dedupe).
- **Interim API hardening shipped** (branch `security/api-hardening`): the four public `/api/*`
  endpoints were fully open — anyone with the URL could burn the OpenAI key. Now gated by a
  short-lived signed session token (`api_security.py`) that only app-served pages embed
  (`window.PASSAGE_TOKEN`, sent as `X-Passage-Token`), plus a per-IP rate limit (30/min default,
  `PASSAGE_API_RATE_LIMIT`), text length cap (8,000 chars, `PASSAGE_MAX_TEXT_CHARS`) and upload
  cap (8 MB, `PASSAGE_MAX_UPLOAD_BYTES`). `PASSAGE_PUBLIC_API=1` disables the gate for local dev.
  This is abuse prevention, NOT auth — real accounts are Phase 4 and replace/absorb this gate.
  Prompt-injection hardening: the translation prompt now pins the model as a translation engine,
  wraps user text in BEGIN/END markers, and instructs it to translate embedded instructions
  literally rather than follow them. Empty model output and refine failures now raise instead of
  silently echoing the source (completes the honest-error contract).
- **Design picked (2026-07-05): Direction A "Press" palette** (warm paper #F6F0E3, letterpress
  ink #2B241C, burgundy #802F3D, Palatino display / Georgia body) **plus Direction C's monospace
  data accents used sparingly** — for metadata, counts, traces, states; NOT for buttons.
  Motion principle from David: subtle animations and loading loops **synced to real data
  movement** (progress reflects actual job/segment state, never decorative spinners) — keeps the
  user in the loop and doubles as debugging/state visibility. Mockup artifact: "Passage — design
  directions" (Press / Gallery / Ledger).
- **Full aesthetics + feature QA pass (2026-07-06)**, per David's ask for a high bar before
  reporting back: live-drove Text/Document/Image/Voice with real translate/OCR/transcript
  round-trips, desktop (1440px) + narrow (390px) viewport, screenshotted every result state
  (not just empty forms). Fixed the raw-white-card debt noted above. Two suspected new bugs
  from screenshots (a washed-out "Translate transcript" button, a mismatched header-tab
  background) both turned out to be **mouse-hover/ripple rendering artifacts from the
  automation, not real bugs** — confirmed via `getComputedStyle` before touching any code
  (background-color and opacity were identical to the correct state in both cases). Lesson: a
  screenshot difference isn't a bug report until the computed style is checked — don't
  "fix" a hover ghost. 5/5 live checks passed, 0 console errors, 95/95 pytest unaffected.
  Reconfirmed the known global-state-collision debt is still live and vivid (a fresh mobile
  session inherits the previous desktop session's mode) — still correctly scoped as a Phase 4
  fix, not something to patch around here.

### Blocked on David (do these once, in the browser)

- [x] Add repo secret: GitHub → Settings → Secrets and variables → Actions → `OPENAI_API_KEY`
      — done, Phase 0 shipped and deployed.
- [x] Merge PR #34 (`fix/boot-crash-and-ci-gate`) — done, Phase 0 shipped and deployed.
- [ ] **Current ask**: share the real Supabase `SUPABASE_URL` + anon key (both designed to be
      public — safe to hand over directly, unlike the service-role key) so Phase 4's auth
      verification (already ported and live-tested against a fake project, see Phase 4 item 1)
      can be pointed at the real one, and so the sign-in UI (item 2) becomes buildable.

## Roadmap

### Phase 0 — Unbreak prod — DONE (pending PR merge + secret above)

### Phase 1 — Brand + design prototypes

1. [x] 2–3 static HTML mockups (Artifact) — done: Press / Gallery / Ledger, interactive comp.
2. [x] David picked: **Press palette + Ledger's mono data accents (sparingly, not buttons);
       motion synced to real data transfer.** Tokens shipped: `static/passage.css` (CSS
       variables + p-btn/p-banner/p-panel/p-well/p-data classes) + `theme.py` (constants).
       ALL ad-hoc Tailwind color strings replaced (blue/purple/indigo/green/red/gray buttons →
       4 semantic button styles; bg-white/gray-50/blue-50 surfaces → paper/panel/well).
3. [x] Brand assets: `static/favicon.svg` (burgundy P on paper), `ui.run(title="Passage",
       favicon=...)`, Passage wordmark in both headers. Still open: OG meta tags;
       `Multilingual.png` is an unused stray file in the repo root (not referenced by any
       code path) — delete during the Phase 2 restructure cleanup pass.

### Phase 2 — UI/UX rehaul (NiceGUI, on the chosen tokens)

Progress (on the `design/press-tokens` branch, PR #37 — David wants NO merge until it
looks properly good; keep committing to this branch):
- [x] Quasar brand colors overridden via `ui.colors` (killed the leftover default-blue toggles/props).
- [x] Header rebuilt: clickable Passage wordmark → `/`, separator, **Text | Document | Image |
      Voice mode tabs** (the comp's interaction model). Default/Advanced toggle REMOVED —
      segment review now always renders when a document has segments.
- [x] Drawer → **Recent Threads**: chats (text translations, recorded in-memory on API success)
      + translated documents, newest first; chat threads reload into the Text workspace.
      Becomes per-user/Supabase-backed in Phase 4.

1. [~] Restructure `TranslationUI.py` into `passage/ui/` — **started 2026-07-06**: the `/voice`
       page (recorder UI, its ~450-line head-injected JS, `api_voice_translate`) moved into
       `passage/ui/voice_page.py` as a `VoicePageMixin` (TranslationUI still inherits it, so
       shared state — `self.backend`, `_check_api_access`, `_inject_theme` — needs no redesign);
       `LANGUAGES`/`_log_event` moved to `passage/ui/common.py` to avoid a circular import back
       to TranslationUI.py. `TranslationUI.py` 1672→1225 lines. Verified: 71/71 pytest (updated
       3 tests that grep JS source text to point at the new file), live browser 7/7 feature pass
       incl. the moved voice route, live voice HTTP round-trip. Still monolithic and NOT split
       further: the workspace/segment-editor methods share heavy `self.*` state and a mixin-only
       split there is lower value than the remaining phases — deferred, not abandoned; revisit
       if the file grows past this point again.
2. [x] One design system: buttons/banners from theme constants; zero raw Tailwind color classes
       remain anywhere in `TranslationUI.py` (verified 2026-07-06). Segment approve/reject/delete
       use `ui.icon`; most other buttons are short text labels by design, not icon candidates.
3. [x] Unified error + loading model — **done 2026-07-06**: `_render_progress_ui(message,
       show_cancel=)` is the one shared loading surface (Text/Image/Document all use it instead
       of ad-hoc `circular_progress` blocks); `self.translate_button` disables the instant
       Translate is clicked and re-enables from every terminal path (`show_error`, `show_result`,
       `show_mobile_image_result`, `show_mobile_voice_result`); `show_error(error, retry=...)`
       renders a "Try again" button that re-invokes the failed action (wired into the Text/Image
       translate-thread failures and Document job-poll failures) and now also clears
       `progress_container` — a real bug caught live: the spinner+label used to stay stuck on
       screen behind the error banner because `show_error` never cleared that container. Image
       mode gained real progress feedback for the first time (it used to block silently with a
       frozen UI during OCR+translate). Verified: 76/76 pytest (5 new), live browser — button
       disables/re-enables, image OCR failure shows the retry button, spinner clears on error.
       Known debt spotted while verifying, not fixed here (pre-existing, out of this task's
       scope): `show_mobile_voice_result`/`show_mobile_image_result` render inside a raw
       `ui.card()` with no Passage classes — floats as a stock white Quasar card instead of
       paper/panel; fold into the theme sweep next time either is touched.
       **Resolved 2026-07-06** in the follow-up aesthetics pass below.
4. [x] Merge the two segment editors into ONE segment review surface — done (commit `f36ea9e`,
       "Merge the segment editors"); no separate Document Editor dialog remains.
5. [x] Responsive layout instead of the `/mobile` UA-redirect; retire `/mobile` — done (commit
       `1adb288`); `/mobile` is now a one-line redirect to `/`.
6. [x] Restyle `/voice` onto the same system — done (commit `9bfc5f9`); voice buttons use
       `p-btn`/theme classes, zero raw colors.
7. [x] Remove dead UI (legacy PPTX `handle_upload`) and dead modules — done: `job_queue.py`
       deleted (commit `26c4868`), legacy PPTX upload flow removed, `translation_metrics.py`
       confirmed live (imported by the backend, not dead).

### Phase 3 — Model optionality

1. [x] Generalize provider to `ChatCompletionsProvider` — **done 2026-07-06**:
       `base_url`+`api_key`+`text_model`, the `openai` SDK's native custom-`base_url` support.
       `OpenAITranslationProvider` kept as an alias (not a rename-in-place) for anything that
       still greps for the old name. `_require_openai_hosted()` makes voice (transcribe/
       synthesize) raise a clear `NotImplementedError` on a non-OpenAI profile instead of
       failing deep inside an SDK call for a capability the target server never had.
2. [ ] Provider profiles + settings UI: **not built** — real scope (per-user Supabase-backed
       profiles, a "Test connection" flow, a picker) that the plan itself sequences alongside
       Phase 4's accounts work. Today, switching providers is an env var
       (`TRANSLATION_PROVIDER=ollama`) — real and tested, just not exposed in the UI yet.
       **Considered building a "local-only, no Supabase needed" slice of this 2026-07-06 and
       deliberately didn't**: `TranslationBackend` is one instance shared across every connected
       client (the same global-state collision flagged elsewhere in this doc), so a provider
       *picker* right now would be a process-wide toggle affecting every open session, not a
       per-user setting — that's new debt in the same shape as the debt already being tracked
       for Phase 4 to fix, not a step toward the real design. Wait for Phase 4's per-session/
       per-user scoping; revisit then.
3. [~] Local default: TranslateGemma via Ollama — **provider layer done and live-verified**
       2026-07-06 against a real local Ollama instance (`http://localhost:11434/v1`, verified
       reachable on this machine): `TRANSLATION_PROVIDER=ollama` boots the backend with **no
       `OPENAI_API_KEY` required** (the old gate incorrectly demanded one for every provider —
       fixed), targets `PASSAGE_OLLAMA_MODEL` (default `gemma3:1b` — a real tag confirmed pulled
       here; NOT "TranslateGemma" specifically, which isn't pulled on this machine — override
       once David confirms his actual tag), and a real translation round-tripped correctly
       ("Good morning, where is the nearest pharmacy?" → "Buenos días, ¿sabe dónde hay una
       farmacia cerca?"). **Still open**: model-tag picker UI (see item 2).
       **Segment-size capping — done 2026-07-06**: `ChatCompletionsProvider.max_input_chars`
       (`None` = uncapped for OpenAI-hosted; `OLLAMA_MAX_INPUT_CHARS`/`PASSAGE_LOCAL_MAX_INPUT_CHARS`,
       default 3200 chars, for Ollama). `translate_text` splits oversized text on sentence
       boundaries (`_split_into_chunks` — hard-splits on whitespace only if a single sentence
       itself exceeds the cap), translates each chunk, joins with spaces, caches the joined
       result under the original key. Live-verified against the real local Ollama instance with
       a forced small cap: a 512-char, 7-sentence paragraph split into 4 chunks and translated
       coherently end to end (sentence order preserved, nothing dropped or duplicated); a
       parallel live check confirmed the OpenAI-hosted path takes the same long text in one
       call, no chunking log line, `max_input_chars is None`. 103/103 pytest (8 new: splitter
       edge cases, capped-vs-uncapped provider behavior, the hard-split path).
4. [x] Hosted tier: `gpt-5.4-nano` (done 2026-07-06), credit-gate via finplatform metering pattern later.
5. [x] Consolidate the model zoo — **done**: verified zero hardcoded model-name literals remain
       outside the roster constants block (`TEXT_MODEL`/`VISION_MODEL`/`TRANSCRIBE_MODEL`/
       `TRANSCRIBE_REST_MODEL`/`TTS_MODEL`/`OLLAMA_MODEL`), all env-overridable.

### Phase 4 — Accounts + workspace (shared finplatform Supabase)

1. [x] Port `finplatform/auth/jwt_verify.py` (**the ES256/JWKS version**) into `passage/auth/` —
       **done 2026-07-06**: ported verbatim (the verification logic is generic, not
       finplatform-specific); `claims_to_identity` simplified to `(user_id, email)` since
       Passage has no Role/credits concept yet. Added `PyJWT[crypto]` to requirements.txt (was
       missing — without it the ES256/JWKS path, the CURRENT Supabase signing scheme per
       finplatform's CLAUDE.md, would silently no-op to anonymous even with a valid token and a
       configured `SUPABASE_URL`; confirmed the install pulls only `cryptography`/`cffi`/
       `pycparser`, no conflicts). New `/api/me` route resolves identity from the
       `Authorization` header, ungated (costs nothing, unlike the OpenAI-backed endpoints).
       **Live-verified end-to-end, twice**: (a) a real ES256 token against a real JWKS endpoint
       (self-generated EC keypair + a throwaway local HTTP server serving the JWKS document —
       proves the asymmetric path isn't just falling through to the "no SUPABASE_URL" None
       case) — verified, tamper rejected too; (b) the real `/api/me` HTTP route with a real
       HS256 token, both signed-in and anonymous. 12 offline unit tests ported from
       finplatform's `test_auth_jwt.py` (throwaway secret, no network). Zero-config confirmed:
       with no `SUPABASE_URL`/`SUPABASE_JWT_SECRET` set, every request resolves anonymous.
       **Genuinely blocked on David** (not attempted, and correctly so): this is real code
       against a placeholder project — I don't have and should not fabricate finplatform's
       actual `SUPABASE_URL`/anon key. The project ref IS known (`wjukgtxlczldgjtcgljv`, from
       finplatform's `CLAUDE.md`) and `SUPABASE_URL`/the anon key are DESIGNED to be public
       (shipped in every Supabase browser bundle) — safe to hand over directly, unlike the
       service-role key which must stay in secret storage. Once David sets those two values,
       real Supabase-issued tokens should verify with no further code change.
2. [ ] Sign-in UI: magic-link + Google via GoTrue REST (mirror finplatform's SDK-less flow).
       Header account chip; signed-out users keep full local use. **Blocked on the anon key**
       above — the verification half (item 1) doesn't need it, but issuing a real magic-link
       request does.
3. [ ] Per-user workspace: `passage_documents`, `passage_segments` tables + `passage-files`
       Storage bucket, RLS by uid. "Recent Documents" reads user rows, not the server
       filesystem — also fixes the global-state collision (scope run state per session/user).
4. [ ] Metering: adapt the `_meter`/402 pattern; no-op unless keys set. Billing unification
       with finplatform is later — schema just must not preclude it.

### Phase 5 — Format-preserving PDF translation + traces (the flagship)

1. [ ] Integrate **pdf2zh / BabelDOC** as an async job (wire in or replace `job_queue.py`).
       Output mono + bilingual variants. DOCX stays on python-docx (+ run-merge for bold/italic).
       **Feasibility check (2026-07-06)**: `uv pip install --dry-run pdf2zh` pulls a large,
       divergent dependency tree — notably **downgrades `starlette` and `websockets`** below
       what NiceGUI and the realtime-voice websocket client need, plus unrelated heavyweight
       deps (`xinference-client`, `tencentcloud-sdk-*`, `shapely`, `tifffile`). Installing it
       into this app's venv risks breaking NiceGUI's ASGI stack and the Phase-just-shipped voice
       pipeline. **Do not `uv add` it directly** — if/when this lands, run it as an isolated
       subprocess/service with its own venv (or a separate container) that Passage shells out to,
       never as an in-process import. Current PyMuPDF overlay translation (`process_pdf`) already
       ships translated PDFs and is the safe fallback until that isolation is built.
2. [ ] HTML/article pipeline NOT built now — keep seams: segments format-agnostic, translation
       exposed as a clean JSON endpoint.
3. [ ] finplatform bridge v1 = the PDF itself: make Passage excellent on finplatform
       print-to-PDF exports; test with real exports.
4. [ ] Traces: JSONL + `passage_traces` (Langfuse data model). Write points = existing
       `record_feedback` + segment editor callbacks. Trace viewer = a tab on the segment
       review surface: per-segment timeline, edit-distance vs machine output, cost per document.
5. [ ] **LLM-as-judge annotations (David, 2026-07-05)**: an advanced mode where a second model
       annotates translation quality per segment (fluency/accuracy/terminology flags). Rides on
       the trace/score model — a judge annotation is just another score row with model attribution.
6. [ ] **Per-user preference dataset (David, 2026-07-06)**: the old Default/Advanced toggle was
       removed in Phase 2 (segment review always renders now), but its successor is an "advanced
       mode" built on Phases 4+5: each user's `segment_map` runs — machine output + their edits +
       judge scores — accumulate as a per-user preference dataset (terminology choices, tone,
       edit patterns) that can seed per-user glossaries/style prompts and later fine-tuning.
       No new storage needed: it's a read view over `passage_segments` + `passage_traces` keyed by uid.

## Sequencing & scope

Order: 0 → 1 → 2 → 3 → 4 → 5. Phases 1–2 are the visible rehaul. 3 before 4 (BYO endpoint has
value signed-out); 4 before 5's storage-backed parts.

**Out of scope now**: React rewrite, self-hosted Langfuse, collaborative editing,
billing unification, TranslateGemma on hosted GPUs.

## Verification (every phase)

- `.venv\Scripts\python.exe -m pytest tests/` green; extend tests alongside changes.
- Boot: `docker build` + `docker run -e PORT=8080` → `/` and `/api/text_translate` respond.
- Deploy: branch push → CI green → merge → Cloud Run revision healthy
  (`gcloud run services describe translation-app`).
- UX: screenshot pass (desktop + narrow) vs the chosen mockup.
- Flagship: real finplatform print-to-PDF export → Passage → side-by-side layout compare;
  every segment has a trace row; a human edit produces a score/event.
