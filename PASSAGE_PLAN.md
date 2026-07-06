# Passage — plan of record

**This file is the single source of truth for the LanguageTranslation → Passage rebuild.**
Any session (human or agent) resuming this work starts here: read "Current state", pick up the
first unchecked item, and update this file as work lands. Keep it committed to git.

Last updated: 2026-07-05.

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
  segment chunking must cap segment size. Hosted tier stays `gpt-4.1-nano` for now.
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
- `job_queue.py` and `translation_metrics.py` are dead code — wire in for the PDF job phase or delete.
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
- **Suite green**: 57/57 passing locally.
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

### Blocked on David (do these once, in the browser)

- [ ] Add repo secret: GitHub → Settings → Secrets and variables → Actions → `OPENAI_API_KEY`
      (deploy fails healthy-boot without it).
- [ ] Merge PR #34 (`fix/boot-crash-and-ci-gate`) once CI is green, confirm the Cloud Run
      revision goes healthy.

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
       favicon=...)`, Passage wordmark in both headers. Still open: OG meta, replace
       `Multilingual.png` hero (do during Phase 2 restructure).

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

1. [ ] Restructure `TranslationUI.py` (~1,700 lines) into `passage/ui/` package
       (`pages.py`, `workspace.py`, `segments.py`, `theme.py`, `errors.py`).
2. [ ] One design system: buttons/banners from theme constants, real icons via `ui.icon`.
3. [ ] Unified error + loading model: single `notify_error()` path, friendly message +
       expandable detail, retry affordance, buttons disabled in flight, shared skeleton/progress.
4. [ ] Merge the two segment editors into ONE segment review surface (inline Advanced editor
       absorbs the Document Editor dialog) — this surface later doubles as the trace viewer.
5. [ ] Responsive layout instead of the `/mobile` UA-redirect; retire `/mobile`.
6. [ ] Restyle `/voice` onto the same system.
7. [ ] Remove dead UI (legacy PPTX `handle_upload`) and dead modules (`job_queue.py` /
       `translation_metrics.py`) or wire them in deliberately — default remove.

### Phase 3 — Model optionality

1. [ ] Generalize provider to `ChatCompletionsProvider` (`base_url`+`api_key`+`model`;
       the `openai` SDK supports this natively). Nothing outside the provider references OpenAI by name.
2. [ ] Provider profiles + settings UI: named profiles stored per-user (Supabase jsonb; local
       JSON signed-out). "Test connection" = `GET /v1/models` (soft) + 1-token completion (hard).
       Model field = free-text + dropdown from `/v1/models`. Generous timeouts (local cold start
       30–120s), non-streaming fallback.
3. [ ] Local default: TranslateGemma via Ollama (`http://host:11434/v1`) — model tag freely
       selectable (4b/smaller experiments; 27b toggle). **Cap segment size for ~2K-token context.**
4. [ ] Hosted tier: keep `gpt-4.1-nano`, credit-gate via finplatform metering pattern later.
5. [ ] Consolidate the model zoo (nano vs mini vs tiktoken's gpt-4o-mini) into one config surface.

### Phase 4 — Accounts + workspace (shared finplatform Supabase)

1. [ ] Port `finplatform/auth/jwt_verify.py` (**the ES256/JWKS version**) into `passage/auth/`.
       Degrade to anon/no-op without keys (zero-config local dev).
2. [ ] Sign-in UI: magic-link + Google via GoTrue REST (mirror finplatform's SDK-less flow).
       Header account chip; signed-out users keep full local use.
3. [ ] Per-user workspace: `passage_documents`, `passage_segments` tables + `passage-files`
       Storage bucket, RLS by uid. "Recent Documents" reads user rows, not the server
       filesystem — also fixes the global-state collision (scope run state per session/user).
4. [ ] Metering: adapt the `_meter`/402 pattern; no-op unless keys set. Billing unification
       with finplatform is later — schema just must not preclude it.

### Phase 5 — Format-preserving PDF translation + traces (the flagship)

1. [ ] Integrate **pdf2zh / BabelDOC** as an async job (wire in or replace `job_queue.py`).
       Output mono + bilingual variants. DOCX stays on python-docx (+ run-merge for bold/italic).
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
