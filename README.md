# Passage

A translation workspace. Paste text, drop a document, snap an image, or speak —
Passage translates it, shows its work segment by segment, and keeps humans in the
loop for review and refinement.

Formerly "LanguageTranslation"; being rebuilt as Passage. The plan of record —
locked decisions, roadmap, current state — lives in [PASSAGE_PLAN.md](PASSAGE_PLAN.md).
Read that first if you're picking up development.

## Run it locally

Requires Python 3.12. The project uses a `uv` virtual environment at `.venv`.

```bash
git clone https://github.com/dneish2/LanguageTranslation.git
cd LanguageTranslation

# create/sync the venv (or use an existing .venv)
uv venv && uv pip install -r requirements.txt

# provide your OpenAI key (a .env file in the repo root also works)
# PowerShell:  $env:OPENAI_API_KEY = "sk-..."
# bash:        export OPENAI_API_KEY=sk-...

.venv/Scripts/python.exe TranslationUI.py   # Windows
# .venv/bin/python TranslationUI.py         # macOS/Linux
```

Open http://localhost:8080. Without an API key the app still boots and serves the
UI; translation calls fail with a clear error until a provider is configured.

## Pages

| Route | What it is |
|-------|------------|
| `/` | The workspace: Text / Document / Image modes via the header tabs, language bar, facing source/translation panels, segment review for documents, Recent Threads drawer |
| `/voice` | Voice translation: record (or paste a transcript), hear the translation spoken back |
| `/mobile` | Mobile layout (being folded into one responsive layout) |

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | Provider key for translation, Whisper, and TTS |
| `PORT` | `8080` | Listen port (Cloud Run injects this) |
| `PASSAGE_PUBLIC_API` | off | Set `1` to disable the API session-token gate (local dev) |
| `PASSAGE_API_RATE_LIMIT` | `30` | Requests per minute per IP on `/api/*` |
| `PASSAGE_MAX_TEXT_CHARS` | `8000` | Max characters per text translation |
| `PASSAGE_MAX_UPLOAD_BYTES` | `8388608` | Max upload size (8 MB) |
| `LIVE_TEXT_STREAMING` | `false` | Enable SSE streaming for long text translations |

## API

The `/api/*` endpoints (`text_translate`, `text_translate_stream`, `voice_translate`,
`image_translate`) are used by the app's own pages and are gated by a short-lived
session token that those pages embed (`X-Passage-Token` header) plus a per-IP rate
limit. They are not a public API; real accounts land later (see the plan, Phase 4).

## Tests

```bash
.venv/Scripts/python.exe -m pytest tests/
```

The suite gates deployment: pushes to `main` run tests in CI and, on green, build
and deploy to Cloud Run (`.github/workflows/deploy.yml`). Secrets (`OPENAI_API_KEY`,
`GCP_SA_KEY`) live in GitHub Actions secrets — never in the repo.

## Design

The visual identity ("Press": warm paper, letterpress ink, burgundy accent;
Palatino display over Georgia body; monospace reserved for data) is defined in
`static/passage.css` and mirrored in `theme.py`. All UI styling flows through
those tokens — no ad-hoc color classes.
