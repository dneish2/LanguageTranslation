# Working in this repo

## Tests

`.venv\Scripts\python.exe -m pytest tests/` — **pytest IS correct here.** The no-pytest rule in
the workspace root `CLAUDE.md` is finplatform's, and does not apply to this repo. uv venv at
`.venv`, Python 3.12 locally; CI runs 3.11, so check both before claiming a suite is green.

Tests here assert **which path ran**, never that a call returned 200. Fake endpoints count their
own calls and stamp their identity into the bytes they return, so a test fails if another engine
served the request *and* fails if nothing ran at all. A test that would still pass when the local
engine silently never ran proves nothing — with no Ollama running, every local path falls back to
hosted and the app looks perfect.

## Branches and PRs

**After a squash merge, start a fresh branch off `main`.** This is not style. PR #40 was
squash-merged from `feat/traces-and-portability-docs`; work continued on that same branch; and
because a squash rewrites the merged commits into one new SHA, `main` and the branch then carried
the same edits under different history. Every file touched by both sides conflicted as add/add,
41 commits later, even though `main` held nothing the branch was missing.

That cost two things worth naming, because neither is obvious:

1. **A conflicting PR shows four green checks.** Mergeability is a banner, not a check run, so it
   is invisible to `gh pr checks` and to the status API rollup.
2. **A conflicting PR silently loses workflows.** GitHub cannot build the merge commit, so every
   `pull_request`-triggered job never fires. The deploy gate did not go red — it went *absent*,
   and an absent check reads exactly like a passing one.

There is now a `mergeable` job in `ci.yml` that turns both into one red X. It is a backstop, not a
substitute for branching off `main`.

**After opening or updating a PR, check mergeability, not just checks:**

```bash
gh pr view <n> --json mergeable,mergeStateStatus --jq '{mergeable,mergeStateStatus}'
# want: MERGEABLE / CLEAN.  DIRTY+CONFLICTING = conflicts.  UNSTABLE = checks still running.
gh pr checks <n>          # and confirm the workflows you EXPECT are all present
```

Never report a PR as green off `gh pr checks` alone.

## Commits

Explain the reasoning, not the diff — a future session should be able to disagree with evidence
rather than guess. Record the fix that was wrong first when there was one; that is usually the
more useful half. **No AI attribution anywhere** — not in commit messages, PR titles, PR bodies
or comments.

## Rules the codebase enforces

- **A prompt is not a mechanism.** Never enforce a model invariant by asking for it in a prompt.
  Mask it, route it, or validate it in code, and test the mechanism. URLs, verbatim reading and
  output format all failed the prompt way first; see `RESEARCH.md` §1.
- **Measure model-dependent results across ≥5 runs**, and report the median plus how many runs
  improved. One run is not a result — `DECISIONS.md` §6. Deterministic changes (masking, routing,
  parsing) can be measured once; the distinction is whether sampling is in the loop.
- **Privacy and metering claims come from what ACTUALLY served the request**, via
  `policy.classify_run`, never from a capability probe read at render time. That one mistake has
  now appeared on five surfaces; if you find a sixth, fix the shape rather than the surface.
- **Browser-verify NiceGUI UI changes.** NiceGUI 3 pipes `ui.html()` through DOMPurify, which
  strips inline event handlers with no console error. Unit tests miss it entirely.
- **Local models share one GPU** — benchmark them serially or you are measuring VRAM contention.

## Scope

Passage is a **portfolio piece**, not a product: no accounts, no billing, no durable storage. See
the "What this deliberately does not do" section of `README.md` before adding anything from it.
`passage/auth/` is ported, live-tested and deliberately unwired.

## The documents

`README.md` is the front door. `RESEARCH.md` holds every measurement and how it was taken —
**do not re-derive what is already in there.** `DECISIONS.md` holds every open call with its
reasoning. `PORTABILITY.md` records what is verified on other hardware and, more importantly,
what is not.
