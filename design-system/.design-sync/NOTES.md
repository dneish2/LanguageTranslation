# Passage design-system sync notes

## Repo-specific facts

- This package (`design-system/`) did not exist before this sync — it was built specifically to give
  claude.ai/design real components to build with, since Passage's actual UI is server-rendered
  Python (NiceGUI) with no JS component boundary. `src/styles.css` is a byte-identical copy of the
  live app's `../static/passage.css`; every component wraps the real `.p-*` classes and CSS custom
  properties 1:1 — never a reimplementation of the visual language, only of the JS component shape.
- Build: `npm run build` (tsup → `dist/index.{js,cjs,d.ts}` + `dist/styles.css`, the latter copied
  verbatim by the build script, not bundled — hence `cfg.cssEntry: "dist/styles.css"`).
- 10 components/exports: `Button`, `Banner`, `SourcePanel`, `TargetPanel`, `Well`, `DataLabel`,
  `Wordmark`, `ModeTab`, `HeaderSeparator`, `ThreadItem`.

## Fonts — accepted substitutes (David's explicit OK, 2026-07-07)

The real app's font stacks are `--p-display: "Palatino Linotype", Palatino, "Book Antiqua", Georgia,
serif`, `--p-body: Georgia, "Times New Roman", serif`, `--p-data: Consolas, "Cascadia Mono",
monospace`. Palatino Linotype, Book Antiqua, Georgia, Times New Roman, and Consolas are
Microsoft/Monotype-licensed fonts bundled with Windows — not freely redistributable as web-font
files, so they are NOT shipped; the bundle falls through to system fonts for these, same as any
browser rendering the live app on a machine without them installed.

**Cascadia Mono** is genuinely open source (Microsoft, SIL Open Font License 1.1) and IS shipped for
real: `fonts/cascadia-mono.css` + `fonts/CascadiaMono-Regular.ttf`, fetched directly from Google
Fonts' CDN (`fonts.gstatic.com/s/cascadiamono/...`), wired via `cfg.extraFonts`. `[FONT_MISSING]`
still fires for the other five families — expected and accepted, not a bug to chase.

## Re-sync risks

- If `static/passage.css` changes in the main app, `design-system/src/styles.css` must be
  re-copied by hand (there's no automated link between the two files) — check `diff` before
  trusting a re-sync's render check.
- If new `.p-*` classes are added to the real app that aren't wrapped by any component here, this
  package silently falls behind — there's no automated coverage check.
