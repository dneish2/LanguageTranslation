# Passage — the "Press" design direction

Warm paper, letterpress ink, one burgundy accent. Palatino-family serif for display type,
Georgia for body text, monospace reserved *only* for data (counts, model names, states) —
never for decoration. No other colors, no drop shadows, no gradients.

## Setup

Import the stylesheet once at the app root — every component and every raw `.p-*` class below
depends on it, and nothing renders styled without it:

```jsx
import "@passage/design-system/styles.css";
```

No provider or context wrapper is needed — every component here is a plain, stateless,
prop-driven function.

## Build with the real components first

`Button`, `Banner`, `SourcePanel`, `TargetPanel`, `Well`, `DataLabel`, `Wordmark`, `ModeTab`,
`HeaderSeparator`, `ThreadItem` — reach for one of these before writing new markup that
duplicates what they already do (a status message is a `Banner`, not a hand-styled div).

```jsx
import { Button, Banner, SourcePanel, TargetPanel, DataLabel } from "@passage/design-system";

<Banner severity="positive">Translation complete — 42 segments, 1,204 tokens.</Banner>
<SourcePanel>
  <p>Original text sits on the warm paper surface.</p>
</SourcePanel>
<TargetPanel>
  <p>Machine output sits on the slightly lighter panel surface.</p>
</TargetPanel>
<div style={{ display: "flex", gap: "0.75rem" }}>
  <Button variant="primary">Translate</Button>
  <Button variant="secondary">Cancel</Button>
</div>
<DataLabel>gpt-5.4-nano · 1,204 tokens</DataLabel>
```

`Button` variants: `primary` (one per view), `secondary` (reversible actions), `danger`
(destructive/stop only), `ok` (record/go affordances only) — never introduce a fifth color.
Sizes: `default`, `sm`, `xl`. `Banner` severities: `info`, `positive`, `negative`, `warning`
— always the same left-rule message shape, never a toast or a modal.

## Building new layout beyond the shipped components

For anything the 10 components don't cover, style with the real CSS custom properties and
utility classes directly — never introduce a new color, a new font stack, or a Tailwind
color class.

**Colors** (`var(--p-*)`): `--p-paper` (page background), `--p-panel` (raised/lighter
surface), `--p-ink` (body text), `--p-muted` (secondary text/data), `--p-accent` (the one
burgundy — links, active states, primary actions), `--p-accent-ink` (text color *on* accent
fills), `--p-rule` / `--p-rule-soft` (borders/hairlines), `--p-ok`, `--p-err`, `--p-warn`
(status only, never decorative).

**Type** (`var(--p-*)`): `--p-display` (headings — Palatino Linotype/Palatino/Book
Antiqua/Georgia, serif), `--p-body` (paragraph text — Georgia/Times New Roman, serif),
`--p-data` (Consolas/Cascadia Mono, monospace — counts, model names, states, traces *only*).

**Utility classes** available globally once the stylesheet is imported: `.p-display` (apply
the display typeface to any element), `.p-data` (the monospace data-accent look — same as
the `DataLabel` component, for cases that aren't a `<span>`), `.p-muted-text` (secondary body
text color), `.p-header` / `.p-drawer` (chrome surfaces), `.p-panel-source` /
`.p-panel-target` / `.p-well` (the same three surfaces the `SourcePanel`/`TargetPanel`/`Well`
components wrap, for cases needing a different element than a `<div>`).

**Radius**: `var(--p-radius)` (3px) on every rounded surface — buttons, banners, panels.
Never a larger radius; this system reads as sharp and press-like, not soft.

## Where the truth lives

`_ds/styles.css` (imported by every design) and its closure, plus each component's
`<Name>.d.ts` for the exact prop contract. When in doubt about a color or class name, grep
the stylesheet rather than guessing — every name in this file was checked against the real
shipped CSS before being written here.
