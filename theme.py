"""Passage design tokens — Direction "Press" (locked 2026-07-05, PASSAGE_PLAN.md).

Python-side constants mirroring static/passage.css. All NiceGUI styling goes
through these; never write raw Tailwind color classes in UI code. Spacing
utilities (px-*, py-*, w-full, ...) stay inline where layout demands them —
only color, type, and surface identity live here.
"""

PALETTE = {
    "paper": "#F6F0E3",
    "panel": "#FBF7ED",
    "ink": "#2B241C",
    "muted": "#7A6F5D",
    "accent": "#802F3D",
    "rule": "#DCD2BE",
    "ok": "#50694E",
    "err": "#8C3B2E",
    "warn": "#8A6A2F",
}

# Buttons — one primary per view; secondary for everything reversible;
# danger only for destructive/stop actions; ok reserved for record/go affordances.
BTN_PRIMARY = "p-btn p-btn-primary px-4 py-2"
BTN_PRIMARY_SM = "p-btn p-btn-primary px-3 py-1"
BTN_PRIMARY_XL = "p-btn p-btn-primary w-full py-3 text-base"
BTN_SECONDARY = "p-btn p-btn-secondary px-4 py-2"
BTN_SECONDARY_SM = "p-btn p-btn-secondary px-3 py-1"
BTN_SECONDARY_XL = "p-btn p-btn-secondary w-full py-3 text-base"
BTN_DANGER = "p-btn p-btn-danger px-4 py-2"
BTN_DANGER_SM = "p-btn p-btn-danger px-3 py-1"
BTN_OK = "p-btn p-btn-ok px-4 py-2"
BTN_OK_SM = "p-btn p-btn-ok px-3 py-1"

# Banners — single message pattern, severity carried by the left rule.
BANNER = {
    "info": "p-banner p-banner-info",
    "positive": "p-banner p-banner-positive",
    "negative": "p-banner p-banner-negative",
    "warning": "p-banner p-banner-warning",
}

# Surfaces — source text sits on paper, machine output sits on panel.
PANEL_SOURCE = "p-panel-source"
PANEL_TARGET = "p-panel-target"
WELL = "p-well"

# Chrome
HEADER = "p-header"
DRAWER = "p-drawer"
WORDMARK = "p-wordmark"

# Data accents (Ledger's contribution): counts, model names, states, traces.
DATA = "p-data"

HEAD_HTML = '<link rel="stylesheet" href="/static/passage.css">'
