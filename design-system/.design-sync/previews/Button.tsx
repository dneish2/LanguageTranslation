import * as React from "react";
import { Button } from "@passage/design-system";

/** The four real variants: one primary per view, secondary for reversible actions, danger/ok reserved. */
export function Variants() {
  return (
    <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
      <Button variant="primary">Translate</Button>
      <Button variant="secondary">Cancel</Button>
      <Button variant="danger">Delete Segment</Button>
      <Button variant="ok">Approve All</Button>
    </div>
  );
}

/** The three real sizes, on the primary variant. */
export function Sizes() {
  return (
    <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }}>
      <Button variant="primary" size="sm">
        Re-translate
      </Button>
      <Button variant="primary" size="default">
        Translate
      </Button>
      <Button variant="primary" size="xl">
        Upload &amp; Translate
      </Button>
    </div>
  );
}

/** Disabled state — used while a translation job is in flight. */
export function Disabled() {
  return (
    <div style={{ display: "flex", gap: "0.75rem" }}>
      <Button variant="primary" disabled>
        Translating…
      </Button>
      <Button variant="secondary" disabled>
        Cancel
      </Button>
    </div>
  );
}
