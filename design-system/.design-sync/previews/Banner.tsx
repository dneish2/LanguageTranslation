import * as React from "react";
import { Banner } from "@passage/design-system";

/** All four severities — the one message pattern used everywhere in the app. */
export function Severities() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", width: "100%" }}>
      <Banner severity="info">Translating "quarterly-report.pdf" to Spanish…</Banner>
      <Banner severity="positive">Translation complete — 42 segments, 1,204 tokens.</Banner>
      <Banner severity="negative">Translation failed: missing result payload.</Banner>
      <Banner severity="warning">Text exceeds this provider's context — split into 4 chunks.</Banner>
    </div>
  );
}
