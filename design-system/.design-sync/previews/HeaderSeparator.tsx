import * as React from "react";
import { Wordmark, HeaderSeparator, ModeTab } from "@passage/design-system";

/** In context: the hairline between the wordmark and the mode tabs — the real header composition. */
export function InHeader() {
  return (
    <div className="p-header" style={{ display: "flex", alignItems: "center", gap: "0.75rem", padding: "0.5rem 1rem" }}>
      <Wordmark onClick={() => {}} />
      <HeaderSeparator />
      <div style={{ display: "flex" }}>
        <ModeTab active>Text</ModeTab>
        <ModeTab>Document</ModeTab>
      </div>
    </div>
  );
}
