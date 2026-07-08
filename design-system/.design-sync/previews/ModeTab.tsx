import * as React from "react";
import { ModeTab } from "@passage/design-system";

/** The real header navigation: Text | Document | Image | Voice, one active. */
export function TabRow() {
  return (
    <div className="p-header" style={{ display: "flex", alignItems: "center", padding: "0.5rem 1rem", gap: 0 }}>
      <ModeTab active>Text</ModeTab>
      <ModeTab>Document</ModeTab>
      <ModeTab>Image</ModeTab>
      <ModeTab>Voice</ModeTab>
    </div>
  );
}
