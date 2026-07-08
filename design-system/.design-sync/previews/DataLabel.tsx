import * as React from "react";
import { DataLabel } from "@passage/design-system";

/** Real examples of the data face in use — counts, model names, states. */
export function Examples() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
      <DataLabel>42 segments translated</DataLabel>
      <DataLabel>gpt-5.4-nano</DataLabel>
      <DataLabel>1,204 tokens</DataLabel>
      <DataLabel>document · Spanish</DataLabel>
    </div>
  );
}
