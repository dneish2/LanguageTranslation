import * as React from "react";
import { ThreadItem } from "@passage/design-system";

/** A document thread and a chat thread — the two real kinds in the Recent Threads drawer. */
export function Kinds() {
  return (
    <div style={{ display: "flex", flexDirection: "column", width: "100%", maxWidth: "20rem" }}>
      <ThreadItem
        label="quarterly-report.pdf"
        kindLabel="document · Spanish"
        onOpen={() => {}}
        onDelete={() => {}}
      />
      <ThreadItem
        label="How do I say 'see you tomorrow' in French?"
        kindLabel="chat · French"
        onOpen={() => {}}
        onDelete={() => {}}
      />
    </div>
  );
}
