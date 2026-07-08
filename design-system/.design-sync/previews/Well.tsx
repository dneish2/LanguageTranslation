import * as React from "react";
import { Well, DataLabel } from "@passage/design-system";

/** A quiet recessed surface for secondary content — here, an upload affordance. */
export function UploadPrompt() {
  return (
    <Well style={{ padding: "1.25rem", maxWidth: "24rem", textAlign: "center" }}>
      <p style={{ margin: "0 0 0.5rem 0", fontFamily: "Georgia, serif" }}>
        Drop a file to translate, or click to browse
      </p>
      <DataLabel>PDF · DOCX · PPTX up to 20MB</DataLabel>
    </Well>
  );
}
