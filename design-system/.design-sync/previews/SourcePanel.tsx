import * as React from "react";
import { SourcePanel } from "@passage/design-system";

/** Source text sitting on the warm paper surface, as it appears before translation. */
export function WithText() {
  return (
    <SourcePanel style={{ padding: "1rem", maxWidth: "28rem" }}>
      <p style={{ margin: 0, fontFamily: "Georgia, 'Times New Roman', serif" }}>
        The quarterly report shows a 12% increase in revenue, driven primarily by
        growth in the European market. Management expects this trend to continue
        into the next fiscal year.
      </p>
    </SourcePanel>
  );
}
