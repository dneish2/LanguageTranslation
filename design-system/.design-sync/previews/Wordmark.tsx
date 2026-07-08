import * as React from "react";
import { Wordmark } from "@passage/design-system";

/** The header wordmark, as it renders in the app chrome. */
export function Default() {
  return (
    <div className="p-header" style={{ padding: "0.5rem 1rem" }}>
      <Wordmark onClick={() => {}} />
    </div>
  );
}
