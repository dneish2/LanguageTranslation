import * as React from "react";

export interface WordmarkProps {
  onClick?: () => void;
}

/** Passage's header wordmark — the accent-colored full stop is the only flourish. */
export function Wordmark({ onClick }: WordmarkProps) {
  return (
    <span className="p-wordmark" onClick={onClick}>
      Passage<b>.</b>
    </span>
  );
}
