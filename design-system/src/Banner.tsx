import * as React from "react";

export type BannerSeverity = "info" | "positive" | "negative" | "warning";

export interface BannerProps {
  /** Severity is carried entirely by the left rule color — one message pattern
   * for every kind of status line in the app. */
  severity?: BannerSeverity;
  children?: React.ReactNode;
}

/** Passage's single status-message pattern (info/positive/negative/warning). */
export function Banner({ severity = "info", children }: BannerProps) {
  return <div className={`p-banner p-banner-${severity}`}>{children}</div>;
}
