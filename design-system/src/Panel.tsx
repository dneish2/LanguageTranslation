import * as React from "react";

export interface PanelProps extends React.HTMLAttributes<HTMLDivElement> {
  children?: React.ReactNode;
}

/** Source-language content surface — sits on the warm paper background. */
export function SourcePanel({ children, ...rest }: PanelProps) {
  return (
    <div className="p-panel-source" {...rest}>
      {children}
    </div>
  );
}

/** Machine-output surface (translations) — sits on the slightly lighter panel tone. */
export function TargetPanel({ children, ...rest }: PanelProps) {
  return (
    <div className="p-panel-target" {...rest}>
      {children}
    </div>
  );
}

/** Quiet recessed surface for secondary content (uploaders, asides) — softer rule than SourcePanel/TargetPanel. */
export function Well({ children, ...rest }: PanelProps) {
  return (
    <div className="p-well" {...rest}>
      {children}
    </div>
  );
}
