import * as React from "react";

export interface DataLabelProps extends React.HTMLAttributes<HTMLSpanElement> {
  children?: React.ReactNode;
}

/**
 * The data accent — Passage's monospace face reserved for counts, model
 * names, states, and traces. Never used for decoration.
 */
export function DataLabel({ children, ...rest }: DataLabelProps) {
  return (
    <span className="p-data" {...rest}>
      {children}
    </span>
  );
}
