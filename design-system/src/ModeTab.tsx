import * as React from "react";

export interface ModeTabProps
  extends Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, "className"> {
  /** True when this tab is the current workspace mode. */
  active?: boolean;
  children?: React.ReactNode;
}

/**
 * A single header navigation tab (Text | Document | Image | Voice). The mode
 * tabs are Passage's only navigation — there is no separate nav bar.
 */
export function ModeTab({ active = false, children, ...rest }: ModeTabProps) {
  return (
    <button
      className={active ? "p-mode-tab p-mode-tab-active" : "p-mode-tab"}
      {...rest}
    >
      {children}
    </button>
  );
}

/** The vertical hairline separating the wordmark from the mode tabs in the header. */
export function HeaderSeparator() {
  return <div className="p-header-sep" />;
}
