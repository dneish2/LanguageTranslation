import * as React from "react";

export type ButtonVariant = "primary" | "secondary" | "danger" | "ok";
export type ButtonSize = "default" | "sm" | "xl";

const SIZE_PADDING: Record<ButtonSize, React.CSSProperties> = {
  default: { padding: "0.5rem 1rem" },
  sm: { padding: "0.25rem 0.75rem" },
  xl: { padding: "0.75rem 0", width: "100%", fontSize: "1rem" },
};

export interface ButtonProps
  extends Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, "className"> {
  /** One primary per view; secondary for everything reversible; danger only for
   * destructive/stop actions; ok reserved for record/go affordances. */
  variant?: ButtonVariant;
  size?: ButtonSize;
  children?: React.ReactNode;
}

/**
 * Passage's press-styled button — burgundy primary, hairline secondary, and the
 * two reserved semantic variants (danger, ok). Label face is always the data
 * (monospace) typeface, per the Press design direction.
 */
export function Button({
  variant = "primary",
  size = "default",
  style,
  children,
  ...rest
}: ButtonProps) {
  return (
    <button
      className={`p-btn p-btn-${variant}`}
      style={{ ...SIZE_PADDING[size], ...style }}
      {...rest}
    >
      {children}
    </button>
  );
}
