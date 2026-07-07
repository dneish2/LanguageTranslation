import * as React from "react";

export interface ThreadItemProps {
  /** The thread's display label (document filename or chat excerpt), already truncated by the caller. */
  label: string;
  /** e.g. "document · Spanish" or "chat · French". */
  kindLabel: string;
  onOpen?: () => void;
  onDelete?: () => void;
}

/**
 * One row in the Recent Threads drawer — a document or chat translation the
 * user can reopen, with a delete affordance. Private per session.
 */
export function ThreadItem({ label, kindLabel, onOpen, onDelete }: ThreadItemProps) {
  return (
    <div style={{ display: "flex", width: "100%", alignItems: "center", gap: 0 }}>
      <button onClick={onOpen} className="p-thread-item" style={{ flexGrow: 1 }}>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 0 }}>
          <span style={{ fontSize: "0.875rem" }}>{label}</span>
          <span className="p-thread-kind">{kindLabel}</span>
        </div>
      </button>
      <button onClick={onDelete} className="p-mode-tab" title="Remove this thread" aria-label="Remove this thread">
        ×
      </button>
    </div>
  );
}
