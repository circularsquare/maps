// Native modal dialogs: the centred panels (saved games, account, leaderboards) and the yes or no
// question (ToolNotes.tsx).

import type { ComponentChildren, RefObject } from "preact";
import { useLayoutEffect, useRef } from "preact/hooks";
import { Icon } from "./widgets";

/** Shows the dialog modally while `open` and gives focus back to what had it when it closes.
 * `again` changing closes and opens it afresh. */
export function useModalDialog(ref: RefObject<HTMLDialogElement | null>, open: boolean, again?: unknown) {
  useLayoutEffect(() => {
    const el = ref.current;
    if (!open || !el) return;
    const before = document.activeElement;
    if (!el.open) el.showModal();
    return () => {
      if (el.open) el.close();
      if (before instanceof HTMLElement && before.isConnected) before.focus();
    };
  }, [open, again]);
}

/** A centred modal: Esc, the close button or a click outside its box closes it. Keys stop at the
 * dialog, since the map clears its selection on Esc. `cls` adds to the panel's classes. */
export function Modal({ open, onClose, title, cls = "", children }: { open: boolean; onClose: () => void; title: string; cls?: string; children: ComponentChildren }) {
  const dialog = useRef<HTMLDialogElement>(null);
  useModalDialog(dialog, open);
  if (!open) return null;
  return <dialog class={"saves-panel " + cls} ref={dialog} aria-label={title}
    onCancel={(e) => { e.preventDefault(); onClose(); }}
    onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    onKeyDown={(e) => { e.stopPropagation(); if (e.key === "Escape") { e.preventDefault(); onClose(); } }}>
    <div class="saves-inner">
      <div class="saves-head">
        <h2>{title}</h2>
        <button class="btn icon" aria-label="Close" data-tip="Close (Esc)" onClick={onClose}><Icon.cross /></button>
      </div>
      {children}
    </div>
  </dialog>;
}
