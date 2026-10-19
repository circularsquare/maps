// The editable name in a station's or line's title (T-534): a double click on the name or the
// pencil opens it; Enter or leaving the box keeps the new name, Escape drops it. A line's editor
// (T-670) is opened and closed from outside (`control`) and holds more than the name: leaving the
// box keeps the name but the editor stays open, Enter keeps it and closes, Done is its own button.

import type { ComponentChildren } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import { Icon } from "../widgets";

export function NameEdit({ name, onRename, prompt = false, onPrompted, control, children }: {
  name: string;
  /** called with the trimmed new name when it is not empty and differs */
  onRename: (name: string) => void;
  /** open the editor now (a new line's name, say) */
  prompt?: boolean;
  onPrompted?: () => void;
  /** the editor's open state kept by the caller */
  control?: { open: boolean; set: (open: boolean) => void };
  /** buttons before the pencil */
  children?: ComponentChildren;
}) {
  const [own, setOwn] = useState(false);
  const editing = control ? control.open : own;
  const setEditing = control ? control.set : setOwn;
  const [draft, setDraft] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const done = useRef(false);
  useEffect(() => {
    if (prompt) {
      setEditing(true);
      onPrompted?.();
    }
  }, [prompt]);
  useEffect(() => {
    setDraft(null);
    if (editing) {
      done.current = false;
      input.current?.focus();
      input.current?.select();
    }
  }, [editing]);
  const commit = (close: boolean) => {
    if (done.current) return;
    const n = (draft ?? name).trim();
    if (n && n !== name) onRename(n);
    if (close) {
      done.current = true;
      setEditing(false);
    }
  };
  return (
    <>
      {editing ? (
        <input
          class="name-edit"
          value={draft ?? name}
          ref={input}
          aria-label="Name"
          onInput={(e) => setDraft(e.currentTarget.value)}
          onBlur={() => commit(!control)}
          onKeyDown={(e) => {
            if (e.key === "Enter") commit(true);
            if (e.key === "Escape") {
              done.current = true;
              setEditing(false);
            }
          }}
        />
      ) : (
        <span class="name" onDblClick={() => setEditing(true)}>
          {name}
        </span>
      )}
      {children}
      {control && editing ? (
        <button class="btn icon on" data-tip="Done" aria-label="Done" onClick={() => setEditing(false)}>
          <Icon.check />
        </button>
      ) : (
        <button class="btn icon" data-tip="Rename, or double click the name" aria-label="Rename" onClick={() => setEditing(true)}>
          <Icon.pencil />
        </button>
      )}
    </>
  );
}
