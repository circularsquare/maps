// The editable name in a station's or line's title (T-534): a double click on the name or the
// pencil opens it; Enter or leaving the box keeps the new name, Escape drops it.

import type { ComponentChildren } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import { Icon } from "../widgets";

export function NameEdit({ name, onRename, prompt = false, onPrompted, children }: {
  name: string;
  /** called with the trimmed new name when it is not empty and differs */
  onRename: (name: string) => void;
  /** open the editor now (a new line's name, say) */
  prompt?: boolean;
  onPrompted?: () => void;
  /** buttons before the pencil */
  children?: ComponentChildren;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const input = useRef<HTMLInputElement>(null);
  const done = useRef(false);
  const open = () => {
    setDraft(name);
    setEditing(true);
  };
  useEffect(() => {
    if (prompt) {
      open();
      onPrompted?.();
    }
  }, [prompt]);
  useEffect(() => {
    if (editing) {
      done.current = false;
      input.current?.focus();
      input.current?.select();
    }
  }, [editing]);
  const commit = () => {
    if (done.current) return;
    done.current = true;
    const n = draft.trim();
    if (n && n !== name) onRename(n);
    setEditing(false);
  };
  return (
    <>
      {editing ? (
        <input
          class="name-edit"
          value={draft}
          ref={input}
          onInput={(e) => setDraft(e.currentTarget.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") commit();
            if (e.key === "Escape") {
              done.current = true;
              setEditing(false);
            }
          }}
        />
      ) : (
        <span class="name" onDblClick={open}>
          {name}
        </span>
      )}
      {children}
      <button class="btn icon" data-tip="Rename, or double click the name" aria-label="Rename" onClick={open}>
        <Icon.pencil />
      </button>
    </>
  );
}
