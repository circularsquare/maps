// The editable name in a station's or line's title (T-534): a double click on the name or the
// pencil opens it; Enter or leaving the box keeps the new name, Escape drops it. A line's editor
// (T-670) is opened and closed from outside (`control`) and holds more than the name: leaving the
// box keeps the name but the editor stays open, Enter keeps it and closes, Done is its own button.

import type { ComponentChildren } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import { touchInput } from "../../map/touch";
import { t } from "../../i18n";
import { Icon } from "../widgets";
import { keyIsTheirs } from "../keys";

export function NameEdit({ name, onRename, prompt = false, onPrompted, control, hotkey, clearable = false, children }: {
  name: string;
  /** called with the trimmed new name when it is not empty (or `clearable`) and differs; while a promise it returns is out, the
   * new name is shown (T-3793) */
  onRename: (name: string) => unknown;
  /** open the editor now (a new line's name, say) */
  prompt?: boolean;
  onPrompted?: () => void;
  /** the editor's open state kept by the caller */
  control?: { open: boolean; set: (open: boolean) => void };
  /** a plain key (compared lowercase) that opens the editor, while it is not open and no text is being typed */
  hotkey?: string;
  /** an emptied name is sent too (T-4080: a bus stop's own name taken off) */
  clearable?: boolean;
  /** buttons before the pencil */
  children?: ComponentChildren;
}) {
  const [own, setOwn] = useState(false);
  const editing = control ? control.open : own;
  const setEditing = control ? control.set : setOwn;
  const [draft, setDraft] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const done = useRef(false);
  const prompted = useRef(false);
  // T-3793: the name just kept, shown until its edit is answered (the world is up to date by then) or the name changes;
  // on a big save the edit takes a moment, and the old name coming back looked like a rename that had not gone through
  const [kept, setKept] = useState<string | null>(null);
  useEffect(() => setKept(null), [name]);
  useEffect(() => {
    if (prompt) {
      prompted.current = !editing;
      setEditing(true);
      onPrompted?.();
    }
  }, [prompt]);
  useEffect(() => {
    setDraft(null);
    if (editing) {
      done.current = false;
      // a new line's name opened by a finger is not focused: the on-screen keyboard would cover the map (T-1756)
      const skip = prompted.current && touchInput.peek();
      prompted.current = false;
      if (!skip) {
        input.current?.focus();
        input.current?.select();
      }
    }
  }, [editing]);
  useEffect(() => {
    if (!hotkey || editing) return;
    const open = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() !== hotkey || e.ctrlKey || e.metaKey || e.altKey || e.shiftKey || e.repeat || e.defaultPrevented || keyIsTheirs(e)) return;
      if (document.querySelector("dialog:modal, .start-menu")) return;
      e.preventDefault();
      setEditing(true);
    };
    window.addEventListener("keydown", open);
    return () => window.removeEventListener("keydown", open);
  }, [hotkey, editing]);
  const commit = (close: boolean) => {
    if (done.current) return;
    const shown = kept ?? name, n = (draft ?? shown).trim();
    if ((n || (clearable && draft !== null)) && n !== shown) {
      const out = onRename(n);
      if (out instanceof Promise) {
        setKept(n);
        const drop = () => setKept((k) => (k === n ? null : k));
        out.then(drop, drop);
      }
    }
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
          value={draft ?? kept ?? name}
          ref={input}
          aria-label={t("nameEdit.name")}
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
          {kept ?? name}
        </span>
      )}
      {children}
      {control && editing ? (
        <button class="btn icon on" data-tip={t("nameEdit.done")} aria-label={t("nameEdit.done")} onClick={() => setEditing(false)}>
          <Icon.check />
        </button>
      ) : (
        <button class="btn icon" data-tip={t("nameEdit.renameTip", { keyed: !!hotkey, key: hotkey?.toUpperCase() ?? "" })} aria-label={t("nameEdit.rename")} onClick={() => setEditing(true)}>
          <Icon.pencil />
        </button>
      )}
    </>
  );
}
