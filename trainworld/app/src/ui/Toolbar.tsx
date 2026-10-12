// Tool hints/messages at the map's bottom left; confirmations in a centered modal.
// Construction controls and the blueprint quote live together in Build (T-100).

import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import { ask, lineDraft, tool, type Tool } from "../game/ui";
import { notice } from "../game/world";

const HINT: Record<Tool, string> = {
  select: "",
  track: "Click where the track should turn. Start or end on track to join it. Click the last point again or press Enter to finish. Backspace takes a point back.",
  station: "Click on track to put a station there.",
  line: "Click stations to insert stops. Press Enter or Escape to finish editing.",
  delete: "Click track, a station or a flyover to remove it. Blueprint is removed at once. Constructed things need a yes first and are not refunded.",
};

/** Native modal keeps map input blocked and traps/restores focus; Escape answers no. */
function Ask() {
  const q = ask.value;
  const dialog = useRef<HTMLDialogElement>(null);
  const cancel = useRef<HTMLButtonElement>(null);
  useLayoutEffect(() => {
    if (!q || !dialog.current) return;
    const el = dialog.current;
    const before = document.activeElement;
    if (!el.open) el.showModal();
    cancel.current?.focus();
    return () => {
      if (el.open) el.close();
      if (before instanceof HTMLElement && before.isConnected) before.focus();
    };
  }, [q]);
  if (!q) return null;
  return (
    <dialog class="confirmation" ref={dialog} aria-labelledby="confirmation-title" aria-describedby="confirmation-text" onCancel={() => ask.value = null} onKeyDown={e => {
      e.stopPropagation();
      if (e.key !== "Tab") return;
      const buttons = [...e.currentTarget.querySelectorAll<HTMLButtonElement>("button:not(:disabled)")];
      const first = buttons[0], last = buttons[buttons.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last?.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first?.focus(); }
    }}>
      <h2 id="confirmation-title">Are you sure?</h2>
      <p id="confirmation-text">{q.text}</p>
      <div class="confirmation-actions">
        <button class="btn text" ref={cancel} onClick={() => (ask.value = null)}>
          {q.no}
        </button>
        <button class="btn text danger" onClick={() => { ask.value = null; q.run(); }}>
          {q.yes}
        </button>
      </div>
    </dialog>
  );
}

const NOTICE_MS = 7000;

function Notice() {
  const n = notice.value;
  const [, tick] = useState(0);
  useEffect(() => {
    if (!n) return;
    const t = setTimeout(() => tick((v) => v + 1), NOTICE_MS);
    return () => clearTimeout(t);
  }, [n]);
  const t = tool.value;
  const draft = lineDraft.value;
  const hint = t === "line" && draft.stops.length === 1 ? "Now click the next station." : HINT[t];
  const show = n && performance.now() - n.at < NOTICE_MS;
  if (!show && !hint) return null;
  return (
    <div id="tool-notes">
      {hint && <p class="muted">{hint}</p>}
      {show && <p class={n.bad ? "bad" : ""}>{n.text}</p>}
    </div>
  );
}

export function Toolbar() {
  return (
    <div id="tools">
      {ask.value ? <Ask /> : <Notice />}
    </div>
  );
}
