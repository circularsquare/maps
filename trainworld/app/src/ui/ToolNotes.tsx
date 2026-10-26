// Tool hints/messages at the map's bottom left; confirmations in a centered modal.

import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import { plural } from "../game/format";
import { ask, lineDraft, selection, tool, type Tool } from "../game/ui";
import { linesByNum, notice, world } from "../game/world";
import { live } from "../map/live";
import { busDraft } from "../map/busDraw";
import { touchInput } from "../map/touch";
import { useModalDialog } from "./Modal";
import { Chip } from "./widgets";

// The build tools carry no hint (Anita, T-1159).
const HINT: Record<Tool, string> = {
  select: "",
  track: "",
  station: "",
  line: "Click a circle to add or remove a stop. Press Enter when done.",
  delete: "",
  trip: "Click where a trip starts, or a station.",
  blueprint: "",
  bus: "",
};

/** Native modal keeps map input blocked and traps/restores focus; Escape answers no. */
function Ask() {
  const q = ask.value;
  const dialog = useRef<HTMLDialogElement>(null);
  const cancel = useRef<HTMLButtonElement>(null);
  useModalDialog(dialog, !!q, q);
  useLayoutEffect(() => { if (q) cancel.current?.focus(); }, [q]);
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
      <h2 id="confirmation-title">Confirm?</h2>
      <p id="confirmation-text">{q.text}</p>
      {q.lines?.length ? (
        <ul class="confirmation-lines">
          {q.lines.map((r) => {
            const l = world.value && linesByNum(world.value).get(r.line);
            return <li key={r.line}>{l && <Chip line={l} size="sm" plain />}<span>{r.text}</span></li>;
          })}
        </ul>
      ) : null}
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
  const sel = selection.value;
  const raw = t === "line" && draft.stops.length === 1 ? "Now click the next station."
    : t === "trip" && sel?.kind === "trip" && !sel.to ? "Now click where the trip ends, or a station."
    : HINT[t];
  // T-1260: a finger taps, and has no Enter key
  const hint = touchInput.value ? raw.replace(" Press Enter when done.", "").replace(/\b([Cc])lick\b/, (_, c) => (c === "C" ? "Tap" : "tap")) : raw;
  const show = n && performance.now() - n.at < NOTICE_MS;
  // T-1464: the bus line tool: which line, how many points, and the way to finish (also on touch)
  const bus = t === "bus" ? busDraft.value : null;
  const busLine = bus && bus.line !== null && world.value ? linesByNum(world.value).get(bus.line) : undefined;
  if (!show && !hint && !bus) return null;
  // stop editing (T-233): which line, how many stops, and the way out
  const editing = t === "line" && draft.line !== null && world.value ? linesByNum(world.value).get(draft.line) : undefined;
  return (
    <div id="tool-notes">
      {bus && (
        <p class="stop-edit-head">
          <span>{busLine ? `Route of ${busLine.name}` : "New bus line"}: {plural(bus.points, "point")}</span>
          <button class="btn text" disabled={bus.points < 2} onClick={() => void live.tools?.bus.finish()}>Done</button>
        </p>
      )}
      {editing && (
        <p class="stop-edit-head">
          <span>Editing stops on {editing.name}: {plural(editing.stops.length, "stop")}</span>
          <button class="btn text" onClick={() => live.tools?.exit()}>Done</button>
        </p>
      )}
      {hint && <p class="muted">{hint}</p>}
      {show && <p class={n.bad ? "bad" : ""}>{n.text}</p>}
    </div>
  );
}

export function ToolNotes() {
  return (
    <div id="tools">
      {ask.value ? <Ask /> : <Notice />}
    </div>
  );
}
