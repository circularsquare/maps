// The bar along the map's bottom left (T-082): undo and redo, the blueprint's cost and Construct,
// which are needed from any tab while working on the map; the build tools themselves are in the
// Build tab (ui/tabs.tsx). Above the bar, a hint for the tool in use and the last message (why an
// edit was refused, what was built).

import { useEffect, useState } from "preact/hooks";
import { money } from "../game/format";
import { ask, lineDraft, tool, type Tool } from "../game/ui";
import { edit, notice, redo, say, undo, world } from "../game/world";
import { Icon } from "./widgets";

const HINT: Record<Tool, string> = {
  select: "",
  track: "Click where the track should turn. Start or end on track to join it. Click the last point again or press Enter to finish. Backspace takes a point back.",
  station: "Click on track to put a station there.",
  line: "Click stations in order. Press Enter or click the last station again when the line is done.",
  delete: "Click track, a station or a flyover to remove it. Blueprint is removed at once. Constructed things need a yes first and are not refunded.",
};

/** A question from the game (T-096: removing something constructed), answered here; Esc is no. */
function Ask() {
  const q = ask.value;
  if (!q) return null;
  return (
    <div id="tool-notes" class="ask">
      <p>{q.text}</p>
      <div class="group">
        <button class="btn text" onClick={q.run}>
          {q.yes}
        </button>
        <button class="btn text" onClick={() => (ask.value = null)}>
          {q.no}
        </button>
      </div>
    </div>
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
  const w = world.value;
  const blueprint = w?.money.blueprintCost ?? 0;
  const construct = async () => {
    const a = await edit({ op: "constructAll" });
    if (a.ok) say(`Constructed for ${money(a.charge * 1e6)}.`);
  };
  return (
    <div id="tools">
      {ask.value ? <Ask /> : <Notice />}
      <div class="tools-row">
        <div class="group">
          <button class="btn icon" title="Undo (Ctrl+Z)" disabled={!w?.canUndo} onClick={() => void undo()}>
            <Icon.undo />
          </button>
          <button class="btn icon" title="Redo (Ctrl+Y)" disabled={!w?.canRedo} onClick={() => void redo()}>
            <Icon.redo />
          </button>
        </div>
        <span class="blueprint">
          Blueprint <b class="num">{money(blueprint)}</b>
        </span>
        <button class="btn text" disabled={blueprint <= 0} onClick={construct}>
          Construct blueprints
        </button>
      </div>
    </div>
  );
}
