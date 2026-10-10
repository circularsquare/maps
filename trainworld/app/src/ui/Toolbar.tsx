// Tool hints, removal questions and the last edit's message, at the map's bottom left.
// Construction controls and the blueprint quote live together in Build (T-100).

import { useEffect, useState } from "preact/hooks";
import { ask, lineDraft, tool, type Tool } from "../game/ui";
import { notice } from "../game/world";

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
  return (
    <div id="tools">
      {ask.value ? <Ask /> : <Notice />}
    </div>
  );
}
