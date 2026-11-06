// Tool hints/messages at the map's bottom left; confirmations in a centered modal.

import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import { ask, lineDraft, selection, tool } from "../game/ui";
import { linesByNum, notice, stationsById, world } from "../game/world";
import { cancelLink, linking } from "../game/links";
import { CLICK_MODE_COLOURS } from "../game/palette";
import { live } from "../map/live";
import { clickMode, type ClickMode } from "./ToolBar";
import { busDraft } from "../map/busDraw";
import { ferryDraft } from "../map/ferryDraw";
import { buildingDraft } from "../game/buildingView";
import { touchInput } from "../map/touch";
import { useModalDialog } from "./Modal";
import { Chip, Icon } from "./widgets";
import { AchievementToasts } from "./Achievements";
import { t } from "../i18n";

// The build tools carry no hint (Anita, T-1159); the line and trip tools do (toolNotes.lineHint, tripHint).

/** T-2284: the modes with a descriptor and no bar of their own here (toolNotes.simple); bus, ferry,
 * stop editing and the station building tool say theirs in their own rows. */
const KEY_ESC = "Esc", KEY_ENTER = "Enter";   // i18n-ignore (key names)
const SIMPLE = new Set<ClickMode>(["track", "station", "delete", "blueprint", "trip"]);

/** T-1649: a removal that breaks dozens of lines lists the first few and counts the rest. */
const MAX_LINE_ROWS = 5;

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
      <h2 id="confirmation-title">{t("toolNotes.confirm")}</h2>
      <p id="confirmation-text">{q.text}</p>
      {q.lines?.length ? (
        <ul class="confirmation-lines">
          {q.lines.slice(0, MAX_LINE_ROWS).map((r) => {
            const l = world.value && linesByNum(world.value).get(r.line);
            return <li key={r.line}>{l && <Chip line={l} size="sm" plain />}<span>{r.text}</span></li>;
          })}
          {q.lines.length > MAX_LINE_ROWS && <li key="more"><span>{t("toolNotes.moreLines", { count: q.lines.length - MAX_LINE_ROWS })}</span></li>}
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
    const timer = setTimeout(() => tick((v) => v + 1), NOTICE_MS);
    return () => clearTimeout(timer);
  }, [n]);
  const tl = tool.value;
  const draft = lineDraft.value;
  const sel = selection.value;
  // T-1260: a finger taps, and has no Enter key
  const touch = touchInput.value;
  const hint = tl === "line" && draft.stops.length === 1 ? t("toolNotes.lineNext", { touch })
    : tl === "trip" && sel?.kind === "trip" && !sel.to ? t("toolNotes.tripEnd", { touch })
    : tl === "line" ? t("toolNotes.lineHint", { touch })
    : tl === "trip" ? t("toolNotes.tripHint", { touch })
    : "";
  const show = n && performance.now() - n.at < NOTICE_MS;
  // T-1464: the bus line tool: which line, how many points, and the way to finish (also on touch)
  const bus = tl === "bus" ? busDraft.value : null;
  const busLine = bus && bus.line !== null && world.value ? linesByNum(world.value).get(bus.line) : undefined;
  // T-2194: the ferry tool's the same, counting ports
  const ferry = tl === "ferry" ? ferryDraft.value : null;
  const ferryLine = ferry && ferry.line !== null && world.value ? linesByNum(world.value).get(ferry.line) : undefined;
  // T-2218: the station building tool: the outline's corners, and Construct or Save outline
  const building = tl === "building" ? buildingDraft.value : null;
  // T-2284: what a click does now, first, by the mode's colour; the simple modes' words and a cross to leave
  const mode = clickMode.value;
  const linkFrom = linking.value;
  const simple = mode === "passage" ? t("toolNotes.passage", { name: (linkFrom !== null && world.value && stationsById(world.value).get(linkFrom)?.name) || t("toolNotes.aStation") }) : SIMPLE.has(mode) ? t("toolNotes.simple", { mode }) : undefined;
  if (!show && !hint && !bus && !ferry && !building && !simple) return null;
  // stop editing (T-233): which line, how many stops, and the way out
  const editing = tl === "line" && draft.line !== null && world.value ? linesByNum(world.value).get(draft.line) : undefined;
  const sq = mode !== "select" && <span class="mode-sq" style={{ background: CLICK_MODE_COLOURS[mode] }} />;
  return (
    <div id="tool-notes">
      {simple && (
        <p class="stop-edit-head mode-desc">
          <span>{sq}{simple}</span>
          <button class="btn icon mini mode-stop" data-tip={t("toolNotes.stopTip")} aria-label={t("toolNotes.stop")} onClick={() => (mode === "passage" ? cancelLink() : live.tools?.exit())}><Icon.cross /></button>
        </p>
      )}
      {bus && (
        <p class="stop-edit-head">
          <span>{sq}{t("toolNotes.busHead", { route: !!busLine, name: busLine?.name, points: bus.points })}</span>
          <span class="bus-acts">
            <button class="btn text" data-tip={KEY_ESC} onClick={() => live.tools?.exit()}>{t("toolNotes.cancel")}</button>
            <button class="btn text" data-tip={KEY_ENTER} disabled={bus.points < 2} onClick={() => void live.tools?.bus.finish()}>{busLine ? t("toolNotes.saveRoute") : t("toolNotes.makeLine")}</button>
          </span>
        </p>
      )}
      {ferry && (
        <p class="stop-edit-head">
          <span>{sq}{t("toolNotes.ferryHead", { route: !!ferryLine, name: ferryLine?.name, ports: ferry.points })}</span>
          <span class="bus-acts">
            <button class="btn text" data-tip={KEY_ESC} onClick={() => live.tools?.exit()}>{t("toolNotes.cancel")}</button>
            <button class="btn text" data-tip={KEY_ENTER} disabled={ferry.points < 2} onClick={() => void live.tools?.ferry.finish()}>{ferryLine ? t("toolNotes.saveRoute") : t("toolNotes.makeLine")}</button>
          </span>
        </p>
      )}
      {building && (
        <p class="stop-edit-head">
          <span>{sq}{t("toolNotes.buildingHead", { isNew: building.id === null, corners: building.corners })}</span>
          <span class="bus-acts">
            <button class="btn text" data-tip={KEY_ESC} onClick={() => live.tools?.exit()}>{t("toolNotes.cancel")}</button>
            <button class="btn text" data-tip={KEY_ENTER} disabled={!building.closed} onClick={() => void live.tools?.building.finish()}>{building.id === null ? t("toolNotes.construct") : t("toolNotes.saveOutline")}</button>
          </span>
        </p>
      )}
      {editing && (
        <p class="stop-edit-head">
          <span>{sq}{t("toolNotes.editing", { name: editing.name, stops: editing.stops.length })}</span>
          <button class="btn text" onClick={() => live.tools?.exit()}>{t("toolNotes.done")}</button>
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
      <AchievementToasts />
      {ask.value ? <Ask /> : <Notice />}
    </div>
  );
}
