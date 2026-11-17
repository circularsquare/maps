// T-908, T-3527: the help panel, opened by the "?" at the top of Settings or F1. The questions players
// ask most, each opening to its answer, then the controls a player would not find by looking (only
// gestures, modifiers and keys that no button shows). Esc or a press outside closes it.

import { signal } from "@preact/signals";
import { useEffect, useRef, useState } from "preact/hooks";
import { t } from "../i18n";
import { mobileDevice } from "../workers/deviceMemory";
import { ACTION_KEYS, BUS_KEY, DEMAND_KEY, HELP_KEY, HIDE_UI_KEY, SWITCH_PICK_KEY, TAB_KEYS, TOOL_KEYS, TOOLS, TRIP_KEY } from "./toolKeys";
import { Icon } from "./widgets";

export const helpOpen = signal(false);

/** F1 opens or closes the panel from anywhere, and never the browser's own help. Nothing while the
 * interface is hidden or a dialog or the start menu covers the dock, so it never opens unseen. */
export function helpKey() {
  window.addEventListener("keydown", (e) => {
    if (e.key !== HELP_KEY || e.ctrlKey || e.metaKey || e.altKey || e.shiftKey) return;
    e.preventDefault();
    if (e.repeat || document.documentElement.classList.contains("ui-hidden")) return;
    if (document.querySelector("dialog:modal, .start-menu")) return;
    helpOpen.value = !helpOpen.peek();
  }, true);
}

/** T-3598: a control's name inside an answer is wrapped in these two private-use characters, and
 * rendered in the theme colour by `marked`. */
const MARK_ON = "", MARK_OFF = "";
const mark = (s: string) => MARK_ON + s + MARK_OFF;
function marked(a: string) {
  return a.split(MARK_ON).map((part, i) => {
    if (i === 0) return part;
    const [name, rest] = part.split(MARK_OFF);
    return [<span class="help-ctl">{name}</span>, rest];
  });
}

/** T-3527: the questions, worded with the names of the buttons and settings they point to, so an answer
 * follows its control's name in every language. Phones get their own gestures and the intercity
 * switch, which only phones have. */
function questions(): [string, string][] {
  const m = mark;
  const touch = mobileDevice();
  const [, selectName, , selectKey] = TOOLS.find(([id]) => id === "blueprint")!;
  const [, deleteName, , deleteKey] = TOOLS.find(([id]) => id === "delete")!;
  const copy = m(t("busInspector.copy"));
  const list: [string, string][] = [
    [t("help.qBranch"), t("help.aBranch", { copy })],
    [t("help.qExpress"), t("help.aExpress", { copy })],
    [t("help.qLoop"), t("help.aLoop", { touch })],
  ];
  if (touch) list.push([t("help.qIntercity"), t("help.aIntercity", { setting: m(t("settingsTab.mobileIntercity")) })]);
  list.push(
    [t("help.qRebuild"), t("help.aRebuild", { touch, tool: m(selectName), key: m(selectKey), rebuild: m(t("blueprint.rebuild")) })],
    [t("help.qEditTrack"), t("help.aEditTrack", { touch, del: m(deleteName), delKey: m(deleteKey), tool: m(selectName), key: m(selectKey), rebuild: m(t("blueprint.rebuild")), remove: m(t("blueprint.remove")) })],
    [t("help.qFourTracks"), t("help.aFourTracks")],
    // T-3527: "Recalculate trips while paused" is on by default, so pausing alone still works trips out after each edit
    [t("help.qRecalc"), t("help.aRecalc", { setting: m(t("settingsTab.recalcWhilePaused")) })],
    [t("help.qLoseMoney"), t("help.aLoseMoney")],
    [t("help.qLoseMoneyIntercity"), t("help.aLoseMoneyIntercity")],
  );
  return list;
}

const keyOf = (act: string) => Object.keys(ACTION_KEYS).find((k) => ACTION_KEYS[k] === act)!.toUpperCase();
const toolKeys = Object.keys(TOOL_KEYS).join(", ");
const tabKeys = Object.keys(TAB_KEYS).map((k) => k.toUpperCase()).join(", ");

/** T-2863: on a phone or tablet, only the rows a finger can do (no keys, no modifiers, no right button) */
function phoneSections(): [string, [string, string][]][] {
  return [
  [t("help.bubblesHead"), [
    [t("help.holdDrag"), t("help.boxFromBubble")],
  ]],
  [t("help.buildingHead"), [
    [t("help.holdBus"), t("help.doDelete")],
    [t("help.dragBusPoint"), t("help.doMove")],
    [t("help.clickBusEnd"), t("help.drawFromEnd")],
  ]],
  [t("help.panelsHead"), [
    [t("help.dblFare"), t("help.doRemove")],
    [t("help.dblName"), t("help.doRename")],
    [t("help.dblRow"), t("help.findOnMap")],
  ]],
  ];
}

function sections(): [string, [string, string][]][] {
  if (mobileDevice()) return phoneSections();
  return [
  [t("help.mapHead"), [
    [t("help.rightDragCtrl"), t("help.rotate")],
  ]],
  [t("help.bubblesHead"), [
    [t("help.shiftDrag"), t("help.boxSelect")],
    [t("help.holdDrag"), t("help.boxFromBubble")],
    [t("help.shiftClick"), t("help.addBubble")],
    [t("help.shiftDragFar"), t("help.exploreTrips")],
  ]],
  [t("help.buildingHead"), [
    [t("help.shiftClick"), t("help.placeTrack")],
    [t("help.doubleClickEnter"), t("help.stopDrawing")],
    [t("help.backspaceRight"), t("help.undoPiece")],
    [t("help.dblTrack"), t("help.addPoint")],
    [t("help.rightClickBus"), t("help.doDelete")],
    [t("help.dragBusPoint"), t("help.doMove")],
    [t("help.clickBusEnd"), t("help.drawFromEnd")],
    [t("help.capsLock"), t("help.keepPlacing")],
    [t("help.rightDrag"), t("help.moveMapBlueprint")],
    [SWITCH_PICK_KEY.toUpperCase(), t("help.togglePicked")],
    [t("help.keyDelete"), t("help.removePicked")],
  ]],
  [t("help.keysHead"), [
    [t("help.shift17"), t("help.levels")],
    [tabKeys, t("help.tabs")],
    [toolKeys, t("help.tools")],
    [TRIP_KEY.toUpperCase(), t("help.exploreTrip")],
    [DEMAND_KEY.toUpperCase(), t("help.cycleDemand")],
    [keyOf("newLine"), t("help.newLine")],
    [BUS_KEY.toUpperCase(), t("help.newBusLine")],
    [keyOf("editStops"), t("help.editStops")],
    [keyOf("addPassed"), t("help.addPassed")],
    [keyOf("construct"), t("help.construct")],
    [t("help.space"), t("help.pause")],
    ["Ctrl Z, Ctrl Y", t("help.undoRedo")], // i18n-ignore
    ["Ctrl S", t("help.saveNow")], // i18n-ignore
    ["/, Ctrl K, Ctrl F", t("help.find")], // i18n-ignore
    [HIDE_UI_KEY, t("help.hideUi")],
    [HELP_KEY, t("help.openHelp")],
    [t("help.esc"), t("help.back")],
  ]],
  [t("help.panelsHead"), [
    [t("help.dblFare"), t("help.doRemove")],
    [t("help.dblName"), t("help.doRename")],
    [t("help.dblRow"), t("help.findOnMap")],
    [t("help.stepTrains"), t("help.stepBy")],
  ]],
  ];
}

export function HelpPanel() {
  const open = helpOpen.value;
  const ref = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState<number | null>(null);
  useEffect(() => {
    if (!open) return;
    // before main.ts's Escape, which would clear the selection too
    const key = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      helpOpen.value = false;
      e.preventDefault();
      e.stopImmediatePropagation();
    };
    const press = (e: PointerEvent) => {
      const el = e.target as Element;
      if (!ref.current?.contains(el) && !el.closest?.(".help-open")) helpOpen.value = false;
    };
    window.addEventListener("keydown", key, true);
    window.addEventListener("pointerdown", press, true);
    return () => {
      window.removeEventListener("keydown", key, true);
      window.removeEventListener("pointerdown", press, true);
    };
  }, [open]);
  if (!open) return null;
  return (
    <div class="help-panel" ref={ref} role="dialog" aria-label={t("help.name")}>
      <ul class="help-qs">
        {questions().map(([q, a], i) => (
          <li key={q}>
            <button class={"help-q" + (i === shown ? " open" : "")} aria-expanded={i === shown} onClick={() => setShown(i === shown ? null : i)}>
              <Icon.chevron /><span>{q}</span>
            </button>
            {i === shown && <p class="help-a">{marked(a)}</p>}
          </li>
        ))}
      </ul>
      {sections().map(([head, rows]) => (
        <section key={head}>
          <h3>{head}</h3>
          <dl>
            {rows.map(([k, v]) => [<dt>{k}</dt>, <dd>{v}</dd>])}
          </dl>
        </section>
      ))}
    </div>
  );
}
