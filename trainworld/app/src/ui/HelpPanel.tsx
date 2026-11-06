// T-908: the controls a player would not find by looking, opened by the "?" at the top of Settings.
// Only gestures, modifiers and keys that no button shows; Esc or a press outside closes it.

import { signal } from "@preact/signals";
import { useEffect, useRef } from "preact/hooks";
import { t } from "../i18n";
import { ACTION_KEYS, BUS_KEY, DEMAND_KEY, HIDE_UI_KEY, SWITCH_PICK_KEY, TAB_KEYS, TOOL_KEYS, TRIP_KEY } from "./toolKeys";

export const helpOpen = signal(false);

const keyOf = (act: string) => Object.keys(ACTION_KEYS).find((k) => ACTION_KEYS[k] === act)!.toUpperCase();
const toolKeys = Object.keys(TOOL_KEYS).join(", ");
const tabKeys = Object.keys(TAB_KEYS).map((k) => k.toUpperCase()).join(", ");

function sections(): [string, [string, string][]][] {
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
    <div class="help-panel" ref={ref} role="dialog" aria-label={t("help.title")}>
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
