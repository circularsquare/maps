// T-908: the controls a player would not find by looking, opened by the "?" in the dock's tab row.
// Only gestures, modifiers and keys that no button shows; Esc or a press outside closes it.

import { signal } from "@preact/signals";
import { useEffect, useRef } from "preact/hooks";
import { ACTION_KEYS, TOOL_KEYS, TRIP_KEY } from "./tabs";

export const helpOpen = signal(false);

const keyOf = (act: string) => Object.keys(ACTION_KEYS).find((k) => ACTION_KEYS[k] === act)!.toUpperCase();
const toolKeys = Object.keys(TOOL_KEYS).join(", ");

const SECTIONS: [string, [string, string][]][] = [
  ["Demand bubbles", [
    ["Shift drag", "Draw a box to select bubbles"],
    ["Hold, then drag", "Draw the box from a bubble"],
    ["Shift click", "Add a bubble to the selection"],
    ["Shift drag over far bubbles", "Explore trips to them"],
  ]],
  ["Building", [
    ["Shift click", "Place track and keep drawing"],
    ["Double click, Enter", "Stop drawing track"],
    ["Backspace", "Take back the last piece of track"],
    ["Caps Lock", "Turn keep placing on or off"],
    ["Right drag", "Move the map in the blueprint tool"],
    ["D, E, Delete", "Disable, enable or remove the picked blueprints"],
  ]],
  ["Keys", [
    [toolKeys, "Track, stations, delete and blueprint tools"],
    [TRIP_KEY.toUpperCase(), "Explore a trip"],
    [keyOf("newLine"), "Make a new line"],
    [keyOf("editStops"), "Edit the selected line's stops"],
    [keyOf("construct"), "Construct blueprints"],
    ["Space", "Pause or carry on"],
    ["Ctrl Z, Ctrl Y", "Undo or redo blueprint edits"],
    ["/, Ctrl K", "Search"],
    ["Esc", "Back out one step, then clear the selection"],
  ]],
  ["Panels", [
    ["Double click a fare node", "Remove it"],
    ["Double click a name", "Rename it"],
    ["Double click a list row", "Find it on the map"],
    ["Shift or Ctrl click −, +", "Step trains by 5 or 20"],
  ]],
];

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
      const t = e.target as Element;
      if (!ref.current?.contains(t) && !t.closest?.(".tab.help")) helpOpen.value = false;
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
    <div class="help-panel" ref={ref} role="dialog" aria-label="Controls">
      {SECTIONS.map(([head, rows]) => (
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
