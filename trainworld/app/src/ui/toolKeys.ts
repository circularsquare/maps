// The build tools and the keys that pick things (main.ts, HelpPanel.tsx, the Build and Settings tabs).

import type { JSX } from "preact";
import { LEVELS, type Level } from "../game/types";
import { tool, type Tool } from "../game/ui";
import { live } from "../map/live";
import { Icon } from "./widgets";

/** The build tools: label, icon, number key and hover (Esc goes back to Select from anywhere, main.ts). */
export const TOOLS: [Tool, string, () => JSX.Element, string, string][] = [
  ["track", "Track", Icon.track, "1", "Draw track"],
  ["station", "Station", Icon.station, "2", "Place stations"],
  ["delete", "Delete", Icon.bin, "3", "Delete track, stations and flyovers"],
  ["blueprint", "Select", Icon.marquee, "5", "Drag a box round track and stations to select several\nDouble click track to select it out to the junctions\nShift+click track to add the track on the way there"],
];

/** The tool each number key picks (T-096). */
export const TOOL_KEYS: Record<string, Tool> = Object.fromEntries(TOOLS.map(([id, , , key]) => [key, id]));
/** The trip explorer's key (T-195, a letter since T-444: the numbers are for building); unlike the
 * build tools it leaves the tab as it is. Compared lowercase, so Caps Lock does not matter. */
export const TRIP_KEY = "t";
/** T-1464: a new bus line, from any tab (handled by map/busDraw.ts, before main.ts's keys). */
export const BUS_KEY = "b";
/** T-1491: hides the interface for screenshots (ui/hideUi.ts). */
export const HIDE_UI_KEY = "F2";
/** T-670: letter keys for new line, edit stops (the selected line's) and construct blueprints, from any tab;
 * A (T-1417) adds every station the selected line passes. */
export const ACTION_KEYS: Record<string, "newLine" | "editStops" | "construct" | "addPassed"> = { n: "newLine", l: "editStops", c: "construct", a: "addPassed" };

/** T-1701: Shift+1 to Shift+7 set the drawing elevation, in the order of the elevation row (+3 down
 * to -3). Matched by key code, since Shift changes the character on some layouts. */
export const levelOfKeyCode = (code: string): Level | null => /^Digit[1-7]$/.test(code) ? LEVELS[Number(code[5]) - 1] : null;
/** the key shown in an elevation button's hover */
export const levelKeyName = (l: Level) => `Shift+${LEVELS.indexOf(l) + 1}`;

/** Pick a build tool; the one in use again, or Select, goes back to selecting. */
export function pickTool(v: Tool) {
  const t = tool.peek();
  live.tools?.exit();
  if (t !== v && v !== "select") tool.value = v;
}
