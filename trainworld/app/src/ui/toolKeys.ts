// The build tools and the keys that pick things (main.ts, HelpPanel.tsx, the Dock, the Build and Settings tabs).

import type { JSX } from "preact";
import { LEVELS, type Level } from "../game/types";
import { tool, type Tab, type Tool } from "../game/ui";
import { live } from "../map/live";
import { Icon } from "./widgets";

/** The build tools: label, icon, letter key and hover (Esc goes back to Select from anywhere, main.ts).
 * T-1819: letters, since the number keys pick the elevation; R rail, P place stations, X cross out,
 * M marquee. T is the trip explorer's. */
export const TOOLS: [Tool, string, () => JSX.Element, string, string][] = [
  ["track", "Track", Icon.track, "R", "Draw track"],
  ["station", "Station", Icon.station, "P", "Place stations"],
  ["delete", "Delete", Icon.bin, "X", "Delete track, stations and flyovers"],
  ["blueprint", "Select", Icon.marquee, "M", "Drag a box round track and stations to select several\nDouble click track to select it out to the junctions\nShift+click track to add the track on the way there"],
];

/** The tool each letter key picks (T-096, letters since T-1819), keyed lowercase. */
export const TOOL_KEYS: Record<string, Tool> = Object.fromEntries(TOOLS.map(([id, , , key]) => [key.toLowerCase(), id]));
/** The trip explorer's key (T-195, a letter since T-444); unlike the build tools it leaves the tab as
 * it is. Compared lowercase, so Caps Lock does not matter. */
export const TRIP_KEY = "t";
/** T-1464: a new bus line, from any tab (handled by map/busDraw.ts, before main.ts's keys). U since T-1819. */
export const BUS_KEY = "u";
/** T-1491: hides the interface for screenshots (ui/hideUi.ts). */
export const HIDE_UI_KEY = "F2";
/** T-670: letter keys for new line, edit stops (the selected line's) and construct blueprints, from any tab;
 * G (T-1417, moved from A by T-1833) adds every station the selected line passes (gaps). */
export const ACTION_KEYS: Record<string, "newLine" | "editStops" | "construct" | "addPassed"> = { n: "newLine", s: "editStops", c: "construct", g: "addPassed" };

/** T-1819: a letter opens each dock tab, from any tab; C stays Construct's, so Cities has Y (Anita's pick,
 * T-1833: A opens Stations and S is Edit stops). */
export const TAB_KEYS: Record<string, Tab> = { b: "build", l: "lines", a: "stations", f: "money", y: "cities" };
/** the key shown in a tab's hover */
export const tabKeyName = (t: Tab) => Object.keys(TAB_KEYS).find((k) => TAB_KEYS[k] === t)?.toUpperCase();

/** T-1819: the number keys 1 to 7 set the drawing elevation (T-1701), in the order of the elevation
 * row (+3 down to -3). Matched by key code, since the digit row is shifted on some layouts. */
export const levelOfKeyCode = (code: string): Level | null => /^(Digit|Numpad)[1-7]$/.test(code) ? LEVELS[Number(code[code.length - 1]) - 1] : null;
/** the key shown in an elevation button's hover */
export const levelKeyName = (l: Level) => `${LEVELS.indexOf(l) + 1}`;

/** Pick a build tool; the one in use again, or Select, goes back to selecting. */
export function pickTool(v: Tool) {
  const t = tool.peek();
  live.tools?.exit();
  if (t !== v && v !== "select") tool.value = v;
}
