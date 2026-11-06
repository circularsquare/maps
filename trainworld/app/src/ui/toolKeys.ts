// The build tools and the keys that pick things (main.ts, HelpPanel.tsx, the Dock, the Build and Settings tabs).

import type { JSX } from "preact";
import { LEVELS, type Level } from "../game/types";
import { tool, type Tab, type Tool } from "../game/ui";
import { live } from "../map/live";
import { Icon } from "./widgets";
import { t, type MessageKey } from "../i18n";

type ToolEntry = [Tool, string, () => JSX.Element, string, string];
/** A tool's entry whose label (index 1) and hover (index 4) are read through `t` each time they are
 * used, so a list kept at module level still follows the language. */
function toolEntry(id: Tool, icon: () => JSX.Element, key: string, label: MessageKey, tip: MessageKey): ToolEntry {
  const e: ToolEntry = [id, "", icon, key, ""];
  Object.defineProperty(e, 1, { get: () => t(label), enumerable: true });
  Object.defineProperty(e, 4, { get: () => t(tip), enumerable: true });
  return e;
}

/** The build tools: label, icon, number key and hover (Esc goes back to Select from anywhere, main.ts).
 * T-1834: back on 1 to 4 (T-1819 had moved them to letters; Anita wants 1234 for track). */
export const TOOLS: ToolEntry[] = [
  toolEntry("track", Icon.track, "1", "toolKeys.track", "toolKeys.trackTip"),
  toolEntry("station", Icon.station, "2", "toolKeys.station", "toolKeys.stationTip"),
  toolEntry("delete", Icon.bin, "3", "toolKeys.delete", "toolKeys.deleteTip"),
  toolEntry("blueprint", Icon.marquee, "4", "toolKeys.select", "toolKeys.selectTip"),
];

/** The tool each number key picks (T-096). */
export const TOOL_KEYS: Record<string, Tool> = Object.fromEntries(TOOLS.map(([id, , , key]) => [key, id]));
/** The trip explorer's key (T-195, a letter since T-444); unlike the build tools it leaves the tab as
 * it is. Compared lowercase, so Caps Lock does not matter. */
export const TRIP_KEY = "t";
/** T-1464: a new bus line, from any tab (handled by map/busDraw.ts, before main.ts's keys). U since T-1819. */
export const BUS_KEY = "u";
/** T-2194: a new ferry line, from any tab (map/ferryDraw.ts, as the bus key). */
export const FERRY_KEY = "o";
/** T-2338: cycles the demand bubbles None, Homes, Destinations, Total, from any tab. */
export const DEMAND_KEY = "d";
/** T-2338: the Select tool's enable or disable of the picked blueprints (map/blueprintBox.ts). */
export const SWITCH_PICK_KEY = "e";
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

/** T-1701: Shift+1 to Shift+7 set the drawing elevation, in the order of the elevation row (+3 down
 * to -3). Matched by key code, since Shift changes the character on some layouts. Digit row only: Shift
 * turns the numpad into cursor keys. */
export const levelOfKeyCode = (code: string): Level | null => /^Digit[1-7]$/.test(code) ? LEVELS[Number(code[5]) - 1] : null;
/** the key shown in an elevation button's hover */
export const levelKeyName = (l: Level) => `Shift+${LEVELS.indexOf(l) + 1}`; // i18n-ignore (key name)

/** Pick a build tool; the one in use again, or Select, goes back to selecting. */
export function pickTool(v: Tool) {
  const cur = tool.peek();
  live.tools?.exit();
  if (cur !== v && v !== "select") tool.value = v;
}
