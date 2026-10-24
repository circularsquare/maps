// The build tools and the keys that pick things (main.ts, HelpPanel.tsx, the Build and Settings tabs).

import type { JSX } from "preact";
import { tool, type Tool } from "../game/ui";
import { live } from "../map/live";
import { Icon } from "./widgets";

/** The build tools: label, icon, number key and hover (Esc goes back to Select from anywhere, main.ts). */
export const TOOLS: [Tool, string, () => JSX.Element, string, string][] = [
  ["track", "Track", Icon.track, "1", "Draw track"],
  ["station", "Station", Icon.station, "2", "Place stations"],
  ["delete", "Delete", Icon.bin, "3", "Delete track, stations and flyovers"],
  ["blueprint", "Select", Icon.marquee, "5", "Drag a box round track and stations to select several"],
];

/** The tool each number key picks (T-096). */
export const TOOL_KEYS: Record<string, Tool> = Object.fromEntries(TOOLS.map(([id, , , key]) => [key, id]));
/** The trip explorer's key (T-195, a letter since T-444: the numbers are for building); unlike the
 * build tools it leaves the tab as it is. Compared lowercase, so Caps Lock does not matter. */
export const TRIP_KEY = "t";
/** T-670: letter keys for new line, edit stops (the selected line's) and construct blueprints, from any tab. */
export const ACTION_KEYS: Record<string, "newLine" | "editStops" | "construct"> = { n: "newLine", l: "editStops", c: "construct" };

/** Pick a build tool; the one in use again, or Select, goes back to selecting. */
export function pickTool(v: Tool) {
  const t = tool.peek();
  live.tools?.exit();
  if (t !== v && v !== "select") tool.value = v;
}
