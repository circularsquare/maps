// UI state: which tab, what is selected, display settings, theme, build tool. Main thread only;
// none of it is in the save. Settings and theme are remembered per browser (localStorage), which
// may be unavailable (private window), so every read and write is guarded.

import { effect, signal } from "@preact/signals";
import type { Level, Selection } from "./types";

export type Tab = "build" | "lines" | "stations" | "money" | "city" | "settings";
export type Theme = "pink" | "blue" | "green";
export const THEMES: Theme[] = ["pink", "blue", "green"];
/** names used before 2026-10-09, so a remembered choice survives the rename */
const OLD_THEME: Record<string, Theme> = { cream: "pink", sky: "blue", matcha: "green" };

export interface DisplaySettings {
  /** Track display: plain lines, traffic-weighted lines, height, or geometry speed limit. */
  trackColour: "line" | "traffic" | "height" | "speed";
  trains: boolean;
  stationNames: boolean;
  capacity: boolean;
  basemapLabels: boolean;
  /** station circles sized by riders a day */
  stationRiders: boolean;
  /** trains filled by how full they are */
  trainLoad: boolean;
  /** Period average or its busiest hour; also selects the train inspector's gauge. */
  trainLoadBasis: "average" | "busiest";
  /** the commuter dot map, coloured by how people get to work */
  commuters: boolean;
  /** the dot map's end of the commute: where commuters live, or where they work */
  commuterEnd: "home" | "work";
  /** commuter bubbles' size, as a multiple of their width (T-099; 1 = BUBBLE_DENSITY's sizes) */
  bubbleSize: number;
}

const DEFAULT_DISPLAY: DisplaySettings = {
  trackColour: "traffic",
  trains: true,
  stationNames: true,
  capacity: true,
  basemapLabels: true,
  stationRiders: true,
  trainLoad: true,
  trainLoadBasis: "busiest",
  commuters: false,
  commuterEnd: "home",
  bubbleSize: 1,
};

const STORE = "trainworld-ui";
/** The dock's width and its tab area's height, px, as the player dragged them (T-085); null =
 * the stylesheet's default. */
export interface DockLayout {
  dockW: number | null;
  paneH: number | null;
}

function load(): Partial<{ tab: Tab; theme: Theme; display: Partial<DisplaySettings>; dock: DockLayout }> {
  try {
    return JSON.parse(localStorage.getItem(STORE) || "{}");
  } catch {
    return {};
  }
}
const saved = load();
// Migrate the former Line + separate width checkbox into the four-way display choice.
const oldDisplay = saved.display as (Partial<DisplaySettings> & { lineLoad?: boolean }) | undefined;
const { lineLoad: legacyLineLoad, ...migratedDisplay } = oldDisplay ?? {};
const trackColour = oldDisplay?.trackColour === "line" && legacyLineLoad !== undefined
  ? legacyLineLoad === false ? "line" : "traffic"
  : oldDisplay?.trackColour ?? DEFAULT_DISPLAY.trackColour;

export const tab = signal<Tab>(saved.tab ?? "lines");
export const selection = signal<Selection>(null);
export const display = signal<DisplaySettings>({ ...DEFAULT_DISPLAY, ...migratedDisplay, trackColour });
export const dockLayout = signal<DockLayout>({ dockW: saved.dock?.dockW ?? null, paneH: saved.dock?.paneH ?? null });
const savedTheme = OLD_THEME[saved.theme as string] ?? saved.theme;
export const theme = signal<Theme>(THEMES.includes(savedTheme as Theme) ? (savedTheme as Theme) : "pink");

/** What a click on the map does: select (pick), draw track, place a station, add stops to a line,
 * remove what is clicked (T-096). */
export type Tool = "select" | "track" | "station" | "line" | "delete";
export const tool = signal<Tool>("select");
/** A question waiting in the hint area above the bottom bar (T-096: removing something
 * constructed): `run` on the yes button, nothing on the no button or Esc. */
export const ask = signal<{ text: string; yes: string; no: string; run: () => void } | null>(null);
/** The line the line tool is adding stops to (null until its first two stops make it). */
export const lineDraft = signal<{ line: number | null; stops: number[] }>({ line: null, stops: [] });
/** Platform length for new stations, m. */
export const buildPlatform = signal(200);
export const buildLevel = signal<Level>(0);
export const singleTrack = signal(false);

export function setDisplay<K extends keyof DisplaySettings>(k: K, v: DisplaySettings[K]) {
  display.value = { ...display.value, [k]: v };
}

effect(() => {
  document.documentElement.dataset.theme = theme.value;
});
effect(() => {
  const s = { tab: tab.value, theme: theme.value, display: display.value, dock: dockLayout.value };
  try {
    localStorage.setItem(STORE, JSON.stringify(s));
  } catch {
    /* storage blocked */
  }
});
