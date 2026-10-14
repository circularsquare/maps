// UI state: which tab, what is selected, display settings, theme, build tool. Main thread only;
// none of it is in the save. Settings and theme are remembered per browser (localStorage), which
// may be unavailable (private window), so every read and write is guarded.

import { computed, effect, signal } from "@preact/signals";
import { currentDay, type DayKind } from "./days";
import type { PurposeFilter } from "./purposes";
import type { Level, Selection } from "./types";

export type Tab = "build" | "lines" | "stations" | "money" | "settings";
/** Tabs a remembered choice can name: the City tab is gone (T-220), so a browser that last had it
 * open starts on Lines. */
const TABS: Tab[] = ["build", "lines", "stations", "money", "settings"];
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
  /** explicit analytical view: translucent circles behind platforms, area by daily entries/exits */
  stationRiders: boolean;
  /** station complexes (T-146): the station inspector and the demand circles take each complex
   * (stations whose platforms nearly touch, game/complexes.ts) as one */
  complexView: boolean;
  /** trains filled by how full they are: each train's own load from the boarding account (T-164) */
  trainLoad: boolean;
  /** Period average or its busiest hour, for a line the boarding account has no riders for yet.
   * No longer a setting (T-164): each train has its own load. */
  trainLoadBasis: "average" | "busiest";
  /** riders waiting on platforms (T-164), drawn as each platform filling up (T-219), or not
   * ("off"). "dots" is the T-164 option T-219 dropped, kept in older browsers' settings: on. */
  platformCrowds: "patch" | "dots" | "off";
  /** the commuter dot map, coloured by how people get to work */
  commuters: boolean;
  /** the dot map's end of the commute: where commuters live, or where they work */
  commuterEnd: "home" | "work";
  /** commuter bubbles' size, as a multiple of their width (T-099; 1 = BUBBLE_DENSITY's sizes) */
  bubbleSize: number;
  /** T-247: the size of the page's panels and text, 0.8-1.2 (UI_SCALES); the map itself is not scaled */
  uiScale: number;
  /** T-266: the day the demand views show: the clock's own (`today`), or always a weekday or a
   * weekend day. The trains and platforms follow the clock whatever this says. */
  dayShown: "today" | "weekday" | "weekend";
}

/** T-247: the UI scale setting's steps */
export const UI_SCALES = [0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2];

const DEFAULT_DISPLAY: DisplaySettings = {
  trackColour: "traffic",
  trains: true,
  stationNames: true,
  capacity: true,
  basemapLabels: true,
  stationRiders: false,
  // T-192 (Anita): one circle per station complex by default
  complexView: true,
  trainLoad: true,
  trainLoadBasis: "busiest",
  platformCrowds: "patch",
  commuters: false,
  commuterEnd: "home",
  bubbleSize: 1,
  uiScale: 1,
  dayShown: "today",
};

const STORE = "trainworld-ui";
/** The dock's width and its tab area's height, px, as the player dragged them (T-085); null =
 * the stylesheet's default. */
export interface DockLayout {
  dockW: number | null;
  paneH: number | null;
}

/** The stored settings' version: 2 since complex view became the default (T-192). Settings stored
 * before it had saved the old default with everything else, so their complex view is dropped once
 * and the new default applies. */
const VERSION = 2;

function load(): Partial<{ tab: Tab; theme: Theme; display: Partial<DisplaySettings>; dock: DockLayout; v: number }> {
  try {
    return JSON.parse(localStorage.getItem(STORE) || "{}");
  } catch {
    return {};
  }
}
const saved = load();
// Migrate the former Line + separate width checkbox into the four-way display choice.
const oldDisplay = saved.display as (Partial<DisplaySettings> & { lineLoad?: boolean; intercity?: boolean }) | undefined;
// `intercity` (hide the longer-trip bubbles) is no longer a setting (T-222): dropped from storage
const { lineLoad: legacyLineLoad, intercity: _intercity, ...migratedDisplay } = oldDisplay ?? {};
if ((saved.v ?? 1) < 2) delete migratedDisplay.complexView;
const trackColour = oldDisplay?.trackColour === "line" && legacyLineLoad !== undefined
  ? legacyLineLoad === false ? "line" : "traffic"
  : oldDisplay?.trackColour ?? DEFAULT_DISPLAY.trackColour;

export const tab = signal<Tab>(saved.tab && TABS.includes(saved.tab) ? saved.tab : "lines");
export const selection = signal<Selection>(null);
export const display = signal<DisplaySettings>({ ...DEFAULT_DISPLAY, ...migratedDisplay, trackColour });
export const dockLayout = signal<DockLayout>({ dockW: saved.dock?.dockW ?? null, paneH: saved.dock?.paneH ?? null });
const savedTheme = OLD_THEME[saved.theme as string] ?? saved.theme;
export const theme = signal<Theme>(THEMES.includes(savedTheme as Theme) ? (savedTheme as Theme) : "pink");

/** What a click on the map does: select (pick), draw track, place a station, add stops to a line,
 * remove what is clicked (T-096), pick a trip's two ends (T-195). */
export type Tool = "select" | "track" | "station" | "line" | "delete" | "trip" | "blueprint";
export const tool = signal<Tool>("select");
/** A question waiting in a centered modal (T-096/T-113: removing something
 * constructed): `run` on the yes button, nothing on the no button or Esc. */
export const ask = signal<{ text: string; yes: string; no: string; run: () => void } | null>(null);
/** The selected line whose stops the map tool is editing. */
export const lineDraft = signal<{ line: number | null; stops: number[] }>({ line: null, stops: [] });
/** Stop editing is on (T-233): the line tool with a line to edit. The map then hides the demand
 * views, dims the other lines and puts a toggle marker on every station (map/stopMarkers.ts). */
export const stopEditing = computed(() => tool.value === "line" && lineDraft.value.line !== null);
/** A newly created line opens its inspector with the default name selected for replacement. */
export const renameLinePrompt = signal<string | null>(null);
/** How the Lines and Stations lists are sorted: the column clicked and its direction (T-310). Kept
 * while the page is open, as the list sort always was. */
export interface ListSort<K extends string> { key: K; dir: 1 | -1 }
export const lineSort = signal<ListSort<"name" | "riders" | "trains" | "near">>({ key: "name", dir: 1 });
export const stationSort = signal<ListSort<"name" | "riders" | "near">>({ key: "riders", dir: -1 });
/** Platform length for new stations, m. */
/** The map's demand drawn for one trip purpose only, or all (T-158). Not remembered: a page
 * load starts on all purposes, so nobody returns to a map that silently leaves riders out. */
export const purposeFilter = signal<PurposeFilter>("all");
export const buildPlatform = signal(200);
export const buildLevel = signal<Level>(0);
/** Track the drawing tool lays: single, double, or quad (two doubles side by side, T-255). */
export const trackCount = signal<1 | 2 | 4>(2);

/** The kind of day every demand view shows (T-260, T-266): the setting, or the clock's day. */
export const shownDay = computed<DayKind>(() => {
  const s = display.value.dayShown;
  return s === "weekday" ? 0 : s === "weekend" ? 1 : currentDay.value;
});

export function setDisplay<K extends keyof DisplaySettings>(k: K, v: DisplaySettings[K]) {
  display.value = { ...display.value, [k]: v };
}

effect(() => {
  document.documentElement.dataset.theme = theme.value;
});
/** T-247: the UI scale as a CSS variable; style.css zooms the panels and sizes the map's tags by it */
export const uiScale = computed(() => {
  const s = Number(display.value.uiScale);
  return Number.isFinite(s) ? Math.min(1.2, Math.max(0.8, s)) : 1;
});
effect(() => {
  document.documentElement.style.setProperty("--ui", String(uiScale.value));
});
effect(() => {
  const s = { tab: tab.value, theme: theme.value, display: display.value, dock: dockLayout.value, v: VERSION };
  try {
    localStorage.setItem(STORE, JSON.stringify(s));
  } catch {
    /* storage blocked */
  }
});
