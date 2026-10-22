// UI state: which tab, what is selected, display settings, theme, build tool. Main thread only;
// none of it is in the save. Settings and theme are remembered per browser (localStorage), which
// may be unavailable (private window), so every read and write is guarded.

import { computed, effect, signal } from "@preact/signals";
import { AVERAGE_DAY, DEMAND_DAY_PHRASE, type DemandDay } from "./days";
import { DEFAULT_FONT, FONT_KEYS, fontStack, loadFont, type FontKey } from "./fonts";
import { DEFAULT_RAMP, setBubbleColours, type BubbleScheme } from "./palette";
import type { PurposeFilter } from "./purposes";
import { LEVELS, type Level, type Selection } from "./types";

export type Tab = "build" | "lines" | "stations" | "money" | "cities" | "settings";
/** Tabs a remembered choice can name: the City tab is gone (T-220), so a browser that last had it
 * open starts on Lines. */
const TABS: Tab[] = ["build", "lines", "stations", "money", "cities", "settings"];
export type Theme = "pink" | "blue" | "green";
export const THEMES: Theme[] = ["pink", "blue", "green"];
/** The light or dark theme (T-039); the colours above apply to either. Map colours: game/shades.ts. */
export type Shade = "light" | "dark";
export const SHADE_NAMES: Shade[] = ["light", "dark"];
/** names used before 2026-10-09, so a remembered choice survives the rename */
const OLD_THEME: Record<string, Theme> = { cream: "pink", sky: "blue", matcha: "green" };

export type StationUseMode = "none" | "origins" | "transfers" | "total";

export interface DisplaySettings {
  /** Track display: plain lines, traffic-weighted lines, height, or geometry speed limit. */
  trackColour: "line" | "traffic" | "height" | "speed";
  trains: boolean;
  stationNames: boolean;
  capacity: boolean;
  /** T-673: the marks on flat crossings and junctions (ramps, portals and refused crossings stay) */
  flatMarks: boolean;
  basemapLabels: boolean;
  /** T-682: faint real-world rail lines from noritetsu's tiles, under the network (map/realNetwork.ts) */
  realNetwork: boolean;
  /** station use circles (T-482): translucent circles behind platforms, area by the riders a day
   * starting a rail trip there, changing lines there, or both (all boardings); or none */
  stationUse: StationUseMode;
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
  /** T-686: sound at stations (a chime for an arriving train, pops for riders getting on and off),
   * from the zoom where platforms are drawn; the volume, 0-1 */
  sound: boolean;
  soundVolume: number;
  /** T-686: faces rising from a platform as riders get off, from the same zoom */
  faces: boolean;
  /** the commuter dot map, coloured by how people get to work */
  commuters: boolean;
  /** the dot map's end of the commute: where commuters live, or where they work */
  commuterEnd: "home" | "work";
  /** commuter bubbles' size, as a multiple of their width (T-099; 1 = BUBBLE_DENSITY's sizes) */
  bubbleSize: number;
  /** bubbles' opacity as a multiple of the built alpha (T-470); the default is 10% under it */
  bubbleOpacity: number;
  /** T-795: how fine the bubbles are, in steps of one aggregation level: 0 groups trips as before
   * the setting existed (res 8 from zoom 11.8 in), 1 is a level finer (the default), and the finest
   * res 9 is never passed (map/demandOverlay.ts `bubbleLevel`) */
  bubbleDetail: number;
  /** T-804: bubbles coloured by the mix of mode colours, or by a ramp over the train share from
   * `rampFrom` at 0% to `rampTo` at 100% (hex) */
  bubbleColours: BubbleScheme;
  rampFrom: string;
  rampTo: string;
  /** T-247: the size of the page's panels and text, 0.8-1.2 (UI_SCALES); the map itself is not scaled */
  uiScale: number;
  /** T-698: the panels' font, a trial (game/fonts.ts); the map's labels stay Zen Maru Gothic */
  font: FontKey;
  /** T-464: the day the demand views show: the average day, a weekday or a weekend day. A stored
   * "today" (T-266's old default) reads as the average. The trains and platforms follow the clock
   * whatever this says. */
  dayShown: "average" | "weekday" | "weekend";
  /** T-381: miles, mph and feet instead of kilometres, km/h and metres (the game itself is metric) */
  usUnits: boolean;
  /** T-381: demand workers by the old rule (half the threads less one, up to 5); read at the next load */
  moreCores: boolean;
  /** T-826: the clock keeps running while the page is hidden (game/clock.ts) */
  runInBackground: boolean;
}

/** T-247: the UI scale setting's steps */
export const UI_SCALES = [0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2];

const DEFAULT_DISPLAY: DisplaySettings = {
  trackColour: "traffic",
  trains: true,
  stationNames: true,
  capacity: true,
  flatMarks: true,
  basemapLabels: true,
  // T-727: faint by default, so the track to snap to is always in sight
  realNetwork: true,
  stationUse: "none",
  // T-192 (Anita): one circle per station complex by default
  complexView: true,
  trainLoad: true,
  trainLoadBasis: "busiest",
  platformCrowds: "patch",
  sound: true,
  soundVolume: 0.6,
  faces: true,
  commuters: false,
  commuterEnd: "home",
  bubbleSize: 1,
  bubbleOpacity: 0.9,
  bubbleDetail: 1,
  bubbleColours: "ramp",
  rampFrom: DEFAULT_RAMP.from,
  rampTo: DEFAULT_RAMP.to,
  uiScale: 0.9,
  font: DEFAULT_FONT,
  dayShown: "average",
  usUnits: false,
  moreCores: false,
  runInBackground: false,
};

const STORE = "trainworld-ui";
/** The dock's width and its tab area's height, px, as the player dragged them (T-085); null =
 * the stylesheet's default. */
export interface DockLayout {
  dockW: number | null;
  paneH: number | null;
}

/** The stored settings' version: 2 since complex view became the default (T-192), 3 since Nunito and
 * 90% did (T-703), 4 since the real network did (T-727), 5 since the ramp went red to blue (T-812), 6 since Train share became the bubble colouring (T-833). Settings stored before a version had
 * saved the old default with everything else, so that setting is dropped once and the new default applies. */
const VERSION = 6;

function load(): Partial<{ tab: Tab; theme: Theme; shade: Shade; display: Partial<DisplaySettings>; dock: DockLayout; build: Partial<{ level: Level; tracks: 1 | 2 | 4; hsr: boolean; keep: boolean; snap: boolean; design: number }>; v: number }> {
  try {
    return JSON.parse(localStorage.getItem(STORE) || "{}");
  } catch {
    return {};
  }
}
const saved = load();
// Migrate the former Line + separate width checkbox into the four-way display choice.
const oldDisplay = saved.display as (Partial<DisplaySettings> & { lineLoad?: boolean; intercity?: boolean; stationRiders?: boolean }) | undefined;
// `intercity` (hide the longer-trip bubbles) is no longer a setting (T-222), nor `stationRiders`
// (the demand circles, T-482): dropped from storage
const { lineLoad: legacyLineLoad, intercity: _intercity, stationRiders: _stationRiders, ...migratedDisplay } = oldDisplay ?? {};
if ((saved.v ?? 1) < 2) delete migratedDisplay.complexView;
// T-703: the font and size defaults changed, and settings stored before had saved the old ones with
// everything else, so both are dropped once
if ((saved.v ?? 1) < 3) delete migratedDisplay.font, delete migratedDisplay.uiScale;
// T-727: the real network became a default (faint), and the old default (off) was stored
if ((saved.v ?? 1) < 4) delete migratedDisplay.realNetwork;
// T-812: the ramp's default went from pink-green to the mode red and blue; the old default was stored
if ((saved.v ?? 1) < 5) delete migratedDisplay.rampFrom, delete migratedDisplay.rampTo;
// T-833: Train share became the default colouring, and the old default (Mode mix) was stored
if ((saved.v ?? 1) < 6) delete migratedDisplay.bubbleColours;
const trackColour = oldDisplay?.trackColour === "line" && legacyLineLoad !== undefined
  ? legacyLineLoad === false ? "line" : "traffic"
  : oldDisplay?.trackColour ?? DEFAULT_DISPLAY.trackColour;

export const tab = signal<Tab>(saved.tab && TABS.includes(saved.tab) ? saved.tab : "lines");
export const selection = signal<Selection>(null);
export const display = signal<DisplaySettings>({ ...DEFAULT_DISPLAY, ...migratedDisplay, trackColour });
// bubble colours are module state in palette.ts (the bubble worker has its own); views that colour
// bubbles read the setting and call this again before building, so the order of effects is moot
effect(() => setBubbleColours(display.value.bubbleColours, display.value.rampFrom, display.value.rampTo));
export const dockLayout = signal<DockLayout>({ dockW: saved.dock?.dockW ?? null, paneH: saved.dock?.paneH ?? null });
const savedTheme = OLD_THEME[saved.theme as string] ?? saved.theme;
export const theme = signal<Theme>(THEMES.includes(savedTheme as Theme) ? (savedTheme as Theme) : "pink");
export const shade = signal<Shade>(saved.shade === "dark" ? "dark" : "light");

/** What a click on the map does: select (pick), draw track, place a station, add stops to a line,
 * remove what is clicked (T-096), pick a trip's two ends (T-195). */
export type Tool = "select" | "track" | "station" | "line" | "delete" | "trip" | "blueprint";
export const tool = signal<Tool>("select");
/** The Build tab's tools: while one is on, the demand views draw beneath the network (T-526). */
export const isBuildTool = (t: Tool) => t === "track" || t === "station" || t === "delete" || t === "blueprint";
/** A question waiting in a centered modal (T-096/T-113: removing something
 * constructed): `run` on the yes button, nothing on the no button or Esc. `lines`: a row each
 * under the text, after the line's bullet (T-859: what a removal does to lines). */
export const ask = signal<{ text: string; yes: string; no: string; run: () => void; lines?: { line: number; text: string }[] } | null>(null);
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
export const lineSort = signal<ListSort<"name" | "city" | "riders" | "money" | "trains" | "near">>({ key: "name", dir: 1 });
export const stationSort = signal<ListSort<"name" | "riders" | "near">>({ key: "riders", dir: -1 });
/** The Money tab's Cities section (T-550): by name, train share, train rides a day or money a day. */
export const citySort = signal<ListSort<"name" | "train" | "rides" | "money" | "points">>({ key: "rides", dir: -1 });
/** Platform length for new stations, m. */
/** The map's demand drawn for one trip purpose only, or all (T-158). Not remembered: a page
 * load starts on all purposes, so nobody returns to a map that silently leaves riders out. */
export const purposeFilter = signal<PurposeFilter>("all");
export const buildPlatform = signal(200);
/** T-468: the three Build tab track settings below are remembered per browser with the rest of the
 * UI state; anything unreadable falls back to the defaults. */
export const buildLevel = signal<Level>(LEVELS.includes(saved.build?.level as Level) ? (saved.build!.level as Level) : 0);
/** Track the drawing tool lays: double or quad (two doubles side by side, T-255). Single track is
 * not offered (T-733), so a remembered 1 reads as double. */
export const trackCount = signal<2 | 4>(saved.build?.tracks === 4 ? 4 : 2);
/** T-445: the track tool starts each segment where the last ended. The Build tab's button and
 * Caps Lock set it; Shift continues a run for one click without it. */
export const keepPlacing = signal(saved.build?.keep === true);
/** T-727: the track tool snaps to the real network (noritetsu's lines); while it does, the layer is
 * drawn at full strength */
export const snapReal = signal(saved.build?.snap === true);
/** T-817: the speed the track tool lays curves out for, km/h (`TrackApi::set_design_speed`). */
export const DESIGN_SPEEDS = [160, 200, 250, 320] as const;
export type DesignSpeed = (typeof DESIGN_SPEEDS)[number];
export const designSpeed = signal<DesignSpeed>(DESIGN_SPEEDS.includes(saved.build?.design as DesignSpeed) ? (saved.build!.design as DesignSpeed) : 160);
/** T-817: the track tool lays high-speed track, always double (`trackCount` is 2 meanwhile). */
export const highSpeed = signal(saved.build?.hsr === true && trackCount.peek() === 2);
/** The design speed before High speed was picked, put back when Double or Quad is picked again. */
let designBeforeHsr: DesignSpeed | null = null;
export type TrackType = 2 | 4 | "hsr";
export const trackType = computed<TrackType>(() => (highSpeed.value ? "hsr" : trackCount.value));
/** The Build tab's Double / Quad / High speed choice. High speed sets the design speed to its top
 * (the player can step it down); leaving it puts back the speed there was before. */
export function setTrackType(t: TrackType) {
  const was = highSpeed.peek();
  if (t === "hsr") {
    if (was) return;
    designBeforeHsr = designSpeed.peek();
    trackCount.value = 2;
    highSpeed.value = true;
    designSpeed.value = DESIGN_SPEEDS[DESIGN_SPEEDS.length - 1];
  } else {
    trackCount.value = t;
    highSpeed.value = false;
    if (was && designBeforeHsr !== null) designSpeed.value = designBeforeHsr;
    designBeforeHsr = null;
  }
}
/** the real network is being snapped to: the switch on and the track tool in hand */
export const snapping = computed(() => snapReal.value && tool.value === "track");

/** The day every demand view shows (T-260, T-464): the setting, the average day by default. */
export const shownDay = computed<DemandDay>(() => {
  const s = display.value.dayShown;
  return s === "weekday" ? 0 : s === "weekend" ? 1 : AVERAGE_DAY;
});
/** The shown day as a hover ends a figure: "per weekday", "per weekend day", "per day". */
export const shownDayPhrase = computed(() => DEMAND_DAY_PHRASE[shownDay.value]);

export function setDisplay<K extends keyof DisplaySettings>(k: K, v: DisplaySettings[K]) {
  display.value = { ...display.value, [k]: v };
}

effect(() => {
  document.documentElement.dataset.theme = theme.value;
  document.documentElement.dataset.shade = shade.value;
});
/** T-247: the UI scale as a CSS variable; style.css zooms the panels and sizes the map's tags by it */
export const uiScale = computed(() => {
  const s = Number(display.value.uiScale);
  return Number.isFinite(s) ? Math.min(1.2, Math.max(0.8, s)) : 1;
});
effect(() => {
  document.documentElement.style.setProperty("--ui", String(uiScale.value));
});
/** T-698: the panels' font as the `--font` variable. Text measured in the old font (station names,
 * capacity tags) is measured again by the "tw-fontchange" event once the new one has loaded. */
effect(() => {
  const k = FONT_KEYS.includes(display.value.font) ? display.value.font : DEFAULT_FONT;
  if (k === DEFAULT_FONT) document.documentElement.style.removeProperty("--font");
  else document.documentElement.style.setProperty("--font", fontStack(k));
  void loadFont(k).then(() => document.dispatchEvent?.(new Event("tw-fontchange")));
});
effect(() => {
  const s = { tab: tab.value, theme: theme.value, shade: shade.value, display: display.value, dock: dockLayout.value, build: { level: buildLevel.value, tracks: trackCount.value, hsr: highSpeed.value, keep: keepPlacing.value, snap: snapReal.value, design: designSpeed.value }, v: VERSION };
  try {
    localStorage.setItem(STORE, JSON.stringify(s));
  } catch {
    /* storage blocked */
  }
});
