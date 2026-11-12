// UI state: which tab, what is selected, display settings, theme, build tool. Main thread only;
// none of it is in the save. Settings and theme are remembered per browser (localStorage), which
// may be unavailable (private window), so every read and write is guarded.

import { t } from "../i18n";
import { batch, computed, effect, signal } from "@preact/signals";
import { AVERAGE_DAY, AWAY_DAY, awayNear, BACK_DAY, DEMAND_DAY_ON, DEMAND_DAY_PHRASE, OUT_DAY, rushRun, whenAway, whenBack, whenOut, type DemandDay } from "./days";
import { defaultCurrency, type Currency } from "./currency";
import { loadNunito } from "./fonts";
import { mobileDevice } from "../workers/deviceMemory";
import { DEFAULT_RAMP, setBubbleColours, setBubbleShade, type BubbleScheme } from "./palette";
import type { PurposeFilter } from "./purposes";
import { LEVELS, type BubbleEnd, type Level, type Selection } from "./types";

export type Tab = "build" | "lines" | "stations" | "money" | "cities" | "settings";
/** Tabs a remembered choice can name. */
const TABS: Tab[] = ["build", "lines", "stations", "money", "cities", "settings"];
export type Theme = "pink" | "blue" | "green";
export const THEMES: Theme[] = ["pink", "blue", "green"];
/** The light or dark theme (T-039); the colours above apply to either. Map colours: game/shades.ts. */
export type Shade = "light" | "dark";
export const SHADE_NAMES: Shade[] = ["light", "dark"];
/** names used before 2026-10-09, so a remembered choice survives the rename */
const OLD_THEME: Record<string, Theme> = { cream: "pink", sky: "blue", matcha: "green" };

export type StationUseMode = "none" | "origins" | "transfers" | "total";
export type MapRow = "bubbles" | "stations" | "track";

export interface DisplaySettings {
  /** Track display: plain lines, traffic-weighted lines, height, or geometry speed limit. */
  trackColour: "line" | "traffic" | "height" | "speed" | "profit";
  trains: boolean;
  stationNames: boolean;
  capacity: boolean;
  /** T-1152: the traffic view's red edges where track is at capacity or trains run packed full */
  fullMarks: boolean;
  /** T-673: the marks on flat crossings and junctions (ramps, portals and refused crossings stay) */
  flatMarks: boolean;
  /** T-3048: the triangles and semicircles on ramps (a refused crossing's mark stays) */
  rampMarks: boolean;
  /** T-1886: lines sharing track drawn side by side (T-1799, T-1848), not over each other; off by default */
  lineBands: boolean;
  basemapLabels: boolean;
  /** T-682: faint real-world rail lines from noritetsu's tiles, under the network (map/realNetwork.ts) */
  realNetwork: boolean;
  /** station use circles (T-482): translucent circles behind platforms, area by the riders a day
   * starting a rail trip there, changing lines there, or both (all boardings); or none */
  stationUse: StationUseMode;
  /** T-1448: the map's upper left rows folded down to their icon ("bubbles", "stations", "track") */
  foldedRows: MapRow[];
  /** trains filled by how full they are: each train's own load from the boarding account (T-164) */
  trainLoad: boolean;
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
  /** the bubbles' end of the trips: homes, destinations, or both summed (T-1903) */
  commuterEnd: BubbleEnd;
  /** commuter bubbles' size, as a multiple of their width (T-099; 1 = BUBBLE_DENSITY's sizes) */
  bubbleSize: number;
  /** bubbles' opacity as a multiple of the built alpha (T-470); the default is 30% under it */
  bubbleOpacity: number;
  /** T-1565: the station ridership circles' size, as a multiple of their own (0.5-2; map/stationCircles.ts) */
  stationCircleSize: number;
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
  /** T-464, T-262: the day the demand views show: the average day, a weekday, a Saturday or a
   * Sunday. A stored "today" (T-266's old default) or "weekend" (before T-262) reads as the average.
   * The trains and platforms follow the clock whatever this says. */
  dayShown: "average" | "weekday" | "saturday" | "sunday" | "away" | "out" | "back";
  /** T-381: miles, mph and feet instead of kilometres, km/h and metres (the game itself is metric) */
  usUnits: boolean;
  /** T-2798: the currency money is shown in (game/currency.ts); the game itself counts US$ */
  currency: Currency;
  /** T-381: demand workers by the old rule (half the threads less one, up to 5); read at the next load */
  moreCores: boolean;
  /** T-826: the clock keeps running while the page is hidden (game/clock.ts) */
  runInBackground: boolean;
  /** T-033: the map as the flat map or as a globe (MapLibre's, flat again from zoom 12) */
  projection: "flat" | "globe";
  /** T-3249: satellite imagery under the game instead of the drawn basemap (map/basemap.ts `setSatellite`) */
  satellite: boolean;
  /** T-1248: the drawn width of lines and track as a multiple of their own, 0.5-2 */
  lineWidth: number;
  /** T-1249: the basemap's contrast, 0-1: below 1 everything but water and the land is mixed that
   * much less toward the land colour (map/basemap.ts, map/realNetwork.ts) */
  basemapContrast: number;
  /** T-1270: hills and mountains shaded on the basemap (map/basemap.ts `setHillshade`) */
  hillshade: boolean;
  /** T-1207: the lighter map (fewer basemap layers, pixel ratio 1): on is Low, off is High (map/lite.ts).
   * T-2840: there was an "auto" too; one stored reads as off. */
  lighterMap: "on" | "off";
  /** T-2031: a right-button drag turns the map (map/mapView.ts `followRotation`) */
  rotateMap: boolean;
  /** T-2660: the timetable chart draws other lines' trains where they share its track or cross it */
  chartOtherLines: boolean;
  /** T-2838: a phone or tablet calculates trips between cities too (the market opens); per browser, since it is
   * about the device. Read at each solve, so it applies from the next one. */
  mobileIntercity: boolean;
}

/** T-247: the UI scale setting's steps */
export const UI_SCALES = [0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2];

const DEFAULT_DISPLAY: DisplaySettings = {
  // T-1101: plain lines to start with; Traffic is one click away
  trackColour: "line",
  trains: true,
  stationNames: false,
  capacity: true,
  fullMarks: true,
  flatMarks: true,
  rampMarks: true,
  lineBands: true,
  basemapLabels: true,
  // T-727: faint by default, so the track to snap to is always in sight
  realNetwork: true,
  stationUse: "none",
  foldedRows: [],
  trainLoad: true,
  platformCrowds: "patch",
  sound: true,
  soundVolume: 0.6,
  faces: true,
  commuters: false,
  commuterEnd: "home",
  bubbleSize: 1,
  bubbleOpacity: 0.7,
  stationCircleSize: 1,
  bubbleDetail: 1,
  bubbleColours: "mode",
  rampFrom: DEFAULT_RAMP.from,
  rampTo: DEFAULT_RAMP.to,
  uiScale: 0.9,
  dayShown: "average",
  usUnits: false,
  currency: defaultCurrency(),
  moreCores: false,
  runInBackground: false,
  projection: "flat",
  satellite: false,
  lineWidth: 1,
  basemapContrast: 1,
  hillshade: false,
  lighterMap: "off",
  rotateMap: false,
  chartOtherLines: true,
  mobileIntercity: false,
};

const STORE = "trainworld-ui";
/** The dock's width and its tab area's height, px, as the player dragged them (T-085); null =
 * the stylesheet's default. */
export interface DockLayout {
  dockW: number | null;
  paneH: number | null;
  /** on a phone held upright, the tab area's share of the width beside the inspector (T-1259) */
  paneW?: number | null;
  /** on a phone held upright, the dock's share of the window's height (T-1884) */
  phoneH?: number | null;
}

/** The stored settings' version: 2 since complex view became the default (T-192), 3 since Nunito and
 * 90% did (T-703), 4 since the real network did (T-727), 5 since the ramp went red to blue (T-812), 6 since Train share became the bubble colouring (T-833), 7 since keep placing is on by default (T-942), 8 since unopened zones' dots are light grey (T-976). Settings stored before a version had
 * saved the old default with everything else, so that setting is dropped once and the new default applies; 9 since the ramp's ends are lighter (T-1138); 10 since the track display starts on Line (T-1244); 11 since the station circles start off (T-1585); 12 since lines side by side are on (T-2346); 13 since everyone starts on High map detail (T-2840); 14 since the currency starts from the browser's region (T-2860). */
const VERSION = 14;

function load(): Partial<{ tab: Tab; theme: Theme; shade: Shade; display: Partial<DisplaySettings>; dock: DockLayout; build: Partial<{ level: Level; ramp: boolean; tracks: 1 | 2 | 4; hsr: boolean; tram: boolean; keep: boolean; snap: boolean; streets: boolean; roads: boolean; beside: boolean; design: number; fly: number; advanced: boolean; platform: number; tramPlatform: number }>; guide: boolean; v: number }> {
  try {
    return JSON.parse(localStorage.getItem(STORE) || "{}");
  } catch {
    return {};
  }
}
const saved = load();
const oldDisplay = saved.display as (Partial<DisplaySettings> & { lineLoad?: boolean; intercity?: boolean; stationRiders?: boolean; unbuiltColour?: string; trainLoadBasis?: string; complexView?: boolean; font?: unknown }) | undefined;
// `lineLoad` (the old width checkbox) and `intercity` (hide the longer-trip bubbles) are no longer settings (T-222), nor `stationRiders`
// (the demand circles, T-482), nor `unbuiltColour` (T-1120), nor `trainLoadBasis` (T-164), nor
// `complexView` (one ridership circle per station complex is the only way since T-1449), nor `font`
// (Nunito is the only font, T-1524): dropped from storage
const { lineLoad: _lineLoad, intercity: _intercity, stationRiders: _stationRiders, unbuiltColour: _unbuiltColour, trainLoadBasis: _trainLoadBasis, complexView: _complexView, font: _font, ...migratedDisplay } = oldDisplay ?? {};
// T-703: the size default changed, and settings stored before had saved the old one with
// everything else, so it is dropped once
if ((saved.v ?? 1) < 3) delete migratedDisplay.uiScale;
// T-727: the real network became a default (faint), and the old default (off) was stored
if ((saved.v ?? 1) < 4) delete migratedDisplay.realNetwork;
// T-812: the ramp's default went from pink-green to the mode red and blue; the old default was stored
if ((saved.v ?? 1) < 5) delete migratedDisplay.rampFrom, delete migratedDisplay.rampTo;
// T-833: Train share became the default colouring, and the old default (Mode mix) was stored
if ((saved.v ?? 1) < 6) delete migratedDisplay.bubbleColours;
// T-1138: the ramp's ends went to lighter red and blue; an end still at the old default was stored, not chosen
if ((saved.v ?? 1) < 9) {
  if (migratedDisplay.rampFrom?.toLowerCase() === "#e61e1e") delete migratedDisplay.rampFrom;
  if (migratedDisplay.rampTo?.toLowerCase() === "#1e3ce6") delete migratedDisplay.rampTo;
}
// T-1585: the circles are off to start with; a choice stored before is dropped once
if ((saved.v ?? 1) < 11) delete migratedDisplay.stationUse;
// T-2346: lines side by side are on to start with; the old default (off) was stored, so it is dropped once
if ((saved.v ?? 1) < 12) delete migratedDisplay.lineBands;
// T-2840: Anita wants High for everyone who played before, a stored Low included; choosing Low after this sticks
if ((saved.v ?? 1) < 13) delete migratedDisplay.lighterMap;
// T-2860: the currency now starts from the browser's region; every earlier player has USD stored, chosen or not, so it is dropped once (a stored other currency stays)
if ((saved.v ?? 1) < 14 && migratedDisplay.currency === "USD") delete migratedDisplay.currency;
// T-1244: Traffic, the default until T-1101, was stored with everything else; Elevation and Max speed stay
const trackColour = (saved.v ?? 1) < 10
  ? oldDisplay?.trackColour === "height" || oldDisplay?.trackColour === "speed" ? oldDisplay.trackColour : DEFAULT_DISPLAY.trackColour
  : oldDisplay?.trackColour ?? DEFAULT_DISPLAY.trackColour;

export const tab = signal<Tab>(saved.tab && TABS.includes(saved.tab) ? saved.tab : "lines");
export const selection = signal<Selection>(null);

/** T-2774: `?video=1`, the launch video's view (video.tsx). Its settings hold for that page only and
 * nothing is stored, so the player's own come back on the next ordinary load. */
export const VIDEO = new URLSearchParams(location.search).get("video") === "1";
const VIDEO_DISPLAY: Partial<DisplaySettings> = {
  trackColour: "line", trains: true, stationNames: false, capacity: false, basemapLabels: false, realNetwork: false,
  stationUse: "none", commuters: false, trainLoad: false, platformCrowds: "patch", sound: true, lighterMap: "off",
};

export const display = signal<DisplaySettings>({ ...DEFAULT_DISPLAY, ...migratedDisplay, trackColour, lighterMap: (migratedDisplay.lighterMap as string) === "on" ? "on" : "off", ...(VIDEO ? VIDEO_DISPLAY : {}) });
/** T-2867: the player chose to play without intercity trips after running out of memory: this tab, across its
 * reloads, and gone with it (one bad evening should not take them away for good) */
const MARKET_DROPPED = "tw-no-intercity";
let droppedHere = false;
try { droppedHere = typeof sessionStorage !== "undefined" && sessionStorage.getItem(MARKET_DROPPED) === "1"; } catch { /* storage unavailable */ }
export const marketDropped = () => droppedHere;
export function dropMarket() {
  droppedHere = true;
  try { sessionStorage.setItem(MARKET_DROPPED, "1"); } catch { /* this page only */ }
}
/** A game opened from the start menu without "Load without inter-city trips" has them again (Anita, 2026-11-10) */
export function keepMarket() {
  droppedHere = false;
  try { sessionStorage.removeItem(MARKET_DROPPED); } catch { /* this page only */ }
}
/** T-2802, T-2838: no market (no trips between cities) on a phone or tablet, unless the player turned it on;
 * T-2867: or after `dropMarket` */
export const marketOff = () => droppedHere || (mobileDevice() && !display.peek().mobileIntercity);
// bubble colours are module state in palette.ts (the bubble worker has its own); views that colour
// bubbles read the setting and call this again before building, so the order of effects is moot
effect(() => setBubbleColours(display.value.bubbleColours, display.value.rampFrom, display.value.rampTo));
export const dockLayout = signal<DockLayout>({ dockW: saved.dock?.dockW ?? null, paneH: saved.dock?.paneH ?? null, paneW: saved.dock?.paneW ?? null, phoneH: saved.dock?.phoneH ?? null });
const savedTheme = OLD_THEME[saved.theme as string] ?? saved.theme;
export const theme = signal<Theme>(THEMES.includes(savedTheme as Theme) ? (savedTheme as Theme) : "pink");
export const shade = signal<Shade>(saved.shade === "dark" ? "dark" : "light");
/** T-3249, T-3267: the shade the map's own ink is drawn in: the dark theme's over satellite imagery shown at half strength or more, the panels' own below it (a faded image reads as the theme's background) */
export const SAT_INK_FLIP = 0.5;
export const mapShade = computed<Shade>(() => (display.value.satellite && (shade.value === "dark" || display.value.basemapContrast >= SAT_INK_FLIP) ? "dark" : shade.value));
effect(() => setBubbleShade(mapShade.value));

/** What a click on the map does: select (pick), draw track, place a station, add stops to a line,
 * remove what is clicked (T-096), pick a trip's two ends (T-195). */
export type Tool = "select" | "track" | "station" | "line" | "delete" | "trip" | "blueprint" | "bus" | "ferry" | "building";
export const tool = signal<Tool>("select");
/** T-1207: the map's camera is moving (set by map/overlay.ts); live counts in panels wait for it. */
export const cameraMoving = { now: false };
/** The Build tab's tools: while one is on, the demand views draw beneath the network (T-526). The
 * station building tool (T-2218) starts from a station's upgrade row, but it builds too. */
export const isBuildTool = (t: Tool) => t === "track" || t === "station" || t === "delete" || t === "blueprint" || t === "bus" || t === "ferry" || t === "building";
/** T-3083: starting to build closes the inspector of a thing on the network (line, station, track and
 * the like). City, trip, commuter and achievement views stay, as the demand views stay under build tools. */
const NETWORK_VIEWS = new Set(["line", "group", "busStop", "station", "track", "junction", "train"]);
export function closeNetworkView() {
  if (NETWORK_VIEWS.has(selection.peek()?.kind ?? "")) selection.value = null;
}
/** A question waiting in a centered modal (T-096/T-113: removing something
 * constructed): `run` on the yes button, `stay` (if any) on the no button or Esc. `lines`: a row each
 * under the text, after the line's bullet (T-859: what a removal does to lines). */
export const ask = signal<{ text: string; yes: string; no: string; run: () => void; stay?: () => void; lines?: { line: number; text: string }[] } | null>(null);
/** The selected line whose stops the map tool is editing. */
export const lineDraft = signal<{ line: number | null; stops: number[] }>({ line: null, stops: [] });
/** Stop editing is on (T-233): the line tool with a line to edit. The map then hides the demand
 * views, dims the other lines and puts a toggle marker on every station (map/stopMarkers.ts). */
export const stopEditing = computed(() => tool.value === "line" && lineDraft.value.line !== null);
/** T-3118: the touch Loop button is on: the next tap on a stop, a route point or a port does a right click's work
 * (close, loop back or open a loop there), then it goes off. Off whenever the tool changes. */
export const loopTap = signal(false);
effect(() => { tool.value; lineDraft.value.line; loopTap.value = false; });
/** T-1899: the line whose first stops are being picked, from New line until its stop editing ends,
 * and the train type the stops last gave it: its clicks pick high-speed or ordinary trains by the
 * track and prefer stations no trains stop at. Null once the player sets the type by hand. */
export const newLine = signal<{ line: number; fast: boolean } | null>(null);
/** A newly created line opens its inspector with the default name selected for replacement. */
export const renameLinePrompt = signal<string | null>(null);
/** How the Lines and Stations lists are sorted: the column clicked and its direction (T-310). Kept
 * while the page is open, as the list sort always was. */
export interface ListSort<K extends string> { key: K; dir: 1 | -1 }
export const lineSort = signal<ListSort<"name" | "city" | "riders" | "money" | "trains" | "near">>({ key: "riders", dir: -1 });
export const stationSort = signal<ListSort<"name" | "riders" | "near">>({ key: "riders", dir: -1 });
/** The Money tab's Cities section (T-550): by name, train share, train rides a day or money a day. */
export const citySort = signal<ListSort<"name" | "train" | "rides" | "money" | "points">>({ key: "rides", dir: -1 });
/** The map's demand drawn for one trip purpose only, or all (T-158). Not remembered: a page
 * load starts on all purposes, so nobody returns to a map that silently leaves riders out. */
export const purposeFilter = signal<PurposeFilter>("all");
/** T-1168: the line panel's stops drawn with the lines related to the line; one setting for every
 * line, on at first (T-1529) and kept while the page is open. Not stored, so there is no old choice
 * to carry over. */
export const relatedLines = signal(true);
/** Platform length for new stations on regular and high-speed track, m; tram track has its own
 * (`buildTramPlatform`, T-2926). Each is remembered in the browser (T-3015). */
export const buildPlatform = signal(savedLength(saved.build?.platform, 60, 400, 20, 200));
export const TRAM_PLATFORM = { min: 20, max: 80, step: 20, start: 40 } as const;
export const buildTramPlatform = signal<number>(savedLength(saved.build?.tramPlatform, TRAM_PLATFORM.min, TRAM_PLATFORM.max, TRAM_PLATFORM.step, TRAM_PLATFORM.start));
function savedLength(v: unknown, min: number, max: number, step: number, fallback: number): number {
  return typeof v === "number" && v >= min && v <= max && v % step === 0 ? v : fallback;
}
/** The station tool's pointer is on tram track, so the station bar steps tram stops' length (T-2926). */
export const stationOnTram = signal(false);
/** T-468: the track bar's settings below are remembered per browser with the rest of the
 * UI state; anything unreadable falls back to the defaults. */
export const buildLevel = signal<Level>(saved.build?.tram === true ? 0 : LEVELS.includes(saved.build?.level as Level) ? (saved.build!.level as Level) : 1);
/** T-1822: this browser has finished the Getting started guide once, so new games start without it. */
export const guideFinished = signal(saved.guide === true);
/** T-2263: the tutorial card is up in this game, so a few hovers it leads to give a longer wording
 * (the road crossings chip, the crowding dot). Set by game/guide.ts. */
export const guideVerbose = signal(false);
/** T-1615: track drawn across other track at its own elevation ramps over it (under it from -1
 * down) where that fits; on unless turned off. */
export const autoRamp = signal(saved.build?.ramp !== false);
/** T-445: the track tool starts each segment where the last ended. The Build tab's button and
 * Caps Lock set it; Shift continues a run for one click without it. */
export const keepPlacing = signal((saved.v ?? 1) < 7 || saved.build?.keep !== false);
/** T-727: the track tool snaps to the real network (noritetsu's lines); while it does, the layer is
 * drawn at full strength */
export const snapReal = signal(saved.build?.snap === true);
/** T-2927: tram track drawn with the track tool follows the streets (street tram track); on unless
 * turned off, and flipped while Ctrl is held (tools.ts) */
export const snapStreets = signal(saved.build?.streets !== false && !(saved.build?.tram === true && saved.build?.tracks === 4));
/** T-3129: rail track drawn with the track tool follows the main roads; off by default, and off whenever the real
 * network or beside track is snapped to (the three exclude each other in the bar) */
export const snapRoads = signal(saved.build?.roads === true && saved.build?.snap !== true && saved.build?.beside !== true);
/** T-1685: track drawn from beside existing track runs alongside it (`TrackApi::set_route_beside`) */
// T-3046: the two snaps exclude each other in the bar; a save holding both keeps the real network
export const snapBeside = signal(saved.build?.beside === true && saved.build?.snap !== true);
/** T-817: the speed the track tool lays curves out for, km/h (`TrackApi::set_design_speed`): T-3121,
 * any speed from DESIGN_MIN to DESIGN_MAX, typed in or stepped through DESIGN_SPEEDS. */
export const DESIGN_SPEEDS = [20, 40, 60, 80, 100, 120, 140, 160, 200, 250, 320] as const;
export const DESIGN_MIN = 20, DESIGN_MAX = 320;
export type DesignSpeed = number;
export const clampDesign = (kmh: number): number => (Number.isFinite(kmh) ? Math.round(Math.max(DESIGN_MIN, Math.min(DESIGN_MAX, kmh)) * 1000) / 1000 : 160);
export const designSpeed = signal<DesignSpeed>(typeof saved.build?.design === "number" ? clampDesign(saved.build.design) : 160);
/** T-3169: the speed real flyovers' crossing legs are laid out for where they have the room, km/h
 * (`TrackApi::set_flyover_speed`; the sim's FLYOVER_KMH before one is set), stepped as the design speed. */
export const FLYOVER_SPEED = 80;
export const flyoverSpeed = signal<DesignSpeed>(typeof saved.build?.fly === "number" ? clampDesign(saved.build.fly) : FLYOVER_SPEED);
/** T-3169: the track bar's Advanced row (the two target speeds) is open. */
export const buildAdvanced = signal(saved.build?.advanced === true);
/** T-817: the track tool lays high-speed track, else regular. Double or quad (T-3126), never single. */
export const highSpeed = signal(saved.build?.hsr === true && saved.build?.tracks !== 1 && saved.build?.tram !== true);
/** T-2926: the track tool lays tram track (always double, ground level to start with), the third type beside
 * Regular and High speed. Drawn by hand it is the off-street kind (`TrackApi::set_route_kind(3)`). */
export const tramTrack = signal(saved.build?.tram === true);
/** T-2931: the track tool lays quad track (two pairs) instead of double; high-speed (T-3126) and tram (T-3166) too. */
export const quadTrack = signal(saved.build?.tracks === 4);
export function setQuadTrack(on: boolean) {
  if (on === quadTrack.peek()) return;
  batch(() => {
    quadTrack.value = on;
    if (on) singleTrack.value = false;
    if (on && tramTrack.peek()) snapStreets.value = false; // quad tram is off-street track (T-3166)
  });
}
/** T-3093: the track tool lays single track (0.6x the cost), beside double and quad; never high-speed or tram. */
export const singleTrack = signal(saved.build?.tracks === 1 && saved.build?.hsr !== true && saved.build?.tram !== true);
export function setSingleTrack(on: boolean) {
  if (on === singleTrack.peek()) return;
  if (on && tramTrack.peek()) return;
  batch(() => {
    singleTrack.value = on;
    if (on) quadTrack.value = false;
    if (on && highSpeed.peek()) setTrackType(2);
  });
}
/** The design speed before High speed was picked, put back when Regular is picked again. */
let designBeforeHsr: DesignSpeed | null = null;
/** The level before Tram was picked, put back when another type is. */
let levelBeforeTram: Level | null = null;
export type TrackType = 2 | "hsr" | "tram";
export const trackType = computed<TrackType>(() => (tramTrack.value ? "tram" : highSpeed.value ? "hsr" : 2));
/** The track bar's Regular / High speed choice. High speed sets the design speed to its top (the
 * player can step it down); leaving it puts back the speed there was before. One batch, so the track
 * tool never sees the old type with the new speed (T-1001). */
export function setTrackType(t: TrackType) {
  const was = highSpeed.peek(), wasTram = tramTrack.peek();
  if ((t === "hsr" && was) || (t === "tram" && wasTram) || (t === 2 && !was && !wasTram)) return;
  batch(() => {
    if (t === "tram") {
      singleTrack.value = false;
      if (quadTrack.peek()) snapStreets.value = false;
      if (was && designBeforeHsr !== null) designSpeed.value = designBeforeHsr;
      designBeforeHsr = null;
      highSpeed.value = false;
      levelBeforeTram = buildLevel.peek();
      tramTrack.value = true;
      buildLevel.value = 0;
      return;
    }
    if (wasTram) {
      tramTrack.value = false;
      buildLevel.value = levelBeforeTram ?? 1;
      levelBeforeTram = null;
    }
    if (t === "hsr") {
      singleTrack.value = false;
      designBeforeHsr = designSpeed.peek();
      highSpeed.value = true;
      designSpeed.value = DESIGN_SPEEDS[DESIGN_SPEEDS.length - 1];
    } else {
      highSpeed.value = false;
      if (was && designBeforeHsr !== null) designSpeed.value = designBeforeHsr;
      designBeforeHsr = null;
    }
  });
}
/** the real network is being snapped to: the switch on and the track tool in hand */
/** T-3049: Snap beside is moot while tram track follows the streets, so its switch is hidden and the stored choice kept */
export const besideOn = computed(() => snapBeside.value && !(tramTrack.value && snapStreets.value));
export const snapping = computed(() => snapReal.value && tool.value === "track" && !tramTrack.value);

/** The day every demand view shows (T-260, T-464): the setting, the average day by default. Away,
 * Out and Back read as the average while no superholiday is near (T-264, T-1373). */
export const shownDay = computed<DemandDay>(() => {
  const s = display.value.dayShown;
  if (s === "away") return awayNear.value ? AWAY_DAY : AVERAGE_DAY;
  if (s === "out" || s === "back") return rushRun.value ? (s === "out" ? OUT_DAY : BACK_DAY) : AVERAGE_DAY;
  return s === "weekday" ? 0 : s === "saturday" ? 1 : s === "sunday" ? 2 : AVERAGE_DAY;
});
/** A superholiday's day shown, as a hover ends a figure, with "per " or "on " before it: "a day of
 * Golden Week, May 3-5"; null for any other day. */
const superDay = computed(() => {
  const d = shownDay.value;
  const when = d === AWAY_DAY ? whenAway.value : d === OUT_DAY ? whenOut.value : d === BACK_DAY ? whenBack.value : null;
  return when || null;
});
/** The shown day as a hover ends a figure: "per weekday", "per Saturday", "per day of Golden Week", "per day". */
export const shownDayPhrase = computed(() => (superDay.value ? t("days.perSuper", { when: superDay.value }) : DEMAND_DAY_PHRASE[shownDay.value]));
/** "on a weekday", "on a day of Golden Week", "a day": the points hovers' ending. */
export const shownDayOn = computed(() => (superDay.value ? t("days.onSuper", { when: superDay.value }) : DEMAND_DAY_ON[shownDay.value]));

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
// T-1695: the device pixel ratio, for style.css's `--px` (one device pixel in the zoomed panels)
if (typeof window !== "undefined") {
  const setDpr = () => document.documentElement.style.setProperty("--dpr", String(window.devicePixelRatio || 1));
  setDpr();
  window.addEventListener("resize", setDpr);
}
// Text measured before Nunito has loaded (station names, capacity tags) is measured again by the
// "tw-fontchange" event.
void loadNunito().then(() => document.dispatchEvent?.(new Event("tw-fontchange")));
effect(() => {
  if (VIDEO) return; // T-2774
  const s = { tab: tab.value, theme:theme.value, shade: shade.value, display: display.value, dock: dockLayout.value, build: { level: buildLevel.value, ramp: autoRamp.value, hsr: highSpeed.value, tram: tramTrack.value, tracks: quadTrack.value ? 4 : singleTrack.value ? 1 : 2, keep: keepPlacing.value, snap: snapReal.value, streets: snapStreets.value, roads: snapRoads.value, beside: snapBeside.value, design: designSpeed.value, fly: flyoverSpeed.value, advanced: buildAdvanced.value, platform: buildPlatform.value, tramPlatform: buildTramPlatform.value }, guide: guideFinished.value, v: VERSION };
  try {
    localStorage.setItem(STORE, JSON.stringify(s));
  } catch {
    /* storage blocked */
  }
});
