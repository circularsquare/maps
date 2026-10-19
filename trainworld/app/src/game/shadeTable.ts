// The light and dark themes' map colours (T-039, T-543), with no page behind them so the clock
// worker can read it (save thumbnails). Everything else imports `game/shades.ts`.

import { NET_INK } from "./palette";
import type { Shade } from "./ui";

// The basemap's palette (T-676), the values to retune by hand: warm paper land, soft teal water,
// sage parks and woods, one road colour for all three road weights, faint grey place names, and
// the warm tone that buildings, boundaries and airfields are mixed towards from the land.
const LIGHT_LAND = "#f4ecdd", LIGHT_WATER = "#b7d9d6", LIGHT_PARK = "#d3e0c4", LIGHT_ROAD = "#ffffff";
const LIGHT_LABEL = "#aaa7a3", LIGHT_WARM = "#6e5b48";
const DARK_LAND = "#292520", DARK_WATER = "#1e3032", DARK_PARK = "#2b3727", DARK_ROAD = "#443e36";
const DARK_LABEL = "#6a655e", DARK_WARM = "#efe4d3";

export interface ShadeColours {
  land: string;
  water: string;
  park: string;
  /** every road, in its three weights (`map/basemap.ts`) */
  road: string;
  /** basemap place names, faint */
  label: string;
  /** what buildings, boundaries and airfields are mixed towards from the land, a little */
  basemapMix: string;
  /** under line and track strokes, and the band between a selected train and its ring */
  casing: string;
  /** a selected track, a selected bubble's outline, walk legs, the map's tags' text, a selected
   * train's outline, ring and glow */
  mapInk: string;
  /** a switched-off blueprint's rails and the middle between them */
  offRail: string;
  offMid: string;
  /** the colour a switched-off blueprint's other colours (a line planned along it, Elevation and
   * Max speed) are mixed 80% towards; a blueprint that is on is mixed 45% towards white in both */
  offFade: string;
  /** the fill of everything outside the focused city, and of a zone still loading, and how opaque */
  outside: string;
  outsideOpacity: number;
  /** the track's grey in a save's thumbnail, drawn over `land` */
  thumbTrack: string;
  /** the crossing, junction, ramp and portal marks (T-629, T-655): the ink of those on built track
   * and the ink of those on blueprint track and while drawing; their fill is `casing` */
  markInk: string;
  markBlueprint: string;
  /** the real-network reference layer (T-682, `map/realNetwork.ts`): main line, other passenger rail, metro */
  realMain: string;
  realOther: string;
  realMetro: string;
  /** a city this game has not opened (T-729, map/greyCities.ts), drawn at the bubbles' opacity */
  greyCity: string;
}

export const SHADES: Record<Shade, ShadeColours> = {
  light: {
    land: LIGHT_LAND, water: LIGHT_WATER, park: LIGHT_PARK, road: LIGHT_ROAD, label: LIGHT_LABEL, basemapMix: LIGHT_WARM,
    casing: "#ffffff", mapInk: NET_INK,
    offRail: "#9aa3ae", offMid: "#f2f4f7", offFade: "#ffffff",
    outside: "#182536", outsideOpacity: 0.075, thumbTrack: "#b9b4ad",
    markInk: NET_INK, markBlueprint: "#2f62b3",
    realMain: "#6f7b93", realOther: "#9ea6b5", realMetro: "#9577a6",
    greyCity: "#8e8a84",
  },
  dark: {
    land: DARK_LAND, water: DARK_WATER, park: DARK_PARK, road: DARK_ROAD, label: DARK_LABEL, basemapMix: DARK_WARM,
    casing: "#141517", mapInk: "#e6e4e0",
    offRail: "#6c747e", offMid: "#383e46", offFade: "#383e46",
    outside: "#000000", outsideOpacity: 0.35, thumbTrack: "#6f6b66",
    markInk: "#e6e4e0", markBlueprint: "#8db6ee",
    realMain: "#6e788c", realOther: "#4e5563", realMetro: "#82699a",
    greyCity: "#8a867f",
  },
};

/** The paint of the fill outside the focused city, in a theme. */
export const outsidePaint = (c: ShadeColours) => ({ "fill-color": c.outside, "fill-opacity": c.outsideOpacity, "fill-antialias": false }) as const;
