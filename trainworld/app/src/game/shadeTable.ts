// The light and dark themes' map colours (T-039, T-543), with no page behind them so the clock
// worker can read it (save thumbnails). Everything else imports `game/shades.ts`.

import { NET_INK } from "./palette";
import type { Shade } from "./ui";

export interface ShadeColours {
  land: string;
  water: string;
  park: string;
  /** basemap place names, faint */
  label: string;
  /** what roads, buildings and boundaries are mixed towards from the land, a little */
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
}

export const SHADES: Record<Shade, ShadeColours> = {
  light: {
    land: "#f2f2ef", water: "#cbd8df", park: "#e3e8df", label: "#a9abad", basemapMix: NET_INK, casing: "#ffffff", mapInk: NET_INK,
    offRail: "#9aa3ae", offMid: "#f2f4f7", offFade: "#ffffff",
    outside: "#182536", outsideOpacity: 0.075, thumbTrack: "#b9b4ad",
    markInk: NET_INK, markBlueprint: "#2f62b3",
  },
  dark: {
    land: "#202225", water: "#121820", park: "#212621", label: "#5b5f64", basemapMix: "#e8e6e3", casing: "#141517", mapInk: "#e6e4e0",
    offRail: "#6c747e", offMid: "#383e46", offFade: "#383e46",
    outside: "#000000", outsideOpacity: 0.35, thumbTrack: "#6f6b66",
    markInk: "#e6e4e0", markBlueprint: "#8db6ee",
  },
};

/** The paint of the fill outside the focused city, in a theme. */
export const outsidePaint = (c: ShadeColours) => ({ "fill-color": c.outside, "fill-opacity": c.outsideOpacity, "fill-antialias": false }) as const;
