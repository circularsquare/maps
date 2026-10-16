// The light and dark themes' map colours (T-039): the basemap, the casing under every line and
// track stroke, and the ink of what the game draws over the map (selection outlines, walk legs, tags'
// text). The panels' colours are CSS tokens (ui/style.css); `--map-land` and `--map-ink` are set
// from this table so the map's DOM tags match the canvas. Line, mode, level and purpose colours are
// the same in both (game/palette.ts). Station dots stay white with a NET_INK rim in both.

import { effect } from "@preact/signals";
import { NET_INK } from "./palette";
import { shade, type Shade } from "./ui";

export interface ShadeColours {
  land: string;
  water: string;
  park: string;
  /** basemap place names, faint */
  label: string;
  /** what roads, buildings and boundaries are mixed towards from the land, a little */
  basemapMix: string;
  /** under line and track strokes */
  casing: string;
  /** a selected track, a selected bubble's outline, walk legs, the map's tags' text */
  mapInk: string;
}

export const SHADES: Record<Shade, ShadeColours> = {
  light: { land: "#f2f2ef", water: "#cbd8df", park: "#e3e8df", label: "#a9abad", basemapMix: NET_INK, casing: "#ffffff", mapInk: NET_INK },
  dark: { land: "#202225", water: "#121820", park: "#212621", label: "#5b5f64", basemapMix: "#e8e6e3", casing: "#141517", mapInk: "#e6e4e0" },
};

/** The current theme's map colours. */
export const shadeColours = (): ShadeColours => SHADES[shade.peek()];

effect(() => {
  const c = SHADES[shade.value], root = document.documentElement.style;
  root.setProperty("--map-land", c.land);
  root.setProperty("--map-ink", c.mapInk);
});
