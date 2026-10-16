// The light and dark themes' map colours (T-039, T-543), table in `game/shadeTable.ts`: the basemap,
// the casing under every line and track stroke, and the ink of what the game draws over the map
// (selection outlines, walk legs, tags' text, a selected train's halo), the blueprint's switched-off
// look, the shade outside the focused city and the save thumbnails. The panels' colours are CSS
// tokens (ui/style.css, which also holds the stop editing markers' greys); `--map-land` and
// `--map-ink` are set from the table so the map's DOM tags match the canvas. Line, mode, level and
// purpose colours are the same in both (game/palette.ts). Station dots stay white with a NET_INK
// rim in both.

import { effect } from "@preact/signals";
import { SHADES, type ShadeColours } from "./shadeTable";
import { shade } from "./ui";

export { outsidePaint, SHADES, type ShadeColours } from "./shadeTable";

/** The current theme's map colours. */
export const shadeColours = (): ShadeColours => SHADES[shade.peek()];

effect(() => {
  const c = SHADES[shade.value], root = document.documentElement.style;
  root.setProperty("--map-land", c.land);
  root.setProperty("--map-ink", c.mapInk);
});
