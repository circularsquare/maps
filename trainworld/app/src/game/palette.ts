// Colours of the network itself. These are not UI theme colours and do not change with the theme
// (SPEC 8: only the UI is styled; lines, trains and stations take the line colours).

import type { Level } from "./types";

/** Default line colours in order: a normal saturated metro palette (SPEC 8). The last five (gray,
 * yellow-green, pink, magenta, light blue; Anita, T-223) keep every pair at least 19 apart in
 * CIEDE2000 (the first eight's closest pair, orange and yellow, is 20); yellow-green, pink and
 * light blue are light enough that their chips take dark letters (`inkOn`). */
export const LINE_PALETTE = ["#d7263d", "#1f6fd1", "#1a9a4a", "#f08a00", "#8a3fb8", "#f2c200", "#8f5a2a", "#0e9a9a",
  "#74797e", "#9ccb19", "#f59dc6", "#e62aa6", "#5cc3f0"];

/** Track colour by height: red-orange, orange and yellow above ground, green at ground, then
 * blue-green, blue and indigo going down (T-413: grey was hard to see on the basemap; T-478: spread
 * so that neighbours are at least 30 apart in CIELAB, was 20). The one source for the drawn track,
 * the build ghost, the Build tab's level buttons and the Height key. */
export const LEVEL_COLOURS: Record<Level, string> = {
  3: "#c2351c",
  2: "#ee7a1a",
  1: "#f0b928",
  0: "#58b04a",
  [-1]: "#16a39a",
  [-2]: "#2f6fd0",
  [-3]: "#4a2c9a",
};

/** The Max speed display's key, km/h and colour; renderer.ts's stroke shader blends the same stops. */
export const SPEED_STOPS: [number, string][] = [[40, "#d85b4b"], [80, "#d9a437"], [120, "#55a886"], [160, "#397bc1"]];

/** The colour the Max speed display gives `kmh`. */
export function speedColour(kmh: number): string {
  const s = SPEED_STOPS;
  if (kmh <= s[0][0]) return s[0][1];
  for (let i = 1; i < s.length; i++) {
    if (kmh > s[i][0]) continue;
    const t = (kmh - s[i - 1][0]) / (s[i][0] - s[i - 1][0]);
    const a = rgb(s[i - 1][1]), b = rgb(s[i][1]);
    return "#" + a.map((v, k) => Math.round(v + (b[k] - v) * t).toString(16).padStart(2, "0")).join("");
  }
  return s[s.length - 1][1];
}

/** Station rings and names on the map. */
export const NET_INK = "#2b2b2b";

/** A platform's edges inside its station box (T-236, Anita: "show the actual platforms as like
 * little blue lines"), on the map and in the complex map. A periwinkle: CIEDE2000 15 from the line
 * blue and the light blue (the palette's blues leave no room for more), 38 from white. */
export const PLATFORM_BLUE = "#8085f0";

/**
 * How commuters travel (T-097, Anita: Subway Builder's scheme, red car, blue train, green walk):
 * the commuter bubbles and the mode-split bar in the top bar use these, whatever the
 * theme. Hand-tunable. Near-pure primaries, so a bubble's mix (`modeMix`) reads as RGB: all car
 * red, all train blue, half and half magenta.
 */
export const MODE_COLOURS = {
  train: "#1e3ce6",
  walk: "#1eaa3c",
  drive: "#e61e1e",
} as const;

/**
 * A bubble's colour for commuters by train, walking and driving, 0-255 rgb: the three mode colours
 * blended by share and scaled to full brightness (Anita chose this RGB mix, T-099). Nobody: grey
 * (never drawn, a bubble with nobody is left out).
 */
export function modeMix(train: number, walk: number, drive: number): [number, number, number] {
  const all = train + walk + drive;
  if (all <= 0) return [163, 155, 143];
  const [t, w, d] = MIX ?? (MIX = [rgb(MODE_COLOURS.train), rgb(MODE_COLOURS.walk), rgb(MODE_COLOURS.drive)]);
  const r = (t[0] * train + w[0] * walk + d[0] * drive) / all;
  const gg = (t[1] * train + w[1] * walk + d[1] * drive) / all;
  const b = (t[2] * train + w[2] * walk + d[2] * drive) / all;
  const k = MIX_TOP / Math.max(r, gg, b, 1);
  return [Math.round(r * k), Math.round(gg * k), Math.round(b * k)];
}
let MIX: [number, number, number][] | null = null;
const MIX_TOP = Math.max(...rgb(MODE_COLOURS.train), ...rgb(MODE_COLOURS.walk), ...rgb(MODE_COLOURS.drive));

/**
 * A selected station's riders on the map (T-084, T-097): the cells they live or work in, opaque on
 * top, and the places at the other end of their trip, filled and partly see-through below them
 * (`DESTINATION_ALPHA`). Both drawn over the network.
 */
export const CATCHMENT_COLOUR = "#c2185b";
export const DESTINATION_COLOUR = "#3a3f8f";
export const DESTINATION_ALPHA = 0.55;
/** Commuter bubbles are this opaque, so the network shows through where they pile up. */
export const BUBBLE_ALPHA = 0.8;

export function rgb(hex: string): [number, number, number] {
  return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)) as [number, number, number];
}

/** Letter colour on a line chip: dark on light lines (yellow), white on the rest. */
export function inkOn(hex: string): string {
  const [r, g, b] = rgb(hex).map((v) => v / 255);
  return 0.299 * r + 0.587 * g + 0.114 * b > 0.62 ? NET_INK : "#ffffff";
}
