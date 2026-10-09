// Colours of the network itself. These are not UI theme colours and do not change with the theme
// (SPEC 8: only the UI is styled; lines, trains and stations take the line colours).

import type { Level } from "./types";

/** Default line colours in order: a normal saturated metro palette (SPEC 8). */
export const LINE_PALETTE = ["#d7263d", "#1f6fd1", "#1a9a4a", "#f08a00", "#8a3fb8", "#f2c200", "#8f5a2a", "#0e9a9a"];

/** Track colour by height (display setting): warm above ground, grey at ground, cool below. */
export const LEVEL_COLOURS: Record<Level, string> = {
  3: "#c2410c",
  2: "#ea7a1a",
  1: "#f2b134",
  0: "#8c8c8c",
  [-1]: "#5aa3d8",
  [-2]: "#2f6fbf",
  [-3]: "#23408e",
};

/** Station rings and names on the map. */
export const NET_INK = "#2b2b2b";

/**
 * How commuters travel (T-097, Anita: Subway Builder's scheme, red car, blue train, green walk):
 * the commuter bubbles, their legend and the mode-split bar in the top bar use these, whatever the
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
