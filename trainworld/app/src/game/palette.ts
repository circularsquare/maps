// Colours of the network itself. These are not UI theme colours and do not change with the theme
// (SPEC 8: only the UI is styled; lines, trains and stations take the line colours).

import { signal } from "@preact/signals";
import type { Level } from "./types";

/** Default line colours: a normal saturated metro palette (SPEC 8), a new line taking a random unused
 * one (T-856). Every pair is at least 19 apart in CIEDE2000 (T-223, T-856) and at least 17 from the
 * map's land, water and parks in both themes (the closest, light blue on light water); yellow-green,
 * pink, light blue and lavender are light enough that their chips take dark letters (`inkOn`). */
export const LINE_PALETTE = ["#d7263d", "#1f6fd1", "#1a9a4a", "#f08a00", "#8a3fb8", "#f2c200", "#8f5a2a", "#0e9a9a",
  "#74797e", "#9ccb19", "#f59dc6", "#e62aa6", "#5cc3f0", "#6f7a0f", "#ee7d63", "#a9a0e8"];

/** Track colour by height: blue-green, blue and indigo going up (open sky), green at ground, then
 * yellow, orange and red-orange going down (deep; T-480). Neighbours are at least 30 apart in
 * CIELAB (T-478). The one source for the drawn track, the build ghost, the Build tab's level
 * buttons and the Height key. */
export const LEVEL_COLOURS: Record<Level, string> = {
  3: "#4a2c9a",
  2: "#2f6fd0",
  1: "#16a39a",
  0: "#58b04a",
  [-1]: "#f0b928",
  [-2]: "#ee7a1a",
  [-3]: "#c2351c",
};

/** The Max speed display's key, km/h and colour; renderer.ts builds its stroke shader's blend from the
 * same stops. Warm red to blue up to 160 (ordinary track), then indigo and violet up to high-speed
 * track's 320 (T-843). */
export const SPEED_STOPS: [number, string][] = [[40, "#d85b4b"], [80, "#d9a437"], [120, "#55a886"], [160, "#397bc1"], [240, "#5b4fc9"], [320, "#a03fc2"]];

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

/** T-833: the colours bars and swatches show modes in under Train share bubbles. Car red and train
 * blue are the ramp's ends; walking is yellow (the ramp has no green, and walkers pull a bubble
 * towards red) and "other" (planes: the one mode that is not car, walk or train) orange, a step from
 * both the yellow and the red. Under Mode mix, planes stay the muted grey. */
const SHARE_MODE_COLOURS = { train: MODE_COLOURS.train, walk: "#f0c814", drive: MODE_COLOURS.drive, other: "#f07a14" } as const;
const MIX_MODE_COLOURS = { ...MODE_COLOURS, other: "var(--muted)" } as const;
/** The scheme in force; a signal, so panels that read `modeColours()` follow a change. */
export const modeScheme = signal<"mode" | "ramp">("mode");
/** The mode colours for bars, swatches and hover tags under the bubble scheme in force. */
export const modeColours = (): { readonly train: string; readonly walk: string; readonly drive: string; readonly other: string } =>
  modeScheme.value === "ramp" ? SHARE_MODE_COLOURS : MIX_MODE_COLOURS;

/**
 * A bubble's colour for commuters by train, walking and driving, 0-255 rgb: the three mode colours
 * blended by share and scaled to full brightness (Anita chose this RGB mix, T-099), then towards a
 * light grey as all three modes near a third each. Nobody: grey
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
  // T-653: an even three-way mix would come out near white; the smallest share, from nothing up to a
  // third, pulls the colour towards one grey (the same in both themes, so baked bubble colours stay valid)
  const g = Math.min(1, 3 * Math.min(train, walk, drive) / all);
  return [Math.round(r * k + (MIX_GREY - r * k) * g), Math.round(gg * k + (MIX_GREY - gg * k) * g), Math.round(b * k + (MIX_GREY - b * k) * g)];
}
const MIX_GREY = 190;
let MIX: [number, number, number][] | null = null;
const MIX_TOP = Math.max(...rgb(MODE_COLOURS.train), ...rgb(MODE_COLOURS.walk), ...rgb(MODE_COLOURS.drive));

/** How the demand bubbles are coloured (T-804): the mode mix above, or a ramp by train share from
 * one colour at 0% to another at 100%. */
export type BubbleScheme = "mode" | "ramp";
export const DEFAULT_RAMP = { from: MODE_COLOURS.drive, to: MODE_COLOURS.train } as const;
/** The ramp's position is the train share to this power, so a 30% share sits halfway (Anita, T-812:
 * 30% purple, 70% clearly blue; 0.3^K = 0.5, and 70% lands at 81%). */
const RAMP_SKEW = Math.log(0.5) / Math.log(0.3);
let schemeKey = "mode";
/** the ramp as 256 rgb triples */
let rampLut: Uint8ClampedArray | null = null;

/** Set the bubble colouring for everything built from now on (the map views, the inspectors'
 * swatches); the same arguments again change nothing. */
export function setBubbleColours(s: BubbleScheme, from: string, to: string) {
  const key = s === "ramp" ? `ramp${from}${to}` : "mode";
  modeScheme.value = s === "ramp" ? "ramp" : "mode";
  if (key === schemeKey) return;
  schemeKey = key;
  rampLut = s === "ramp" ? buildRamp(HEX.test(from) ? from : DEFAULT_RAMP.from, HEX.test(to) ? to : DEFAULT_RAMP.to) : null;
}
/** Names the scheme set now ("mode", or "ramp" with its two colours), to tell built discs apart. */
export const bubbleSchemeKey = () => schemeKey;
const HEX =/^#[0-9a-f]{6}$/i;

/** A bubble's colour under the chosen scheme: `modeMix`, or the ramp at the train share. */
export function bubbleColour(train: number, walk: number, drive: number): [number, number, number] {
  if (!rampLut) return modeMix(train, walk, drive);
  const all = train + walk + drive;
  if (all <= 0) return [163, 155, 143];
  const k = Math.round((train / all) * 255) * 3;
  return [rampLut[k], rampLut[k + 1], rampLut[k + 2]];
}

// OKLab (Ottosson): even steps of lightness and hue between the two ends
const toLinear = (v: number) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
const fromLinear = (v: number) => (v <= 0.0031308 ? v * 12.92 : 1.055 * v ** (1 / 2.4) - 0.055);
function toOklab(hex: string): [number, number, number] {
  const [r, g, b] = rgb(hex).map((v) => toLinear(v / 255));
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s, 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
}
function fromOklab([L, a, b]: [number, number, number]): [number, number, number] {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  return [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s, -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s]
    .map((v) => Math.round(255 * fromLinear(Math.min(1, Math.max(0, v))))) as [number, number, number];
}
function buildRamp(from: string, to: string): Uint8ClampedArray {
  const a = toOklab(from), b = toOklab(to), out = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) ** RAMP_SKEW;
    out.set(fromOklab([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]), i * 3);
  }
  return out;
}

/** T-976: the colour of the small dots that stand for zones the game has not opened (the zoomed-out
 * overview, map/intercityView.ts); a setting. */
export const UNBUILT_COLOURS = {
  light: { black: "#000000", maroon: "#800000", grey: "#7a7a7a" },
  // lifted so they read on the dark basemap: "black" becomes a pale ink
  dark: { black: "#e4dfd6", maroon: "#c4585c", grey: "#9c9c9c" },
} as const;
export type UnbuiltColour = keyof (typeof UNBUILT_COLOURS)["light"];
export const UNBUILT_KEYS = Object.keys(UNBUILT_COLOURS.light) as UnbuiltColour[];

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
