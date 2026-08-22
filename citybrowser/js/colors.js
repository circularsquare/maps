// Colour scales. Pure functions, no dependencies — safe for one owner to hold.
//
// LIGHT BACKGROUND. Every ramp here is calibrated against a pale ground, which
// mainly means the low/mid end cannot be a light yellow: it vanishes. Where a
// dark-mode ramp would run green -> yellow, this runs medium-green -> amber.

const clamp01 = t => t < 0 ? 0 : t > 1 ? 1 : t;

function lerpStops(stops, t) {
  t = clamp01(t);
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const [a, ca] = stops[i - 1], [b, cb] = stops[i];
      const u = (b - a) ? (t - a) / (b - a) : 0;
      return [0, 1, 2].map(k => Math.round(ca[k] + (cb[k] - ca[k]) * u));
    }
  }
  return stops[stops.length - 1][1];
}

export const rgb = c => `rgb(${c[0]},${c[1]},${c[2]})`;

// --- population: green (low) -> yellow -> orange -> red -> purple (highest).
//
// These are the FULL-STRENGTH stops. An earlier version darkened them on the
// theory that yellow needs help on a light ground; in practice that plus the
// untouched-fade compounded into mud. Saturation reads better than darkness
// here — leave these bright and let fade() do the muting.
const POP_STOPS = [
  [0.00, [ 63, 163,  77]],
  [0.34, [212, 212,  74]],
  [0.56, [232, 145,  42]],
  [0.78, [214,  59,  47]],
  [1.00, [142,  63, 168]],
];
const POP_LO = Math.log10(1e4), POP_HI = Math.log10(3e7);
export const popT = p => (Math.log10(Math.max(p || 1e4, 1e4)) - POP_LO) / (POP_HI - POP_LO);
export const popColor = p => lerpStops(POP_STOPS, popT(p));
export const popRampCSS = () => rampCSS(POP_STOPS);

// --- altitude: hypsometric. Green lowland -> tan -> orange-red -> pale grey
// summits, the convention people already read on physical maps.
const ELEV_STOPS = [
  [0.00, [106, 153,  85]],
  [0.25, [201, 180, 106]],
  [0.50, [193, 129,  63]],
  [0.75, [173,  79,  60]],
  [1.00, [206, 200, 203]],
];
export const elevT = m => clamp01((m || 0) / 5000);
export const elevColor = m => lerpStops(ELEV_STOPS, elevT(m));
export const elevRampCSS = () => rampCSS(ELEV_STOPS);

// --- gdp per capita: dark blue -> teal -> amber -> red. Log scale, since the
// spread from Kinshasa to Zurich is two orders of magnitude.
const GDP_STOPS = [
  [0.00, [ 33,  56, 122]],
  [0.35, [ 42, 130, 140]],
  [0.65, [214, 158,  46]],
  [1.00, [178,  44,  40]],
];
const GDP_LO = Math.log10(500), GDP_HI = Math.log10(120000);
export const gdpT = v => clamp01((Math.log10(Math.max(v || 500, 500)) - GDP_LO) / (GDP_HI - GDP_LO));
export const gdpColor = v => lerpStops(GDP_STOPS, gdpT(v));
export const gdpRampCSS = () => rampCSS(GDP_STOPS);

function rampCSS(stops) {
  const out = [];
  for (let i = 0; i <= 12; i++) out.push(`${rgb(lerpStops(stops, i / 12))} ${i * 100 / 12}%`);
  return `linear-gradient(90deg,${out.join(',')})`;
}

// --- languages: the SAME language must get the SAME colour in every city, so
// the palette is keyed by name, not by position in that city's list. Curated
// entries first (the languages that recur most), hashed fallback after — the
// hash is stable, so a language keeps its colour across sessions and rebuilds.
const NAMED = {
  English: '#2f6fb5', Spanish: '#d4762a', Mandarin: '#c0392b', Chinese: '#c0392b',
  Hindi: '#e0952b', Arabic: '#2e8b57', Portuguese: '#3f9c52', Bengali: '#1a9187',
  Russian: '#6d5bab', Japanese: '#c2417a', French: '#4054a8', German: '#8a6d3b',
  Korean: '#0f7f92', Italian: '#4f8f3a', Turkish: '#b03a48', Vietnamese: '#b8862b',
  Urdu: '#2e7d5b', Indonesian: '#a0522d', Persian: '#7b5ea7', Swahili: '#b5603a',
  Thai: '#9b4f96', Polish: '#8f3f5f', Dutch: '#e07b39', Tagalog: '#3a8f7a',
};
const HASHED = [
  '#4a7fb0','#b5703c','#5a9440','#a04a5e','#7a68a8','#3f8e8e','#a8873c','#8c5a9c',
  '#4f7a5c','#b0524a','#5b7fa0','#96683f','#6f9350','#9c4a7a','#4a6ea8','#8a7b3f',
];
export function languageColor(name) {
  if (!name) return '#9aa4ae';
  if (NAMED[name]) return NAMED[name];
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) | 0;
  return HASHED[Math.abs(h) % HASHED.length];
}

// --- curation state.
//
// Untouched cities are desaturated toward grey. Keep this SLIGHT: curation runs
// for months, so <1% touched is the normal state for a long time, and a heavy
// fade means the map spends most of its life looking muddy and dead. At 0.30
// the population ramp still reads on untouched dots; curated ones are picked
// out by an outline (see map.js) rather than by everything else being drab.
const GREY = [150, 156, 163];
export function fade(c, amount = 0.30) {
  return [0, 1, 2].map(k => Math.round(c[k] + (GREY[k] - c[k]) * amount));
}
