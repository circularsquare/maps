// T-1907: the faces' skin tones and hair colours by the station's country (ISO 3166-1 alpha-2, the
// registry's `country`). A fixed table of a few broad profiles, each weights over six plain skin
// tones and the four hair colours; a country not listed keeps the mood-coloured cartoon face and
// evenly drawn hair. Natural variation within each, nothing drawn differently but the colours.

/** Skin tones, lightest first: fill, edge, and the ink of the eyes and mouth. */
export const SKIN = [
  { fill: "#f7dcc6", edge: "#d9b08f", ink: "#6b4a3a" },
  { fill: "#eccaa6", edge: "#c99e74", ink: "#5e402f" },
  { fill: "#d9ab80", edge: "#b0814f", ink: "#4e3424" },
  { fill: "#bf8b60", edge: "#93653d", ink: "#3f291b" },
  { fill: "#9a6a47", edge: "#714a2d", ink: "#2e1e14" },
  { fill: "#71503a", edge: "#4f3524", ink: "#24170f" },
];

/** Weights over `SKIN`, and over the hair colours (faces.ts `HAIR`: brown, black, auburn, honey). */
interface Profile { skin: number[]; hair: number[] }

const P: Record<string, Profile> = {
  eastAsia: { skin: [3, 5, 2, 0, 0, 0], hair: [1, 9, 0, 0] },
  southeastAsia: { skin: [1, 3, 4, 2, 0.5, 0], hair: [0.5, 9.5, 0, 0] },
  southAsia: { skin: [0, 0.5, 2.5, 4, 2.5, 0.5], hair: [0.5, 9.5, 0, 0] },
  middleEast: { skin: [1, 3.5, 3.5, 1.5, 0.5, 0], hair: [3, 6.8, 0, 0.2] },
  northEurope: { skin: [5.5, 3.5, 0.7, 0.2, 0.1, 0], hair: [5, 1.2, 0.6, 3.2] },
  southEurope: { skin: [3.5, 4.5, 1.5, 0.3, 0.2, 0], hair: [6, 3, 0.3, 0.7] },
  africa: { skin: [0, 0, 0.2, 1.3, 4, 4.5], hair: [0.2, 9.8, 0, 0] },
  southAfrica: { skin: [0.8, 0.4, 0.6, 1.2, 3.5, 3.5], hair: [1.2, 8.2, 0.1, 0.5] },
  northAmerica: { skin: [4, 2, 1.5, 1, 1, 0.5], hair: [4.5, 3.5, 0.4, 1.6] },
  latinAmerica: { skin: [1.5, 3, 3, 1.5, 0.8, 0.2], hair: [5.5, 4, 0.1, 0.4] },
  brazil: { skin: [2, 2.5, 2, 1.5, 1.2, 0.8], hair: [5, 4.4, 0.1, 0.5] },
  southernCone: { skin: [3.5, 4, 1.5, 0.5, 0.3, 0.2], hair: [6, 3, 0.2, 0.8] },
  oceania: { skin: [4.5, 2.5, 1.2, 0.8, 0.6, 0.4], hair: [4.5, 2.5, 0.5, 2.5] },
};

const COUNTRIES: Record<string, string[]> = {
  eastAsia: ["JP", "KR", "KP", "CN", "TW", "HK", "MO", "MN"],
  southeastAsia: ["VN", "TH", "KH", "LA", "MM", "MY", "SG", "ID", "PH", "BN", "TL"],
  southAsia: ["IN", "PK", "BD", "LK", "NP", "BT", "MV"],
  middleEast: ["TR", "IR", "IQ", "SA", "AE", "QA", "KW", "BH", "OM", "YE", "JO", "LB", "SY", "IL", "PS", "EG", "LY", "TN", "DZ", "MA", "AF", "AZ", "AM", "GE", "KZ", "UZ", "TM", "KG", "TJ"],
  northEurope: ["GB", "IE", "DE", "NL", "BE", "LU", "DK", "SE", "NO", "FI", "IS", "PL", "CZ", "SK", "AT", "CH", "HU", "EE", "LV", "LT", "RU", "UA", "BY", "MD"],
  southEurope: ["FR", "ES", "PT", "IT", "GR", "HR", "SI", "RS", "BA", "ME", "MK", "AL", "XK", "BG", "RO", "CY", "MT", "AD", "MC", "SM"],
  africa: ["NG", "GH", "KE", "ET", "TZ", "UG", "RW", "CD", "CG", "CM", "CI", "SN", "ML", "BF", "NE", "TD", "SD", "SS", "SO", "AO", "ZM", "ZW", "MZ", "MW", "MG", "BJ", "TG", "GN", "SL", "LR", "GA", "BW", "NA", "LS", "SZ", "ER", "DJ", "GM", "GW", "MR", "BI", "CF", "GQ"],
  southAfrica: ["ZA"],
  northAmerica: ["US", "CA", "PR"],
  latinAmerica: ["MX", "GT", "HN", "SV", "NI", "CR", "PA", "CU", "DO", "HT", "JM", "TT", "CO", "VE", "EC", "PE", "BO", "PY"],
  brazil: ["BR"],
  southernCone: ["AR", "CL", "UY"],
  oceania: ["AU", "NZ"],
};

const OF = new Map<string, Profile>();
for (const [p, list] of Object.entries(COUNTRIES)) for (const cc of list) OF.set(cc, P[p]);

/** A draw from `weights` with `u` in [0, 1). */
function drawFrom(weights: number[], u: number): number {
  const total = weights.reduce((a, b) => a + b, 0);
  let t = u * total;
  for (let i = 0; i < weights.length; i++) if ((t -= weights[i]) < 0) return i;
  return weights.length - 1;
}

/** A face's skin tone (index into `SKIN`, -1 for the mood-coloured face) and hair colour (index
 * into `hairs` colours) in `country`, from two uniform draws. */
export function toneFor(country: string | null | undefined, hairs: number, us: number, uh: number): { skin: number; tint: number } {
  const p = country ? OF.get(country) : undefined;
  if (!p) return { skin: -1, tint: Math.floor(uh * hairs) };
  return { skin: drawFrom(p.skin, us), tint: drawFrom(p.hair, uh) };
}

const hex = (s: string) => [1, 3, 5].map((i) => parseInt(s.slice(i, i + 2), 16));
/** `a` mixed towards `b` by `t`, as #rrggbb. */
export function mix(a: string, b: string, t: number): string {
  const x = hex(a), y = hex(b);
  return "#" + x.map((v, i) => Math.round(v + (y[i] - v) * t).toString(16).padStart(2, "0")).join("");
}
