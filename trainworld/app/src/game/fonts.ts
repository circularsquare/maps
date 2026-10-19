// The panels' font choices (T-698, T-703). Nunito is the default, self-hosted in `public/fonts/` (the
// Latin and Latin extended subsets of Google's variable file, SIL OFL); it has no Japanese, so kana
// and kanji fall back to Zen Maru Gothic, which index.html loads from Google Fonts. The others are
// fetched from Google Fonts (all SIL OFL) only when chosen. The map's own labels are glyph tiles and
// stay Zen Maru Gothic.

export type FontKey = "zen" | "mplus" | "kosugi" | "nunito";
export const DEFAULT_FONT: FontKey = "nunito";

export interface FontChoice {
  label: string;
  /** the CSS family */
  family: string;
  /** Google Fonts css2 query for this family; null when the page already has it */
  query: string | null;
  /** self-hosted faces under `public/fonts/` (one variable file each), with their unicode ranges */
  files?: { file: string; range: string }[];
}

export const FONTS: Record<FontKey, FontChoice> = {
  nunito: {
    label: "Nunito", family: '"Nunito"', query: null,
    files: [
      { file: "Nunito-latin.woff2", range: "U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+0304, U+0308, U+0329, U+2000-206F, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD" },
      { file: "Nunito-latin-ext.woff2", range: "U+0100-02BA, U+02BD-02C5, U+02C7-02CC, U+02CE-02D7, U+02DD-02FF, U+0304, U+0308, U+0329, U+1D00-1DBF, U+1E00-1E9F, U+1EF2-1EFF, U+2020, U+20A0-20AB, U+20AD-20C4, U+2113, U+2C60-2C7F, U+A720-A7FF" },
    ],
  },
  zen: { label: "Zen Maru Gothic", family: '"Zen Maru Gothic"', query: null },
  mplus: { label: "M PLUS Rounded 1c", family: '"M PLUS Rounded 1c"', query: "family=M+PLUS+Rounded+1c:wght@400;700" },
  kosugi: { label: "Kosugi Maru", family: '"Kosugi Maru"', query: "family=Kosugi+Maru" },
};
export const FONT_KEYS = Object.keys(FONTS) as FontKey[];

/** the `--font` value for a choice: its family, then the page font for what it lacks */
export const fontStack = (k: FontKey) => (k === "zen" ? `"Zen Maru Gothic", system-ui, sans-serif` : `${FONTS[k].family}, "Zen Maru Gothic", system-ui, sans-serif`);

const added = new Map<FontKey, Promise<unknown>>();
/** Add the choice's faces once (its Google Fonts stylesheet, or its own files); resolves when they
 * have loaded for the page's text. */
export async function loadFont(k: FontKey): Promise<void> {
  const c = FONTS[k];
  if (!added.has(k)) {
    if (c.query) {
      added.set(k, new Promise<void>((done) => {
        const link = document.createElement("link");
        link.rel = "stylesheet";
        link.href = `https://fonts.googleapis.com/css2?${c.query}&display=swap`;
        link.onload = link.onerror = () => done();
        document.head.appendChild(link);
      }));
    } else if (c.files && typeof FontFace !== "undefined") {
      added.set(k, Promise.all(c.files.map(({ file, range }) => {
        const face = new FontFace("Nunito", `url("${new URL(`fonts/${file}`, document.baseURI).href}") format("woff2")`, { weight: "200 1000", unicodeRange: range, display: "swap" });
        document.fonts.add(face);
        return face.load();
      })).catch(() => undefined));
    }
  }
  try {
    await added.get(k);
    await Promise.all([`14px ${c.family}`, `700 14px ${c.family}`].map((f) => document.fonts?.load(f, "Aé")));
  } catch {
    // offline: the fallback stays
  }
}
