// The page font (T-698, T-703, T-1524): Nunito, self-hosted in `public/fonts/` (the Latin and Latin
// extended subsets of Google's variable file, SIL OFL). It has no Japanese, so kana and kanji fall
// back to Zen Maru Gothic, which index.html loads from Google Fonts. The map's own labels are glyph
// tiles and are Zen Maru Gothic.

const FILES = [
  { file: "Nunito-latin.woff2", range: "U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+0304, U+0308, U+0329, U+2000-206F, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD" },
  { file: "Nunito-latin-ext.woff2", range: "U+0100-02BA, U+02BD-02C5, U+02C7-02CC, U+02CE-02D7, U+02DD-02FF, U+0304, U+0308, U+0329, U+1D00-1DBF, U+1E00-1E9F, U+1EF2-1EFF, U+2020, U+20A0-20AB, U+20AD-20C4, U+2113, U+2C60-2C7F, U+A720-A7FF" },
];

let loaded: Promise<void> | null = null;
/** Add Nunito's faces once; resolves when they have loaded for the page's text. */
export function loadNunito(): Promise<void> {
  loaded ??= (async () => {
    try {
      if (typeof FontFace !== "undefined") {
        await Promise.all(FILES.map(({ file, range }) => {
          const face = new FontFace("Nunito", `url("${new URL(`fonts/${file}`, document.baseURI).href}") format("woff2")`, { weight: "200 1000", unicodeRange: range, display: "swap" });
          document.fonts.add(face);
          return face.load();
        }));
      }
      await Promise.all(['14px "Nunito"', '700 14px "Nunito"'].map((f) => document.fonts?.load(f, "Aé")));
    } catch {
      // offline: the fallback stays
    }
  })();
  return loaded;
}
