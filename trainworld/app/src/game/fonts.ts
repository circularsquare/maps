// The panels' font choices (T-698, a trial). Zen Maru Gothic is the page's own font, loaded by
// index.html; the others are fetched from Google Fonts (all SIL OFL) only when chosen, and Nunito,
// which has no Japanese, falls back to Zen Maru Gothic for kana and kanji. The map's own labels are
// glyph tiles and stay Zen Maru Gothic.

export type FontKey = "zen" | "mplus" | "kosugi" | "nunito";

export interface FontChoice {
  label: string;
  /** the CSS family */
  family: string;
  /** Google Fonts css2 query for this family; null when index.html already loads it */
  query: string | null;
}

export const FONTS: Record<FontKey, FontChoice> = {
  zen: { label: "Zen Maru Gothic", family: '"Zen Maru Gothic"', query: null },
  mplus: { label: "M PLUS Rounded 1c", family: '"M PLUS Rounded 1c"', query: "family=M+PLUS+Rounded+1c:wght@400;700" },
  kosugi: { label: "Kosugi Maru", family: '"Kosugi Maru"', query: "family=Kosugi+Maru" },
  nunito: { label: "Nunito", family: '"Nunito"', query: "family=Nunito:wght@400;700" },
};
export const FONT_KEYS = Object.keys(FONTS) as FontKey[];

/** the `--font` value for a choice: its family, then the page font for what it lacks */
export const fontStack = (k: FontKey) => (k === "zen" ? `"Zen Maru Gothic", system-ui, sans-serif` : `${FONTS[k].family}, "Zen Maru Gothic", system-ui, sans-serif`);

const sheets = new Map<FontKey, Promise<void>>();
/** Add the choice's stylesheet once; resolves when its faces for the page's text have loaded. */
export async function loadFont(k: FontKey): Promise<void> {
  const q = FONTS[k].query;
  if (q && !sheets.has(k)) {
    sheets.set(k, new Promise<void>((done) => {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = `https://fonts.googleapis.com/css2?${q}&display=swap`;
      link.onload = link.onerror = () => done();
      document.head.appendChild(link);
    }));
  }
  try {
    await sheets.get(k);
    await Promise.all([`14px ${FONTS[k].family}`, `700 14px ${FONTS[k].family}`].map((f) => document.fonts?.load(f, "Aあ漢")));
  } catch {
    // offline: the fallback stays
  }
}
