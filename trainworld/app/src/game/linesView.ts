// T-3816, T-3799: the Lines tab's gear: the kinds of line and the cities it lists, and the columns it shows.
// Remembered per browser (localStorage), never in the save. What is kept is what the player turned off, so a
// game with other cities or kinds lists them all to start with.

import { effect, signal } from "@preact/signals";
import { VIDEO } from "./ui";

/** The columns a player may turn off; their ids are the sort keys. Name always shows; Near shows while sorted on. */
export type LineCol = "city" | "riders" | "money" | "trains" | "kind";
export const LINE_COLS: readonly LineCol[] = ["city", "riders", "money", "trains", "kind"];

export interface LinesView {
  /** the line types turned off, as the Type column's ranks (0 train, 1 high speed, 2 tram, 3 bus, 4 ferry) */
  modes: readonly number[];
  /** the cities (zone ids) turned off */
  cities: readonly string[];
  cols: readonly LineCol[];
}

const KEY = "trainworld-lines-view";
const EMPTY: LinesView = { modes: [], cities: [], cols: [] };

function load(): LinesView {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) || "null") as Partial<LinesView> | null;
    if (!v || typeof v !== "object") return EMPTY;
    return {
      modes: Array.isArray(v.modes) ? v.modes.filter((m) => Number.isInteger(m) && m >= 0 && m <= 4) : [],
      cities: Array.isArray(v.cities) ? v.cities.filter((c) => typeof c === "string") : [],
      cols: Array.isArray(v.cols) ? v.cols.filter((c) => LINE_COLS.includes(c)) : [],
    };
  } catch {
    return EMPTY;
  }
}

export const linesView = signal<LinesView>(VIDEO ? EMPTY : load());

effect(() => {
  const v = linesView.value;
  if (VIDEO) return;
  try { localStorage.setItem(KEY, JSON.stringify(v)); } catch { /* this page only */ }
});

/** Turn `vals` of one list on (`off` false) or off. */
export function setOff<K extends keyof LinesView>(k: K, vals: LinesView[K], off: boolean) {
  const cur = linesView.peek()[k] as readonly (string | number)[];
  const drop = new Set<string | number>(vals);
  const next = off ? [...cur.filter((x) => !drop.has(x)), ...drop] : cur.filter((x) => !drop.has(x));
  linesView.value = { ...linesView.peek(), [k]: next };
}
