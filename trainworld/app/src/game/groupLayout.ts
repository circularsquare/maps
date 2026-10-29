// T-1791 stage 2: the group view's combined diagram, laid out. Rows top to bottom, one column per
// line. A row is a slot, not a station: each column has at most one station on it (a stop or a
// station it passes), so a row's stations can differ by column and the hover names them. Rows
// start from the group's line with most stops (the spine) in its path order; each other line is
// matched to the rows by place (a station complex, `key`), turned round if it runs the other way,
// and its own places go in new rows where it leaves the rows it shares. A line sharing nothing
// takes the rows from the top, so unrelated lines stand side by side. Pure, for checks/t1791-layout.mjs.

/** One place on a line's path, in path order: a stop or a station it passes. */
export interface PathPlace { key: string; station: string; stop: boolean; /** the stop's index in its line's stops, -1 for a station passed */ index: number }
export interface GroupLineIn { num: number; places: PathPlace[] }
export interface GroupCell { station: string; stop: boolean; index: number }
export interface GroupLayout {
  /** rows, each a cell per column (by the input's order) or null where the line has no place there */
  rows: (GroupCell | null)[][];
  /** per column, the first and last rows its bar runs between (-1, -1 for a line with no places) */
  spans: [number, number][];
  /** per column, the rows its bar does not run into from its previous place (a branch moved away) */
  breaks: number[][];
}

/** A line's places from its stops and the stations it passes (`[gap, node]`, gap i after stop i). */
export function pathPlaces(stops: readonly string[], passed: readonly { gap: number; node: number }[], key: (station: string) => string): PathPlace[] {
  const out: PathPlace[] = [];
  stops.forEach((s, i) => {
    out.push({ key: key(s), station: s, stop: true, index: i });
    for (const p of passed) if (p.gap === i) out.push({ key: key(String(p.node)), station: String(p.node), stop: false, index: -1 });
  });
  return out;
}

/** A loop's path ends where it began: its places once round. */
function onceRound(p: PathPlace[]): PathPlace[] {
  return p.length > 2 && p[0].key === p[p.length - 1].key ? p.slice(0, -1) : p;
}

export function groupLayout(lines: GroupLineIn[]): GroupLayout {
  const n = lines.length;
  const rows: (GroupCell | null)[][] = [];
  const keys: (string | null)[] = []; // a row's place, the first a line put there; null for none
  const empty = () => Array<GroupCell | null>(n).fill(null);
  if (!n) return { rows, spans: [], breaks: [] };
  const breaks: number[][] = lines.map(() => []);
  const order = lines.map((_, k) => k);
  const stops = (k: number) => lines[k].places.filter((p) => p.stop).length;
  const spine = order.reduce((b, k) => (stops(k) > stops(b) ? k : b), 0);
  for (const p of onceRound(lines[spine].places)) {
    const r = empty();
    r[spine] = { station: p.station, stop: p.stop, index: p.index };
    rows.push(r);
    keys.push(p.key);
  }
  for (const k of order) {
    if (k === spine) continue;
    let places = onceRound(lines[k].places);
    const at = (p: PathPlace) => keys.indexOf(p.key);
    // turned round when its shared places run up the rows
    const hits = places.map(at).filter((r) => r >= 0);
    if (hits.length >= 2 && hits[0] > hits[hits.length - 1]) places = [...places].reverse();
    if (!places.some((p) => at(p) >= 0)) {
      // nothing shared: down the rows from the top, side by side
      places.forEach((p, i) => {
        if (i >= rows.length) { rows.push(empty()); keys.push(null); }
        rows[i][k] = { station: p.station, stop: p.stop, index: p.index };
      });
      continue;
    }
    let last = -1;
    let pending: PathPlace[] = [];
    const put = (before: number) => {
      // its own places since the last shared row go in new rows just before `before`
      pending.forEach((p, j) => {
        const r = empty();
        r[k] = { station: p.station, stop: p.stop, index: p.index };
        rows.splice(before + j, 0, r);
        keys.splice(before + j, 0, p.key);
      });
      const added = pending.length;
      pending = [];
      for (const b of breaks) b.forEach((x, i) => { if (x >= before) b[i] = x + added; });
      return added;
    };
    for (const p of places) {
      const r = keys.indexOf(p.key, last + 1);
      if (r < 0 || rows[r][k]) { pending.push(p); continue; }
      last = r + put(r);
      rows[last][k] = { station: p.station, stop: p.stop, index: p.index };
    }
    // a branch past the shared rows goes after all the rows, its column broken where it leaves
    const away = pending.length > 0 && rows.length > last + 1;
    const end = rows.length;
    put(away ? end : last + 1);
    if (away) breaks[k].push(end);
  }
  const spans = lines.map((_, k): [number, number] => {
    let a = -1, b = -1;
    rows.forEach((r, i) => { if (r[k]) { if (a < 0) a = i; b = i; } });
    return [a, b];
  });
  return { rows, spans, breaks };
}
