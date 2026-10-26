// The related-lines diagram's layout (T-1168, notes/T-1168.md "Layout"): rows, columns, turns,
// stop pills and hover texts, from `decodeRelated`'s groups. Pure, so a Node check can run it
// (checks/t1168-layout.mjs); drawn by ui/inspector/RelatedStops.tsx.

import type { RelatedGroup } from "./relatedLines";

/** Column pitch, line width, a gutter with bullets and one without, px. A line on its own track
 * sits OWN_GAP further out than one sharing the selected line's track (Anita, 2026-10-24). */
export const COL = 8, OWN_GAP = 3, BAR = 6, GUT = 28, NO_GUT = 6;
/** Turns leaving at one gap fan out this far apart; their bullets stack at least this far apart, px. */
const FAN = 7, CHIP_GAP = 17;
/** At most this many columns a side, and only as many as fit in this share of the panel's width. */
const MAX_SIDE = 3, WIDTH_SHARE = 0.4;
/** A station of the other line's within this of a row of the same complex joins that row, m. */
const SAME_ROW_M = 400;

export interface LayoutOpts {
  /** The selected line's number, stop count and each stop's offset along its first run, m. */
  line: number;
  stopS: ArrayLike<number>;
  /** The panel's inner width, px: the columns take at most 40% of it. */
  width: number;
  /** Centre of row `i`, px (the panel measures its rows); default 25 px rows. */
  rowY?: (i: number) => number;
  lineName: (line: number) => string;
  /** A line's stop name, the selected line's included. */
  stopName: (line: number, stop: number) => string;
  stopCount: (line: number) => number;
  dist?: (m: number) => string;
}

export interface LayoutRow {
  /** The selected line's stop here, or -1 for a row only side lines stop at. */
  lStop: number;
  /** Whose stop names the row. */
  ref: { line: number; stop: number };
  sL: number;
  /** Lines met here by this classification that are not drawn as a column here. */
  cross: number[];
  /** Lines drawn as columns over this row. */
  drawn: number[];
  y: number;
}

export interface LayoutColumn { line: number; side: "L" | "R"; x: number; r0: number; r1: number; up: boolean; down: boolean; d: string; tip: string }
export interface LayoutBullet { line: number; x: number; y: number; tip: string }
export interface LayoutPill { row: number; x0: number; x1: number; big: boolean; tip: string }

export interface RelatedLayout {
  rows: LayoutRow[];
  width: number;
  spineX: number;
  spine: [number, number];
  columns: LayoutColumn[];
  bullets: LayoutBullet[];
  pills: LayoutPill[];
  /** The hover of a crossing bar at a row, by line, where the classification has words for it. */
  crossTip: (row: number, line: number) => string;
}

interface Row { lStop: number; ref: { line: number; stop: number }; sL: number; complex: number; stops: Set<number>; cross: number[] }
interface Item { g: number; line: number; r0: number; r1: number; up: boolean; down: boolean; shared: number; own: number; side?: "L" | "R"; slot?: number }

const km = (m: number) => `${(m / 1000).toFixed(1)} km`;

function andList(xs: string[]): string {
  return xs.length <= 1 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`;
}
const stops = (n: number) => `${n} ${n === 1 ? "stop" : "stops"}`;

export function relatedLayout(groups: RelatedGroup[], o: LayoutOpts): RelatedLayout {
  const L = o.line, dist = o.dist ?? km;
  const rowY = o.rowY ?? ((i: number) => i * 25 + 12.5);
  const me = o.lineName(L);
  // ---- rows: the selected line's stops, then side lines' stations at the nearest point of it
  const rows: Row[] = Array.from({ length: o.stopS.length }, (_, i) => ({ lStop: i, ref: { line: L, stop: i }, sL: o.stopS[i], complex: -1, stops: new Set<number>(), cross: [] }));
  for (const g of groups) for (const m of g.meetings) if (m.stopL >= 0 && rows[m.stopL]) rows[m.stopL].complex = m.complex;
  const rowAt = (complex: number, sL: number, make: () => Row | null): Row | null => {
    let best: Row | null = null;
    for (const r of rows) if (r.complex === complex && Math.abs(r.sL - sL) <= SAME_ROW_M && (!best || Math.abs(r.sL - sL) < Math.abs(best.sL - sL))) best = r;
    if (best) return best;
    const r = make();
    if (r) rows.push(r);
    return r;
  };
  const beside = groups.map((g, gi) => [g, gi] as const).filter(([g]) => g.kind === "beside");
  for (const [g, gi] of beside) {
    for (const m of g.meetings) {
      if (m.stopO < 0) continue;
      const r = m.stopL >= 0 ? rows[m.stopL] : rowAt(m.complex, m.sL, () => ({ lStop: -1, ref: { line: g.line, stop: m.stopO }, sL: m.sL, complex: m.complex, stops: new Set(), cross: [] }));
      r?.stops.add(gi);
    }
    for (const s of g.own) rowAt(s.complex, s.sL, () => ({ lStop: -1, ref: { line: g.line, stop: s.stop }, sL: s.sL, complex: s.complex, stops: new Set(), cross: [] }))?.stops.add(gi);
  }
  const sortRows = (rs: Row[]) => rs.sort((a, b) => a.sL - b.sL || (a.lStop >= 0 && b.lStop >= 0 ? a.lStop - b.lStop : 0));
  sortRows(rows);

  // ---- columns: each beside group over the rows it stops at and those inside its span
  const itemsOn = (rs: Row[]): Item[] => beside.map(([g, gi]) => {
    let r0 = Infinity, r1 = -Infinity;
    rs.forEach((r, i) => { if (r.stops.has(gi) || (r.sL >= g.sL0 - 1 && r.sL <= g.sL1 + 1)) { r0 = Math.min(r0, i); r1 = Math.max(r1, i); } });
    // the other line goes on past the span's top (towards the first stop) or bottom
    const [before, after] = g.dir > 0 ? [g.stopsBefore, g.stopsAfter] : [g.stopsAfter, g.stopsBefore];
    let shared = 0, own = 0;
    for (let i = r0; i <= r1; i++) if (rs[i]?.stops.has(gi)) { if (rs[i].lStop >= 0) shared++; else own++; }
    return { g: gi, line: g.line, r0, r1, up: before > 0, down: after > 0, shared, own };
  });
  let items = itemsOn(rows);
  // a side line beside the spine over fewer than three rows is drawn as crossings at its rows
  const asCrossing = (it: Item, rs: Row[]) => {
    const g = groups[it.g];
    for (const r of rs) if ((r.stops.has(it.g) || g.meetings.some((m) => m.complex === r.complex && Math.abs(m.sL - r.sL) <= SAME_ROW_M)) && !r.cross.includes(it.line)) r.cross.push(it.line);
  };
  const short = items.filter((it) => !(it.r1 - it.r0 >= 2));
  for (const it of short) asCrossing(it, rows);
  for (const r of rows) for (const it of short) r.stops.delete(it.g);
  // rows nobody drawn stops at are left out
  const kept = rows.filter((r) => r.lStop >= 0 || r.stops.size > 0);
  items = itemsOn(kept).filter((it) => Number.isFinite(it.r0) && !short.some((s) => s.g === it.g));
  for (const g of groups) {
    if (g.kind !== "crossing") continue;
    for (const m of g.meetings) {
      const r = kept.find((x) => x.complex === m.complex && Math.abs(x.sL - m.sL) <= SAME_ROW_M);
      if (r && !r.cross.includes(g.line)) r.cross.push(g.line);
    }
  }
  // sides: a line stopping more often than the spine over its span goes right, as often or less
  // left; most stops in common nearest; the other side when its own is full
  items.sort((a, b) => b.shared - a.shared || (b.r1 - b.r0) - (a.r1 - a.r0));
  const perSide = Math.max(1, Math.min(MAX_SIDE, Math.floor((o.width * WIDTH_SHARE - 2 * GUT - 2 * NO_GUT) / (2 * COL))));
  const taken: Record<string, [number, number][]> = {};
  const over: Item[] = [];
  for (const it of items) {
    let skipped = 0;
    for (let i = it.r0; i <= it.r1; i++) if (kept[i].lStop >= 0 && !kept[i].stops.has(it.g)) skipped++;
    const pref: "L" | "R" = it.own > skipped ? "R" : "L";
    const a = it.r0 - (it.up ? 0.6 : 0.1), b = it.r1 + (it.down ? 0.6 : 0.1);
    const order: ["L" | "R", number][] = [];
    for (let k = 1; k <= perSide; k++) order.push([pref, k]);
    for (let k = 1; k <= perSide; k++) order.push([pref === "L" ? "R" : "L", k]);
    const slot = order.find(([s, k]) => (taken[s + k] ?? []).every(([x0, x1]) => b < x0 || a > x1));
    if (!slot) { over.push(it); continue; }
    (taken[slot[0] + slot[1]] ??= []).push([a, b]);
    [it.side, it.slot] = slot;
  }
  for (const it of over) asCrossing(it, kept);
  items = items.filter((it) => it.side);

  // ---- geometry
  const off = (it: Item) => it.slot! * COL + (groups[it.g].shared ? 0 : OWN_GAP);
  const nL = Math.max(0, ...items.filter((it) => it.side === "L").map(off));
  const nR = Math.max(0, ...items.filter((it) => it.side === "R").map(off));
  const X0 = (nL ? GUT : NO_GUT) + nL + NO_GUT;
  const W = X0 + nR + NO_GUT + (nR ? GUT : NO_GUT);
  const xOf = (it: Item) => X0 + (it.side === "R" ? 1 : -1) * off(it);
  const gx = (side: "L" | "R") => (side === "L" ? 14 : W - 14);
  const yAt = (r: number) => {
    const n = kept.length, step = n > 1 ? (rowY(n - 1) - rowY(0)) / (n - 1) : 25;
    if (r <= 0) return rowY(0) + r * step;
    if (r >= n - 1) return rowY(n - 1) + (r - n + 1) * step;
    const i = Math.floor(r);
    return rowY(i) + (r - i) * (rowY(i + 1) - rowY(i));
  };
  const name = (i: number) => o.stopName(kept[i].ref.line, kept[i].ref.stop);

  // hover texts
  const colTip = (it: Item) => {
    const g = groups[it.g], on = o.lineName(it.line);
    const t = [`${on} runs beside ${me} from ${name(it.r0)} to ${name(it.r1)}`, `${stops(it.shared)} in common`];
    if (it.own) t.push(`${stops(it.own)} of its own`);
    if (g.shared) t.push("On the same track");
    const gaps = g.gaps.filter((x) => x.run > 1);
    if (gaps.length) {
      const run = gaps.reduce((a, x) => a + x.run, 0), ls = gaps.reduce((a, x) => a + x.lStep, 0), st = Math.max(...gaps.map((x) => x.stray));
      t.push(`${dist(run)} on its own track beside ${dist(ls)} of ${me}`, `Up to ${Math.round(st)} m from ${me}'s track`);
    }
    return t.join("\n");
  };
  // the same line beside again further along, in its own order
  const sameLine = (it: Item, later: boolean) => {
    const g = groups[it.g];
    const others = items.filter((j) => j !== it && j.line === it.line && (groups[j.g].sO0 > g.sO0) === later);
    return others.sort((a, b) => (later ? groups[a.g].sO0 - groups[b.g].sO0 : groups[b.g].sO0 - groups[a.g].sO0))[0];
  };
  const turnTip = (it: Item, end: "up" | "down") => {
    const g = groups[it.g], on = o.lineName(it.line), n = o.stopCount(it.line);
    // towards the other line's own end: down the diagram is on along it when it runs our way
    const later = (end === "down") === (g.dir > 0);
    const back = sameLine(it, later);
    if (back) {
      const b = groups[back.g];
      const between = later ? b.stopsBefore - (n - g.stopsAfter) : g.stopsBefore - (n - b.stopsAfter);
      const at = end === "down" ? name(back.r0) : name(back.r1);
      return end === "down" ? `${on} leaves here\nBack beside ${me} at ${at} after ${stops(Math.max(0, between))} of its own`
        : `${on} joins here\nIt left ${me} at ${at} ${stops(Math.max(0, between))} back`;
    }
    const past = later ? g.stopsAfter : g.stopsBefore;
    const term = o.stopName(it.line, later ? n - 1 : 0);
    return `${on} goes on to ${term}\n${past} more ${past === 1 ? "stop" : "stops"}`;
  };

  // turns out to the gutter, by side and gap: leaving first, outermost first, then joining,
  // innermost first, FAN px apart so a bundle leaving together never crosses itself
  const turns = new Map<string, { side: "L" | "R"; gap: number; ev: [number, number, Item, "up" | "down"][] }>();
  for (const it of items) {
    for (const end of ["up", "down"] as const) {
      if (!it[end]) continue;
      const gap = end === "up" ? it.r0 - 0.5 : it.r1 + 0.5, key = `${it.side}${gap}`;
      const e = turns.get(key) ?? { side: it.side!, gap, ev: [] };
      e.ev.push(end === "up" ? [1, it.slot!, it, end] : [0, -it.slot!, it, end]);
      turns.set(key, e);
    }
  }
  const turnY = new Map<string, number>();
  const chips: { side: "L" | "R"; y: number; it: Item; tip: string }[] = [];
  for (const { side, gap, ev } of turns.values()) {
    ev.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const mid = yAt(gap), n = ev.length;
    ev.forEach(([, , it, end], j) => {
      turnY.set(`${it.g}${end}`, mid + (j - (n - 1) / 2) * FAN);
      chips.push({ side, y: mid + (j - (n - 1) / 2) * CHIP_GAP, it, tip: turnTip(it, end) });
    });
  }
  const columns: LayoutColumn[] = [];
  for (const it of [...items].sort((a, b) => b.slot! - a.slot!)) {
    const x = xOf(it), s = it.side === "L" ? -1 : 1, g0 = gx(it.side!);
    const yUp = turnY.get(`${it.g}up`), yDown = turnY.get(`${it.g}down`);
    const yEnd = yDown !== undefined ? yDown - 5 : yAt(it.r1);
    let d = yUp !== undefined ? `M${g0},${yUp} H${x + s * 5} Q${x},${yUp} ${x},${yUp + 5} V${yEnd}` : `M${x},${yAt(it.r0)} V${yEnd}`;
    if (yDown !== undefined) d += ` M${x},${yDown - 5} Q${x},${yDown} ${x + s * 5},${yDown} H${g0}`;
    columns.push({ line: it.line, side: it.side!, x, r0: it.r0, r1: it.r1, up: it.up, down: it.down, d, tip: colTip(it) });
  }
  for (const it of items) if (!it.up && !it.down) chips.push({ side: it.side!, y: yAt(it.r0), it, tip: colTip(it) });
  const last = { L: -Infinity, R: -Infinity };
  const bullets: LayoutBullet[] = chips.sort((a, b) => a.y - b.y).map((c) => {
    const y = Math.max(c.y, last[c.side] + CHIP_GAP);
    last[c.side] = y;
    return { line: c.it.line, x: gx(c.side), y, tip: c.tip };
  });

  // stop pills: one over each run of neighbouring columns that stop at the row
  const lRows = kept.map((r, i) => (r.lStop >= 0 ? i : -1)).filter((i) => i >= 0);
  const spine: [number, number] = [lRows[0] ?? 0, lRows[lRows.length - 1] ?? 0];
  const xs = [{ x: X0, it: null as Item | null }, ...items.map((it) => ({ x: xOf(it), it }))].sort((a, b) => a.x - b.x);
  const pills: LayoutPill[] = [];
  kept.forEach((r, i) => {
    const on = xs.filter(({ it }) => (it ? it.r0 <= i && i <= it.r1 : spine[0] <= i && i <= spine[1]));
    const stopsHere = (it: Item | null) => (it ? r.stops.has(it.g) : r.lStop >= 0);
    const who = (it: Item | null) => o.lineName(it ? it.line : L);
    const stopping = on.filter(({ it }) => stopsHere(it)).map(({ it }) => who(it));
    const passing = on.filter(({ it }) => !stopsHere(it)).map(({ it }) => who(it));
    const tip = `${andList(stopping)} ${stopping.length === 1 ? "stops" : "stop"} at ${name(i)}` + (passing.length ? `\n${andList(passing)} ${passing.length === 1 ? "passes" : "pass"}` : "");
    let run: number[] = [];
    for (const c of [...on, null]) {
      if (c && stopsHere(c.it)) run.push(c.x);
      else if (run.length) {
        pills.push({ row: i, x0: run[0], x1: run[run.length - 1], big: r.cross.length > 0, tip });
        run = [];
      }
    }
  });

  const out: LayoutRow[] = kept.map((r, i) => ({
    lStop: r.lStop, ref: r.ref, sL: r.sL, cross: r.cross,
    drawn: items.filter((it) => it.r0 <= i && i <= it.r1).map((it) => it.line), y: yAt(i),
  }));
  const crossTip = (i: number, line: number) => {
    const t = [`${o.lineName(line)} crosses at ${name(i)}`];
    const r = kept[i];
    const gs = groups.map((g, gi) => [g, gi] as const).filter(([g]) => g.line === line).sort((a, b) => a[0].sO0 - b[0].sO0);
    const k = gs.findIndex(([g]) => g.kind === "crossing" && g.meetings.some((m) => m.complex === r.complex));
    const g = gs[k]?.[0];
    if (g?.split && Number.isFinite(g.split.stray) && k > 0) {
      const prev = gs[k - 1][0].meetings.at(-1);
      const at = prev ? (prev.stopL >= 0 ? o.stopName(L, prev.stopL) : prev.stopO >= 0 ? o.stopName(line, prev.stopO) : "") : "";
      t.push(`It left ${me}${at ? " at " + at : ""} ${dist(g.split.run)} back`, `Its track strays up to ${dist(g.split.stray)} from ${me}'s in between`);
    }
    return t.join("\n");
  };
  return { rows: out, width: W, spineX: X0, spine, columns, bullets, pills, crossTip };
}
