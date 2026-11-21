import type { DayDemand, LineDemand, StationDemand } from "../game/demand";

/**
 * T-2624: one step of a local result as its periods come in (the morning peak first). The lines in `ids` take
 * the solved periods' figures from `est` (a day's views worked out with every unsolved period's place filled by a
 * solved one, never read); each unsolved period keeps the shown figures, except on a line the edit changed (its
 * stops, trains or cars), where a period with a solved period of its demand level gets that level's change in
 * proportion (riders new / riders shown over the level's solved periods). A line's riders a day and its boardings
 * by stop are summed again from its periods; its stations move by its change. Everything else stays as shown:
 * other lines, the zones, the purposes, the bubbles, the view's version and final flag. `base`: the views shown
 * before the result's first step, which every step starts from (put into `shown`, the views shown now). `levels`: each period's
 * demand level; `layoutNow`/`layoutWas`: each line's stops and run 1's platforms (T-3348) now and as shown.
 * `keep(id)`: whether a line may take the result (T-3303's fares rule).
 */
export function localStep(
  shown: DayDemand, base: DayDemand, est: DayDemand, solved: readonly boolean[], levels: readonly string[], ids: readonly string[],
  layoutNow: ReadonlyMap<string, { stops: string[]; back?: string[] }>, layoutWas: ReadonlyMap<string, { stops: string[]; back?: string[] }>,
  keep: (id: string) => boolean,
): DayDemand {
  const view = shown.view, from = base.view, got = est.view;
  if (!view || !from || !got) return shown;
  const P = solved.length;
  const lines = new Map(view.lines), stations = new Map(view.stations);
  const delta = new Map<string, [number, number]>();
  const add = (id: string, b: number, a: number) => { const d = delta.get(id); if (d) { d[0] += b; d[1] += a; } else delta.set(id, [b, a]); };
  // a line's boardings and alightings a day at its stations, signed
  const stationsOf = (l: LineDemand, at: { stops: string[]; back?: string[] }, sign: number) => {
    const n = at.stops.length;
    for (let q = 0; q < P; q++) {
      const b = l.board?.[q], a = l.alight?.[q];
      if (!b || !a || b.length !== 2 * n) continue;
      for (let k = 0; k < n; k++) {
        const i = k, j = 2 * n - 1 - k, sa = at.stops[k], sb = at.back?.[k] ?? sa;
        add(sa, sign * b[i], sign * a[i]);
        add(sb, sign * b[j], sign * a[j]);
      }
    }
  };
  const sum = (v: Float32Array | undefined) => { let s = 0; if (v) for (let k = 0; k < v.length; k++) s += v[k]; return s; };
  const scaled = (v: Float32Array, f: number) => { if (f === 1) return v; const o = new Float32Array(v.length); for (let k = 0; k < v.length; k++) o[k] = v[k] * f; return o; };
  for (const id of ids) {
    const was = from.lines.get(id), now = got.lines.get(id), at = layoutNow.get(id);
    // (a line with no shown figures, new or from a cached result, waits for the whole local result)
    if (!was?.board || !was.alight || !was.loads || !was.stops || !now?.board || !now.alight || !now.loads || !at || !keep(id)) continue;
    if (was.board.length !== P || now.board.length !== P) continue;
    const sameStops = was.stops.join() === at.stops.join();
    const edited = !sameStops || (!!was.tph && !!now.tph && was.tph.join() !== now.tph.join()) || (was.cars !== undefined && now.cars !== undefined && was.cars !== now.cars);
    const riders = (l: LineDemand, q: number) => sum(l.board![q]);
    const level = new Map<string, { now: number; was: number; q: number }>();
    for (let q = 0; q < P; q++) {
      if (!solved[q]) continue;
      const v = level.get(levels[q]) ?? { now: 0, was: 0, q };
      v.now += riders(now, q); v.was += riders(was, q);
      level.set(levels[q], v);
    }
    const first = solved.indexOf(true);
    const board: Float32Array[] = [], alight: Float32Array[] = [], arrive: Float32Array[] = [], loads: Float32Array[] = [];
    const fareX: number[] = [], baseX: number[] = [];
    for (let q = 0; q < P; q++) {
      if (solved[q]) {
        board.push(now.board[q]); alight.push(now.alight[q]); arrive.push((now.arrive ?? now.board)[q]); loads.push(now.loads[q]);
        fareX.push(now.fareX?.[q] ?? 0); baseX.push(now.baseX?.[q] ?? 0);
        continue;
      }
      const lv = level.get(levels[q]);
      const f = edited && lv && lv.was > 0 ? lv.now / lv.was : 1;
      fareX.push((was.fareX?.[q] ?? 0) * f); baseX.push((was.baseX?.[q] ?? 0) * f);
      if (sameStops && was.loads[q]?.length === now.loads[q]?.length) {
        board.push(scaled(was.board[q], f)); alight.push(scaled(was.alight[q], f)); arrive.push(scaled((was.arrive ?? was.board)[q], f)); loads.push(scaled(was.loads[q], f));
      } else {
        // on new stops: a solved period's figures (its level's, else the first), sized to the shown period's riders
        const from = lv?.q ?? first, g = riders(now, from) > 0 ? (riders(was, q) * f) / riders(now, from) : 0;
        board.push(scaled(now.board[from], g)); alight.push(scaled(now.alight[from], g)); arrive.push(scaled((now.arrive ?? now.board)[from], g)); loads.push(scaled(now.loads[from], g));
      }
    }
    const n = at.stops.length, boardings = at.stops.map(() => 0);
    for (const b of board) for (let k = 0; k < n; k++) boardings[k] += b[k] + b[2 * n - 1 - k];
    const allHigh = levels.every((l, q) => l !== "high" || solved[q]);
    const line: LineDemand = { ...now, ridersPerDay: boardings.reduce((x, y) => x + y, 0), fullestPct: allHigh ? now.fullestPct : was.fullestPct, boardings, loads, board, alight, arrive, fareX, baseX };
    lines.set(id, line);
    const wasAt = layoutWas.get(id);
    stationsOf(was, wasAt?.stops.join() === was.stops.join() ? wasAt : { stops: was.stops }, -1);
    stationsOf(line, at, 1);
  }
  // (only stations the result has: the whole local result replaces every one of them)
  for (const [id, [b, a]] of delta) {
    if (!got.stations.has(id)) continue;
    const s = from.stations.get(id) ?? { boardings: 0, alightings: 0 };
    const out: StationDemand = { ...s, boardings: Math.max(0, s.boardings + b), alightings: Math.max(0, s.alightings + a) };
    stations.set(id, out);
  }
  const v = { ...view, lines, stations };
  const cities = new Map(shown.cities);
  for (const [id, c] of shown.cities) cities.set(id, { ...c, lines, stations });
  return { ...shown, view: v, cities };
}
