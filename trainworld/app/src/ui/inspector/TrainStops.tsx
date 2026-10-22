// T-868: a selected train's next few stops with the riders getting off at each. The train's own
// loads on each stretch come from the boarding account (game/boarding.ts); the riders getting off
// at a stop are its load arriving times the line's alighting share there, as the account itself
// lets them off (`alight_shares` in sim/src/track/boarding.rs), so the figures agree with the
// train's loads: arriving less getting off plus boarding is the load leaving.

import type { BoardHour } from "../../game/boarding";
import { tripLoad } from "../../game/boarding";
import { clock } from "../../game/clock";
import { duration } from "../../game/format";

const count = (v: number) => Math.round(v).toLocaleString("en-US");
import { periodAt, PERIODS } from "../../game/periods";
import type { LineDemand } from "../../game/demand";
import type { LineInput } from "../../game/types";
import { world } from "../../game/world";
import { profileS } from "../../map/network";
import { complexes } from "../../game/complexView";
import { membersOf } from "../../game/complexes";
import { Chip } from "../widgets";
import { live } from "../../map/live";

/** Stops listed ahead of the train. */
const AHEAD = 4;

/** The share of the riders arriving at each stop of run `r` who get off there, in period `p`
 * (travel order): demand's alightings over the riders it has arriving, the day's where none
 * arrive in the period. A port of `alight_shares` in boarding.rs. */
function alightShares(dm: LineDemand, n: number, r: number, wrap: boolean): number[][] {
  const out: number[][] = [];
  const dayIn = new Array(n).fill(0), dayOff = new Array(n).fill(0);
  for (let p = 0; p < PERIODS.length; p++) {
    const b = dm.board?.[p], a = dm.alight?.[p];
    const row = new Array(n).fill(-1);
    out.push(row);
    if (!b || !a) continue;
    const at = (v: Float32Array, k: number) => Math.max(0, v[r * n + k] ?? 0);
    let on = 0;
    if (wrap) {
      let x = 0, low = 0;
      for (let k = 0; k < n; k++) { x -= at(a, k); low = Math.min(low, x); x += at(b, k); }
      on = -low;
    }
    for (let k = 0; k < n; k++) {
      const off = at(a, k);
      dayIn[k] += on;
      dayOff[k] += off;
      row[k] = on > 1e-6 ? Math.min(1, Math.max(0, off / on)) : -1;
      on = Math.max(0, on + at(b, k) - off);
    }
  }
  for (const row of out) {
    for (let k = 0; k < n; k++) if (row[k] < 0) row[k] = dayIn[k] > 1e-6 ? Math.min(1, Math.max(0, dayOff[k] / dayIn[k])) : 0;
    if (!wrap) row[n - 1] = 1;
  }
  return out;
}

/** Seconds into the trip at which it reaches offset `s` (m along its run). */
function reachAt(profile: number, s: number, from: number, to: number): number {
  const net = live.overlay?.renderer.net;
  if (!net) return NaN;
  let lo = from, hi = to;
  if (profileS(net, profile, hi) < s) return NaN;
  for (let k = 0; k < 40 && hi - lo > 0.5; k++) {
    const mid = (lo + hi) / 2;
    if (profileS(net, profile, mid) >= s) hi = mid;
    else lo = mid;
  }
  return hi;
}

export function TrainStops({ line, dm, board, trip, run, seg, stopS, profile, dep }: {
  line: LineInput; dm: LineDemand; board: BoardHour; trip: number; run: number; seg: number; stopS: number[]; profile: number; dep: number;
}) {
  const w = world.value;
  const n = line.stops.length;
  if (!w || !dm.board || !dm.alight || n < 2) return null;
  const wrap = w.lineStats[line.id]?.loop === "circle";
  // a circle's trip ends at its first stop again and carries on round as the next trip
  const last = wrap ? n - 2 : n - 1;
  const names = new Map(w.save.stations.map((s) => [s.id, s.name]));
  const stops = run === 0 ? line.stops : [...line.stops].reverse();
  const shares = alightShares(dm, n, run, wrap);
  const load = (t: number) => (t < 0 || t >= n - 1 ? 0 : tripLoad(board, trip, run === 0 ? t : n - 2 - t) ?? 0);
  const net = live.overlay?.renderer.net;
  const tripS = net ? net.meta[profile * 4 + 2] : 0;
  const now = clock.now() - dep;
  const shift = w.lineStats[line.id]?.shift ?? 0;
  // T-886: the other lines calling at each stop or its complex, for changing trains
  const cx = complexes.value, linesAt = new Map<string, LineInput[]>();
  for (const l of w.save.lines) for (const s of new Set(l.stops)) if (l.id !== line.id) linesAt.set(s, [...(linesAt.get(s) ?? []), l]);
  const transfers = (s: string) => [...new Map(membersOf(cx, s).flatMap((m) => linesAt.get(m) ?? []).map((l) => [l.id, l])).values()];
  const rows: { name: string; inS: number; off: number; on: number; arriving: number; last: boolean; lines: LineInput[] }[] = [];
  for (let k = seg + 1; k <= last && rows.length < AHEAD; k++) {
    const arriving = load(k - 1);
    const at = reachAt(profile, (stopS[k] ?? 0) - 1, Math.max(0, now), tripS);
    const inS = Number.isFinite(at) ? Math.max(0, at - now) : NaN;
    const p = periodAt(clock.minute.value * 60 + shift + (Number.isFinite(inS) ? inS : 0));
    const end = k === n - 1;
    const off = end ? arriving : arriving * (shares[p]?.[k] ?? 0);
    const on = end ? 0 : Math.max(0, load(k) - (arriving - off));
    rows.push({ name: names.get(stops[k]) ?? "a station", inS, off, on, arriving, last: end, lines: transfers(stops[k]) });
  }
  if (!rows.length) return null;
  return (
    <>
      <div class="h">Next stops<span class="right muted small">getting off</span></div>
      <ul class="rows train-stops">
        {rows.map((r) => (
          <li data-tip={r.last
            ? `Everyone aboard gets off at ${r.name}, the end of the line`
            : `${count(r.off)} of the ${count(r.arriving)} riders aboard get off at ${r.name}, and ${count(r.on)} get on`}>
            <span class="grow stop-name"><span class="name">{r.name}</span>{r.lines.map((l) => <Chip line={l} size="sm" />)}</span>
            {Number.isFinite(r.inS) && (
              <span class="muted small" data-tip={`Arrives in ${duration(r.inS)}`}>{r.inS < 60 ? "<1 min" : `${Math.round(r.inS / 60)} min`}</span>
            )}
            <span class="val w-num" data-tip={`${count(r.off)} ${Math.round(r.off) === 1 ? "rider gets" : "riders get"} off at ${r.name}`}>{count(r.off)}</span>
          </li>
        ))}
      </ul>
    </>
  );
}
