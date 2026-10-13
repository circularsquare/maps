// The network tag's text for trains, stations, platform crowds and the station demand circles
// (T-194, Anita: "we probably shouldn't have much that is drawn but does not have a tooltip
// explaining quantitatively what is drawn"). trackHover.ts shows it, after the same rest as the
// track's: each says in words the number the drawing stands for.
//
// - A train: its line, and the riders aboard against seats and packed full (the gauge its core is
//   filled with), from the boarding account as the train inspector counts them.
// - A station: its name; with demand circles on, the boardings and alightings a day its circle
//   stands for (its complex's in complex view); where platform crowds are drawn, the riders
//   waiting per line and direction and how full the platform is drawn (T-219: the crowd is inside
//   the box now, so the station's own hover covers it). Zoomed out, a complex's one dot is its
//   busiest station's, and says so (T-217).
// - A demand circle away from any station: the same count, for the circle's station or complex.

import { crowdAt, QUEUE_STRIDE, tripLoad } from "../game/boarding";
import { clock } from "../game/clock";
import { complexes, complexOf } from "../game/complexView";
import { demandView, stationRidersFor } from "../game/demand";
import { PERIODS, periodAt } from "../game/periods";
import { purposeName } from "../game/purposes";
import type { WorldView } from "../game/types";
import { display, purposeFilter } from "../game/ui";
import { stationScale } from "./demandLayers";
import { tripAt } from "./network";
import type { Overlay } from "./overlay";
import { stationHaloRadius } from "./stationGeometry";

/** Seats and riders at crush load per 20 m car (sim/src/demand/api.rs; ui/inspector/others.tsx). */
const SEATS_PER_CAR = 44, CRUSH_PER_CAR = 160;

const n0 = (v: number) => Math.round(v).toLocaleString("en-US");
/** A forecast count, rounded as the track tag rounds one (trackHover.ts `riders`). */
const forecast = (n: number) => (n >= 1000 ? Math.round(n / 100) * 100 : n >= 100 ? Math.round(n / 10) * 10 : Math.round(n)).toLocaleString("en-US");

/** "12,400 boardings and alightings a day", with what the purpose filter counts. */
function circleLines(riders: number, members: number): string[] {
  const p = purposeFilter.peek();
  const where = members > 1 ? `, at the complex's ${members} stations` : "";
  const out = [`${forecast(riders)} boardings and alightings a day${where}`];
  if (p !== "all") out.push(p === "longer" ? "Longer trips only" : `${purposeName(p)} trips only`);
  return out;
}

/** The demand circle's members and riders for station `id`: its complex's in complex view. */
function circleOf(w: WorldView, id: string): { members: string[]; riders: number } | null {
  const dv = demandView.peek();
  if (!dv) return null;
  const built = new Set(w.save.stations.filter((s) => s.built).map((s) => s.id));
  let members = [id];
  if (display.peek().complexView) {
    const cx = complexes.peek();
    const k = cx.of.get(id);
    if (k !== undefined) members = cx.members[k].filter((m) => built.has(m));
  }
  let riders = 0;
  for (const m of members) {
    const d = stationRidersFor(dv, m, purposeFilter.peek());
    if (d) riders += d.boardings + d.alightings;
  }
  return { members, riders };
}

/** The tag for a train (`pick`'s train hit). */
export function trainTip(ov: Overlay, w: WorldView, line: string, profile: number, dep: number): string[] | null {
  const l = w.save.lines.find((x) => x.id === line);
  const net = ov.renderer.net;
  if (!l || !net) return null;
  const name = `${l.name} train`;
  const r = ov.renderer;
  const i = r.board ? r.tripIndex(profile, dep - r.tripsEpoch) : -1;
  // a train standing at a terminal for its next departure is drawn past its trip (T-206)
  const at = tripAt(net, profile, dep, clock.now(), i >= 0 && r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity);
  const st = w.lineStats[l.id];
  if (!at || !st || !display.peek().trainLoad) return [name];
  const dv = demandView.peek();
  if (!dv) return [name, "Calculating demand"];
  const n = l.stops.length, run = Math.floor(profile / 3) % 2;
  const stopS = st.stopS[run] ?? [];
  let seg = 0;
  while (seg + 1 < n - 1 && stopS[seg + 1] <= at.s + 1) seg++;
  const atEnd = at.s >= (stopS[n - 1] ?? Infinity) - 1;
  const own = r.board && i >= 0 ? tripLoad(r.board, i, run === 0 ? seg : n - 2 - seg) : null;
  const seats = SEATS_PER_CAR * st.cars, crush = CRUSH_PER_CAR * st.cars;
  const cap = `${n0(seats)} seats, ${n0(crush)} packed full`;
  // packed full: drawn with a red edge (T-207)
  if (own !== null) return [name, `${n0(atEnd ? 0 : own)} riders aboard${!atEnd && own >= crush - 0.5 ? ", packed full" : ""}`, cap];
  // no account for this train yet (its line's stops just changed): no gauge either, and no
  // forecast average in its place (T-230, Anita)
  return [name, "Calculating demand"];
}

/** Riders waiting at station `i`'s platforms now, a line per line and direction with anyone, and
 * how full the platform is drawn (T-219). */
function crowdLines(ov: Overlay, w: WorldView, id: string, i: number): string[] {
  const r = ov.renderer, b = r.board;
  if (!b || display.peek().platformCrowds === "off" || !r.crowdsShown) return [];
  // zoomed out, the dot's tint (T-228): everyone waiting at it and its fullest platform
  const dot = r.dotCrowd(i);
  if (dot) {
    if (dot.waiting < 1) return ["Nobody waiting on the platforms"];
    const where = r.dotStations(i) > 1 ? " at these stations" : "";
    const fullest = dot.share >= 1 ? "full" : `${Math.round(dot.share * 100)}% full`;
    const out = [`${n0(dot.waiting)} waiting${where}`, `Fullest platform ${fullest}`];
    if (dot.leftBehind) out.push("A full train left some behind");
    return out;
  }
  const node = Number(id), t = clock.now() - r.tripsEpoch;
  const names = new Map(w.save.stations.map((s) => [s.id, s.name]));
  const out: string[] = [];
  for (let q = 0; q < b.queueCount; q++) {
    const base = q * QUEUE_STRIDE;
    if (b.queues[base] !== node) continue;
    const { waiting, leftBehind } = crowdAt(b, q, t);
    if (waiting < 1) continue;
    const l = w.save.lines.find((x) => x.id === String(b.queues[base + 1]));
    if (!l) continue;
    const to = names.get(b.queues[base + 2] === 0 ? l.stops[l.stops.length - 1] : l.stops[0]) ?? "the end";
    out.push(leftBehind ? `${n0(waiting)} waiting for ${l.name} to ${to} after a full train left some behind` : `${n0(waiting)} waiting for ${l.name} to ${to}`);
  }
  if (!out.length) return ["Nobody waiting on the platforms"];
  // the fill drawn: each side's share of the room to stand there
  const share = r.crowdShare(i);
  const full = (v: number) => (v >= 1 ? "full" : `${Math.round(v * 100)}% full`);
  if (share?.shares.length === 2) out.push(`Platform ${full(share.shares[0])} on one side and ${full(share.shares[1])} on the other (room for ${forecast(share.room)} a side)`);
  else if (share) out.push(`Platform ${full(share.shares[0])} (room for ${forecast(share.room)})`);
  return out;
}

/** The tag for station `i` of the network buffers. */
export function stationTip(ov: Overlay, w: WorldView, i: number): string[] | null {
  const net = ov.renderer.net;
  if (!net) return null;
  const id = net.stationIds[i];
  const s = w.save.stations.find((x) => x.id === id);
  if (!s) return null;
  const out = [s.name];
  // zoomed out a complex is one dot, standing for its busiest station (T-217)
  const dot = ov.renderer.dotStations(i);
  // a complex the player linked is a dot per group of platforms, joined by thin lines (T-225)
  const all = ov.renderer.stationStyle.boxes ? 1 : complexOf(id).length;
  if (dot > 1 || all > dot) out.push(all > dot ? (dot > 1 ? `One dot for ${dot} of the complex's ${all} stations` : `One of the complex's ${all} stations`) : `One dot for the complex's ${dot} stations`);
  if (!s.built) out.push("Blueprint");
  else if (display.peek().stationRiders) {
    const c = circleOf(w, id);
    out.push(...(c ? circleLines(c.riders, c.members.length) : ["Calculating demand"]));
  }
  if (s.built) out.push(...crowdLines(ov, w, id, i));
  return out;
}

/** The tag for a station demand circle under the point (circles are behind the network), or null. */
export function circleTip(ov: Overlay, w: WorldView, px: number, py: number, zoom: number): string[] | null {
  const net = ov.renderer.net, dv = demandView.peek();
  if (!net || !dv || !display.peek().stationRiders) return null;
  const R = stationHaloRadius(zoom);
  const index = new Map(net.stationIds.map((id, i) => [id, i]));
  const names = new Map(w.save.stations.map((s) => [s.id, s.name]));
  const seen = new Set<string>();
  let best: { members: string[]; riders: number } | null = null, bestF = 1;
  for (const s of w.save.stations) {
    if (!s.built || seen.has(s.id)) continue;
    const c = circleOf(w, s.id);
    if (!c || c.riders <= 0) continue;
    for (const m of c.members) seen.add(m);
    // the circle's centre: its stations' mean position (one station's own position otherwise)
    let x = 0, y = 0, k = 0;
    for (const m of c.members) {
      const j = index.get(m);
      if (j === undefined) continue;
      x += net.stations[j * 3];
      y += net.stations[j * 3 + 1];
      k++;
    }
    if (!k) continue;
    const [sx, sy] = ov.toScreen(x / k, y / k);
    const r = R * stationScale(c.riders);
    const f = r > 0 ? Math.hypot(sx - px, sy - py) / r : Infinity;
    // the circle whose centre the pointer is relatively nearest, of those it is inside
    if (f <= bestF) [best, bestF] = [c, f];
  }
  if (!best) return null;
  const busiest = best.members.map((m) => ({ m, d: stationRidersFor(dv, m) })).sort((a, b) => (b.d ? b.d.boardings + b.d.alightings : 0) - (a.d ? a.d.boardings + a.d.alightings : 0))[0];
  return [names.get(busiest.m) ?? "Station", ...circleLines(best.riders, best.members.length)];
}
