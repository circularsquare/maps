// The network tag's text for trains, stations, platform crowds and the station use circles
// (T-194, Anita: "we probably shouldn't have much that is drawn but does not have a tooltip
// explaining quantitatively what is drawn"). trackHover.ts shows it, after the same rest as the
// track's: each says in words the number the drawing stands for.
//
// - A train: its line, and the riders aboard against seats and packed full (the gauge its core is
//   filled with), from the boarding account as the train inspector counts them.
// - A station: its name; with use circles on, the riders a day its circle stands for (its
//   complex's in complex view); where platform crowds are drawn, the riders
//   waiting per line and direction, red where that platform side is full (T-219: the crowd is
//   inside the box now, so the station's own hover covers it). Zoomed out, a complex's one dot is its
//   busiest station's, and says so (T-217).
// - A use circle away from any station: the same count, for the circle's station or complex.

import { crowdAt, QUEUE_STRIDE, tripLoad } from "../game/boarding";
import { clock } from "../game/clock";
import { complexes } from "../game/complexView";
import { demandView } from "../game/demand";
import { PERIODS, periodAt } from "../game/periods";
import { useFigure } from "../game/stationUse";
import type { WorldView } from "../game/types";
import { display, purposeFilter, shownDayPhrase } from "../game/ui";
import { stationScale } from "./demandLayers";
import { tripAt } from "./network";
import type { Overlay } from "./overlay";
import { stationHaloRadius } from "./stationGeometry";
import { factRow, lineBullet, lineBullets } from "./trackTagRows";

/** A line of a tag: plain text, or a row built as DOM. */
export type TipLine = string | HTMLElement;

/** Seats and riders at crush load per 20 m car (sim/src/demand/api.rs; ui/inspector/others.tsx). */
const SEATS_PER_CAR = 44, CRUSH_PER_CAR = 160;

const n0 = (v: number) => Math.round(v).toLocaleString("en-US");
/** A forecast count, rounded as the track tag rounds one (trackHover.ts `riders`). */
const forecast = (n: number) => (n >= 1000 ? Math.round(n / 100) * 100 : n >= 100 ? Math.round(n / 10) * 10 : Math.round(n)).toLocaleString("en-US");

/** "12,400 entries per day" (T-482, T-722), as the station use selector counts, whatever the number
 * of stations in a complex. The circles count every purpose, whatever the purpose filter says. */
function circleLines(riders: number): string[] {
  const m = display.peek().stationUse;
  const what = m === "origins" ? "entries" : m === "transfers" ? "transfers" : "entries and transfers";
  const out = [`${forecast(riders)} ${what} ${shownDayPhrase.peek()}`];
  if (purposeFilter.peek() !== "all") out.push("All purposes");
  return out;
}

/** The use circle's members and riders for station `id`: its complex's in complex view; null
 * while the figure is not known. */
function circleOf(w: WorldView, id: string): { members: string[]; riders: number } | null {
  const dv = demandView.peek(), mode = display.peek().stationUse;
  if (!dv) return null;
  const built = new Set(w.save.stations.filter((s) => s.built).map((s) => s.id));
  let members = [id];
  if (display.peek().complexView) {
    const cx = complexes.peek();
    const k = cx.of.get(id);
    if (k !== undefined) members = cx.members[k].filter((m) => built.has(m));
  }
  let riders = 0, known = false;
  for (const m of members) {
    const v = useFigure(dv, mode, m);
    if (v !== undefined) (riders += v), (known = true);
  }
  return known ? { members, riders } : null;
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
  const cap = `${n0(seats)} seats, ${n0(crush)} full`;
  // full: drawn with a red edge (T-207)
  // T-324: a new or changed line's first figures, from an estimate the solve is still refining
  const est = dv.lines.get(l.id)?.estimate ? " (estimate)" : "";
  if (own !== null) return [name, `${n0(atEnd ? 0 : own)} riders aboard${!atEnd && own >= crush - 0.5 ? ", full" : ""}${est}`, cap];
  // no account for this train yet (its line's stops just changed): no gauge either, and no
  // forecast average in its place (T-230, Anita)
  return [name, "Calculating demand"];
}

/** A tag row of an optional line bullet, a label and a count, the count in the alert colour when
 * `full` (T-559). */
function waitRow(bullet: HTMLElement | null, label: string, count: string, full: boolean): HTMLDivElement {
  const c = document.createElement("span");
  c.textContent = count;
  c.style.whiteSpace = "nowrap";
  if (full) c.style.color = "var(--bad)";
  const l = Object.assign(document.createElement("span"), { textContent: label });
  l.style.minWidth = "0";
  const d = factRow(...(bullet ? [bullet] : []), ...(label ? [l] : []), c);
  d.style.gap = "5px";
  return d;
}

/** Riders waiting at station `i`'s platforms now: a row per line and direction with anyone, its
 * count red while that platform side is full (T-559). */
function crowdLines(ov: Overlay, w: WorldView, id: string, i: number): TipLine[] {
  const r = ov.renderer, b = r.board;
  if (!b || display.peek().platformCrowds === "off" || !r.crowdsShown) return [];
  // zoomed out, the dot's pie (T-228, T-553): everyone waiting at it and its fullest platform
  const dot = r.dotCrowd(i);
  if (dot) {
    if (dot.waiting < 1) return ["Nobody waiting on the platforms"];
    const fullest = dot.share >= 1 ? "full" : `${Math.round(dot.share * 100)}% full`;
    return [waitRow(null, "", `${n0(dot.waiting)} waiting`, dot.share >= 1), `Fullest platform ${fullest}`];
  }
  const node = Number(id), t = clock.now() - r.tripsEpoch;
  const names = new Map(w.save.stations.map((s) => [s.id, s.name]));
  const out: TipLine[] = [];
  for (let q = 0; q < b.queueCount; q++) {
    const base = q * QUEUE_STRIDE;
    if (b.queues[base] !== node) continue;
    const { waiting } = crowdAt(b, q, t);
    if (waiting < 1) continue;
    const l = w.save.lines.find((x) => x.id === String(b.queues[base + 1]));
    if (!l) continue;
    const to = names.get(b.queues[base + 2] === 0 ? l.stops[l.stops.length - 1] : l.stops[0]) ?? "the end";
    out.push(waitRow(lineBullet(l, true), `to ${to}`, `${n0(waiting)} waiting`,(r.queueShare(q) ?? 0) >= 1));
  }
  return out.length ? out : ["Nobody waiting on the platforms"];
}

/** The tag for station `i` of the network buffers. */
export function stationTip(ov: Overlay, w: WorldView, i: number): TipLine[] | null {
  const net = ov.renderer.net;
  if (!net) return null;
  const id = net.stationIds[i];
  const s = w.save.stations.find((x) => x.id === id);
  if (!s) return null;
  const out: TipLine[] = [s.name];
  // T-305: the lines serving the station (zoomed out too, where every station is its own dot since
  // T-313), as bullets (T-559)
  const serving = [...new Map(w.save.lines.filter((l) => l.stops.includes(id)).map((l) => [l.letter + l.colour, l])).values()];
  if (serving.length) out.push(lineBullets(serving));
  if (!s.built) out.push("Blueprint");
  else if (display.peek().stationUse !== "none") {
    const c = circleOf(w, id);
    out.push(...(c ? circleLines(c.riders) : ["Calculating demand"]));
  }
  if (s.built) out.push(...crowdLines(ov, w, id, i));
  return out;
}

/** The tag for a station use circle under the point (circles are behind the network), or null. */
export function circleTip(ov: Overlay, w: WorldView, px: number, py: number, zoom: number): string[] | null {
  const net = ov.renderer.net, dv = demandView.peek(), mode = display.peek().stationUse;
  if (!net || !dv || mode === "none") return null;
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
  const busiest = best.members.map((m) => ({ m, v: useFigure(dv, mode, m) ?? 0 })).sort((a, b) => b.v - a.v)[0];
  return [names.get(busiest.m) ?? "Station", ...circleLines(best.riders)];
}
