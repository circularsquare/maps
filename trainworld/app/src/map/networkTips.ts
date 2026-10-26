// The network tag's text for trains, stations, platform crowds and the station use circles
// (T-194). trackHover.ts shows it, at once like the track's: each says in words the number the
// drawing stands for.
//
// - A train: its line, and the riders aboard against seats and packed full (the gauge its core is
//   filled with), from the boarding account as the train inspector counts them.
// - A station: its name; with ridership circles on, the riders a day its complex's circle stands
//   for; where platform crowds are drawn, the riders
//   waiting per line and direction, red where that platform side is full (T-219). Zoomed out, a complex's one dot is its
//   busiest station's, and says so (T-217).
// - A ridership circle away from any station: the same count, for the circle's complex.

import { CRUSH_PER_CAR, crowdAt, QUEUE_STRIDE, SEATS_PER_CAR, tripLoad } from "../game/boarding";
import { clock } from "../game/clock";
import { complexes } from "../game/complexView";
import { demandView } from "../game/demand";
import { exact, forecast } from "../game/format";
import { PERIODS, periodAt } from "../game/periods";
import type { WorldView } from "../game/types";
import { UPGRADE, UPGRADE_TEXT } from "../game/upgrades";
import { display, purposeFilter, shownDayPhrase } from "../game/ui";
import { tripAt } from "./network";
import type { Overlay } from "./overlay";
import { circleColours, circleList, stationCircles, type Circle } from "./stationCircles";
import { factRow, lineBullet, lineBullets } from "./trackTagRows";

/** A line of a tag: plain text, or a row built as DOM. */
export type TipLine = string | HTMLElement;

/** A ridership circle's riders (T-482, T-722, T-1448), as the station ridership choice counts: under
 * Total its entries and transfers apart, each by its colour on the map. The circles count every
 * purpose, whatever the purpose filter says. */
function circleLines(c: Pick<Circle, "riders" | "transfers">): TipLine[] {
  const m = display.peek().stationUse, day = shownDayPhrase.peek(), col = circleColours();
  const out: TipLine[] = m === "total"
    ? [`${forecast(c.riders)} riders ${day}`, part(col.entries, `${forecast(c.riders - c.transfers)} entries`), part(col.transfers, `${forecast(c.transfers)} transfers`)]
    : [part(m === "origins" ? col.entries : col.transfers, `${forecast(c.riders)} ${m === "origins" ? "entries" : "transfers"} ${day}`)];
  if (purposeFilter.peek() !== "all") out.push("All purposes");
  return out;
}

/** A tag row: a dot in a circle's colour, then its count. */
function part(colour: string, text: string): HTMLDivElement {
  const dot = document.createElement("span");
  dot.className = "tag-dot";
  dot.style.background = colour;
  return factRow(dot, Object.assign(document.createElement("span"), { textContent: text }));
}

/** The ridership circle of station `id`'s complex; null while its figure is not known. */
function circleOf(ov: Overlay, w: WorldView, id: string): Circle | null {
  const net = ov.renderer.net, dv = demandView.peek(), mode = display.peek().stationUse;
  if (!net || !dv || mode === "none") return null;
  const k = complexes.peek().of.get(id);
  const members = new Set(k === undefined ? [id] : complexes.peek().members[k]);
  return circleList(net, w, dv, mode).find((c) => members.has(c.station)) ?? null;
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
  const cap = `${exact(seats)} seats, ${exact(crush)} full`;
  // full: drawn with a red edge (T-207)
  // T-324: a new or changed line's first figures, from an estimate the solve is still refining
  const est = dv.lines.get(l.id)?.estimate ? " (estimate)" : "";
  if (own !== null) return [name, `${exact(atEnd ? 0 : own)} riders aboard${!atEnd && own >= crush - 0.5 ? ", full" : ""}${est}`, cap];
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
    return [waitRow(null, "", `${exact(dot.waiting)} waiting`, dot.share >= 1), `Fullest platform ${fullest}`];
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
    out.push(waitRow(lineBullet(l, true), `to ${to}`, `${exact(waiting)} waiting`,(r.queueShare(q) ?? 0) >= 1));
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
    const c = circleOf(ov, w, id);
    out.push(...(c ? circleLines(c) : ["Calculating demand"]));
  }
  if (s.built) out.push(...crowdLines(ov, w, id, i));
  return out;
}

/** The tag for an upgrade drawn on station `i`'s box (T-1416): its name and the station's; a
 * waiting room's riders too, as it is drawn filled with them. */
export function upgradeTip(ov: Overlay, w: WorldView, i: number, bit: number): TipLine[] | null {
  const id = ov.renderer.net?.stationIds[i];
  const s = w.save.stations.find((x) => x.id === id);
  const u = UPGRADE_TEXT.find((x) => x.bit === bit);
  if (!s || !u || id === undefined) return null;
  const out: TipLine[] = [u.name, s.name];
  if (bit === UPGRADE.waitingRoom && s.built) out.push(...crowdLines(ov, w, id, i));
  return out;
}

/** The tag for a ridership circle under the point, or null. */
export function circleTip(px: number, py: number): TipLine[] | null {
  if (display.peek().stationUse === "none") return null;
  const c = stationCircles.now?.at(px, py);
  return c ? [c.name, ...circleLines(c)] : null;
}
