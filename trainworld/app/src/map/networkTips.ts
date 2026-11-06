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

import { t, tNumber } from "../i18n";
import { capacity, crowdAt, QUEUE_STRIDE, tripLoad } from "../game/boarding";
import { clock } from "../game/clock";
import { complexes } from "../game/complexView";
import { demandView } from "../game/demand";
import { exact, forecast } from "../game/format";
import type { WorldView } from "../game/types";
import { hasUpgrade, UPGRADE, UPGRADE_TEXT } from "../game/upgrades";
import { display, purposeFilter, shownDayPhrase } from "../game/ui";
import { tripAt } from "./network";
import type { Overlay } from "./overlay";
import { circleColour, circleColours, circleList, stationCircles, type Circle } from "./stationCircles";
import { factRow, lineBullet, lineBullets } from "./trackTagRows";

/** A line of a tag: plain text, or a row built as DOM. */
export type TipLine = string | HTMLElement;

/** A ridership circle's riders (T-482, T-722, T-1448), as the station ridership choice counts: under
 * Total its entries and transfers apart, each after a dot in its end colour, and the total after a
 * dot in the circle's own, between them (T-1531). The circles count every
 * purpose, whatever the purpose filter says. */
function circleLines(c: Pick<Circle, "riders" | "transfers">): TipLine[] {
  const m = display.peek().stationUse, day = shownDayPhrase.peek(), col = circleColours();
  const out: TipLine[] = m === "total"
    ? [part(circleColour(c.transfers / c.riders, col), t("networkTips.riders", { n: forecast(c.riders), when: day })), part(col.entries, t("networkTips.entries", { n: forecast(c.riders - c.transfers) })), part(col.transfers, t("networkTips.transfers", { n: forecast(c.transfers) }))]
    : [part(m === "origins" ? col.entries : col.transfers, t("networkTips.use", { mode: m, n: forecast(c.riders), when: day }))];
  if (purposeFilter.peek() !== "all") out.push(t("networkTips.allPurposes"));
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
  const name = t("networkTips.train", { name: l.name });
  const r = ov.renderer;
  const i = r.tripIndex(profile, dep - r.tripsEpoch);
  // a train standing at a terminal for its next departure is drawn past its trip (T-206), and one
  // the guard holds behind its timetable (T-1965)
  const at = tripAt(net, profile, dep + r.tripLag(i), clock.now(), i >= 0 && r.board && r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity);
  const st = w.lineStats[l.id];
  if (!at || !st || !display.peek().trainLoad) return [name];
  const dv = demandView.peek();
  if (!dv) return [name, t("networkTips.calculating")];
  const n = l.stops.length, run = Math.floor(profile / 3) % 2;
  const stopS = st.stopS[run] ?? [];
  let seg = 0;
  while (seg + 1 < n - 1 && stopS[seg + 1] <= at.s + 1) seg++;
  const atEnd = at.s >= (stopS[n - 1] ?? Infinity) - 1;
  const own = r.board && i >= 0 ? tripLoad(r.board, i, run === 0 ? seg : n - 2 - seg) : null;
  const { crush } = capacity(st.cars);
  // full: drawn with a red edge (T-207)
  // T-324: a new or changed line's first figures, from an estimate the solve is still refining
  const est = !!dv.lines.get(l.id)?.estimate;
  if (own !== null) return [name, t("networkTips.onBoard", { n: exact(atEnd ? 0 : own), cap: exact(crush), full: !atEnd && own >= crush - 0.5, est })];
  // no account for this train yet (its line's stops just changed): no gauge either, and no
  // forecast average in its place (T-230, Anita)
  return [name, t("networkTips.calculating")];
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
    if (dot.waiting < 1) return [t("networkTips.nobody")];
    const fullest = dot.share >= 1 ? t("networkTips.fullestFull") : t("networkTips.fullestPct", { share: tNumber(Math.round(dot.share * 100)) });
    return [waitRow(null, "", t("networkTips.waiting", { n: exact(dot.waiting) }), dot.share >= 1), fullest];
  }
  const node = Number(id), now = clock.now() - r.tripsEpoch;
  const names = new Map(w.save.stations.map((s) => [s.id, s.name]));
  const out: TipLine[] = [];
  for (let q = 0; q < b.queueCount; q++) {
    const base = q * QUEUE_STRIDE;
    if (b.queues[base] !== node) continue;
    const { waiting } = crowdAt(b, q, now);
    if (waiting < 1) continue;
    const l = w.save.lines.find((x) => x.id === String(b.queues[base + 1]));
    if (!l) continue;
    const to = names.get(b.queues[base + 2] === 0 ? l.stops[l.stops.length - 1] : l.stops[0]) ?? "";
    out.push(waitRow(lineBullet(l, true), to ? t("networkTips.toStation", { name: to }) : t("networkTips.toEnd"), t("networkTips.waiting", { n: exact(waiting) }),(r.queueShare(q) ?? 0) >= 1));
  }
  return out.length ? out : [t("networkTips.nobody")];
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
  if (!s.built) out.push(t("networkTips.blueprint"));
  else if (display.peek().stationUse !== "none") {
    const c = circleOf(ov, w, id);
    out.push(...(c ? circleLines(c) : [t("networkTips.calculating")]));
  }
  if (s.built) out.push(...crowdLines(ov, w, id, i));
  return out;
}

/** T-2163: the network diagram's tag for a complex, `ids` its stations, as `stationTip` says a station:
 * its name, the lines stopping at any of them, and its ridership circle's riders (the platforms' crowds
 * are each station's, so none). */
export function complexTip(ov: Overlay, w: WorldView, ids: readonly string[], name: string): TipLine[] {
  const at = new Set(ids), out: TipLine[] = [name];
  const serving = [...new Map(w.save.lines.filter((l) => l.stops.some((s) => at.has(s))).map((l) => [l.letter + l.colour, l])).values()];
  if (serving.length) out.push(lineBullets(serving));
  if (!w.save.stations.some((s) => at.has(s.id) && s.built)) out.push(t("networkTips.blueprint"));
  else if (display.peek().stationUse !== "none") {
    const c = circleOf(ov, w, ids[0]);
    out.push(...(c ? circleLines(c) : [t("networkTips.calculating")]));
  }
  return out;
}

/** The tag for an upgrade drawn on station `i`'s box or beside it (T-1416, T-1528): its name and the
 * station's, how many of the complex's stations share it; a waiting room's riders too, as it is
 * drawn filled with them. */
export function upgradeTip(ov: Overlay, w: WorldView, i: number, bit: number): TipLine[] | null {
  const id = ov.renderer.net?.stationIds[i];
  const s = w.save.stations.find((x) => x.id === id);
  const u = UPGRADE_TEXT.find((x) => x.bit === bit);
  if (!s || !u || id === undefined) return null;
  const out: TipLine[] = [u.name, s.name];
  if (bit !== UPGRADE.terminal) {
    const byId = new Map(w.save.stations.map((x) => [x.id, x]));
    const k = complexes.peek().of.get(id), members = k === undefined ? [id] : complexes.peek().members[k];
    const sharing = members.filter((m) => hasUpgrade(byId.get(m)?.upgrades, bit)).length;
    if (sharing > 1) out.push(t("networkTips.shared", { count: sharing }));
  }
  if (bit === UPGRADE.waitingRoom && s.built) out.push(...crowdLines(ov, w, id, i));
  return out;
}

/** The tag for a ridership circle under the point, or null. */
export function circleTip(px: number, py: number): TipLine[] | null {
  if (display.peek().stationUse === "none") return null;
  const c = stationCircles.now?.at(px, py);
  return c ? [c.name, ...circleLines(c)] : null;
}
