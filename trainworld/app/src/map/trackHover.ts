// Hovering track (T-157): a tag by the pointer says what the current track display draws there:
// the line's bullet and name and its riders per day on that stop-to-stop stretch (any view,
// T-1907), its average load there (Traffic, T-2007), and always the elevation and speed limit there;
// bare track also lists the lines on it (T-473, T-526). It shows at once. The tag follows the map's pick order (T-133): with the select tool,
// demand bubbles, capacity tags and editing handles above the network hide it; with a build tool
// the demand views are beneath the network, and the tag sits above the pointer, clear of the
// tool's own tag. Trains, stations, platform crowds and station demand circles have tags of their
// own (T-194, networkTips.ts).

import { effect } from "@preact/signals";
import type { Map as MlMap } from "maplibre-gl";
import { clock } from "../game/clock";
import { demandView, lineLoadsFor, type DemandView } from "../game/demand";
import { purposeName, type PurposeFilter } from "../game/purposes";
import { DEMAND_INDEX, type LineInput, type WorldView } from "../game/types";
import { ask, display, isBuildTool, purposeFilter, selection, shownDay, shownDayPhrase, tool } from "../game/ui";
import { world } from "../game/world";
import { hoverAt } from "./demandHover";
import { restHolds } from "./hoverRest";
import { markAt, markTip } from "./gradeMarks";
import { hoverHeld } from "../ui/blurHold";
import { live } from "./live";
import { levelAtStart, type NetworkBuffers } from "./network";
import { circleTip, stationTip, trainTip, upgradeTip, type TipLine } from "./networkTips";
import { busLineTip, busStopTip } from "./busLayer";
import type { Overlay } from "./overlay";
import { pick, SLOP_PX, topSeg } from "./pick";
import { hiddenRealmTip } from "./realmOverlay";
import { setPointerNet, type Hot } from "./hovered";
import { TAG_OFFSET_PX, tagLines } from "./labelLayout";
import { complexOf, stopKey } from "../game/complexView";
import { clockTime, distUnit, forecast, money, moneyNegative, perDistance } from "../game/format";
import { stretchMoney, type StretchFigures } from "../game/stretchMoney";
import type { CrowdedStretch } from "../game/boarding";
import { loadText } from "../game/today";
import { t, tNumber, tRich, type MessageKey } from "../i18n";
import type { ComponentChild, ComponentChildren } from "preact";
import { chip, kids, elevationFact, factRow, lineBullet, lineBullets, speedFact, trainSpeedFact } from "./trackTagRows";
import { stretchLoad, tripsLanded } from "./stretchLoad";
import { crowdColour, profitColour } from "../game/palette";
import { floorAt } from "./buildingFloors";
import { buildingMembersById, hallWaiting } from "../game/buildingView";
import type { BuildingInput } from "../game/types";

/** Daily riders on run 0's stretch `k` of a line with these `stops`, both directions: every
 * period's load summed; null when demand has no answer for it that fits its stops (T-1419: after an
 * edit, the last result's figures carried onto the new stops until the new one lands).
 * `purpose`: the map's purpose filter (T-158), that purpose's riders only. */
export function dailyRiders(dv: DemandView | null, lineId: string, stopIds: readonly string[], k: number, purpose: PurposeFilter = "all"): number | null {
  const stops = stopIds.length;
  const loads = lineLoadsFor(dv, lineId, purpose, stopIds, stopKey);
  if (!loads?.length || stops < 2 || k < 0 || k > stops - 2) return null;
  let sum = 0;
  for (const p of loads) {
    if (p.length !== 2 * (stops - 1)) return null;
    // run 1 runs the stops backwards: its stretch j covers run 0's stretch n - 2 - j
    sum += p[k] + p[stops - 1 + (stops - 2 - k)];
  }
  return sum;
}

/** T-1907: how big line `num` is at its stroke segment `seg`, for a click on lines drawn over each
 * other: its riders per day on that stretch, else (no demand yet) its trains an hour times cars,
 * scaled so any riders outrank it. */
export function lineSizeAt(ov: Overlay, num: number, seg: number): number {
  const w = world.peek(), l = w?.save.lines.find((x) => x.num === num);
  if (!w || !l) return 0;
  const k = ov.renderer.net?.lines.stretch?.[seg] ?? -1;
  const day = dailyRiders(demandView.peek(), l.id, l.stops, k);
  if (day !== null) return day + 1;
  return ((l.tph[clock.demand.peek()] ?? 0) * (w.lineStats[l.id]?.cars ?? 1)) / 1e6;
}

/** The tag's lines for a line stroke segment `i`. */
function lineTip(n: NetworkBuffers, w: WorldView | null, dv: DemandView | null, traffic: boolean, i: number, full: CrowdedStretch | null): TipLine[] {
  const num = n.lines.colour[i];
  const l = w?.save.lines.find((x) => x.num === num);
  const name = l ? named(l) : t("trackHover.line", { num });
  const status = l ? w?.lineStats[l.id]?.status : undefined;
  if (status === "planned") return [name, t("trackHover.notRunningYet")];
  if (status === "broken") return [name, t("trackHover.notRunning")];
  if (!l) return [name];
  // T-1907: the riders in every track view; outside Traffic a tag without them says nothing of demand
  if (!dv) return traffic ? [name, t("trackHover.calculating")] : [name];
  const k = n.lines.stretch ? n.lines.stretch[i] : -1;
  const purpose = purposeFilter.peek();
  const day = dailyRiders(dv, l.id, l.stops, k, purpose);
  if (day === null) return traffic ? [name, t("trackHover.calculating")] : [name];
  const count = t("trackHover.riders", { n: day < 0.5 ? "0" : forecast(day), when: shownDayPhrase.peek() });
  // the purpose filter (T-158) says what it counted
  const out: TipLine[] = purpose === "all" ? [name, count] : [name, count, t("trackHover.tripsOnly", { purpose: purposeName(purpose) })];
  if (!traffic) return out;
  // T-2007: the stretch's average load, every purpose's riders against the day's trips at the rated load
  const st = w?.lineStats[l.id];
  const all = purpose === "all" ? day : dailyRiders(dv, l.id, l.stops, k, "all");
  const load = st && w && all !== null ? stretchLoad(num, w.version, st.cars, shownDay.peek(), all) : null;
  if (load !== null && load !== undefined) out.push(t("trackHover.avgLoad", { share: tNumber(Math.round(load * 100)) }));
  // T-819: the red outline drawn on track at capacity, at the level in force as the capacity tags
  const lev = DEMAND_INDEX[clock.demand.peek()];
  if (display.peek().fullMarks && w?.fullTrack[lev]?.has(n.lines.edge[i])) {
    out.push(Object.assign(document.createElement("div"), { textContent: t("trackHover.atCapacity", { demand: t("inspector.demand", { demand: clock.demand.peek() }) }), style: "color: var(--bad)" }));
  }
  // T-2236: the crowding mark's load (the boarding account's busiest hour), on its crowding colour
  // as the line panel's dots
  if (full) {
    const d = document.createElement("div");
    d.append(...kids(tRich("trackHover.peakLoad", { load: chip(loadText(full.load), crowdColour(full.load)) as unknown as ComponentChild, time: clockTime(full.hour * 60) })));
    out.push(d);
  }
  return out;
}

/** T-2007: a line's bullet, then its name, on one row. */
function named(l: LineInput): HTMLDivElement {
  const d = factRow(lineBullet(l), Object.assign(document.createElement("span"), { textContent: l.name }));
  d.style.gap = "5px";
  return d;
}

/** A running bus or ferry line's rows on the stretch between its stops under the pointer, as a rail
 * line's tag and the track's say them: in Profit the stretch's money first (`profitRows`), then its
 * riders (`lineTip`), then in Speed its vehicles' speed there (`trackFacts`), in the mode's own word. */
function busRows(ov: Overlay, w: WorldView, line: number, px: number, py: number, view: string): TipLine[] {
  const l = w.save.lines.find((x) => x.num === line);
  if (!l) return [];
  const ferry = !!w.lineStats[l.id]?.bus?.ferry, vehicles = ferry ? "boats" : "buses";
  const k = ov.renderer.buses.stretchAt(ov, px, py, line);
  const out: TipLine[] = [];
  const dv = demandView.peek();
  if (view === "profit") {
    const sm = stretchMoney.peek(), s = k >= 0 ? sm?.buses.get(line)?.stretches[k] : undefined;
    if (!sm) { if (!dv) out.push(t("trackHover.calculating")); }
    else if (s?.bare) out.push(t("trackHover.noVehicles", { vehicles }));
    else if (s) out.push(...stretchMoneyRows(s, vehicles));
  }
  const purpose = purposeFilter.peek();
  const day = dv ? dailyRiders(dv, l.id, l.stops, k, purpose) : null;
  if (day !== null) {
    out.push(t("trackHover.riders", { n: day < 0.5 ? "0" : forecast(day), when: shownDayPhrase.peek() }));
    if (purpose !== "all") out.push(t("trackHover.tripsOnly", { purpose: purposeName(purpose) }));
  } else if (view === "traffic") out.push(t("trackHover.calculating"));
  if (view === "speed") {
    const v = ov.renderer.buses.speedAt(ov, px, py, line);
    out.push(v !== null ? factRow(trainSpeedFact(v, ferry ? "boat" : "bus")) : t("trackHover.noVehicles", { vehicles }));
  }
  return out;
}

/** The tag's lines at a point (CSS px in the map), or null: what the current track display draws
 * on the line or track `pick` finds there; for a train, a station, a platform's crowd or a station
 * demand circle, what they stand for (T-194, networkTips.ts); for a crossing, junction, ramp or
 * portal mark, its name (T-629). Nothing for a junction without a mark, nor for the station
 * circles while a build tool has the demand views beneath the network. */
export function trackTip(ov: Overlay, px: number, py: number, zoom = 12, out?: { hot?: Partial<Hot> }): TipLine[] | null {
  const n = ov.renderer.net;
  if (!n || !ov.camMatrix) return null;
  const d = display.peek();
  const w = world.peek();
  // T-1124: riders changing trains on their walk
  const bub = ov.cssW > 0 && ov.cssH > 0 ? ov.renderer.bubbles.at(px / ov.cssW, py / ov.cssH) : null;
  if (bub) {
    const to = w?.save.lines.find((x) => x.num === bub.line)?.name ?? t("trackHover.line", { num: bub.line });
    return [t("trackHover.changing", { n: forecast(bub.riders), line: to })];
  }
  // a build tool is about the track: no train under the pointer, a search through every train on
  // each pointer move (T-1394)
  const hit = pick(ov, px, py, d.trains && !isBuildTool(tool.peek()));
  if (w && hit?.kind === "train") return trainTip(ov, w, hit.line, hit.profile, hit.dep);
  // T-2218: a station building's floor where no box or car park is (pick gives its complex)
  const floor = hit?.kind === "station" && floorAt(ov, px, py);
  if (floor && hit?.kind === "station" && ov.renderer.stationAt(px, py, (x, y) => ov.toScreen(x, y), 2) < 0 && ov.renderer.lotAt(px, py, (x, y) => ov.toScreen(x, y)) < 0) {
    if (out) out.hot = { stations: complexOf(hit.station) };
    return buildingTip(floor);
  }
  if (w && hit?.kind === "station") {
    const i = n.stationIds.indexOf(hit.station);
    if (i >= 0 && out) out.hot = { stations: complexOf(hit.station) };
    if (i < 0) return null;
    const bit = ov.renderer.upgradeAt(i, px, py, (x, y) => ov.toScreen(x, y));
    return bit ? upgradeTip(ov, w, i, bit) : stationTip(ov, w, i);
  }
  // the marks for crossings, junctions, ramps and portals sit on the track, under stations and trains (T-629)
  if (n.marks) {
    const i = markAt(ov, n.marks, px, py, zoom);
    if (i >= 0) return markTip(n.marks, i);
  }
  // bus lines: a stop's tag, else the line's (map/busLayer.ts); the stops are drawn over the line
  const buses = ov.renderer.buses;
  if (w && buses.any && (!hit || (hit.kind === "line" && buses.has(Number(hit.line))))) {
    const stop = buses.stopAt(ov, px, py);
    if (stop) {
      if (out) out.hot = { line: stop.line };
      return busStopTip(w, stop.line, stop.end, stop.id, ov);
    }
    if (hit) {
      const num = Number(hit.line);
      if (out) out.hot = { line: num };
      return busLineTip(w, num, busRows(ov, w, num, px, py, d.trackColour));
    }
  }
  // the ridership circles: over the bubbles, but the network beneath them keeps the hover (T-1448)
  if (!hit) {
    // an empty building's floor picks nothing
    const empty = floorAt(ov, px, py);
    if (empty) return buildingTip(empty);
    return w && !isBuildTool(tool.peek()) ? circleTip(px, py) : null;
  }
  const reach = Math.max(SLOP_PX, ov.renderer.laneReachPx);
  const slop = reach * reach + 1;
  if (hit.kind === "line") {
    const L = n.lines;
    // a selected line's highlight is drawn above the others (pick.ts)
    const sel = selection.peek();
    // as wide as drawn (traffic thickness, T-227), as pick() takes it
    const wide = ov.renderer.lineWidths, widePx = ov.renderer.slotPx;
    // side by side (T-1799) the band pick() hit, the selected line's only when that is it
    const bandPx = ov.renderer.pickBandPx, B = ov.renderer.pickBand;
    let [i] = sel?.kind === "line" && (!(bandPx > 0) || sel.line === hit.line) ? topSeg(ov, L.seg, L.count, px, py, slop, B, bandPx, (j) => String(L.colour[j]) === sel.line, wide, widePx) : [-1];
    if (i < 0) [i] = topSeg(ov, L.seg, L.count, px, py, slop, B, bandPx, (j) => String(L.colour[j]) === hit.line, wide, widePx);
    if (i < 0) return null;
    if (out) out.hot = { line: L.colour[i] };
    // T-473: the track under the line says its elevation and speed too; T-2387: the other lines on it
    // are left out, as they read as part of the hovered line
    return [...lineTip(n, w, demandView.peek(), d.trackColour === "traffic", i, ov.renderer.fullStretch(L.colour[i], L.stretch ? L.stretch[i] : -1)), ...trackFacts(ov, L.edge[i], px, py, slop)];
  }
  if (hit.kind !== "track") return null;
  const facts = trackFacts(ov, hit.edge, px, py, slop);
  if (!facts.length) return null;
  if (out) out.hot = { edge: hit.edge };
  // T-473: always the elevation, speed and lines on it, whatever the track display; no lines row
  // where none runs (T-526). Track under a bundle of lines loses to the lines, except at its edges.
  const on = linesOn(w, hit.edge);
  const money = d.trackColour === "profit" ? profitRows(hit.edge, edgeVehicles(w, hit.edge)) : [];
  if (on.length) {
    return [...money, ...edgeRiders(n, hit.edge, on), ...facts, lineBullets(on)];
  }
  return !w && lineOnEdge(n, hit.edge) ? [...facts, t("trackHover.lineRuns")] : [...money, ...facts];
}

/** T-2420: the Profit view's rows for the track stretch an edge is on: the share of its costs the fares cover on
 * its colour, its profit, per km, then fares, running trains and upkeep; bare track its upkeep. None for blueprint track. */
function profitRows(edge: number, vehicles: string): TipLine[] {
  const sm = stretchMoney.peek();
  if (!sm) return demandView.peek() ? [] : [t("trackHover.calculating")];
  const s = sm.stretchOf.get(edge);
  if (!s) return [];
  if (s.bare) return [t("trackHover.noVehicles", { vehicles }), moneyRow("trackHover.upkeepDay", -s.upkeep)];
  return stretchMoneyRows(s, vehicles);
}

/** A money row, negative money red. */
function moneyRow(key: MessageKey, v: number, more: Record<string, string> = {}): HTMLDivElement {
  const d = document.createElement("div");
  const m = Object.assign(document.createElement("span"), { textContent: money(v) });
  if (moneyNegative(v)) m.className = "money-neg";
  d.append(...kids(tRich(key, { amount: m as unknown as ComponentChild, ...more })));
  return d;
}


/** The Profit view's rows for a stretch something runs on, its running cost named for `vehicles`
 * (T-2452: "buses", "boats"). */
function stretchMoneyRows(s: StretchFigures, vehicles: string): TipLine[] {
  const row = moneyRow;
  if (s.noFigures) return [t("trackHover.calculating")];
  const profit = s.fares - s.running - s.upkeep, perKm = s.km > 0 ? profit / s.km : 0, costs = s.running + s.upkeep;
  const rows = [row("trackHover.profit", profit, { when: shownDayPhrase.peek() }), row("trackHover.perKm", perDistance(perKm), { unit: distUnit() }), row("trackHover.fares", s.fares), row("trackHover.running", -s.running, { vehicles }), row("trackHover.upkeep", -s.upkeep)];
  if (!(costs > 0)) return rows;
  const ratio = s.fares / costs, top = document.createElement("div");
  top.append(...kids(tRich("trackHover.faresCover", { share: chip(`${tNumber(Math.round(100 * ratio))}%`, profitColour(ratio)) as unknown as ComponentChild })));
  return [top, ...rows];
}

/** What runs on an edge, in the plural: trams where every line on it is a tram line (tram and train track never join). */
function edgeVehicles(w: WorldView | null, edge: number): "trams" | "trains" {
  const on = linesOn(w, edge);
  return on.length > 0 && on.every((l) => w!.lineStats[l.id]?.tram) ? "trams" : "trains";
}

/** The lines whose path uses an edge, in the save's order. */
function linesOn(w: WorldView | null, edge: number): LineInput[] {
  const e = w?.edges.find((x) => x.id === edge);
  return e ? w!.save.lines.filter((l) => e.lines.includes(l.id)) : [];
}

/** T-2218: a station building's floor: its name, and that it has no station inside when so. */
function buildingTip(b: BuildingInput): TipLine[] {
  if (!(buildingMembersById.peek().get(b.id)?.length)) return [b.name, t("trackHover.building"), t("trackHover.noStation")];
  // T-2219: the hall's crowd, while platform crowds are drawn
  const hall = hallWaiting.peek().get(b.id) ?? 0;
  return hall >= 1 ? [b.name, t("trackHover.building"), t("trackHover.hall", { n: tNumber(Math.round(hall)) })] : [b.name, t("trackHover.building")];
}

/** The elevation and speed limit at a point of an edge's track stroke, on one row (T-473, T-526,
 * T-2007); none off the stroke. */
function trackFacts(ov: Overlay, edge: number, px: number, py: number, slop: number): TipLine[] {
  const T = ov.renderer.net!.track;
  const [i] = topSeg(ov, T.seg, T.count, px, py, slop, undefined, 0, (j) => T.edge[j] === edge);
  if (i < 0) return [];
  const lv = levelAtStart(T.level[i]);
  const rows: TipLine[] = [T.speed?.[i] !== undefined ? factRow(elevationFact(lv), speedFact(T.speed[i])) : factRow(elevationFact(lv))];
  // T-1271: the Speed view draws the trains' speed where they run, the limit elsewhere; say which
  if (display.peek().trackColour === "speed") {
    const v = T.trainSpeed?.[i] ?? -1;
    const tram = edgeVehicles(world.peek(), edge) === "trams";
    rows.unshift(v >= 0 ? factRow(trainSpeedFact(v, tram ? "tram" : "train")) : t("trackHover.noVehicles", { vehicles: tram ? "trams" : "trains" }));
  }
  if (world.peek()?.edges.find((x) => x.id === edge)?.oneWay) rows.push(t("trackHover.oneWay"));
  return rows;
}

/** T-1907: the riders per day of every line on an edge, each on its stretch there, as a tag row;
 * none until demand has an answer for each. Height and Speed draw no line strokes, so their tag is
 * the track's, and this is where they say the riders. */
function edgeRiders(n: NetworkBuffers, edge: number, on: LineInput[]): string[] {
  const dv = demandView.peek();
  if (!dv || !n.lines.stretch) return [];
  const k = new Map<number, number>();
  for (let i = 0; i < n.lines.count && k.size < on.length; i++) if (n.lines.edge[i] === edge && !k.has(n.lines.colour[i])) k.set(n.lines.colour[i], n.lines.stretch[i]);
  const purpose = purposeFilter.peek();
  let sum = 0;
  for (const l of on) {
    const day = dailyRiders(dv, l.id, l.stops, k.get(l.num) ?? -1, purpose);
    if (day === null) return [];
    sum += day;
  }
  const count = t("trackHover.riders", { n: sum < 0.5 ? "0" : forecast(sum), when: shownDayPhrase.peek() });
  return purpose === "all" ? [count] : [count, t("trackHover.tripsOnly", { purpose: purposeName(purpose) })];
}

function lineOnEdge(n: NetworkBuffers, edge: number): boolean {
  for (let i = 0; i < n.lines.count; i++) if (n.lines.edge[i] === edge) return true;
  return false;
}

/** The tag in the map: shown while the pointer is on track, moved and refilled on moves. */
export function startTrackHover(map: MlMap, ov: Overlay) {
  const box = map.getContainer();
  const tag = document.createElement("div");
  tag.className = "tool-tag demand-tag track-tag";
  tag.style.display = "none";
  box.appendChild(tag);
  let shown = false;
  let last = "";
  /** the pointer, CSS px in the map; null when it is not over the map's canvas */
  let at: [number, number] | null = null;

  const hide = () => {
    shown = false;
    tag.style.display = "none";
    setPointerNet(null);
  };
  /** The tag's lines at the pointer, or null where something above the track is, or a tool
   * without it is on (the stop editor and the trip tool). Lights what it names with the select
   * tool only: a build tool's ghost is what matters there (T-730). */
  const tipAt = (px: number, py: number): TipLine[] | null => {
    const t = tool.peek();
    if (ask.peek()) return null;
    if (t === "select") {
      if (live.nodeDrag?.dragging) return null;
      if (hoverAt(map, px, py)) return null; // a demand bubble above the network
      if (live.nodeDrag?.hit(px, py)?.movable) return null; // a blueprint node handle
    } else if (!isBuildTool(t)) return null;
    if (t === "bus" && live.tools?.bus.onHandle) return null; // its own tag says what the point does (T-1514)
    if (t === "ferry" || t === "building") return null; // its own tag says what a click there does (T-2194, T-2218)
    const out: { hot?: Partial<Hot> } = {};
    const lines = trackTip(ov, px, py, map.getZoom(), out);
    setPointerNet(lines && t === "select" ? out.hot ?? null : null);
    return lines ?? hiddenRealmTip(map, px, py); // T-1151
  };
  const show = (lines: TipLine[], px: number, py: number) => {
    const key = lines.map((s) => (typeof s === "string" ? s : s.outerHTML)).join("\n");
    if (key !== last) {
      tagLines(tag, lines);
      last = key;
    }
    tag.style.display = "";
    // a build tool's own tag is below right of the pointer, so this one goes above
    tag.style.transform = isBuildTool(tool.peek())
      ? `translate(${Math.round(px + TAG_OFFSET_PX)}px, ${Math.round(py - 10)}px) translateY(-100%)`
      : `translate(${Math.round(px + TAG_OFFSET_PX)}px, ${Math.round(py + TAG_OFFSET_PX)}px)`;
    shown = true;
  };
  const update = () => {
    if (!at) return hide();
    const t = tipAt(at[0], at[1]);
    if (t) show(t, at[0], at[1]);
    else hide();
  };

  // On the container, not MapLibre's events: capacity tags and handles are DOM targets above the
  // canvas that MapLibre does not hear, and they must hide the tag too.
  // the container's place on the page, read again only when it may have moved: reading it on each
  // move lays the page out again after the build tool's writes (T-1394)
  let rect: DOMRect | null = null;
  const stale = () => (rect = null);
  new ResizeObserver(stale).observe(box);
  addEventListener("resize", stale);
  addEventListener("scroll", stale, { capture: true, passive: true });
  box.addEventListener("mousemove", (e) => {
    const r = (rect ??= box.getBoundingClientRect());
    const px = e.clientX - r.left, py = e.clientY - r.top;
    if (restHolds(px, py)) return; // on the way to the resting hover's list (T-732)
    // anything but the bare map canvas (a capacity tag, a handle, a control) is above the track
    if (e.buttons || e.target !== map.getCanvas()) {
      at = null;
      return hide();
    }
    at = [px, py];
    update();
  });
  box.addEventListener("mousedown", () => hide());
  box.addEventListener("mouseleave", () => {
    if (hoverHeld()) return;
    at = null;
    hide();
  });
  box.addEventListener("wheel", () => hide(), { passive: true });
  map.on("movestart", hide);
  // What is drawn changed under a resting pointer: say what is there now. After the current task,
  // as the clock client sets the world before handing the renderer its network.
  effect(() => {
    world.value;
    demandView.value;
    display.value;
    purposeFilter.value;
    selection.value;
    tool.value;
    ask.value;
    clock.minute.value; // trains move and crowds change under a resting pointer (T-194)
    tripsLanded.value;
    if (shown) queueMicrotask(() => shown && update());
  });
}
