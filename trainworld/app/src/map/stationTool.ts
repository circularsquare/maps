// The station tool (T-023): the name, the trips near, the ghost and its verdict, and the click that places a station (T-2374, moved out of tools.ts).

import { t as tr, tNumber } from "../i18n";
import { type MapMouseEvent } from "maplibre-gl";
import { EARTH_R } from "../game/frame";
import { describeIssues } from "../game/issues";
import { count, unitMetres, unitShort } from "../game/format";
import { activeCity, getWorldFrame, worldFrameEpoch } from "../game/cities";
import { cellModes, cityCells, demandView } from "../game/demand";
import { metresBetween } from "../game/walks";
import { stationKind, stationOnTram, tool } from "../game/ui";
import { newStationName, trackDirection } from "./stationNames";
import { edit, rightHand, say, stationNames, world } from "../game/world";
import { type ClockEdge, type ClockNode } from "../workers/protocol";
import { toLngLat, toLocal } from "./geo";
import { type TrackEdge } from "../game/types";
import { circleOnScreen } from "./curveCircle";
import { stationOutside } from "./stationSide";
import { splitsAt } from "./terminalBays";
import { PLATFORM_HALF_M, platformShape, reversed, stationPath, stationShape, waysFrom, type PlatformShape, type StationShape } from "./stationGeometry";
import { placeAt } from "./screenPlace";
import { firstAccepted, fractionAlong, layoutIssues, layoutOf, likelyFirst, nearestOn, pointAlong, roomNear, type Candidate } from "./stationSnap";
import { M_CROSSING, MARK_STRIDE } from "./markData";
import { playBuild } from "./buildSound";
import { SNAP_EDGE_PX, type Snap, type Spot, type Verdict, HOLD_AT, KNOWN_PX, LAYOUT_ISSUES, lengthOf } from "./toolParts";
import { portClick } from "./stationPort";
import { freeClick, freeStart, freeStarted } from "./stationFree";
import type { Tools } from "./tools";

export function nameKey(tk: Tools, px: number, py: number) {
  const c = tk.map.getCenter();
  return `${px},${py},${c.lng},${c.lat},${tk.map.getZoom()},${tk.map.getBearing()},${tk.map.getPitch()},${world.peek()?.version}`;
}

export function nameAhead(tk: Tools, px: number, py: number) {
  clearTimeout(tk.namedTimer);
  tk.namedTimer = setTimeout(() => {
    if (tool.peek() !== "station" || !tk.map.areTilesLoaded()) return;
    const key = nameKey(tk, px, py);
    if (tk.named?.key !== key) tk.named = { key, name: nameFor(tk, px, py) };
  }, 150) as unknown as number;
}

/** T-3381: the name the station placed at `at` shows (a station joining a complex takes its name), not the one made for it */
export function shownNameAt(_tk: Tools, lng: number, lat: number): string | undefined {
  const w = world.peek();
  if (!w) return undefined;
  const k = Math.cos((lat * Math.PI) / 180);
  let best: string | undefined, bd = 4e-8;
  for (const s of w.save.stations) {
    const d = ((s.lng - lng) * k) ** 2 + (s.lat - lat) ** 2;
    if (d < bd) (bd = d), (best = s.id);
  }
  return best === undefined ? undefined : stationNames(w).get(best);
}

/** T-3378: a BRT station's platform, m (sim `params::BRT_STATION_PLATFORM`) */
export const BRT_STATION_PLATFORM = 40;
/** T-3378: a BRT station's centre is this many ground metres off its road's centre line, and a click this near a
 * road a bus may use places one */
const BRT_SIDE_M = 10;
export const BRT_REACH_M = 50;

/** T-3999: a BRT station snaps to a bus route within this many screen pixels of ground (146 m at zoom 15 at 40°
 * latitude), at least `BUS_ROUTE_MIN_M` and at most `BUS_ROUTE_MAX_M`; a pointer within `ON_ROAD_M` of another road
 * nearer than the route takes that road */
const BUS_ROUTE_PX = 80, BUS_ROUTE_MIN_M = 50, BUS_ROUTE_MAX_M = 400, ON_ROAD_M = 8;
const M_PER_DEG = 111_320;

/** Ground metres a screen pixel spans at `lat` (512 px tiles). */
export function metresPerPx(tk: Tools, lat: number): number {
  return (78271.517 * Math.cos((lat * Math.PI) / 180)) / 2 ** tk.map.getZoom();
}

/** T-3999: the nearest point within `m` ground metres on a bus line's route (no ferry's), a point 10 m along it and
 * how far off it is; null where none is. */
function busRouteNear(lng: number, lat: number, m: number): { at: { lng: number; lat: number }; along: { lng: number; lat: number }; d: number } | null {
  const w = world.peek();
  if (!w) return null;
  const kx = M_PER_DEG * Math.cos((lat * Math.PI) / 180), ky = M_PER_DEG;
  let best = m, out: ReturnType<typeof busRouteNear> = null;
  for (const st of Object.values(w.lineStats)) {
    const b = st.bus;
    if (!b || b.ferry) continue;
    const r = b.routeLL;
    for (let i = 0; i + 3 < r.length; i += 2) {
      const ax = (r[i] - lng) * kx, ay = (r[i + 1] - lat) * ky, dx = (r[i + 2] - r[i]) * kx, dy = (r[i + 3] - r[i + 1]) * ky, l2 = dx * dx + dy * dy;
      if (!l2) continue;
      const k = Math.max(0, Math.min(1, -(ax * dx + ay * dy) / l2)), px = ax + k * dx, py = ay + k * dy, d = Math.hypot(px, py);
      if (d >= best) continue;
      const l = Math.sqrt(l2);
      best = d;
      out = { at: { lng: lng + px / kx, lat: lat + py / ky }, along: { lng: lng + (px + (dx / l) * 10) / kx, lat: lat + (py + (dy / l) * 10) / ky }, d };
    }
  }
  return out;
}

/** T-3378: where a click at `lng, lat` places a BRT station: T-3999: beside the nearest bus route within
 * `BUS_ROUTE_PX`, else beside the nearest road a bus may use within `BRT_REACH_M`; on the click's side of it, its
 * platform along the road (`heading` in the world frame); null where there is none, undefined while the roads load
 * (and no route is near). */
export function brtSpot(tk: Tools, lng: number, lat: number): { lx: number; ly: number; x: number; y: number; heading: number; ways: Float32Array[]; perM: number } | null | undefined {
  const route = busRouteNear(lng, lat, Math.max(BUS_ROUTE_MIN_M, Math.min(BUS_ROUTE_MAX_M, BUS_ROUTE_PX * metresPerPx(tk, lat))));
  const road = tk.street.busRoads.kerb({ lng, lat }, BRT_REACH_M);
  // a pointer on another road takes that road, so the street beside a route can still have its own station
  const onRoad = road ? metresBetween(lng, lat, road.at.lng, road.at.lat) : Infinity;
  const k = route && !(onRoad <= ON_ROAD_M && route.d > onRoad + ON_ROAD_M) ? route : road;
  if (!k) return k;
  const [ax, ay] = toLocal(k.at.lng, k.at.lat), [bx, by] = toLocal(k.along.lng, k.along.lat), [cx, cy] = toLocal(lng, lat);
  const l = Math.hypot(bx - ax, by - ay) || 1, ux = (bx - ax) / l, uy = (by - ay) / l;
  const side = ux * (cy - ay) - uy * (cx - ax) < 0 ? -1 : 1, perM = 1 / Math.cos((lat * Math.PI) / 180);
  const lx = ax - uy * side * BRT_SIDE_M * perM, ly = ay + ux * side * BRT_SIDE_M * perM;
  const p = tk.pack(lx, ly), q = tk.pack(lx + ux * perM, ly + uy * perM), reach = 100 * perM;
  // the platform's two ways out along the road, as a station's along its track
  const ways = [Float32Array.of(lx, ly, 0, lx + ux * reach, ly + uy * reach, 0), Float32Array.of(lx, ly, 0, lx - ux * reach, ly - uy * reach, 0)];
  return { lx, ly, x: p.x, y: p.y, heading: Math.atan2(q.y - p.y, q.x - p.x), ways, perM };
}

export async function stationClick(tk: Tools, e: MapMouseEvent) {
  const kind = stationKind.peek();
  if (kind === "port") return portClick(tk, e);
  if (kind === "bus") {
    // T-3378: a BRT station beside a road; T-3999: only with the bus kind picked, never snapped to track
    let b = brtSpot(tk, e.lngLat.lng, e.lngLat.lat);
    // a click before the roads there have loaded waits for them a few seconds
    for (let i = 0; b === undefined && i < 50; i++) {
      await new Promise((r) => setTimeout(r, 100));
      b = brtSpot(tk, e.lngLat.lng, e.lngLat.lat);
    }
    if (!b) {
      say(tr("tools.notBesideRoad"), true);
      return;
    }
    playBuild("station");
    const key = nameKey(tk, e.point.x, e.point.y), ahead = tk.named?.key === key && tk.map.areTilesLoaded() ? tk.named.name : null;
    tk.named = null;
    const name = await (ahead ?? nameFor(tk, e.point.x, e.point.y));
    const r = await edit({ op: "addBrtStation", x: b.x, y: b.y, heading: b.heading, name }, true, true);
    if (r.ok) {
      tk.lastPlaced = toLngLat(b.lx, b.ly);
      say(tr("tools.blueprintStation", { name: shownNameAt(tk, tk.lastPlaced[0], tk.lastPlaced[1]) ?? name }));
    } else if (r.issues.length) say(describeIssues(r.issues, undefined, undefined, true), true);
    return;
  }
  // T-3999: on open ground the first click is the platform's first end and the second its direction
  if (freeStarted()) return freeClick(tk, e, (px, py) => nameFor(tk, px, py));
  const first = tk.snap(e.point.x, e.point.y, e.lngLat);
  if (first.kind === "free") return freeStart(tk, first);
  // T-671, T-695: the spot the ghost shows, else the first the track model takes of what is under the pointer
  const v = tk.verdict();
  // T-1474: heard at the click, not when the edit is answered; not when the ghost shows a refusal
  if (!(v && !v.ok && !v.held)) playBuild("station");
  const pend = v?.ok && !v.held && tk.hover && tk.hover.kind !== "free" ? addPending(tk) : null;
  const tries = [...(v?.ok && v.spot && !v.held ? [v.spot] : []), ...tk.alts];
  const key = nameKey(tk, e.point.x, e.point.y);
  const ahead = tk.named?.key === key && tk.map.areTilesLoaded() ? tk.named.name : null;
  tk.named = null;
  const name = await (ahead ?? nameFor(tk, e.point.x, e.point.y));
  const stationAt = (p: Exclude<Snap, { kind: "free" }>) => ({ op: "addStation" as const, at: p.kind === "node" ? { kind: "node" as const, node: p.node } : { kind: "edge" as const, edge: p.edge, ...tk.pack(p.lx, p.ly) }, platform: tk.platFor(p), name });
  const one = tk.oneAt();
  const stationAt1 = (p: Exclude<Snap, { kind: "free" }>) => { const o = stationAt(p); return { ...o, at: { ...o.at, ...one } }; };
  const r = await firstAccepted(tries, async (p) => { const a = await edit(stationAt1(p), true, true); return a.ok ? null : a.issues; }, (p) => roomSnap(tk, p), () => false);
  dropPending(tk, pend);
  if (r.spot) {
    tk.lastPlaced = toLngLat(r.spot.lx, r.spot.ly); // T-312: the hover tag's "km from" (found again by position)
    say(tr("tools.blueprintStation", { name: shownNameAt(tk, tk.lastPlaced[0], tk.lastPlaced[1]) ?? name }));
  } else if (r.issues.length) say(describeIssues(r.issues, undefined, undefined, true), true);
}

/** Ask for the legs of the demand shown, unless held or asked already (T-412). T-2430: an answer
 * for a demand shown before is kept while nothing newer is held (a big zone's answer can take
 * longer than a solve's round, and dropping it left the tag without trips until the solve ended),
 * and one that fails is asked again 2 s later while the station tool is in hand. */
export function askLegs(tk: Tools) {
  const dv = demandView.peek();
  if (!dv) return;
  const city = activeCity.peek(), key = `${dv.version}:${dv.round}:${dv.day}:${city}:${worldFrameEpoch.peek()}`;
  if (tk.legs?.key === key || tk.legsAsked === key) return;
  tk.legsAsked = key;
  const seq = ++tk.legsSeq;
  void Promise.all([cityCells(), cellModes("all")]).then(([cells, m]) => {
    if (!cells || !m) {
      setTimeout(() => {
        if (tk.legsAsked !== key) return;
        tk.legsAsked = "";
        if (tool.peek() === "station") askLegs(tk);
      }, 2000);
      return;
    }
    // a newer answer is held already, or the zone or frame has changed (its cells are another set)
    if ((tk.legs && tk.legs.seq > seq) || activeCity.peek() !== city || !key.endsWith(`:${worldFrameEpoch.peek()}`)) return;
    const legs = new Float32Array(cells.n);
    for (const a of [m.homeRail, m.homeWalk, m.homeDrive, m.workRail, m.workWalk, m.workDrive, m.homeNone, m.workNone])
      if (a.length === cells.n) for (let i = 0; i < cells.n; i++) legs[i] += a[i];
    tk.legs = { key, xy: cells.xy, legs, seq };
    if (tool.peek() === "station") tk.place();
  });
}

/** "12k trips within 1km" of the spot (T-412, T-1058; 1 mile in US units), from the cells behind
 * the Homes and Destinations bubbles; null until they are in. A trip with both ends inside counts
 * twice, and a spot near a zone's edge counts the viewed zone's cells only. */
export function tripsNear(tk: Tools, lng: number, lat: number): { text: string; legs: number } | null {
  askLegs(tk);
  const L = tk.legs;
  if (!L) return null;
  // the same answer until the pointer, the unit or the legs change: tk runs on every place()
  const r = unitMetres(), kept = tk.nearKept;
  if (kept && kept.legs === L && kept.lng === lng && kept.lat === lat && kept.r === r) return kept.out;
  const out = sumTripsNear(tk, L, lng, lat, r);
  tk.nearKept = { legs: L, lng, lat, r, out };
  return out;
}

export function sumTripsNear(_tk: Tools, L: NonNullable<Tools["legs"]>, lng: number, lat: number, r: number): { text: string; legs: number } | null {
  // the radius in world metres each way, so any frame's scale is right here
  const f = getWorldFrame();
  const [x, y] = f.fromLonLat(lng, lat);
  const dLat = (r / EARTH_R) * (180 / Math.PI), dLng = dLat / Math.cos((lat * Math.PI) / 180);
  const [ex] = f.fromLonLat(lng + dLng, lat), [, ny] = f.fromLonLat(lng, lat + dLat);
  const rx = Math.abs(ex - x), ry = Math.abs(ny - y);
  let sum = 0, cells = 0;
  for (let i = 0; i < L.legs.length; i++) {
    const dx = (L.xy[i * 2] - x) / rx, dy = (L.xy[i * 2 + 1] - y) / ry;
    if (dx * dx + dy * dy <= 1) (sum += L.legs[i]), cells++;
  }
  if (!cells) return null;
  // two significant figures: the figure is rough by nature
  const p = 10 ** Math.max(0, Math.floor(Math.log10(Math.max(sum, 1))) - 1), r2 = Math.round(sum / p) * p;
  const n = sum < 50 ? null : r2 >= 10000 ? count(r2) : tNumber(r2);
  return { text: n === null ? tr("tools.nearUnder", { unit: unitShort() }) : tr("tools.near", { n, unit: unitShort() }), legs: sum };
}

/** The station placed last in tk session (T-312), found by its position so a rename shows. */
export function lastPlacedName(tk: Tools): { name: string; lng: number; lat: number } | null {
  const p = tk.lastPlaced;
  if (!p) return null;
  let best: { name: string; lng: number; lat: number } | null = null, bestD = 30;
  for (const s of world.peek()?.save.stations ?? []) {
    const d = metresBetween(p[0], p[1], s.lng, s.lat);
    if (d < bestD) (bestD = d), (best = s);
  }
  if (!best) tk.lastPlaced = null; // taken out again
  return best;
}

/** A name for a new station where the pointer is (map/stationNames.ts): the cross street, a
 * small neighbourhood, or the fallbacks, skipping names in use. */
export function nameFor(tk: Tools, px: number, py: number): Promise<string> {
  const s = tk.snap(px, py, tk.map.unproject([px, py]));
  const [lng, lat] = toLngLat(s.lx, s.ly);
  const net = tk.overlay.renderer.net;
  const track = net ? trackDirection(net, ...toLocal(lng, lat, net.anchor)) : null;
  return newStationName(tk.map, lng, lat, track, () => new Set(world.peek()?.save.stations.map((st) => st.name) ?? []));
}

export function el(tk: Tools, i: number, cls: string): HTMLDivElement {
  let m = tk.marks[i];
  if (!m) {
    m = tk.marks[i] = document.createElement("div");
    tk.box.appendChild(m);
  }
  m.className = cls;
  m.textContent = "";
  m.style.display = "";
  return m;
}

/** A 1 px SVG in the marks layer (`placeAt` moves it) with an empty child per tag in `kids`, at
 * the back of the layer or the front. */
export function svgMark(tk: Tools, kids: string[], back: boolean): SVGSVGElement {
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("width", "1");
  svg.setAttribute("height", "1");
  for (const k of kids) svg.appendChild(document.createElementNS(NS, k));
  if (back) tk.box.insertBefore(svg, tk.box.firstChild);
  else tk.box.appendChild(svg);
  return svg;
}

export function addPending(tk: Tools) {
  const b = tk.lastBox;
  if (!b) return null;
  const el = svgMark(tk, ["path"], false);
  el.setAttribute("class", "station-ghost ok placed");
  const p = { b, el };
  tk.pendingSt.push(p);
  placePending(tk);
  return p;
}

export function dropPending(tk: Tools, p: { b: unknown; el: SVGSVGElement } | null) {
  if (!p) return;
  p.el.remove();
  tk.pendingSt = tk.pendingSt.filter((x) => x !== p);
}

export function placePending(tk: Tools) {
  for (const { b, el } of tk.pendingSt) {
    const shape = ghostShape(tk, b.lx, b.ly, b.ways, b.perM, b.plat, b.side, b.tram, b.mid);
    placeAt(el, shape.x, shape.y, 0, 0, false);
    ghostPath(tk, el.firstElementChild as SVGPathElement, shape);
  }
}

/** The box of a station to place, from the same `stationShape` call as a built one (renderer.ts's
 * stationShapeAt), so the ghost, the pending mark and the station keep one width: always the full
 * box at its true length, whatever the zoom (T-311), the map's own style drawing dots below STATION_BOX_ZOOM */
export function ghostShape(tk: Tools, lx: number, ly: number, ways: ArrayLike<number>[], perM: number, plat: number, side: number, tram: boolean, mid = false) {
  const style = { ...tk.overlay.renderer.stationStyle, boxes: true };
  return stationShape(lx, ly, platformShape(ways, 0, plat / 2 * perM), PLATFORM_HALF_M * perM, (x, y) => tk.screen(x, y), style, false, false, side, { tram, mid });
}

/** A ghost's outline, as wide as a built station's rim and inside the box's edge as that is (a stroke is
 * centred on its path, so the path is drawn half a rim in) */
export function ghostPath(tk: Tools, el: SVGPathElement, shape: StationShape) {
  const ring = tk.overlay.renderer.stationStyle.ringPx;
  el.setAttribute("d", stationPath(shape, -ring / 2));
  el.style.strokeWidth = `${ring}px`;
}

export function stationGhost(tk: Tools, h: Exclude<Snap,{kind:"free"}>) {
  const n=tk.overlay.renderer.net,w=world.peek();
  if (!n || !w) return;
  const candidates=h.kind==="edge"?[h.edge]:w.edges.filter(e=>e.a===h.node||e.b===h.node).map(e=>e.id);
  const usage=new Map<number,Set<number>>();
  for (let i=0;i<n.lines.count;i++) {
    const edge=n.lines.edge[i];
    if (!candidates.includes(edge)) continue;
    if (!usage.has(edge)) usage.set(edge,new Set());
    usage.get(edge)!.add(n.lines.colour[i]);
  }
  candidates.sort((a,b)=>(usage.get(b)?.size??0)-(usage.get(a)?.size??0)||a-b);
  const edge=candidates[0];
  const [,lat]=toLngLat(h.lx,h.ly);
  const perM=1/Math.cos(lat*Math.PI/180);
  // the track either way from the point, as network.ts lays out a built station's platform
  const pts=n.edgePts.get(edge);
  const ways:ArrayLike<number>[]=h.kind==="edge"
    ? pts ? waysFrom(pts,h.lx,h.ly) : []
    : candidates.map((id)=>{ const e=w.edges.find((x)=>x.id===id), p=n.edgePts.get(id); return !e||!p?new Float32Array(0):e.a===h.node?p:reversed(p); });
  // on quad track the pair clicked only (T-1687); T-3332: a one-track station's half on double track
  const double=candidates.some((id)=>w.edges.find((x)=>x.id===id)?.tracks===2);
  // network.ts's `single`: every track under the platform is single and not a crossover
  const ends=candidates.map((id)=>w.edges.find((x)=>x.id===id));
  // a snap made before an undo can name track the undo took away: no ghost until the next snap
  if (!ends.length||!ends.every((e):e is TrackEdge=>!!e)) return;
  const single=!double&&ends.every((e)=>e.tracks===1&&!e.crossover);
  const outside=single?(sh:PlatformShape)=>outsideSide(tk, h,ends,ways,sh):null;
  boxGhost(tk, h.lx,h.ly,ways,perM,tk.verdict()?.ok===false,null,undefined,double,outside);
}

/** The side of its platform a station on one-way single track (or single track joining double) draws its one bar, from the same `stationOutside` a
 * built one uses. Part way along an edge the new station's node is -1, with the edge's two halves either side. */
export function outsideSide(tk: Tools, h: Exclude<Snap,{kind:"free"}>, ends: TrackEdge[], ways: ArrayLike<number>[], sh: PlatformShape): number {
  const clock=(e: TrackEdge, over: Partial<ClockEdge> = {}) => ({ ...e, off: !!e.off, hsr: !!e.hsr, kind: e.kind ?? 0, oneWay: e.oneWay ?? 0, join: e.join ?? 0, crossover: !!e.crossover, ...over }) as unknown as ClockEdge;
  let node: number, at: ClockEdge[], way: (n: number, id: number) => ArrayLike<number>, platformEdge: ClockEdge;
  if (h.kind==="node") {
    node=h.node;
    at=ends.map((e)=>clock(e));
    way=(_n,id)=>ways[ends.findIndex((e)=>e.id===id)] ?? new Float32Array(0);
    platformEdge=at[0];
  } else {
    const e=ends[0], half=e.lengthM/2;
    const toB=clock(e,{ id: -1, a: -1, join: (e.join ?? 0) >> 2 << 2, lengthM: half });
    const toA=clock(e,{ id: -2, b: -1, join: (e.join ?? 0) & 3, lengthM: half });
    node=-1; at=[toB,toA];
    way=(_n,id)=>id===-1?ways[0]:ways[1];
    platformEdge=toB;
  }
  // two-way single track splits through the station, a full box (network.ts `split`): no bar
  if (at.some((e)=>splitsAt(e, e.a===node?0:1, { platform: 1, one: 0 } as unknown as ClockNode))) return 0;
  const others=(n: number): readonly ClockEdge[] => {
    if (n===node) return at;
    const w=world.peek();
    return (w?.edges ?? []).filter((e)=>(e.a===n||e.b===n)&&!(h.kind==="edge"&&e.id===h.edge)).map((e)=>clock(e));
  };
  return stationOutside(node, platformEdge, others, (n,id)=>{
    if (n===node) return way(n,id);
    const e=world.peek()?.edges.find((x)=>x.id===id), p=tk.overlay.renderer.net?.edgePts.get(id);
    return !e||!p?new Float32Array(0):e.a===n?p:reversed(p);
  }, sh.ax, sh.ay, rightHand.peek()===false);
}

/** The platform box at (lx, ly) along `ways` (the track from the spot each way), and the twin's
 * on quad. */
export function boxGhost(tk: Tools, lx: number, ly: number, ways: ArrayLike<number>[], perM: number, bad: boolean, twinAt: { x: number; y: number; ways: ArrayLike<number>[] } | null, plat = tk.platFor(tk.stationSpot()), double = false, outside: ((sh: PlatformShape) => number) | null = null, tramLook = false) {
  const sh=platformShape(ways,0,plat/2*perM);
  // T-3332: a platform on one track of double track: its half of the box, on the pointer's side
  const one=tk.oneSideAt(lx,ly,sh,double);
  // a station on single track is one bar on its outside, as network.ts draws it (T-3540)
  const side=one||outside?.(sh)||0, mid=!one&&side!==0;
  const tram=tramLook||stationOnTram.peek();
  tk.lastBox={lx,ly,ways,perM,plat,side,tram,mid};
  const shape=ghostShape(tk, lx,ly,ways,perM,plat,side,tram,mid);
  tk.ghost ??= svgMark(tk, ["path"], false);
  tk.ring.style.display="none";
  tk.ghost.style.display="";
  tk.ghost.setAttribute("class",`station-ghost ${bad?"bad":"ok"}`);
  placeAt(tk.ghost,shape.x,shape.y,0,0,false);
  ghostPath(tk, tk.ghost.firstElementChild as SVGPathElement,shape);
  let twin=tk.ghost.children[1] as SVGPathElement|undefined;
  if (twinAt) {
    const ts=ghostShape(tk, twinAt.x,twinAt.y,twinAt.ways,perM,plat,0,tram);
    if (!twin) twin=tk.ghost.appendChild(document.createElementNS("http://www.w3.org/2000/svg","path"));
    ghostPath(tk, twin,ts);
    twin.setAttribute("transform",`translate(${ts.x-shape.x},${ts.y-shape.y})`);
    twin.style.display="";
  } else if (twin) twin.style.display="none";
}

export function stationRings(tk: Tools, h: Snap) {
  const step = stationOnTram.peek() ? 0.5 : 1;
  if (!tk.rings) {
    tk.rings=svgMark(tk, ["circle","text","circle","text"],true);
    tk.rings.setAttribute("class","station-rings");
  }
  const unit=circleOnScreen([h.lx,h.ly,unitMetres()],(x,y)=>tk.screen(x,y),toLngLat,toLocal);
  if (!unit) return;
  const {x:cx,y:cy,r:km1}=unit;
  tk.rings.style.display="";
  placeAt(tk.rings,cx,cy,0,0,false);
  const kids=tk.rings.children;
  for (let i=0;i<2;i++) {
    const r=km1*step*(i+1), c=kids[i*2], t=kids[i*2+1];
    c.setAttribute("r",String(r));
    t.setAttribute("y",String(-r-3));
    t.textContent=`${tNumber(step*(i+1))} ${unitShort()}`;
  }
}

export function checkGhost(tk: Tools) {
  const h=tk.hover, w=world.peek(), n=tk.overlay.renderer.net;
  if (!h || h.kind==="free" || !w || !n) return;
  const key=tk.altsKey(h);
  if (key===tk.guess?.key) return;
  const spots=tk.alts.length?tk.alts:[h];
  const platform=tk.platFor(h), lay=layout(tk, w);
  const id=(s:Spot)=>s.kind==="node"?`n${s.node}`:`e${s.edge}`;
  const scope=`${platform}:${w.version}`;
  if (tk.tried.scope!==scope) tk.tried={scope,at:new Map()};
  const tried=tk.tried.at;
  // the worker's answer for tk track from a spot within KNOWN_PX of tk one: a refusal the
  // layout cannot explain stands for tk spot; a spot taken there still has the layout checked
  const known=(s:Spot)=>{
    const r=tried.get(id(s));
    if (!r) return undefined;
    const [ax,ay]=tk.screen(r.lx,r.ly), [bx,by]=tk.screen(s.lx,s.ly);
    return Math.hypot(ax-bx,ay-by)<=KNOWN_PX ? r.issues : undefined;
  };
  const cand=(s:Spot):Candidate=>s.kind==="node"?{kind:"node",node:s.node}:{kind:"edge",edge:s.edge,frac:fractionAlong(n.edgePts.get(s.edge)??[],s.lx,s.ly)};
  const roomOf=(s:Spot)=>roomSnap(tk, s);
  const g=likelyFirst(spots,(s)=>known(s)||layoutIssues(cand(s),platform,lay),roomOf);
  let guess:Verdict={key,spot:g.spot??spots[0],ok:!!g.spot,why:g.spot?"":describeIssues(g.issues,undefined,undefined,true)||tr("tools.cannotPlace")};
  if (spots.length>=HOLD_AT && !(g.spot?known(g.spot)===null:spots.every((s)=>known(s)))) {
    // a bundle of tracks where the worker has not yet confirmed the guess: there it is often
    // refused for what the page cannot see, so the ghost stays as it was until the answer (if it
    // was by the pointer; else none)
    const was=tk.shown, at=was?.spot;
    let near=false;
    if (at) {
      const [ax,ay]=tk.screen(at.lx,at.ly), [bx,by]=tk.screen(h.lx,h.ly);
      near=Math.hypot(ax-bx,ay-by)<=SNAP_EDGE_PX;
    }
    guess=near&&was ? {...was,key,held:true} : {key,spot:null,ok:true,why:"",held:true};
  }
  tk.guess=guess;
  const one=tk.oneAt();
  const atOf=(s:Spot)=>s.kind==="node"?{kind:"node" as const,node:s.node,...one}:{kind:"edge" as const,edge:s.edge,...tk.pack(s.lx,s.ly),...one};
  // a track the worker refused a few px away is not asked again, so while the pointer moves each
  // new spot's answer gets further down the list
  const trial=(s:Spot)=>{
    const k=known(s);
    return k ? Promise.resolve(k) : new Promise<string[]|null>((res)=>tk.client.checkStation(atOf(s),platform,(ok,issues)=>{
      // refusals the layout explains depend on the spot; the layout finds those itself, and a split
      // refused where the layout finds no room is by a node, not on a ramp (T-1870)
      const byNode=issues.includes("SplitPoint") && layoutIssues(cand(s),platform,lay).length>0;
      if (ok || !issues.every((i)=>LAYOUT_ISSUES.has(i) || (i==="SplitPoint" && byNode))) tried.set(id(s),{issues:ok?null:issues,lx:s.lx,ly:s.ly});
      else tried.delete(id(s));
      res(ok?null:issues);
    }));
  };
  void firstAccepted(spots,trial,roomOf,()=>key!==tk.guess?.key).then(({spot,issues})=>{
    if (key!==tk.guess?.key) return; // a newer spot is already asked
    const why=spot?"":describeIssues(issues,undefined,undefined,true)||tr("tools.cannotPlace");
    tk.answer={key,spot:spot??(guess.held?spots[0]:guess.spot),ok:!!spot,why};
    if (tool.peek()==="station" && (tk.answer.ok!==guess.ok || tk.answer.why!==guess.why || tk.answer.spot!==guess.spot)) tk.place();
  });
}

export function layout(tk: Tools, w: NonNullable<ReturnType<typeof world.peek>>) {
  if (tk.lay?.version===w.version) return tk.lay;
  const net=tk.overlay.renderer.net, marks=net?.marks, crossings=new Map<number, number[]>();
  const lay=layoutOf(w.edges,w.nodes,{
    crossingsOn:(e)=>{
      let out=crossings.get(e);
      if (out) return out;
      out=[];
      const pts=net?.edgePts.get(e), edge=lay.byId.get(e);
      if (pts && edge && marks) {
        // local units per metre along tk edge; a crossing within 2 m of it is on it
        const total=lengthOf(pts), perM=total/Math.max(1,edge.lengthM);
        let x0=Infinity,y0=Infinity,x1=-Infinity,y1=-Infinity;
        for (let k=0;k<pts.length;k+=3) x0=Math.min(x0,pts[k]),x1=Math.max(x1,pts[k]),y0=Math.min(y0,pts[k+1]),y1=Math.max(y1,pts[k+1]);
        const pad=2*perM;
        for (let k=0;k+MARK_STRIDE<=marks.length;k+=MARK_STRIDE) {
          const x=marks[k],y=marks[k+1];
          if (marks[k+2]!==M_CROSSING || x<x0-pad || x>x1+pad || y<y0-pad || y>y1+pad) continue;
          const [f,off]=nearestOn(pts,x,y);
          if (off<=pad) out.push(f*edge.lengthM);
        }
      }
      crossings.set(e,out);
      return out;
    },
  });
  return tk.lay={...lay,version:w.version};
}

/** The spots near an edge spot where the platform may have room (`roomNear`, T-671, T-1870). */
export function roomSnap(tk: Tools, h: Spot): Spot[] {
  const w=world.peek(), net=tk.overlay.renderer.net, pts=h.kind==="edge"?net?.edgePts.get(h.edge):null;
  if (!w || !net || !pts || pts.length<6 || h.kind!=="edge") return [];
  return roomNear({edge:h.edge,frac:fractionAlong(pts,h.lx,h.ly)},tk.platFor(h),layout(tk, w)).flatMap((c):Spot[]=>{
    if (c.kind==="node") {
      const node=w.nodes.get(c.node);
      if (!node) return [];
      const [lx,ly]=toLocal(node.lng,node.lat);
      return [{kind:"node",node:c.node,lx,ly}];
    }
    const to=net.edgePts.get(c.edge);
    if (!to) return [];
    const [lx,ly]=pointAlong(to,c.frac);
    return [{kind:"edge",edge:c.edge,lx,ly}];
  });
}
