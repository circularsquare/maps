// The delete tool (T-096): what a click removes, hovered and clicked (T-2374, moved out of tools.ts).

import { type MapMouseEvent } from "maplibre-gl";
import { lngLatToPack } from "../game/coords";
import { tool } from "../game/ui";
import { removeThing, stationStub, stationTrim, type Removable } from "../game/remove";
import { pointerLocal } from "./pick";
import { edgesNear } from "./snapIndex";
import { world } from "../game/world";
import { toLocal } from "./geo";
import { strokesFromParts } from "./network";
import { fractionAlong } from "./stationSnap";
import { touchInput } from "./touch";
import { buildingAtPack } from "../game/buildingView";
import { BuildingFloors } from "./buildingFloors";
import { type Snap, trimPolyline } from "./toolParts";
import { touchStop } from "./toolTouch";
import type { Tools } from "./tools";

/** What a click with the delete tool would remove: a station, a flyover, else a stretch of
 * track, else a bus line (train lines are removed from the Lines tab). */
export function deletePick(tk: Tools, px: number, py: number): (Removable & { lx: number; ly: number }) | null {
  const w = world.peek();
  const n = tk.overlay.renderer.net;
  if (!w || !n) return null;
  const near = (lx: number, ly: number, slop: number) => {
    const [x, y] = tk.screen(lx, ly);
    const d = Math.hypot(x - px, y - py);
    return d < slop ? d : Infinity;
  };
  let best = Infinity, hit: (Removable & { lx: number; ly: number }) | null = null;
  const station=tk.overlay.renderer.stationAt(px,py,(x,y)=>tk.screen(x,y),5);
  if (station>=0) {
    // T-3629: track under the box that is not the station's own platform stretch (a siding) is the pick
    const node = Number(n.stationIds[station]);
    const s = tk.snap(px, py, tk.map.unproject([px, py]));
    if (s.kind === "edge" && !w.edges.some((e) => e.id === s.edge && (e.a === node || e.b === node))) return { kind: "track", edge: s.edge, lx: s.lx, ly: s.ly };
    return {kind:"station",node,lx:n.stations[station*3],ly:n.stations[station*3+1]};
  }
  for (const node of w.nodes.values()) {
    if (!node.flying || node.ports < 3) continue;
    const [lx, ly] = toLocal(node.lng, node.lat);
    const d = near(lx, ly, 10);
    if (d < best) [best, hit] = [d, { kind: "flyover", node: node.id, lx, ly }];
  }
  if (hit) return hit;
  const s = tk.snap(px, py, tk.map.unproject([px, py]));
  const under = stationUnder(tk, s);
  if (under) return under;
  if (s.kind === "edge") return { kind: "track", edge: s.edge, lx: s.lx, ly: s.ly };
  if (s.kind === "node") {
    // a plain node (track end, junction): the stretch nearest to the pointer there
    let e = -1, bd = Infinity;
    const tryPoint = (edge: number, pts: Float32Array, k: number) => {
      const d = near(pts[k], pts[k + 1], 14);
      if (d < bd) [bd, e] = [d, edge];
    };
    const at = pointerLocal(tk.overlay, px, py), close = at ? edgesNear(n, at[0], at[1], 14 * at[2]) : null;
    if (close) {
      for (const { edge, pts, ks } of close) for (const k of ks) tryPoint(edge, pts, k), tryPoint(edge, pts, k + 3);
    } else {
      for (const [edge, pts] of n.edgePts) for (let k = 0; k < pts.length; k += 3) tryPoint(edge, pts, k);
    }
    if (e >= 0) return { kind: "track", edge: e, lx: s.lx, ly: s.ly };
  }
  // a bus line has no track: under the rail, so only where no rail is (T-1513)
  const buses = tk.overlay.renderer.buses;
  if (buses.any) {
    const line = buses.stopAt(tk.overlay, px, py)?.line ?? buses.lineAt(tk.overlay, px, py);
    const at = tk.overlay.screenToLocal(px, py);
    if (line >= 0 && at) return { kind: "bus", line, lx: at[0], ly: at[1] };
  }
  // T-2218: a station building's floor, under everything else
  if (!BuildingFloors.shownAt(tk.map.getZoom())) return null;
  const ll = tk.map.unproject([px, py]), q = lngLatToPack(ll.lng, ll.lat), b = buildingAtPack(q.x, q.y);
  const here = b && tk.overlay.screenToLocal(px, py);
  if (b && here) return { kind: "building", id: b.id, lx: here[0], ly: here[1] };
  return null;
}

/** T-2412: the delete tool on a station's node or its platform's stretch of track (half the platform
 * and 4 m either side, what `stationTrim` keeps; all of a terminus's platform) takes the station,
 * whose own mark is small zoomed out; the track stays, as when the station is clicked. So does a
 * station's short end piece, bare end included, which a removal could only lay back (T-2797). Not
 * where stations are hidden (T-205). */
export function stationUnder(tk: Tools, s: Snap): (Removable & { lx: number; ly: number }) | null {
  const w = world.peek(), n = tk.overlay.renderer.net;
  if (!w || !n || tk.overlay.renderer.stationStyle.hidden || (s.kind !== "node" && s.kind !== "edge")) return null;
  const station = (id: number) => {
    const node = w.nodes.get(id);
    return node && node.platform > 0 && !node.bus && !node.port ? node : null;
  };
  const hit = (id: number) => {
    const node = w.nodes.get(id)!, [lx, ly] = toLocal(node.lng, node.lat);
    return { kind: "station" as const, node: id, lx, ly };
  };
  const stub = (edge: number | undefined) => {
    const id = edge === undefined ? null : stationStub(edge);
    return id !== null && station(id) ? hit(id) : null;
  };
  if (s.kind === "node") {
    if (station(s.node)) return hit(s.node);
    return w.nodes.get(s.node)?.ports === 1 ? stub(w.edges.find((x) => x.a === s.node || x.b === s.node)?.id) : null;
  }
  const held = stub(s.edge);
  if (held) return held;
  const e = w.edges.find((x) => x.id === s.edge), pts = n.edgePts.get(s.edge);
  if (!e || !pts || e.crossover) return null;
  const along = fractionAlong(pts, s.lx, s.ly) * e.lengthM;
  const reach = (id: number, d: number) => {
    const node = station(id);
    return node ? (node.ports <= 1 ? node.platform : node.platform / 2) + 4 - d : -Infinity;
  };
  const ra = reach(e.a, along), rb = reach(e.b, e.lengthM - along);
  if (ra < 0 && rb < 0) return null;
  return hit(ra >= rb ? e.a : e.b);
}

/** T-1872: the red stretch is the shared preview, so it goes with the hover whichever way the pointer or tool leaves. */
export function clearDelHover(tk: Tools) {
  if (!tk.delHover) return;
  tk.delHover = null;
  tk.clearPreview();
}

export function deleteHover(tk: Tools, px: number, py: number) {
  const h = deletePick(tk, px, py);
  const same = (a: Removable | null, b: Removable | null) => JSON.stringify(a && { ...a, lx: 0, ly: 0 }) === JSON.stringify(b && { ...b, lx: 0, ly: 0 });
  const changed = !same(h, tk.delHover);
  tk.delHover = h;
  tk.delAt = [px, py];
  if (changed) {
    // the stretch that would go, drawn red over itself
    // a crossover goes whole: both diagonals (T-391)
    const edgePts = tk.overlay.renderer.net?.edgePts;
    const partner = h?.kind === "track" ? world.peek()?.edges.find((x) => x.id === h.edge)?.partner ?? -1 : -1;
    const parts = h?.kind === "track" && edgePts
      ? [h.edge, partner].map((id) => edgePts.get(id)).filter((p): p is Float32Array => !!p && p.length >= 6)
      : [];
    // T-1161: past a station, only what goes
    if (h?.kind === "track" && parts.length) {
      const [ka, kb] = stationTrim(h.edge), len = world.peek()?.edges.find((x) => x.id === h.edge)?.lengthM ?? 0;
      if ((ka || kb) && len > 0) parts[0] = trimPolyline(parts[0], ka / len, kb / len);
    }
    tk.overlay.renderer.setPreview(parts.length ? strokesFromParts(parts, true, false, h?.kind === "track" && world.peek()?.edges.find((x) => x.id === h.edge)?.tracks === 1 ? 0 : 255) : null);
    tk.overlay.invalidate();
  }
  tk.map.getCanvas().style.cursor = h ? "pointer" : "crosshair";
  tk.place();
}

export function deleteClick(tk: Tools, e: MapMouseEvent) {
  const h = deletePick(tk, e.point.x, e.point.y);
  if (!h) return;
  // on touch nothing is shown until the next tap (T-1260)
  void removeThing(h).then(() => touchInput.peek() ? touchStop(tk) : tool.peek() === "delete" && deleteHover(tk, e.point.x, e.point.y));
}
