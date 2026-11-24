// The tools' marks on the map: the curve circle and tangent, the band, the marks and the tag (T-2374, moved out of tools.ts).

import { render } from "preact";
import { t as tr } from "../i18n";
import { Icon } from "../ui/widgets";
import { distance } from "../game/format";
import { metresBetween } from "../game/walks";
import { buildLevel, besideOn, buswayTrack, stationKind, tool, tramTrack } from "../game/ui";
import { describeRemovable } from "../game/remove";
import { realStationUnder } from "./realStations";
import { world } from "../game/world";
import { type PreviewAnswer } from "../workers/clockClient";
import { toLngLat, toLocal } from "./geo";
import { circleOnScreen, drawTangent, endHeading } from "./curveCircle";
import { placeAt } from "./screenPlace";
import { curveLabels, legCurveLabels, placeCurveLabels, placementTag, rampLabels } from "./trackTagRows";
import { MODE_ICON_COLOURS } from "../game/palette";
import { TRACK_SPECS } from "../game/trackKinds";
import { logShare } from "./modeBar";
import { NEAR_BAR_LO, NEAR_BAR_HI, BESIDE_PX, type Snap, tooFarText, withName } from "./toolParts";
import { tripsNear, lastPlacedName, el, svgMark, stationGhost, stationRings, checkGhost, brtSpot, boxGhost, BRT_STATION_PLATFORM } from "./stationTool";
import { hidePortGhost, portMarks } from "./stationPort";
import { freeGhost, freeHint, freeStarted } from "./stationFree";
import type { Tools } from "./tools";

export function curveCircle(tk: Tools, pv: PreviewAnswer | null) {
  const c = circleOnScreen(pv?.arc, (x, y) => tk.screen(x, y), toLngLat, toLocal);
  if (!c) {
    if (tk.circle) tk.circle.style.display = "none";
    return;
  }
  tk.circle ??= svgMark(tk, ["circle"], true);
  tk.circle.style.display = "";
  tk.circle.setAttribute("class", "curve-circle" + (pv!.ok ? "" : " bad"));
  placeAt(tk.circle, c.x, c.y, 0, 0, false);
  tk.circle.firstElementChild!.setAttribute("r", String(c.r));
}

export function curveTangent(tk: Tools, pv: PreviewAnswer | null) {
  const straightFromFree = tk.start?.end.kind === "free" && !(pv?.arc && pv.arc.length >= 3 && pv.arc[2] > 0);
  const h = pv && !straightFromFree ? endHeading(pv.verts, pv.pts, pv.arc, toLngLat, toLocal) : null;
  if (!h) {
    if (tk.tangent) tk.tangent.style.display = "none";
    return;
  }
  tk.tangent ??= svgMark(tk, ["line"], true);
  drawTangent(tk.tangent, h, pv!.ok, (x, y) => tk.screen(x, y));
}

export function drawBand(tk: Tools) {
  const n = tk.overlay.renderer.net;
  const on = tool.peek() === "track" && besideOn.peek() && !!n;
  if (!on) {
    if (tk.bandEl) {
      tk.bandEl.style.display = "none";
      tk.bandEl.width = tk.bandEl.height = 0; // T-3497: no backing store while hidden
    }
    tk.bandKey = "";
    return;
  }
  const c = tk.map.getCenter(), W = tk.overlay.cssW, H = tk.overlay.cssH, dpr = devicePixelRatio || 1;
  const key = `${c.lng},${c.lat},${tk.map.getZoom()},${tk.map.getBearing()},${tk.map.getPitch()},${W},${H},${world.peek()?.version},${document.documentElement.dataset.shade}`;
  if (tk.bandEl && key === tk.bandKey) return;
  tk.bandKey = key;
  if (!tk.bandEl) {
    tk.bandEl = document.createElement("canvas");
    tk.bandEl.className = "beside-band";
    tk.box.prepend(tk.bandEl);
  }
  const el = tk.bandEl;
  el.style.display = "";
  if (el.width !== Math.round(W * dpr) || el.height !== Math.round(H * dpr)) {
    el.width = Math.round(W * dpr);
    el.height = Math.round(H * dpr);
    el.style.width = `${W}px`;
    el.style.height = `${H}px`;
  }
  const g = el.getContext("2d")!;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  g.beginPath();
  for (const pts of n!.edgePts.values()) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    const xy: number[] = [];
    for (let k = 0; k + 2 < pts.length; k += 3) {
      const [x, y] = tk.screen(pts[k], pts[k + 1]);
      xy.push(x, y);
      x0 = Math.min(x0, x), y0 = Math.min(y0, y), x1 = Math.max(x1, x), y1 = Math.max(y1, y);
    }
    if (x1 < -BESIDE_PX || y1 < -BESIDE_PX || x0 > W + BESIDE_PX || y0 > H + BESIDE_PX) continue;
    g.moveTo(xy[0], xy[1]);
    for (let k = 2; k < xy.length; k += 2) g.lineTo(xy[k], xy[k + 1]);
  }
  g.lineWidth = BESIDE_PX * 2;
  g.lineCap = g.lineJoin = "round";
  g.strokeStyle = getComputedStyle(el).getPropertyValue("--beside").trim() || "#3d82d6";
  g.globalAlpha = 0.1;
  g.stroke();
}

export function placeMarks(tk: Tools) {
  let k = 0;
  const put = (m: HTMLDivElement, lx: number, ly: number, dx = 0, dy = 0) => {
    const [x, y] = tk.screen(lx, ly);
    placeAt(m, x, y, dx, dy);
  };
  const t = tool.peek();
  if (t === "track" && tk.start) {
    const pv = tk.shownPreview;
    // where the segment starts, which from a station is past its platform (T-1155)
    if (pv && pv.verts.length >= 2) put(el(tk, k++, "pi-mark"), pv.verts[0], pv.verts[1]);
    else put(el(tk, k++, "pi-mark"), ...toLocal(tk.start.lng, tk.start.lat));
    if (pv) {
      const from = k;
      k += placeCurveLabels([...legCurveLabels(pv.legs), ...rampLabels(pv.ramps), ...curveLabels(pv.verts)], (x, y) => tk.screen(x, y), tk.overlay.cssW, tk.overlay.cssH, (i) => el(tk, from + i, "pi-label"));
    }
  }
  for (let i = k; i < tk.marks.length; i++) tk.marks[i].style.display = "none";
  curveCircle(tk, t === "track" && tk.start ? tk.shownPreview : null);
  curveTangent(tk, t === "track" && tk.start ? tk.shownPreview : null);

  // the platform ghost shows only while the station tool snaps (below)
  if (tk.ghost) tk.ghost.style.display = "none";
  if (tk.rings) tk.rings.style.display = "none";
  if (t === "bus") {
    tk.ring.style.display = "none";
    tk.bus.place();
    return;
  }
  if (t === "ferry") {
    tk.ring.style.display = "none";
    tk.ferry.place();
    return;
  }
  if (t === "building") {
    tk.ring.style.display = "none";
    tk.building.place();
    return;
  }
  // the delete tool: a ring on the station or flyover, and what a click removes by the cursor
  if (t === "delete") {
    const d = tk.delHover, what = d && describeRemovable(d);
    if (d && d.kind !== "track" && d.kind !== "bus" && d.kind !== "building") {
      tk.ring.className = "snap-ring";
      tk.ring.style.display = "";
      put(tk.ring, d.lx, d.ly);
    } else tk.ring.style.display = "none";
    if (what) showTag(tk, tr("tools.removeWhat", { what: what.what }), false, ...tk.delAt);
    else tk.tag.style.display = "none";
    return;
  }
  // the snap ring and the cursor tag
  const kind = t === "station" ? stationKind.peek() : null;
  if (kind !== "port") hidePortGhost();
  if (kind === "port") {
    tk.ring.style.display = "none";
    tk.shown = null;
    // T-1777: a pointer gone off the map (no hover) leaves no ghost or tag
    if (!tk.hover) return void (hidePortGhost(), (tk.tag.style.display = "none"));
    return portMarks(tk, (c, bad, x, y) => showTag(tk, c, bad, x, y));
  }
  // T-3802: the real station ring under the pointer, named first in the tag (it holds the hover's back)
  const real = t === "station" && tk.pointer ? realStationUnder(tk.pointer.lng, tk.pointer.lat) : null;
  if (kind === "bus") {
    // T-3378: a BRT station's ghost along the road or bus route by the pointer; T-3999: the bus kind only
    tk.ring.style.display = "none";
    tk.shown = null;
    const p = tk.pointer;
    if (!p || !tk.hover || tk.ghostSpent) return void (tk.tag.style.display = "none");
    const b = brtSpot(tk, p.lng, p.lat), here: Snap = { kind: "free", lx: 0, ly: 0 };
    if (b) {
      boxGhost(tk, b.lx, b.ly, b.ways, b.perM, false, null, BRT_STATION_PLATFORM, false, null, true);
      stationRings(tk, { ...here, lx: b.lx, ly: b.ly });
      const lines = helpLines(tk, { ...here, lx: b.lx, ly: b.ly }, null);
      if (lines.length) showTag(tk, lines, false, p.px, p.py);
      else tk.tag.style.display = "none";
    } else if (b === null) showTag(tk, ...withName(real, tr("tools.notBesideRoad")), p.px, p.py);
    else tk.tag.style.display = "none";
    return;
  }
  const row = (s: string) => Object.assign(document.createElement("div"), { textContent: s });
  if (t === "station" && freeStarted()) {
    // T-3999 (T-794, T-907): after the first click on open ground, the station and its own piece toward the pointer
    tk.ring.style.display = "none";
    tk.shown = null;
    const g = freeGhost(tk, (lx, ly, ways, perM, bad, plat, tram) => boxGhost(tk, lx, ly, ways, perM, bad, null, plat, false, null, tram));
    const p = tk.pointer;
    if (!g || !p) return void (tk.tag.style.display = "none");
    stationRings(tk, g.mid);
    if (g.ok === false) return void showTag(tk, g.why, true, p.px, p.py);
    const lines = helpLines(tk, g.mid, freeHint());
    if (g.text) lines.splice(1, 0, row(g.text));
    showTag(tk, lines, false, p.px, p.py);
    return;
  }
  if (t === "station") checkGhost(tk);
  const h = t === "station" ? (tk.ghostSpent ? null : tk.stationSpot()) : tk.hover;
  const v = t === "station" ? tk.verdict() : null;
  if (t === "station") tk.shown = h && h.kind !== "free" ? v : null;
  if (t === "station" && h?.kind === "free") {
    // T-3999: open ground takes a station by two clicks (its first end, then its direction); the ring marks the first
    tk.ring.className = "snap-ring";
    tk.ring.style.display = "";
    put(tk.ring, h.lx, h.ly);
    const lines = helpLines(tk, h, real);
    lines.splice(real ? 1 : 0, 0, row(freeHint()));
    showTag(tk, lines, false, ...tk.screen(h.lx, h.ly));
    return;
  }
  // T-1721: the track tool's point: red branching off a piece, yellow joining at a node, blue beside track
  const hb = t === "track" && h?.kind === "free" ? tk.hoverBeside : null;
  tk.ring.className = "snap-ring" + (t !== "track" ? "" : hb ? " beside" : h?.kind === "edge" ? " branch" : h?.kind === "node" ? (h.merge ? " join merge" : " join") : "");
  if ((t === "track" || t === "station") && h && h.kind !== "free") {
    tk.ring.style.display = "";
    if (t === "station") {
      stationRings(tk, h);
      stationGhost(tk, h);
    } else put(tk.ring, h.lx, h.ly, ...(h.dot ?? [0, 0]));
  } else if (hb) {
    tk.ring.style.display = "";
    put(tk.ring, hb.lx, hb.ly);
  } else if (t === "track" && tk.real.ringAt) {
    tk.ring.style.display = "";
    put(tk.ring, ...toLocal(...tk.real.ringAt));
  } else if (t === "track" && !tk.start && h?.kind === "free" && tk.street.ringAt) {
    tk.ring.style.display = "";
    put(tk.ring, ...toLocal(...tk.street.ringAt));
  } else tk.ring.style.display = "none";
  const pv = tk.shownPreview, rt = t === "track" && tk.start && h ? tk.real.tag() ?? tk.street.tag() : null;
  const at = h ? tk.screen(h.lx, h.ly) : null;
  if (rt) {
    // T-727d: a real route still being found (the spinner once it takes a while), or none
    const kids: (string | Node)[] = [];
    if (rt.spin) kids.push(Object.assign(document.createElement("i"), { className: "spin" }));
    if (rt.text) kids.push(rt.text);
    if (kids.length) showTag(tk, kids, rt.bad, ...at!);
    else tk.tag.style.display = "none";
  } else if (t === "track" && tk.start && h && !tk.snapped && tk.tooFar(h)) {
    showTag(tk, tooFarText(), true, ...at!);
  } else if (t === "track" && tk.start && h && (pv || tk.waiting())) {
    const rows: HTMLDivElement[] = pv ? placementTag(pv, buildLevel.peek(), !tramTrack.peek() ? TRACK_SPECS.regular : buswayTrack.peek() ? TRACK_SPECS.busway : tk.street.active ? TRACK_SPECS.streetTram : TRACK_SPECS.tram) : [document.createElement("div")];
    // T-1039: still working on the newest ghost or the click
    if (tk.waiting()) rows[0].prepend(Object.assign(document.createElement("i"), { className: "spin row-dot" }));
    showTag(tk, rows, false, ...at!);
  } else if (t === "station" && h && h.kind !== "free" && v?.ok === false) {
    // why a click here would be refused (T-311)
    showTag(tk, ...withName(real, v.why), ...at!);
  } else if (t === "station" && h && h.kind !== "free") {
    const lines = helpLines(tk, h, real);
    if (lines.length) showTag(tk, lines, false, ...at!);
    else tk.tag.style.display = "none";
  } else tk.tag.style.display = "none";
}

const KIND_ICON = { rail: Icon.train, tram: Icon.tram, bus: Icon.brt, port: Icon.anchor };
const KIND_COLOUR = { rail: MODE_ICON_COLOURS.rail, tram: MODE_ICON_COLOURS.tram, bus: MODE_ICON_COLOURS.brt, port: MODE_ICON_COLOURS.ferry };

/** The cursor tag with `content`, red when `bad`, by the screen point (x, y). */
export function showTag(tk: Tools, content: string | (string | Node)[], bad: boolean, x: number, y: number) {
  tk.tag.style.display = "";
  tk.tag.className = "tool-tag build-tag" + (bad ? " bad" : "");
  if (typeof content === "string") tk.tag.textContent = content;
  else tk.tag.replaceChildren(...content);
  placeAt(tk.tag, x, y, 14, 12);
}

/** The station tool's tag lines: `first` if any, how far from the station placed last (T-312),
 * the trips near (T-412) over a bar of their size on a log scale (T-2271). */
export function helpLines(tk: Tools, h: Snap, first: string | null): HTMLDivElement[] {
  const row = (s: string) => Object.assign(document.createElement("div"), { textContent: s });
  const lines = first ? [row(first)] : [];
  const last = lastPlacedName(tk), [lng, lat] = toLngLat(h.lx, h.ly);
  if (last) lines.push(row(tr("tools.fromLast", { distance: distance(metresBetween(lng, lat, last.lng, last.lat)), name: last.name })));
  const trips = tripsNear(tk, lng, lat);
  if (trips) {
    const colour = KIND_COLOUR[stationKind.peek()];
    const bar = Object.assign(document.createElement("div"), { className: "size-bar" });
    bar.style.setProperty("--kind", colour);
    const fill = document.createElement("i");
    fill.style.width = `${(100 * logShare(trips.legs, NEAR_BAR_LO, NEAR_BAR_HI)).toFixed(1)}%`;
    bar.append(fill);
    const text = row(trips.text);
    // T-4105, T-4135: the row starts with the icon of the kind a click would place; no row names the kind
    const icon = Object.assign(document.createElement("span"), { className: "kind-icon" });
    icon.style.color = colour;
    render(KIND_ICON[stationKind.peek()](), icon);
    text.prepend(icon);
    text.className = "near-row";
    lines.push(text, bar);
  }
  return lines;
}
