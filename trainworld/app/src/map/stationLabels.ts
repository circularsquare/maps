// Station names as DOM labels above the overlay canvas, in the UI font (Zen Maru Gothic). A
// MapLibre symbol layer would sit under the overlay, with track drawn through the names. Labels
// move on MapLibre's frames only (the camera does not change otherwise). A flat pan translates
// one compositor layer; other camera changes re-place names. A greedy pass hides any label that
// overlaps a more important one (transfers first, then lines, then constructed). Stations of one
// complex that share a name have one label between them (T-169, labelGroups below). Capacity tags
// are placed after the names and keep off them (T-184, rects() and onPlace).
//
// Winning over basemap place names (T-032): an invisible symbol layer on top of the basemap holds
// the same names at the same spots. MapLibre places symbols from the top layer down, so basemap
// labels that would collide with a station name are dropped by MapLibre itself; the shadow is
// never drawn (opacity 0) and is only rebuilt when stations change.

import type { FeatureCollection } from "geojson";
import type { GeoJSONSource, Map as MlMap } from "maplibre-gl";
import type { Complexes } from "../game/complexes";
import { complexes } from "../game/complexView";
import type { StationInput, WorldView } from "../game/types";
import { anchor, toLocal } from "./geo";
import type { Overlay } from "./overlay";
import { perfOff, perfTry } from "../perfFlags";
import { MIN_LABEL_ZOOM, labelPadding } from "./labelLayout";
import { plainBoxHalfPx } from "./stationGeometry";

/** ?perfOff=labels (T-045, measurement only): DOM labels never placed. */
const NO_LABELS = perfOff("labels");
const LAYERED = perfTry("labelLayers");
const PAN = !perfOff("labelPan");

interface Label {
  el: HTMLDivElement;
  x: number;
  y: number;
  /** label to the right of the dot (track runs north-south) or below it (east-west) */
  below: boolean;
  w: number;
  h: number;
}

const GAP = 10; // px from the station centre
const SHADOW = "tw-station-shadow";
// Keep names beyond the viewport ready to enter during a pan. Rebase before this runs out.
const PAN_MARGIN = 256;

export class StationLabels {
  private box: HTMLDivElement;
  private plane: HTMLDivElement;
  private labels: Label[] = [];
  private visible = true;
  private shown = false;
  private shadow: FeatureCollection = { type: "FeatureCollection", features: [] };
  private key = "";
  private placedMatrix: number[] | null = null;
  private placedSize: [number, number] = [0, 0];
  private anchor: [number, number] = [0, 0];
  private needsPlacement = true;
  /** Shown labels' boxes in screen px at the last full placement, before the pan offset. */
  private placed: [number, number, number, number][] = [];
  private offset: [number, number] = [0, 0];
  /** Bumped whenever the shown names change place, so capacity tags know to place again (T-184). */
  version = 0;
  /** Called after the names change place outside a pan (capacity tags follow, T-184). */
  onPlace: (() => void) | null = null;

  constructor(container: HTMLElement, private overlay: Overlay, private map: MlMap) {
    this.box = document.createElement("div");
    this.box.className = "stn-labels";
    this.box.style.display = "none";
    container.appendChild(this.box);
    this.plane = document.createElement("div");
    this.plane.className = "stn-label-plane";
    if (PAN) this.plane.style.willChange = "transform";
    this.box.appendChild(this.plane);
    overlay.afterMapFrame.push(() => this.place());
    // moveend can precede the final rendered camera: place on that frame, not the old matrix.
    map.on("moveend", () => {
      this.needsPlacement = true;
      map.triggerRepaint();
    });
    // widths measured before the web font arrived are wrong: measure again once it has
    document.fonts?.ready.then(() => {
      for (const l of this.labels) l.w = 0;
      this.needsPlacement = true;
      this.place();
    });
    map.on("style.load", () => this.addShadow());
    if (map.isStyleLoaded()) this.addShadow();
  }

  /** The invisible copy of the names, above every basemap layer. */
  private addShadow() {
    const map = this.map;
    if (map.getLayer(SHADOW) || perfOff("shadow")) return;
    // The basemap's own label font, so the glyphs exist on its glyph server.
    const font = map.getStyle().layers.find((l) => l.type === "symbol" && (l.layout as any)?.["text-font"])?.layout as any;
    if (!map.getSource(SHADOW)) map.addSource(SHADOW, { type: "geojson", data: this.shadow });
    map.addLayer({
      id: SHADOW,
      minzoom: MIN_LABEL_ZOOM,
      type: "symbol",
      source: SHADOW,
      layout: {
        "text-field": ["get", "name"],
        "text-font": font?.["text-font"] ?? ["Noto Sans Regular"],
        "text-size": 13,
        "text-anchor": ["get", "anchor"],
        "text-offset": ["get", "offset"],
        "text-allow-overlap": true,
        "text-ignore-placement": false,
        "text-padding": 4,
      },
      paint: { "text-opacity": 0 },
    });
  }

  set(w: WorldView) {
    const stations = w.save.stations;
    const lines = new Map<string, Set<string>>();
    for (const l of w.save.lines) for (const s of l.stops) {
      const set = lines.get(s);
      if (set) set.add(l.id);
      else lines.set(s, new Set([l.id]));
    }
    // Most edits leave the stations as they were: keep the labels and the shadow layer then
    // (rebuilding 470 labels and the shadow source took ~100 ms an edit on a 2,000 km network, T-063).
    // Length and the player's links are in the key because they decide complexes (T-209).
    const links = w.save.links ?? [];
    // and the draw anchor: the names' positions are in its local units (T-307)
    const key = `${anchor().version}\n` + stations.map((s) => `${s.id}|${s.name}|${s.built}|${s.lng}|${s.lat}|${s.heading}|${s.length}|${[...(lines.get(s.id) ?? [])].join(",")}`).join("\n") + "\n" + links.join("|");
    if (key === this.key) return;
    this.key = key;
    // the world's complexes (game/complexView.ts), built once for every user of this world
    const names = labelGroups(stations, lines, complexes.peek());
    this.needsPlacement = true;
    // A label whose text and look are unchanged keeps its element and measured size (T-496): a new
    // station then costs one new element, not a rebuild and a layout of every name. Shown labels
    // never overlap, so their order in the DOM draws nothing differently.
    const spare = new Map<string, Label[]>();
    for (const l of this.labels) {
      const k = l.el.className + "\u0000" + l.el.textContent;
      const v = spare.get(k);
      if (v) v.push(l);
      else spare.set(k, [l]);
    }
    this.labels = names.map((g) => {
      const cls = "stn-label" + (g.built ? "" : " plan");
      const [x, y] = toLocal(g.lng, g.lat);
      const below = Math.abs(Math.cos(g.heading)) > Math.abs(Math.sin(g.heading));
      const old = spare.get(cls + "\u0000" + g.name)?.pop();
      if (old) return { ...old, x, y, below };
      const el = document.createElement("div");
      el.className = cls;
      if (LAYERED) el.style.willChange = "transform"; // ?perfTry=labelLayers (T-045)
      el.textContent = g.name;
      this.plane.appendChild(el);
      return { el, x, y, below, w: 0, h: 0 };
    });
    for (const v of spare.values()) for (const l of v) l.el.remove();
    this.shadow = {
      type: "FeatureCollection",
      features: names.map((g, i) => ({
        type: "Feature",
        geometry: { type: "Point", coordinates: [g.lng, g.lat] },
        properties: { name: g.name, anchor: this.labels[i].below ? "top" : "left", offset: this.labels[i].below ? [0, 0.6] : [0.7, 0] },
      })),
    };
    (this.map.getSource(SHADOW) as GeoJSONSource | undefined)?.setData(this.shadow);
    this.place();
  }

  setVisible(on: boolean) {
    this.visible = on;
    if (this.map.getLayer(SHADOW)) this.map.setLayoutProperty(SHADOW, "visibility", on ? "visible" : "none");
    this.needsPlacement = true;
    this.place();
  }

  /** The shown names' boxes in screen px as they are now, pan included; capacity tags keep off
   * them (T-184). */
  rects(): [number, number, number, number][] {
    if (!this.shown) return [];
    const [dx, dy] = this.offset;
    return this.placed.map((r) => [r[0] + dx, r[1] + dy, r[2] + dx, r[3] + dy]);
  }

  place() {
    const zoom = this.map.getZoom();
    const show = this.visible && zoom >= MIN_LABEL_ZOOM && !NO_LABELS;
    if (show !== this.shown) {
      this.shown = show;
      this.box.style.display = show ? "" : "none";
      this.needsPlacement = true;
      this.version++;
      if (!show) this.onPlace?.();
    }
    if (!show || !this.overlay.camMatrix) return;
    const W = this.overlay.cssW, H = this.overlay.cssH;
    const m = this.overlay.camMatrix;
    const prev = this.placedMatrix;
    // A flat mercator pan changes only translation. Zoom, rotation, pitch, padding and resize
    // fall back to individual placement, using the same camera as the network's draw.
    const same = (a: number, b: number) => Math.abs(a - b) <= 1e-10 * Math.max(1, Math.abs(a), Math.abs(b));
    if (PAN && !this.needsPlacement && prev && W === this.placedSize[0] && H === this.placedSize[1]
      && m[3] === 0 && m[7] === 0
      && prev.every((v, i) => i === 12 || i === 13 || i === 14 || same(v, m[i]))) {
      const [x, y] = this.overlay.toScreen(0, 0);
      const dx = x - this.anchor[0], dy = y - this.anchor[1];
      if (Math.abs(dx) < PAN_MARGIN / 2 && Math.abs(dy) < PAN_MARGIN / 2) {
        const transform = `translate(${dx}px, ${dy}px)`;
        if (this.plane.style.transform !== transform) this.plane.style.transform = transform;
        this.offset = [dx, dy];
        return;
      }
    }
    this.needsPlacement = false;
    this.placedMatrix = Array.from(m);
    this.placedSize = [W, H];
    this.anchor = this.overlay.toScreen(0, 0);
    this.plane.style.transform = "";
    this.offset = [0, 0];
    this.placed = [];
    const margin = PAN ? PAN_MARGIN : 0;
    // Reserve breathing room around each name, especially at neighbourhood scale.
    const [padX, padY] = labelPadding(zoom);
    const taken: [number, number, number, number][] = [];
    // measure every new label first, then move them: reads between writes force a layout each
    for (const l of this.labels) {
      if (!l.w) {
        l.w = l.el.offsetWidth;
        l.h = l.el.offsetHeight;
      }
    }
    // clear of a platform box, which widens as the map zooms in (T-204), or of a transfer dot (T-313)
    const ss = this.overlay.renderer.stationStyle;
    const gap = Math.max(GAP, plainBoxHalfPx(ss, zoom) + 4, ss.boxes ? 0 : ss.radius[1] + 3);
    for (const l of this.labels) {
      const [sx, sy] = this.overlay.toScreen(l.x, l.y);
      const x = l.below ? sx - l.w / 2 : sx + gap;
      const y = l.below ? sy + gap - 2 : sy - l.h / 2;
      const r: [number, number, number, number] = [x, y, x + l.w, y + l.h];
      const collision: [number, number, number, number] = [r[0] - padX, r[1] - padY, r[2] + padX, r[3] + padY];
      const off = r[2] < -margin || r[0] > W + margin || r[3] < -margin || r[1] > H + margin;
      const hit = !off && taken.some((t) => collision[0] < t[2] && collision[2] > t[0] && collision[1] < t[3] && collision[3] > t[1]);
      if (off || hit) {
        l.el.style.visibility = "hidden";
        continue;
      }
      taken.push(collision);
      l.el.style.visibility = "";
      const rx = Math.round(x), ry = Math.round(y);
      l.el.style.transform = `translate(${rx}px, ${ry}px)`;
      this.placed.push([rx, ry, rx + l.w, ry + l.h]);
    }
    this.version++;
    this.onPlace?.();
  }
}

interface NameGroup {
  name: string;
  lng: number;
  lat: number;
  heading: number;
  built: boolean;
  /** distinct lines stopping at any of its stations */
  lines: number;
}

/** One label per distinct name per station complex (T-169): stations of one complex that share a
 * name (three Times Sq platforms) get one label at their mean position, oriented like the member
 * with the most lines; differently named members keep their own. Most important first: most
 * lines, then constructed, then save order. The player's links count, as everywhere a complex is
 * used (T-197, T-209). */
function labelGroups(stations: readonly StationInput[], lines: ReadonlyMap<string, ReadonlySet<string>>, cx: Complexes): NameGroup[] {
  const groups = new Map<string, StationInput[]>();
  for (const s of stations) {
    const k = `${cx.of.get(s.id)}\u0000${s.name}`;
    const g = groups.get(k);
    if (g) g.push(s);
    else groups.set(k, [s]);
  }
  const out: NameGroup[] = [];
  for (const members of groups.values()) {
    const ids = new Set<string>();
    let lng = 0, lat = 0, lead = members[0];
    for (const s of members) {
      lng += s.lng;
      lat += s.lat;
      for (const l of lines.get(s.id) ?? []) ids.add(l);
      if ((lines.get(s.id)?.size ?? 0) > (lines.get(lead.id)?.size ?? 0)) lead = s;
    }
    out.push({ name: lead.name, lng: lng / members.length, lat: lat / members.length, heading: lead.heading, built: members.some((s) => s.built), lines: ids.size });
  }
  return out.sort((a, b) => b.lines - a.lines || Number(b.built) - Number(a.built));
}
