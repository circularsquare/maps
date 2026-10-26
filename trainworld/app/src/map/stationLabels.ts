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
import { complexes } from "../game/complexView";
import { PATCH_MODE } from "../game/patchMode";
import type { WorldView } from "../game/types";
import { NameIndex, labelGroups, linesAt, type NameGroup, type NameOp } from "./labelNames";
import { anchor, toLocal } from "./geo";
import type { Overlay } from "./overlay";
import { PlaceThrottle } from "./moving";
import { perfOff, perfTry } from "../perfFlags";
import { BoxGrid, MIN_LABEL_ZOOM, labelPadding } from "./labelLayout";
import { styleUp } from "./styleReady";
import { plainBoxHalfPx } from "./stationGeometry";
import { afterPaint } from "../game/busy";
import { ribbonColours } from "../game/openingDay";

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
  /** what was last written to its style, so a placement writes only what changed (T-689) */
  hidden?: boolean;
  transform?: string;
  /** the opening-day bunting's background (T-1434), "" for none */
  rib: string;
}

const GAP = 10; // px from the station centre
/** more stations than this named unlike every label drawn wait for the paint (T-1357) */
const DEFER_NAMES = 200;
/** side of the screen grid the collision pass files placed names in, CSS px */
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
  private deferred: WorldView | null = null;
  /** the names to draw, kept up to date from the stations that changed (T-1380); `labels` and the
   * shadow's features follow it index for index */
  private names = new NameIndex();
  /** the shadow's features' text, the same order as `labels`, to tell a change from none */
  private shadowKeys: string[] = [];
  private shadowKey = "";
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

  private throttle: PlaceThrottle;

  constructor(container: HTMLElement, private overlay: Overlay, private map: MlMap) {
    this.throttle = new PlaceThrottle(() => overlay.moving);
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
    const remeasure = () => {
      for (const l of this.labels) l.w = 0;
      this.needsPlacement = true;
      this.place();
    };
    document.fonts?.ready.then(remeasure);
    document.addEventListener("tw-fontchange", remeasure);
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
        visibility: this.visible ? "visible" : "none",
      },
      paint: { "text-opacity": 0 },
    });
  }

  set(w: WorldView, painted = false) {
    // the build waiting for the paint takes the latest world
    if (this.deferred) { this.deferred = w; return; }
    const stations = w.save.stations;
    // Most edits leave the stations as they were, or change a few: only the names those are in or
    // join are worked out again, and only their labels and shadow features change (T-063, T-1380).
    // The names are per complex, so the world's complexes (game/complexView.ts, built once for every
    // user of this world) and the player's links count (T-209); so do the draw anchor, since the
    // names' positions are in its local units (T-307), and the opening-day bunting.
    const rib = ribbonColours(w);
    const cx = complexes.peek();
    const plan = this.names.plan(stations, w.save.lines, cx, rib, anchor().version);
    if (!plan) return;
    if (PATCH_MODE === "0") plan.full = true;
    // T-1357: many names not drawn before (a game's first network) are built once the network has
    // been painted
    let fresh = 0;
    if (plan.full && !painted && stations.length - this.labels.length > DEFER_NAMES) {
      const had = new Set(this.labels.map((l) => l.el.textContent));
      for (const s of stations) if (!had.has(s.name) && ++fresh > DEFER_NAMES) break;
    }
    if (fresh > DEFER_NAMES) {
      this.deferred = w;
      afterPaint(() => { const d = this.deferred!; this.deferred = null; this.set(d, true); });
      return;
    }
    const patch = this.names.apply(plan, cx, rib);
    if (patch.full) this.rebuild();
    else if (patch.ops.length) this.patch(patch.ops);
    else return;
    this.needsPlacement = true;
    if (PATCH_MODE === "check") this.check(w, rib);
    this.place();
  }

  /** The label for a name, from a removed one with the same text and look if there is one. A label
   * whose text and look are unchanged keeps its element and measured size (T-496): a new station
   * then costs one new element, not a rebuild and a layout of every name. Shown labels never
   * overlap, so their order in the DOM draws nothing differently. */
  private labelFor(g: NameGroup, spare: Map<string, Label[]>): Label {
    const cls = "stn-label" + (g.built ? "" : " plan") + (g.park ? " park" : "") + (g.rib ? " opening" : "");
    const [x, y] = toLocal(g.lng, g.lat);
    const below = Math.abs(Math.cos(g.heading)) > Math.abs(Math.sin(g.heading));
    const old = spare.get(cls + "\u0000" + g.name + "\u0000" + g.rib)?.pop();
    if (old) return { ...old, x, y, below };
    const el = document.createElement("div");
    el.className = cls;
    if (LAYERED) el.style.willChange = "transform"; // ?perfTry=labelLayers (T-045)
    if (g.rib) el.style.setProperty("--rib", g.rib);
    el.textContent = g.name;
    this.plane.appendChild(el);
    return { el, x, y, below, w: 0, h: 0, rib: g.rib };
  }

  private keepSpare(spare: Map<string, Label[]>, l: Label) {
    const k = l.el.className + "\u0000" + l.el.textContent + "\u0000" + l.rib;
    const v = spare.get(k);
    if (v) v.push(l);
    else spare.set(k, [l]);
  }

  /** Every label and the shadow made again from the whole list of names. */
  private rebuild() {
    const names = this.names.list;
    const spare = new Map<string, Label[]>();
    for (const l of this.labels) this.keepSpare(spare, l);
    this.labels = names.map((g) => this.labelFor(g, spare));
    for (const v of spare.values()) for (const l of v) l.el.remove();
    // A stop edit changes a station's lines, mostly not the shadow: MapLibre then keeps its layout
    // instead of tiling and placing every name again (T-648). The features kept then keep their ids,
    // which are those the source holds.
    const keys = names.map((g, i) => shadowText(g, this.labels[i].below));
    const shadowKey = keys.join("\n");
    if (shadowKey !== this.shadowKey) {
      this.shadowKey = shadowKey;
      this.shadowKeys = keys;
      this.shadow = { type: "FeatureCollection", features: names.map((g, i) => shadowFeature(g, this.labels[i].below)) };
      (this.map.getSource(SHADOW) as GeoJSONSource | undefined)?.setData(this.shadow);
    }
  }

  /** The labels and shadow features followed through the names that left and entered (T-1380). */
  private patch(ops: NameOp[]) {
    const features = this.shadow.features, spare = new Map<string, Label[]>();
    type Moved = { at: number; text: string; feature: (typeof features)[number] };
    const left: Moved[] = [], entered: Moved[] = [];
    for (const op of ops) {
      if (op.kind !== "remove") continue;
      const [l] = this.labels.splice(op.at, 1);
      this.keepSpare(spare, l);
      left.push({ at: op.at, text: this.shadowKeys.splice(op.at, 1)[0], feature: features.splice(op.at, 1)[0] });
    }
    for (const op of ops) {
      if (op.kind !== "insert") continue;
      const l = this.labelFor(op.group, spare), text = shadowText(op.group, l.below), feature = shadowFeature(op.group, l.below);
      this.labels.splice(op.at, 0, l);
      this.shadowKeys.splice(op.at, 0, text);
      features.splice(op.at, 0, feature);
      entered.push({ at: op.at, text, feature });
    }
    for (const v of spare.values()) for (const l of v) l.el.remove();
    left.reverse();
    if (left.length === entered.length && left.every((x, i) => x.at === entered[i].at && x.text === entered[i].text)) {
      // the same features in the same places: the source keeps the ones it has
      entered.forEach((x, i) => { features[x.at] = left[i].feature; });
      return;
    }
    this.shadowKey = "";
    const src = this.map.getSource(SHADOW) as GeoJSONSource | undefined;
    if (!src) return;
    const refill = () => { src.setData(this.shadow); };
    try {
      src.updateData({ remove: left.map((x) => x.feature.id!), add: entered.map((x) => x.feature) }).catch(refill);
    } catch {
      refill();
    }
  }

  /** `?patch=check`: the labels and shadow features against the names worked out from nothing. */
  private check(w: WorldView, rib: ReadonlyMap<string, string> | null) {
    const want = labelGroups(w.save.stations, linesAt(w.save.lines), complexes.peek(), rib);
    const feats = this.shadow.features;
    let bad = this.labels.length !== want.length || feats.length !== want.length || this.shadowKeys.length !== want.length ? "lengths" : "";
    for (let i = 0; !bad && i < want.length; i++) {
      const g = want[i], l = this.labels[i], below = Math.abs(Math.cos(g.heading)) > Math.abs(Math.sin(g.heading));
      const f = feats[i] as unknown as { properties: { name: string }; geometry: { coordinates: number[] } };
      if (l.el.textContent !== g.name || l.rib !== g.rib || l.below !== below || l.el.className.includes("plan") === g.built || l.el.className.includes("park") !== g.park) bad = `label ${i} (${g.name})`;
      else if (f.properties.name !== g.name || f.geometry.coordinates[0] !== g.lng || f.geometry.coordinates[1] !== g.lat || this.shadowKeys[i] !== shadowText(g, below)) bad = `shadow feature ${i} (${g.name})`;
    }
    if (bad) console.error(`T-1380 check: the station names patched from changes differ from a whole build: ${bad}`);
  }

  setVisible(on: boolean) {
    this.visible = on;
    if (styleUp(this.map) && this.map.getLayer(SHADOW)) this.map.setLayoutProperty(SHADOW, "visibility", on ? "visible" : "none");
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
    // while the camera moves, a full placement at most every MOVING_PLACE_MS (T-1207); a pan between
    // them moves the names as one, past the margin if need be
    const due = this.throttle.due();
    if (PAN && (!this.needsPlacement || !due) && prev && W === this.placedSize[0] && H === this.placedSize[1]
      && m[3] === 0 && m[7] === 0
      && prev.every((v, i) => i === 12 || i === 13 || i === 14 || same(v, m[i]))) {
      const [x, y] = this.overlay.toScreen(0, 0);
      const dx = x - this.anchor[0], dy = y - this.anchor[1];
      if (!due || (Math.abs(dx) < PAN_MARGIN / 2 && Math.abs(dy) < PAN_MARGIN / 2)) {
        const transform = `translate(${dx}px, ${dy}px)`;
        if (this.plane.style.transform !== transform) this.plane.style.transform = transform;
        this.offset = [dx, dy];
        return;
      }
    }
    if (!due) return;
    this.throttle.done();
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
    const placedBoxes = new BoxGrid();
    const hide = (l: Label) => {
      if (l.hidden !== true) l.el.style.visibility = "hidden";
      l.hidden = true;
    };
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
      if (off || placedBoxes.hits(collision)) {
        hide(l);
        continue;
      }
      placedBoxes.add(collision);
      if (l.hidden !== false) l.el.style.visibility = "";
      l.hidden = false;
      const rx = Math.round(x), ry = Math.round(y);
      const transform = `translate(${rx}px, ${ry}px)`;
      if (l.transform !== transform) l.el.style.transform = l.transform = transform;
      this.placed.push([rx, ry, rx + l.w, ry + l.h]);
    }
    this.version++;
    this.onPlace?.();
  }
}

function shadowText(g: Omit<NameGroup, "uid">, below: boolean): string {
  return `${g.name}|${g.lng}|${g.lat}|${below}`;
}

function shadowFeature(g: NameGroup, below: boolean) {
  return {
    type: "Feature" as const,
    id: g.uid,
    geometry: { type: "Point" as const, coordinates: [g.lng, g.lat] },
    properties: { name: g.name, anchor: below ? "top" : "left", offset: below ? [0, 0.6] : [0.7, 0] },
  };
}
