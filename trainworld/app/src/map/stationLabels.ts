// Station names as DOM labels above the overlay canvas, in the UI font (Zen Maru Gothic). A
// MapLibre symbol layer would sit under the overlay, with track drawn through the names. Labels
// move on MapLibre's frames only (the camera does not change otherwise). A flat pan translates
// one compositor layer; other camera changes re-place names. A greedy pass hides any label that
// overlaps a more important one (transfers first, then lines, then constructed).
//
// Winning over basemap place names (T-032): an invisible symbol layer on top of the basemap holds
// the same names at the same spots. MapLibre places symbols from the top layer down, so basemap
// labels that would collide with a station name are dropped by MapLibre itself; the shadow is
// never drawn (opacity 0) and is only rebuilt when stations change.

import type { FeatureCollection } from "geojson";
import type { GeoJSONSource, Map as MlMap } from "maplibre-gl";
import type { WorldView } from "../game/types";
import { toLocal } from "./geo";
import type { Overlay } from "./overlay";
import { perfOff, perfTry } from "../perfFlags";
import { MIN_LABEL_ZOOM, labelPadding } from "./labelLayout";

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
    const lines = new Map<string, number>();
    for (const l of w.save.lines) for (const s of new Set(l.stops)) lines.set(s, (lines.get(s) ?? 0) + 1);
    const st = [...w.save.stations].sort((a, b) => (lines.get(b.id) ?? 0) - (lines.get(a.id) ?? 0) || Number(b.built) - Number(a.built));
    // Most edits leave the stations as they were: keep the labels and the shadow layer then
    // (rebuilding 470 labels and the shadow source took ~100 ms an edit on a 2,000 km network, T-063).
    const key = st.map((s) => `${s.id}|${s.name}|${s.built}|${s.lng}|${s.lat}|${s.heading}|${lines.get(s.id) ?? 0}`).join("\n");
    if (key === this.key) return;
    this.key = key;
    this.plane.textContent = "";
    this.needsPlacement = true;
    this.labels = st.map((s) => {
      const el = document.createElement("div");
      el.className = "stn-label" + (s.built ? "" : " plan");
      if (LAYERED) el.style.willChange = "transform"; // ?perfTry=labelLayers (T-045)
      el.textContent = s.name;
      this.plane.appendChild(el);
      const [x, y] = toLocal(s.lng, s.lat);
      return { el, x, y, below: Math.abs(Math.cos(s.heading)) > Math.abs(Math.sin(s.heading)), w: 0, h: 0 };
    });
    this.shadow = {
      type: "FeatureCollection",
      features: st.map((s, i) => ({
        type: "Feature",
        geometry: { type: "Point", coordinates: [s.lng, s.lat] },
        properties: { name: s.name, anchor: this.labels[i].below ? "top" : "left", offset: this.labels[i].below ? [0, 0.6] : [0.7, 0] },
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

  place() {
    const zoom = this.map.getZoom();
    const show = this.visible && zoom >= MIN_LABEL_ZOOM && !NO_LABELS;
    if (show !== this.shown) {
      this.shown = show;
      this.box.style.display = show ? "" : "none";
      this.needsPlacement = true;
    }
    if (!show || !this.overlay.camMatrix) return;
    const W = this.overlay.canvas.clientWidth, H = this.overlay.canvas.clientHeight;
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
        return;
      }
    }
    this.needsPlacement = false;
    this.placedMatrix = Array.from(m);
    this.placedSize = [W, H];
    this.anchor = this.overlay.toScreen(0, 0);
    this.plane.style.transform = "";
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
    for (const l of this.labels) {
      const [sx, sy] = this.overlay.toScreen(l.x, l.y);
      const x = l.below ? sx - l.w / 2 : sx + GAP;
      const y = l.below ? sy + GAP - 2 : sy - l.h / 2;
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
      l.el.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
    }
  }
}
