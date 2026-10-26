// T-1270 stage 3: the terrain classes as a faint brown wash while a build tool is on and Terrain is
// shown, drawn from the grids the clock worker put in the track model (game/terrain.ts), so what is
// shown is what is charged. One image over the view and a margin around it, redrawn when the camera
// stops: its rows are even in mercator, each looking up the class at its latitude, since the grids
// are on a degree lattice and a degree box is no mercator quad. Flat ground is left bare.

import * as maplibregl from "maplibre-gl";
import type { Map as MlMap } from "maplibre-gl";
import { computed, effect } from "@preact/signals";
import { display, isBuildTool, shade, tool } from "../game/ui";
import { terrainGrids, terrainText, terrainVersion, type ClassGrid } from "../game/terrain";
import { firstSymbolId } from "./basemap";
import { styleUp } from "./styleReady";

const ID = "terrain-tint";
/** below this the classes' cells are a few pixels and the wash only muddies the map */
const MIN_ZOOM = 7;
/** the image covers the view and this much of its size again on each side, so a pan shows it */
const MARGIN = 0.5;
/** canvas pixels at most a side, and at least this many to a lattice cell's width */
const MAX_PX = 2048;
const PX_PER_CELL = 12;
/** RGB and the alpha of rolling, hilly and mountain ground, in each theme */
const WASH = {
  light: { rgb: [150, 88, 30], alpha: [0, 0.08, 0.15, 0.24] },
  dark: { rgb: [230, 160, 90], alpha: [0, 0.06, 0.11, 0.18] },
};

const shown = computed(() => display.value.hillshade && isBuildTool(tool.value));
let visibleNow = false;

/** The class at a point from the grids in hand (first that covers it, as the sim's `Terrain`), or
 * -1 where none does. */
function classAt(lon: number, lat: number): number {
  for (const g of terrainGrids.values()) {
    const c = Math.floor((lon + 180) * g.perDeg) - g.col0, r = Math.floor((lat + 90) * g.perDeg) - g.row0;
    if (c >= 0 && r >= 0 && c < g.w && r < g.h) return g.cls[r * g.w + c];
  }
  return -1;
}

/** The hover over washed ground while the wash is shown: its class and what it does to the price. */
export function terrainTip(map: MlMap, px: number, py: number): string[] | null {
  if (!visibleNow || map.getZoom() < MIN_ZOOM) return null;
  const at = map.unproject([px, py]);
  const c = classAt(at.lng, at.lat);
  return c > 0 ? [terrainText(c)] : null;
}

export function startTerrainTint(map: MlMap): () => void {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 1;
  const blank = canvas.toDataURL();
  let url: string | null = null;
  let drawn = "";

  const add = () => {
    if (!styleUp(map)) return;
    ensure();
    drawn = "";
    redraw();
  };
  const ensure = () => {
    if (!map.getSource(ID)) map.addSource(ID, { type: "image", url: blank, coordinates: [[-1, 1], [1, 1], [1, -1], [-1, -1]] });
    if (!map.getLayer(ID)) {
      // over the land cover and the hillshade, under the water (never charged) and the roads and labels
      const before = ["water_over_shade", "waterway"].find((id) => map.getLayer(id)) ?? firstSymbolId(map);
      map.addLayer({
        id: ID, type: "raster", source: ID, layout: { visibility: "none" },
        paint: { "raster-resampling": "nearest", "raster-fade-duration": 0, "raster-opacity": 1 },
      }, before);
    }
  };

  const setVisible = (on: boolean) => {
    visibleNow = on;
    if (map.getLayer(ID) && (map.getLayoutProperty(ID, "visibility") === "visible") !== on) map.setLayoutProperty(ID, "visibility", on ? "visible" : "none");
  };

  const redraw = () => {
    if (!styleUp(map)) return;
    ensure();
    if (!shown.peek() || map.getZoom() < MIN_ZOOM || !terrainGrids.size) return setVisible(false);
    const box = map.getContainer();
    const b = map.getBounds();
    const nw = maplibregl.MercatorCoordinate.fromLngLat([b.getWest(), Math.min(85, b.getNorth())]);
    const se = maplibregl.MercatorCoordinate.fromLngLat([b.getEast(), Math.max(-85, b.getSouth())]);
    const dx = (se.x - nw.x) * MARGIN, dy = (se.y - nw.y) * MARGIN;
    const top = maplibregl.MercatorCoordinate.fromLngLat([0, 85]).y, bottom = maplibregl.MercatorCoordinate.fromLngLat([0, -85]).y;
    const x0 = nw.x - dx, x1 = se.x + dx, y0 = Math.max(top, nw.y - dy), y1 = Math.min(bottom, se.y + dy);
    const lonW = x0 * 360 - 180, lonE = x1 * 360 - 180;
    const perDeg = terrainGrids.values().next().value!.perDeg;
    const cssW = box.clientWidth * (1 + 2 * MARGIN), cssH = box.clientHeight * (1 + 2 * MARGIN);
    const W = Math.max(1, Math.round(Math.min(MAX_PX, cssW / 2, (lonE - lonW) * perDeg * PX_PER_CELL)));
    const H = Math.max(1, Math.round(Math.min(MAX_PX, W * (y1 - y0) / (x1 - x0))));
    const s = shade.peek();
    const key = [x0, y0, x1, y1, W, H, s, terrainVersion.peek()].join();
    setVisible(true);
    if (key === drawn) return;
    drawn = key;

    const wash = WASH[s];
    const px = new Uint8ClampedArray(W * H * 4);
    const grids = [...terrainGrids.values()];
    // per grid, the lattice column of each canvas column (-1 outside it)
    const cols = grids.map((g) => {
      const out = new Int32Array(W);
      for (let x = 0; x < W; x++) {
        const c = Math.floor((lonW + (lonE - lonW) * (x + 0.5) / W + 180) * g.perDeg) - g.col0;
        out[x] = c >= 0 && c < g.w ? c : -1;
      }
      return out;
    });
    const rows: { g: ClassGrid; cols: Int32Array; base: number }[] = [];
    let any = false;
    for (let y = 0; y < H; y++) {
      const my = y0 + (y1 - y0) * (y + 0.5) / H;
      const lat = mercLat(my);
      rows.length = 0;
      grids.forEach((g, k) => {
        const r = Math.floor((lat + 90) * g.perDeg) - g.row0;
        if (r >= 0 && r < g.h) rows.push({ g, cols: cols[k], base: r * g.w });
      });
      if (!rows.length) continue;
      for (let x = 0; x < W; x++) {
        for (const { g, cols: cc, base } of rows) {
          const c = cc[x];
          if (c < 0) continue;
          const k = g.cls[base + c];
          if (k) {
            const i = (y * W + x) * 4;
            px[i] = wash.rgb[0], px[i + 1] = wash.rgb[1], px[i + 2] = wash.rgb[2], px[i + 3] = Math.round(wash.alpha[k] * 255);
            any = true;
          }
          break;
        }
      }
    }
    canvas.width = W, canvas.height = H;
    if (any) canvas.getContext("2d")!.putImageData(new ImageData(px, W, H), 0, 0);
    const coordinates: [[number, number], [number, number], [number, number], [number, number]] = [
      [lonW, mercLat(y0)], [lonE, mercLat(y0)], [lonE, mercLat(y1)], [lonW, mercLat(y1)],
    ];
    canvas.toBlob((blob) => {
      if (!blob || drawn !== key) return;
      const src = map.getSource(ID) as maplibregl.ImageSource | undefined;
      if (!src) return;
      const old = url;
      url = URL.createObjectURL(blob);
      src.updateImage({ url, coordinates });
      // an image still loading when the next is asked for keeps its URL a while
      if (old) window.setTimeout(() => URL.revokeObjectURL(old), 10_000);
    });
  };

  map.on("style.load", add);
  map.on("moveend", redraw);
  if (styleUp(map)) add();
  // grids come in bursts as the view reaches new zones: one redraw for a burst
  let soon = 0;
  const dispose = effect(() => {
    void shown.value, terrainVersion.value, shade.value;
    window.clearTimeout(soon);
    soon = window.setTimeout(redraw, shown.peek() ? 50 : 0);
  });
  return () => {
    dispose();
    window.clearTimeout(soon);
    map.off("style.load", add);
    map.off("moveend", redraw);
  };
}

const mercLat = (y: number) => (Math.atan(Math.sinh(Math.PI * (1 - 2 * y))) * 180) / Math.PI;
