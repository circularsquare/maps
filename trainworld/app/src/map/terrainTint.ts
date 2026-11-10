// T-1270 stage 3: the terrain classes as a faint brown wash while a build tool is on and Terrain is
// shown, drawn from the blocks the clock worker has sent (game/terrain.ts), hex by hex
// as the price reads them (T-2890), so what is shown is what is charged. One image over the view and
// a margin around it, redrawn when the camera stops: its rows are even in mercator, the plane the
// hexes are regular on. Flat ground is left bare.

import * as maplibregl from "maplibre-gl";
import type { Map as MlMap } from "maplibre-gl";
import { computed, effect } from "@preact/signals";
import { display, isBuildTool, shade, tool } from "../game/ui";
import { BLOCK_CELLS, HEX_ROW, HEX_SHIFT, HEX_WIDTH, mercLat, terrainGrids, terrainVersion, type ClassGrid } from "../game/terrain";
import { firstSymbolId } from "./basemap";
import { styleUp } from "./styleReady";

const ID = "terrain-tint";
/** below this a view is wider than the view fetch reaches, and a patch of tint would show its edge (T-2901) */
const MIN_ZOOM = 9;
/** the image covers the view and this much of its size again on each side, so a pan shows it */
const MARGIN = 0.5;
/** canvas pixels at most a side */
const MAX_PX = 2048;
/** RGB and the alpha of rolling, hilly and mountain ground, in each theme */
const WASH = {
  light: { rgb: [150, 88, 30], alpha: [0, 0.08, 0.15, 0.24] },
  // a warm grey with little colour, so it reads as ground beside the parks' green (Anita picked it, T-2745)
  dark: { rgb: [190, 178, 164], alpha: [0, 0.06, 0.11, 0.18] },
};

const shown = computed(() => display.value.hillshade && isBuildTool(tool.value));

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
        paint: { "raster-resampling": "linear", "raster-fade-duration": 0, "raster-opacity": 1 },
      }, before);
    }
  };

  const setVisible = (on: boolean) => {
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
    // about a screen pixel to an image pixel, so with linear resampling a hex's edge is a smooth diagonal at any zoom (T-2902)
    const fine = Math.min(MAX_PX, box.clientWidth * (1 + 2 * MARGIN));
    // a hex under a pixel of the image has no shape to show: half the image, a quarter of the work
    const W = Math.max(1, Math.round(fine * HEX_WIDTH / (x1 - x0) < 1 ? fine / 2 : fine));
    const H = Math.max(1, Math.round(Math.min(MAX_PX, W * (y1 - y0) / (x1 - x0))));
    const s = shade.peek();
    const key = [x0, y0, x1, y1, W, H, s, terrainVersion.peek()].join();
    setVisible(true);
    if (key === drawn) return;
    drawn = key;

    const [red, green, blue] = WASH[s].rgb;
    const alpha = WASH[s].alpha.map((a) => Math.round(a * 255));
    const px = new Uint8ClampedArray(W * H * 4);
    // the hex rows the image can reach, and per row the key prefix of its block and the row inside it
    const rMin = Math.floor(y0 / HEX_ROW) - 1, rows = Math.ceil(y1 / HEX_ROW) + 1 - rMin + 1;
    const rowKey = new Int32Array(rows), rowCell = new Int32Array(rows);
    for (let j = 0; j < rows; j++) {
      const lat = mercLat((rMin + j) * HEX_ROW), la = Math.max(0, Math.min(179, Math.floor(lat + 90)));
      rowKey[j] = la * 360;
      rowCell[j] = Math.floor((lat + 90) * BLOCK_CELLS) - la * BLOCK_CELLS;
    }
    const colScale = HEX_WIDTH * 180 * BLOCK_CELLS, colShift = HEX_SHIFT * 360 * BLOCK_CELLS, worldCols = 360 * BLOCK_CELLS;
    // each pixel wears the class of its hex's centre, as track there is charged
    const xScale = (x1 - x0) / W / HEX_WIDTH, xStart = (x0 - HEX_SHIFT) / HEX_WIDTH + xScale * 0.5;
    let any = false;
    let lastKey = -1, lastGrid: ClassGrid | undefined;
    for (let y = 0; y < H; y++) {
      const rf = (y0 + (y1 - y0) * (y + 0.5) / H) / HEX_ROW;
      let lastQ = NaN, lastR = NaN, lastK = 0;
      const half = rf / 2;
      for (let x = 0; x < W; x++) {
        const qf = xStart + x * xScale - half, sf = -qf - rf;
        let q = Math.floor(qf + 0.5), r = Math.floor(rf + 0.5);
        const s = Math.floor(sf + 0.5), dq = Math.abs(q - qf), dr = Math.abs(r - rf), ds = Math.abs(s - sf);
        if (dq > dr && dq > ds) q = -r - s;
        else if (dr > ds) r = -q - s;
        let k = lastK;
        if (q !== lastQ || r !== lastR) {
          k = 0;
          const j = r - rMin, col = (((Math.floor((2 * q + r) * colScale + colShift)) % worldCols) + worldCols) % worldCols;
          const lo = Math.floor(col / BLOCK_CELLS), key = rowKey[j] + lo;
          if (key !== lastKey) lastKey = key, lastGrid = terrainGrids.get(key);
          const g = lastGrid;
          if (g) {
            const i = rowCell[j] * BLOCK_CELLS + col - lo * BLOCK_CELLS;
            k = (g.cls[i >> 2] >> ((i & 3) << 1)) & 3;
          }
          lastQ = q, lastR = r, lastK = k;
        }
        if (k) {
          const i = (y * W + x) * 4;
          px[i] = red, px[i + 1] = green, px[i + 2] = blue, px[i + 3] = alpha[k];
          any = true;
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
  // blocks come in bursts as the view reaches new land: one redraw for a burst
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

