// T-1270 stage 3: the terrain classes as a faint brown wash while a build tool is on and Terrain is
// shown, drawn from the blocks the clock worker has sent (game/terrain.ts), cell by cell as the price
// reads them, so what is shown is what is charged. One image over the view and a margin around it,
// redrawn when the camera stops: its rows are even in mercator, each looking up the cell at its
// latitude, since the grid is on a degree lattice and a degree box is no mercator quad. Flat ground is
// left bare.

import * as maplibregl from "maplibre-gl";
import type { Map as MlMap } from "maplibre-gl";
import { computed, effect } from "@preact/signals";
import { display, isBuildTool, mapShade, tool } from "../game/ui";
import { BLOCK_CELLS, mercLat, terrainGrids, terrainVersion, type ClassGrid } from "../game/terrain";
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
    // about a screen pixel to an image pixel, so with linear resampling a cell's edge is a thin soft line at any zoom
    const W = Math.max(1, Math.round(Math.min(MAX_PX, box.clientWidth * (1 + 2 * MARGIN))));
    const H = Math.max(1, Math.round(Math.min(MAX_PX, W * (y1 - y0) / (x1 - x0))));
    const s = mapShade.peek();
    const key = [x0, y0, x1, y1, W, H, s, terrainVersion.peek()].join();
    setVisible(true);
    if (key === drawn) return;
    drawn = key;

    const [red, green, blue] = WASH[s].rgb;
    const alpha = WASH[s].alpha.map((a) => Math.round(a * 255));
    const px = new Uint8ClampedArray(W * H * 4);
    // per canvas column, its world cell column and the block's key prefix
    const worldCols = 360 * BLOCK_CELLS;
    const colCell = new Int32Array(W), colBlock = new Int32Array(W);
    for (let x = 0; x < W; x++) {
      const c = Math.floor((lonW + (lonE - lonW) * (x + 0.5) / W + 180) * BLOCK_CELLS);
      const col = ((c % worldCols) + worldCols) % worldCols;
      colBlock[x] = Math.floor(col / BLOCK_CELLS);
      colCell[x] = col - colBlock[x] * BLOCK_CELLS;
    }
    let any = false;
    for (let y = 0; y < H; y++) {
      const lat = mercLat(y0 + (y1 - y0) * (y + 0.5) / H), la = Math.max(0, Math.min(179, Math.floor(lat + 90)));
      const rowBase = (Math.floor((lat + 90) * BLOCK_CELLS) - la * BLOCK_CELLS) * BLOCK_CELLS;
      let lastBlock = -1, lastGrid: ClassGrid | undefined;
      for (let x = 0; x < W; x++) {
        if (colBlock[x] !== lastBlock) lastBlock = colBlock[x], lastGrid = terrainGrids.get(la * 360 + lastBlock);
        const g = lastGrid;
        if (!g) continue;
        const i = rowBase + colCell[x];
        const k = (g.cls[i >> 2] >> ((i & 3) << 1)) & 3;
        if (k) {
          const o = (y * W + x) * 4;
          px[o] = red, px[o + 1] = green, px[o + 2] = blue, px[o + 3] = alpha[k];
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
    void shown.value, terrainVersion.value, mapShade.value;
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

