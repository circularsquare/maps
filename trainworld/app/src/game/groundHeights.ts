// T-2734: ground heights for the line cross-section, read from the hillshade's Terrain Tiles
// (Terrarium PNGs, map/basemap.ts), each tile decoded once and kept.

import { signal } from "@preact/signals";
import { demBytes } from "../map/demFlat";
import { terrainUrl } from "../map/basemap";

/** the finest zoom read; finer tiles outside the US are this one's data upsampled */
export const GROUND_MAX_Z = 12;
const KEEP = 96;
const N = 256;

const decoded = new Map<string, Float32Array | null>();
const pending = new Map<string, Promise<void>>();
/** bumped as tiles land, for drawings reading `groundAt` */
export const groundVersion = signal(0);

/** T-2734: the cross section is flavour, so tiles are decoded when the page is idle */
const idle = () => new Promise<void>((r) => (typeof requestIdleCallback === "function" ? requestIdleCallback(() => r(), { timeout: 2000 }) : setTimeout(r, 50)));

async function decode(bytes: ArrayBuffer): Promise<Float32Array> {
  await idle();
  const bmp = await createImageBitmap(new Blob([bytes], { type: "image/png" }), { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const c = new OffscreenCanvas(N, N).getContext("2d", { willReadFrequently: true })!;
  c.drawImage(bmp, 0, 0);
  bmp.close();
  const px = c.getImageData(0, 0, N, N).data, out = new Float32Array(N * N);
  // Terrarium: R*256 + G + B/256 - 32768, the sea floor below 0
  for (let i = 0; i < N * N; i++) out[i] = px[i * 4] * 256 + px[i * 4 + 1] + px[i * 4 + 2] / 256 - 32768;
  return out;
}

function load(z: number, x: number, y: number): Promise<void> {
  const key = `${z}/${x}/${y}`;
  if (decoded.has(key)) return Promise.resolve();
  let p = pending.get(key);
  if (!p) {
    p = demBytes(terrainUrl, z, x, y)
      .then((b) => (b ? decode(b) : null))
      .catch(() => null)
      .then((h) => {
        decoded.set(key, h);
        if (decoded.size > KEEP) decoded.delete(decoded.keys().next().value!);
        pending.delete(key);
        groundVersion.value++;
      });
    pending.set(key, p);
  }
  return p;
}

/** The tile pixel (fractional, from the north-west) of a place at zoom `z`. */
function pixel(z: number, lng: number, lat: number): [number, number] {
  const n = 2 ** z * N, r = (Math.max(-85, Math.min(85, lat)) * Math.PI) / 180;
  return [((lng + 180) / 360) * n, ((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * n];
}

function readPx(z: number, gx: number, gy: number): number | undefined {
  const n = 2 ** z;
  const tx = (((Math.floor(gx / N) % n) + n) % n), ty = Math.floor(gy / N);
  const t = decoded.get(`${z}/${tx}/${ty}`);
  if (!t) return undefined;
  return t[(((gy % N) + N) % N) * N + (((gx % N) + N) % N)];
}

/** The ground height at a place from the tiles already decoded at zoom `z` (bilinear), or undefined
 * where none is; asks for the missing tile. */
export function groundAt(z: number, lng: number, lat: number): number | undefined {
  const [fx, fy] = pixel(z, lng, lat);
  const x0 = Math.floor(fx - 0.5), y0 = Math.floor(fy - 0.5), tx = fx - 0.5 - x0, ty = fy - 0.5 - y0;
  const a = readPx(z, x0, y0), b = readPx(z, x0 + 1, y0), c = readPx(z, x0, y0 + 1), d = readPx(z, x0 + 1, y0 + 1);
  if (a === undefined || b === undefined || c === undefined || d === undefined) {
    const n = 2 ** z;
    for (const [gx, gy] of [[x0, y0], [x0 + 1, y0 + 1]]) {
      const ty2 = Math.floor(gy / N);
      if (ty2 >= 0 && ty2 < n) void load(z, ((Math.floor(gx / N) % n) + n) % n, ty2);
    }
    return a ?? b ?? c ?? d;
  }
  return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
}

/** The zoom whose pixels are about `metres` across at `lat`, at most GROUND_MAX_Z. */
export function zoomFor(metres: number, lat: number): number {
  const z0 = Math.log2((40075016.686 * Math.cos((lat * Math.PI) / 180)) / N / Math.max(1, metres));
  return Math.max(3, Math.min(GROUND_MAX_Z, Math.round(z0)));
}

/** Loads every tile under the points (lng, lat pairs) at zoom `z`. */
export function loadUnder(z: number, lngLat: ArrayLike<number>): Promise<void> {
  const want = new Set<string>(), all: Promise<void>[] = [];
  const n = 2 ** z;
  for (let i = 0; i + 1 < lngLat.length; i += 2) {
    const [fx, fy] = pixel(z, lngLat[i], lngLat[i + 1]);
    const tx = ((Math.floor(fx / N) % n) + n) % n, ty = Math.floor(fy / N);
    const key = `${tx}/${ty}`;
    if (want.has(key) || ty < 0 || ty >= n) continue;
    want.add(key);
    all.push(load(z, tx, ty));
  }
  return Promise.all(all).then(() => {});
}
