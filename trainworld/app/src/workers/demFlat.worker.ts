// T-1881: a Terrain Tiles height tile with its relief taken out where the track is priced flat
// (map/demFlat.ts). The public tiles are a surface model in much of the world, so towers, raised
// roads and SRTM's speckle read as hills; where the terrain class is flat the heights are blurred
// (BLUR_M), so only the lie of the land over kilometres is left, blended back to the tile's own heights
// by `mask`. The 3 x 3 tiles round it give the blur its reach, so neighbouring tiles agree at their edge.

/** `bytes`: the PNGs of the tile and its 8 neighbours, row by row from the north-west (null: none,
 * the tile's own edge stands in); `mask`: 0-255 a pixel, how flat (255 all). */
export type DemFlatIn = { id: number; keys: string[]; bytes: (ArrayBuffer | null)[]; mask: Uint8Array; pxM: number };
export type DemFlatOut = { id: number; bitmap?: ImageBitmap; error?: string };

const N = 256;
/** the blur's sigma, m: a tower block or a raised road, a few hundred metres across, is spread to
 * a few metres over a kilometre or two */
const BLUR_M = 400;

const decoded = new Map<string, Float32Array>();
const DECODED_MAX = 96;

async function heights(key: string, bytes: ArrayBuffer): Promise<Float32Array> {
  const hit = decoded.get(key);
  if (hit) {
    decoded.delete(key);
    decoded.set(key, hit);
    return hit;
  }
  const bmp = await createImageBitmap(new Blob([bytes]), { premultiplyAlpha: "none", colorSpaceConversion: "none" });
  const cv = new OffscreenCanvas(N, N);
  const cx = cv.getContext("2d", { willReadFrequently: true })!;
  cx.drawImage(bmp, 0, 0);
  bmp.close();
  const px = cx.getImageData(0, 0, N, N).data;
  const h = new Float32Array(N * N);
  for (let i = 0; i < N * N; i++) h[i] = px[4 * i] * 256 + px[4 * i + 1] + px[4 * i + 2] / 256 - 32768;
  decoded.set(key, h);
  if (decoded.size > DECODED_MAX) decoded.delete(decoded.keys().next().value!);
  return h;
}

/** A mean of `2r + 1` along rows or columns, in place, the line's end values standing beyond it. */
function boxPass(a: Float32Array, width: number, height: number, r: number, alongRows: boolean) {
  const len = alongRows ? width : height, lines = alongRows ? height : width;
  const step = alongRows ? 1 : width, lineStep = alongRows ? width : 1;
  const line = new Float32Array(len), w = 2 * r + 1;
  for (let l = 0; l < lines; l++) {
    const o = l * lineStep;
    for (let i = 0; i < len; i++) line[i] = a[o + i * step];
    const at = (i: number) => line[i < 0 ? 0 : i >= len ? len - 1 : i];
    let s = 0;
    for (let i = -r; i <= r; i++) s += at(i);
    for (let i = 0; i < len; i++) {
      a[o + i * step] = s / w;
      s += at(i + r + 1) - at(i - r);
    }
  }
}

/** Near enough a Gaussian of `sigma` pixels: three box means each way. */
function blur(a: Float32Array, width: number, height: number, sigma: number) {
  const r = Math.round((Math.sqrt(4 * sigma * sigma + 1) - 1) / 2);
  if (r < 1) return;
  for (let k = 0; k < 3; k++) {
    boxPass(a, width, height, r, true);
    boxPass(a, width, height, r, false);
  }
}

async function flatten(m: DemFlatIn): Promise<ImageBitmap> {
  const tiles = await Promise.all(m.bytes.map((b, i) => (b ? heights(m.keys[i], b) : null)));
  const own = tiles[4]!;
  const sigma = BLUR_M / m.pxM;
  const M = Math.min(N, Math.ceil(3 * sigma) + 1);
  const W = N + 2 * M;
  // the tile with M pixels of its neighbours round it (its own edge where a neighbour is missing); the
  // sea floor counts as sea level, so a shore is not dragged down to it
  const big = new Float32Array(W * W);
  for (let y = 0; y < W; y++)
    for (let x = 0; x < W; x++) {
      const gx = x - M + N, gy = y - M + N; // in the 3 x 3 mosaic
      const tx = Math.floor(gx / N), ty = Math.floor(gy / N);
      const t = tiles[ty * 3 + tx];
      big[y * W + x] = Math.max(0, t ? t[(gy - ty * N) * N + (gx - tx * N)] : own[Math.min(N - 1, Math.max(0, gy - N)) * N + Math.min(N - 1, Math.max(0, gx - N))]);
    }
  blur(big, W, W, sigma);
  const px = new Uint8ClampedArray(N * N * 4);
  for (let y = 0; y < N; y++)
    for (let x = 0; x < N; x++) {
      const i = y * N + x, f = m.mask[i] / 255;
      const raw = own[i];
      const h = f > 0 ? f * big[(y + M) * W + x + M] + (1 - f) * raw : raw;
      const v = Math.max(0, h + 32768);
      px[4 * i] = Math.floor(v / 256);
      px[4 * i + 1] = Math.floor(v) % 256;
      px[4 * i + 2] = Math.floor((v - Math.floor(v)) * 256);
      px[4 * i + 3] = 255;
    }
  return createImageBitmap(new ImageData(px, N, N), { premultiplyAlpha: "none", colorSpaceConversion: "none" });
}

self.onmessage = async (e: MessageEvent<DemFlatIn>) => {
  const w = self as unknown as Worker;
  try {
    const bitmap = await flatten(e.data);
    w.postMessage({ id: e.data.id, bitmap } satisfies DemFlatOut, [bitmap]);
  } catch (err) {
    w.postMessage({ id: e.data.id, error: String(err) } satisfies DemFlatOut);
  }
};
