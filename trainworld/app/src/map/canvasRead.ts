// T-3705: Firefox with resistFingerprinting on (or its fingerprinting protection set to block canvas reads, as
// LibreWolf and the Mullvad and Tor browsers have it) makes up the pixels of every canvas read: getImageData,
// toBlob, toDataURL, convertToBlob. It hides VideoFrame too, MapLibre's way round it. A height tile read through a
// canvas was then noise, drawn as vertical bars. Where reads are made up, PNGs are decoded and encoded here
// without a canvas, and what can only come from a canvas read is left out (map/basemap.ts `demReadable`).

let readsTrue: boolean | undefined;

/** Whether a canvas hands back the pixels drawn into it (false: the browser makes them up). Asked once a thread:
 * the page and each worker that reads pixels. */
export function canvasReadsTrue(): boolean {
  if (readsTrue !== undefined) return readsTrue;
  readsTrue = true;
  try {
    if (typeof OffscreenCanvas === "undefined") return readsTrue;
    const c = new OffscreenCanvas(5, 5).getContext("2d", { willReadFrequently: true });
    if (!c) return readsTrue;
    const img = new ImageData(5, 5);
    for (let i = 0; i < 25; i++) img.data.set([i * 4, i * 4 + 1, i * 4 + 2, 255], i * 4);
    c.putImageData(img, 0, 0);
    const d = c.getImageData(0, 0, 5, 5).data;
    for (let i = 0; i < 100; i++) if (i % 4 !== 3 && d[i] !== i) { readsTrue = false; break; }
  } catch { /* no 2D canvas here: nothing reads one */ }
  return readsTrue;
}

const SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10];

async function zlib(z: Uint8Array<ArrayBuffer>, how: "decompress" | "compress"): Promise<Uint8Array> {
  const stream = new Blob([z]).stream().pipeThrough(how === "decompress" ? new DecompressionStream("deflate") : new CompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** An 8-bit, not interlaced PNG (greyscale, RGB, palette, with or without alpha) as RGBA, read without a canvas.
 * Throws on anything else. */
export async function decodePng(bytes: ArrayBuffer): Promise<{ width: number; height: number; rgba: Uint8ClampedArray }> {
  const b = new Uint8Array(bytes), v = new DataView(bytes);
  if (b.length < 8 || SIGNATURE.some((s, i) => b[i] !== s)) throw new Error("not a PNG");
  let p = 8, width = 0, height = 0, depth = 0, type = 0, interlace = 0, size = 0;
  let palette: Uint8Array | null = null, trns: Uint8Array | null = null;
  const idat: Uint8Array[] = [];
  while (p + 8 <= b.length) {
    const len = v.getUint32(p), kind = String.fromCharCode(b[p + 4], b[p + 5], b[p + 6], b[p + 7]), body = b.subarray(p + 8, p + 8 + len);
    if (kind === "IHDR") [width, height, depth, type, interlace] = [v.getUint32(p + 8), v.getUint32(p + 12), b[p + 16], b[p + 17], b[p + 20]];
    else if (kind === "PLTE") palette = body;
    else if (kind === "tRNS") trns = body;
    else if (kind === "IDAT") { idat.push(body); size += len; }
    else if (kind === "IEND") break;
    p += 12 + len;
  }
  const channels = ({ 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 } as Record<number, number>)[type];
  if (!width || !height || depth !== 8 || interlace !== 0 || !channels || (type === 3 && !palette)) throw new Error(`PNG not read here: depth ${depth}, type ${type}, interlace ${interlace}`);
  const z = new Uint8Array(size);
  for (let i = 0, at = 0; i < idat.length; at += idat[i].length, i++) z.set(idat[i], at);
  const raw = await zlib(z, "decompress");
  const stride = width * channels;
  if (raw.length < (stride + 1) * height) throw new Error("PNG data short");
  const out = new Uint8ClampedArray(width * height * 4);
  let prev = new Uint8Array(stride), cur = new Uint8Array(stride);
  for (let y = 0; y < height; y++) {
    const at = y * (stride + 1), filter = raw[at];
    for (let x = 0; x < stride; x++) {
      const a = x >= channels ? cur[x - channels] : 0, up = prev[x], c = x >= channels ? prev[x - channels] : 0;
      let pred = 0;
      if (filter === 1) pred = a;
      else if (filter === 2) pred = up;
      else if (filter === 3) pred = (a + up) >> 1;
      else if (filter === 4) {
        const q = a + up - c, pa = Math.abs(q - a), pb = Math.abs(q - up), pc = Math.abs(q - c);
        pred = pa <= pb && pa <= pc ? a : pb <= pc ? up : c;
      }
      cur[x] = (raw[at + 1 + x] + pred) & 255;
    }
    for (let x = 0, o = y * width * 4; x < width; x++, o += 4) {
      const s = x * channels;
      if (type === 6) { out[o] = cur[s]; out[o + 1] = cur[s + 1]; out[o + 2] = cur[s + 2]; out[o + 3] = cur[s + 3]; }
      else if (type === 2) { out[o] = cur[s]; out[o + 1] = cur[s + 1]; out[o + 2] = cur[s + 2]; out[o + 3] = 255; }
      else if (type === 3) { const k = cur[s] * 3; out[o] = palette![k]; out[o + 1] = palette![k + 1]; out[o + 2] = palette![k + 2]; out[o + 3] = trns && cur[s] < trns.length ? trns[cur[s]] : 255; }
      else { out[o] = out[o + 1] = out[o + 2] = cur[s]; out[o + 3] = type === 4 ? cur[s + 1] : 255; }
    }
    [prev, cur] = [cur, prev];
  }
  return { width, height, rgba: out };
}

let crcTable: Uint32Array | null = null;
function crc32(b: Uint8Array): number {
  if (!crcTable) {
    crcTable = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      crcTable[n] = c >>> 0;
    }
  }
  let c = 0xffffffff;
  for (let i = 0; i < b.length; i++) c = crcTable[(c ^ b[i]) & 255] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(kind: string, body: Uint8Array): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(12 + body.length), v = new DataView(out.buffer);
  v.setUint32(0, body.length);
  for (let i = 0; i < 4; i++) out[4 + i] = kind.charCodeAt(i);
  out.set(body, 8);
  v.setUint32(8 + body.length, crc32(out.subarray(4, 8 + body.length)));
  return out;
}

/** RGBA pixels (not premultiplied) as a PNG, made without a canvas. */
export async function encodePng(rgba: Uint8ClampedArray, width: number, height: number): Promise<Blob> {
  const stride = width * 4, raw = new Uint8Array((stride + 1) * height);
  for (let y = 0; y < height; y++) raw.set(rgba.subarray(y * stride, (y + 1) * stride), y * (stride + 1) + 1);
  const head = new Uint8Array(13), hv = new DataView(head.buffer);
  hv.setUint32(0, width);
  hv.setUint32(4, height);
  head.set([8, 6, 0, 0, 0], 8);
  const z = await zlib(raw, "compress");
  return new Blob([new Uint8Array(SIGNATURE), chunk("IHDR", head), chunk("IDAT", z), chunk("IEND", new Uint8Array(0))], { type: "image/png" });
}
