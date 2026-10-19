// The real network as a routing graph, one file per noritetsu region (T-727, notes/T-727.md): the
// track the real-network layer draws, as chains of vertices between junctions. Written by
// pipeline/realgraph.py (its docstring has the layout) into data/packs/realgraph/, served beside
// the packs and copied by a production build (vite.config.ts). The track tool's router reads it.

export interface RealGraph {
  cc: string;
  /** junctions, degrees */
  jLon: Float64Array;
  jLat: Float64Array;
  /** each chain's junctions: `a` at its first vertex, `b` at its last */
  a: Uint32Array;
  b: Uint32Array;
  /** 0 rail, 2 subway */
  kind: Uint8Array;
  /** 0 main line, 1 other */
  rank: Uint8Array;
  /** 1 passenger route, 2 bridge, 4 tunnel, 8 high speed */
  flags: Uint8Array;
  /** chain c's vertices are `off[c]` to `off[c + 1] - 1` */
  off: Uint32Array;
  lon: Float64Array;
  lat: Float64Array;
  /** junctions shared with a neighbouring region's file, and their OSM node ids */
  shared: Uint32Array;
  sharedId: Float64Array;
}

const loaded = new Map<string, Promise<RealGraph | null>>();

/** A region's graph, fetched once; null if there is none. `base`: the folder of the files (a worker
 * has no document to find it from). */
export function realGraph(cc: string, base?: string): Promise<RealGraph | null> {
  let p = loaded.get(cc);
  if (!p) {
    p = (async () => {
      const r = await fetch(new URL(`${cc}.rgr`, base ?? new URL("packs/realgraph/", document.baseURI)));
      if (!r.ok || !r.body) return null;
      const raw = await new Response(r.body.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
      return decodeRealGraph(cc, raw);
    })().catch(() => null);
    loaded.set(cc, p);
  }
  return p;
}

export function decodeRealGraph(cc: string, raw: ArrayBuffer): RealGraph {
  const v = new DataView(raw);
  if (new TextDecoder().decode(new Uint8Array(raw, 0, 4)) !== "TWRG" || v.getUint32(4, true) !== 1) throw new Error(`${cc}: not a real-network graph`);
  const [J, C, V, S] = [8, 12, 16, 20].map((o) => v.getUint32(o, true));
  let o = 24;
  // the arrays are not 4-byte aligned after the u8 ones, so each is copied out
  const u32 = (n: number) => { const a = new Uint32Array(n); for (let i = 0; i < n; i++, o += 4) a[i] = v.getUint32(o, true); return a; };
  const deg = (n: number) => { const a = new Float64Array(n); for (let i = 0; i < n; i++, o += 4) a[i] = v.getInt32(o, true) / 1e6; return a; };
  const u8 = (n: number) => { const a = new Uint8Array(raw.slice(o, o + n)); o += n; return a; };
  const jLon = deg(J), jLat = deg(J);
  const a = u32(C), b = u32(C);
  const kind = u8(C), rank = u8(C), flags = u8(C);
  const off = u32(C + 1);
  const bytes = new Uint8Array(raw);
  const lon = new Float64Array(V), lat = new Float64Array(V);
  let x = 0, y = 0, c = 0;
  const varint = () => {
    let r = 0, s = 0, k: number;
    do {
      k = bytes[o++];
      r += (k & 0x7f) * 2 ** s;
      s += 7;
    } while (k & 0x80);
    return r % 2 ? -(r + 1) / 2 : r / 2;
  };
  for (let i = 0; i < V; i++) {
    const dx = varint(), dy = varint();
    if (i === off[c]) (x = dx), (y = dy), c++;
    else (x += dx), (y += dy);
    lon[i] = x / 1e6;
    lat[i] = y / 1e6;
  }
  const shared = u32(S);
  const sharedId = new Float64Array(S);
  for (let i = 0; i < S; i++, o += 8) sharedId[i] = v.getFloat64(o, true);
  if (o !== raw.byteLength) throw new Error(`${cc}: ${raw.byteLength - o} bytes left over`);
  return { cc, jLon, jLat, a, b, kind, rank, flags, off, lon, lat, shared, sharedId };
}
