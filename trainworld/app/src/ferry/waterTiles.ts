// The water a ferry sails on (T-2192, notes/T-2177.md 3.3): the polygons of one OpenMapTiles vector tile's
// `water` layer, and the bridges of its `transportation` layer, which cut many canals' water polygons
// (Amsterdam, Venice). Read at zoom 12, where the basemap carries the same canals as at zoom 14 at a tenth
// of the bytes. Pure: no DOM, no map.

import { decodeRoads } from "../map/roadTiles";

export const WATER_ZOOM = 12;

export interface WaterTile {
  extent: number;
  /** each polygon feature's rings, x, y pairs in tile units; filled even-odd, one feature at a time */
  polys: Int32Array[][];
  /** bridge lines (any road or railway class), x, y pairs in tile units */
  bridges: Int32Array[];
}

export const EMPTY_TILE: WaterTile = { extent: 4096, polys: [], bridges: [] };

class Reader {
  pos = 0;
  constructor(public b: Uint8Array, public end = b.length) {}
  varint(): number {
    let r = 0, mul = 1, byte: number;
    do {
      byte = this.b[this.pos++];
      r += (byte & 0x7f) * mul;
      mul *= 128;
    } while (byte & 0x80);
    return r;
  }
  len(): number {
    const n = this.varint();
    return this.pos + n;
  }
  skip(wire: number) {
    if (wire === 0) this.varint();
    else if (wire === 1) this.pos += 8;
    else if (wire === 2) this.pos = this.len();
    else if (wire === 5) this.pos += 4;
    else throw new Error("bad wire type");
  }
}

type Value = string | number | boolean | null;

function value(r: Reader, end: number): Value {
  let out: Value = null;
  while (r.pos < end) {
    const tag = r.varint(), f = tag >> 3, w = tag & 7;
    if (f === 1 && w === 2) {
      const e = r.len();
      out = new TextDecoder().decode(r.b.subarray(r.pos, e));
      r.pos = e;
    } else if (f === 2 && w === 5) {
      out = new DataView(r.b.buffer, r.b.byteOffset + r.pos, 4).getFloat32(0, true);
      r.pos += 4;
    } else if (f === 3 && w === 1) {
      out = new DataView(r.b.buffer, r.b.byteOffset + r.pos, 8).getFloat64(0, true);
      r.pos += 8;
    } else if ((f === 4 || f === 5) && w === 0) out = r.varint();
    else if (f === 6 && w === 0) {
      const v = r.varint();
      out = v % 2 ? -(v + 1) / 2 : v / 2;
    } else if (f === 7 && w === 0) out = r.varint() !== 0;
    else r.skip(w);
  }
  return out;
}

/** The water polygons a boat may use: not swimming pools, water in tunnels or intermittent water. */
function sailable(props: Map<string, Value>): boolean {
  if (props.get("class") === "swimming_pool") return false;
  if (props.get("brunnel") === "tunnel") return false;
  const im = props.get("intermittent");
  return !(im === 1 || im === true);
}

/** A tile's water polygons and bridge lines. */
export function decodeWater(buf: ArrayBuffer): WaterTile {
  const b = new Uint8Array(buf);
  const t = new Reader(b);
  const out: WaterTile = { extent: 4096, polys: [], bridges: [] };
  while (t.pos < t.end) {
    const tag = t.varint(), f = tag >> 3, w = tag & 7;
    if (f !== 3 || w !== 2) {
      t.skip(w);
      continue;
    }
    const end = t.len();
    const r = new Reader(b, end);
    r.pos = t.pos;
    t.pos = end;
    let name = "", extent = 4096;
    const feats: [number, number][] = [], keys: string[] = [], values: Value[] = [];
    while (r.pos < end) {
      const ft = r.varint(), ff = ft >> 3, fw = ft & 7;
      if (ff === 1 && fw === 2) {
        const e = r.len();
        name = new TextDecoder().decode(b.subarray(r.pos, e));
        r.pos = e;
      } else if (ff === 2 && fw === 2) {
        const e = r.len();
        feats.push([r.pos, e]);
        r.pos = e;
      } else if (ff === 3 && fw === 2) {
        const e = r.len();
        keys.push(new TextDecoder().decode(b.subarray(r.pos, e)));
        r.pos = e;
      } else if (ff === 4 && fw === 2) {
        const e = r.len();
        values.push(value(r, e));
        r.pos = e;
      } else if (ff === 5 && fw === 0) extent = r.varint();
      else r.skip(fw);
    }
    if (name !== "water") continue;
    out.extent = extent;
    for (const [s, e] of feats) {
      const fr = new Reader(b, e);
      fr.pos = s;
      let type = 0, g0 = -1, g1 = -1;
      const props = new Map<string, Value>();
      while (fr.pos < e) {
        const ft = fr.varint(), ff = ft >> 3, fw = ft & 7;
        if (ff === 2 && fw === 2) {
          const te = fr.len();
          while (fr.pos < te) {
            const k = fr.varint(), v = fr.varint();
            if (keys[k] !== undefined) props.set(keys[k], values[v] ?? null);
          }
        } else if (ff === 3 && fw === 0) type = fr.varint();
        else if (ff === 4 && fw === 2) {
          g1 = fr.len();
          g0 = fr.pos;
          fr.pos = g1;
        } else fr.skip(fw);
      }
      if (type !== 3 || g0 < 0 || !sailable(props)) continue;
      const g = new Reader(b, g1);
      g.pos = g0;
      const rings: Int32Array[] = [];
      let x = 0, y = 0, cur: number[] = [];
      while (g.pos < g1) {
        const cmd = g.varint(), id = cmd & 7;
        let count = cmd >> 3;
        if (id === 7) {
          if (cur.length >= 6) rings.push(Int32Array.from(cur));
          cur = [];
          continue;
        }
        while (count--) {
          const dx = g.varint(), dy = g.varint();
          x += (dx >> 1) ^ -(dx & 1);
          y += (dy >> 1) ^ -(dy & 1);
          if (id === 1) {
            if (cur.length >= 6) rings.push(Int32Array.from(cur));
            cur = [x, y];
          } else cur.push(x, y);
        }
      }
      if (cur.length >= 6) rings.push(Int32Array.from(cur));
      if (rings.length) out.polys.push(rings);
    }
  }
  out.bridges = decodeRoads(buf, (_c, brunnel) => brunnel === "bridge").lines;
  return out;
}
