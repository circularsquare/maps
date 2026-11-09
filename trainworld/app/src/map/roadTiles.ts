// The road lines of one OpenMapTiles vector tile (T-1465): a small reader for the protobuf, since the
// game needs only the `transportation` layer's lines and the map library keeps its own reader to
// itself. Pure: no DOM, no map.

/** The road classes a bus may drive (OpenMapTiles `transportation.class`); motorways, service ways,
 * tracks, paths, rail and ferries are left out (notes/T-1445.md). */
export const BUS_CLASSES = new Set(["trunk", "primary", "secondary", "tertiary", "minor", "busway"]);

export interface RoadLines {
  extent: number;
  /** each line's x, y pairs in tile units, 0..extent across the tile (a little beyond at its edge) */
  lines: Int32Array[];
  /** each line's weight, what `keep` gave it (a plain true is 1) */
  weights?: number[];
  /** each line's level: 0 on the ground, anything else on a bridge, in a tunnel or on an OSM layer */
  levels?: number[];
}

/** A road's `class` and `brunnel` (bridge, tunnel, ford or none) in OpenMapTiles. */
export type RoadKeep = (cls: string, brunnel: string | null) => boolean | number;

/** What a road counts for where track at ground level crosses it (T-1929): the bus classes 1, a
 * motorway 3; a road on a bridge or in a tunnel is already out of the way, 0. */
export const crossingWeight: RoadKeep = (cls, brunnel) =>
  brunnel === "bridge" || brunnel === "tunnel" ? 0 : cls === "motorway" ? 3 : BUS_CLASSES.has(cls) ? 1 : 0;

const UTF8 = new TextDecoder();

class Reader {
  pos = 0;
  b: Uint8Array;
  end: number;
  constructor(b: Uint8Array, end = b.length) {
    this.b = b;
    this.end = end;
  }
  varint(): number {
    let r = 0, mul = 1, byte: number;
    do {
      byte = this.b[this.pos++];
      r += (byte & 0x7f) * mul;
      mul *= 128;
    } while (byte & 0x80);
    return r;
  }
  /** a length-delimited field's end */
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
  string(end: number): string {
    const s = UTF8.decode(this.b.subarray(this.pos, end));
    this.pos = end;
    return s;
  }
}

/** A value message as a string, an integer's raw varint in decimal. */
function valueString(r: Reader, end: number): string | null {
  let out: string | null = null;
  while (r.pos < end) {
    const tag = r.varint(), f = tag >> 3, w = tag & 7;
    if (f === 1 && w === 2) out = r.string(r.len());
    else if (w === 0) out = String(r.varint());
    else r.skip(w);
  }
  return out;
}

/** A layer's name, every other field skipped; the reader is left anywhere in the layer. */
function layerName(r: Reader, end: number): string {
  while (r.pos < end) {
    const tag = r.varint(), f = tag >> 3, w = tag & 7;
    if (f === 1 && w === 2) return r.string(r.len());
    r.skip(w);
  }
  return "";
}

/** The lines of a tile's `transportation` layer that `keep` accepts (true or a weight above 0); none
 * when the tile has no such layer. */
export function decodeRoads(buf: ArrayBuffer, keep: RoadKeep = (c) => BUS_CLASSES.has(c)): RoadLines {
  const b = new Uint8Array(buf);
  const t = new Reader(b);
  while (t.pos < t.end) {
    const tag = t.varint(), f = tag >> 3, w = tag & 7;
    if (f !== 3 || w !== 2) {
      t.skip(w);
      continue;
    }
    const end = t.len();
    // T-2227: another layer is skipped on its name alone; decoding the values of the building and
    // place layers took most of a tile's 2-22 ms
    const r = new Reader(b, end);
    r.pos = t.pos;
    if (layerName(r, end) !== "transportation") {
      t.pos = end;
      continue;
    }
    r.pos = t.pos;
    let name = "", extent = 4096;
    const feats: [number, number][] = [], keys: string[] = [], values: (string | null)[] = [];
    while (r.pos < end) {
      const ft = r.varint(), ff = ft >> 3, fw = ft & 7;
      if (ff === 1 && fw === 2) name = r.string(r.len());
      else if (ff === 2 && fw === 2) {
        const e = r.len();
        feats.push([r.pos, e]);
        r.pos = e;
      } else if (ff === 3 && fw === 2) keys.push(r.string(r.len()));
      else if (ff === 4 && fw === 2) {
        const e = r.len();
        values.push(valueString(r, e));
        r.pos = e;
      } else if (ff === 5 && fw === 0) extent = r.varint();
      else r.skip(fw);
    }
    t.pos = end;
    if (name !== "transportation") continue;
    const classKey = keys.indexOf("class"), brunnelKey = keys.indexOf("brunnel"), layerKey = keys.indexOf("layer");
    const lines: Int32Array[] = [], weights: number[] = [], levels: number[] = [];
    for (const [s, e] of feats) {
      const fr = new Reader(b, e);
      fr.pos = s;
      let type = 0, geomStart = -1, geomEnd = -1, cls: string | null = null, brunnel: string | null = null, layer = 0;
      while (fr.pos < e) {
        const ft = fr.varint(), ff = ft >> 3, fw = ft & 7;
        if (ff === 2 && fw === 2) {
          const te = fr.len();
          while (fr.pos < te) {
            const k = fr.varint(), v = fr.varint();
            if (k === classKey) cls = values[v] ?? null;
            else if (k === brunnelKey) brunnel = values[v] ?? null;
            else if (k === layerKey) layer = Number(values[v]) || 0;
          }
        } else if (ff === 3 && fw === 0) type = fr.varint();
        else if (ff === 4 && fw === 2) {
          geomEnd = fr.len();
          geomStart = fr.pos;
          fr.pos = geomEnd;
        } else fr.skip(fw);
      }
      if (type !== 2 || geomStart < 0 || cls === null) continue;
      const kept = keep(cls, brunnel);
      const weight = kept === true ? 1 : kept === false ? 0 : kept;
      if (!(weight > 0)) continue;
      const level = (brunnel === "bridge" ? 1000 : brunnel === "tunnel" ? 2000 : 0) + layer;
      const g = new Reader(b, geomEnd);
      g.pos = geomStart;
      let x = 0, y = 0, cur: number[] | null = null;
      while (g.pos < geomEnd) {
        const cmd = g.varint(), id = cmd & 7;
        let count = cmd >> 3;
        if (id === 7) continue;
        while (count--) {
          const dx = g.varint(), dy = g.varint();
          x += (dx >> 1) ^ -(dx & 1);
          y += (dy >> 1) ^ -(dy & 1);
          if (id === 1) {
            if (cur && cur.length >= 4) lines.push(Int32Array.from(cur)), weights.push(weight), levels.push(level);
            cur = [x, y];
          } else cur?.push(x, y);
        }
      }
      if (cur && cur.length >= 4) lines.push(Int32Array.from(cur)), weights.push(weight), levels.push(level);
    }
    return { extent, lines, weights, levels };
  }
  return { extent: 4096, lines: [], weights: [] };
}
