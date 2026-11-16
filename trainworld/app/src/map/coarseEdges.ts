// T-3491: the coarse outlines drawn on desktop, every zone at every zoom, held as the file's packed deltas
// (coarseOutlines.ts) and one number a point, the other zone holding the border from that point to the next.
// OutlineEdges' maps and the decoded rings took about 140 MB of the page for the world's 4,045 zones
// (notes/T-3472.md); this takes about 5 MB beside the packed file, worked out once off the main thread.
import type { Feature, MultiLineString, Position } from "geojson";
import type { PackedOutlines } from "./coarseOutlines";

/** `partner[p]`: the other zone (an index into the packed ids) holding the border from point `p` (data index / 2)
 * to the next point of its ring; -1 for none, -2 for two or more others, listed in `more` with the zone itself. */
export type CoarseTopology = { partner: Int32Array; more: Map<number, number[]> };

/** Each point's partner across the border it starts: the file repeats a shared run's vertices in both zones (T-799),
 * so a border two zones hold is the same pair of points in each. About 0.3 s and 60 MB passing for the world. */
export function coarseTopology(p: PackedOutlines): CoarseTopology {
  const { data, zoneRing, ringAt } = p, P = data.length >> 1;
  let size = 1 << 12;
  while (size < P * 1.5) size *= 2;
  const mask = size - 1;
  const hash = (a: number, b: number) => { const s = Math.imul(a ^ Math.imul(b, 0x9e3779b1), 0x85ebca6b); return (s ^ (s >>> 15)) & mask; };
  // each point's vertex number, the same for the same place
  const vid = new Int32Array(P), zoneOf = new Int32Array(P);
  {
    const kx = new Int32Array(size), ky = new Int32Array(size), kid = new Int32Array(size).fill(-1);
    let n = 0;
    for (let z = 0; z < p.ids.length; z++) for (let r = zoneRing[z]; r < zoneRing[z + 1]; r++) {
      let x = 0, y = 0;
      for (let i = ringAt[r]; i < ringAt[r + 1]; i += 2) {
        x += data[i]; y += data[i + 1];
        const pt = i >> 1;
        zoneOf[pt] = z;
        for (let s = hash(x, y); ; s = (s + 1) & mask) {
          const k = kid[s];
          if (k < 0) { kx[s] = x; ky[s] = y; kid[s] = n; vid[pt] = n++; break; }
          if (kx[s] === x && ky[s] === y) { vid[pt] = k; break; }
        }
      }
    }
  }
  // each border by its two vertices: the first and second point starting it, any more apart
  const su = new Int32Array(size), sv = new Int32Array(size), first = new Int32Array(size).fill(-1), second = new Int32Array(size).fill(-1);
  const extra = new Map<number, number[]>();
  const partner = new Int32Array(P).fill(-1);
  for (let r = 0; r < ringAt.length - 1; r++) {
    const a0 = ringAt[r] >> 1, a1 = ringAt[r + 1] >> 1;
    for (let pt = a0; pt < a1; pt++) {
      let u = vid[pt], v = vid[pt + 1 < a1 ? pt + 1 : a0];
      if (u === v) continue;
      if (u > v) { const t = u; u = v; v = t; }
      for (let s = hash(u, v); ; s = (s + 1) & mask) {
        const q = first[s];
        if (q < 0) { su[s] = u; sv[s] = v; first[s] = pt; break; }
        if (su[s] !== u || sv[s] !== v) continue;
        // (a zone's ring touching itself along a border: still the zone's alone)
        if (zoneOf[q] === zoneOf[pt]) break;
        if (second[s] < 0) { second[s] = pt; partner[q] = zoneOf[pt]; partner[pt] = zoneOf[q]; }
        else { const l = extra.get(s); if (l) l.push(pt); else extra.set(s, [pt]); }
        break;
      }
    }
  }
  const more = new Map<number, number[]>();
  for (const [s, pts] of extra) {
    const all = [first[s], second[s], ...pts], zones = [...new Set(all.map((q) => zoneOf[q]))].sort((a, b) => a - b);
    for (const q of all) { partner[q] = -2; more.set(q, zones); }
  }
  return { partner, more };
}

/** The coarse zones' outlines with each shared border drawn once, as OutlineEdges draws them: by the first zone in
 * the file's order holding it among those still drawn coarse, and by none when the focused zone holds it. A zone
 * whose own outline is in leaves (`set(id, false)`) and its borders pass to its neighbours. */
export class CoarseEdges {
  private index = new Map<string, number>();
  private active: Uint8Array;
  private focus = -1;
  /** each zone's feature as last drawn (null: none), while its borders' holders and the focus stay */
  private drawn = new Map<number, Feature<MultiLineString> | null>();

  constructor(private p: PackedOutlines, private t: CoarseTopology) {
    p.ids.forEach((id, k) => this.index.set(id, k));
    this.active = new Uint8Array(p.ids.length).fill(1);
  }

  /** whether zone `id` is drawn from the coarse file */
  has(id: string) { const k = this.index.get(id); return k !== undefined && this.active[k] === 1; }

  /** Zone `id` drawn from the coarse file, or not (its own outline is in). */
  set(id: string, on: boolean) {
    const k = this.index.get(id);
    if (k === undefined || (this.active[k] === 1) === on) return;
    this.active[k] = on ? 1 : 0;
    this.touch(k);
  }

  /** the zone and those sharing a border with it redraw */
  private touch(k: number) {
    const { zoneRing, ringAt } = this.p, { partner, more } = this.t;
    this.drawn.delete(k);
    for (let pt = ringAt[zoneRing[k]] >> 1, end = ringAt[zoneRing[k + 1]] >> 1; pt < end; pt++) {
      const o = partner[pt];
      if (o >= 0) this.drawn.delete(o);
      else if (o === -2) for (const z of more.get(pt)!) this.drawn.delete(z);
    }
  }

  private setFocus(id: string) {
    const k = this.index.get(id) ?? -1;
    if (k === this.focus) return;
    for (const f of [this.focus, k]) if (f >= 0) this.touch(f);
    this.focus = k;
  }

  /** whether zone `k` draws the border from point `pt` */
  private draws(k: number, pt: number) {
    const o = this.t.partner[pt];
    if (o === -1) return true;
    for (const z of o >= 0 ? [o] : this.t.more.get(pt)!) if (z !== k && this.active[z] && (z < k || z === this.focus)) return false;
    return true;
  }

  /** One feature per coarse zone with the lines it draws (property `id`), none for `focused` or for a zone not
   * drawn coarse; only the zones in `only` when given. Features of zones left out are forgotten. */
  lines(focused: string, only?: Set<string>): Feature<MultiLineString>[] {
    this.setFocus(focused);
    const out: Feature<MultiLineString>[] = [], ids = this.p.ids;
    if (only) for (const k of this.drawn.keys()) if (!only.has(ids[k])) this.drawn.delete(k);
    const one = (k: number) => {
      if (!this.active[k] || k === this.focus) return;
      let f = this.drawn.get(k);
      if (f === undefined) this.drawn.set(k, (f = this.zoneLines(k)));
      if (f) out.push(f);
    };
    if (only) for (const id of only) { const k = this.index.get(id); if (k !== undefined) one(k); }
    else for (let k = 0; k < ids.length; k++) one(k);
    return out;
  }

  private zoneLines(k: number): Feature<MultiLineString> | null {
    const { data, zoneRing, ringAt, q } = this.p, coordinates: Position[][] = [];
    for (let r = zoneRing[k]; r < zoneRing[k + 1]; r++) {
      const a0 = ringAt[r], n = (ringAt[r + 1] - a0) >> 1, base = a0 >> 1;
      if (n < 2) continue;
      const ring: Position[] = new Array(n + 1);
      for (let i = 0, x = 0, y = 0; i < n; i++) { x += data[a0 + 2 * i]; y += data[a0 + 2 * i + 1]; ring[i] = [x / q, y / q]; }
      ring[n] = ring[0];
      let start = -1;
      for (let i = 0; i < n && start < 0; i++) if (!this.draws(k, base + i)) start = i;
      if (start < 0) { coordinates.push(ring); continue; }
      // begin just after a gap, so a run that crosses the ring's seam stays one line
      let line: Position[] = [];
      for (let j = 1; j <= n; j++) {
        const i = (start + j) % n;
        if (this.draws(k, base + i)) {
          if (!line.length) line.push(ring[i]);
          line.push(ring[i + 1]);
        } else if (line.length) {
          coordinates.push(line);
          line = [];
        }
      }
      if (line.length) coordinates.push(line);
    }
    return coordinates.length ? { type: "Feature", properties: { id: this.p.ids[k] }, geometry: { type: "MultiLineString", coordinates } } : null;
  }
}
