// The world split into regions of at most `cap` stations (T-4113, notes/T-4113.md), so an edit's work can be
// bounded to the regions it touched and done there first.
//
// A region is a leaf of a quadtree over the track model's plane: a cell is split exactly when it holds more than
// `cap` stations and is wider than `MIN_HALF` (stacked platforms could put more than `cap` on one spot, so the
// cap can be passed at that floor). The leaves are a function of the station set alone, so an edit moves only the
// leaves where stations came or went, and an undo gives back the regions from before. Every node and edge belongs
// to one leaf (an edge to its a end's); a leaf is dirty from a change to anything in it until it is taken.

/** half the root cell's width, plane units: wider than any frame reaches */
const ROOT_HALF = 2 ** 26;
/** a cell this wide or less (about 1 km) is never split */
const MIN_HALF = 2 ** 9;
const LEVELS = Math.log2(ROOT_HALF / MIN_HALF);
const SPAN = 2 ** (LEVELS + 1);

/** a cell's key: its level and its column and row at that level */
type Key = number;
const keyOf = (level: number, ix: number, iy: number): Key => (level * SPAN + ix) * SPAN + iy;
const levelOf = (k: Key) => Math.floor(k / (SPAN * SPAN));
const halfOf = (level: number) => ROOT_HALF / 2 ** level;

function cellAt(level: number, x: number, y: number): Key {
  const n = 2 ** level, w = (2 * ROOT_HALF) / n;
  const ix = Math.min(n - 1, Math.max(0, Math.floor((x + ROOT_HALF) / w)));
  const iy = Math.min(n - 1, Math.max(0, Math.floor((y + ROOT_HALF) / w)));
  return keyOf(level, ix, iy);
}
const parentOf = (k: Key): Key => {
  const l = levelOf(k), r = k - l * SPAN * SPAN, ix = Math.floor(r / SPAN), iy = r % SPAN;
  return keyOf(l - 1, ix >> 1, iy >> 1);
};

interface Leaf {
  key: Key;
  stations: number;
  nodes: Set<number>;
  edges: Set<number>;
}

interface NodeAt { x: number; y: number; station: boolean; leaf: Key }

/** what `take` hands over: every node and edge in the leaves taken (`all`: every one there is) */
export interface Taken {
  all: boolean;
  nodes: Set<number>;
  edges: Set<number>;
  /** leaves taken, and leaves still dirty after (to be taken by the next read) */
  leaves: number;
  left: number;
}

/** where the player is: the edit's own records and the view, taken first */
export interface Focus {
  nodes?: Iterable<number>;
  /** plane x0, y0, x1, y1 */
  rect?: [number, number, number, number];
}

export class Regions {
  private leaves = new Map<Key, Leaf>();
  /** stations in each cell above a leaf */
  private inner = new Map<Key, number>();
  private at = new Map<number, NodeAt>();
  /** each edge's ends, its leaf its a end's */
  private ends = new Map<number, [number, number]>();
  /** the edges whose a end each node is */
  private from = new Map<number, Set<number>>();
  private dirty = new Set<Key>();
  private everything = true;
  /** leaves split or merged so far (for the notes' figures) */
  reshapes = 0;

  constructor(readonly cap: number) {
    this.reset();
  }

  reset() {
    this.leaves.clear();
    this.inner.clear();
    this.at.clear();
    this.ends.clear();
    this.from.clear();
    this.dirty.clear();
    const k = keyOf(0, 0, 0);
    this.leaves.set(k, { key: k, stations: 0, nodes: new Set(), edges: new Set() });
    this.everything = true;
  }

  get size() { return this.leaves.size; }

  private leafAt(x: number, y: number): Leaf {
    for (let level = 0; ; level++) {
      const l = this.leaves.get(cellAt(level, x, y));
      if (l) return l;
    }
  }

  /** the region a point is in */
  regionAt(x: number, y: number): Key {
    return this.leafAt(x, y).key;
  }

  /** The nodes (`nodes_info`, 10 numbers each) and edges (`edges_info`) as the model has them now: each one new,
   * gone, moved, or a station or not anew marks its region (old and new) dirty. */
  sync(ni: ArrayLike<number>, ei: ArrayLike<number>, only?: { nodes: Set<number>; edges: Set<number> }) {
    // `only`: the records an edit touched, the rest as they were unless the count of either moved
    if (only && ni.length / 10 === this.at.size && ei.length / 10 === this.ends.size) {
      for (let i = 0; i < ni.length; i += 10) {
        if (!only.nodes.has(ni[i])) continue;
        const id = ni[i], x = ni[i + 1], y = ni[i + 2], station = ni[i + 7] > 0, was = this.at.get(id);
        if (was && was.x === x && was.y === y && was.station === station) continue;
        if (was) this.dropNode(id, was);
        this.addNode(id, x, y, station);
      }
      for (let i = 0; i < ei.length; i += 10) {
        if (!only.edges.has(ei[i])) continue;
        const id = ei[i], a = ei[i + 1], b = ei[i + 2], was = this.ends.get(id);
        if (was && was[0] === a && was[1] === b) continue;
        if (was) this.dropEdge(id, was);
        this.addEdge(id, a, b);
      }
      return;
    }
    const seen = new Set<number>();
    for (let i = 0; i < ni.length; i += 10) {
      const id = ni[i], x = ni[i + 1], y = ni[i + 2], station = ni[i + 7] > 0;
      seen.add(id);
      const was = this.at.get(id);
      if (was && was.x === x && was.y === y && was.station === station) continue;
      if (was) this.dropNode(id, was);
      this.addNode(id, x, y, station);
    }
    if (this.at.size > seen.size) for (const [id, was] of this.at) if (!seen.has(id)) this.dropNode(id, was);
    const seenE = new Set<number>();
    for (let i = 0; i < ei.length; i += 10) {
      const id = ei[i], a = ei[i + 1], b = ei[i + 2];
      seenE.add(id);
      const was = this.ends.get(id);
      if (was && was[0] === a && was[1] === b) continue;
      if (was) this.dropEdge(id, was);
      this.addEdge(id, a, b);
    }
    if (this.ends.size > seenE.size) for (const [id, was] of this.ends) if (!seenE.has(id)) this.dropEdge(id, was);
  }

  private addNode(id: number, x: number, y: number, station: boolean) {
    let leaf = this.leafAt(x, y);
    this.at.set(id, { x, y, station, leaf: leaf.key });
    leaf.nodes.add(id);
    for (const e of this.edgesFrom(id)) leaf.edges.add(e);
    this.dirty.add(leaf.key);
    if (!station) return;
    for (let k = leaf.key; levelOf(k) > 0;) { k = parentOf(k); this.inner.set(k, (this.inner.get(k) ?? 0) + 1); }
    leaf.stations++;
    while (leaf.stations > this.cap && halfOf(levelOf(leaf.key)) > MIN_HALF) {
      this.split(leaf);
      leaf = this.leafAt(x, y);
    }
  }

  private dropNode(id: number, was: NodeAt) {
    const leaf = this.leaves.get(was.leaf)!;
    leaf.nodes.delete(id);
    for (const e of this.edgesFrom(id)) leaf.edges.delete(e);
    this.at.delete(id);
    this.dirty.add(leaf.key);
    if (!was.station) return;
    leaf.stations--;
    // the highest cell above that now fits becomes one leaf again
    let top: Key | null = null;
    for (let k = leaf.key; levelOf(k) > 0;) {
      k = parentOf(k);
      const n = (this.inner.get(k) ?? 0) - 1;
      this.inner.set(k, n);
      if (n <= this.cap) top = k;
    }
    if (top !== null) this.merge(top);
  }

  private addEdge(id: number, a: number, b: number) {
    this.ends.set(id, [a, b]);
    let f = this.from.get(a);
    if (!f) this.from.set(a, (f = new Set()));
    f.add(id);
    const la = this.at.get(a)?.leaf;
    const leaf = la === undefined ? undefined : this.leaves.get(la);
    leaf?.edges.add(id);
    this.markNode(a);
    this.markNode(b);
  }

  private dropEdge(id: number, was: [number, number]) {
    this.ends.delete(id);
    const f = this.from.get(was[0]);
    if (f) { f.delete(id); if (!f.size) this.from.delete(was[0]); }
    const la = this.at.get(was[0])?.leaf;
    if (la !== undefined) this.leaves.get(la)?.edges.delete(id);
    this.markNode(was[0]);
    this.markNode(was[1]);
  }

  /** the edges whose a end is node `id` */
  private edgesFrom(id: number): Iterable<number> {
    return this.from.get(id) ?? [];
  }

  private split(leaf: Leaf) {
    this.reshapes++;
    const level = levelOf(leaf.key) + 1;
    this.leaves.delete(leaf.key);
    this.inner.set(leaf.key, leaf.stations);
    const dirty = this.dirty.delete(leaf.key);
    const kids = new Map<Key, Leaf>();
    const kid = (x: number, y: number) => {
      const k = cellAt(level, x, y);
      let l = kids.get(k);
      if (!l) kids.set(k, (l = { key: k, stations: 0, nodes: new Set(), edges: new Set() }));
      return l;
    };
    // all four children exist, empty ones too, so every point lands in a leaf
    const p = this.at.get(leaf.nodes.values().next().value!)!;
    const base = cellAt(level - 1, p.x, p.y), r = base - (level - 1) * SPAN * SPAN, ix = Math.floor(r / SPAN), iy = r % SPAN;
    for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) { const k = keyOf(level, 2 * ix + dx, 2 * iy + dy); kids.set(k, { key: k, stations: 0, nodes: new Set(), edges: new Set() }); }
    for (const id of leaf.nodes) {
      const n = this.at.get(id)!, l = kid(n.x, n.y);
      l.nodes.add(id);
      if (n.station) l.stations++;
      n.leaf = l.key;
    }
    for (const e of leaf.edges) {
      const a = this.ends.get(e)![0], l = this.leaves.get(this.at.get(a)!.leaf) ?? kids.get(this.at.get(a)!.leaf)!;
      l.edges.add(e);
    }
    for (const l of kids.values()) {
      this.leaves.set(l.key, l);
      if (dirty) this.dirty.add(l.key);
    }
  }

  /** every leaf under cell `top` as one leaf */
  private merge(top: Key) {
    this.reshapes++;
    const into: Leaf = { key: top, stations: 0, nodes: new Set(), edges: new Set() };
    let dirty = false;
    const level = levelOf(top);
    for (const [k, l] of this.leaves) {
      if (levelOf(k) <= level) continue;
      let up = k;
      while (levelOf(up) > level) up = parentOf(up);
      if (up !== top) continue;
      this.leaves.delete(k);
      if (this.dirty.delete(k)) dirty = true;
      into.stations += l.stations;
      for (const n of l.nodes) { into.nodes.add(n); this.at.get(n)!.leaf = top; }
      for (const e of l.edges) into.edges.add(e);
    }
    for (const k of [...this.inner.keys()]) {
      let up = k;
      while (levelOf(up) > level) up = parentOf(up);
      if (up === top) this.inner.delete(k);
    }
    this.leaves.set(top, into);
    if (dirty) this.dirty.add(top);
  }

  markNode(id: number) {
    const n = this.at.get(id);
    if (n) this.dirty.add(n.leaf);
  }
  /** an edge's two ends' regions */
  markEdge(id: number) {
    const e = this.ends.get(id);
    if (e) { this.markNode(e[0]); this.markNode(e[1]); }
  }
  markPoint(x: number, y: number) {
    this.dirty.add(this.leafAt(x, y).key);
  }
  markAll() {
    this.everything = true;
  }
  /** whether a point is in a region of `t` (one `take` just handed over) */
  isTaken(t: Taken, x: number, y: number): boolean {
    return t.all || this.takenKeys.has(this.leafAt(x, y).key);
  }
  private takenKeys = new Set<Key>();

  /** The dirty regions' nodes and edges, no longer dirty. With a `focus` and more than `budget` nodes and edges
   * dirty, only the regions in it (the edit's own records' and those the view overlaps); the rest stay dirty for
   * the next take. */
  take(focus?: Focus, budget = Infinity): Taken {
    const keys = this.everything ? [...this.leaves.keys()] : [...this.dirty].filter((k) => this.leaves.has(k));
    let size = 0;
    for (const k of keys) { const l = this.leaves.get(k)!; size += l.nodes.size + l.edges.size; }
    let chosen = keys;
    if (focus && size > budget) {
      const want = new Set<Key>();
      for (const n of focus.nodes ?? []) { const at = this.at.get(n); if (at) want.add(at.leaf); }
      if (focus.rect) {
        const [x0, y0, x1, y1] = focus.rect;
        for (const k of keys) {
          const l = levelOf(k), h = halfOf(l), r = k - l * SPAN * SPAN, cx = -ROOT_HALF + (2 * Math.floor(r / SPAN) + 1) * h, cy = -ROOT_HALF + (2 * (r % SPAN) + 1) * h;
          if (cx + h >= x0 && cx - h <= x1 && cy + h >= y0 && cy - h <= y1) want.add(k);
        }
      }
      chosen = keys.filter((k) => want.has(k));
    }
    const all = this.everything && chosen.length === keys.length;
    const nodes = new Set<number>(), edges = new Set<number>();
    this.takenKeys = new Set(chosen);
    for (const k of chosen) {
      const l = this.leaves.get(k)!;
      for (const n of l.nodes) nodes.add(n);
      for (const e of l.edges) edges.add(e);
      this.dirty.delete(k);
    }
    if (this.everything) {
      this.everything = false;
      if (!all) for (const k of keys) if (!this.takenKeys.has(k)) this.dirty.add(k);
    }
    return { all, nodes, edges, leaves: chosen.length, left: this.dirty.size };
  }

  /** stations per leaf, for the notes */
  stats(): { leaves: number; max: number; mean: number; empty: number } {
    let max = 0, sum = 0, empty = 0;
    for (const l of this.leaves.values()) { max = Math.max(max, l.stations); sum += l.stations; if (!l.nodes.size) empty++; }
    return { leaves: this.leaves.size, max, mean: sum / Math.max(1, this.leaves.size - empty), empty };
  }

  /** The leaves as a fresh build from the same nodes and edges would make them (the fuzz's check that they are
   * a function of the station set): leaf key to its stations, nodes and edges, sorted. */
  shape(): string {
    return [...this.leaves.values()].filter((l) => l.nodes.size || l.edges.size).sort((p, q) => p.key - q.key)
      .map((l) => `${l.key}:${l.stations}:${[...l.nodes].sort((a, b) => a - b).join(",")}:${[...l.edges].sort((a, b) => a - b).join(",")}`).join("|");
  }
}
