// Where a complex's waiting room and car park go (T-1528, notes/T-1528.md), in the clock worker.
// Each is drawn once per station complex, however many of its stations have one. The waiting room
// is an annex joined to one box (stationGeometry.ts `stationAnnexes`): of the stations with one and
// the two sides of each, the spot with the fewest other platforms, tracks, car parks and the name
// label near it. The car park is a lot of its own beside the complex (map/parkingLots.ts): the
// nearest spot along one of the complex's platforms that overlaps no box, track, waiting room, other
// lot or the label, shrinking if none does, and the least bad spot if nothing fits.

import { ShapeGrid } from "./shapeGrid";
import { PLATFORM_POINTS, ST_FLIP, ST_ONE, ST_ONE_NEG, ST_PARK, ST_ROOM } from "./stationGeometry";

/** A lot's numbers in `NetworkBuffers.lots`: centre x, y (local); half its length along it as a
 * vector (local); half length and half depth, m; bays; flags (LOT_*); the station its hover names
 * (index); how many of the complex's stations have park and ride. */
export const LOT_STRIDE = 10;
export const LOT_BLUEPRINT = 1, LOT_OFF = 2;
/** A car space and the aisle between two rows, m: a bay is a row each side of an aisle. */
export const SPACE_W = 2.5, SPACE_L = 5, AISLE = 6, BAY_M = 2 * SPACE_L + AISLE;
/** A lot runs along the longest platform with park and ride, within these lengths, m. */
const LOT_MIN_M = 80, LOT_MAX_M = 300, LOT_SHORT_M = 60;
/** At most this many bays deep: one for each station with park and ride. */
const MAX_BAYS = 3;
/** How far a lot's near edge is from the platform's centre line, m: past a box drawn at zoom 14,
 * where it is about 32 m wide on the ground (T-553). */
const LOT_GAPS_M = [20, 30, 42, 56, 72];
/** Kept clear round a lot, m: a box's centre line (half a zoom 14 box plus a little), a track, a
 * waiting room or another lot. */
const CLEAR_BOX_M = 18, CLEAR_TRACK_M = 6, CLEAR_LOT_M = 4;
/** Where a waiting room reaches, m from the platform's centre line and along it from its middle:
 * about the drawn size between zoom 14.5 and 16, where a box is 16-30 m wide on the ground. */
const ACROSS_M = [14, 24, 34], ALONG_M = [-28, -10, 10, 28];
const ROOM_NEAR_M = 14, ROOM_TRACK_M = 5;
/** The name label's patch beside the complex, m (stationLabels.ts: below the station where its
 * track runs east-west, else to its right), about its size at zoom 16. */
const LABEL_BELOW = { halfWide: 90, deep: 50 }, LABEL_RIGHT = { wide: 150, halfDeep: 25 };
const CELL_M = 40;

export interface Rect { x: number; y: number; ux: number; uy: number; hl: number; hd: number }

/** A station building's outline (local x, y pairs) and its members (station indices), which car parks
 * stay out of (T-2219). */
export interface LotOutline { corners: ArrayLike<number>; members: readonly number[] }

/** Where station `i`'s waiting room stands on the floor, as placed (ST_ROOM, ST_FLIP): the patch the
 * room search reserves, half sizes in m. */
export function roomRect(i: number, stations: Float32Array, stationLine: Float32Array, perM: number): Rect {
  const f = platformFrame(i, stations, stationLine);
  const side = Math.round(stations[i * 3 + 2]) & ST_FLIP ? 1 : -1;
  return { x: f.mx + f.nx * side * 24 * perM, y: f.my + f.ny * side * 24 * perM, ux: f.dx, uy: f.dy, hl: 30, hd: 12 };
}

/** A platform's middle, its way there and the left of it (+), local units. */
function platformFrame(i: number, stations: Float32Array, stationLine: Float32Array) {
  const x = stations[i * 3], y = stations[i * 3 + 1], o = i * 2 * PLATFORM_POINTS, m = PLATFORM_POINTS >> 1;
  let dx = stationLine[o + m * 2 + 2] - stationLine[o + m * 2 - 2], dy = stationLine[o + m * 2 + 3] - stationLine[o + m * 2 - 1];
  const dl = Math.hypot(dx, dy) || 1;
  dx /= dl; dy /= dl;
  return { mx: x + stationLine[o + m * 2], my: y + stationLine[o + m * 2 + 1], dx, dy, nx: -dy, ny: dx };
}

/** T-3803: what the last placement read and placed, kept between the clock worker's builds (`NetCache.facilities`), so
 * a build after an edit places again only the complexes whose inputs changed. A complex's spots depend on its
 * stations, the segments in the grid cells its search read, the station buildings, and the rooms and lots placed
 * before it that come near its tries; each is compared with what the last build had, so nothing kept can be stale
 * (an undo, a load or a merge is just more changed inputs). The grid of segments near any complex is updated edge by
 * edge (an edge's points are the same array while it is unchanged) and station by station; it is made again when the
 * near cells move (a complex with a facility gained, lost or moved a station) or the scale does. */
export interface FacilityCache {
  /** local units a metre the grid was made at; NaN before any */
  perM: number;
  /** the facility complexes' stations' positions and platforms the near cells were made from */
  nearOf: number[];
  /** the near cells in tiles of 32 x 32, each 32 row masks (bit cx & 31 of row cy & 31), by `cellKey` of the tile */
  near: Map<number, Int32Array>;
  /** each segment's x0, y0, x1, y1, owner (an edge id, or -1 - a station's id), by id; `free`: ids to reuse */
  seg: number[];
  free: number[];
  grid: CellLists;
  edges: Map<number, { pts: Float32Array; segs: number[] }>;
  /** the network's points map `edges` was made from (map/network.ts keeps it while no edge changed) */
  edgePts: Map<number, Float32Array> | null;
  stations: Map<number, { line: number[]; segs: number[] }>;
  /** the station buildings as last read */
  outlines: number[];
  /** by the complex's stations' ids */
  complexes: Map<string, Placed>;
}

/** One complex's placement and what it read. */
interface Placed {
  inputs: number[];
  /** the grid cells its search read: cx0, cy0, cx1, cy1 */
  cells: number[];
  /** the box (local) its room spots and lot tries cover, which a room or lot placed before it must come near to matter */
  env: number[];
  room: { id: number; side: number; rect: Rect } | null;
  lot: { rect: Rect; bays: number; flags: number; park0: number; parks: number } | null;
}

export const newFacilityCache = (): FacilityCache => ({ perM: NaN, nearOf: [], near: new Map(), seg: [], free: [], grid: new CellLists(), edges: new Map(), edgePts: null, stations: new Map(), outlines: [], complexes: new Map() });

/** T-4187: the grid's segments by cell, each cell a list through two pooled arrays (an entry's segment and the next entry,
 * -1 at the end), by `cellKey`, as an array a cell held a few ids in room for 17. A cell's segments come in no set order:
 * every read of one takes them as a set (`placeOne`). */
class CellLists {
  private head = new Map<number, number>();
  /** per entry: its segment, and the cell's next entry */
  seg = new Int32Array(1024);
  next = new Int32Array(1024);
  private used = 0;
  private free = -1;
  /** the cell's first entry, -1 none */
  first(key: number): number {
    return this.head.get(key) ?? -1;
  }
  add(key: number, k: number) {
    let e = this.free;
    if (e >= 0) this.free = this.next[e];
    else {
      e = this.used++;
      if (e === this.seg.length) {
        const seg = new Int32Array(e * 2), next = new Int32Array(e * 2);
        seg.set(this.seg);
        next.set(this.next);
        this.seg = seg;
        this.next = next;
      }
    }
    this.seg[e] = k;
    this.next[e] = this.head.get(key) ?? -1;
    this.head.set(key, e);
  }
  remove(key: number, k: number) {
    let e = this.head.get(key)!, before = -1;
    while (this.seg[e] !== k) {
      before = e;
      e = this.next[e];
    }
    const after = this.next[e];
    if (before >= 0) this.next[before] = after;
    else if (after >= 0) this.head.set(key, after);
    else this.head.delete(key);
    this.next[e] = this.free;
    this.free = e;
  }
}

const sameNums = (a: readonly number[], b: readonly number[]) => a.length === b.length && a.every((v, i) => Object.is(v, b[i]));
const cellKey = (cx: number, cy: number) => cx * 1_000_003 + cy;

/** Place every complex's waiting room and car park. `stations` flags: ST_ROOM and ST_PARK as built
 * at each station on entry; on return ST_ROOM is left only on the one station per complex whose box
 * draws the room, with ST_FLIP for its side. `complexOf`: each node's complex (0xffffffff none);
 * absent, each station is its own. `outlines`: the station buildings, which lots go outside of
 * (T-2219). `cache`: the last build's (T-3803); a new one places every complex. Returns the lots, LOT_STRIDE numbers
 * each. */
export function placeFacilities(
  st: { id: number; lat: number; stationBuilt?: boolean; stationOff?: boolean; platform: number }[],
  stations: Float32Array,
  stationLine: Float32Array,
  stationShape: Float32Array,
  edgePts: Map<number, Float32Array>,
  waysAt: Map<number, number[]>,
  complexOf?: ArrayLike<number>,
  outlines: readonly LotOutline[] = [],
  cache: FacilityCache = newFacilityCache(),
): Float32Array {
  const flag = (i: number) => Math.round(stations[i * 3 + 2]);
  const want: number[] = [];
  for (let i = 0; i < st.length; i++) if (flag(i) & (ST_ROOM | ST_PARK)) want.push(i);
  if (!want.length) {
    cache.perM = NaN;
    cache.complexes.clear();
    return new Float32Array(0);
  }
  const perM = 1 / Math.cos(st[want[0]].lat * Math.PI / 180), cell = CELL_M * perM;
  const keyOf = (i: number) => {
    const c = complexOf?.[st[i].id];
    return c === undefined || c === 0xffffffff ? -1 - i : c;
  };
  // the complexes with a facility, each with all its stations, in node order
  const groups = new Map<number, number[]>();
  for (const i of want) groups.set(keyOf(i), []);
  for (let i = 0; i < st.length; i++) groups.get(keyOf(i))?.push(i);

  // the near cells: only what lies near a complex with a facility goes in the grid, its platforms and a lot's reach past them
  const nearOf: number[] = [];
  for (const members of groups.values()) for (const i of members) nearOf.push(stations[i * 3], stations[i * 3 + 1], st[i].platform);
  let allDirty = false;
  if (!Object.is(cache.perM, perM) || !sameNums(cache.nearOf, nearOf)) {
    // (T-3659: a square is marked a word per row, however many platforms share its cells)
    const near = new Map<number, Int32Array>();
    for (const members of groups.values()) for (const i of members) {
      const cx = Math.floor(stations[i * 3] / cell), cy = Math.floor(stations[i * 3 + 1] / cell);
      const r = Math.ceil(((st[i].platform / 2 + LOT_MAX_M / 2 + LOT_GAPS_M[LOT_GAPS_M.length - 1] + MAX_BAYS * BAY_M) * perM) / cell) + 1;
      for (let tx = Math.floor((cx - r) / 32); tx <= Math.floor((cx + r) / 32); tx++) {
        const lo = Math.max(0, cx - r - tx * 32), hi = Math.min(31, cx + r - tx * 32);
        const mask = (-1 << lo) & (0xffffffff >>> (31 - hi));
        for (let ty = Math.floor((cy - r) / 32); ty <= Math.floor((cy + r) / 32); ty++) {
          let tile = near.get(cellKey(tx, ty));
          if (!tile) near.set(cellKey(tx, ty), (tile = new Int32Array(32)));
          for (let y = Math.max(0, cy - r - ty * 32); y <= Math.min(31, cy + r - ty * 32); y++) tile[y] |= mask;
        }
      }
    }
    Object.assign(cache, { perM, nearOf, near, seg: [], free: [], grid: new CellLists(), edges: new Map(), edgePts: null, stations: new Map() });
    cache.complexes.clear();
    allDirty = true;
  }
  const { seg, grid, near } = cache;
  const isNear = (x: number, y: number) => {
    const cx = Math.floor(x / cell), cy = Math.floor(y / cell), tile = near.get(cellKey(Math.floor(cx / 32), Math.floor(cy / 32)));
    return tile !== undefined && (tile[cy & 31] & (1 << (cx & 31))) !== 0;
  };
  // a straight run of track can be one long segment: tried a cell apart along it
  const passesNear = (x0: number, y0: number, x1: number, y1: number) => {
    const n = Math.ceil(Math.hypot(x1 - x0, y1 - y0) / cell);
    for (let j = 0; j <= n; j++) if (isNear(x0 + (x1 - x0) * j / Math.max(1, n), y0 + (y1 - y0) * j / Math.max(1, n))) return true;
    return false;
  };
  /** the grid cells of each segment put in or taken out: cx0, cy0, cx1, cy1 */
  const changed: number[] = [];
  const cellsOf = (k: number) => [Math.floor(Math.min(seg[k * 5], seg[k * 5 + 2]) / cell), Math.floor(Math.min(seg[k * 5 + 1], seg[k * 5 + 3]) / cell), Math.floor(Math.max(seg[k * 5], seg[k * 5 + 2]) / cell), Math.floor(Math.max(seg[k * 5 + 1], seg[k * 5 + 3]) / cell)];
  const add = (x0: number, y0: number, x1: number, y1: number, owner: number, into: number[]) => {
    const k = cache.free.pop() ?? seg.length / 5;
    seg[k * 5] = x0; seg[k * 5 + 1] = y0; seg[k * 5 + 2] = x1; seg[k * 5 + 3] = y1; seg[k * 5 + 4] = owner;
    const c = cellsOf(k);
    for (let cx = c[0]; cx <= c[2]; cx++) for (let cy = c[1]; cy <= c[3]; cy++) grid.add(cellKey(cx, cy), k);
    if (!allDirty) changed.push(...c);
    into.push(k);
  };
  const drop = (segs: number[]) => {
    for (const k of segs) {
      const c = cellsOf(k);
      for (let cx = c[0]; cx <= c[2]; cx++) for (let cy = c[1]; cy <= c[3]; cy++) grid.remove(cellKey(cx, cy), k);
      changed.push(...c);
      cache.free.push(k);
    }
  };
  // the edges: an unchanged one keeps its segments
  // (T-4153: none looked at while the network's points are the same map, the edges' own then)
  if (cache.edgePts !== edgePts) {
    const edgesNow = new Map<number, { pts: Float32Array; segs: number[] }>();
    for (const [id, p] of edgePts) {
      let e = cache.edges.get(id);
      if (e?.pts !== p) {
        if (e) drop(e.segs);
        e = { pts: p, segs: [] };
        for (let k = 0; k + 5 < p.length; k += 3) if (passesNear(p[k], p[k + 1], p[k + 3], p[k + 4])) add(p[k], p[k + 1], p[k + 3], p[k + 4], id, e.segs);
      }
      edgesNow.set(id, e);
    }
    for (const [id, e] of cache.edges) if (!edgesNow.has(id)) drop(e.segs);
    cache.edges = edgesNow;
    cache.edgePts = edgePts;
  }
  // the platforms near a complex: one that kept its place and shape keeps its segments
  const stationsNow = new Map<number, { line: number[]; segs: number[] }>();
  for (let i = 0; i < st.length; i++) {
    const x = stations[i * 3], y = stations[i * 3 + 1], o = i * 2 * PLATFORM_POINTS;
    if (!isNear(x, y)) continue;
    const line = [x, y, ...stationLine.subarray(o, o + 2 * PLATFORM_POINTS)];
    let s = cache.stations.get(st[i].id);
    if (!s || !sameNums(s.line, line)) {
      if (s) drop(s.segs);
      s = { line, segs: [] };
      for (let j = 0; j + 1 < PLATFORM_POINTS; j++) add(x + stationLine[o + j * 2], y + stationLine[o + j * 2 + 1], x + stationLine[o + j * 2 + 2], y + stationLine[o + j * 2 + 3], -1 - st[i].id, s.segs);
    }
    stationsNow.set(st[i].id, s);
  }
  for (const [id, s] of cache.stations) if (!stationsNow.has(id)) drop(s.segs);
  cache.stations = stationsNow;
  // the station buildings, by their stations' ids
  const outlineKey: number[] = [];
  for (const o of outlines) outlineKey.push(o.corners.length, ...Array.from(o.corners), o.members.length, ...o.members.map((i) => st[i].id));
  if (!sameNums(cache.outlines, outlineKey)) {
    cache.outlines = outlineKey;
    allDirty = true;
  }

  const index = new Map<number, number>();
  for (let i = 0; i < st.length; i++) index.set(st[i].id, i);
  const lots: number[] = [], lotRects = new ShapeGrid<Rect>(256 * perM), rooms = new ShapeGrid<Rect>(256 * perM);
  // the station buildings by their boxes, and the first one holding a station of each complex
  const buildings = new ShapeGrid<LotOutline>(256 * perM), homeOf = new Map<number, LotOutline>();
  for (const o of outlines) {
    for (const i of o.members) if (!homeOf.has(keyOf(i))) homeOf.set(keyOf(i), o);
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let k = 0; k + 1 < o.corners.length; k += 2) {
      x0 = Math.min(x0, o.corners[k]); x1 = Math.max(x1, o.corners[k]);
      y0 = Math.min(y0, o.corners[k + 1]); y1 = Math.max(y1, o.corners[k + 1]);
    }
    buildings.add(o, x0, y0, x1, y1);
  }
  /** rooms and lots placed or taken away since the last build: a complex after one is placed again if its tries come near */
  const moved: Rect[] = [];
  const order = [...groups.values()].sort((a, b) => st[a[0]].id - st[b[0]].id);
  const keys = order.map((members) => members.map((i) => st[i].id).join());
  const placedNow = new Map<string, Placed>();
  if (!allDirty) {
    const kept = new Set(keys);
    for (const [k, p] of cache.complexes) if (!kept.has(k)) pushRects(moved, p);
  }
  const ctx: PlaceCtx = { st, stations, stationLine, stationShape, waysAt, buildings, perM, cell, seg, grid, lotRects, rooms, seen: new Uint32Array(seg.length / 5), stamp: 0, found: [] };
  order.forEach((members, n) => {
    const inputs: number[] = [];
    for (const i of members) {
      const ways = waysAt.get(st[i].id) ?? [];
      inputs.push(st[i].id, st[i].stationBuilt !== false ? 1 : 0, st[i].stationOff ? 1 : 0, st[i].platform, stations[i * 3], stations[i * 3 + 1],
        flag(i) & (ST_ROOM | ST_PARK | ST_ONE | ST_ONE_NEG | ST_FLIP), stationShape[i * 4 + 2], ...stationLine.subarray(i * 2 * PLATFORM_POINTS, (i + 1) * 2 * PLATFORM_POINTS), ways.length, ...ways);
    }
    const was = allDirty ? undefined : cache.complexes.get(keys[n]);
    let p: Placed;
    if (was && sameNums(was.inputs, inputs) && !cellsMeet(was.cells, changed) && !moved.some((r) => rectNearBox(r, was.env, perM))) {
      p = was;
      for (const i of members) if (flag(i) & ST_ROOM) stations[i * 3 + 2] -= ST_ROOM;
      if (p.room) {
        setRoom(stations, index.get(p.room.id)!, p.room.side);
        addRect(rooms, p.room.rect, perM);
      }
    } else {
      p = placeOne(ctx, members, inputs, homeOf.get(keyOf(members[0])));
      if (was && !samePlaced(was, p)) pushRects(moved, was);
      if (!was || !samePlaced(was, p)) pushRects(moved, p);
    }
    placedNow.set(keys[n], p);
    if (p.lot) {
      const { rect: r, bays, flags, park0, parks } = p.lot;
      addRect(lotRects, r, perM);
      lots.push(r.x, r.y, r.ux * r.hl * perM, r.uy * r.hl * perM, r.hl, r.hd, bays, flags, index.get(park0)!, parks);
    }
  });
  cache.complexes = placedNow;
  return new Float32Array(lots);
}

interface PlaceCtx {
  st: { id: number; stationBuilt?: boolean; stationOff?: boolean; platform: number }[];
  stations: Float32Array;
  stationLine: Float32Array;
  stationShape: Float32Array;
  waysAt: Map<number, number[]>;
  buildings: ShapeGrid<LotOutline>;
  perM: number;
  cell: number;
  seg: number[];
  grid: CellLists;
  /** the lots and rooms placed so far this build */
  lotRects: ShapeGrid<Rect>;
  rooms: ShapeGrid<Rect>;
  /** by segment id, the last query that took it (`stamp`); `found`, that query's segments */
  seen: Uint32Array;
  stamp: number;
  found: number[];
}

/** A room or lot into its grid, by its centre and the furthest its corners reach (local); one with no way (a
 * zero vector) in every query. */
function addRect(grid: ShapeGrid<Rect>, r: Rect, perM: number) {
  const reach = r.ux || r.uy ? (r.hl + r.hd) * perM : Infinity;
  grid.add(r, r.x - reach, r.y - reach, r.x + reach, r.y + reach);
}

/** Station `i` draws its complex's waiting room, on `side`. */
function setRoom(stations: Float32Array, i: number, side: number) {
  const flag = () => Math.round(stations[i * 3 + 2]);
  stations[i * 3 + 2] += ST_ROOM;
  if (side > 0 && !(flag() & ST_FLIP)) stations[i * 3 + 2] += ST_FLIP;
  if (side < 0 && (flag() & ST_FLIP)) stations[i * 3 + 2] -= ST_FLIP;
}

function pushRects(out: Rect[], p: Placed) {
  if (p.room) out.push(p.room.rect);
  if (p.lot) out.push(p.lot.rect);
}

const sameRect = (a: Rect, b: Rect) => a.x === b.x && a.y === b.y && a.ux === b.ux && a.uy === b.uy && a.hl === b.hl && a.hd === b.hd;
function samePlaced(a: Placed, b: Placed): boolean {
  if (!a.room !== !b.room || !a.lot !== !b.lot) return false;
  if (a.room && (a.room.id !== b.room!.id || a.room.side !== b.room!.side || !sameRect(a.room.rect, b.room!.rect))) return false;
  return !a.lot || sameRect(a.lot.rect, b.lot!.rect);
}

/** Whether any of the changed segments' cell boxes (4 numbers each) meets `c`. */
function cellsMeet(c: number[], changed: number[]): boolean {
  for (let k = 0; k < changed.length; k += 4) if (changed[k] <= c[2] && c[0] <= changed[k + 2] && changed[k + 1] <= c[3] && c[1] <= changed[k + 3]) return true;
  return false;
}

/** Whether a room or lot could meet anything tried in box `b` (local; each try's half length plus half depth around
 * it): `rectsMeet`'s separating axes with their pad allow up to about sqrt(2) times its half sizes and twice the pad
 * along x or y, so twice both bounds it. */
function rectNearBox(r: Rect, b: number[], perM: number): boolean {
  const reach = 2 * (r.hl + r.hd + CLEAR_LOT_M) * perM;
  return r.x + reach >= b[0] && r.x - reach <= b[2] && r.y + reach >= b[1] && r.y - reach <= b[3];
}

/** One complex's waiting room and car park, against the grid and the rooms and lots placed before it; its stations'
 * flags are set for the room as `placeFacilities` says. */
function placeOne(ctx: PlaceCtx, members: number[], inputs: number[], home: LotOutline | undefined): Placed {
  const { st, stations, stationLine, stationShape, waysAt, buildings, perM, cell, seg, grid, lotRects, rooms } = ctx;
  const flag = (i: number) => Math.round(stations[i * 3 + 2]);
  const frame = (i: number) => platformFrame(i, stations, stationLine);
  const cells = [Infinity, Infinity, -Infinity, -Infinity], env = [Infinity, Infinity, -Infinity, -Infinity];
  const read = (cx0: number, cy0: number, cx1: number, cy1: number) => {
    cells[0] = Math.min(cells[0], cx0); cells[1] = Math.min(cells[1], cy0); cells[2] = Math.max(cells[2], cx1); cells[3] = Math.max(cells[3], cy1);
  };
  const cover = (x: number, y: number, ext: number) => {
    env[0] = Math.min(env[0], x - ext); env[1] = Math.min(env[1], y - ext); env[2] = Math.max(env[2], x + ext); env[3] = Math.max(env[3], y + ext);
  };
  // the segments in a box of cells, each once (a stamp per segment, since one long segment is in many cells)
  const segsIn = (cx0: number, cy0: number, cx1: number, cy1: number) => {
    const { seen, found } = ctx;
    found.length = 0;
    const stamp = ++ctx.stamp;
    for (let cx = cx0; cx <= cx1; cx++)
      for (let cy = cy0; cy <= cy1; cy++) {
        for (let e = grid.first(cellKey(cx, cy)); e >= 0; e = grid.next[e]) {
          const k = grid.seg[e];
          if (seen[k] !== stamp) { seen[k] = stamp; found.push(k); }
        }
      }
    return found;
  };
  const room = members.filter((i) => flag(i) & ST_ROOM), park = members.filter((i) => flag(i) & ST_PARK);
  // the name label's patch: below or right of the complex's middle, by its first station's way
  let lx = 0, ly = 0;
  for (const i of members) { lx += stations[i * 3]; ly += stations[i * 3 + 1]; }
  lx /= members.length; ly /= members.length;
  const f0 = frame(members[0]), below = Math.abs(f0.dx) > Math.abs(f0.dy);
  // local y runs south, as the screen's does
  const label: Rect = below
    ? { x: lx, y: ly + LABEL_BELOW.deep / 2 * perM, ux: 1, uy: 0, hl: LABEL_BELOW.halfWide, hd: LABEL_BELOW.deep / 2 }
    : { x: lx + LABEL_RIGHT.wide / 2 * perM, y: ly, ux: 1, uy: 0, hl: LABEL_RIGHT.wide / 2, hd: LABEL_RIGHT.halfDeep };
  const out: Placed = { inputs, cells, env, room: null, lot: null };

  // the waiting room: one box of the complex draws it
  for (const i of room) stations[i * 3 + 2] -= ST_ROOM;
  if (room.length) {
    let best: { i: number; side: number; bad: number } | null = null;
    for (const i of room) {
      const f = frame(i), own = new Set(waysAt.get(st[i].id) ?? []), mine = -1 - st[i].id;
      // + is to the left of the platform's way; it bends towards + when phi > 0, so its outside is -
      const outside = stationShape[i * 4 + 2] < 0 ? 1 : -1;
      // (T-3540: a station with one platform has its room on the platform's side, joined to it)
      for (const sg of (flag(i) & ST_ONE ? [flag(i) & ST_ONE_NEG ? -1 : 1] : [outside, -outside])) {
        let bad = 0;
        for (const a of ALONG_M) for (const c of ACROSS_M) {
          const px = f.mx + (f.dx * a + f.nx * sg * c) * perM, py = f.my + (f.dy * a + f.ny * sg * c) * perM;
          let worst = 0;
          const cx = Math.floor(px / cell), cy = Math.floor(py / cell);
          read(cx - 1, cy - 1, cx + 1, cy + 1);
          cover(px, py, 0);
          for (let u = -1; u <= 1; u++) for (let v = -1; v <= 1; v++) for (let e = grid.first(cellKey(cx + u, cy + v)); e >= 0; e = grid.next[e]) {
            const k = grid.seg[e], owner = seg[k * 5 + 4];
            if (owner === mine || own.has(owner)) continue;
            const d = distSeg(px, py, seg[k * 5], seg[k * 5 + 1], seg[k * 5 + 2], seg[k * 5 + 3]) / perM;
            if (owner < 0 && d < ROOM_NEAR_M) worst = 2;
            else if (owner >= 0 && d < ROOM_TRACK_M) worst = Math.max(worst, 1);
          }
          if (lotRects.near(px, py, 0).some((r) => inRect(r, px, py, perM, 0))) worst = 2;
          if (inRect(label, px, py, perM, 0)) worst = Math.max(worst, 1);
          bad += worst;
        }
        if (!best || bad < best.bad) best = { i, side: sg, bad };
      }
    }
    const b = best!;
    setRoom(stations, b.i, b.side);
    const f = frame(b.i);
    out.room = { id: st[b.i].id, side: b.side, rect: { x: f.mx + f.nx * b.side * 24 * perM, y: f.my + f.ny * b.side * 24 * perM, ux: f.dx, uy: f.dy, hl: 30, hd: 12 } };
    // (its own car park keeps clear of it too)
    addRect(rooms, out.room.rect, perM);
  }

  // the car park: a lot of its own beside the complex
  if (!park.length) return out;
  const longest = Math.max(...park.map((i) => st[i].platform));
  const full = Math.max(LOT_MIN_M, Math.min(LOT_MAX_M, longest)), bays = Math.min(park.length, MAX_BAYS);
  const sizes: [number, number][] = [[full, bays]];
  if (bays > 1) sizes.push([full, 1]);
  if (full * 0.6 >= LOT_SHORT_M) sizes.push([Math.max(LOT_SHORT_M, full * 0.6), 1]);
  // along every platform of the complex, those with park and ride first, longest first
  const along = [...members].sort((a, b) => Number(!!(flag(b) & ST_PARK)) - Number(!!(flag(a) & ST_PARK)) || st[b].platform - st[a].platform);
  // A try's badness, cut short once it reaches `limit` (the best so far: a try only wins with less, so the cut one
  // loses either way). What it would read is recorded first, so a kept complex depends on all of it.
  const badness = (r: Rect, limit: number) => {
    cover(r.x, r.y, (r.hl + r.hd) * perM);
    const ext = Math.hypot(r.hl, r.hd) * perM + CLEAR_BOX_M * perM;
    const cx0 = Math.floor((r.x - ext) / cell), cx1 = Math.floor((r.x + ext) / cell), cy0 = Math.floor((r.y - ext) / cell), cy1 = Math.floor((r.y + ext) / cell);
    read(cx0, cy0, cx1, cy1);
    let bad = rectsMeet(r, label, perM, 0) ? 1 : 0;
    // (the rooms' and lots' grid holds each at its centre and reach, so this finds every one rectsMeet could meet)
    const reach = (r.hl + r.hd + 2 * CLEAR_LOT_M) * perM;
    for (const o of rooms.near(r.x, r.y, reach)) if (rectsMeet(r, o, perM, CLEAR_LOT_M)) bad += 2;
    for (const o of lotRects.near(r.x, r.y, reach)) if (rectsMeet(r, o, perM, CLEAR_LOT_M)) bad += 2;
    if (bad >= limit) return bad;
    for (const k of segsIn(cx0, cy0, cx1, cy1)) {
      const owner = seg[k * 5 + 4], box = owner < 0;
      if (segHitsRect(seg[k * 5], seg[k * 5 + 1], seg[k * 5 + 2], seg[k * 5 + 3], r, perM, box ? CLEAR_BOX_M : CLEAR_TRACK_M) && (bad += box ? 3 : 2) >= limit) return bad;
    }
    for (const o of buildings.near(r.x, r.y, reach)) if (outlineMeets(o.corners, r, perM, CLEAR_LOT_M)) bad += 3;
    return bad;
  };
  let pick: { r: Rect; bays: number; bad: number } | null = null;
  // a complex in a station building: along the outside of its edges first, longest edge first
  if (home) {
    const c = home.corners, n = c.length / 2;
    const edges = Array.from({ length: n }, (_, k) => k).sort((a, b) => edgeLen(c, b) - edgeLen(c, a));
    let cx = 0, cy = 0;
    for (let k = 0; k < n; k++) { cx += c[2 * k] / n; cy += c[2 * k + 1] / n; }
    edge: for (const [len, nb] of sizes) {
      const hl = Math.floor(len / SPACE_W) * SPACE_W / 2, hd = nb * BAY_M / 2;
      for (const k of edges) {
        const j = (k + 1) % n, ex = c[2 * j] - c[2 * k], ey = c[2 * j + 1] - c[2 * k + 1], el = Math.hypot(ex, ey) || 1;
        const ux = ex / el, uy = ey / el;
        // the side away from the outline's middle
        const mx = (c[2 * k] + c[2 * j]) / 2, my = (c[2 * k + 1] + c[2 * j + 1]) / 2;
        const sg = (cx - mx) * -uy + (cy - my) * ux > 0 ? -1 : 1, off = (hd + CLEAR_LOT_M + 2) * perM;
        for (const a of [0, -0.3, 0.3]) {
          const t = a * el;
          const r: Rect = { x: mx + ux * t - uy * sg * off, y: my + uy * t + ux * sg * off, ux, uy, hl, hd };
          const bad = badness(r, pick?.bad ?? Infinity);
          if (!pick || bad < pick.bad) pick = { r, bays: nb, bad };
          if (bad === 0) break edge;
        }
      }
    }
  }
  if (!pick || pick.bad > 0) search: for (const [len, nb] of sizes) {
    const hl = Math.floor(len / SPACE_W) * SPACE_W / 2, hd = nb * BAY_M / 2;
    for (const gap of LOT_GAPS_M) for (const i of along) {
      const f = frame(i);
      // the side away from the label first
      const sides = (below ? f.ny : f.nx) > 0 ? [-1, 1] : [1, -1];
      for (const sg of sides) for (const a of [0, -0.25, 0.25, -0.5, 0.5]) {
        const off = gap + hd, t = a * st[i].platform;
        const r: Rect = { x: f.mx + (f.dx * t + f.nx * sg * off) * perM, y: f.my + (f.dy * t + f.ny * sg * off) * perM, ux: f.dx, uy: f.dy, hl, hd };
        const bad = badness(r, pick?.bad ?? Infinity);
        if (!pick || bad < pick.bad) pick = { r, bays: nb, bad };
        if (bad === 0) break search;
      }
    }
  }
  const p = pick!;
  const built = park.some((i) => st[i].stationBuilt !== false), on = park.some((i) => !st[i].stationOff);
  out.lot = { rect: p.r, bays: p.bays, flags: (built ? 0 : LOT_BLUEPRINT) | (on ? 0 : LOT_OFF), park0: st[park[0]].id, parks: park.length };
  return out;
}

/** Whether a point (local) is within `pad` m of a rectangle. */
function inRect(r: Rect, px: number, py: number, perM: number, pad: number) {
  const u = ((px - r.x) * r.ux + (py - r.y) * r.uy) / perM, v = (-(px - r.x) * r.uy + (py - r.y) * r.ux) / perM;
  return Math.abs(u) <= r.hl + pad && Math.abs(v) <= r.hd + pad;
}

/** Whether a segment (local) comes within `pad` m of a rectangle: clipped to it grown by `pad`. */
export function segHitsRect(x0: number, y0: number, x1: number, y1: number, r: Rect, perM: number, pad: number) {
  const tu = (x: number, y: number) => ((x - r.x) * r.ux + (y - r.y) * r.uy) / perM;
  const tv = (x: number, y: number) => (-(x - r.x) * r.uy + (y - r.y) * r.ux) / perM;
  const u0 = tu(x0, y0), v0 = tv(x0, y0), du = tu(x1, y1) - u0, dv = tv(x1, y1) - v0;
  const hu = r.hl + pad, hv = r.hd + pad;
  // the part of the segment (0-1) inside each pair of the rectangle's sides, without an array a call (T-3659)
  let lo = 0, hi = 1;
  if (Math.abs(du) < 1e-12) {
    if (u0 + hu < 0 || hu - u0 < 0) return false;
  } else {
    const a = (-hu - u0) / du, b = (hu - u0) / du;
    lo = Math.max(lo, Math.min(a, b)); hi = Math.min(hi, Math.max(a, b));
    if (lo > hi) return false;
  }
  if (Math.abs(dv) < 1e-12) return v0 + hv >= 0 && hv - v0 >= 0;
  const a = (-hv - v0) / dv, b = (hv - v0) / dv;
  return Math.max(lo, Math.min(a, b)) <= Math.min(hi, Math.max(a, b));
}

function edgeLen(c: ArrayLike<number>, k: number) {
  const j = (k + 1) % (c.length / 2);
  return Math.hypot(c[2 * j] - c[2 * k], c[2 * j + 1] - c[2 * k + 1]);
}

/** Whether a rectangle comes within `pad` m of a station building's floor (an outline, local x, y pairs). */
function outlineMeets(c: ArrayLike<number>, r: Rect, perM: number, pad: number) {
  const n = c.length / 2;
  let odd = false;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    if (segHitsRect(c[2 * j], c[2 * j + 1], c[2 * i], c[2 * i + 1], r, perM, pad)) return true;
    const ay = c[2 * i + 1], by = c[2 * j + 1];
    if ((ay > r.y) !== (by > r.y) && r.x < c[2 * i] + ((c[2 * j] - c[2 * i]) * (r.y - ay)) / (by - ay)) odd = !odd;
  }
  // no edge near it: inside the floor if its middle is
  return odd;
}

/** Whether two rectangles come within `pad` m of each other (separating axes). */
function rectsMeet(a: Rect, b: Rect, perM: number, pad: number) {
  const dx = (b.x - a.x) / perM, dy = (b.y - a.y) / perM;
  for (const [ax, ay] of [[a.ux, a.uy], [-a.uy, a.ux], [b.ux, b.uy], [-b.uy, b.ux]]) {
    const ra = a.hl * Math.abs(a.ux * ax + a.uy * ay) + a.hd * Math.abs(-a.uy * ax + a.ux * ay);
    const rb = b.hl * Math.abs(b.ux * ax + b.uy * ay) + b.hd * Math.abs(-b.uy * ax + b.ux * ay);
    if (Math.abs(dx * ax + dy * ay) > ra + rb + pad) return false;
  }
  return true;
}

function distSeg(px: number, py: number, x0: number, y0: number, x1: number, y1: number) {
  const dx = x1 - x0, dy = y1 - y0, l2 = dx * dx + dy * dy;
  const t = l2 > 0 ? Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / l2)) : 0;
  return Math.hypot(x0 + t * dx - px, y0 + t * dy - py);
}
