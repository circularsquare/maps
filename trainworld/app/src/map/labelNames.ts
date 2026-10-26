// The station names to draw, kept up to date from the stations that changed (T-1380). One name per
// distinct name per station complex (T-169): stations of one complex that share a name (three Times
// Sq platforms) get one label at their mean position, oriented like the member with the most lines;
// differently named members keep their own. Most important first: most lines, then constructed, then
// save order. The player's links count, as everywhere a complex is used (T-197, T-209).
//
// `labelGroups` is the whole list worked out from nothing. `NameIndex` keeps the same list, in the
// same order, and after an edit works out again only the names a changed station is in or joins:
// `plan` finds what changed without touching the index, `apply` brings the list up to date and says
// which names left it and entered it, at which places, for the labels and the shadow layer to follow
// (map/stationLabels.ts). No DOM here: checks/t1380-labels.mjs runs it in node.

import type { Complexes } from "../game/complexes";
import type { LineId, LineInput, StationId, StationInput } from "../game/types";
import { hasUpgrade, UPGRADE } from "../game/upgrades";

export interface NameGroup {
  /** stays with the name while its stations change */
  uid: number;
  name: string;
  lng: number;
  lat: number;
  heading: number;
  built: boolean;
  /** a member has a park and ride (T-1281): a P beside the name */
  park: boolean;
  /** the opening-day bunting's background (T-1434), "" for none */
  rib: string;
  /** distinct lines stopping at any of its stations */
  lines: number;
}

/** The whole list from nothing, the reference `NameIndex` must match. */
export function labelGroups(stations: readonly StationInput[], lines: ReadonlyMap<string, ReadonlySet<string>>, cx: Complexes, rib: ReadonlyMap<string, string> | null): Omit<NameGroup, "uid">[] {
  const groups = new Map<string, StationInput[]>();
  for (const s of stations) {
    const k = `${cx.of.get(s.id)}\u0000${s.name}`;
    const g = groups.get(k);
    if (g) g.push(s);
    else groups.set(k, [s]);
  }
  const out: Omit<NameGroup, "uid">[] = [];
  for (const members of groups.values()) {
    const ids = new Set<string>();
    let lng = 0, lat = 0, lead = members[0];
    for (const s of members) {
      lng += s.lng;
      lat += s.lat;
      for (const l of lines.get(s.id) ?? []) ids.add(l);
      if ((lines.get(s.id)?.size ?? 0) > (lines.get(lead.id)?.size ?? 0)) lead = s;
    }
    out.push({ name: lead.name, lng: lng / members.length, lat: lat / members.length, heading: lead.heading, built: members.some((s) => s.built), park: members.some((s) => hasUpgrade(s.upgrades, UPGRADE.parkRide)), rib: rib ? members.map((s) => rib.get(s.id)).find(Boolean) ?? "" : "", lines: ids.size });
  }
  return out.sort((a, b) => b.lines - a.lines || Number(b.built) - Number(a.built));
}

/** The stations' line sets, as `labelGroups` takes them. */
export function linesAt(lines: readonly LineInput[]): Map<string, Set<string>> {
  const out = new Map<string, Set<string>>();
  for (const l of lines) for (const s of l.stops) {
    const set = out.get(s);
    if (set) set.add(l.id);
    else out.set(s, new Set([l.id]));
  }
  return out;
}

interface Mem {
  s: StationInput;
  id: StationId;
  /** the first station of its complex in save order: names the complex whatever number it has */
  root: StationId;
  rib: string;
  /** its place in the list of stations last given */
  idx: number;
  group: Group | null;
  seen: number;
}

interface Group extends NameGroup {
  key: string;
  members: Mem[];
  first: Mem;
  inList: boolean;
}

export type NameOp = { kind: "remove" | "insert"; at: number; group: NameGroup };
/** `ops`: first the names that left, from the last index to the first, then those that entered, from
 * the first place to the last; each `at` is the place when its turn comes. */
export type NamePatch = { full: true } | { full: false; ops: NameOp[] };

interface Dirty {
  m: Mem | null;
  s: StationInput;
  root: StationId;
  rib: string;
}

export interface NamePlan {
  full: boolean;
  stations: readonly StationInput[];
  lines: readonly LineInput[];
  av: number;
  dirty: Dirty[];
  removed: Mem[];
  /** [the line as last seen, the line now]; one of them missing for a line new or gone */
  lineChanges: [LineInput | undefined, LineInput | undefined][];
}

/** More stations changed than this share of all: the list is made anew. */
const REBUILD_SHARE = 8;

const rootOf = (cx: Complexes, id: StationId): StationId => {
  const c = cx.of.get(id);
  return c === undefined ? id : cx.members[c][0];
};

export class NameIndex {
  /** the names, most important first */
  list: Group[] = [];
  private mems = new Map<StationId, Mem>();
  private groups = new Map<string, Group>();
  private stationLines = new Map<StationId, Set<LineId>>();
  private lineSeen = new Map<LineId, LineInput>();
  private ready = false;
  private av = 0;
  private stamp = 0;
  private uids = 0;
  private lastStations: readonly StationInput[] | null = null;
  private lastLines: readonly LineInput[] | null = null;
  private lastRoots: Complexes | null = null;
  private lastRib: ReadonlyMap<string, string> | null = null;

  /** What differs from what the index holds, or null for nothing. Leaves the index as it is. */
  plan(stations: readonly StationInput[], lines: readonly LineInput[], cx: Complexes, rib: ReadonlyMap<string, string> | null, av: number): NamePlan | null {
    const p: NamePlan = { full: false, stations, lines, av, dirty: [], removed: [], lineChanges: [] };
    if (!this.ready || av !== this.av) {
      p.full = true;
      return p;
    }
    // the same lists as last time, the same complexes and bunting: nothing to look at
    if (stations === this.lastStations && lines === this.lastLines && cx === this.lastRoots && rib === this.lastRib) return null;

    const lineDirty = new Set<StationId>();
    if (lines !== this.lastLines) {
      let found = 0;
      for (const l of lines) {
        const old = this.lineSeen.get(l.id);
        if (old) found++;
        if (old === l) continue;
        if (old && sameStops(old.stops, l.stops)) this.lineSeen.set(l.id, l);
        else p.lineChanges.push([old, l]);
      }
      if (this.lineSeen.size > found) {
        const ids = new Set(lines.map((l) => l.id));
        for (const [id, old] of this.lineSeen) if (!ids.has(id)) p.lineChanges.push([old, undefined]);
      }
      for (const [old, now] of p.lineChanges) {
        for (const id of old?.stops ?? []) lineDirty.add(id);
        for (const id of now?.stops ?? []) lineDirty.add(id);
      }
    }

    // the same stations, complexes and bunting: only the lines can have changed something
    if (stations === this.lastStations && cx === this.lastRoots && rib === this.lastRib) {
      for (const id of lineDirty) {
        const m = this.mems.get(id);
        if (m) p.dirty.push({ m, s: m.s, root: m.root, rib: m.rib });
      }
      if (p.dirty.length * REBUILD_SHARE > stations.length + REBUILD_SHARE) {
        p.full = true;
        return p;
      }
      if (!p.dirty.length && !p.lineChanges.length) {
        this.lastLines = lines;
        return null;
      }
      return p;
    }

    const stamp = ++this.stamp;
    let survivors = 0, prev = -1, reordered = false;
    for (let i = 0; i < stations.length; i++) {
      const s = stations[i];
      const root = rootOf(cx, s.id), ribc = rib?.get(s.id) ?? "";
      const m = this.mems.get(s.id);
      if (!m) {
        p.dirty.push({ m: null, s, root, rib: ribc });
        continue;
      }
      if (m.seen === stamp) {
        p.full = true;
        return p;
      }
      m.seen = stamp;
      survivors++;
      if (m.idx <= prev) reordered = true;
      prev = m.idx;
      const o = m.s;
      if ((o !== s && (o.name !== s.name || o.built !== s.built || o.lng !== s.lng || o.lat !== s.lat || o.heading !== s.heading || (o.upgrades ?? 0) !== (s.upgrades ?? 0))) || m.root !== root || m.rib !== ribc || lineDirty.has(s.id)) {
        p.dirty.push({ m, s, root, rib: ribc });
      }
    }
    if (this.mems.size > survivors) for (const m of this.mems.values()) if (m.seen !== stamp) p.removed.push(m);
    if (reordered || (p.dirty.length + p.removed.length) * REBUILD_SHARE > stations.length + REBUILD_SHARE) {
      p.full = true;
      return p;
    }
    if (!p.dirty.length && !p.removed.length && !p.lineChanges.length) {
      // only identities differ: remember the new lists so the next call can stop at once
      this.lastStations = stations;
      this.lastLines = lines;
      this.lastRoots = cx;
      this.lastRib = rib;
      for (let i = 0; i < stations.length; i++) this.mems.get(stations[i].id)!.s = stations[i];
      return null;
    }
    return p;
  }

  /** Bring the index up to date with a plan from `plan`. */
  apply(p: NamePlan, cx: Complexes, rib: ReadonlyMap<string, string> | null): NamePatch {
    this.lastStations = p.stations;
    this.lastLines = p.lines;
    this.lastRoots = cx;
    this.lastRib = rib;
    if (p.full) return this.rebuild(p, cx, rib);
    const ops: NameOp[] = [];

    // lines first: a station's line set is what its name's count comes from
    for (const [old, now] of p.lineChanges) {
      if (old) for (const id of old.stops) this.stationLines.get(id)?.delete(old.id);
      if (now) {
        for (const id of now.stops) {
          const set = this.stationLines.get(id);
          if (set) set.add(now.id);
          else this.stationLines.set(id, new Set([now.id]));
        }
        this.lineSeen.set(now.id, now);
      } else if (old) this.lineSeen.delete(old.id);
    }

    // the names to work out again, and which of them are in the list now
    const affected = new Set<Group>();
    for (const d of p.dirty) if (d.m?.group) affected.add(d.m.group);
    for (const m of p.removed) if (m.group) affected.add(m.group);
    for (const d of p.dirty) {
      const g = this.groups.get(`${d.root}\u0000${d.s.name}`);
      if (g) affected.add(g);
    }
    const gone: { at: number; g: Group }[] = [];
    for (const g of affected) {
      if (!g.inList) continue;
      const at = this.find(g);
      if (at < 0) return this.rebuild(p, cx, rib);
      gone.push({ at, g });
    }
    gone.sort((a, b) => b.at - a.at);
    for (const x of gone) {
      this.list.splice(x.at, 1);
      x.g.inList = false;
      ops.push({ kind: "remove", at: x.at, group: x.g });
    }

    // stations out of their names, into theirs
    const touched = new Set<Group>(affected);
    for (const m of p.removed) {
      if (m.group) removeMember(m.group, m);
      m.group = null;
      this.mems.delete(m.id);
    }
    for (const d of p.dirty) {
      let m = d.m;
      if (m) {
        if (m.group) removeMember(m.group, m);
        m.s = d.s;
        m.root = d.root;
        m.rib = d.rib;
      } else {
        m = { s: d.s, id: d.s.id, root: d.root, rib: d.rib, idx: -1, group: null, seen: 0 };
        this.mems.set(m.id, m);
      }
      const key = `${d.root}\u0000${d.s.name}`;
      let g = this.groups.get(key);
      if (!g) this.groups.set(key, (g = { uid: ++this.uids, key, members: [], first: m, inList: false, name: "", lng: 0, lat: 0, heading: 0, built: false, park: false, rib: "", lines: 0 }));
      g.members.push(m);
      m.group = g;
      touched.add(g);
    }
    // places move only when a station came or went
    if (p.removed.length || p.dirty.some((d) => !d.m)) {
      for (let i = 0; i < p.stations.length; i++) {
        const m = this.mems.get(p.stations[i].id)!;
        m.idx = i;
        m.s = p.stations[i];
      }
    }

    const enter: Group[] = [];
    for (const g of touched) {
      if (!g.members.length) {
        this.groups.delete(g.key);
        continue;
      }
      g.members.sort((a, b) => a.idx - b.idx);
      this.settle(g);
      enter.push(g);
    }
    enter.sort(compare);
    for (const g of enter) {
      const at = this.lowerBound(g);
      this.list.splice(at, 0, g);
      g.inList = true;
      ops.push({ kind: "insert", at, group: g });
    }
    return { full: false, ops };
  }

  private rebuild(p: NamePlan, cx: Complexes, rib: ReadonlyMap<string, string> | null): NamePatch {
    this.mems = new Map();
    this.groups = new Map();
    this.stationLines = linesAt(p.lines);
    this.lineSeen = new Map(p.lines.map((l) => [l.id, l]));
    p.stations.forEach((s, i) => {
      const root = rootOf(cx, s.id), m: Mem = { s, id: s.id, root, rib: rib?.get(s.id) ?? "", idx: i, group: null, seen: 0 };
      this.mems.set(s.id, m);
      const key = `${root}\u0000${s.name}`;
      let g = this.groups.get(key);
      if (!g) this.groups.set(key, (g = { uid: ++this.uids, key, members: [], first: m, inList: true, name: "", lng: 0, lat: 0, heading: 0, built: false, park: false, rib: "", lines: 0 }));
      g.members.push(m);
      m.group = g;
    });
    for (const g of this.groups.values()) this.settle(g);
    this.list = [...this.groups.values()].sort(compare);
    this.av = p.av;
    this.ready = true;
    return { full: true };
  }

  /** A name's fields from its members, which are in save order. */
  private settle(g: Group) {
    const ms = g.members;
    let lng = 0, lat = 0, lead = ms[0];
    const ids = new Set<LineId>();
    for (const m of ms) {
      lng += m.s.lng;
      lat += m.s.lat;
      const ls = this.stationLines.get(m.id);
      if (ls) for (const l of ls) ids.add(l);
      if ((ls?.size ?? 0) > (this.stationLines.get(lead.id)?.size ?? 0)) lead = m;
    }
    g.first = ms[0];
    g.name = lead.s.name;
    g.lng = lng / ms.length;
    g.lat = lat / ms.length;
    g.heading = lead.s.heading;
    g.built = ms.some((m) => m.s.built);
    g.park = ms.some((m) => hasUpgrade(m.s.upgrades, UPGRADE.parkRide));
    g.rib = ms.find((m) => m.rib)?.rib ?? "";
    g.lines = ids.size;
  }

  /** Where `g` is, or where it would go, in the list. */
  private lowerBound(g: Group): number {
    let lo = 0, hi = this.list.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (compare(this.list[mid], g) < 0) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  private find(g: Group): number {
    const at = this.lowerBound(g);
    return this.list[at] === g ? at : -1;
  }
}

/** The list's order: most lines, then constructed, then the first station in save order. */
function compare(a: Group, b: Group): number {
  return b.lines - a.lines || Number(b.built) - Number(a.built) || a.first.idx - b.first.idx;
}

function removeMember(g: Group, m: Mem) {
  const at = g.members.indexOf(m);
  if (at >= 0) g.members.splice(at, 1);
}

function sameStops(a: readonly string[], b: readonly string[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}
