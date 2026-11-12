"""T-3273: each institution's Wikidata student count shared over all its campuses in the world, so a
parent whose campuses fall in several packs is counted once.

    python -I pipeline/world_run.py step campus_pools build [--into DIR]   the world table
    python -I pipeline/world_run.py step campus_pools show <Q>... [--into DIR]   an institution's pool

A site's institution is `assign`'s: its counted Wikidata id; else, the smallest of its counted parents
(T-3216) and the counted institutions whose label its name holds (`name_index`: a label of two words
and NAME_MIN characters, or NAME_MIN_CJK in Chinese, Japanese or Korean, at word boundaries, that no
other institution has and fewer than GENERIC_HOLDERS sister ids' names hold; the site within NAME_KM
of one of the institution's Wikidata-linked sites: Ming Chuan's untagged Taipei and Taoyuan campuses,
Kennesaw State's Marietta campus, whose Wikidata parent is the University System of Georgia). An
institution with a label naming it an open or distance university (tier0.OPEN_UNIVERSITY) is an open
one, its sites left out.

The table (`data/raw/tier0/campus/pools.json`, or DIR's), from the campus bands
(`campus_outlines.py fetch`) and the Wikidata caches: each institution's count less the counts of its
counted members with sites of their own (a system's or a federal university's count holds its
members', which they place themselves), each of its sites' point, hectares (CAMPUS_POINT_HA for a
point) and students (`allot`), its main campus, sites left out by name not among them; the name
index, the anchors and the open institutions `assign` needs. `tier0.university_sizes` gives an
institution in a pack its count less what `allot` gives its sites outside the pack, split over its
sites in the pack the same way. Reads only caches: the light pass's `wikidata_prefill.py` asks
Wikidata for the bands' ids and builds this at its end.
Data (c) OpenStreetMap contributors, ODbL; Wikidata CC0.
"""
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import shapely

import campus_outlines
import overpass

POOLS = os.path.join(campus_outlines.TIER0, "campus", "pools.json")
NAME_MIN, NAME_MIN_CJK = 12, 4
NAME_KM = 500.0
GENERIC_HOLDERS = 2
ANCHORS_MAX = 40  # an institution's Wikidata-linked points kept for the distance test
_CJK = re.compile("[぀-ヿ㐀-鿿가-힯]")
_loaded, _index = {}, {}


def world_sites():
    """kind, name, lat, lon, m2, wikidata of every university and college in the bands, as
    campus_outlines.rows builds a pack's (a point inside a mapped site of its kind left out)."""
    t0 = time.time()
    seen, out, pts = set(), [], []
    polys = {k: [] for k in campus_outlines.KINDS}
    for w, e in campus_outlines.BANDS:
        for k in campus_outlines.KINDS:
            n = campus_outlines.band_name(k, w, e)
            if not overpass.cache_path(n):
                raise SystemExit(f"{n}: not fetched (campus_outlines.py fetch)")
            for el in overpass.read_cache(n).get("elements", []):
                key = (el["type"], el["id"])
                tags = el.get("tags", {})
                kind = tags.get("amenity")
                if key in seen or kind not in polys:
                    continue
                seen.add(key)
                if el["type"] == "node":
                    pts.append((kind, tags.get("name", ""), el["lat"], el["lon"], tags.get("wikidata", "")))
                    continue
                g = overpass.polygon(el)
                if g is None:
                    continue
                try:
                    g = shapely.make_valid(g)
                    c = g.representative_point()
                    a = overpass.area_m2(g)
                except shapely.errors.GEOSException:
                    continue
                out.append((kind, tags.get("name", ""), c.y, c.x, a, tags.get("wikidata", "")))
                polys[kind].append(g)
    trees = {k: shapely.STRtree(v) for k, v in polys.items() if v}
    n_in = 0
    for kind, nm, lat, lon, wd in pts:
        if kind in trees and len(trees[kind].query(shapely.Point(lon, lat), predicate="within")):
            n_in += 1
            continue
        out.append((kind, nm, lat, lon, 0.0, wd))
    df = pd.DataFrame(out, columns=["kind", "name", "lat", "lon", "m2", "wikidata"])
    print(f"{len(df):,} sites in the bands ({n_in:,} points inside a site left out), {time.time() - t0:.0f} s", flush=True)
    return df


def _fold(s):
    return " ".join((s or "").split()).casefold()


def name_index(labels, insts, sites=None, parents=None):
    """{casefolded label: institution}: the institutions' labels long enough (two words at least
    outside CJK), no other institution's, and held by the names of sites of fewer than
    GENERIC_HOLDERS other Wikidata ids that are not its members (`sites`' name and wikidata;
    国際大学, "École polytechnique", also "Texas A&M University", which its sister universities'
    names hold)."""
    seen = {}
    for q in insts:
        for lab in labels.get(q, ()):
            lab = _fold(lab)
            cjk = bool(_CJK.search(lab))
            if len(lab) >= (NAME_MIN_CJK if cjk else NAME_MIN) and (cjk or " " in lab):
                seen.setdefault(lab, set()).add(q)
    names = {lab: next(iter(s)) for lab, s in seen.items() if len(s) == 1}
    if sites is not None:
        idx, holders = _Names(names), {}
        for nm, w in zip(sites.name, sites.wikidata):
            if w:
                for lab, q in idx.held_labels(nm):
                    if q != w and q not in (parents.get(w) or {}):
                        holders.setdefault(lab, set()).add(w)
        names = {lab: q for lab, q in names.items() if len(holders.get(lab, ())) < GENERIC_HOLDERS}
    return names


class _Names:
    """Which labels a name holds (a CJK label anywhere, another at word boundaries), by the labels'
    first four characters."""
    def __init__(self, names):
        self.by = {}
        for lab, q in names.items():
            self.by.setdefault(lab[:4], []).append((lab, q, bool(_CJK.search(lab))))

    def held_labels(self, name):
        s = _fold(name)
        got = []
        for i in range(max(len(s) - 3, 0)):
            for lab, q, cjk in self.by.get(s[i:i + 4], ()):
                j = i + len(lab)
                if s.startswith(lab, i) and (cjk or ((i == 0 or not s[i - 1].isalnum()) and (j == len(s) or not s[j].isalnum()))):
                    got.append((lab, q))
        return got

    def held(self, name):
        return {q for _, q in self.held_labels(name)}


def _near(lat, lon, pts, km=NAME_KM):
    p = np.asarray(pts, float).reshape(-1, 2)
    dy = (p[:, 0] - lat) * 111.2
    dx = (p[:, 1] - lon) * 111.2 * np.cos(np.radians(lat))
    return bool((dx * dx + dy * dy <= km * km).any())


def assign(df, counts, parents, table=None):
    """Each site's (institution, its count, how: "count", "parent Q", "named Q" or "area") and
    whether that institution is an open university, as lists. `df` has name, lat, lon, wikidata;
    without `table` only the Wikidata links (tier0.institution_of)."""
    import tier0
    got = [tier0.institution_of(q, counts, parents) for q in df.wikidata]
    inst, count, how = [g[0] for g in got], [g[1] for g in got], [g[2] for g in got]
    table = table or {}
    names, anchors, pools = table.get("names"), table.get("anchors", {}), table.get("pools", {})
    if names:
        idx = _index.get(id(names))
        if idx is None:
            idx = _index[id(names)] = _Names(names)
        for j, (nm, lat, lon) in enumerate(zip(df.name, df.lat, df.lon)):
            if how[j] == "count":
                continue
            best = None
            for q in idx.held(nm):
                if q in pools and q in anchors and _near(lat, lon, anchors[q]):
                    c = pools[q]["raw"]
                    if best is None or c < best[1]:
                        best = (q, c)
            if best and (not inst[j] or best[1] < count[j]):
                inst[j], count[j], how[j] = best[0], float(best[1]), f"named {best[0]}"
    open_ = set(table.get("open", ()))
    return inst, count, how, np.array([bool(q) and q in open_ for q in inst], bool)


def allot(count, ha, main):
    """An institution's count over its sites (hectares `ha`) by area, each site but its main campus
    (index `main`) at most the curve for its area (tier0.CAMPUS_SIZE_C x ha^CAMPUS_SIZE_POW, the
    median institution of that size), the rest to the main: a branch has more land a student than
    its main campus (Tamkang's Lanyang campus 34 ha to Tamsui's 20, National Taiwan University's
    Yunlin 53 ha), and so does a farm or a forest named after the university."""
    import tier0
    caps = tier0.CAMPUS_SIZE_C * np.asarray(ha, float) ** tier0.CAMPUS_SIZE_POW
    caps[main] = np.inf
    return tier0.campus_fill(count, ha, caps)[0]


def build(sites, counts, parents, labels):
    """The table from the world's sites and the Wikidata caches ({q: (students, year)},
    {q: {parent: students}}, {q: [labels]})."""
    import tier0
    own = {q for q in sites.wikidata if q and (counts.get(q) or (None,))[0]}
    sites = sites[~tier0.not_campus(sites.name)]
    link = [tier0.institution_of(q, counts, parents) for q in sites.wikidata]
    anchors, raw = {}, {}
    for (q, c, _), lat, lon in zip(link, sites.lat, sites.lon):
        if q:
            raw[q] = c
            a = anchors.setdefault(q, [])
            if len(a) < ANCHORS_MAX:
                a.append([round(lat, 4), round(lon, 4)])
    names = name_index(labels, set(raw), sites, parents)
    named = set(names.values())
    table = {"names": names, "anchors": {q: a for q, a in anchors.items() if q in named},
             "open": sorted(q for q, ls in labels.items() if any(tier0.OPEN_UNIVERSITY.search(lab) for lab in ls)),
             "pools": {q: {"raw": c} for q, c in raw.items()}}
    inst, count, how, opened = assign(sites, counts, parents, table)
    pools, own_site = {}, {}
    for q, c, o, lat, lon, m2, w in zip(inst, count, opened, sites.lat, sites.lon, sites.m2, sites.wikidata):
        if not q or o:
            continue
        p = pools.setdefault(q, {"raw": c, "sites": []})
        ha = m2 / 1e4 if m2 > 0 else tier0.CAMPUS_POINT_HA
        if w == q:
            own_site.setdefault(q, set()).add(len(p["sites"]))
        p["sites"].append([round(lat, 5), round(lon, 5), round(ha, 3)])
    # a counted id with sites of its own takes its count off each counted parent's: a system's or a
    # federal university's count holds its members' (a pair naming each other is left alone)
    less = {}
    for q in own:
        for p in parents.get(q) or {}:
            if p != q and p in pools and q not in (parents.get(p) or {}):
                less[p] = less.get(p, 0.0) + float(counts[q][0])
    for q, p in pools.items():
        p["less"] = less.get(q, 0.0)
        p["count"] = max(p["raw"] - p["less"], 0.0)
        ha = np.array([s[2] for s in p["sites"]])
        p["main"] = main = max(own_site.get(q) or range(len(ha)), key=lambda k: ha[k])
        for s, n in zip(p["sites"], allot(p["count"], ha, main)):
            s.append(round(float(n), 1))
    table["pools"] = pools
    n_named = sum(1 for h in how if h.startswith("named"))
    return table, n_named


def write(table, path=POOLS):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)
    _loaded.pop(path, None)


def load(path=POOLS):
    """The table, or {} when it has not been built (each pack then shares a count over its own sites)."""
    if path not in _loaded:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                _loaded[path] = json.load(f)
        else:
            _loaded[path] = {}
    return _loaded[path]


def outside(boundary, insts, table):
    """{institution: (its count for the world, the students `allot` gives its sites outside
    `boundary`, its main campus's (lat, lon) when that is inside, else None)} for the institutions
    in the table."""
    pools = table.get("pools", {})
    shapely.prepare(boundary)
    got = {}
    for q in insts:
        p = pools.get(q)
        if p is None:
            continue
        s = np.asarray(p["sites"], float).reshape(-1, 4)
        inb = shapely.contains_xy(boundary, s[:, 1], s[:, 0])
        m = p["main"]
        got[q] = (float(p["count"]), float(s[~inb, 3].sum()), (s[m, 0], s[m, 1]) if inb[m] else None)
    return got


def main(args):
    import tier0
    d = os.path.abspath(args[args.index("--into") + 1]) if "--into" in args else None
    args = [a for k, a in enumerate(args) if a != "--into" and (k == 0 or args[k - 1] != "--into")]
    cache = (lambda n: os.path.join(d, n)) if d else tier0.raw
    path = os.path.join(d, "pools.json") if d else POOLS
    if args[0] == "show":
        pools = load(path).get("pools", {})
        for q in args[1:]:
            p = pools.get(q)
            print(q, "not in the table" if p is None else
                  f"count {p['count']:,.0f} (raw {p['raw']:,.0f} less {p['less']:,.0f}), {len(p['sites'])} sites, "
                  f"{sum(s[2] for s in p['sites']):,.1f} ha, main #{p['main']}: "
                  + ", ".join(f"{s[0]:.3f},{s[1]:.3f} {s[2]:,.1f} ha {s[3]:,.0f}" for s in p["sites"][:12]))
        return
    t0 = time.time()
    caches = []
    for n in ("wikidata_students.json", "wikidata_parents.json", "wikidata_labels.json"):
        caches.append({})
        if os.path.exists(cache(n)):
            with open(cache(n), encoding="utf-8") as f:
                caches[-1] = json.load(f)
    counts = {q: tuple(v) for q, v in caches[0].items()}
    sites = world_sites()
    table, n_named = build(sites, counts, caches[1], caches[2])
    write(table, path)
    pools = table["pools"]
    ids = set(sites.wikidata) - {""}
    print(f"{len(pools):,} institutions ({sum(1 for p in pools.values() if p['less'] > 0):,} less their counted members, "
          f"{len(table['open'])} open universities left out), {n_named:,} sites joined by name; "
          f"{len(ids - set(counts)):,} of the bands' {len(ids):,} ids not in the counts cache; {path}; {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] not in ("build", "show"):
        sys.exit(__doc__)
    main(a)
