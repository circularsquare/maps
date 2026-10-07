"""Level 4 for Bangladesh: unions, paurashavas and city corporation wards.

Called from fetch.py after levels 1-3 are built. Writes
  data/bangladesh/level4.gpkg          polygons (EPSG:4326), one per level-4 unit
  data/bangladesh/level4_lineage.csv   every 2022 row and 2011 union/ward behind each unit
and returns the population rows (code, 4, year, pop) for population.csv.

Counts
  2022  Union Statistics (BBS, May 2025) Table U 01 for the 4,584 unions;
        National Report Vol I Table P34 for the 327 paurashavas and Table P32 for
        the city corporation wards (level4_tables.py parses all three).
        Whatever an upazila's P35 figure holds beyond its unions and
        paurashavas (cantonments and other non-union areas, 75,158 people in
        all) becomes an "Other areas" member of that upazila.
  2011  The 5,161 unions/wards of the 2011 census (USCB transcription), each
        already placed on one level-3 unit by fetch.py (unions_2011_to_v03.csv).

Polygons
  No open polygons of the 2022 unions exist: COD-AB v03 stops at upazila, OSM
  has almost no union boundaries in Bangladesh, and geoBoundaries' ADM4 is the
  same 2011-era BBS/OCHA layer as the USCB one. So level 4 is drawn from the
  2011 union/ward polygons (USCB Bangladesh.gdb), and a level-4 unit is the
  smallest set of 2011 and 2022 units that can be matched to each other:
    - matched by name inside the level-3 unit (most unions: one to one);
    - a union split since 2011 ("Bharella" -> "Bharella Uttar" + "Bharella
      Dakshin") keeps its 2011 polygon and gets both 2022 rows;
    - whatever is left unmatched on both sides in a level-3 unit is merged;
    - a 2011 union left over alone (absorbed by a paurashava or a neighbour)
      joins the paurashava if the upazila has one, else its longest-border
      neighbour; a 2022 union left over alone (carved out of neighbours) joins
      the unit whose 2022/2011 ratio it brings closest to the upazila's.
    - neighbours whose ratios are off in opposite directions are merged
      (rebalance()), so a new union the names could not place does not show
      a thirteen-fold rise beside a halving.
  City corporations: Barishal, Chattogram, Khulna, Rajshahi and Sylhet by ward,
  matched to the 2011 ward polygons by ward number. Dhaka North and South by
  thana (Table P33), because the 2011 ward numbers are the undivided city's and
  were renumbered at the split in a way no open table records; the 2011 side is
  the 2011 metropolitan thana. Gazipur, Narayanganj, Cumilla, Mymensingh and
  Rangpur were paurashavas and unions in 2011: their 2022 thanas are matched to
  those by name (and HAND_LINKS), which splits Gazipur into five units and
  leaves the others whole or nearly.
  Every unit is then fitted to its v03 level-3 polygon: clipped to it, and the
  parts of the level-3 polygon no 2011 polygon covers (the two layers are a
  few hundred metres apart, and big rivers) go to the nearest unit.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import difflib
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.ops import unary_union

import level4_tables
import report

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "helper1m/data/bangladesh"
USCB_GDB = REPO / "religiondots/data/raw/bd/Bangladesh.gdb"
LINEAGE3 = DATA / "unions_2011_to_v03.csv"
OUT_GPKG = DATA / "level4.gpkg"
OUT_LIN = DATA / "level4_lineage.csv"
METRIC = "EPSG:3106"  # Gulshan 303 / Bangladesh TM

# City corporations whose 2011 wards have polygons; the rest stay one unit.
WARD_CCS = {"Barishal", "Chattogram", "Khulna", "Rajshahi", "Sylhet"}

# ---------------------------------------------------------------- names

DIRS = {"uttar": "N", "north": "N", "dakshin": "S", "dakkhin": "S", "dakhin": "S",
        "dakshinbhag": None, "south": "S", "purba": "E", "east": "E", "paschim": "W",
        "pashchim": "W", "pachim": "W", "west": "W", "maddho": "M", "madhya": "M"}
DROP = {"union", "paurashava", "pourashava", "model", "bazar", "sadar"}


def words(s):
    s = s.lower().replace("(", " ").replace(")", " ")
    return [w for w in re.split(r"[^a-z]+", s) if w]


def canon(s, stem=False):
    """(direction tokens, sorted other tokens) of a union name."""
    d, rest = [], []
    for w in words(s):
        if DIRS.get(w):
            d.append(DIRS[w])
        elif w not in DROP:
            rest.append(w)
    return ("" if stem else "".join(sorted(d))), "".join(sorted(rest))


def skel(s):
    s = s.replace("ph", "f").replace("w", "o").replace("v", "b").replace("z", "j")
    s = s.replace("y", "i").replace("ksh", "kk").replace("sh", "s").replace("ch", "s")
    s = re.sub(r"h", "", s)
    s = re.sub(r"[aeiou]+", "a", s)
    return re.sub(r"(.)\1+", r"\1", s)


def variants(s):
    """The name, the name without its bracketed part, and the bracketed part:
    2011 "Bitghar (Tiara)" is 2022 "Bitghar"; "Dharmapur (Pananagar)" is
    "Pananagar"."""
    out = [s]
    m = re.match(r"^(.*?)\((.*?)\)(.*)$", s)
    if m:
        out += [m.group(1) + m.group(3), m.group(2)]
    return out


def score(a, b, stem=False):
    best = 0.0
    for x in variants(a):
        for y in variants(b):
            dx, rx = canon(x, stem)
            dy, ry = canon(y, stem)
            if not rx or not ry:
                continue
            if dx != dy and dx and dy:
                continue  # Uttar X is not Dakshin X
            pen = 1.0 if dx == dy else 0.97  # one side has a direction, the other none
            if rx == ry:
                s = 1.0
            else:
                s = 0.99 * max(difflib.SequenceMatcher(None, rx, ry).ratio(),
                               difflib.SequenceMatcher(None, skel(rx), skel(ry)).ratio())
            best = max(best, s * pen)
    return best


def zila_norm(s):
    return re.sub(r"[^a-z]", "", s.lower())


# ---------------------------------------------------------------- inputs

def load_2022(v3):
    t = pd.DataFrame(level4_tables.parse())
    v3k = v3.assign(k=v3.adm2_name.map(zila_norm) + "|" + v3.adm3_name.map(zila_norm))
    u = t[(t.table == "U01") & (t.name != "") & (t.name != "TOTAL")].copy()
    u["k"] = u.district.map(zila_norm) + "|" + u.upazila.map(zila_norm)
    u = u.merge(v3k[["k", "adm3_pcode"]], on="k", how="left")
    assert u.adm3_pcode.notna().all(), u[u.adm3_pcode.isna()]
    up = t[(t.table == "U01") & (t.name == "")]
    assert len(up) == (~v3.adm3_name.str.endswith("City Corporation")).sum()
    p = t[(t.table == "P34") & (t.name != "TOTAL")].copy()
    w = t[(t.table == "P32") & (t.name != "")].copy()
    return u, p, w, t


def load_2011():
    lin = pd.read_csv(LINEAGE3, keep_default_na=False)
    g = gpd.read_file(USCB_GDB, layer="BD_GEOG_ADM4_2011_uscb_202107")
    g = g[["GEO_MATCH", "NSO_NAME", "geometry"]].to_crs(METRIC)
    g["geometry"] = g.geometry.buffer(0)
    lin = lin.merge(g, on="GEO_MATCH", how="left")
    assert lin.NSO_NAME.notna().all()
    lin["kind"] = lin.NSO_NAME.str.extract(r"(Paurashava|Ward|Cantonment|Forest)",
                                           expand=False).fillna("Union")
    lin["ward"] = pd.to_numeric(lin.NSO_NAME.str.extract(r"Ward No-?\s*(\d+)", expand=False))
    return gpd.GeoDataFrame(lin, geometry="geometry", crs=METRIC)


# ---------------------------------------------------------------- paurashavas

def place_paurashavas(p, u, l11, v3, p35):
    """adm3_pcode for every 2022 paurashava (P34 gives only its district).
    By name against the 2011 paurashavas (whose level-3 unit is known), then
    against the upazila names of its district, then by room: each upazila's
    P35 figure minus its unions is what its paurashavas (and any cantonment)
    hold, and a paurashava not placed by name goes where it fits.
    A paurashava that 2011 split over two upazilas (Bogura: Bogura Sadar and
    Shajahanpur) is split the same way: by its 2011 parts, then trimmed so no
    upazila holds more than its room."""
    v3d = v3.assign(dk=v3.adm2_name.map(zila_norm))
    p = p.assign(dk=p.district.map(zila_norm))
    assert set(p.dk) <= set(v3d.dk), set(p.dk) - set(v3d.dk)
    adm2_of = v3.set_index("adm3_pcode").adm2_name.map(zila_norm)
    old = l11[l11.kind == "Paurashava"].copy()
    old["dk"] = old.adm3_pcode.map(adm2_of)
    room = (p35 - u.groupby("adm3_pcode")["pop"].sum()).to_dict()
    out, how, parts = {}, {}, {}
    for r in p.itertuples():
        cand = old[old.dk == r.dk]
        sc = [(score(r.name, n), c, int(q)) for n, c, q in
              zip(cand.NSO_NAME, cand.adm3_pcode, cand.pop_2011)]
        sc = [x for x in sc if x[0] >= 0.85]
        if sc:
            out[r.Index], how[r.Index] = max(sc)[1], "2011 paurashava"
            top = max(x[0] for x in sc)
            pp = {c: q for s, c, q in sc if s >= top - 0.02}
            if len(pp) > 1:
                parts[r.Index] = pp
            continue
        ups = v3d[(v3d.dk == r.dk) & ~v3d.adm3_name.str.endswith("City Corporation")]
        sc = [(score(r.name, n), c) for n, c in zip(ups.adm3_name, ups.adm3_pcode)]
        sc = [x for x in sc if x[0] >= 0.85]
        if sc:
            out[r.Index], how[r.Index] = max(sc)[1], "upazila name"
            continue
        out[r.Index], how[r.Index] = None, ""
    # what is left: by room (the only upazilas of the district that can hold it)
    used = pd.Series({i: p.loc[i, "pop"] for i in out if out[i]}).groupby(
        pd.Series({i: out[i] for i in out if out[i]})).sum()
    for i in [i for i in out if out[i] is None]:
        r = p.loc[i]
        ups = v3d[(v3d.dk == r.dk) & ~v3d.adm3_name.str.endswith("City Corporation")].adm3_pcode
        left = {c: room.get(c, 0) - used.get(c, 0) for c in ups}
        fits = {c: v - r["pop"] for c, v in left.items() if v - r["pop"] >= 0}
        assert fits, (r["name"], left)
        c = min(fits, key=fits.get)
        out[i], how[i] = c, "room in upazila"
        used[c] = used.get(c, 0) + r["pop"]
    p = p.assign(adm3_pcode=pd.Series(out), placed_by=pd.Series(how))
    # split paurashavas: share by 2011 parts, then move any excess over an
    # upazila's room to the other parts
    extra = []
    for i, pp in parts.items():
        r = p.loc[i]
        others = p[(p.index != i)].groupby("adm3_pcode")["pop"].sum()
        free = {c: room.get(c, 0) - others.get(c, 0) for c in pp}
        tot = sum(pp.values())
        alloc = {c: round(r["pop"] * q / tot) for c, q in pp.items()}
        alloc[max(pp, key=pp.get)] += r["pop"] - sum(alloc.values())
        for c in alloc:
            if alloc[c] > free[c]:
                excess = alloc[c] - free[c]
                alloc[c] = free[c]
                for d in alloc:
                    if d != c:
                        take = min(excess, free[d] - alloc[d])
                        alloc[d] += take
                        excess -= take
                assert excess == 0, (r["name"], alloc, free)
        for k, (c, v) in enumerate(alloc.items()):
            row = r.copy()
            row["pop"], row["adm3_pcode"] = v, c
            row["placed_by"] = f"2011 paurashava, split over {len(alloc)} upazilas by room"
            extra.append(row)
    p = pd.concat([p.drop(index=list(parts)), pd.DataFrame(extra)], ignore_index=True)
    # every upazila must hold its paurashavas, with a small remainder at most
    rem = pd.Series(room) - p.groupby("adm3_pcode")["pop"].sum().reindex(list(room)).fillna(0)
    assert (rem >= 0).all(), rem[rem < 0]
    return p, rem


# ---------------------------------------------------------------- matching

class Groups:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, k):
        while self.parent[k] != k:
            self.parent[k] = self.parent[self.parent[k]]
            k = self.parent[k]
        return k

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)

    def sets(self):
        out = {}
        for k in self.parent:
            out.setdefault(self.find(k), []).append(k)
        return list(out.values())


def greedy(pairs, threshold):
    """One-to-one by best score first."""
    ua, ub, out = set(), set(), []
    for s, a, b in sorted(pairs, key=lambda x: -x[0]):
        if s < threshold:
            break
        if a in ua or b in ub:
            continue
        ua.add(a), ub.add(b)
        out.append((a, b, s))
    return out


def match_unit(new, old, how_log, neighbours, ratio_target, hand=()):
    """new: list of dict(key, name, kind, pop); old: list of dict(key, name, kind,
    pop, ward). Returns list of member-key lists. Every set has at least one
    2011 member (it carries the polygon). `hand`: (2022 name, [2011 names])
    links decided by hand, applied first."""
    keys = [n["key"] for n in new] + [o["key"] for o in old]
    G = Groups(keys)
    N = {n["key"]: n for n in new}
    O = {o["key"]: o for o in old}
    matched = set()

    def link(a, b, why):
        G.union(a, b)
        matched.update([a, b])
        how_log[a] = how_log.get(a) or why
        how_log[b] = how_log.get(b) or why

    hand_old = set()
    for nname, onames in hand:
        nk = [n["key"] for n in new if n["name"] == nname]
        assert len(nk) == 1, (nname, [n["name"] for n in new])
        for on in onames:
            ok = [o["key"] for o in old if o["name"] == on]
            assert len(ok) == 1, (on, [o["name"] for o in old])
            # only the 2022 side counts as matched: the 2011 unit still gets
            # its own same-named 2022 unit below, which joins the same set
            G.union(nk[0], ok[0])
            matched.add(nk[0])
            hand_old.add(ok[0])
            how_log[nk[0]] = "by hand (HAND_LINKS)"
            how_log.setdefault(ok[0], "by hand (HAND_LINKS)")

    # wards by number
    nw = {n["ward"]: n["key"] for n in new if n.get("ward") is not None}
    for o in old:
        if o.get("ward") is not None and o["ward"] in nw:
            link(nw[o["ward"]], o["key"], "ward number")
    # paurashava to paurashava, union to union, by name
    for kind, th in (("Paurashava", 0.8), ("Union", 0.8)):
        a = [n for n in new if n["kind"] == kind and n["key"] not in matched]
        b = [o for o in old if o["kind"] == kind and o["key"] not in matched]
        for x, y, s in greedy([(score(n["name"], o["name"]), n["key"], o["key"])
                               for n in a for o in b], th):
            link(x, y, "name" if s == 1 else f"name ~{s:.2f}")
    matched |= hand_old
    # cantonments and other non-union areas together
    oth_new = [n["key"] for n in new if n["kind"] == "Other"]
    oth_old = [o["key"] for o in old if o["kind"] in ("Cantonment", "Forest")]
    for x in oth_new:
        for y in oth_old:
            link(x, y, "cantonment / other area")
    # splits: an unmatched 2022 union whose name, without its direction, is a
    # 2011 union's (and the other way round)
    for x in [n for n in new if n["key"] not in matched and n["kind"] == "Union"]:
        sc = [(score(x["name"], o["name"], stem=True), o["key"]) for o in old if o["kind"] == "Union"]
        sc = [s for s in sc if s[0] >= 0.88]
        if sc:
            link(x["key"], max(sc)[1], "split since 2011")
    for y in [o for o in old if o["key"] not in matched and o["kind"] == "Union"]:
        sc = [(score(n["name"], y["name"], stem=True), n["key"]) for n in new if n["kind"] == "Union"]
        sc = [s for s in sc if s[0] >= 0.88]
        if sc:
            link(max(sc)[1], y["key"], "merged since 2011")
    # everything still unmatched on both sides: one set
    ln = [k for k in N if k not in matched]
    lo = [k for k in O if k not in matched]
    if ln and lo:
        for k in ln[1:] + lo:
            G.union(k, ln[0])
        for k in ln + lo:
            how_log[k] = "left over: merged"
        ln, lo = [], []
    sets = G.sets()

    def old_pop(s):
        return sum(O[k]["pop"] for k in s if k in O)

    def new_pop(s):
        return sum(N[k]["pop"] for k in s if k in N)

    # a 2011 unit alone: to the paurashava's set, else its longest-border neighbour
    for k in lo:
        own = [s for s in sets if k in s][0]
        targets = [s for s in sets if s is not own and any(m in N for m in s)]
        paur = [s for s in targets if any(N.get(m, {}).get("kind") == "Paurashava" for m in s)]
        if paur:
            dest, why = paur[0], "left over 2011: into the paurashava"
        else:
            nb = neighbours(k, [[m for m in s if m in O] for s in targets])
            dest, why = targets[nb], "left over 2011: into longest-border neighbour"
        dest.extend(own)
        sets.remove(own)
        how_log[k] = why
    # a 2022 unit alone: where it brings the 2022/2011 ratio closest to the target
    for k in ln:
        own = [s for s in sets if k in s][0]
        targets = [s for s in sets if s is not own and old_pop(s) > 0]
        if not targets:
            raise RuntimeError(f"no 2011 unit to attach {k} to")
        best = min(targets, key=lambda s: abs((new_pop(s) + N[k]["pop"]) / old_pop(s)
                                              - ratio_target) - abs(new_pop(s) / old_pop(s) - ratio_target))
        best.extend(own)
        sets.remove(own)
        how_log[k] = "left over 2022: where the ratio fits best"
    # a set with 2011 members only (a 2011 union with no 2022 people, e.g. a
    # forest range) or 2022 only must not survive
    for s in list(sets):
        if not any(m in N for m in s) or not any(m in O for m in s):
            raise RuntimeError(f"one-sided set {s}")
    return sets


HIGH, LOW, TINY = 1.5, 0.7, 500


def rebalance(sets, N, O, geo, target, how_log, allow_far=True):
    """Merge neighbouring sets whose 2022/2011 ratios are off in opposite
    directions. A union carved out of its neighbours after 2011 that the names
    could not tie to them shows as one set growing several-fold next to sets
    that shrank (Ghatail: Lakkhindar, Sagardighi and Sangrampur x13, beside
    Rasulpur x0.42); so does a paurashava that took in parts of the unions
    around it. Real growth (Savar, Ashulia) has no shrinking neighbour and is
    left alone. Also folds in sets too small to carry a trend (< TINY people
    in either year: a forest range, a cantonment fragment)."""
    sets = [list(s) for s in sets]

    def tag(k, what):
        if what not in how_log.get(k, ""):
            how_log[k] = (how_log.get(k, "") + "; " + what).lstrip("; ")

    def pops(s):
        return (sum(O[k]["pop"] for k in s if k in O), sum(N[k]["pop"] for k in s if k in N))

    def dev(s):
        a, b = pops(s)
        return np.inf if a == 0 else (b / a) / target

    def shape(s):
        return unary_union([geo[k] for k in s if k in O]).buffer(30)

    def border(s, t):
        return shape(s).intersection(shape(t)).area

    changed = True
    while changed and len(sets) > 1:
        changed = False
        for s in sorted(sets, key=lambda s: min(pops(s))):
            if min(pops(s)) < TINY:
                cand = [t for t in sets if t is not s]
                b = [border(s, t) for t in cand]
                t = cand[int(np.argmax(b))] if max(b) > 0 else max(cand, key=lambda t: sum(pops(t)))
                t.extend(s)
                sets.remove(s)
                for k in s:
                    tag(k, "folded in: too small")
                changed = True
                break
        if changed:
            continue
        bad = [s for s in sets if dev(s) > HIGH or dev(s) < LOW]
        best = None
        # neighbours first; a pair that does not touch only when nothing else
        # is left (the 2011 polygons of a union and the paurashava that took
        # its centre need not share a border)
        for adjacent in ((True, False) if allow_far else (True,)):
            for s in bad:
                hi = dev(s) > HIGH
                for t in sets:
                    if t is s or not bool(dev(t) < LOW if hi else dev(t) > HIGH):
                        continue
                    if (border(s, t) > 0) != adjacent:
                        continue
                    before = max(abs(np.log(dev(s))), abs(np.log(dev(t))))
                    after = abs(np.log(dev(s + t)))
                    if after < before and (best is None or after < best[0]):
                        best = (after, s, t)
            if best:
                break
        if best:
            _, s, t = best
            t.extend(s)
            sets.remove(s)
            for k in t:
                tag(k, "merged: neighbours with opposite ratios")
            changed = True
    return sets


# ---------------------------------------------------------------- geometry

def fit_to_unit(unit_poly, pieces, spacing=150.0):
    """Clip each set's 2011 polygon to the level-3 polygon and give every gap
    to the nearest set (Voronoi cells of points along the clipped borders).
    pieces: {set id: geometry or None}. Returns {set id: geometry}.
    All overlay work is on a 0.5 m grid, which keeps GEOS out of topology
    trouble with the two layers' near-coincident edges."""
    G = 0.5
    unit_poly = polygonal(shapely.set_precision(shapely.make_valid(unit_poly), G))
    out, taken = {}, None
    for k, g in pieces.items():
        c = None
        if g is not None and not g.is_empty:
            g = polygonal(shapely.set_precision(shapely.make_valid(g), G))
            c = polygonal(shapely.intersection(g, unit_poly, grid_size=G))
            if taken is not None and not c.is_empty:
                c = polygonal(shapely.difference(c, taken, grid_size=G))
        if c is not None and not c.is_empty and c.area > 1.0:
            out[k] = polygonal(c)
            taken = out[k] if taken is None else polygonal(shapely.union(taken, out[k], grid_size=G))
        else:
            out[k] = None
    empty = [k for k, g in out.items() if g is None]
    gap = polygonal(shapely.difference(unit_poly, taken, grid_size=G)) if taken is not None else unit_poly
    # a set whose 2011 polygon lies wholly outside: give it the gap nearest its old place
    for k in empty:
        g = pieces[k]
        parts = [p for p in getattr(gap, "geoms", [gap]) if p.area > 1.0]
        if not parts:
            raise RuntimeError("set with no area and no gap to take")
        near = min(parts, key=lambda p: p.distance(g.centroid))
        out[k] = near
        gap = polygonal(shapely.difference(gap, near, grid_size=G))
    if gap.is_empty or gap.area < 1.0:
        return out
    pts, lab = [], []
    for k, g in out.items():
        for ring in shapely.get_parts(shapely.boundary(g)):
            n = max(4, int(ring.length / spacing))
            d = np.linspace(0, ring.length, n, endpoint=False)
            pts.append(shapely.line_interpolate_point(ring, d))
            lab += [k] * n
    pts = np.concatenate(pts)
    cells = shapely.voronoi_polygons(shapely.multipoints(pts), extend_to=unit_poly.buffer(1000))
    cells = np.array(cells.geoms)
    tree = shapely.STRtree(pts)
    # each cell contains exactly one generator
    ci, pi = tree.query(cells, predicate="contains")
    owner = {}
    for c, p in zip(ci, pi):
        owner[c] = lab[p]
    by = {}
    for c, cell in enumerate(cells):
        if c in owner:
            by.setdefault(owner[c], []).append(cell)
    for k, cl in by.items():
        add = polygonal(shapely.intersection(shapely.union_all(cl, grid_size=G), gap, grid_size=G))
        if not add.is_empty:
            out[k] = polygonal(shapely.union(out[k], add, grid_size=G))
    return out


def polygonal(g):
    """Drop stray lines and points that clipping leaves behind."""
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    parts = [p for p in getattr(g, "geoms", [g]) if p.geom_type in ("Polygon", "MultiPolygon")]
    return unary_union(parts)


# ---------------------------------------------------------------- main

def build(v1, v2, v3, p22_l3, p11_l3):
    """p22_l3 / p11_l3: {adm3_pcode: pop} as fetch.py computed for level 3."""
    u, p, w, t = load_2022(v3)
    rec = pd.DataFrame(report.parse())
    p33 = rec[(rec.table == "P33") & (rec.name != "")]
    l11 = load_2011()
    p35 = pd.Series({c: v for c, v in p22_l3.items()
                     if not v3.set_index("adm3_pcode").adm3_name[c].endswith("City Corporation")})
    p, remainder = place_paurashavas(p, u, l11, v3, p35)
    print(f"  2022: {len(u)} unions, {len(p)} paurashavas ({p.placed_by.value_counts().to_dict()}), "
          f"{len(w)} city corporation ward rows; other areas {int(remainder.sum()):,} people "
          f"in {int((remainder > 0).sum())} upazilas")

    v3m = v3.to_crs(METRIC).set_index("adm3_pcode")
    meta = v3.set_index("adm3_pcode")
    feats, lineage, rows = [], [], []
    how_log = {}
    import time
    t0 = time.time()
    for i_c, c in enumerate(v3.adm3_pcode):
        if i_c % 100 == 0:
            print(f"    level 4: {i_c}/{len(v3)} level-3 units, {time.time() - t0:.0f} s", flush=True)
        name3 = meta.adm3_name[c]
        is_cc = name3.endswith("City Corporation")
        old = l11[l11.adm3_pcode == c]
        olds = [dict(key=f"o:{r.GEO_MATCH}", name=r.NSO_NAME, kind=r.kind, pop=int(r.pop_2011),
                     ward=(int(r.ward) if r.kind == "Ward" and pd.notna(r.ward) else None))
                for r in old.itertuples()]
        geo = old.set_index(old.GEO_MATCH.radd("o:")).geometry
        expand = None
        if is_cc and name3.replace(" City Corporation", "") in WARD_CCS:
            ws = w[w.cc == name3]
            news = [dict(key=f"w:{c}:{r.name}", name=r.name,
                         kind="Ward" if r.name.startswith("Ward") else "Other", pop=int(r.pop),
                         ward=(int(r.name.split()[1]) if re.match(r"^Ward \d+$", r.name) else None))
                    for r in ws.itertuples()]
            sets = match_unit(news, olds, how_log, make_neighbours(geo), p22_l3[c] / p11_l3[c])
        elif is_cc:
            # By thana (Table P33). In Dhaka the 2011 side is the 2011
            # metropolitan thana each ward piece or union belonged to; in the
            # other four it is the 2011 paurashava or union itself.
            ccname = name3.replace(" City Corporation", "")
            th = p33[p33.cc == name3]
            news = [dict(key=f"t:{c}:{r.name}", name=r.name, kind="Union", pop=int(r.pop))
                    for r in th.itertuples()]
            if ccname.startswith("Dhaka"):
                tname = old.unit_2011.str.replace(r"\s*[Tt]hana$", "", regex=True)
                # Dhaka's 2011 thanas: one 2011 record per thana
                byth = {}
                for k, tn in zip(old.GEO_MATCH.radd("o:"), tname):
                    byth.setdefault(tn, []).append(k)
                expand = {f"T:{c}:{tn}": ks for tn, ks in byth.items()}
                O0 = {o["key"]: o for o in olds}
                olds = [dict(key=vk, name=vk.split(":", 2)[2], kind="Union",
                             pop=sum(O0[k]["pop"] for k in ks), ward=None)
                        for vk, ks in expand.items()]
                vgeo = gpd.GeoSeries({vk: unary_union([geo[k] for k in ks])
                                      for vk, ks in expand.items()}, crs=METRIC)
                sets = match_unit(news, olds, how_log, make_neighbours(vgeo),
                                  p22_l3[c] / p11_l3[c], HAND_LINKS.get(ccname, ()))
                for vk, ks in expand.items():
                    for k in ks:
                        how_log[k] = how_log.get(vk, "")
                sets = [[m for k in s for m in (expand[k] if k in expand else [k])] for s in sets]
                olds = [o for o in O0.values()]
            else:
                for o in olds:
                    o["kind"] = "Union"  # paurashavas and unions alike stand for a thana
                sets = match_unit(news, olds, how_log, make_neighbours(geo),
                                  p22_l3[c] / p11_l3[c], HAND_LINKS.get(ccname, ()))
        else:
            uu = u[u.adm3_pcode == c]
            pp = p[p.adm3_pcode == c]
            news = [dict(key=f"u:{c}:{r.name}", name=r.name, kind="Union", pop=int(r.pop))
                    for r in uu.itertuples()]
            news += [dict(key=f"p:{c}:{r.name}", name=r.name + " Paurashava", kind="Paurashava",
                          pop=int(r.pop)) for r in pp.itertuples()]
            rem = int(remainder.get(c, 0))
            if rem > 0:
                news.append(dict(key=f"x:{c}", name="Other areas", kind="Other", pop=rem))
            sets = match_unit(news, olds, how_log, make_neighbours(geo), p22_l3[c] / p11_l3[c])
        N = {n["key"]: n for n in news}
        O = {o["key"]: o for o in olds}
        sets = rebalance(sets, N, O, geo, p22_l3[c] / p11_l3[c], how_log, allow_far=not is_cc)
        assert sum(n["pop"] for n in news) == p22_l3[c], (c, sum(n["pop"] for n in news), p22_l3[c])
        assert sum(o["pop"] for o in olds) == p11_l3[c], c
        assert sorted(k for s in sets for k in s) == sorted(list(N) + list(O)), c
        # order sets: biggest first, numbered within the level-3 unit
        sets.sort(key=lambda s: -sum(N[k]["pop"] for k in s if k in N))
        pieces = {i: unary_union([geo[k] for k in s if k in O]) for i, s in enumerate(sets)}
        fitted = fit_to_unit(v3m.geometry[c], pieces)
        for i, s in enumerate(sets):
            code = f"{c}{i + 1:03d}"
            names = [N[k]["name"] for k in s if k in N]
            names = sorted(names, key=lambda n: -[N[k]["pop"] for k in s if k in N and N[k]["name"] == n][0])
            label = names[0] if len(names) == 1 else (
                f"{names[0]} + {len(names) - 1} more" if len(names) > 3 else " + ".join(names))
            pop22 = sum(N[k]["pop"] for k in s if k in N)
            pop11 = sum(O[k]["pop"] for k in s if k in O)
            feats.append(dict(code=code, name=label, adm3_pcode=c, adm3_name=name3,
                              adm2_pcode=meta.adm2_pcode[c], adm1_pcode=meta.adm1_pcode[c],
                              geometry=polygonal(fitted[i])))
            rows += [(code, 4, 2011, pop11), (code, 4, 2022, pop22)]
            for k in s:
                m = N.get(k) or O.get(k)
                lineage.append(dict(code=code, unit=label, year=2022 if k in N else 2011,
                                    member=m["name"], kind=m["kind"], pop=m["pop"],
                                    how=how_log.get(k, "")))
    gdf = gpd.GeoDataFrame(feats, geometry="geometry", crs=METRIC).to_crs("EPSG:4326")
    tmp = OUT_GPKG.with_suffix(".part.gpkg")
    gdf.to_file(tmp, driver="GPKG")
    os.replace(tmp, OUT_GPKG)
    lin = pd.DataFrame(lineage)
    lin.to_csv(OUT_LIN, index=False)
    print(f"  level 4: {len(gdf)} units; members by rule: "
          + ", ".join(f"{k} {v}" for k, v in lin.how.value_counts().items()))
    return rows


def make_neighbours(geo):
    """f(key, candidate sets of 2011 keys) -> index of the set sharing the
    longest border with the 2011 unit `key`. geo: key -> geometry."""

    def nb(key, cands):
        g = geo[key].buffer(50)
        lens = [sum(g.intersection(geo[m].boundary).length for m in s) for s in cands]
        if max(lens) == 0:
            d = [min(geo[key].distance(geo[m]) for m in s) if s else np.inf for s in cands]
            return int(np.argmin(d))
        return int(np.argmax(lens))
    return nb


# City corporation thanas of 2022 with no 2011 namesake, and the 2011 units
# they were carved from (by hand; README "Level 4"). A 2022 thana listed here
# joins the set of every 2011 unit named, and with it whatever 2022 thana those
# match by name, so "Rupnagar": ["Mirpur", "Pallabi"] makes one unit of the
# 2022 thanas Mirpur, Pallabi and Rupnagar.
HAND_LINKS = {
    "Dhaka North": [
        ("Banani", ["Gulshan"]),                # 2013, out of Gulshan
        ("Bhatara", ["Badda"]),                 # out of Badda
        ("Hatirjheel", ["Tejgaon", "Rampura"]),  # Tejgaon and Rampura lost what it holds
        ("Rupnagar", ["Mirpur", "Pallabi"]),    # out of Mirpur and Pallabi
        ("Bhasantek", ["Kafrul"]),              # out of Kafrul
        # renamed or spelt so the name matcher misses them
        ("Tejgaon Shilpa Elaka", ["Tejgaon Ind. Area"]),
        ("Uttarkhan", ["Uttar Khan"]),
        ("Bimanbandar", ["Biman Bandar"]),
    ],
    "Dhaka South": [
        ("Wari", ["Sutrapur"]),
        ("Mugda", ["Sabujbagh"]),
        ("Shahjahanpur", ["Motijheel"]),
        # 2011 Kamrangir Char thana held only Sultanganj union; the 2022 thana
        # also holds what was Lalbagh's (Lalbagh halved, Kamrangichar x4)
        ("Kamrangichar", ["Lalbagh", "Kamrangir Char"]),
        ("Chakbazar", ["Chak Bazar"]),
        ("Newmarket", ["New Market"]),
    ],
    "Gazipur": [("Joydebpur", ["Gazipur Paurashava"])],
    "Narayanganj": [("Bandar", ["Kadam Rasul Paurashava"])],
    "Cumilla": [("Adarsha Sadar", ["Comilla Paurashava"]),
                ("Sadar Dakkhin", ["Comilla Dakshin Paurashava"])],
}
