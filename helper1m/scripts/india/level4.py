"""India level 4: the level-3 sub-districts, with the ones over 2 million split.

Called by fetch.py (level4.build(level-3 rows)); also runs standalone for a look
without rewriting population.csv. Writes
  data/india/level4.gpkg       one polygon per level-4 unit (code, name, parent, group)
  data/india/level4_parts.csv  every census ward, town and village behind each part
and returns the level-4 population rows.

Units and codes
  Every level-3 unit is carried as itself (code = level-3 code + "00") except the
  ones over 2,000,000 in 2026, which are replaced by parts (code + "01", "02", ...).
  parent = the level-3 code. A part's population is its share of the unit's 2011
  census total times the unit's level-3 figure for each year, the same method as
  level 3, apportioned by largest remainder so parts add up to their unit exactly
  in every year.

Sources (downloads under maps/data/asia1m/india/wards/)
  pca_tv/pca_tv_<pc11_d_id>.xlsx  Census 2011 PCA TV ("Primary census abstract at
      town, village and ward level"), one workbook per district, from
      censusindia.gov.in/nada (catalog search "PCA TV"). Every census ward, town
      and village with TOT_P. Per unit, the WARD rows plus the VILLAGE rows add up
      to the SUB-DISTRICT row exactly in all 30 units (checked at every build: a
      mismatch is logged).
  datameet/  DataMeet Municipal_Spatial_Data (github.com/datameet, CC BY-SA 2.5 IN):
      ward maps, used only where they match the 2011 census wards (below).
  religiondots/data/geo/in/shrug-village-pc11.parquet (read only): SHRUG village and
      town polygons with census codes, which place every village and every town that
      is not split by ward.

Method
  Atoms are census wards (for a town split by ward), whole towns and villages,
  each with its 2011 count and a set of sample points: a grid inside its polygon
  (ward-map polygon clipped to the town's SHRUG polygon, or the SHRUG village/town
  polygon), plus one representative point. Parts are groups of atoms; part
  outlines are the Voronoi cells of all the points, clipped to the level-3 outline
  and dissolved by part, so parts tile their unit exactly. Shapes are therefore a
  rough gist of where the people are, not legal boundaries.
  Grouping: the ward-split town by a named grouping where there is one (Mumbai
  admin wards, Bangalore assembly constituencies, Chennai zones), else by recursive
  bisection at the population median until each group is under 0.8M (2026). Other
  towns over 0.2M (2026) are parts of their own. The rest (villages and smaller
  towns) is bisected the same way. A part under 0.1M joins its nearest neighbour.
  Barrackpur-II also takes the 99999-row towns of North 24 Parganas that fetch.py's
  place_towns puts in it (Panihati, Kamarhati, Baranagar, the Dum Dums and more),
  which is why it is 2.6M though its own census row is 0.2M.

Ward maps, and how each match was checked
  Mumbai and Mumbai Suburban (99999 units): census ward codes are 4 digits whose
  first two (01-24) are BMC's 24 administrative wards in the order A, B, C, D, E,
  F/S, F/N, G/S, G/N, H/E, H/W, K/E, K/W, P/S, P/N, R/S, R/C, R/N, L, M/E, M/W, N, S,
  T. Each prefix's census total equals BMC's published 2011 ward population (A
  185,014; K/E 823,885; P/N 941,366; M/E 807,720; ...), exactly, for all 24. Shapes:
  BMC_Wards.geojson (admin wards, unchanged since before 2011). 24 parts.
  Bangalore 99999: BBMP_oldWards.geojson, the 198 wards of the 2010 delimitation
  that the census used (198 census wards, numbers 1-198 all present). Check: the
  map carries each ward's 2001 population; census 2011 / map 2001 runs 0.76-4.15
  (median 1.27, 5th percentile 0.91), growth only, as a right join gives and a
  wrong one would not. Grouped by the map's 2008 assembly constituency: 27 parts.
  Chennai 99999: Wards-2008.geojson, the old corporation's 155 wards in 10 zones,
  the city the census counted before the 2011 expansion. 155 = 155, every ward
  inside the unit, densities 6,000-134,000 per km2. Grouped by zone: 10 parts.
  Kolkata 99999: kolkata.geojson, 141 wards = the 141 census wards (KMC added
  142-144 only in 2015). All inside the unit. Bisection: 8 parts.
  Delhi, Saraswati Vihar: Delhi_Wards.geojson, the 272 MCD wards of the 2007
  delimitation (plus cantonment and NDMC charges, ignored). Check: of the 30 DMC
  wards the census puts in this tehsil, 27 lie inside it on the map; of the 16 DMC
  wards the census puts partly elsewhere, 3. Bisection on DMC (U) wards, plus
  Kirari Suleman Nagar and Sultan Pur Majra census towns: 7 parts.
  Jaipur tehsil: Jaipur_Wards.geojson, the 77 wards of the 2004 delimitation (the
  2014 one has 91). Check: 56 of the 59 census wards in Jaipur tehsil lie inside it
  on the map, 0 of the 18 the census puts in other tehsils do. 5 parts.
  Lucknow tehsil: Lucknow_ward_boundary.geojson, 110 numbered wards (plus "0",
  the airport), the census's count, but both the 2006 and the 2017 delimitation
  have 110 wards and the file is undated, and the whole city is in one tehsil so
  the tehsil test above cannot tell them apart. Placed by number anyway, as
  approximate: census ward populations on the map's neighbours are as spatially
  alike as in Chennai, Kolkata and Jaipur (statistic 0.50 against 0.47-0.62 there,
  0.07 for the known mismatch Bhopal), so numbers run through the city the same
  way; at 11-19 wards per part a renumbering's drift mostly stays inside a part.
  8 ward parts + villages: the ward parts' shapes are approximate.
  Every ward map's numbers were tested the same way before use; the ones that do
  not match 2011 were not used (below).

Left whole or only partly split, and why
  Ahmadabad City (6.85M): census has 57 wards, DataMeet's map is the 2015
      delimitation's 48. No 2011 ward map found. Carried whole.
  Surat City (5.48M), Nagpur (Urban) (2.73M): one town each, no ward map. Whole.
  Pune City (4.22M): kept whole, see KEEP_WHOLE (PMC's 144 census wards of the
      2007 delimitation have no map; DataMeet has the 2012/2017/2022 prabhags).
  Towns split off but not split further (2026): Kanpur 3.05M (DataMeet's Kanpur
      file has 58 of 110 wards and does not overlay), Indore 2.58M, Thane (M Corp.)
      2.43M, Ghaziabad 2.31M, Bhopal 2.29M (the DataMeet map is the 2014
      delimitation's 85 wards, the census has 70, and the number test fails),
      Pimpri Chinchwad 2.21M (map has 2017 prabhags), Vadodara 2.15M, Patna
      2.13M, Agra 1.96M, Ludhiana 1.84M, Nashik 1.80M, Kalyan-Dombivli 1.65M,
      Meerut 1.51M, Navi Mumbai 1.50M (map is 2015's 111 wards, census 89),
      Varanasi 1.42M, Moradabad, Bareilly, Aligarh, Mira-Bhayandar (about 1.1M).

Approximate shapes to know about
  Bangalore: SHRUG's outline for the 99999 unit is 507 km2 of BBMP's 713, missing
      most of the south, so constituencies there (Bangalore South, Jaya Nagar,
      Padmanaba Nagar, B.T.M. Layout) are slivers on the unit's edge; their people
      are right, their shapes are not.
  Haveli: SHRUG's Pimpri Chinchwad polygon is about half the corporation, so the
      "villages west" part is drawn over some of PCMC.
  Village parts take the gaps between SHRUG polygons, so they can look larger
      than their villages.
  One Varanasi village (Karaundi, 7,271) has no SHRUG polygon; it is counted in
      its part but not drawn.
"""
from __future__ import annotations

import os
import re
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402
from shapely.geometry import MultiPoint, Point  # noqa: E402
from shapely.ops import nearest_points  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

HELPER = Path(__file__).resolve().parents[2]
REPO = HELPER.parent
WARDS = REPO / "data" / "asia1m" / "india" / "wards"
PCA = WARDS / "pca_tv"            # pca_tv_<pc11_d_id>.xlsx, Census 2011 PCA TV per district
DM = WARDS / "datameet"           # DataMeet Municipal_Spatial_Data ward maps
SUBDIST_SHP = REPO / "data" / "asia1m" / "india" / "subdistrict.shp"
VILLAGES = REPO / "religiondots" / "data" / "geo" / "in" / "shrug-village-pc11.parquet"
OUT_GPKG = HELPER / "data" / "india" / "level4.gpkg"
OUT_PARTS = HELPER / "data" / "india" / "level4_parts.csv"

SPLIT_OVER = 2_000_000     # a level-3 unit over this in SPLIT_YEAR is split
SPLIT_YEAR = 2026
PART_MAX = 800_000         # split a group again while it is over this (SPLIT_YEAR)
TOWN_PART = 200_000        # a town this big (SPLIT_YEAR) is a part of its own
MERGE_UNDER = 100_000      # a part smaller than this joins its nearest neighbour
CRS = 7755                 # WGS 84 / India NSF LCC, metres

# Ward-split towns: census wards placed on a ward map by ward number.
#   town  - census town code (Town/Village column)
#   map   - DataMeet file under DM; num - its ward-number column
#   group - map column that groups wards into parts (else recursive bisection)
#   label - how a part is named
CITIES = {
    "51899999": dict(town="802794", map="Mumbai__BMC_Wards.geojson", mumbai=True),
    "51999999": dict(town="802794", map="Mumbai__BMC_Wards.geojson", mumbai=True),
    "57299999": dict(town="803162", map="Bangalore__BBMP_oldWards.geojson", num="WARD_NO",
                     group="ASS_CONST1", label="BBMP wards in {g} constituency"),
    "60399999": dict(town="803339", map="Chennai__Wards-2008.geojson", num="ward",
                     group="zone", label="Chennai zone {g}"),
    "34299999": dict(town="801742", map="Kolkata__kolkata.geojson", num="WARD"),
    "09000431": dict(town="800441", map="Delhi__Delhi_Wards.geojson", num="Ward_No"),
    "11000546": dict(town="800522", map="Jaipur__Jaipur_Wards.geojson", num="WARD_NO"),
    "15700821": dict(town="800951", map="Lucknow__Lucknow_ward_boundary.geojson",
                     num="Ward Num"),
}
# Units the method could split but should not (see the docstring).
KEEP_WHOLE = {
    "52104194": "SHRUG's Pune (M Corp.) polygon is 88 km2 of the corporation's 244, so "
                "the 0.23M cantonments-and-village part would be drawn as a ring over "
                "the rest of the city",
}
# Census ward prefix (first two digits) -> BMC administrative ward. Checked: each
# prefix's census total equals BMC's published 2011 figure for that ward.
MUMBAI_PREFIX = dict(zip(
    [f"{i:02d}" for i in range(1, 25)],
    ["A", "B", "C", "D", "E", "F/S", "F/N", "G/S", "G/N", "H/E", "H/W", "K/E", "K/W",
     "P/S", "P/N", "R/S", "R/C", "R/N", "L", "M/E", "M/W", "N", "S", "T"]))


def log(*a):
    print("  level 4:", *a)


def ward_num(v):
    """A map's ward number, or None for anything that is not a plain number
    (Delhi's CANT_1, NDMC charges and the like)."""
    s = str(v).strip()
    if re.fullmatch(r"\d+(\.0+)?", s):
        return int(float(s))
    return None


def ranges(nums):
    nums = sorted(set(nums))
    out, start, prev = [], None, None
    for n in nums:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append(f"{start}-{prev}" if prev > start else f"{start}")
            start = prev = n
    if start is not None:
        out.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ", ".join(out)


def compass(dx, dy):
    if dx * dx + dy * dy < 1e6:     # within 1 km of the centre
        return "central"
    a = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    return ["north", "northeast", "east", "southeast", "south", "southwest", "west",
            "northwest"][int((a + 22.5) // 45) % 8]


def clean_town(name):
    return re.sub(r"\s*\(Part\)\s*$", "", re.sub(r"\s+", " ", str(name))).strip()


# ---- inputs ---------------------------------------------------------------

_pca = {}


def pca(d):
    if d not in _pca:
        df = pd.read_excel(PCA / f"pca_tv_{d}.xlsx", dtype=str)
        df["TOT_P"] = pd.to_numeric(df["TOT_P"])
        _pca[d] = df
    return _pca[d]


def shrug(districts):
    import pyarrow.parquet as pq
    t = pq.read_table(VILLAGES, columns=["pc11_d_id", "pc11_sd_id", "pc11_tv_id", "geometry"],
                      filters=[("pc11_d_id", "in", sorted(districts))]).to_pandas()
    g = gpd.GeoDataFrame(t.drop(columns="geometry"),
                         geometry=shapely.from_wkb(t["geometry"].values), crs=4326)
    return g.to_crs(CRS)


def ward_map(cfg, unit_poly):
    g = gpd.read_file(DM / cfg["map"], on_invalid="ignore")
    g = g[g.geometry.notna()].to_crs(CRS)
    g["geometry"] = g.buffer(0)
    if cfg.get("mumbai"):
        g["key"] = g["name"].astype(str).str.strip()
    else:
        g["key"] = g[cfg["num"]].map(ward_num)
        g = g[g.key.notna()]
        g["key"] = g.key.astype(int)
    keep = ["key"] + ([cfg["group"]] if cfg.get("group") else [])
    return g[keep + ["geometry"]].dissolve("key", as_index=False, aggfunc="first")


# ---- geometry helpers -------------------------------------------------------

def sample(poly, spacing):
    """Grid points inside poly, plus one representative point."""
    if poly is None or poly.is_empty:
        return np.empty((0, 2))
    x0, y0, x1, y1 = poly.bounds
    xs = np.arange(x0 + spacing / 2, x1, spacing)
    ys = np.arange(y0 + spacing / 2, y1, spacing)
    pts = np.empty((0, 2))
    if len(xs) and len(ys):
        gx, gy = np.meshgrid(xs, ys)
        gx, gy = gx.ravel(), gy.ravel()
        inside = shapely.contains_xy(poly, gx, gy)
        pts = np.column_stack([gx[inside], gy[inside]])
    rp = poly.representative_point()
    return np.vstack([pts, [[rp.x, rp.y]]])


def into(poly, pt):
    """pt if it lies in poly, else the nearest point just inside poly."""
    if poly.contains(pt):
        return pt
    inner = poly.buffer(-20)
    target = inner if not inner.is_empty else poly
    return nearest_points(target, pt)[0]


def voronoi_parts(unit_poly, atoms, part_of):
    """Voronoi cells of every atom's points, clipped to the unit and dissolved
    by part. Parts tile the unit exactly."""
    xy, lab = [], []
    for i, a in enumerate(atoms):
        for p in a["pts"]:
            xy.append(p)
            lab.append(part_of[i])
    xy = np.asarray(xy)
    # drop exact duplicates, which voronoi cannot take
    xy, idx = np.unique(np.round(xy, 1), axis=0, return_index=True)
    lab = np.asarray(lab)[idx]
    if len(set(lab)) == 1:
        return {lab[0]: unit_poly}
    cells = shapely.voronoi_polygons(MultiPoint(xy), extend_to=unit_poly.envelope.buffer(5000))
    cells = gpd.GeoDataFrame(geometry=list(cells.geoms), crs=CRS)
    pts = gpd.GeoDataFrame({"part": lab}, geometry=shapely.points(xy), crs=CRS)
    j = gpd.sjoin(cells, pts, predicate="contains", how="inner")
    j["geometry"] = j.geometry.intersection(unit_poly)
    out = {}
    for part, g in j.groupby("part"):
        out[part] = shapely.union_all(g.geometry.values)
    return out


def bisect(atoms, idx, limit):
    """Recursive bisection of atoms[idx] by population at the weighted median,
    across the longer axis, until each group is at most limit (or one atom)."""
    pop = np.array([atoms[i]["p26"] for i in idx])
    if pop.sum() <= limit or len(idx) == 1:
        return [list(idx)]
    xy = np.array([atoms[i]["xy"] for i in idx])
    w = pop / pop.sum()
    spread = np.sqrt((w[:, None] * (xy - (w[:, None] * xy).sum(0)) ** 2).sum(0))
    ax = int(np.argmax(spread))
    order = np.argsort(xy[:, ax], kind="stable")
    cum = np.cumsum(pop[order])
    cut = int(np.searchsorted(cum, cum[-1] / 2))
    cut = min(max(cut, 0), len(idx) - 2) + 1
    left = [idx[k] for k in order[:cut]]
    right = [idx[k] for k in order[cut:]]
    return bisect(atoms, left, limit) + bisect(atoms, right, limit)


# ---- per unit ---------------------------------------------------------------

def unit_atoms(code, vil):
    """Atoms of one level-3 unit: census wards of ward-split towns, other towns
    whole, villages. Each with 2011 population and sample points."""
    d, sd = code[:3], code[3:]
    df = pca(d)
    s = df[df["Subdistt"] == sd]
    total = s[(s["Level"] == "SUB-DISTRICT") & (s["TRU"] == "Total")]["TOT_P"].sum()
    wards = s[s["Level"] == "WARD"]
    vills = s[s["Level"] == "VILLAGE"]
    towns = s[(s["Level"] == "TOWN") & (s["TRU"] == "Urban")]
    names = {}
    for _, t in towns.iterrows():   # prefer the "(M Corp.)" name over "+ OG" twin
        names.setdefault(t["Town/Village"], clean_town(t["Name"]))
    extra = []
    if sd != "99999":
        # towns of the district's polygon-less 99999 row that SHRUG files here
        # (fetch.py's place_towns puts their people in this sub-district)
        z = df[(df["Subdistt"] == "99999")]
        filed = set(vil[(vil.pc11_d_id == d) & (vil.pc11_sd_id == sd)].pc11_tv_id)
        zt = z[(z["Level"] == "TOWN") & (z["TRU"] == "Urban") & z["Town/Village"].isin(filed)]
        if len(zt):
            for _, t in zt.iterrows():
                names.setdefault(t["Town/Village"], clean_town(t["Name"]))
            extra = list(zt["Town/Village"].unique())
            wards = pd.concat([wards, z[(z["Level"] == "WARD") & z["Town/Village"].isin(extra)]])
            total += z[(z["Level"] == "TOWN") & (z["TRU"] == "Urban")
                       & z["Town/Village"].isin(extra)].drop_duplicates("Town/Village")["TOT_P"].sum()
    town_pop = wards.groupby("Town/Village")["TOT_P"].sum()
    atoms_sum = town_pop.sum() + vills["TOT_P"].sum()
    if atoms_sum != total:
        log(f"{code}: wards+villages {atoms_sum:,} vs sub-district row {total:,}")
    return wards, vills, town_pop, names, extra


def split_unit(code, unit_poly, sd_name, pops, vil):
    d = code[:3]
    wards, vills, town_pop, names, extra = unit_atoms(code, vil)
    cfg = CITIES.get(code)
    p11 = town_pop.sum() + vills["TOT_P"].sum()
    f26 = pops[SPLIT_YEAR] / p11
    spacing = float(np.clip(np.sqrt(unit_poly.area / 6000), 100, 500))
    poly_of = {}
    mine = vil[(vil.pc11_d_id == d)]
    for tv, g in mine.groupby("pc11_tv_id"):
        here = g[g.pc11_sd_id == code[3:]]
        poly_of[tv] = shapely.union_all((here if len(here) else g).geometry.values)
    centre = unit_poly.centroid

    atoms = []   # dict(kind, key, pop, pts, xy, p26, ...)

    def add(kind, key, pop, poly, **kw):
        if poly is not None and not poly.is_empty:
            poly = poly.intersection(unit_poly)
        pts = sample(poly, spacing) if poly is not None and not poly.is_empty else np.empty((0, 2))
        atoms.append(dict(kind=kind, key=key, pop=int(pop), pts=pts, p26=pop * f26, **kw))

    # 1. the ward-split town
    split_town = cfg["town"] if cfg else None
    missing = []
    if cfg:
        wm = ward_map(cfg, unit_poly)
        tpoly = poly_of.get(split_town)
        tpoly = unit_poly if tpoly is None else tpoly.intersection(unit_poly)
        w = wards[wards["Town/Village"] == split_town]
        if cfg.get("mumbai"):
            grp = w.groupby(w["Ward"].str[:2])["TOT_P"].sum()
            for pre, pop in grp.items():
                aw = MUMBAI_PREFIX[pre]
                geom = wm.loc[wm.key == aw, "geometry"]
                add("ward", aw, pop, geom.iloc[0].intersection(tpoly) if len(geom) else None,
                    group=aw)
        else:
            w = w.groupby(w["Ward"].astype(int))["TOT_P"].sum()
            byk = wm.set_index("key")
            for n, pop in w.items():
                if n in byk.index:
                    geom = byk.loc[n, "geometry"]
                    gval = byk.loc[n, cfg["group"]] if cfg.get("group") else None
                    inside = geom.intersection(tpoly)
                    if inside.is_empty or inside.area < 1:
                        inside = into(tpoly, geom.centroid).buffer(1)
                    add("ward", n, pop, inside, group=gval)
                else:
                    missing.append(n)
                    add("ward", n, pop, None, group=None)
    # 2. other towns, whole
    for tv, pop in town_pop.items():
        if tv == split_town:
            continue
        add("town", tv, pop, poly_of.get(tv), name=names.get(tv, tv))
    # 3. villages
    for _, v in vills.iterrows():
        add("village", v["Town/Village"], v["TOT_P"], poly_of.get(v["Town/Village"]),
            name=str(v["Name"]).strip())
    no_pts = [a for a in atoms if len(a["pts"]) == 0]
    for a in atoms:
        a["xy"] = a["pts"].mean(0) if len(a["pts"]) else np.array([centre.x, centre.y])
    # a census ward missing from the map goes with the located ward of the
    # nearest number (ward numbers run through a city in order)
    located = [a for a in atoms if a["kind"] == "ward" and len(a["pts"])]
    for a in atoms:
        if (a["kind"] == "ward" and not len(a["pts"]) and located
                and isinstance(a["key"], int)):
            nb = min(located, key=lambda b: abs(b["key"] - a["key"]))
            a["xy"], a["group"] = nb["xy"], nb.get("group")

    # ---- grouping into parts
    groups = []   # (label, [atom idx])
    wi = [i for i, a in enumerate(atoms) if a["kind"] == "ward"]
    if wi:
        tname = names.get(split_town, split_town)
        if cfg.get("mumbai"):
            for i in wi:
                groups.append((f"{atoms[i]['key']} ward", [i]))
        elif cfg.get("group"):
            by = {}
            for i in wi:
                by.setdefault(atoms[i]["group"], []).append(i)
            for g, idx in sorted(by.items(), key=lambda kv: str(kv[0])):
                groups.append((cfg["label"].format(g=g), idx))
        else:
            for idx in bisect(atoms, wi, PART_MAX):
                groups.append((f"{tname} wards " + ranges(atoms[i]["key"] for i in idx), idx))
        if missing:
            log(f"{code}: census wards not on the map, placed with the nearest number: {missing}")
    rest = []
    for i, a in enumerate(atoms):
        if a["kind"] == "town" and a["p26"] >= TOWN_PART:
            groups.append((a["name"], [i]))
        elif a["kind"] != "ward":
            rest.append(i)
    if rest:
        tehsil = sd_name
        rg = bisect(atoms, rest, PART_MAX)
        for idx in rg:
            nv = sum(atoms[i]["kind"] == "village" for i in idx)
            tw = sorted((atoms[i] for i in idx if atoms[i]["kind"] == "town"),
                        key=lambda a: -a["pop"])
            xy = np.average([atoms[i]["xy"] for i in idx], axis=0,
                            weights=[max(atoms[i]["pop"], 1) for i in idx])
            where = compass(xy[0] - centre.x, xy[1] - centre.y) if len(rg) > 1 else ""
            vw = f"{nv} village{'s' if nv > 1 else ''}"
            if nv and tw:
                lab = f"{tehsil} tehsil {vw} {where}, with {tw[0]['name']}"
                if len(tw) > 1:
                    lab += f" and {len(tw) - 1} more town{'s' if len(tw) > 2 else ''}"
            elif nv:
                lab = f"{tehsil} tehsil {vw} {where}"
            else:
                lab = ", ".join(a["name"] for a in tw[:3]) + (
                    f" and {len(tw) - 3} more towns" if len(tw) > 3 else "")
            groups.append((re.sub(r"\s+,", ",", re.sub(r"\s+", " ", lab)).strip(), idx))
    # merge parts under MERGE_UNDER into the nearest other part
    def gpop(idx):
        return sum(atoms[i]["p26"] for i in idx)

    def gxy(idx):
        return np.average([atoms[i]["xy"] for i in idx], axis=0,
                          weights=[max(atoms[i]["pop"], 1) for i in idx])
    changed = True
    while changed and len(groups) > 1:
        changed = False
        small = min(range(len(groups)), key=lambda k: gpop(groups[k][1]))
        if gpop(groups[small][1]) >= MERGE_UNDER:
            break
        lab, idx = groups.pop(small)
        c = gxy(idx)
        near = min(range(len(groups)), key=lambda k: np.hypot(*(gxy(groups[k][1]) - c)))
        nl, ni = groups[near]
        groups[near] = (f"{nl}, with {lab}", ni + idx)
        changed = True
    part_of = {}
    for k, (lab, idx) in enumerate(groups):
        for i in idx:
            part_of[i] = k
    if len(groups) == 1:
        return None, atoms, groups
    geoms = voronoi_parts(unit_poly, atoms, part_of)
    if no_pts:
        log(f"{code}: {len(no_pts)} atoms without a location "
            f"({sum(a['pop'] for a in no_pts):,} people): "
            f"{[a.get('name', a['key']) for a in no_pts][:8]}")
    return geoms, atoms, groups


def apportion(shares, total):
    """Integers proportional to shares summing exactly to total."""
    raw = np.asarray(shares, float) / np.sum(shares) * total
    out = np.floor(raw).astype(int)
    rem = int(total - out.sum())
    if rem:
        out[np.argsort(-(raw - out))[:rem]] += 1
    return out


def build(l3_rows):
    """l3_rows: [(code, 3, year, pop)] from fetch.py. Writes OUT_GPKG and
    OUT_PARTS; returns level-4 rows [(code, 4, year, pop)]."""
    if not (PCA.exists() and SUBDIST_SHP.exists() and VILLAGES.exists()):
        log("inputs missing - skipped")
        return []
    pops = {}
    for code, _, year, pop in l3_rows:
        pops.setdefault(code, {})[year] = pop
    big = sorted(c for c, p in pops.items() if p.get(SPLIT_YEAR, 0) > SPLIT_OVER)
    sd = gpd.read_file(SUBDIST_SHP).to_crs(CRS)
    sd["code"] = sd.pc11_d_id + sd.pc11_sd_id
    sd = sd.set_index("code")
    vil = shrug({c[:3] for c in big})
    recs, rows, lines = [], [], []
    split = {}
    for code in big:
        if code in KEEP_WHOLE:
            log(f"{code} {sd.loc[code].sd_name}: kept whole ({KEEP_WHOLE[code]})")
            continue
        poly = shapely.make_valid(sd.loc[code].geometry)
        geoms, atoms, groups = split_unit(code, poly, sd.loc[code].sd_name, pops[code], vil)
        if geoms is None:
            log(f"{code} {sd.loc[code].sd_name}: one part only, carried whole")
            continue
        p11 = [sum(atoms[i]["pop"] for i in idx) for _, idx in groups]
        years = sorted(pops[code])
        per_year = {y: apportion(p11, pops[code][y]) for y in years}
        split[code] = True
        for k, (lab, idx) in enumerate(groups):
            pc = f"{code}{k + 1:02d}"
            recs.append(dict(code=pc, name=lab, parent=code, group=sd.loc[code].pc11_s_id,
                             geometry=geoms.get(k)))
            for y in years:
                rows.append((pc, 4, y, int(per_year[y][k])))
            for i in idx:
                a = atoms[i]
                lines.append(dict(part=pc, part_name=lab, kind=a["kind"], census_code=a["key"],
                                  name=a.get("name", ""), pop2011=a["pop"]))
        sizes = sorted(per_year[SPLIT_YEAR])
        log(f"{code} {sd.loc[code].sd_name}: {len(groups)} parts, "
            f"{sizes[0] / 1e6:.2f}-{sizes[-1] / 1e6:.2f}M in {SPLIT_YEAR}")
    for code, r in sd.iterrows():
        if code in split:
            continue
        recs.append(dict(code=f"{code}00", name=r.sd_name, parent=code, group=r.pc11_s_id,
                         geometry=r.geometry))
        for y, p in sorted(pops.get(code, {}).items()):
            rows.append((f"{code}00", 4, y, p))
    out = gpd.GeoDataFrame(recs, geometry="geometry", crs=CRS)
    empty = out[out.geometry.isna() | out.geometry.is_empty]
    if len(empty):
        log(f"WARNING {len(empty)} level-4 units with no geometry: {list(empty.code)}")
    out = out.to_crs(4326)
    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_GPKG.with_suffix(".part.gpkg")
    if tmp.exists():
        tmp.unlink()
    out.to_file(tmp, driver="GPKG")
    os.replace(tmp, OUT_GPKG)
    pd.DataFrame(lines).to_csv(OUT_PARTS, index=False, encoding="utf-8")
    log(f"{len(out)} units ({len(split)} sub-districts split) -> {OUT_GPKG.name}")
    return rows


if __name__ == "__main__":
    # standalone: read level-3 rows from the existing population.csv
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    df = pd.read_csv(HELPER / "data" / "india" / "population.csv", dtype={"code": str})
    l3 = [(r.code, 3, int(r.year), int(r.pop)) for r in df[df.level == 3].itertuples()]
    rows = build(l3)
    print(len(rows), "level-4 rows (population.csv not rewritten; run fetch.py for that)")
