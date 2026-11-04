"""India level 4: the level-3 sub-districts, with the ones over 750,000 split.

Called by fetch.py (level4.build(level-3 rows)); also runs standalone for a look
without rewriting population.csv. Writes
  data/india/level4.gpkg       one polygon per level-4 unit (code, name, parent, group)
  data/india/level4_parts.csv  every census ward, town and village behind each part
  data/india/level3.gpkg       the level-3 outlines, SHRUG's with subdistricts()'s fixes
  data/india/level2.gpkg       SHRUG's districts, REDRAW ones the union of their new tehsils
and returns the level-4 population rows.

Units and codes
  Every level-3 unit is carried as itself (code = level-3 code + "00") except the
  ones over 750,000 in 2026 (2,000,000 until 2026-10-25, her call), which are replaced by parts (code + "01", "02", ...).
  parent = the level-3 code. A part's population is its share of the unit's 2011
  census total times the unit's level-3 figure for each year, the same method as
  level 3, apportioned by largest remainder so parts add up to their unit exactly
  in every year.

Sources (downloads under maps/data/asia1m/india/wards/)
  pca_tv/pca_tv_<pc11_d_id>.xlsx  Census 2011 PCA TV ("Primary census abstract at
      town, village and ward level"), one workbook per district, from
      censusindia.gov.in/nada, downloaded by fetch_pca_tv.py. Every census ward, town
      and village with TOT_P. Per unit, the WARD rows plus the VILLAGE rows add up
      to the SUB-DISTRICT row exactly in all 265 units (checked at every build: a
      mismatch is logged).
  Result (2026-11-02 build): all 265 units over 0.75M are split. Before that
      day 13 were carried whole (one town each with no 2011 ward map: Ahmadabad,
      Surat, Nagpur, ...) plus Pune City, and 72 level-4 units were still over
      0.75M because big towns stayed one part; now every town over TOWN_SPLIT
      is split too (town_pieces()), and the only part over 1M is Mumbai's P/N
      ward, a real BMC ward.
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
  A town over TOWN_SPLIT (0.75M in 2026) that is not split by census ward is
  split anyway, with less to go on (town_pieces()): Pune and Ahmadabad over the
  wards of a later delimitation drawn on the 2011 census, every ward near the
  same population and the outer limit the 2011 one (EQUAL_WARDS: PMC's 76
  prabhags of 2012, AMC's 48 wards of 2015), each ward an equal share; every
  other town over GHSL 2025's 1 km cells clipped to its polygon, people shared
  by GHSL with no cell over GHSL_CAP per km2. Parts are named by those ward
  numbers or by direction from the town's centre ("Surat (M Corp.) north 2").
  These shapes say less than census wards do: GHSL misplaces people at the
  edges (see audit_ghsl.py), and equal shares miss the 10% each ward may vary.
  Grouping: the ward-split town by a named grouping where there is one (Mumbai
  admin wards, Bangalore assembly constituencies, Chennai zones), else by recursive
  bisection at the population median until each group is under 0.5M (2026). Other
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
  Delhi (all 27 tehsils, since 2026-11-02; Saraswati Vihar alone before):
  Delhi_Wards.geojson, the 272 MCD wards of the 2007 delimitation plus 9 NDMC and
  8 cantonment charges. Every census ward is on it: DMC (U) ward n = map ward n,
  NDMC (Part) ward n = NDMC charge n, Delhi Cantonment ward n = CANT charge n, and
  a census town's rows carry the number of the MCD ward it lies in. Checks: each
  tehsil's census wards make one block on the map (Gandhi Nagar and Patel Nagar
  bar a ward holding one census town), with the map's names in the right place
  (ward 234 "Gandhi Nagar" in Gandhi Nagar tehsil); neighbouring map wards share
  a tehsil 64% of the time, 5% when tehsils are shuffled; SHRUG's census-town
  polygons lie in the wards their rows name (median 92% of the area). Big
  tehsils are split by bisection on DMC (U) wards; other towns as everywhere.
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
  Since 2026-11-02 all the towns below are split by town_pieces() instead; the
  notes say why census wards could not be used.
  Ahmadabad City (6.85M): census has 57 wards, DataMeet's map is the 2015
      delimitation's 48. No 2011 ward map found. Split by EQUAL_WARDS.
  Surat City (5.48M), Nagpur (Urban) (2.73M): one town each, no ward map. GHSL.
  Pune City (4.22M): PMC's 144 census wards of the 2007 delimitation have no
      map; DataMeet has the 2012/2017/2022 prabhags. Split by the 2012 ones.
  Towns once split off but not split further (2026): Kanpur 3.05M (DataMeet's Kanpur
      file has 58 of 110 wards and does not overlay), Indore 2.58M, Thane (M Corp.)
      2.43M, Ghaziabad 2.31M, Bhopal 2.29M (the DataMeet map is the 2014
      delimitation's 85 wards, the census has 70, and the number test fails),
      Pimpri Chinchwad 2.21M (map has 2017 prabhags), Vadodara 2.15M, Patna
      2.13M, Agra 1.96M, Ludhiana 1.84M, Nashik 1.80M, Kalyan-Dombivli 1.65M,
      Meerut 1.51M, Navi Mumbai 1.50M (map is 2015's 111 wards, census 89),
      Varanasi 1.42M, Moradabad, Bareilly, Aligarh, Mira-Bhayandar (about 1.1M).

Approximate shapes to know about
  Bangalore: SHRUG's outline for the 99999 unit is 504 km2 of the ward map's 708,
      missing most of the south, which SHRUG gives to Bangalore South taluk (186
      km2 of city), so the south's constituencies were slivers and the taluk's 0.28M
      were spread over the southern city. Since 2026-10-31 (she spotted it) the unit
      takes in the ward map and that area is taken out of the four taluks; level 3
      gets the same outlines through level3.gpkg.
  Pune (2026-11-02, she spotted it too): SHRUG's PMC polygon is 88 km2 of the 2012
      ward map's 243, and 39 km2 of PMC sat in Haveli's "villages west" part; its
      Pimpri Chinchwad polygon is 84 km2 of PCMC's 171, so PCMC's 2.2M were drawn
      at 26,000/km2 with the villages part over the rest of it.
  Both fixed by TOWN_OUTLINES: a town listed there is drawn as its DataMeet map,
      whose outer limit must match the corporation's 2011 one (PMC 2012 wards;
      PCMC electoral wards at 171 km2, its limit since 1997 bar Talawade; BBMP's
      2010 wards; VMC's admin wards at 154 km2, before the 2020 expansion); its
      unit takes in that area from its district's other sub-districts
      (subdistricts()), and other SHRUG polygons inside it give the area up.
      Checked the same way and left: Chennai, Kolkata, Ahmedabad (map within 8
      km2 of SHRUG), Mira-Bhayandar and Navi Mumbai (within 30 km2, maps of later
      limits), Bhopal (map is the 2014 limits, 392 km2).
  Delhi (2026-11-02, found by audit_ghsl.py): SHRUG's DMC (U) polygon in each
      tehsil does not follow the census's tehsils (Gandhi Nagar's 5.0 km2 has 19% of
      its area in the tehsil's census wards), so Gandhi Nagar was drawn at 101,000
      per km2, Seema Puri 99,000 and Shahdara 98,000 with Vivek Vihar pale beside
      them, and one level-4 part was 0.58M on 2 km2. REDRAW rebuilds all 27 from
      census membership (redraw()): villages and census towns keep their SHRUG
      polygons; each map ward, less those, goes to the tehsil holding its people;
      84 of 289 wards are split between tehsils (1.2M in the minority shares), and
      their area is carved by nearness to what each tehsil holds for sure (a tehsil
      under REDRAW_MINOR of a ward's people gets none); five central tehsils with no
      whole ward (Sadar Bazar, Kotwali, Connaught Place, Chanakya Puri, Parliament
      Street) seed that carve from SHRUG's polygon; 37.5 km2 off the map goes to the
      nearest tehsil. The 27 tile SHRUG's Delhi (1,476 km2) with no gap or overlap;
      level 2 follows through level2.gpkg. Km2 before -> after: Gandhi Nagar 5.0 ->
      9.7, Seema Puri 7.6 -> 12.2, Shahdara 4.6 -> 5.3 (86,000/km2 in 2026, which its
      own wards bear out), Vivek Vihar 22.3 -> 9.7, Rajouri Garden 10.1 -> 15.0,
      Sadar Bazar 3.4 -> 7.4, Pahar Ganj 4.1 -> 2.9. Densest level-4 parts now about
      105,000/km2 on their own wards (north east Delhi), against 291,000 before.
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
DISTRICT_SHP = REPO / "data" / "asia1m" / "india" / "district.shp"
VILLAGES = REPO / "religiondots" / "data" / "geo" / "in" / "shrug-village-pc11.parquet"
OUT_GPKG = HELPER / "data" / "india" / "level4.gpkg"
OUT_L3 = HELPER / "data" / "india" / "level3.gpkg"
OUT_L2 = HELPER / "data" / "india" / "level2.gpkg"
OUT_PARTS = HELPER / "data" / "india" / "level4_parts.csv"

SPLIT_OVER = 750_000       # a level-3 unit over this in SPLIT_YEAR is split
SPLIT_YEAR = 2026
PART_MAX = 500_000         # split a group again while it is over this (SPLIT_YEAR)
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
    "11000546": dict(town="800522", map="Jaipur__Jaipur_Wards.geojson", num="WARD_NO"),
    "15700821": dict(town="800951", map="Lucknow__Lucknow_ward_boundary.geojson",
                     num="Ward Num"),
}
# Towns whose SHRUG polygon is far smaller than the corporation was in 2011, drawn
# instead as the union of a ward map whose outer limit matches 2011 (see the
# docstring). census town code -> (level-3 unit it is counted in, DataMeet file)
TOWN_OUTLINES = {
    "803162": ("57299999", "Bangalore__BBMP_oldWards.geojson"),      # 488 -> 708 km2
    "802814": ("52104194", "Pune__pune-electoral-wards_2012.geojson"),  # 88 -> 243
    "802811": ("52104193", "PCMC__pcmc-electoral-wards.geojson"),    # 84 -> 171
    "802596": ("48603898", "Vadodara__vardodara_wards.geojson"),     # 112 -> 154
}
# Districts whose SHRUG sub-district lines do not follow the 2011 census, redrawn
# from census membership (see the docstring): each tehsil is its villages and
# census towns (SHRUG polygons) plus the ward-map area of the wards the census
# counts in it, a ward shared between tehsils carved by nearness.
#   districts - pc11_d_id; map, num - DataMeet ward map and its ward column
#   towns     - census town code -> prefix of its wards' keys on the map
#   split     - the town every big tehsil is split by ward at level 4 (as CITIES)
REDRAW = {
    "Delhi": dict(districts=[f"{d:03d}" for d in range(90, 99)],
                  map="Delhi__Delhi_Wards.geojson", num="Ward_No",
                  towns={"800441": "", "800442": "NDMC_", "800443": "CANT_"},
                  split="800441"),
}
REDRAW_MINOR = 0.03        # a tehsil with under this share of a ward's people gets none of its area
REDRAW_SLIVER = 100_000    # m2; a detached piece smaller than this joins its neighbour
# Units the method could split but should not (see the docstring).
KEEP_WHOLE = {}
# A town at least this big (SPLIT_YEAR) with no 2011 ward map is split anyway
# (split_town()): over the wards of a map drawn on the 2011 census at equal
# population (EQUAL_WARDS), else by GHSL.
TOWN_SPLIT = SPLIT_OVER
# census town code -> (DataMeet file, ward column, year). Each is a delimitation
# made from the 2011 census with every ward near the same population, inside the
# town's 2011 limits, so each ward gets an equal share. The column's leading
# number is the ward number ("47 VATVA" is 47).
EQUAL_WARDS = {
    "802814": ("Pune__pune-electoral-wards_2012.geojson", "wardnum", 2012),  # 76 prabhags
    "802484": ("Ahmedabad__Wards.geojson", "Name", 2015),                    # 48 wards
}
GHSL = (REPO / "trainworld" / "data" / "raw" / "worldplan"
        / "GHS_POP_E2025_GLOBE_R2023A_54009_1000_V1_0.tif")
# GHSL piles people into its densest cells (Surat's top 1 km cells at 250,000-
# 315,000 per km2, where the densest census wards anywhere here are about
# 135,000, Chennai), so a GHSL piece holds at most this many 2011 people per km2
GHSL_CAP = 100_000
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


_outline = {}


def town_outline(town):
    """TOWN_OUTLINES town as one polygon (CRS)."""
    if town not in _outline:
        g = gpd.read_file(DM / TOWN_OUTLINES[town][1], on_invalid="ignore")
        g = g[g.geometry.notna()].to_crs(CRS)
        _outline[town] = shapely.union_all(g.buffer(0).values)
    return _outline[town]


_redrawn_towns = {}   # (level-3 code, census town) -> that town's area in it, from redraw()


def map_key(v):
    """A ward map's key: plain numbers as "17", anything else as written (CANT_3)."""
    n = ward_num(v)
    return str(n) if n is not None else str(v).strip()


def redraw_cfg(code):
    """The CITIES-style ward split for a level-3 unit in a REDRAW district."""
    for cfg in REDRAW.values():
        if code[:3] in cfg["districts"]:
            return dict(town=cfg["split"], map=cfg["map"], num=cfg["num"])
    return None


def redraw(cfg, old):
    """REDRAW entry, {code: SHRUG outline} -> {code: new outline}, tiling the
    same ground (CRS), built from census membership. Fills _redrawn_towns with
    each ward-mapped town's area per tehsil, for level 4."""
    state = shapely.union_all(list(old.values()))
    ds = cfg["districts"]
    df = pd.concat([pca(d) for d in ds])
    df["code"] = df["District"].str.zfill(3) + df["Subdistt"]
    towns = cfg["towns"]
    pieces = []   # (code, ward-mapped town or "", geometry)

    # 1. villages and census towns: SHRUG's polygon, in the tehsil the census
    #    counts them in
    vil = shrug(ds)
    poly = {tv: shapely.union_all(g.geometry.values) for tv, g in vil.groupby("pc11_tv_id")}
    tv = df[(df.Level == "VILLAGE")
            | ((df.Level == "TOWN") & (df.TRU == "Urban") & ~df["Town/Village"].isin(towns))]
    tv = tv.drop_duplicates(["Town/Village", "code"])
    twice = tv[tv.duplicated("Town/Village", keep=False)]
    if len(twice):
        log(f"redraw: in two tehsils, placed in the first: {sorted(set(twice['Town/Village']))}")
    nopoly = []
    for t, c in tv.drop_duplicates("Town/Village")[["Town/Village", "code"]].values:
        if t in poly:
            pieces.append((c, "", poly[t].intersection(state)))
        else:
            nopoly.append(t)
    if nopoly:
        log(f"redraw: {len(nopoly)} villages/towns without a SHRUG polygon: {nopoly[:8]}")
    taken = shapely.union_all([p for _, _, p in pieces])

    # 2. census people per ward and tehsil. A town's ward row carries the town's
    #    ward number; a census town's row carries the number of the ward it lies in
    w = df[df.Level == "WARD"].copy()
    pre = w["Town/Village"].map(towns)
    w["key"] = pre.fillna("") + w["Ward"].astype(int).astype(str)
    w["town"] = w["Town/Village"].where(pre.notna(), "")
    in_town = w[w.town != ""].groupby(["key", "code"])["TOT_P"].sum()
    in_any = w.groupby(["key", "code"])["TOT_P"].sum()
    town_of = w[w.town != ""].drop_duplicates("key").set_index("key")["town"]

    wm = gpd.read_file(DM / cfg["map"], on_invalid="ignore")
    wm = wm[wm.geometry.notna()].to_crs(CRS)
    wm["key"] = wm[cfg["num"]].map(map_key)
    wm["geometry"] = wm.buffer(0)
    wm = wm.dissolve("key", as_index=False)
    missing = sorted(set(in_any.index.get_level_values(0)) - set(wm.key))
    if missing:
        log(f"redraw: census wards not on the map: {missing}")

    # 3. each map ward minus the villages and census towns in it goes to the
    #    tehsil(s) holding its town's people (else its census towns' people)
    tree = shapely.STRtree(wm.geometry.values)
    shared = []
    for i, (key, geom) in enumerate(zip(wm.key, wm.geometry)):
        rest = geom.intersection(state).difference(taken)
        prev = [j for j in tree.query(geom) if j < i]
        if prev:
            rest = rest.difference(shapely.union_all(wm.geometry.values[prev]))
        if rest.is_empty:
            continue
        src = in_town if key in in_town.index.get_level_values(0) else in_any
        if key not in src.index.get_level_values(0):
            continue   # no census rows (the map's unnumbered sliver): left to the gap fill
        o = src.loc[key]
        o = o[o >= REDRAW_MINOR * o.sum()]
        town = town_of.get(key, "")
        if len(o) == 1:
            pieces.append((o.index[0], town, rest))
        else:
            shared.append((key, town, rest, list(o.index)))

    # 4. a shared ward is carved between its tehsils by nearness to the area
    #    they surely hold (Voronoi of points in their other pieces)
    seeds = {}
    for c, _, p in pieces:
        seeds.setdefault(c, []).append(sample(p, 150))
    seeds = {c: np.vstack(v) for c, v in seeds.items()}
    # a tehsil that holds no ward outright is seeded by SHRUG's polygon for it
    bare = sorted({c for *_, codes in shared for c in codes} - set(seeds))
    for c in bare:
        seeds[c] = sample(old[c], 150)
    if bare:
        log(f"redraw: seeded by SHRUG's own polygon (no ward of their own): {bare}")

    def carve(area, codes, near):
        xy, lab = [], []
        for c in codes:
            s = seeds[c]
            box = near.bounds
            k = ((s[:, 0] > box[0]) & (s[:, 0] < box[2]) & (s[:, 1] > box[1]) & (s[:, 1] < box[3]))
            if not k.any():   # nothing nearby: its 20 nearest points
                d = np.hypot(s[:, 0] - near.centroid.x, s[:, 1] - near.centroid.y)
                k = np.argsort(d)[:20]
            xy.append(s[k])
            lab += [c] * len(s[k])
        xy, idx = np.unique(np.round(np.vstack(xy), 1), axis=0, return_index=True)
        lab = np.asarray(lab)[idx]
        cells = shapely.voronoi_polygons(MultiPoint(xy), extend_to=area.envelope.buffer(5000))
        cells = gpd.GeoDataFrame(geometry=list(cells.geoms), crs=CRS)
        pts = gpd.GeoDataFrame({"code": lab}, geometry=shapely.points(xy), crs=CRS)
        j = gpd.sjoin(cells, pts, predicate="contains", how="inner")
        j["geometry"] = j.geometry.intersection(area)
        return {c: shapely.union_all(g.geometry.values) for c, g in j.groupby("code")}

    for key, town, rest, codes in shared:
        for c, p in carve(rest, codes, rest.buffer(1500)).items():
            if not p.is_empty:
                pieces.append((c, town, p))

    # 5. what is left of the state (off the map, slivers) goes to the nearest tehsil
    gap = state.difference(shapely.union_all([p for _, _, p in pieces]))
    if gap.area > 1:
        seeds = {}
        for c, _, p in pieces:
            seeds.setdefault(c, []).append(sample(p, 300))
        seeds = {c: np.vstack(v) for c, v in seeds.items()}
        for c, p in carve(gap, sorted(seeds), state).items():
            if not p.is_empty:
                pieces.append((c, "", p))
        log(f"redraw: {gap.area / 1e6:.1f} km2 not on the map or in a SHRUG polygon, "
            f"given to the nearest tehsil")

    out = {}
    for c in sorted({c for c, _, _ in pieces}):
        out[c] = shapely.make_valid(shapely.union_all([p for cc, _, p in pieces if cc == c]))
    # slivers (a piece under REDRAW_SLIVER away from its tehsil's main body) go
    # to the neighbour they share most border with
    for c in list(out):
        gs = sorted(shapely.get_parts(out[c]), key=lambda g: -g.area)
        for s in gs[1:]:
            if s.area >= REDRAW_SLIVER:
                continue
            ring = s.buffer(5)
            nb = max((o for o in out if o != c), key=lambda o: out[o].intersection(ring).area)
            out[c] = out[c].difference(s)
            out[nb] = out[nb].union(s)
    for (c, town) in {(c, t) for c, t, _ in pieces if t}:
        _redrawn_towns[(c, town)] = shapely.make_valid(
            shapely.union_all([p for cc, t, p in pieces if cc == c and t == town]))
    return out


def subdistricts():
    """SHRUG's sub-district outlines (CRS, indexed by code), with REDRAW
    districts redrawn from the census, and each TOWN_OUTLINES town added to the
    unit it is counted in, clipped to the district, and taken out of the
    district's other sub-districts."""
    sd = gpd.read_file(SUBDIST_SHP).to_crs(CRS)
    sd["code"] = sd.pc11_d_id + sd.pc11_sd_id
    sd = sd.set_index("code")
    sd["geometry"] = shapely.make_valid(sd.geometry.values)
    for name, cfg in REDRAW.items():
        sib = sd.pc11_d_id.isin(cfg["districts"])
        new = redraw(cfg, sd.loc[sib].geometry.to_dict())
        moved = []
        for c in sd.index[sib]:
            old = sd.loc[c].geometry.area
            sd.loc[c, "geometry"] = new.get(c, Point())
            moved.append(f"{sd.loc[c].sd_name} {old / 1e6:.0f}->{sd.loc[c].geometry.area / 1e6:.0f}")
        log(f"{name} redrawn from census wards (km2): {', '.join(moved)}")
    for town, (code, f) in TOWN_OUTLINES.items():
        sib = sd.pc11_d_id == code[:3]
        old = sd.loc[code].geometry.area
        city = town_outline(town).intersection(shapely.union_all(sd.loc[sib].geometry.values))
        moved = []
        for c in sd.index[sib]:
            if c == code:
                continue
            g = sd.loc[c].geometry
            lost = g.intersection(city).area
            if lost > 1e6:
                moved.append(f"{sd.loc[c].sd_name} {lost / 1e6:.0f}")
            sd.loc[c, "geometry"] = g.difference(city)
        new = sd.loc[code].geometry.union(city)
        sd.loc[code, "geometry"] = new
        log(f"{code} {sd.loc[code].sd_name} with {f}: {old / 1e6:.0f} -> {new.area / 1e6:.0f} km2, "
            f"taken from {', '.join(moved) or 'nothing'} (km2)")
    return sd


def town_pieces(town, tpoly):
    """A TOWN_SPLIT town's pieces as [(key, weight, polygon)] (CRS) and how they
    were made: EQUAL_WARDS map wards weighted by the share of each inside the
    town, else GHSL 1 km cells weighted by their people inside the town."""
    if town in EQUAL_WARDS:
        f, col, year = EQUAL_WARDS[town]
        g = gpd.read_file(DM / f, on_invalid="ignore")
        g = g[g.geometry.notna()].to_crs(CRS)
        g["key"] = g[col].astype(str).str.extract(r"^\s*(\d+)")[0].astype(float)
        g = g[g.key.notna()]
        g["geometry"] = g.buffer(0)
        g = g.dissolve("key", as_index=False)
        out = []
        for k, geom in zip(g.key.astype(int), g.geometry):
            inside = geom.intersection(tpoly)
            if inside.area > 1:
                out.append((k, inside.area / geom.area, inside))
        return out, f"{year} wards"
    import rasterio
    from rasterio.windows import Window
    p = gpd.GeoSeries([tpoly], crs=CRS).to_crs("ESRI:54009").iloc[0]
    with rasterio.open(GHSL) as src:
        t = src.transform
        x0, y0, x1, y1 = p.bounds
        c0, c1 = int((x0 - t.c) // t.a), int((x1 - t.c) // t.a) + 1
        r0, r1 = int((t.f - y1) // -t.e), int((t.f - y0) // -t.e) + 1
        arr = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(float)
    arr[arr < 0] = 0
    rr, cc = np.nonzero(arr > 0)
    xs, ys = t.c + (c0 + cc) * t.a, t.f + (r0 + rr) * t.e
    cells = shapely.box(xs, ys + t.e, xs + t.a, ys)
    inside = shapely.intersection(cells, p)
    w = arr[rr, cc] * shapely.area(inside) / (t.a * -t.e)
    k = w > 0
    geo = gpd.GeoSeries(inside[k], crs="ESRI:54009").to_crs(CRS)
    return [(i, wi, gi) for i, (wi, gi) in enumerate(zip(w[k], geo.values))], "GHSL"


def cap_density(w, km2, total):
    """Weights scaled to total with no piece over GHSL_CAP per km2, the excess
    spread over the others in proportion (repeated until nothing is over)."""
    x = w / w.sum() * total
    cap = GHSL_CAP * km2
    for _ in range(50):
        over = x > cap
        if not over.any():
            break
        spare = (x[over] - cap[over]).sum()
        x[over] = cap[over]
        free = ~over & (x < cap)
        if not free.any():
            break
        x[free] += spare * x[free] / x[free].sum()
    return x


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
    cfg = CITIES.get(code) or redraw_cfg(code)
    p11 = town_pop.sum() + vills["TOT_P"].sum()
    f26 = pops[SPLIT_YEAR] / p11
    spacing = float(np.clip(np.sqrt(unit_poly.area / 6000), 100, 500))
    poly_of = {}
    mine = vil[(vil.pc11_d_id == d)]
    for tv, g in mine.groupby("pc11_tv_id"):
        here = g[g.pc11_sd_id == code[3:]]
        poly_of[tv] = shapely.union_all((here if len(here) else g).geometry.values)
    # a TOWN_OUTLINES town takes its outline, and SHRUG polygons lying in it
    # give that area up (unless that would leave them nothing)
    for town, (c, _) in TOWN_OUTLINES.items():
        if c != code:
            continue
        out = town_outline(town)
        for tv, p in poly_of.items():
            rest = p.difference(out)
            if not rest.is_empty and rest.area > 1:
                poly_of[tv] = rest
        poly_of[town] = out
    # a REDRAW district's ward-mapped towns take their redrawn area
    for (c, town), p in _redrawn_towns.items():
        if c == code:
            poly_of[town] = p
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
    # 2. other towns, whole, but a town over TOWN_SPLIT in pieces
    how = {}
    for tv, pop in town_pop.items():
        if tv == split_town:
            continue
        tpoly = poly_of.get(tv)
        if pop * f26 >= TOWN_SPLIT and tpoly is not None:
            pieces, how[tv] = town_pieces(tv, tpoly.intersection(unit_poly))
            if len(pieces) > 1:
                w = np.array([w for _, w, _ in pieces], float)
                if how[tv] == "GHSL":
                    w = cap_density(w, np.array([p.area / 1e6 for _, _, p in pieces]), pop)
                for (k, _, p), n in zip(pieces, apportion(w, pop)):
                    add("tpart", k, n, p, town=tv, name=names.get(tv, tv))
                continue
        add("town", tv, pop, tpoly, name=names.get(tv, tv))
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
    # a town split by town_pieces(): bisection, parts named by ward numbers or
    # by direction from the town's centre
    for tv, h in how.items():
        ti = [i for i, a in enumerate(atoms) if a["kind"] == "tpart" and a["town"] == tv]
        if not ti:
            continue
        tname = atoms[ti[0]]["name"]
        tc = np.average([atoms[i]["xy"] for i in ti], axis=0,
                        weights=[max(atoms[i]["pop"], 1) for i in ti])
        seen = {}
        for idx in bisect(atoms, ti, PART_MAX):
            if h == "GHSL":
                xy = np.average([atoms[i]["xy"] for i in idx], axis=0,
                                weights=[max(atoms[i]["pop"], 1) for i in idx])
                lab = f"{tname} {compass(xy[0] - tc[0], xy[1] - tc[1])}"
                seen[lab] = seen.get(lab, 0) + 1
                if seen[lab] > 1:
                    lab += f" {seen[lab]}"
            else:
                lab = f"{tname} {h} " + ranges(atoms[i]["key"] for i in idx)
            groups.append((lab, idx))
        log(f"{code}: {tname} split by {h}")
    rest = []
    for i, a in enumerate(atoms):
        if a["kind"] == "tpart":
            continue
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


def write_gpkg(g, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part.gpkg")
    if tmp.exists():
        tmp.unlink()
    g.to_file(tmp, driver="GPKG")
    os.replace(tmp, path)


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
    sd = subdistricts()
    l3 = sd.reset_index()[["pc11_s_id", "pc11_d_id", "pc11_sd_id", "sd_name", "geometry"]]
    write_gpkg(gpd.GeoDataFrame(l3, geometry="geometry", crs=CRS).to_crs(4326), OUT_L3)
    # districts of REDRAW are the union of their new tehsils
    dist = gpd.read_file(DISTRICT_SHP).to_crs(CRS)
    redrawn = sorted(d for cfg in REDRAW.values() for d in cfg["districts"])
    for d in redrawn:
        k = dist.index[dist.pc11_d_id == d]
        if len(k):
            dist.loc[k[0], "geometry"] = shapely.union_all(sd[sd.pc11_d_id == d].geometry.values)
    write_gpkg(dist.to_crs(4326), OUT_L2)
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
    write_gpkg(out.to_crs(4326), OUT_GPKG)
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
