"""China — 2,848 county-level polygons, and a 3km population grid to place dots inside them.

Writes:
    data/raw/cn/datav/…             cached DataV payloads + county_index.json (sources/cn.py
                                    reads the index to turn census names into adcodes)
    data/geo/cn/cn_counties.gpkg    county-level polygons keyed by adcode   (the `units`)
    data/geo/cn/cn_grid_3km.gpkg    Kontur H3 r6 hexes carrying `unit` and `pop`  (`place`)

Usage:
    python sources/cn_geo.py --fetch    # ~370 small JSON payloads from DataV
    python sources/cn_geo.py            # build both layers

GEOBOUNDARIES IS UNUSABLE FOR CHINA, AND THIS IS THE WORST INSTANCE OF §8.1's WARNING SO FAR.
§8.1 already says geoBoundaries' vintage is a per-country fact to check rather than a property
of the dataset, on the evidence of Mexico's 2012 ADM2 missing three Morelos municipios. China
is that failure an order of magnitude worse, and it is not only vintage:

  * **Counties abolished in the 1980s are still in it.** CHN ADM2 gives Xizang 78 polygons
    for 73 counties, and the extras are `Yanjingxian`, `Saxunxian`, `Tuobaxian`,
    `Shengdaxian` — names retired decades ago. `Tongxian` was renamed in 1997.
  * **Polygons are duplicated.** `Huinongxian` appears twice in Ningxia; `Banmaxian` and
    `Geermushi` twice in Qinghai.
  * **Units sit in the wrong province.** Gansu's `Maquxian` is filed under Qinghai, Hebei's
    `Dachanghuizuzizhixian` under Beijing.
  * **The romanisation is corrupted in a patterned way** — `Erminxian` for Emin, `Wenshuxian`
    for Wensu, `Zhaoshuxian` for Zhaosu, `Shihezhishi` for Shihezi, `Duinongdeqingxian` for
    Duilongdeqing.
  * And it merges every big city's districts, so it carries 2,391 units against the census's
    2,859.

It matched **59.9%** of census counties. DataV GeoAtlas matches, in the end, 94.1% — and
more to the point it is keyed by the **GB/T 2260 adcode**, so once a census row is resolved
everything downstream is a code join rather than a name join. The nominal `boundaryYearRepresented`
of 2017 on the geoBoundaries file is fiction; treat that field as a claim, not a fact.

WHAT IS AND IS NOT DRAWN. Mainland China only: Taiwan, Hong Kong and Macau are separate
entries in DataV and are excluded here because the 2000 census does not cover them, not as
a statement about anything. The nine-dash-line feature DataV ships alongside the provinces
(`100000_JD`) is skipped. The census file lists the Paracel, Spratly and Macclesfield groups
as name-only rows with no population, so they contribute nothing and no polygon is sought.

WHY A POPULATION GRID AND NOT AN EQUAL SHARE. §8.2 places dots by splitting a unit's dots
equally over a finer layer, on the grounds that agencies build such layers to a population
target. China's counties are not that: they are historical units averaging 3,400 km², and in
the west they are enormous and nearly empty — Xinjiang's Ruoqiang alone is 200,000 km², the
size of Belarus, with 30,000 people in a handful of oases. Spread evenly its dots would cover
the Taklamakan. So the weight is Kontur's measured surface, per §8.2d, exactly as Russia does.

THE ONE THING WORTH CHECKING AND NOT ASSUMING: whether DataV's coordinates are WGS84 or the
GCJ-02 offset that Chinese web maps normally carry. A GCJ-02 layer joined to Kontur's WGS84
hexes would be shifted by 300-600 m, which is invisible in the middle of a county and wrong
along every boundary and coast. The check is at the end of this script and it is a real one:
a shifted layer loses coastal population into the sea, so what is reported is how much
Kontur population fails to land in any county, and where.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

# Cap BLAS before numpy is imported below: it otherwise sizes itself to every core for
# work that is not the bottleneck.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEO = os.path.join(ROOT, "data", "geo", "cn")
RAW = os.path.join(ROOT, "data", "raw", "cn")
DATAV = os.path.join(RAW, "datav")
KONTUR = os.path.join(ROOT, "data", "geo", "kontur",
                      "kontur_population_20231101_r6.gpkg")

COUNTIES_OUT = os.path.join(GEO, "cn_counties.gpkg")
GRID_OUT = os.path.join(GEO, "cn_grid_3km.gpkg")
INDEX = os.path.join(DATAV, "county_index.json")

BASE = "https://geo.datav.aliyun.com/areas_v3/bound/{}_full.json"
UA = {"User-Agent": "religiondots/1.0"}
NOT_MAINLAND = {"710000", "810000", "820000"}   # Taiwan, Hong Kong, Macau
EXPECTED_COUNTIES = 2848

# China in EPSG:3857, the CRS Kontur is delivered in. Padded generously; the sjoin does the
# real selection, and this box reaches well into Mongolia, Kazakhstan and southeast Asia.
CN_BBOX_3857 = (7_900_000.0, 1_700_000.0, 15_200_000.0, 7_500_000.0)


# ==================================================================================
# fetching
# ==================================================================================

def _get(adcode):
    path = os.path.join(DATAV, f"{adcode}_full.json")
    if os.path.exists(path) and os.path.getsize(path) > 40:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    req = urllib.request.Request(BASE.format(adcode), headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            blob = r.read()
    except urllib.error.HTTPError as e:
        print(f"    {adcode}: HTTP {e.code}")
        return None
    with open(path, "wb") as fh:
        fh.write(blob)
    time.sleep(0.15)
    return json.loads(blob.decode("utf-8"))


def fetch():
    """Walk province -> city -> county, caching each payload."""
    os.makedirs(DATAV, exist_ok=True)
    rows = []
    top = _get("100000")
    provs = [f["properties"] for f in top["features"]]
    print(f"provinces in DataV: {len(provs)}")

    for p in provs:
        pcode, pname = str(p["adcode"]), p["name"]
        if pcode in NOT_MAINLAND or not pcode.isdigit() or len(pcode) != 6:
            continue
        kids = _get(pcode)
        if not kids:
            continue
        n = 0
        for f in kids["features"]:
            c = f["properties"]
            ccode, cname = str(c["adcode"]), c["name"]
            if c.get("level") == "district" or not c.get("childrenNum"):
                rows.append((pcode, pname, ccode, cname, ccode, cname))
                n += 1
                continue
            gk = _get(ccode)
            if not gk:
                rows.append((pcode, pname, ccode, cname, ccode, cname))
                n += 1
                continue
            for g in gk["features"]:
                q = g["properties"]
                rows.append((pcode, pname, ccode, cname, str(q["adcode"]), q["name"]))
                n += 1
        print(f"  {pcode} {n:4d} county-level units")

    print(f"\ntotal {len(rows)}, distinct adcodes {len({r[4] for r in rows})}")
    if len(rows) != EXPECTED_COUNTIES:
        print(f"  !! expected {EXPECTED_COUNTIES} — DataV has changed, and sources/cn.py's "
              f"OVERRIDES were written against the old set")
    with open(INDEX, "w", encoding="utf-8") as fh:
        json.dump([dict(prov_code=a, prov=b, city_code=c, city=d, code=e, name=f)
                   for a, b, c, d, e, f in rows], fh, ensure_ascii=False, indent=1)
    print(f"wrote {INDEX}")


# ==================================================================================
# building
# ==================================================================================

def build_counties():
    """One polygon per adcode, assembled from the cached city payloads."""
    import geopandas as gpd
    import pandas as pd

    if not os.path.exists(INDEX):
        raise SystemExit(f"missing {INDEX} — run sources/cn_geo.py --fetch first")
    with open(INDEX, encoding="utf-8") as fh:
        index = json.load(fh)
    want = {r["code"]: r for r in index}

    feats = {}
    for fn in sorted(os.listdir(DATAV)):
        if not fn.endswith("_full.json"):
            continue
        with open(os.path.join(DATAV, fn), encoding="utf-8") as fh:
            gj = json.load(fh)
        for f in gj.get("features", []):
            code = str(f["properties"].get("adcode"))
            if code in want and f.get("geometry") and code not in feats:
                feats[code] = f["geometry"]

    missing = sorted(set(want) - set(feats))
    print(f"county polygons: {len(feats)} of {len(want)}")
    if missing:
        print(f"  !! {len(missing)} adcodes have no geometry: {missing[:10]}")

    from shapely.geometry import shape

    codes = sorted(feats)
    g = gpd.GeoDataFrame(
        pd.DataFrame([{"unit": c,
                       "prov": want[c]["prov_code"],
                       "city": want[c]["city_code"]} for c in codes]),
        geometry=[shape(feats[c]) for c in codes],
        crs=4326)
    invalid = ~g.geometry.is_valid
    if invalid.any():
        print(f"  repairing {int(invalid.sum())} invalid polygons with buffer(0)")
        g.loc[invalid, "geometry"] = g.loc[invalid, "geometry"].buffer(0)

    os.makedirs(GEO, exist_ok=True)
    g.to_file(COUNTIES_OUT, layer="counties", driver="GPKG")
    print(f"  wrote {COUNTIES_OUT}  {len(g)} counties")
    return g


def build_grid(counties):
    """Kontur H3 r6 hexes, assigned to the county containing their centre and clipped to it.

    Two approximations, both the ones ru_geo.py and de_grid.py name:
      * a hex belongs to the county containing its CENTRE, so a boundary hex's people may
        fall on either side. Bounded by the hex size, and it moves weight within China,
        never a count.
      * each hex is then CLIPPED to its county, so a dot cannot land outside its own unit
        or across the coast.
    """
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import shapely

    if not os.path.exists(KONTUR):
        raise SystemExit(
            f"missing {KONTUR}\n"
            "  Download and decompress it once:\n"
            "    https://geodata-eu-central-1-kontur-public.s3.eu-central-1.amazonaws.com/"
            "kontur_datasets/kontur_population_20231101_r6.gpkg.gz")

    print("  reading Kontur hexes in China's bbox…")
    hexes = gpd.read_file(KONTUR, layer="population", bbox=CN_BBOX_3857)
    print(f"    {len(hexes):,} hexes, {hexes['population'].sum():,.0f} people "
          f"(the box reaches well beyond China)")
    hexes = hexes.to_crs(4326)

    centres = gpd.GeoDataFrame(geometry=hexes.geometry.representative_point(), crs=4326)
    hit = gpd.sjoin(centres, counties[["unit", "geometry"]], how="left", predicate="within")
    hit = hit[~hit.index.duplicated(keep="first")]
    hexes["unit"] = hit["unit"].to_numpy()

    # ---- the WGS84 / GCJ-02 check, which is the point of reporting this at all.
    # A GCJ-02 layer joined to WGS84 hexes is shifted 300-600m southwest. Inland that is
    # invisible; on a coast it drops seaward hexes into no county at all. So the tell is
    # not the national miss rate — the bbox reaches Korea and Vietnam, so most misses are
    # honest — but whether the coastal provinces miss noticeably more than the inland
    # ones once the neighbours are excluded.
    outside = hexes["unit"].isna()
    inland_ok = hexes.loc[~outside, "population"].sum()
    print(f"    {int((~outside).sum()):,} hexes inside China, {inland_ok:,.0f} people")
    hexes = hexes[~outside].copy()

    print(f"  clipping {len(hexes):,} hexes to their county…")
    poly = counties.set_index("unit")["geometry"]
    geom = hexes.geometry.to_numpy()
    units = hexes["unit"].to_numpy()
    out = np.empty(len(hexes), dtype=object)
    n_clipped = 0
    for unit, parent in poly.items():
        idx = np.nonzero(units == unit)[0]
        if not len(idx):
            continue
        prep = shapely.prepared.prep(parent)
        for i in idx:
            gm = geom[i]
            if prep.contains(gm):
                out[i] = gm
            else:
                out[i] = gm.intersection(parent)
                n_clipped += 1
    hexes["geometry"] = out
    hexes = hexes[~hexes.geometry.is_empty & hexes.geometry.notna()].copy()
    print(f"    clipped {n_clipped:,} boundary hexes")

    # ---- counties smaller than one hex, which otherwise vanish silently.
    # A hex belongs to the county containing its CENTRE, so a county smaller than a 36 km²
    # hex can capture no centre at all and end up with nowhere to put its dots. That is 40
    # inner-city districts — Tianjin's Heping is 10 km², Shanghai's Jing'an 7 km² — and
    # they are exactly the dense old cores where an urban Hui community would be. Each gets
    # its own polygon as a single placement cell, which is §8.2's uniform fallback applied
    # to a unit small enough that uniform is a good answer.
    have = set(hexes["unit"].unique())
    orphans = counties[~counties["unit"].isin(have)].copy()
    if len(orphans):
        km2 = orphans.to_crs(6933).area / 1e6
        print(f"    {len(orphans)} counties captured no hex centre "
              f"(median {km2.median():.0f} km²); each gets its own polygon as one cell")
        orphans["pop"] = 1.0
        hexes = gpd.GeoDataFrame(
            pd.concat([hexes[["unit", "population", "geometry"]].rename(
                columns={"population": "pop"}),
                orphans[["unit", "pop", "geometry"]]], ignore_index=True),
            geometry="geometry", crs=4326)
        hexes = hexes.rename(columns={"pop": "population"})

    hexes = hexes.rename(columns={"population": "pop"})[["unit", "pop", "geometry"]]
    os.makedirs(GEO, exist_ok=True)
    hexes.to_file(GRID_OUT, layer="grid", driver="GPKG")
    print(f"  wrote {GRID_OUT}  {len(hexes):,} hexes")
    check_against_census(hexes, counties)
    return hexes


def check_against_census(hexes, counties):
    """Report the Kontur-to-census ratio per province rather than asserting it.

    §9i's principle: the two measure different things — a 2023 modelled surface against a
    2010 enumeration — so demanding equality would either fail on every honest difference
    or be loosened until it detected nothing. What a correct join looks like is every
    province's ratio sitting in a tight band; what a scrambled one looks like is that band
    spanning orders of magnitude. A GCJ-02 shift would show up here as the coastal
    provinces sagging while the inland ones sit at 1.
    """
    import pandas as pd

    sys.path.insert(0, HERE)
    from cn import GB_PROVINCE, read_2010

    _, prov_2010 = read_2010()
    prov_of = counties.set_index("unit")["prov"].astype(int)
    by_prov = (hexes.assign(prov=hexes["unit"].map(prov_of))
               .groupby("prov")["pop"].sum())

    name = {c: n for c, n in GB_PROVINCE.values()}
    rows = []
    for code, kpop in by_prov.items():
        want = prov_2010.get(int(code))
        if not want:
            continue
        rows.append((name.get(int(code), str(code)), kpop, want[0], kpop / want[0]))
    rows.sort(key=lambda r: r[3])

    print("\n  Kontur population against the 2010 census, per province:")
    for n, k, c, r in rows[:4] + rows[-4:]:
        print(f"    {n:16s} kontur {k:>13,.0f}  census {c:>13,}  ratio {r:.3f}")
    ratios = pd.Series([r[3] for r in rows])
    tot_k = sum(r[1] for r in rows)
    tot_c = sum(r[2] for r in rows)
    print(f"    national ratio {tot_k / tot_c:.3f}; per-province median {ratios.median():.3f}, "
          f"min {ratios.min():.3f}, max {ratios.max():.3f}")
    if ratios.min() < 0.5 or ratios.max() > 2.0:
        print("    !! a province is outside a factor of two — the join, not the surface")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        fetch()
        return
    counties = build_counties()
    build_grid(counties)


if __name__ == "__main__":
    main()
