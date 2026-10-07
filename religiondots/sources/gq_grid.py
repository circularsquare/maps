"""Equatorial Guinea: the placement layer, Kontur 400 m population hexagons keyed to province.

Writes data/geo/gq/gq_hexes.gpkg. Copied in shape from `sources/cu_grid.py`, without its
municipality layer; `sources/gq.md` §5 is the record.

THE JOIN IS ON HEX CENTROIDS. A hex whose centroid is outside every province is snapped to the
nearest one within `SNAP_KM` when it holds people (the coasts of Bioko, Annobón and Corisco), and
dropped beyond it. Those beyond are printed by distance band, since Kontur's `GQ` extract runs over
the borders with Cameroon and Gabon.

THREE CHECKS AGAINST THE 2015 CENSUS:

  * **the national ratio**, Kontur 2023 over the 2015 count, inside `EXPECTED_RATIO` +/-
    `TOLERANCE` (the US government's mid-2023 figure is 1.7 million, so Kontur reads high);
  * **the rank witness**: Kontur people per district against Tabla 3.1's 18 districts is not
    possible without a district layer, so the witness is per province, Spearman over 7 against
    every one of the 5,040 orderings;
  * **no town lost**: every GeoNames seat of `SEAT_MIN_POP` or more, Kontur within 5 and 10 km.

Every hex is then scaled to its province's census count.

Usage:
    python sources/gq_grid.py --fetch    Kontur GQ and GeoNames GQ.zip
    python sources/gq_grid.py            rebuild from data/raw/gq/
"""

import gzip
import itertools
import os
import shutil
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # kontur_cap
os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "gq")
GEO = os.path.join(ROOT, "data", "geo", "gq")
UNITS = os.path.join(GEO, "gq_provinces.gpkg")
OUT = os.path.join(GEO, "gq_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_GQ_20231101.gpkg.gz")
GZ_NAME = "kontur_population_GQ_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_GQ_20231101.gpkg"
GEONAMES_URL = "https://download.geonames.org/export/dump/GQ.zip"
GEONAMES = os.path.join(RAW, "geonames_GQ.zip")

EXPECTED_UNITS = 7
# Raw Kontur blocks reaching the cap, (lon, lat) of the peak -> (action, name). Filled after review.
# One block, 2026-10-03: 10 hexes and 191,402 raw Kontur people 3-4 km south-east of GeoNames'
# Malabo point, where Kontur puts most of the city (within 2 km of the GeoNames point it holds only
# 11,073). Capping it to its 3 km ring's median would take Malabo's weight down to about 10,000 and
# the province calibration would then spread the city over Baney and the rural hexes, below the
# census's own urban count (Bioko Norte 272,249 urban, preliminary Tabla 5.1). Left, as
# Afghanistan's Kabul and Herat were (sources/af_grid.py), with the urban bound asserted below.
BLOCKS = {
    (8.8148, 3.7316): ("left", "Malabo, south-east of the old centre"),
}
URBAN_BOUND = {(8.8148, 3.7316): ("GQ199", 272_249)}
EXPECTED_RATIO = 1.30
TOLERANCE = 0.30
SNAP_KM = 2.0
SNAP_BANDS_KM = (0.5, 1.0, 2.0, 5.0, 10.0, 25.0)

SEAT_MIN_POP = 5_000
HOLE_KM = 5.0
WIDE_KM = 10.0
HOLE_RATIO = 0.10
KONTUR_HOLES = set()

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
GEONAMES_COLS = ["geonameid", "name", "asciiname", "alternatenames", "lat", "lon", "fclass",
                 "fcode", "cc", "cc2", "admin1", "admin2", "admin3", "admin4", "population",
                 "elevation", "dem", "timezone", "modified"]
METRIC = "EPSG:32632"


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    if not os.path.exists(GEONAMES):
        req = urllib.request.Request(GEONAMES_URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        if data[:2] != b"PK":
            raise SystemExit(f"{GEONAMES_URL} is not a zip")
        with open(GEONAMES, "wb") as fh:
            fh.write(data)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 100_000:
        print("already have", gpkg)
        return
    if not os.path.exists(gz):
        print("GET", GZ_URL)
        r = requests.get(GZ_URL, timeout=1800, stream=True, headers={"User-Agent": UA})
        r.raise_for_status()
        with open(gz + ".part", "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        os.replace(gz + ".part", gz)
    with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")


def km(lat0, lon0, lat, lon):
    p = np.radians
    a = (np.sin(p(lat - lat0) / 2) ** 2
         + np.cos(p(lat0)) * np.cos(p(lat)) * np.sin(p(lon - lon0) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def seat_check(out, units, names, rel):
    import geopandas as gpd

    with zipfile.ZipFile(GEONAMES) as zf:
        t = pd.read_csv(zf.open("GQ.txt"), sep="\t", header=None, names=GEONAMES_COLS,
                        quoting=3, dtype=str, keep_default_na=False)
    s = t[t["fclass"] == "P"].copy()
    s["lat"], s["lon"] = s["lat"].astype(float), s["lon"].astype(float)
    s["population"] = pd.to_numeric(s["population"], errors="coerce").fillna(0).astype(int)
    pts = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s["lon"], s["lat"]), crs=4326)
    j = gpd.sjoin(pts, units[["unit", "geometry"]], how="inner", predicate="within")
    rows = []
    for _i, r in j[j["population"] >= SEAT_MIN_POP].iterrows():
        h = out[out["unit"] == r["unit"]]
        d = km(r["lat"], r["lon"], h["lat"].to_numpy(), h["lon"].to_numpy())
        rows.append((r["unit"], r["name"], int(r["population"]),
                     float(h.loc[d <= HOLE_KM, "kontur"].sum()), float(h.loc[d <= WIDE_KM, "kontur"].sum())))
    c = pd.DataFrame(rows, columns=["unit", "seat", "geonames", "kontur", "kontur_wide"])
    c["ratio"] = c["kontur"] / c["geonames"]
    c["ratio_wide"] = c["kontur_wide"] / c["geonames"]
    c["region"] = c["unit"].map(rel)
    c = c.sort_values("ratio")
    print(f"\n  Kontur within {HOLE_KM:g} and {WIDE_KM:g} km of each GeoNames place of {SEAT_MIN_POP:,}+, "
          f"all {len(c)}, with the province's own Kontur/census ratio over the national one:")
    for _i, r in c.iterrows():
        print(f"      {names[r['unit']]:<12} {r['seat']:<22} GeoNames {r['geonames']:>9,}   "
              f"Kontur {r['kontur']:>9,.0f} ({r['ratio']:.2f})  {r['kontur_wide']:>9,.0f} "
              f"({r['ratio_wide']:.2f})   province {r['region']:.2f}")
    holes = set(c.loc[c["ratio"] < HOLE_RATIO, "seat"])
    if holes != KONTUR_HOLES:
        raise SystemExit(f"places with under {HOLE_RATIO:.0%} of their people in Kontur within "
                         f"{HOLE_KM:g} km: {sorted(holes)}, not {sorted(KONTUR_HOLES)}")


def main():
    import geopandas as gpd
    from geo_checks import read_layer

    gpkg = os.path.join(RAW, GPKG_NAME)
    if "--fetch" in sys.argv or not os.path.exists(gpkg) or not os.path.exists(GEONAMES):
        fetch()
    if not os.path.exists(UNITS):
        raise SystemExit(f"missing {UNITS}; run sources/gq_geo.py first")

    hexes = read_layer(gpkg, "Kontur GQ")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, population {hexes[popcol].sum():,.0f}")

    units = gpd.read_file(UNITS)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{UNITS} has {len(units)} provinces, expected {EXPECTED_UNITS}")

    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()}, geometry=cent,
                           crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)

    u = units[["unit", "geometry"]]
    joined = gpd.sjoin(pts, u, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)
    outside = joined["unit"].isna().to_numpy()
    if outside.any():
        near = gpd.sjoin_nearest(pts.loc[outside].to_crs(METRIC), u.to_crs(METRIC),
                                 how="left", distance_col="d")
        near = near[~near.index.duplicated(keep="first")].reindex(pts.index[outside])
        print(f"\n  hexes whose centroid is outside every province: {int(outside.sum()):,} "
              f"({pts.loc[outside, popcol].sum():,.0f} people); people by distance to one:")
        for b in SNAP_BANDS_KM:
            m = near["d"] <= b * 1000
            print(f"      within {b:>4g} km: {int(m.sum()):>6,} hexes, "
                  f"{pts.loc[near.index[m], popcol].sum():>10,.0f} people")
        snap = near["d"] <= SNAP_KM * 1000
        joined.loc[near.index[snap], "unit"] = near.loc[snap, "unit"]
    dropped = joined["unit"].isna()
    print(f"  dropped: {int(dropped.sum()):,} hexes, {pts.loc[dropped, popcol].sum():,.0f} "
          f"people ({100.0 * pts.loc[dropped, popcol].sum() / pts[popcol].sum():.3f}%)")

    keep = ~dropped
    out = gpd.GeoDataFrame({"unit": joined.loc[keep, "unit"].to_numpy(),
                            "kontur": pts.loc[keep, popcol].to_numpy(dtype=float),
                            "lat": pts.loc[keep].geometry.y.to_numpy(),
                            "lon": pts.loc[keep].geometry.x.to_numpy()},
                           geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)
    out = out.reset_index(drop=True)

    est = dict(zip(units["unit"], units["pop"]))
    names = dict(zip(units["unit"], units["name"]))
    tot = float(out["kontur"].sum())
    ratio = tot / sum(est.values())
    print(f"\n  Kontur {tot:,.0f} vs census 2015 {sum(est.values()):,}: ratio {ratio:.3f} "
          f"(expected about {EXPECTED_RATIO})")
    if abs(ratio - EXPECTED_RATIO) > TOLERANCE:
        raise SystemExit("Kontur and the census disagree beyond the band")
    per_p = out.groupby("unit")["kontur"].sum()
    missing = sorted(set(est) - set(per_p.index))
    if missing:
        raise SystemExit(f"provinces with no populated hex: {missing}")
    rel_p = {x: (per_p[x] / est[x]) / ratio for x in sorted(est)}
    print("  per-province raw Kontur 2023 / census 2015, over the national ratio:")
    for x in sorted(rel_p, key=rel_p.get):
        print(f"      {names[x]:<12} {per_p[x]:>11,.0f}  {est[x]:>10,}  {rel_p[x]:5.2f}")

    keys = sorted(est)
    a = np.array([per_p[x] for x in keys])
    b = np.array([est[x] for x in keys], dtype=float)
    rho = stats.spearmanr(a, b).statistic
    perm = np.array([stats.spearmanr(a, np.array(p)).statistic for p in itertools.permutations(b)])
    beaten = int((perm >= rho - 1e-12).sum())
    print(f"  join witness: Spearman(Kontur, census) over 7 provinces = {rho:+.3f}; {beaten} of "
          f"{len(perm):,} orderings reach it (the true one among them)")
    if beaten > 10:
        raise SystemExit("the rank witness fails; the province join may be permuted")

    seat_check(out, units, names, rel_p)

    # ---- Kontur's density cap, read on the RAW layer (sources/mr_grid.py's method) ----
    import kontur_cap

    raw = out[["unit", "geometry"]].copy()
    raw["pop"] = out["kontur"].to_numpy()
    raw = raw.to_crs(4326)
    blk = kontur_cap.find_blocks(raw)
    at_cap = [idx for idx in blk["groups"] if (blk["dens"][idx] >= kontur_cap.AT_CAP).any()]
    print(f"\n  raw Kontur: densest hex {blk['dens'].max():,.0f}/km2; {len(blk['groups'])} blocks "
          f"over {kontur_cap.HI:,.0f}/km2, {len(at_cap)} of them reaching the cap of "
          f"{kontur_cap.CAP:,.0f}")
    weight = out["kontur"].to_numpy(dtype=float).copy()
    found, unlisted, left_blocks = set(), [], {}
    for idx in at_cap:
        pk = idx[np.argmax(blk["dens"][idx])]
        lon, lat = float(blk["lon"][pk]), float(blk["lat"][pk])
        print(f"      block of {len(idx)} hexes at ({lon:.4f}, {lat:.4f}), {blk['pop'][idx].sum():,.0f} "
              f"people, in {sorted(set(names[x] for x in out.loc[idx, 'unit']))}")
        key = next((k for k in BLOCKS if km(k[1], k[0], lat, lon) <= 1.0), None)
        if key is None:
            unlisted.append((lon, lat))
            continue
        found.add(key)
        action, name = BLOCKS[key]
        if action == "left":
            print(f"        {name}: left as Kontur has it (BLOCKS)")
            left_blocks[key] = idx
            continue
        if action != "capped":
            raise SystemExit(f"BLOCKS action {action!r} is neither `capped` nor `left`")
        inblock = np.zeros(len(weight), dtype=bool)
        inblock[idx] = True
        near = blk["tree"].query_ball_point(blk["xy"][idx], kontur_cap.RING_KM * 1000.0)
        ring = np.unique(np.concatenate([np.asarray(n, dtype=np.int64) for n in near]))
        ring = ring[~inblock[ring] & (weight[ring] > 0)]
        if len(ring) == 0:
            raise SystemExit(f"{name} has no populated hex in its ring")
        ceiling = float(np.median(blk["dens"][ring]))
        weight[idx] = np.minimum(weight[idx], ceiling * blk["area"][idx])
        print(f"        {name}: lowered to the {kontur_cap.RING_KM:g} km ring's median of "
              f"{ceiling:,.0f}/km2, now {weight[idx].sum():,.0f} people")
    if unlisted:
        raise SystemExit(f"raw Kontur blocks at the cap not in BLOCKS: {unlisted}; review them and name them there")
    if found != set(BLOCKS):
        raise SystemExit(f"BLOCKS rows matching no block at the cap: {sorted(set(BLOCKS) - found)}")

    # ---- calibrate every hex to its province's 2015 census count ----
    out["capped"] = weight
    per_c = out.groupby("unit")["capped"].sum()
    out["pop"] = weight * out["unit"].map(lambda x: est[x] / per_c[x]).to_numpy()
    dens = out["pop"].to_numpy() / (out.to_crs(6933).geometry.area.to_numpy() / 1e6)
    top = int(np.argmax(dens))
    print(f"  calibrated densest hex {dens.max():,.0f}/km2 in {names[out.loc[top, 'unit']]} "
          f"({'above' if dens.max() > kontur_cap.OVER_CAP else 'not above'} Kontur's limit, so "
          f"kontur_cap.py {'skips' if dens.max() > kontur_cap.OVER_CAP else 'checks'} this layer)")
    for key, idx in left_blocks.items():
        unit, urban = URBAN_BOUND[key]
        held = float(out.loc[idx, "pop"].sum())
        print(f"  {BLOCKS[key][1]}: the calibrated block holds {held:,.0f}, {names[unit]}'s urban "
              f"count is {urban:,}")
        if held > urban:
            raise SystemExit("a block left at Kontur's cap holds more than its province's urban count")
    chk = out.groupby("unit")["pop"].sum()
    worst = max(abs(chk[x] - est[x]) for x in est)
    if worst > 0.5:
        raise SystemExit(f"calibration leaves a province {worst:.2f} people off the census count")
    moved = 0.5 * sum(abs(per_p[x] / tot - est[x] / sum(est.values())) for x in est)
    print(f"  calibrated: every province sums to the 2015 census (worst {worst:.3f}); raw Kontur had "
          f"{moved:.1%} of its people in a different province from the census")

    os.makedirs(GEO, exist_ok=True)
    out[["unit", "pop", "geometry"]].to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} cells)")


if __name__ == "__main__":
    main()
