"""Gabon: the placement layer, Kontur 400 m population hexagons keyed to province.

Writes data/geo/ga/ga_hexes.gpkg. Copied in shape from `sources/gq_grid.py`; `sources/ga.md` §5 is
the record.

THE JOIN IS ON HEX CENTROIDS. A hex whose centroid is outside every province is snapped to the
nearest one within `SNAP_KM` when it holds people (the coast and the estuary islands), and dropped
beyond it: Kontur's `GA` extract runs over the borders with Equatorial Guinea, Cameroon and the
Republic of the Congo, and a hex outside the land border is the neighbour's town
(`playbooks/geography.md`, Bhutan). Hexes Equatorial Guinea's own layer already holds are counted.

CHECKS AGAINST THE 2026 CENSUS (all residents, since Kontur models everyone):

  * **the national ratio**, Kontur 2023 over the 2026 count, inside `EXPECTED_RATIO` +/-
    `TOLERANCE` (the 2026 count is about a million above the UN's and World Bank's estimates, so
    Kontur reads low);
  * **the rank witness**: Spearman over the 9 provinces against every one of the 362,880 orderings;
  * **per province**, raw Kontur over the census, over the national ratio, printed: this is where
    COD's Ogooué-Lolo line (`sources/ga_geo.py` `AREA_PINNED`) would show if it moved people;
  * **no town lost**: every GeoNames place of `SEAT_MIN_POP` or more, Kontur within 5 and 10 km.

Every hex is then scaled to its province's GABONESE count (`ga_lookup.csv` `pop`), which is what is
drawn. Kontur's weight inside a province includes its foreign residents, who live mostly in the
towns, so the citizens' dots lean a little urban; nothing finer than the province says by how much.

Usage:
    python sources/ga_grid.py --fetch    Kontur GA and GeoNames GA.zip
    python sources/ga_grid.py            rebuild from data/raw/ga/
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
RAW = os.path.join(ROOT, "data", "raw", "ga")
GEO = os.path.join(ROOT, "data", "geo", "ga")
UNITS = os.path.join(GEO, "ga_provinces.gpkg")
LOOKUP = os.path.join(GEO, "ga_lookup.csv")
OUT = os.path.join(GEO, "ga_hexes.gpkg")
GQ_PLACE = os.path.join(ROOT, "data", "geo", "gq", "gq_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_GA_20231101.gpkg.gz")
GZ_NAME = "kontur_population_GA_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_GA_20231101.gpkg"
GEONAMES_URL = "https://download.geonames.org/export/dump/GA.zip"
GEONAMES = os.path.join(RAW, "geonames_GA.zip")

EXPECTED_UNITS = 9
# Raw Kontur blocks reaching the cap, (lon, lat) of the peak -> (action, name). Filled after review.
BLOCKS = {}
EXPECTED_RATIO = 0.70
TOLERANCE = 0.25
SNAP_KM = 2.0
SNAP_BANDS_KM = (0.5, 1.0, 2.0, 5.0, 10.0, 25.0)

SEAT_MIN_POP = 5_000
HOLE_KM = 5.0
WIDE_KM = 10.0
HOLE_RATIO = 0.10
# Places under HOLE_RATIO within 5 km, reviewed 2026-10-03 (sources/ga.md §5):
#   Tsogni (13,725) and Oyam (30,100): villages whose GeoNames figures (edited 2025-11) cannot be a
#     town's; Kontur has 101 and 4,138 people within 10 km, and no source names a town there.
#   Gamba (12,565): the point is a rounded -2.65, 10.00; Kontur holds 7,466 within 10 km.
#   Akanda (41,524): the point is on the peninsula's tip; the commune is Libreville's northern
#     suburbs, and Kontur holds 491,678 within 20 km.
#   Ntoum (62,445): a real thin spot. Kontur holds 5,065 within 5 km and 14,745 within 20; no 2026
#     department or commune count exists to fill it from, so the province scaling places Ntoum's
#     people in proportion to the rest of Estuaire, mostly Libreville, about 35 km west.
KONTUR_HOLES = {"Tsogni", "Oyam", "Gamba", "Akanda", "Ntoum"}

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
GEONAMES_COLS = ["geonameid", "name", "asciiname", "alternatenames", "lat", "lon", "fclass",
                 "fcode", "cc", "cc2", "admin1", "admin2", "admin3", "admin4", "population",
                 "elevation", "dem", "timezone", "modified"]
METRIC = "EPSG:32732"


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
        t = pd.read_csv(zf.open("GA.txt"), sep="\t", header=None, names=GEONAMES_COLS,
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
        print(f"      {names[r['unit']]:<16} {r['seat']:<22} GeoNames {r['geonames']:>9,}   "
              f"Kontur {r['kontur']:>9,.0f} ({r['ratio']:.2f})  {r['kontur_wide']:>9,.0f} "
              f"({r['ratio_wide']:.2f})   province {r['region']:.2f}")
    holes = set(c.loc[c["ratio"] < HOLE_RATIO, "seat"])
    if holes != KONTUR_HOLES:
        raise SystemExit(f"places with under {HOLE_RATIO:.0%} of their people in Kontur within "
                         f"{HOLE_KM:g} km: {sorted(holes)}, not {sorted(KONTUR_HOLES)}")


def neighbour_overlap(out):
    """Hexes Equatorial Guinea's place layer already holds (centroids within 10 m), printed."""
    import geopandas as gpd

    if not os.path.exists(GQ_PLACE):
        print("  (no gq_hexes.gpkg to compare against)")
        return
    gq = gpd.read_file(GQ_PLACE).to_crs(METRIC)
    a = gpd.GeoDataFrame(geometry=out.geometry.centroid.to_crs(METRIC) if out.crs.is_projected
                         else out.to_crs(METRIC).geometry.centroid, crs=METRIC)
    b = gpd.GeoDataFrame({"gq_pop": gq["pop"].to_numpy()}, geometry=gq.geometry.centroid, crs=METRIC)
    j = gpd.sjoin_nearest(a, b, how="inner", max_distance=10.0)
    j = j[~j.index.duplicated()]
    print(f"  hexes also in Equatorial Guinea's place layer: {len(j):,} "
          f"({out.loc[j.index, 'kontur'].sum():,.0f} raw Kontur people here)")
    return j.index


def main():
    import geopandas as gpd
    from geo_checks import read_layer

    gpkg = os.path.join(RAW, GPKG_NAME)
    if "--fetch" in sys.argv or not os.path.exists(gpkg) or not os.path.exists(GEONAMES):
        fetch()
    if not os.path.exists(UNITS):
        raise SystemExit(f"missing {UNITS}; run sources/ga_geo.py first")

    hexes = read_layer(gpkg, "Kontur GA")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, population {hexes[popcol].sum():,.0f}")

    units = gpd.read_file(UNITS)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{UNITS} has {len(units)} provinces, expected {EXPECTED_UNITS}")
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})

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
    held = neighbour_overlap(out)
    if held is not None and len(held):
        # Left to the neighbour, which already places dots on them (Somalia's rule, so_grid.py).
        out = out.drop(index=held).reset_index(drop=True)
        print(f"  left to Equatorial Guinea: {len(held)} hexes dropped here")

    total = dict(zip(lut["geo_id"], lut["total_2026"]))
    est = dict(zip(units["unit"], units["pop"]))
    names = dict(zip(units["unit"], units["name"]))
    tot = float(out["kontur"].sum())
    ratio = tot / sum(total.values())
    print(f"\n  Kontur {tot:,.0f} vs census 2026 {sum(total.values()):,} (all residents): ratio "
          f"{ratio:.3f} (expected about {EXPECTED_RATIO})")
    if abs(ratio - EXPECTED_RATIO) > TOLERANCE:
        raise SystemExit("Kontur and the census disagree beyond the band")
    per_p = out.groupby("unit")["kontur"].sum()
    missing = sorted(set(est) - set(per_p.index))
    if missing:
        raise SystemExit(f"provinces with no populated hex: {missing}")
    rel_p = {x: (per_p[x] / total[x]) / ratio for x in sorted(est)}
    print("  per-province raw Kontur 2023 / census 2026, over the national ratio:")
    for x in sorted(rel_p, key=rel_p.get):
        print(f"      {names[x]:<16} {per_p[x]:>11,.0f}  {total[x]:>10,}  {rel_p[x]:5.2f}")

    keys = sorted(est)
    a = np.array([per_p[x] for x in keys])
    b = np.array([total[x] for x in keys], dtype=float)
    rho = stats.spearmanr(a, b).statistic
    perm = np.array([stats.spearmanr(a, np.array(p)).statistic for p in itertools.permutations(b)])
    beaten = int((perm >= rho - 1e-12).sum())
    print(f"  join witness: Spearman(Kontur, census) over 9 provinces = {rho:+.3f}; {beaten} of "
          f"{len(perm):,} orderings reach it (the true one among them)")
    # Bar: 1 in 1,000 orderings. Measured 2026-10-03: +0.933, 136 of 362,880 (Haut-Ogooué's
    # Kontur excess is the one big disorder; the province join is by p-code and name in ga_geo.py).
    if beaten > len(perm) // 1000:
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
    found, unlisted = set(), []
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

    # ---- calibrate every hex to its province's Gabonese count ----
    out["capped"] = weight
    per_c = out.groupby("unit")["capped"].sum()
    out["pop"] = weight * out["unit"].map(lambda x: est[x] / per_c[x]).to_numpy()
    dens = out["pop"].to_numpy() / (out.to_crs(6933).geometry.area.to_numpy() / 1e6)
    top = int(np.argmax(dens))
    print(f"  calibrated densest hex {dens.max():,.0f}/km2 in {names[out.loc[top, 'unit']]} "
          f"({'above' if dens.max() > kontur_cap.OVER_CAP else 'not above'} Kontur's limit, so "
          f"kontur_cap.py {'skips' if dens.max() > kontur_cap.OVER_CAP else 'checks'} this layer)")
    chk = out.groupby("unit")["pop"].sum()
    worst = max(abs(chk[x] - est[x]) for x in est)
    if worst > 0.5:
        raise SystemExit(f"calibration leaves a province {worst:.2f} people off its Gabonese count")
    moved = 0.5 * sum(abs(per_p[x] / tot - total[x] / sum(total.values())) for x in est)
    print(f"  calibrated: every province sums to its Gabonese count (worst {worst:.3f}); raw Kontur "
          f"had {moved:.1%} of its people in a different province from the 2026 census")

    os.makedirs(GEO, exist_ok=True)
    out[["unit", "pop", "geometry"]].to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} cells)")


if __name__ == "__main__":
    main()
