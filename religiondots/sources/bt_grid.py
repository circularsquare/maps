"""Bhutan: the placement layer, Kontur 400 m population hexagons calibrated to dzongkhag.

Writes data/geo/bt/bt_hexes.gpkg.

Bhutan is 38,400 km2 of mountain; Gasa, Bumthang, Lhuentse and Wangdue Phodrang are a third of it
and hold 11% of its people. Spread flat, a dzongkhag's dots would sit on glaciers.

THE JOIN IS ON HEX CENTROIDS, to COD-AB's 20 dzongkhags (`sources/bt_geo.py`), so a hex on a line
belongs to one side.

**Hexes whose centroid is outside Bhutan are dropped, not snapped** (`SNAP_KM = 0`). Bhutan has no
coast, so a hex outside is in India or China. The `BT` extract has 116 of them with 72,617 people,
all within 2 km of the line, and 15 beside Chhukha hold 52,177: that is Jaigaon, the Indian town
across the gate from Phuentsholing. Snapped, it made Chhukha read 1.83 of its census share over the
national ratio and would have pulled Chhukha's dots onto the border; dropped, Chhukha reads 1.31
and Samdrup Jongkhar 1.28, the two highest, and the share of Kontur's people in a different
dzongkhag from the census falls from 11.2% to 7.6% (2026-10-03, fafd1067-bt). Both still lean to
their border towns, which calibration cannot see inside a dzongkhag; left as is. The next largest groups are beside Samdrup Jongkhar (10
hexes, 8,144 people) and Samtse (52 hexes, 7,969), both border towns with an Indian twin.

## CALIBRATED TO THE 2017 CENSUS

Each hex is scaled so its dzongkhag's hexes add up to the census's 2017 population there, Bhutanese
and non-Bhutanese together (Table A2.8), so Kontur decides only where people are inside a
dzongkhag. Both halves of the country's dots are placed on this one layer.

## CHECKS

  * **the national ratio**, Kontur over the census, inside `EXPECTED_RATIO` +/- `TOLERANCE`.
    Written before the first run: Kontur 2023 is scaled to a modelled national figure for a later
    year than 2017, so a ratio a little above 1 is expected;
  * **the rank witness for the dzongkhag join**: Kontur people per dzongkhag against the census,
    Spearman, against `N_PERM` shuffles, and each dzongkhag's ratio printed;
  * **Kontur's density cap**, scanned on the raw layer before calibration; every block at the cap is
    named in `BLOCKS` with what is done to it.

Usage:
    python sources/bt_grid.py --fetch    one gzipped gpkg from Kontur
    python sources/bt_grid.py            rebuild from data/raw/bt/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # kontur_cap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bt")
GEO = os.path.join(ROOT, "data", "geo", "bt")
UNITS = os.path.join(GEO, "bt_dzongkhags.gpkg")
OUT = os.path.join(GEO, "bt_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_BT_20231101.gpkg.gz")
GZ_NAME = "kontur_population_BT_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_BT_20231101.gpkg"

EXPECTED_UNITS = 20
EXPECTED_RATIO = 1.10
TOLERANCE = 0.25
N_PERM = 20_000
SNAP_KM = 0.0
EXPECTED_DROPPED = (116, 72_617)     # hexes and Kontur people outside Bhutan, asserted
SNAP_BANDS_KM = (0.5, 1.0, 2.0, 5.0, 10.0)
# Raw Kontur blocks at the 46,200/km2 cap, by (lon, lat) of the block's peak, matched within 1 km.
# Every block at the cap must be named here: "capped" lowers it to its 3 km ring's median before
# calibration (kontur_cap.py's method), "left" keeps Kontur's shape.
BLOCKS = {}

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
METRIC = "EPSG:32646"


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
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


def main():
    import geopandas as gpd
    from geo_checks import read_layer

    gpkg = os.path.join(RAW, GPKG_NAME)
    if "--fetch" in sys.argv or not os.path.exists(gpkg):
        fetch()
    if not os.path.exists(UNITS):
        raise SystemExit(f"missing {UNITS}; run sources/bt_geo.py first")

    hexes = read_layer(gpkg, "Kontur BT")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, population {hexes[popcol].sum():,.0f}")

    units = gpd.read_file(UNITS)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{UNITS} has {len(units)} dzongkhags, expected {EXPECTED_UNITS}")

    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()}, geometry=cent,
                           crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)
    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    if outside.any():
        near = gpd.sjoin_nearest(pts.loc[outside].to_crs(METRIC),
                                 units[["unit", "geometry"]].to_crs(METRIC),
                                 how="left", distance_col="d")
        near = near[~near.index.duplicated(keep="first")]
        print(f"\n  hexes whose centroid is outside every dzongkhag: {int(outside.sum()):,} "
              f"({pts.loc[outside, popcol].sum():,.0f} people); people by distance:")
        for b in SNAP_BANDS_KM:
            m = near["d"] <= b * 1000
            print(f"      within {b:>4g} km: {int(m.sum()):>5,} hexes, "
                  f"{pts.loc[near.index[m], popcol].sum():>9,.0f} people")
        by_unit = pts.loc[near.index, popcol].groupby(near["unit"]).agg(["size", "sum"])
        print("  outside hexes by nearest dzongkhag: " + ", ".join(
            f"{u} {int(r['size'])} hexes {r['sum']:,.0f}" for u, r in
            by_unit.sort_values("sum", ascending=False).iterrows()))
        snapped = near["d"] <= SNAP_KM * 1000
        joined.loc[near.index[snapped], "unit"] = near.loc[snapped, "unit"]
        print(f"  snapped within {SNAP_KM:g} km: {int(snapped.sum()):,} hexes")
    dropped = joined["unit"].isna()
    print(f"  dropped (outside Bhutan): {int(dropped.sum()):,} hexes, "
          f"{pts.loc[dropped, popcol].sum():,.0f} people "
          f"({100.0 * pts.loc[dropped, popcol].sum() / pts[popcol].sum():.3f}%)")
    got = (int(dropped.sum()), round(float(pts.loc[dropped, popcol].sum())))
    if got != EXPECTED_DROPPED:
        raise SystemExit(f"dropped {got}, expected {EXPECTED_DROPPED}; the extract or COD changed")

    keep = ~dropped
    out = gpd.GeoDataFrame({"unit": joined.loc[keep, "unit"].to_numpy(),
                            "kontur": pts.loc[keep, popcol].to_numpy(dtype=float),
                            "lat": pts.loc[keep].geometry.y.to_numpy(),
                            "lon": pts.loc[keep].geometry.x.to_numpy()},
                           geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    per = out.groupby("unit")["kontur"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index[per["sum"] > 0]))
    if missing:
        raise SystemExit(f"dzongkhags with no populated hex: {missing}")
    census = dict(zip(units["unit"], units["pop"].astype(int)))
    names = dict(zip(units["unit"], units["name"]))
    tot = float(out["kontur"].sum())
    ratio = tot / sum(census.values())
    print(f"\n  Kontur {tot:,.0f} vs the 2017 census {sum(census.values()):,}: ratio {ratio:.3f} "
          f"(band {EXPECTED_RATIO - TOLERANCE:.2f} to {EXPECTED_RATIO + TOLERANCE:.2f})")
    if abs(ratio - EXPECTED_RATIO) > TOLERANCE:
        raise SystemExit("Kontur and the census disagree beyond the band")

    u = sorted(census)
    a = np.array([per.loc[x, "sum"] for x in u])
    b = np.array([census[x] for x in u], dtype=float)
    rho = stats.spearmanr(a, b).statistic
    rng = np.random.default_rng(0)
    perm = np.array([stats.spearmanr(a, rng.permutation(b)).statistic for _ in range(N_PERM)])
    beaten = int((perm >= rho).sum())
    print(f"  join witness: Spearman(Kontur, census) over {len(u)} dzongkhags = {rho:+.3f}; "
          f"{beaten} of {N_PERM:,} shuffles reach it (best {perm.max():+.3f})")
    if beaten:
        raise SystemExit("the rank witness fails; the dzongkhag join may be permuted")
    rel = {x: (per.loc[x, "sum"] / census[x]) / ratio for x in u}
    print("  per-dzongkhag Kontur / census, over the national ratio:")
    for x in sorted(rel, key=rel.get):
        print(f"      {x} {names[x]:<17} {int(per.loc[x, 'size']):>6,} hexes  census "
              f"{census[x]:>8,}  Kontur {per.loc[x, 'sum']:>9,.0f}  {rel[x]:5.2f}")
    moved = 0.5 * sum(abs(per.loc[x, "sum"] / tot - census[x] / sum(census.values())) for x in u)
    print(f"  share of Kontur's people in a different dzongkhag from the census: {moved:.1%}")

    # ---- Kontur's density cap, read on the RAW layer ----
    import kontur_cap

    out = out.reset_index(drop=True)
    raw = out[["unit", "geometry"]].copy()
    raw["pop"] = out["kontur"].to_numpy()
    raw = raw.to_crs(4326)
    blk = kontur_cap.find_blocks(raw)
    at_cap = [idx for idx in blk["groups"] if (blk["dens"][idx] >= kontur_cap.AT_CAP).any()]
    print(f"\n  raw Kontur: densest hex {blk['dens'].max():,.0f}/km2; {len(blk['groups'])} blocks "
          f"over {kontur_cap.HI:,.0f}/km2, {len(at_cap)} of them reaching the cap of "
          f"{kontur_cap.CAP:,.0f}")
    weight = out["kontur"].to_numpy(dtype=float).copy()
    found, unnamed = set(), []
    for idx in at_cap:
        pk = idx[np.argmax(blk["dens"][idx])]
        lon, lat = float(blk["lon"][pk]), float(blk["lat"][pk])
        u_in = out.loc[idx, "unit"]
        shares = ", ".join(f"{p} {out.loc[idx, 'kontur'][u_in == p].sum() / per.loc[p, 'sum']:.1%}"
                           for p in sorted(set(u_in)))
        print(f"      block of {len(idx)} hexes at ({lon:.4f}, {lat:.4f}), "
              f"{blk['pop'][idx].sum():,.0f} people; share of each dzongkhag's Kontur: {shares}")
        key = next((k for k in BLOCKS if km(k[1], k[0], lat, lon) <= 1.0), None)
        if key is None:
            unnamed.append((round(lon, 4), round(lat, 4)))
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
    if unnamed:
        raise SystemExit(f"raw Kontur blocks at the cap not in BLOCKS: {unnamed}; review each and "
                         "name it there")
    if found != set(BLOCKS):
        raise SystemExit(f"BLOCKS rows matching no block at the cap: {sorted(set(BLOCKS) - found)}")
    out["capped"] = weight
    per_c = out.groupby("unit")["capped"].sum()

    # ---- calibrate every hex to its dzongkhag's census count ----
    out["pop"] = out["capped"] * out["unit"].map(lambda x: census[x] / per_c[x])
    dens = out["pop"].to_numpy() / (out.to_crs(6933).geometry.area.to_numpy() / 1e6)
    print(f"  calibrated densest hex {dens.max():,.0f}/km2")
    chk = out.groupby("unit")["pop"].sum()
    worst = max(abs(chk[x] - census[x]) for x in u)
    if worst > 0.5:
        raise SystemExit(f"calibration leaves a dzongkhag {worst:.2f} people off its count")
    print(f"\n  calibrated: every dzongkhag's hexes sum to its 2017 count (worst {worst:.4f})")

    os.makedirs(GEO, exist_ok=True)
    out[["unit", "pop", "kontur", "geometry"]].to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} cells)")


if __name__ == "__main__":
    main()
