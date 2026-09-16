"""Afghanistan: the placement layer, Kontur 400 m population hexagons calibrated to province.

Writes data/geo/af/af_hexes.gpkg.

Afghanistan is 643,000 km2 and most of it is mountain or desert: Helmand, Herat, Kandahar, Farah,
Nimroz and Ghor are 46% of the country's area and 19% of NSIA's settled people. Spread flat, a
province's dots would sit on the Registan and the Hindu Kush; with Kontur an empty hex takes none.

THE JOIN IS ON HEX CENTROIDS, to COD-AB's 34 provinces (`sources/af_geo.py`), so a hex on a line
belongs to one side. Hexes whose centroid is outside every province (the `AF` extract's edge along
the borders) are snapped to the nearest province within `SNAP_KM` and dropped beyond it.

## CALIBRATED TO NSIA'S PROVINCE ESTIMATES

Each hex is scaled so its province's hexes add up to NSIA's 1404 settled estimate (Iran's, Iceland's
and Mauritania's method, `playbooks/geography.md`), so Kontur decides only where people are inside a
province. NSIA also prints settled population by district (Tables 6 onwards, one per province), but
in Dari and Pashto only and over the de facto authorities' 457 district-level units, where COD-AB
draws AGCHO's 401; that join was not attempted (`sources/af.md` §5).

**Sar-e Pol is Kontur's low outlier and is kept.** Kontur holds 88,475 people there against NSIA's
678,598 (0.11 of it over the national ratio; the median hex holds 1 person). Its pattern inside the
province is not the problem: against COD-PS 2026's district figures, a separate model, only 5.9% of
Kontur's Sar-e Pol people sit in a different district (Spearman +0.86 over 7 districts), better than
the 10.9% NSIA-weighted mean across provinces. So calibration raises the level and keeps the shape.
Measured 2026-09-15 with a scratch script, not asserted here.

## CHECKS

  * **the national ratio**, Kontur over NSIA's settled population, inside `EXPECTED_RATIO` +/-
    `TOLERANCE`. Written before the first run: NSIA projects from a 2002-2005 household listing and
    runs below every modelled estimate (COD-PS 2026 is 1.39x it, `sources/af_geo.py`), and Kontur
    2023 is scaled to a modelled national figure, so the ratio is expected above 1. First run 1.211;
  * **the rank witness for the province join**: Kontur people per province against NSIA, Spearman,
    against `N_PERM` shuffles;
  * **no town lost**: every GeoNames provincial seat or capital (PPLA, PPLC) of `SEAT_MIN_POP` or
    more is compared with Kontur's people within `HOLE_KM`, gated on its province's own ratio as in
    `sources/mr_grid.py`; `KONTUR_HOLES` names the holes;
  * **Kontur's density cap**, scanned on the raw layer before calibration; every block at the cap
    is named in `BLOCKS` with what is done to it;
  * **a block left at the cap holds no more than its province's urban population** once
    calibrated (NSIA Table 3's urban column, which neither Kontur nor the cap decides).

Usage:
    python sources/af_grid.py --fetch    one gzipped gpkg from Kontur (10.8 MB); GeoNames AF.zip
    python sources/af_grid.py            rebuild from data/raw/af/
"""

import gzip
import os
import shutil
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # kontur_cap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "af")
GEO = os.path.join(ROOT, "data", "geo", "af")
PROVINCES = os.path.join(GEO, "af_provinces.gpkg")
LOOKUP = os.path.join(GEO, "af_lookup.csv")
OUT = os.path.join(GEO, "af_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_AF_20231101.gpkg.gz")
GZ_NAME = "kontur_population_AF_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_AF_20231101.gpkg"
GEONAMES_URL = "https://download.geonames.org/export/dump/AF.zip"
GEONAMES = os.path.join(RAW, "geonames_AF.zip")

EXPECTED_UNITS = 34
EXPECTED_RATIO = 1.15
TOLERANCE = 0.35
N_PERM = 20_000
SNAP_KM = 2.0
SNAP_BANDS_KM = (0.5, 1.0, 2.0, 5.0, 10.0)

SEAT_MIN_POP = 15_000
HOLE_KM = 5.0
WIDE_KM = 10.0
HOLE_RATIO = 0.10
LOW_RATIO = 0.5
KONTUR_HOLES = set()      # province pcodes whose seat Kontur has lost; asserted
# Raw Kontur blocks at the 46,200/km2 cap, by (lon, lat) of the block's peak, matched within 1 km.
# Every block at the cap must be named here with what is done to it: "capped" lowers it to its
# 3 km ring's median before calibration (kontur_cap.py's method), "left" keeps Kontur's shape.
# Reviewed 2026-09-15 (cb8b206e-af) against GeoNames places within 5 km and NSIA Table 3's urban
# column. The rule applied: a block is left only where capping would take a city below NSIA's own
# urban count for its province; every other block is one to five hexes in a ring of 2 to 2,106
# people per km2 with no settlement of its size nearby, which is Kontur's false city.
BLOCKS = {
    # 187 hexes, 4,583,840 people, 70.4% of Kabul's Kontur. Kabul city; NSIA's urban Kabul is
    # 5,361,333 (Table 7's city districts 5,333,284). Capping to the ring's 1,014/km2 leaves 158,469.
    (69.2253, 34.5142): ("left", "Kabul city"),
    # 41 hexes, 938,391 people, 26.8% of Herat's Kontur, 2.4 km from GeoNames' Herat (574,300).
    # NSIA's urban Herat province is 760,907; capping to the ring's 868/km2 leaves 29,044.
    (62.2221, 34.3371): ("left", "Herat city"),
    # Helmand, 4 hexes, 95,503 people at Gereshk (GeoNames 43,588), 84% of NSIA's urban Helmand
    # (112,991) in 3 km2 while Lashkar Gah is the larger town. Ring 247/km2; 174,531 within 10 km
    # raw, so Gereshk's surroundings keep about 79,000 after the cap.
    (64.5649, 31.8235): ("capped", "Gereshk, Helmand"),
    # Herat, 5 hexes, 166,462 people at Torghundi on the Turkmen border (GeoNames population 0).
    (62.2882, 35.2447): ("capped", "Torghundi, Herat"),
    # Herat, 1 hex, 37,634 people at Farsi district seat (GeoNames 0); ring 63/km2.
    (63.2414, 33.7801): ("capped", "Farsi, Herat"),
    # Ghor, 3 hexes, 86,505 people at Chaghcharan (Fayroz Koh, GeoNames 15,000); NSIA puts all of
    # Ghor's urban population at 9,128. Ring 2,106/km2.
    (65.2569, 34.5209): ("capped", "Chaghcharan, Ghor"),
    # Ghor, 1 hex, 37,945 people at Shahrak (GeoNames 15,967); ring 19/km2.
    (64.3057, 34.1031): ("capped", "Shahrak, Ghor"),
    # Ghor, 2 hexes, 52,419 people; nearest named places population 0; ring 23/km2.
    (64.8476, 33.6884): ("capped", "Pasaband area, Ghor"),
    # Ghor, 3 hexes, 92,726 people; ring 23/km2.
    (66.2808, 34.4942): ("capped", "Lal wa Sarjangal area, Ghor"),
    # Ghor, 2 hexes, 53,058 people at Tulak district seat (GeoNames 0); ring 13/km2.
    (63.7263, 33.9767): ("capped", "Tulak, Ghor"),
    # Daykundi, 1 hex, 38,177 people (6.1% of the province); ring 2/km2. NSIA: no urban Daykundi.
    (67.3777, 33.8164): ("capped", "Khak-e Folad area, Daykundi"),
    # Daykundi, 2 hexes, 51,568 people (8.2%); ring 4/km2.
    (65.9886, 33.8699): ("capped", "Shush area, Daykundi"),
    # Daykundi, 2 hexes, 58,607 people (9.3%); ring 6/km2.
    (66.1270, 34.2615): ("capped", "Watarmah area, Daykundi"),
    # Kandahar, 1 hex, 36,904 people in Panjwayi; ring 359/km2.
    (65.4342, 31.5853): ("capped", "Panjwayi, Kandahar"),
    # Faryab, 2 hexes, 53,524 people; ring 327/km2.
    (64.2900, 35.6839): ("capped", "Yangi Tashqul area, Faryab"),
    # Samangan, 1 hex, 39,172 people (5.5% of the province); ring 121/km2.
    (67.5216, 36.2423): ("capped", "Orlamish area, Samangan"),
    # Maidan Wardak, 2 hexes, 76,742 people (7.6%); ring 2/km2; NSIA urban Maidan Wardak 3,984.
    (67.8944, 34.1684): ("capped", "Zard Joy area, Maidan Wardak"),
    # Badakhshan, 1 hex, 39,703 people; ring 21/km2.
    (70.3054, 37.3476): ("capped", "Langar area, Badakhshan"),
}

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")

GEONAMES_COLS = ["geonameid", "name", "asciiname", "alternatenames", "lat", "lon", "fclass",
                 "fcode", "cc", "cc2", "admin1", "admin2", "admin3", "admin4", "population",
                 "elevation", "dem", "timezone", "modified"]
METRIC = "EPSG:32642"


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
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 1_000_000:
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


def seat_check(out, units, rel):
    """Kontur people near every GeoNames seat of SEAT_MIN_POP or more; holes gated on `rel`."""
    import geopandas as gpd

    with zipfile.ZipFile(GEONAMES) as zf:
        t = pd.read_csv(zf.open("AF.txt"), sep="\t", header=None, names=GEONAMES_COLS,
                        quoting=3, dtype=str, keep_default_na=False)
    s = t[t["fcode"].isin(["PPLA", "PPLC"])].copy()
    s["lat"], s["lon"] = s["lat"].astype(float), s["lon"].astype(float)
    s["population"] = pd.to_numeric(s["population"], errors="coerce").fillna(0).astype(int)
    pts = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s["lon"], s["lat"]), crs=4326)
    j = gpd.sjoin(pts, units[["unit", "geometry"]], how="inner", predicate="within")
    rows = []
    for _i, r in j[j["population"] >= SEAT_MIN_POP].iterrows():
        h = out[out["unit"] == r["unit"]]
        d = km(r["lat"], r["lon"], h["lat"].to_numpy(), h["lon"].to_numpy())
        rows.append((r["unit"], r["name"], int(r["population"]),
                     float(h.loc[d <= HOLE_KM, "kontur"].sum()),
                     float(h.loc[d <= WIDE_KM, "kontur"].sum())))
    c = pd.DataFrame(rows, columns=["unit", "seat", "geonames", "kontur", "kontur_wide"])
    c["ratio"] = c["kontur"] / c["geonames"]
    c["ratio_wide"] = c["kontur_wide"] / c["geonames"]
    c["u_ratio"] = c["unit"].map(rel)
    c = c.sort_values("ratio")
    print(f"\n  GeoNames: {len(s)} PPLA/PPLC places; Kontur within {HOLE_KM:g} and {WIDE_KM:g} km of "
          f"each of the {len(c)} of {SEAT_MIN_POP:,}+, with the province's Kontur/NSIA ratio:")
    for _i, r in c.iterrows():
        print(f"      {r['unit']}  {r['seat']:<22} GeoNames {r['geonames']:>9,}   Kontur "
              f"{r['kontur']:>9,.0f} ({r['ratio']:.2f})  {r['kontur_wide']:>9,.0f} "
              f"({r['ratio_wide']:.2f})   province {r['u_ratio']:.2f}")
    missing = sorted(set(units["unit"]) - set(c["unit"]))
    if missing:
        print(f"  provinces with no GeoNames seat of {SEAT_MIN_POP:,}+: {missing}")
    holes = set(c.loc[(c["ratio"] < HOLE_RATIO) & (c["u_ratio"] < LOW_RATIO), "unit"])
    if holes != KONTUR_HOLES:
        raise SystemExit(f"seats with under {HOLE_RATIO:.0%} of their people in Kontur, in a "
                         f"province under {LOW_RATIO} of its estimate: {sorted(holes)}, not "
                         f"{sorted(KONTUR_HOLES)}")


def main():
    import geopandas as gpd
    from geo_checks import read_layer

    gpkg = os.path.join(RAW, GPKG_NAME)
    if "--fetch" in sys.argv or not os.path.exists(gpkg) or not os.path.exists(GEONAMES):
        fetch()
    if not os.path.exists(PROVINCES) or not os.path.exists(LOOKUP):
        raise SystemExit(f"missing {PROVINCES} or {LOOKUP}; run sources/af_geo.py first")

    hexes = read_layer(gpkg, "Kontur AF")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, population {hexes[popcol].sum():,.0f}")

    units = gpd.read_file(PROVINCES)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{PROVINCES} has {len(units)} provinces, expected {EXPECTED_UNITS}")
    urban = pd.read_csv(LOOKUP).set_index("unit")["urban"].astype(int).to_dict()

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
        print(f"\n  hexes whose centroid is outside every province: {int(outside.sum()):,} "
              f"({pts.loc[outside, popcol].sum():,.0f} people); people by distance:")
        for b in SNAP_BANDS_KM:
            m = near["d"] <= b * 1000
            print(f"      within {b:>4g} km: {int(m.sum()):>5,} hexes, "
                  f"{pts.loc[near.index[m], popcol].sum():>9,.0f} people")
        snapped = near["d"] <= SNAP_KM * 1000
        joined.loc[near.index[snapped], "unit"] = near.loc[snapped, "unit"]
        print(f"  snapped within {SNAP_KM:g} km: {int(snapped.sum()):,} hexes")
    dropped = joined["unit"].isna()
    print(f"  dropped (beyond {SNAP_KM:g} km of every province): {int(dropped.sum()):,} hexes, "
          f"{pts.loc[dropped, popcol].sum():,.0f} people "
          f"({100.0 * pts.loc[dropped, popcol].sum() / pts[popcol].sum():.3f}%)")

    keep = ~dropped
    out = gpd.GeoDataFrame({"unit": joined.loc[keep, "unit"].to_numpy(),
                            "kontur": pts.loc[keep, popcol].to_numpy(dtype=float),
                            "lat": pts.loc[keep].geometry.y.to_numpy(),
                            "lon": pts.loc[keep].geometry.x.to_numpy()},
                           geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    per = out.groupby("unit")["kontur"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index[per["sum"] > 0]))
    if missing:
        raise SystemExit(f"provinces with no populated hex: {missing}")
    nsia = dict(zip(units["unit"], units["pop"].astype(int)))
    names = dict(zip(units["unit"], units["name"]))
    tot = float(out["kontur"].sum())
    ratio = tot / sum(nsia.values())
    print(f"\n  Kontur {tot:,.0f} vs NSIA 1404 settled {sum(nsia.values()):,}: ratio {ratio:.3f} "
          f"(band {EXPECTED_RATIO - TOLERANCE:.2f} to {EXPECTED_RATIO + TOLERANCE:.2f})")
    if abs(ratio - EXPECTED_RATIO) > TOLERANCE:
        raise SystemExit("Kontur and NSIA disagree beyond the band")

    # ---- the join witness: Kontur's rank of the provinces against NSIA ----
    u = sorted(nsia)
    a = np.array([per.loc[x, "sum"] for x in u])
    b = np.array([nsia[x] for x in u], dtype=float)
    rho = stats.spearmanr(a, b).statistic
    rng = np.random.default_rng(0)
    perm = np.array([stats.spearmanr(a, rng.permutation(b)).statistic for _ in range(N_PERM)])
    beaten = int((perm >= rho).sum())
    print(f"  join witness: Spearman(Kontur, NSIA) over {len(u)} provinces = {rho:+.3f}; "
          f"{beaten} of {N_PERM:,} shuffles reach it (best {perm.max():+.3f})")
    if beaten:
        raise SystemExit("the rank witness fails; the province join may be permuted")

    rel = {x: (per.loc[x, "sum"] / nsia[x]) / ratio for x in u}
    print("  per-province Kontur / NSIA, over the national ratio:")
    for x in sorted(rel, key=rel.get):
        print(f"      {x} {names[x]:<14} {int(per.loc[x, 'size']):>7,} hexes  NSIA "
              f"{nsia[x]:>9,}  Kontur {per.loc[x, 'sum']:>11,.0f}  {rel[x]:5.2f}")
    moved = 0.5 * sum(abs(per.loc[x, "sum"] / tot - nsia[x] / sum(nsia.values())) for x in u)
    print(f"  share of Kontur's people in a different province from NSIA: {moved:.1%}")

    seat_check(out, units, rel)

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
    found, unnamed, left = set(), [], []
    for idx in at_cap:
        pk = idx[np.argmax(blk["dens"][idx])]
        lon, lat = float(blk["lon"][pk]), float(blk["lat"][pk])
        u_in = out.loc[idx, "unit"]
        shares = ", ".join(f"{p} {out.loc[idx, 'kontur'][u_in == p].sum() / per.loc[p, 'sum']:.1%}"
                           for p in sorted(set(u_in)))
        print(f"      block of {len(idx)} hexes at ({lon:.4f}, {lat:.4f}), "
              f"{blk['pop'][idx].sum():,.0f} people; share of each province's Kontur: {shares}")
        key = next((k for k in BLOCKS if km(k[1], k[0], lat, lon) <= 1.0), None)
        if key is None:
            unnamed.append((round(lon, 4), round(lat, 4)))
            continue
        found.add(key)
        action, name = BLOCKS[key]
        if action == "left":
            print(f"        {name}: left as Kontur has it (BLOCKS)")
            left.append((idx, name, out.loc[pk, "unit"]))
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
    print(f"  capping moved {per['sum'].sum() - per_c.sum():,.0f} raw Kontur people out of the "
          f"capped blocks; calibration spreads them over the rest of each province")

    # ---- calibrate every hex to its province's NSIA estimate ----
    out["pop"] = out["capped"] * out["unit"].map(lambda x: nsia[x] / per_c[x])
    dens = out["pop"].to_numpy() / (out.to_crs(6933).geometry.area.to_numpy() / 1e6)
    print(f"  calibrated densest hex {dens.max():,.0f}/km2 "
          f"({'above' if dens.max() > kontur_cap.OVER_CAP else 'not above'} Kontur's limit, so "
          f"kontur_cap.py {'skips' if dens.max() > kontur_cap.OVER_CAP else 'checks'} this layer)")
    chk = out.groupby("unit")["pop"].sum()
    worst = max(abs(chk[x] - nsia[x]) for x in u)
    if worst > 0.5:
        raise SystemExit(f"calibration leaves a province {worst:.2f} people off its estimate")
    print(f"\n  calibrated: every province's hexes sum to NSIA's estimate (worst {worst:.4f})")

    # ---- a block left at the cap cannot hold more than its province's urban population ----
    for idx, name, unit in left:
        held = float(out.loc[idx, "pop"].sum())
        print(f"  {name}: {held:,.0f} calibrated people in the block, against NSIA's urban "
              f"{names[unit]} {urban[unit]:,} ({held / urban[unit]:.2f})")
        if held > urban[unit]:
            raise SystemExit(f"{name} holds more than {names[unit]}'s urban population; review BLOCKS")

    os.makedirs(GEO, exist_ok=True)
    out[["unit", "pop", "kontur", "geometry"]].to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} cells)")


if __name__ == "__main__":
    main()
