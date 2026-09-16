"""Mauritania: the placement layer, Kontur 400 m population hexagons calibrated to moughataa.

Writes data/geo/mr/mr_hexes.gpkg.

Mauritania is 1.03 million km2 and its 2023 census counts 29.4% of its people in Nouakchott's
three wilayas, which are 1,149 km2. Tiris Zemmour, Adrar and Tagant are 59% of the country and
hold 5.4%. Spread flat, a wilaya's dots would sit on the dunes; with Kontur an empty hex takes none.

THE JOIN IS ON HEX CENTROIDS, to COD-AB's 63 moughataas (`sources/mr_geo.py`), so a hex on a line
belongs to one side. Hexes whose centroid is outside every moughataa (the `MR` extract overruns the
coast and the borders with Senegal, Mali, Algeria and Western Sahara) are snapped to the nearest
moughataa within `SNAP_KM` and dropped beyond it.

## CALIBRATED TO THE CENSUS'S MOUGHATAA COUNTS

The census prints population by moughataa (Thème 1, Tableau A.1.4), a table 63 units deep beside
the 15 the religion counts use. Each hex is scaled so its moughataa's hexes add up to that count
(Iran's and Iceland's method, `playbooks/geography.md`), so Kontur decides only where people are
inside a moughataa. Kontur/census per moughataa is printed before and after.

## CHECKS

  * **the national ratio**, Kontur over the census, inside `EXPECTED_RATIO` +/- `TOLERANCE`. Both
    are 2023 and both count everyone living in the country;
  * **the rank witness for the moughataa join**: Kontur people per moughataa against the census,
    Spearman, against `N_PERM` shuffles of the census;
  * **no town lost**: every GeoNames seat (PPLA, PPLC) of `SEAT_MIN_POP` or more is compared with
    Kontur's people within `HOLE_KM`, gated on its moughataa's own ratio as in `sources/ly_grid.py`;
    `KONTUR_HOLES` names the holes;
  * **the Mbera camps**. Thème 1 (printed p.13) puts the camps at 33.4% of Bassiknou moughataa's
    123,337 people, about 41,200. Its communes (Tableau A.1.4) are Bassiknou 21,252, Elmegve 15,232,
    Vassala 79,508 and Dhar 7,345, so only Vassala (COD's `Vessale`, the border town of Fassala) is
    large enough to hold them. GeoNames' `Mbera` point (15.856N, 5.791W, population 58,985) lies in
    COD's El Megve and cannot be the camps as counted; Kontur has 6,332 people in that commune.
    So the check is Kontur's share of Bassiknou within `CAMP_KM` of GeoNames' `Vassala`, which has
    to be at least the camps' share and inside `CAMP_BAND`. Measured 2026-09-15 on COD's communes,
    not asserted here: Kontur holds 184,907 in Vessale against the census's 79,508, and 6,332 in El
    Megve against 15,232, so after calibration Vessale still carries about 30,000 of the
    moughataa's weight too many, placed within 60 km of where the census counted them.

Usage:
    python sources/mr_grid.py --fetch    one gzipped gpkg from Kontur (3.5 MB); GeoNames MR.zip
    python sources/mr_grid.py            rebuild from data/raw/mr/
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
RAW = os.path.join(ROOT, "data", "raw", "mr")
GEO = os.path.join(ROOT, "data", "geo", "mr")
MOUGHATAAS = os.path.join(GEO, "mr_moughataas.gpkg")
OUT = os.path.join(GEO, "mr_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_MR_20231101.gpkg.gz")
GZ_NAME = "kontur_population_MR_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_MR_20231101.gpkg"
GEONAMES_URL = "https://download.geonames.org/export/dump/MR.zip"
GEONAMES = os.path.join(RAW, "geonames_MR.zip")

EXPECTED_UNITS = 63
EXPECTED_RATIO = 1.0
TOLERANCE = 0.3
N_PERM = 20_000
SNAP_KM = 2.0
SNAP_BANDS_KM = (0.5, 1.0, 2.0, 5.0, 10.0)

SEAT_MIN_POP = 15_000
HOLE_KM = 5.0
WIDE_KM = 10.0
HOLE_RATIO = 0.10
LOW_RATIO = 0.5
KONTUR_HOLES = set()      # moughataa pcodes whose seat Kontur has lost; asserted
# Raw Kontur blocks at the 46,200/km2 cap, by (lon, lat) of the block's peak, matched within 1 km.
# Every block at the cap must be named here with what is done to it: "capped" lowers it to its
# ring's median before calibration (kontur_cap.py's method), "left" keeps Kontur's shape.
# Reviewed 2026-09-15:
BLOCKS = {
    # 10 hexes, 244,404 people: 86% of Toujounine's Kontur weight and 31% of Dar Naim's
    # (Nouakchott-Nord). Toujounine is one commune, 303,882 people on 56 km2, so no census figure
    # says where inside it they live. Capping was tried and is worse: the 3 km ring's median is
    # 2,378/km2, the block kept 17,989 people, and calibration then pushed Toujounine's edge hexes
    # to 76,322/km2. Left as Kontur has it; its dots stay inside Toujounine and Dar Naim.
    (-15.9374, 18.1289): ("left", "Toujounine and Dar Naim, Nouakchott-Nord"),
}

BASSIKNOU = "MR012"
VASSALA = (15.55710, -5.51600)  # GeoNames 4028 `Vassala`, PPL (Fassala)
CAMP_KM = 10.0
CAMP_CENSUS_SHARE = 0.334       # Thème 1 printed p.13, the camps' share of Bassiknou moughataa
CAMP_BAND = (0.334, 0.70)       # at least the camps; at most Vassala commune's 64.5% plus slack

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")

GEONAMES_COLS = ["geonameid", "name", "asciiname", "alternatenames", "lat", "lon", "fclass",
                 "fcode", "cc", "cc2", "admin1", "admin2", "admin3", "admin4", "population",
                 "elevation", "dem", "timezone", "modified"]
METRIC = "EPSG:32628"


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
        t = pd.read_csv(zf.open("MR.txt"), sep="\t", header=None, names=GEONAMES_COLS,
                        quoting=3, dtype=str, keep_default_na=False)
    s = t[t["fcode"].isin(["PPLA", "PPLC"])].copy()
    s["lat"], s["lon"] = s["lat"].astype(float), s["lon"].astype(float)
    s["population"] = pd.to_numeric(s["population"], errors="coerce").fillna(0).astype(int)
    pts = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s["lon"], s["lat"]), crs=4326)
    j = gpd.sjoin(pts, units[["moughataa", "geometry"]], how="inner", predicate="within")
    rows = []
    for _i, r in j[j["population"] >= SEAT_MIN_POP].iterrows():
        h = out[out["moughataa"] == r["moughataa"]]
        d = km(r["lat"], r["lon"], h["lat"].to_numpy(), h["lon"].to_numpy())
        rows.append((r["moughataa"], r["name"], int(r["population"]),
                     float(h.loc[d <= HOLE_KM, "kontur"].sum()),
                     float(h.loc[d <= WIDE_KM, "kontur"].sum())))
    c = pd.DataFrame(rows, columns=["moughataa", "seat", "geonames", "kontur", "kontur_wide"])
    c["ratio"] = c["kontur"] / c["geonames"]
    c["ratio_wide"] = c["kontur_wide"] / c["geonames"]
    c["m_ratio"] = c["moughataa"].map(rel)
    c = c.sort_values("ratio")
    print(f"\n  GeoNames: {len(s)} PPLA/PPLC places; Kontur within {HOLE_KM:g} and {WIDE_KM:g} km of "
          f"each of the {len(c)} of {SEAT_MIN_POP:,}+, with the moughataa's Kontur/census ratio:")
    for _i, r in c.iterrows():
        print(f"      {r['moughataa']}  {r['seat']:<16} GeoNames {r['geonames']:>9,}   Kontur "
              f"{r['kontur']:>9,.0f} ({r['ratio']:.2f})  {r['kontur_wide']:>9,.0f} "
              f"({r['ratio_wide']:.2f})   moughataa {r['m_ratio']:.2f}")
    holes = set(c.loc[(c["ratio"] < HOLE_RATIO) & (c["m_ratio"] < LOW_RATIO), "moughataa"])
    if holes != KONTUR_HOLES:
        raise SystemExit(f"seats with under {HOLE_RATIO:.0%} of their people in Kontur, in a "
                         f"moughataa under {LOW_RATIO} of its count: {sorted(holes)}, not "
                         f"{sorted(KONTUR_HOLES)}")


def camp_check(out):
    h = out[out["moughataa"] == BASSIKNOU]
    d = km(VASSALA[0], VASSALA[1], h["lat"].to_numpy(), h["lon"].to_numpy())
    share = float(h.loc[d <= CAMP_KM, "kontur"].sum() / h["kontur"].sum())
    print(f"\n  Mbera camps: Kontur holds {share:.1%} of Bassiknou within {CAMP_KM:g} km of Vassala "
          f"(Fassala), the only commune large enough to hold the camps' {CAMP_CENSUS_SHARE:.1%}")
    if not CAMP_BAND[0] <= share <= CAMP_BAND[1]:
        raise SystemExit(f"Kontur's share near Vassala is outside {CAMP_BAND}; read the docstring")


def main():
    import geopandas as gpd
    from geo_checks import read_layer

    gpkg = os.path.join(RAW, GPKG_NAME)
    if "--fetch" in sys.argv or not os.path.exists(gpkg) or not os.path.exists(GEONAMES):
        fetch()
    if not os.path.exists(MOUGHATAAS):
        raise SystemExit(f"missing {MOUGHATAAS}; run sources/mr_geo.py first")

    hexes = read_layer(gpkg, "Kontur MR")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, population {hexes[popcol].sum():,.0f}")

    units = gpd.read_file(MOUGHATAAS)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{MOUGHATAAS} has {len(units)} moughataas, expected {EXPECTED_UNITS}")

    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()}, geometry=cent,
                           crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)
    joined = gpd.sjoin(pts, units[["moughataa", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["moughataa"].isna()
    if outside.any():
        near = gpd.sjoin_nearest(pts.loc[outside].to_crs(METRIC),
                                 units[["moughataa", "geometry"]].to_crs(METRIC),
                                 how="left", distance_col="d")
        near = near[~near.index.duplicated(keep="first")]
        print(f"\n  hexes whose centroid is outside every moughataa: {int(outside.sum()):,} "
              f"({pts.loc[outside, popcol].sum():,.0f} people); people by distance:")
        for b in SNAP_BANDS_KM:
            m = near["d"] <= b * 1000
            print(f"      within {b:>4g} km: {int(m.sum()):>5,} hexes, "
                  f"{pts.loc[near.index[m], popcol].sum():>9,.0f} people")
        snapped = near["d"] <= SNAP_KM * 1000
        joined.loc[near.index[snapped], "moughataa"] = near.loc[snapped, "moughataa"]
        print(f"  snapped within {SNAP_KM:g} km: {int(snapped.sum()):,} hexes")
    dropped = joined["moughataa"].isna()
    print(f"  dropped (beyond {SNAP_KM:g} km of every moughataa): {int(dropped.sum()):,} hexes, "
          f"{pts.loc[dropped, popcol].sum():,.0f} people "
          f"({100.0 * pts.loc[dropped, popcol].sum() / pts[popcol].sum():.3f}%)")

    keep = ~dropped
    mo = joined.loc[keep, "moughataa"].to_numpy()
    out = gpd.GeoDataFrame({"moughataa": mo, "unit": [m[:4] for m in mo],
                            "kontur": pts.loc[keep, popcol].to_numpy(dtype=float),
                            "lat": pts.loc[keep].geometry.y.to_numpy(),
                            "lon": pts.loc[keep].geometry.x.to_numpy()},
                           geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    per = out.groupby("moughataa")["kontur"].agg(["size", "sum"])
    missing = sorted(set(units["moughataa"]) - set(per.index[per["sum"] > 0]))
    if missing:
        raise SystemExit(f"moughataas with no populated hex: {missing}")
    census = dict(zip(units["moughataa"], units["pop"].astype(int)))
    names = dict(zip(units["moughataa"], units["name"]))
    tot = float(out["kontur"].sum())
    ratio = tot / sum(census.values())
    print(f"\n  Kontur {tot:,.0f} vs RGPH 2023 {sum(census.values()):,}: ratio {ratio:.3f} "
          f"(expected about {EXPECTED_RATIO})")
    if abs(ratio - EXPECTED_RATIO) > TOLERANCE:
        raise SystemExit("Kontur and the census disagree beyond the band")

    # ---- the join witness: Kontur's rank of the moughataas against the census ----
    u = sorted(census)
    a = np.array([per.loc[x, "sum"] for x in u])
    b = np.array([census[x] for x in u], dtype=float)
    rho = stats.spearmanr(a, b).statistic
    rng = np.random.default_rng(0)
    perm = np.array([stats.spearmanr(a, rng.permutation(b)).statistic for _ in range(N_PERM)])
    beaten = int((perm >= rho).sum())
    print(f"  join witness: Spearman(Kontur, census) over {len(u)} moughataas = {rho:+.3f}; "
          f"{beaten} of {N_PERM:,} shuffles reach it (best {perm.max():+.3f})")
    if beaten:
        raise SystemExit("the rank witness fails; the moughataa join may be permuted")

    rel = {x: (per.loc[x, "sum"] / census[x]) / ratio for x in u}
    wil = out.groupby("unit")["kontur"].sum()
    wcen = units.groupby("unit")["pop"].sum()
    print("\n  per-wilaya Kontur / census, over the national ratio:")
    for w in sorted(wcen.index, key=lambda w: wil[w] / wcen[w]):
        print(f"      {w}  {wil[w] / wcen[w] / ratio:5.2f}")
    print("  per-moughataa Kontur / census, over the national ratio, lowest and highest eight:")
    order = sorted(rel, key=rel.get)
    for x in order[:8] + ["..."] + order[-8:]:
        if x == "...":
            print("      ...")
            continue
        print(f"      {x} {names[x]:<18} {int(per.loc[x, 'size']):>6,} hexes  census "
              f"{census[x]:>8,}  {rel[x]:5.2f}")
    moved = 0.5 * sum(abs(per.loc[x, "sum"] / tot - census[x] / sum(census.values())) for x in u)
    print(f"  share of Kontur's people in a different moughataa from the census: {moved:.1%}")

    seat_check(out, units, rel)
    camp_check(out)

    # ---- Kontur's density cap, read on the RAW layer ----
    # kontur_cap.apply skips a layer with any hex above Kontur's limit, and a calibrated layer can
    # go above it, so the scan for blocks at the cap is run here on Kontur's own figures, before
    # calibration, where it can still see them.
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
    found = set()
    for idx in at_cap:
        pk = idx[np.argmax(blk["dens"][idx])]
        lon, lat = float(blk["lon"][pk]), float(blk["lat"][pk])
        mo_in = out.loc[idx, "moughataa"]
        shares = ", ".join(f"{m} {out.loc[idx, 'kontur'][mo_in == m].sum() / per.loc[m, 'sum']:.0%}"
                           for m in sorted(set(mo_in)))
        print(f"      block of {len(idx)} hexes at ({lon:.4f}, {lat:.4f}), "
              f"{blk['pop'][idx].sum():,.0f} people; share of each moughataa's Kontur: {shares}")
        key = next((k for k in BLOCKS if km(k[1], k[0], lat, lon) <= 1.0), None)
        if key is None:
            raise SystemExit("a raw Kontur block at the cap is not in BLOCKS; review it and name it "
                             "there")
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
    if found != set(BLOCKS):
        raise SystemExit(f"BLOCKS rows matching no block at the cap: {sorted(set(BLOCKS) - found)}")
    out["capped"] = weight
    per_c = out.groupby("moughataa")["capped"].sum()

    # ---- calibrate every hex to its moughataa's census count ----
    out["pop"] = out["capped"] * out["moughataa"].map(lambda x: census[x] / per_c[x])
    dens = out["pop"].to_numpy() / (out.to_crs(6933).geometry.area.to_numpy() / 1e6)
    print(f"  calibrated densest hex {dens.max():,.0f}/km2 "
          f"({'above' if dens.max() > kontur_cap.OVER_CAP else 'not above'} Kontur's limit, so "
          f"kontur_cap.py {'skips' if dens.max() > kontur_cap.OVER_CAP else 'checks'} this layer)")
    chk = out.groupby("moughataa")["pop"].sum()
    worst = max(abs(chk[x] - census[x]) for x in u)
    if worst > 0.5:
        raise SystemExit(f"calibration leaves a moughataa {worst:.2f} people off its census count")
    wchk = out.groupby("unit")["pop"].sum()
    print(f"\n  calibrated: every moughataa's hexes sum to its census count (worst {worst:.4f}); "
          f"wilayas {', '.join(f'{w} {int(round(wchk[w])):,}' for w in sorted(wchk.index)[:3])} ...")

    os.makedirs(GEO, exist_ok=True)
    out[["unit", "moughataa", "pop", "kontur", "geometry"]].to_file(OUT, layer="hexes",
                                                                  driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} cells)")


if __name__ == "__main__":
    main()
