"""Iran - the 31 provinces (ostan), and a placement grid calibrated to the 429 counties.

Writes:
    data/geo/ir/ir_provinces.gpkg      the 31 counted units (`units`)
    data/geo/ir/ir_hexes.gpkg          Kontur 400 m hexes with `unit`, `county`, `kontur_pop` and
                                       `pop` (`place`); `pop` is calibrated to the census county
    data/geo/ir/ir_lookup.csv          province -> census 1395 population, Kontur 2023 population
    data/geo/ir/ir_county_lookup.csv   county -> census 1395 population, Kontur, scale factor

Usage:
    python sources/ir_geo.py --fetch    COD-AB shapefile zip (~11 MB), Kontur IR (~17 MB) and
                                        COD-PS ADM2 (~77 KB)
    python sources/ir_geo.py            rebuild from data/raw/ir/ and data/geo/kontur/

THE BOUNDARIES ARE OCHA COD-AB IRAN (`cod-ab-irn`, valid_on 2019-05-14, reviewed 2024-10), the
shapefile bundle read with `engine="fiona"`. ADM1 is the 31 provinces, which is the tier the 1395
census printed religion on: Alborz was split from Tehran in 2010, before both the 1390 and the 1395
censuses. ADM2 is the 429 counties, the same 429 the 1395 census counted.

THE PROVINCE JOIN USES TWO KEYS THAT HAVE TO AGREE. `sources/ir.py` carries an English name for each
province, written by hand beside the yearbook's Persian name; that English name must equal COD's
`adm1_name`, and COD's own Persian `adm1_name1` for the same polygon must equal the yearbook's
Persian name once Arabic yeh and kaf are folded to Persian. A third witness neither key determines:
each province's Kontur 2023 population against its census total, whose spread of log ratios must
sit below every one of 2,000 shuffled pairings.

KONTUR PUTS 18% OF IRAN IN THE WRONG COUNTY, SO THE GRID IS CALIBRATED TO THE CENSUS COUNTY.
Measured 2026-09-14. Kontur's Fars is 1.90x the census, and the excess is three rural counties south-
east of Shiraz: Sarvestan (census 38,114, Kontur 1,779,613), Kavar (83,883 against 1,788,778) and
Kherameh (54,864 against 1,677,708), each a false city of hexes at Kontur's 46,200/km2 limit, while
Shiraz county itself (1,869,001) gets 1,238,528. The same shape recurs across the country: Torghabe-
o-Shandiz outside Mashhad at 15.9x its census share of Razavi Khorasan, Bavi in Khuzestan 11.5x,
Sareyn in Ardabil 22.8x, Famenin in Hamadan 13.2x, and 78 blocks at the limit in all. Within their
provinces, 14.6 million people are placed in the wrong county. kontur_cap.py's per-block cap could not
fix this: capping a block lowers it to its ring's median, and the ring around a false city is its own
ramp, so most of the false weight would stay in Kavar.

The county figures are COD-PS Iran ADM2 2016 (`cod-ps-irn`, `irn_admpop_adm2_2016_v2.csv`), which
is the 1395 census: its 429 counties sum to every province total in SCI's Table 3-18 to the person,
men plus women close on every row, and its pcodes are exactly COD-AB's 429 polygons with the same
parents. So each hex keeps Kontur's shape inside its county and takes the county's census total:
`pop = kontur_pop x county census / county Kontur`. Placement only; no count moves, and the counts
stay at province. Spec §12's Haiti rule (a false commune is scaled, not capped) applied to every
county, because here the county figure is a census count rather than a projection.

The calibrated layer goes above Kontur's 46,200/km2 where a real core was under-weighted (Shiraz
county is scaled up 1.51x), so kontur_cap.py reads it as not raw Kontur and skips it, as it does
cn, kr and bg. What is left is Kontur's surface inside a county, which no county figure reaches.
"""

import csv
import gzip
import glob
import os
import shutil
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ir")
SHP_DIR = os.path.join(RAW, "cod_ab")
GEO = os.path.join(ROOT, "data", "geo", "ir")
KONTUR = os.path.join(ROOT, "data", "geo", "kontur")
NORM = os.path.join(ROOT, "data", "normalized", "ir.csv")

ZIP_NAME = "irn_admin_boundaries.shp.zip"
ZIP_URL = ("https://data.humdata.org/dataset/247b4026-79ff-4b16-95b9-0f366792d2cc/resource/"
           "3652644d-b572-4736-8d66-834c1cbc6bc7/download/irn_admin_boundaries.shp.zip")

PS_NAME = "irn_admpop_adm2_2016_v2.csv"
PS_URL = ("https://data.humdata.org/dataset/07f4ec78-42c7-4606-ae62-4f1bff918c45/resource/"
          "81700d7b-fbea-49ab-9972-c867231cb3c0/download/irn_admpop_adm2_2016_v2.csv")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_IR_20231101.gpkg.gz")
GZ_NAME = "kontur_population_IR_20231101.gpkg.gz"
GZ_SIZE = 17_300_063
GPKG_NAME = "kontur_population_IR_20231101.gpkg"

OUT_UNITS = os.path.join(GEO, "ir_provinces.gpkg")
OUT_HEXES = os.path.join(GEO, "ir_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "ir_lookup.csv")
OUT_COUNTY = os.path.join(GEO, "ir_county_lookup.csv")

UNITS = 31
COUNTIES = 429

# Census November 2016 against Kontur 2023-11: seven years at Iran's 1.24% a year between 1390 and
# 1395 (Table 3-17's totals) is about 1.09x. The bands are wide around it.
KONTUR_RATIO_MIN = 0.90
KONTUR_RATIO_MAX = 1.40
PROVINCE_RATIO_MIN = 0.70
PROVINCE_RATIO_MAX = 1.70

# A province over the band on RAW Kontur because the grid is wrong there, not because the join is.
# Fars's 1.90x is Sarvestan, Kavar and Kherameh (above); the calibration below removes it.
KONTUR_EXCESS = {"Fars": 2.00}

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

sys.path.insert(0, HERE)
from ir import fold  # noqa: E402  the same Persian fold the source module uses
from geo_checks import read_layer  # noqa: E402


def _get(url, dst, min_size, magic=None, size=None, head_text=None):
    import requests

    if os.path.exists(dst) and os.path.getsize(dst) > min_size:
        print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
        return
    print("GET", url)
    r = requests.get(url, timeout=1800, stream=True, headers=UA)
    r.raise_for_status()
    with open(dst + ".part", "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    with open(dst + ".part", "rb") as fh:
        head = fh.read(64)
    if magic is not None and not head.startswith(magic):
        raise SystemExit(f"{dst}: starts {head[:8]!r}, expected {magic!r}")
    if head_text is not None and not head.decode("utf-8-sig", "replace").startswith(head_text):
        raise SystemExit(f"{dst}: starts {head[:16]!r}, expected {head_text!r}")
    got = os.path.getsize(dst + ".part")
    if got < min_size or (size is not None and got != size):
        raise SystemExit(f"{dst}: {got:,} bytes (expected {size or f'over {min_size:,}'})")
    os.replace(dst + ".part", dst)
    print(f"  got {got:,} bytes")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(KONTUR, exist_ok=True)
    z = os.path.join(RAW, ZIP_NAME)
    _get(ZIP_URL, z, 5_000_000, magic=b"PK")
    with zipfile.ZipFile(z) as zf:
        zf.extractall(SHP_DIR)
    _get(PS_URL, os.path.join(RAW, PS_NAME), 50_000, head_text="ADM0_EN,")

    gz = os.path.join(KONTUR, GZ_NAME)
    gpkg = os.path.join(KONTUR, GPKG_NAME)
    _get(GZ_URL, gz, 10_000_000, magic=b"\x1f\x8b", size=GZ_SIZE)
    if not (os.path.exists(gpkg) and os.path.getsize(gpkg) > 10_000_000):
        with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def read_counties(units, census):
    """COD-PS ADM2 2016 and COD-AB ADM2, asserted to be the census's 429 counties on one pcode list."""
    import geopandas as gpd
    import pandas as pd

    ps = pd.read_csv(os.path.join(RAW, PS_NAME), dtype=str, encoding="utf-8-sig")
    ps.columns = [c.strip() for c in ps.columns]
    for c in ("T_TL", "M_TL", "F_TL"):
        ps[c] = ps[c].str.strip().astype(int)
    ps["ADM2_PCODE"] = ps["ADM2_PCODE"].str.strip()
    ps["ADM1_PCODE"] = ps["ADM1_PCODE"].str.strip()
    if len(ps) != COUNTIES or ps["ADM2_PCODE"].nunique() != COUNTIES:
        raise SystemExit(f"COD-PS ADM2: {len(ps)} rows, {ps['ADM2_PCODE'].nunique()} pcodes, "
                         f"expected {COUNTIES}")
    if not (ps["M_TL"] + ps["F_TL"] == ps["T_TL"]).all():
        raise SystemExit("COD-PS ADM2: men + women != total on some row")
    pcode_unit = dict(zip(units["adm1_pcode"], units["unit"]))
    ps["unit"] = ps["ADM1_PCODE"].map(pcode_unit)
    if ps["unit"].isna().any():
        raise SystemExit(f"COD-PS ADM2 parents not in COD-AB ADM1: "
                         f"{sorted(set(ps.loc[ps['unit'].isna(), 'ADM1_PCODE']))}")
    sums = ps.groupby("unit")["T_TL"].sum()
    off = {u: int(sums.get(u, 0) - census[u]) for u in census.index if sums.get(u, 0) != census[u]}
    if off:
        raise SystemExit(f"COD-PS ADM2 does not sum to the 1395 census by province: {off}")
    print(f"\nCOD-PS ADM2 2016: {COUNTIES} counties; they sum to every province total in Table 3-18 "
          "to the person, and men + women close on every row: it is the 1395 census by county")

    shp = glob.glob(os.path.join(SHP_DIR, "**", "*admin2.shp"), recursive=True)
    if len(shp) != 1:
        raise SystemExit(f"expected one *admin2.shp under {SHP_DIR}, found {shp}")
    a2 = gpd.read_file(shp[0], engine="fiona")
    if len(a2) != COUNTIES or set(a2["adm2_pcode"]) != set(ps["ADM2_PCODE"]):
        raise SystemExit(f"COD-AB ADM2 ({len(a2)} polygons) and COD-PS ADM2 disagree on pcodes: "
                         f"{sorted(set(a2['adm2_pcode']) ^ set(ps['ADM2_PCODE']))[:10]}")
    par = a2.set_index("adm2_pcode")["adm1_pcode"]
    wrong = [p for p, q in zip(ps["ADM2_PCODE"], ps["ADM1_PCODE"]) if par[p] != q]
    if wrong:
        raise SystemExit(f"COD-AB and COD-PS put these counties in different provinces: {wrong}")
    print(f"  COD-AB ADM2 has the same {COUNTIES} pcodes, each under the same province")
    a2 = a2.merge(ps[["ADM2_PCODE", "ADM2_EN", "unit", "T_TL"]], left_on="adm2_pcode",
                  right_on="ADM2_PCODE")
    return a2[["adm2_pcode", "ADM2_EN", "unit", "T_TL", "geometry"]].rename(
        columns={"adm2_pcode": "county", "ADM2_EN": "county_name", "unit": "county_unit",
                 "T_TL": "county_pop"})


def main():
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()
    shps = glob.glob(os.path.join(SHP_DIR, "**", "*admin1.shp"), recursive=True)
    gpkg = os.path.join(KONTUR, GPKG_NAME)
    if len(shps) != 1:
        raise SystemExit(f"expected one *admin1.shp under {SHP_DIR}, found {shps} - run --fetch")
    for p in (gpkg, NORM, os.path.join(RAW, PS_NAME)):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} - run sources/ir.py --fetch and sources/ir_geo.py --fetch")
    os.makedirs(GEO, exist_ok=True)

    # ---- 1. the counted units, from the normalised file
    df = pd.read_csv(NORM, keep_default_na=False, na_values=[""])
    census = df.groupby("geo_id")["count"].sum()
    persian = df.groupby("geo_id")["geo_name"].first()
    if len(census) != UNITS:
        raise SystemExit(f"{len(census)} units in ir.csv, expected {UNITS}")

    g = gpd.read_file(shps[0], engine="fiona")
    print(f"COD-AB Iran admin1: {len(g)} features, crs={g.crs}")
    if len(g) != UNITS or g["adm1_pcode"].nunique() != UNITS:
        raise SystemExit(f"{shps[0]} has {len(g)} features, expected {UNITS} unique pcodes")
    if "valid_on" in g.columns:
        print(f"  valid_on {sorted(set(map(str, g['valid_on'])))}")

    lut = {}
    for name in census.index:
        hits = g.index[g["adm1_name"] == name].tolist()
        if len(hits) != 1:
            raise SystemExit(f"census province {name!r} matched {len(hits)} COD polygons: "
                             f"{sorted(g['adm1_name'])}")
        lut[name] = hits[0]
    if len(set(lut.values())) != UNITS:
        raise SystemExit("two census provinces matched the same polygon")
    wrong = [(n, persian[n], g.loc[i, "adm1_name1"]) for n, i in lut.items()
             if fold(g.loc[i, "adm1_name1"]) != fold(persian[n])]
    if wrong:
        raise SystemExit(f"English key and Persian key disagree: {wrong}")
    print(f"  join: {UNITS}/{UNITS} provinces on COD's English adm1_name, and COD's Persian "
          f"adm1_name1 equals the yearbook's Persian name for every one of them")

    units = g.loc[list(lut.values())].copy()
    units["unit"] = list(lut.keys())
    units["census_pop"] = units["unit"].map(census).astype(int)
    units = units[["unit", "adm1_pcode", "adm1_name1", "census_pop", "geometry"]]
    bb = units.geometry.bounds
    if ((bb["maxx"] - bb["minx"]) > 30).any() or ((bb["maxy"] - bb["miny"]) > 30).any():
        raise SystemExit("a province's bounding box is over 30 degrees across")
    units.to_file(OUT_UNITS, layer="provinces", driver="GPKG")
    print(f"wrote {OUT_UNITS} ({len(units)} provinces)")

    counties = read_counties(units, census)

    # ---- 2. Kontur, joined on hex CENTROIDS to province and county
    hexes = read_layer(gpkg, "Kontur IR")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"\nKontur hexes: {len(hexes):,}, crs={hexes.crs}, population {hexes[popcol].sum():,.0f}")

    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)
    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)
    cj = gpd.sjoin(pts[["geometry"]], counties[["county", "county_unit", "geometry"]].to_crs(units.crs),
                   how="left", predicate="within")
    cj = cj[~cj.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"  hexes whose centroid is in no province: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%), dropped")
    if lost / pts[popcol].sum() > 0.01:
        raise SystemExit("over 1% of Kontur's people fall outside every province")
    keep = ~outside
    nocounty = keep & cj["county"].isna()
    if nocounty.any():
        raise SystemExit(f"{int(nocounty.sum())} hexes are in a province and in no county")
    clash = keep & (cj["county_unit"] != joined["unit"])
    if clash.any():
        raise SystemExit(f"{int(clash.sum())} hexes sit in a county of another province")

    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "county": cj.loc[keep, "county"].to_numpy(),
         "kontur_pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    per = out.groupby("unit")["kontur_pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"provinces with no populated hex: {missing}")

    tot = float(out["kontur_pop"].sum())
    census_total = int(units["census_pop"].sum())
    ratio = tot / census_total
    print(f"\n  Kontur 2023 {tot:,.0f} vs the 1395 census {census_total:,}: ratio {ratio:.3f}")
    if not KONTUR_RATIO_MIN <= ratio <= KONTUR_RATIO_MAX:
        raise SystemExit(f"ratio {ratio:.3f} outside [{KONTUR_RATIO_MIN}, {KONTUR_RATIO_MAX}]")

    # ---- 3. the province join witness, on raw Kontur
    kp = per["sum"].reindex(census.index).to_numpy(dtype=float)
    cp = census.to_numpy(dtype=float)
    r = kp / cp
    bad = [(u, round(x, 3)) for u, x in zip(census.index, r)
           if not PROVINCE_RATIO_MIN <= x <= KONTUR_EXCESS.get(u, PROVINCE_RATIO_MAX)]
    if bad:
        raise SystemExit(f"province Kontur/census ratios outside [{PROVINCE_RATIO_MIN}, "
                         f"{PROVINCE_RATIO_MAX}] (or a KONTUR_EXCESS bound): {bad}")
    spread = float(np.std(np.log(r)))
    rng = np.random.default_rng(1395)
    null = np.array([np.std(np.log(kp / rng.permutation(cp))) for _ in range(2000)])
    print(f"  per-province ratio {r.min():.3f}-{r.max():.3f}; spread of log ratio {spread:.3f} "
          f"against shuffled pairings {null.min():.3f} (min) / {np.median(null):.3f} (median)")
    if not spread < null.min():
        raise SystemExit("the real join's spread is not below every shuffled pairing")

    # ---- 4. calibrate each county to its census population
    kc = out.groupby("county")["kontur_pop"].agg(["size", "sum"])
    cc = counties.set_index("county")
    empty = sorted(set(cc.index) - set(kc.index[kc["sum"] > 0]))
    if empty:
        raise SystemExit(f"counties with no populated hex: {empty}")
    lk = cc[["county_name", "county_unit", "county_pop"]].join(kc)
    lk["factor"] = lk["county_pop"] / lk["sum"]
    kprov = lk.groupby("county_unit")["sum"].transform("sum")
    cprov = lk.groupby("county_unit")["county_pop"].transform("sum")
    lk["share_ratio"] = (lk["sum"] / kprov) / (lk["county_pop"] / cprov)
    wrong_county = float((lk["sum"] / kprov * cprov - lk["county_pop"]).clip(lower=0).sum())
    print(f"\n  Kontur places {wrong_county:,.0f} people ({100 * wrong_county / census_total:.1f}% "
          "of Iran) in the wrong county within their province")
    q = lk["share_ratio"].quantile([0, .05, .5, .95, 1]).to_numpy()
    print(f"  Kontur's county share over the census county share: min {q[0]:.2f}, p5 {q[1]:.2f}, "
          f"median {q[2]:.2f}, p95 {q[3]:.2f}, max {q[4]:.2f}")
    for pc, row in lk.sort_values("share_ratio", ascending=False).head(8).iterrows():
        print(f"    {row['county_unit']:<24} {row['county_name']:<20} census {row['county_pop']:>9,} "
              f"Kontur {row['sum']:>11,.0f}  x{row['factor']:.3f}")
    for pc, row in lk.sort_values("share_ratio").head(4).iterrows():
        print(f"    {row['county_unit']:<24} {row['county_name']:<20} census {row['county_pop']:>9,} "
              f"Kontur {row['sum']:>11,.0f}  x{row['factor']:.3f}")

    out["pop"] = out["kontur_pop"] * out["county"].map(lk["factor"]).to_numpy(dtype=float)
    after = out.groupby("unit")["pop"].sum()
    worst = float((after.reindex(census.index) / census - 1).abs().max())
    if worst > 1e-9:
        raise SystemExit(f"calibrated province weights miss the census by up to {worst:.2e}")
    dens = out["pop"].to_numpy() / (out.to_crs(6933).geometry.area.to_numpy() / 1e6)
    print(f"  calibrated: every province's weight equals its census total; the densest hex is "
          f"{dens.max():,.0f}/km2 (Kontur's own limit is 46,200), so kontur_cap.py reads this "
          "layer as not raw Kontur and skips it")

    out = out[["unit", "county", "kontur_pop", "pop", "geometry"]]
    out.to_file(OUT_HEXES, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT_HEXES} ({len(out):,} hexes)")

    with open(OUT_COUNTY, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["county", "county_name", "province", "census_pop_2016", "kontur_pop_2023",
                    "hexes", "factor", "kontur_share_over_census_share"])
        for pc, row in lk.sort_values(["county_unit", "county_name"]).iterrows():
            w.writerow([pc, row["county_name"], row["county_unit"], int(row["county_pop"]),
                        round(row["sum"], 1), int(row["size"]), round(row["factor"], 4),
                        round(row["share_ratio"], 3)])
    print(f"wrote {OUT_COUNTY}")

    pl = units[["unit", "adm1_pcode", "census_pop"]].merge(
        per.rename(columns={"size": "hexes", "sum": "kontur_pop"}),
        left_on="unit", right_index=True, how="left")
    pl["kontur_over_census"] = pl["kontur_pop"] / pl["census_pop"]
    with open(OUT_LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "adm1_pcode", "census_pop_2016", "kontur_pop_2023", "hexes",
                    "kontur_over_census"])
        for row in pl.sort_values("unit").itertuples(index=False):
            w.writerow([row.unit, row.adm1_pcode, int(row.census_pop), round(row.kontur_pop, 1),
                        int(row.hexes), round(row.kontur_over_census, 3)])
    print(f"wrote {OUT_LOOKUP}")


if __name__ == "__main__":
    main()
