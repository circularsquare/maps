"""Senegal — the 1988 census units and the placement grid.

Writes:
    data/geo/sn/sn_units.gpkg      the 12 counted units (9 régions of 1988, Diourbel's 3 départements)
    data/geo/sn/sn_hexes.gpkg      Kontur 400 m hexes with `unit` and `pop` (`place`)
    data/geo/sn/sn_lookup.csv      unit -> 1988 census population, Kontur population, areas

Usage:
    python sources/sn_geo.py --fetch    COD-AB shapefile zip (~20 MB) + Kontur SN (~5 MB)
    python sources/sn_geo.py            rebuild from data/raw/sn/

THE UNITS ARE 1988's, REBUILT FROM COD-AB SENEGAL v02 (valid 2024-05-20) BY PCODE. Senegal had ten
régions in 1988 and has fourteen now: Matam was cut from Saint-Louis in 2002, and Kaffrine (from
Kaolack), Kédougou (from Tambacounda) and Sédhiou (from Kolda) in 2008. Each 1988 région is the
union of its successors' ADM1 polygons. Diourbel is drawn at its three départements (Bambey,
Diourbel, Mbacké), which are unchanged in name and number since 1988 and are ADM2 SN0201-SN0203.
There is no name join: every pcode's COD name and parent are asserted against the list below.

THE AREAS ARE THE WITNESS, AND FIVE UNITS DO NOT MATCH 1988's. Tableau 1.2 of the national report
prints each région's 1988 population and density in whole people per km2, which bounds its area;
Tableau 1.2 of the Diourbel report prints its départements' areas. Against those, COD's Louga is
8.5% smaller than its band, Fatick 11% smaller and Saint-Louis with Matam 5% larger, and inside
Diourbel the Diourbel département is 9.5% larger and Mbacké 22% larger while Bambey matches
(-1.3%). The national total matches (196,794 km2 against 196,722). So some border land has moved
between régions since 1988 (the Ferlo between Louga and Matam, and land around Mbacké), or the 1988
areas were measured on older maps; no 1988 boundary file exists to say which. The mismatched units
are pinned in AREA_PINNED so a changed boundary file or transcription fails loudly.

KONTUR IS THE PLACEMENT WEIGHT, AND IT IS 35 YEARS NEWER THAN THE COUNTS. Kontur 2023-11 inside a
1988 unit places that unit's 1988 dots where people live now. That matters most in Mbacké, where
Touba has grown several times over since 1988, and in Dakar's outer départements. The counts per
unit are 1988's and do not move; only the placement inside a unit is modern.
"""

import csv
import gzip
import os
import shutil
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
RAW = os.path.join(ROOT, "data", "raw", "sn")
SHP_DIR = os.path.join(RAW, "shp")
GEO = os.path.join(ROOT, "data", "geo", "sn")
KONTUR = os.path.join(ROOT, "data", "geo", "kontur")
NORM = os.path.join(ROOT, "data", "normalized", "sn.csv")

ZIP_NAME = "sen_admin_boundaries.shp.zip"
ZIP_URL = ("https://data.humdata.org/dataset/bd9bc484-155d-41a3-87cf-064310a94492/resource/"
           "3d0110df-e2c9-4546-bf0f-2770c04641cc/download/sen_admin_boundaries.shp.zip")
GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_SN_20231101.gpkg.gz")
GZ_NAME = "kontur_population_SN_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_SN_20231101.gpkg"

OUT_UNITS = os.path.join(GEO, "sn_units.gpkg")
OUT_HEXES = os.path.join(GEO, "sn_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "sn_lookup.csv")

# COD-AB v02 pcodes -> the names and parents COD must carry. Asserted, so a renumbered release
# stops here rather than handing Kaffrine's polygon to the wrong 1988 région.
ADM1_NAMES = {
    "SN01": "Dakar", "SN02": "Diourbel", "SN03": "Fatick", "SN04": "Kaffrine", "SN05": "Kaolack",
    "SN06": "Kédougou", "SN07": "Kolda", "SN08": "Louga", "SN09": "Matam", "SN10": "Saint-Louis",
    "SN11": "Sédhiou", "SN12": "Tambacounda", "SN13": "Thiès", "SN14": "Ziguinchor",
}
REGIONS_1988 = {                      # 1988 région -> its successors today
    "Dakar": ["SN01"],
    "Ziguinchor": ["SN14"],
    "Saint-Louis": ["SN10", "SN09"],  # Matam cut out in 2002
    "Tambacounda": ["SN12", "SN06"],  # Kédougou cut out in 2008
    "Kaolack": ["SN05", "SN04"],      # Kaffrine cut out in 2008
    "Thiès": ["SN13"],
    "Louga": ["SN08"],
    "Fatick": ["SN03"],
    "Kolda": ["SN07", "SN11"],        # Sédhiou cut out in 2008
}
DEPTS_1988 = {"Bambey": "SN0201", "Diourbel": "SN0202", "Mbacké": "SN0203"}
UNITS = len(REGIONS_1988) + len(DEPTS_1988)

# National report Tableau 1.2: 1988 population and density (whole people per km2), so each
# région's area lies in [pop / (d + 0.5), pop / (d - 0.5)]. Diourbel is here for its total.
DENSITY_1988 = {
    "Dakar": (1_488_941, 2707), "Ziguinchor": (398_337, 54), "Diourbel": (619_245, 142),
    "Saint-Louis": (660_282, 15), "Tambacounda": (385_982, 6), "Kaolack": (811_258, 51),
    "Thiès": (941_151, 143), "Louga": (490_077, 17), "Fatick": (509_702, 64),
    "Kolda": (591_833, 28),
}
# Diourbel report Tableau 1.2: superficie by département, km2 (région 4,359).
AREA_DEPTS_1988 = {"Bambey": 1351, "Diourbel": 1175, "Mbacké": 1833}
# Units whose COD area sits more than 3% outside the 1988 figure: measured 2026-09-15, see the
# docstring. The run asserts this set exactly.
AREA_PINNED = {"Louga", "Fatick", "Saint-Louis", "Diourbel", "Mbacké"}
AREA_TOL = 0.03
AREA_WORST = 0.25

# 1988 census (27 May) against Kontur 2023-11: Senegal's population was about 2.6 times 1988's.
KONTUR_RATIO_MIN = 1.8
KONTUR_RATIO_MAX = 3.5
SNAP_M = 500                          # a coastal hex centroid just offshore joins its nearest unit

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}


def _get(url, dst, magic, min_size):
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
        head = fh.read(len(magic))
    if head != magic:
        raise SystemExit(f"{dst}: starts {head!r}, expected {magic!r}")
    os.replace(dst + ".part", dst)
    print(f"  got {os.path.getsize(dst):,} bytes")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(KONTUR, exist_ok=True)
    z = os.path.join(RAW, ZIP_NAME)
    _get(ZIP_URL, z, b"PK", 10_000_000)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(SHP_DIR)
    gz = os.path.join(KONTUR, GZ_NAME)
    gpkg = os.path.join(KONTUR, GPKG_NAME)
    _get(GZ_URL, gz, b"\x1f\x8b", 4_000_000)
    if not (os.path.exists(gpkg) and os.path.getsize(gpkg) > 4_000_000):
        with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def build_units(census):
    import geopandas as gpd
    import pandas as pd
    from geo_checks import read_layer

    a1 = read_layer(os.path.join(SHP_DIR, "sen_admin1.shp"), "COD-AB SEN admin1", engine="fiona")
    a2 = read_layer(os.path.join(SHP_DIR, "sen_admin2.shp"), "COD-AB SEN admin2", engine="fiona")
    print(f"COD-AB Senegal: admin1 {len(a1)} features, admin2 {len(a2)}; "
          f"valid_on {sorted(set(map(str, a1['valid_on'])))}, version {sorted(set(a1['version']))}")
    if len(a1) != 14 or len(a2) != 46:
        raise SystemExit("expected 14 régions and 46 départements in COD-AB v02")
    got = dict(zip(a1["adm1_pcode"], a1["adm1_name"]))
    if got != ADM1_NAMES:
        raise SystemExit(f"COD admin1 pcodes/names changed: {got}")
    for dept, pc in DEPTS_1988.items():
        row = a2[a2["adm2_pcode"] == pc]
        if len(row) != 1 or row.iloc[0]["adm2_name"] != dept or row.iloc[0]["adm1_pcode"] != "SN02":
            raise SystemExit(f"COD admin2 {pc} is not {dept} in Diourbel: {row.to_dict('records')}")
    if sorted(a2.loc[a2["adm1_pcode"] == "SN02", "adm2_pcode"]) != sorted(DEPTS_1988.values()):
        raise SystemExit("Diourbel has other than its three départements in COD admin2")
    used = sorted(pc for pcs in REGIONS_1988.values() for pc in pcs) + ["SN02"]
    if sorted(used) != sorted(ADM1_NAMES):
        raise SystemExit("the 1988 régions plus Diourbel do not use every COD région exactly once")
    print("  pcodes: 14 COD régions used once each; Diourbel's 3 départements named and parented")

    rows = []
    for reg, pcs in REGIONS_1988.items():
        geom = a1.loc[a1["adm1_pcode"].isin(pcs), "geometry"].union_all()
        rows.append({"unit": reg, "level": "region", "pcodes": "+".join(pcs), "geometry": geom})
    for dept, pc in DEPTS_1988.items():
        geom = a2.loc[a2["adm2_pcode"] == pc, "geometry"].union_all()
        rows.append({"unit": dept, "level": "department", "pcodes": pc, "geometry": geom})
    units = gpd.GeoDataFrame(rows, geometry="geometry", crs=a1.crs)

    missing = sorted(set(units["unit"]) ^ set(census.index))
    if missing:
        raise SystemExit(f"units in only one of sn.csv and the polygons: {missing}")
    units["census_pop"] = units["unit"].map(census).astype(int)

    # ---- the area witness
    km2 = units.to_crs(6933).geometry.area / 1e6
    units["km2"] = km2.round(1)
    d1 = a1.to_crs(6933)
    diourbel_adm1 = float(d1.loc[a1["adm1_pcode"] == "SN02"].geometry.area.sum() / 1e6)
    diourbel_depts = float(km2[units["level"] == "department"].sum())
    if abs(diourbel_depts / diourbel_adm1 - 1) > 0.002:
        raise SystemExit(f"Diourbel's départements {diourbel_depts:,.0f} km2 against its région "
                         f"polygon {diourbel_adm1:,.0f}")
    print(f"\n  areas against 1988 (bands from Tableau 1.2's whole-number densities):")
    off = {}
    for r, a in zip(units.itertuples(), km2):
        if r.level == "region":
            pop, d = DENSITY_1988[r.unit]
            lo, hi = pop / (d + 0.5), pop / (d - 0.5)
            ref = f"{lo:,.0f}-{hi:,.0f}"
        else:
            lo = hi = AREA_DEPTS_1988[r.unit]
            ref = f"{lo:,}"
        dev = 0.0 if lo <= a <= hi else (a / lo - 1 if a < lo else a / hi - 1)
        off[r.unit] = dev
        print(f"    {r.unit:<12} {r.level:<10} COD {a:>9,.0f} km2   1988 {ref:>15}   {dev:+6.1%}"
              f"{'   pinned' if r.unit in AREA_PINNED else ''}")
    lo, hi = DENSITY_1988["Diourbel"][0] / 142.5, DENSITY_1988["Diourbel"][0] / 141.5
    print(f"    (Diourbel région: COD {diourbel_adm1:,.0f} km2 against 1988 {lo:,.0f}-{hi:,.0f}; "
          "the Diourbel report prints 4,359)")
    over = {u for u, v in off.items() if abs(v) > AREA_TOL}
    if over != AREA_PINNED:
        raise SystemExit(f"units more than {AREA_TOL:.0%} off 1988's area: {sorted(over)}, "
                         f"pinned {sorted(AREA_PINNED)}")
    worst = max(off, key=lambda u: abs(off[u]))
    if abs(off[worst]) > AREA_WORST:
        raise SystemExit(f"{worst} is {off[worst]:+.1%} off 1988's area")
    total = float(km2.sum())
    print(f"  the {AREA_PINNED and len(AREA_PINNED)} pinned units are the only ones over "
          f"{AREA_TOL:.0%}; worst {worst} {off[worst]:+.1%}; all units {total:,.0f} km2")
    return units


def main():
    import geopandas as gpd
    import pandas as pd
    from geo_checks import read_layer

    if "--fetch" in sys.argv:
        fetch()
    gpkg = os.path.join(KONTUR, GPKG_NAME)
    for p in (os.path.join(SHP_DIR, "sen_admin1.shp"), gpkg, NORM):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} — run sources/sn.py --fetch and "
                             "sources/sn_geo.py --fetch")
    os.makedirs(GEO, exist_ok=True)

    df = pd.read_csv(NORM, keep_default_na=False, na_values=[""])
    census = df.groupby("geo_id")["count"].sum()
    if len(census) != UNITS:
        raise SystemExit(f"{len(census)} units in sn.csv, expected {UNITS}")
    units = build_units(census)
    units.to_file(OUT_UNITS, layer="units", driver="GPKG")
    print(f"wrote {OUT_UNITS} ({len(units)} units)")

    # ---- Kontur, joined on hex centroids; an offshore centroid snaps to the nearest unit
    hexes = read_layer(gpkg, "Kontur SN")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"\nKontur hexes: {len(hexes):,}, crs={hexes.crs}, population {hexes[popcol].sum():,.0f}")
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()}, geometry=cent,
                           crs=hexes.crs).to_crs(units.crs)
    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)
    unit = joined["unit"].copy()

    outside = unit.isna()
    if outside.any():
        utm = 32628
        near = gpd.sjoin_nearest(pts.loc[outside].to_crs(utm),
                                 units[["unit", "geometry"]].to_crs(utm),
                                 how="left", max_distance=SNAP_M, distance_col="dist")
        near = near[~near.index.duplicated(keep="first")]
        snapped = near["unit"].dropna()
        unit.loc[snapped.index] = snapped
        print(f"  centroids in no unit: {int(outside.sum()):,} "
              f"({pts.loc[outside, popcol].sum():,.0f} people); {len(snapped):,} within "
              f"{SNAP_M} m snapped ({pts.loc[snapped.index, popcol].sum():,.0f} people)")
    lost = unit.isna()
    lost_pop = float(pts.loc[lost, popcol].sum())
    share = lost_pop / float(pts[popcol].sum())
    print(f"  dropped: {int(lost.sum()):,} hexes, {lost_pop:,.0f} people ({share:.3%})")
    if share > 0.01:
        raise SystemExit("more than 1% of Kontur's people fall outside every unit")

    keep = (~lost).to_numpy()
    out = gpd.GeoDataFrame({"unit": unit[keep].to_numpy(),
                            "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
                           geometry=hexes.geometry[keep].to_crs(units.crs).to_numpy(),
                           crs=units.crs)
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    bad = sorted(set(units["unit"]) - set(per.index[per["sum"] > 0]))
    if bad:
        raise SystemExit(f"units with no populated hex: {bad}")

    tot = float(out["pop"].sum())
    census_total = int(units["census_pop"].sum())
    ratio = tot / census_total
    print(f"\n  Kontur 2023 {tot:,.0f} against the 1988 drawn population {census_total:,}: "
          f"ratio {ratio:.3f}")
    if not KONTUR_RATIO_MIN <= ratio <= KONTUR_RATIO_MAX:
        raise SystemExit(f"ratio {ratio:.3f} outside [{KONTUR_RATIO_MIN}, {KONTUR_RATIO_MAX}]")
    out.to_file(OUT_HEXES, layer="hexes", driver="GPKG")
    print(f"wrote {OUT_HEXES} ({len(out):,} hexes)")

    lk = units[["unit", "level", "pcodes", "census_pop", "km2"]].merge(
        per.rename(columns={"size": "hexes", "sum": "kontur_pop"}),
        left_on="unit", right_index=True, how="left")
    lk["kontur_over_census"] = lk["kontur_pop"] / lk["census_pop"]
    with open(OUT_LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "level", "pcodes", "census_pop_1988", "cod_km2", "kontur_pop_2023",
                    "hexes", "kontur_over_census"])
        for r in lk.sort_values("unit").itertuples(index=False):
            w.writerow([r.unit, r.level, r.pcodes, int(r.census_pop), r.km2,
                        round(r.kontur_pop, 1), int(r.hexes), round(r.kontur_over_census, 3)])
    print(f"wrote {OUT_LOOKUP}")
    print("\n  Kontur 2023 over the 1988 count, per unit (growth and migration since 1988):")
    for r in lk.sort_values("kontur_over_census").itertuples(index=False):
        print(f"    {r.unit:<12} {r.kontur_over_census:5.2f}x  {int(r.hexes):>7,} hexes  "
              f"median {r.km2 / max(int(r.hexes), 1):.2f} km2/hex")


if __name__ == "__main__":
    main()
