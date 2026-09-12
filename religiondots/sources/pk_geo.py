"""Pakistan — district boundaries and the placement grid.

Writes:
    data/geo/pk/pk_districts.gpkg    the 135 districts with religion data (`units`)
    data/geo/pk/pk_hexes.gpkg        Kontur H3 r8 hexes with `unit` and `pop` (`place`)
    data/geo/pk/pk_lookup.csv        unit -> names, census population, Kontur population

Usage:
    python sources/pk_geo.py --fetch    one 27 MB gzipped gpkg from Kontur
    python sources/pk_geo.py            rebuild from data/raw/pk/

THE JOIN IS AN IDENTITY, as in Ethiopia (§9u): `PK_GEOG1_ADM3_2017` and
`PK_RELIGION_GEOG1_2017census` are two layers of one geodatabase, both keyed on `GEO_MATCH`,
155 polygons against 155 rows and nothing to match. That is the USCB seam's whole value
(sources.md §11h) and it is the second time it has paid.

THE TWENTY MISSING DISTRICTS ARE A HOLE IN THE NORTH AND IT IS DELIBERATE. Azad Kashmir's
ten and Gilgit-Baltistan's ten have polygons and no religion data — PBS did not publish the
disputed regions, and USCB's metadata says so outright. They are dropped from `units`, so
they take no dots and draw as empty ground. `note_public` says what the hole is; an
unexplained blank on a map reads as "nobody lives here", which is the one thing it must not.

KONTUR IS NEEDED FOR BALOCHISTAN, and the numbers are starker than Ethiopia's. Balochistan is
44% of Pakistan's land and 6% of its people; **Chagai district alone is 44,748 km² with
226,508 people** — a density under 6/km² — and the province is 99.4% Muslim. Placing
uniformly inside districts would fill nearly half the map with an even wash of one colour
over the Makran and the Kharan desert. Same failure as Kenya (§8.2) and Ethiopia, on the
largest empty quarter of the three.

THE RATIO BAND, AND WHY IT IS NARROWER THAN ETHIOPIA'S. Ethiopia's census is 2007 against a
2023 grid, so its band had to allow for a doubling. Pakistan's census is **2017** — six
years, not sixteen — so the expected ratio is much closer to 1 and the band can be tight
again. This is the point `et_geo.py` makes in the other direction: the tolerance is a
statement about the VINTAGE GAP and must be re-derived per country rather than copied.
"""

import csv
import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "pk")
GEO = os.path.join(ROOT, "data", "geo", "pk")
KONTUR = os.path.join(ROOT, "data", "geo", "kontur")

GDB = os.path.join(RAW, "Pakistan.gdb")
XLSX = os.path.join(RAW, "pakistan_uscb_202401.xlsx")
LAYER_ADM3 = "PK_GEOG1_ADM3_2017_uscb_202401"

OUT_UNITS = os.path.join(GEO, "pk_districts.gpkg")
OUT_HEXES = os.path.join(GEO, "pk_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "pk_lookup.csv")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_PK_20231101.gpkg.gz")
GZ_NAME = "kontur_population_PK_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_PK_20231101.gpkg"

ADM3_FEATURES = 155           # polygons in the layer
DISTRICTS = 135               # of those, the ones with 2017 religion counts
CENSUS_POPULATION = 207_684_626

# Kontur is 2023, the census 2017 — six years, and Pakistan grows ~2%/yr, so ~1.1-1.2 is
# expected. Wider than that at either end means a truncated file or a scrambled join.
KONTUR_RATIO_MIN = 0.90
KONTUR_RATIO_MAX = 1.45

CATS = ["RLG_MUS", "RLG_CHR", "RLG_HIN", "RLG_QAD", "RLG_SCH", "RLG_OTH"]


def fetch():
    import requests

    os.makedirs(KONTUR, exist_ok=True)
    gz = os.path.join(KONTUR, GZ_NAME)
    gpkg = os.path.join(KONTUR, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 50_000_000:
        print("already have", gpkg)
        return
    if not os.path.exists(gz) or os.path.getsize(gz) < 20_000_000:
        print("GET", GZ_URL)
        r = requests.get(GZ_URL, timeout=1800, stream=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(gz, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        print(f"  {os.path.getsize(gz):,} bytes")
    with gzip.open(gz, "rb") as src, open(gpkg, "wb") as dst:
        shutil.copyfileobj(src, dst)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()
    if not os.path.isdir(GDB):
        raise SystemExit(f"missing {GDB} — run: python sources/pk.py --fetch")
    gpkg = os.path.join(KONTUR, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} — run: python sources/pk_geo.py --fetch")

    os.makedirs(GEO, exist_ok=True)

    # ---- 1. the districts
    g = gpd.read_file(GDB, layer=LAYER_ADM3)
    if len(g) != ADM3_FEATURES:
        raise SystemExit(f"{LAYER_ADM3} has {len(g)} features, expected {ADM3_FEATURES}")

    rel = pd.read_excel(XLSX, sheet_name="Religion", header=0, skiprows=[1])
    for c in CATS:
        rel[c] = pd.to_numeric(rel[c], errors="coerce")
    rel["census_pop"] = rel[CATS].sum(axis=1, min_count=1)
    rel = rel[rel["ADM_LEVEL"] == 3]

    g = g.merge(rel[["GEO_MATCH", "census_pop"]], on="GEO_MATCH", how="left")
    units = g[g["census_pop"].notna()].copy()
    dropped = g[g["census_pop"].isna()]
    if len(units) != DISTRICTS:
        raise SystemExit(f"{len(units)} districts carry counts, expected {DISTRICTS}")

    regions = sorted(set(dropped["ADM1_NAME"].dropna()))
    print(f"ADM3 polygons {len(g)}: {len(units)} districts with religion data, "
          f"{len(dropped)} without — all of them in {', '.join(regions)}.")
    print("   PBS did not publish religion for the disputed regions; they draw blank.")

    units["unit"] = units["GEO_MATCH"].astype(str)
    units[["unit", "AREA_NAME", "ADM1_NAME", "ADM2_NAME", "census_pop", "geometry"]] \
        .to_file(OUT_UNITS, layer="districts", driver="GPKG")
    print(f"\nwrote {OUT_UNITS} ({len(units):,} districts)")

    # ---- 2. Kontur, joined on hex CENTROIDS
    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"\nKontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)

    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"  hexes whose centroid is in no drawn district: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.2f}%)")
    print("     mostly Azad Kashmir and Gilgit-Baltistan, which are deliberately not "
          "drawn,\n     plus the border overrun; dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"districts with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"districts whose hexes sum to zero population: {zero}")
    print(f"  every one of the {len(units)} drawn districts has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur 2023 {tot:,.0f} vs 2017 census {CENSUS_POPULATION:,} — "
          f"ratio {ratio:.3f}")
    if not KONTUR_RATIO_MIN <= ratio <= KONTUR_RATIO_MAX:
        raise SystemExit(
            f"ratio {ratio:.3f} outside [{KONTUR_RATIO_MIN}, {KONTUR_RATIO_MAX}] — six "
            "years of Pakistani growth should put it near 1.1; check the download")
    print("     six years apart, so this band is tight where Ethiopia's had to be wide.")

    out.to_file(OUT_HEXES, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT_HEXES} ({len(out):,} hexes)")

    # ---- 3. the lookup and the per-district sanity record
    per = per.rename(columns={"size": "hexes", "sum": "kontur_pop"})
    lut = units[["unit", "AREA_NAME", "ADM1_NAME", "ADM2_NAME", "census_pop"]] \
        .merge(per, left_on="unit", right_index=True, how="left")
    lut["kontur_over_census"] = lut["kontur_pop"] / lut["census_pop"]
    with open(OUT_LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "district", "province", "division", "census_pop_2017",
                    "kontur_pop_2023", "hexes", "kontur_over_census"])
        for r in lut.sort_values("unit").itertuples(index=False):
            w.writerow([r.unit, r.AREA_NAME, r.ADM1_NAME, r.ADM2_NAME,
                        int(r.census_pop), round(r.kontur_pop, 1), int(r.hexes),
                        round(r.kontur_over_census, 3)])
    print(f"wrote {OUT_LOOKUP}")

    q = lut["kontur_over_census"].quantile([.01, .25, .5, .75, .99])
    print(f"\n  per-district Kontur/census ratio: median {q[.5]:.2f}, "
          f"quartiles {q[.25]:.2f}–{q[.75]:.2f}, 1–99% {q[.01]:.2f}–{q[.99]:.2f}")
    print("  the 6 districts where Kontur and the census disagree most:")
    for r in lut.reindex(lut["kontur_over_census"].sort_values(
            ascending=False).index).head(6).itertuples(index=False):
        print(f"     {r.kontur_over_census:6.2f}x  {str(r.AREA_NAME)[:30]:30s} "
              f"{str(r.ADM1_NAME)[:14]:14s} census {int(r.census_pop):>10,}")


if __name__ == "__main__":
    main()
