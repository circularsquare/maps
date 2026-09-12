"""Central African Republic — commune boundaries and the placement grid.

Writes:
    data/geo/cf/cf_communes.gpkg    the 177 counted communes (`units`)
    data/geo/cf/cf_hexes.gpkg       Kontur H3 hexes with `unit` and `pop` (`place`)
    data/geo/cf/cf_lookup.csv       unit -> names, census population, Kontur population

Usage:
    python sources/cf_geo.py --fetch    one 2.9 MB gzipped gpkg from Kontur
    python sources/cf_geo.py            rebuild from data/raw/cf/

THE BOUNDARIES COME OUT OF THE SAME FILE AS THE COUNTS, and here they are also the same
VINTAGE. Ethiopia (§9u) had 2007 counts on 2021 woredas and USCB's re-cutting to trust;
CAR's religion layer keys to `CF_GEOG1_ADM3_2003`, the 2003 set, so there is no re-cutting at
all. The workbook also ships a GEOG2/2021 boundary set with **181** communes for its
`Population` and `Displacement` tables — reading that one instead would misalign four units
and look like a join failure rather than a vintage error, so the layer name is asserted.

THE JOIN IS AN IDENTITY, AND IT IS MEASURED RATHER THAN ASSUMED. 177 polygons, 177 ADM3 count
rows, zero `GEO_MATCH` on either side alone, zero duplicates. §11h's claim, checked on the
sixth country of the series.

KONTUR IS NEEDED, AND THE NUMBER THAT SAYS SO IS 73/29. **The 50 largest communes by area are
73.0% of the country's land and 28.9% of its people.** Yalinga is 42,260 km² with 4,768
people and Djémah 37,065 km² with 1,845 — 0.11 and 0.05 people per km². Spread those
uniformly and the whole east fills with an even wash over country that is very nearly empty,
which is the failure ke_grid.py and et_geo.py exist to prevent. CAR is a stronger case for a
grid than either: a larger share of the land is in the largest units.

**THE RATIO BAND IS NOT A CHECK HERE, AND THAT IS THE FINDING THIS COUNTRY CONTRIBUTES.**
The census is 2003 and Kontur is 2023, so a ratio near 1.0 would be evidence of a bad
download; the national figure is 1.500, which is about right for twenty years of growth. But
the PER-COMMUNE spread is the tell: **78.5% of the 177 communes fall within ±5% of the median
ratio, against 34.0% of Ethiopia's 738 woredas.** Independent modelling does not do that.
Kontur's CAR extract is, at commune level, very close to a constant rescale of the same 2003
census this map is drawing — which makes sense, because there is no newer CAR census for it
to have used. CAR's last enumeration is RGPH03.

So the agreement between Kontur and the census here proves nothing about either. **What the
grid still buys is the only thing it is actually used for: WHERE INSIDE a commune the people
are**, which comes from settlement and building footprints and is genuinely not in the
census. The level is never read — Kontur is a within-commune weight — so the derivation does
not contaminate anything. It is recorded because et_geo.py and ke_grid.py both treat the
band as evidence, and on this country it is not. `main()` measures the concentration on every
run rather than leaving it as a claim in a docstring.

The wide tail is real and is the war: CAR has had a displacement crisis since 2012, several
hundred thousand people left the country and several hundred thousand more moved inside it.
Ouakanga at 8.62x and the Bangui arrondissements above 2x are that, not error.
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
RAW = os.path.join(ROOT, "data", "raw", "cf")
GEO = os.path.join(ROOT, "data", "geo", "cf")
KONTUR = os.path.join(ROOT, "data", "geo", "kontur")

GDB = os.path.join(RAW, "Central_African_Republic.gdb")
XLSX = os.path.join(RAW, "central_african_republic_uscb_202303.xlsx")

# GEOG1 is the 2003 vintage, which is what the religion layer keys to. GEOG2 is 2021 and has
# 181 communes; it belongs to the Population and Displacement tables and is NOT this.
LAYER_ADM3 = "CF_GEOG1_ADM3_2003_uscb_202303"

OUT_UNITS = os.path.join(GEO, "cf_communes.gpkg")
OUT_HEXES = os.path.join(GEO, "cf_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "cf_lookup.csv")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_CF_20231101.gpkg.gz")
GZ_NAME = "kontur_population_CF_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_CF_20231101.gpkg"

COMMUNES = 177
CENSUS_POPULATION = 3_836_736     # the religion universe, not the RGPH03 total (see cf.py)

# Twenty years, and a war in the middle. Near 1.0 is the suspicious value, not the good one.
KONTUR_RATIO_MIN = 1.05
KONTUR_RATIO_MAX = 2.30

CATS = ["RLG_CAT", "RLG_PRO", "RLG_MUS", "RLG_OTHR", "RLG_NR"]


def fetch():
    import requests

    os.makedirs(KONTUR, exist_ok=True)
    gz = os.path.join(KONTUR, GZ_NAME)
    gpkg = os.path.join(KONTUR, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 5_000_000:
        print("already have", gpkg)
        return
    if not os.path.exists(gz) or os.path.getsize(gz) < 1_000_000:
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
    # §5a: a 200 is not a download, and a gunzip that runs is not a GeoPackage.
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
        raise SystemExit(f"missing {GDB} — run: python sources/cf.py --fetch")
    gpkg = os.path.join(KONTUR, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} — run: python sources/cf_geo.py --fetch")

    os.makedirs(GEO, exist_ok=True)

    # ---- 1. the communes, and the identity join §11h promises
    g = gpd.read_file(GDB, layer=LAYER_ADM3)
    if len(g) != COMMUNES:
        raise SystemExit(f"{LAYER_ADM3} has {len(g)} features, expected {COMMUNES} — "
                         "the 2021 GEOG2 layer has 181 and is the wrong vintage")

    rel = pd.read_excel(XLSX, sheet_name="Religion", header=0, skiprows=[1])
    rel = rel[rel["ADM_LEVEL"].astype(int) == 3].copy()
    for c in CATS:
        rel[c] = pd.to_numeric(rel[c], errors="coerce")
    rel["census_pop"] = rel[CATS].sum(axis=1, min_count=1)

    a, b = set(g["GEO_MATCH"]), set(rel["GEO_MATCH"])
    if a != b or len(g) != g["GEO_MATCH"].nunique():
        raise SystemExit(
            f"the join is not an identity: {len(a - b)} polygons with no counts, "
            f"{len(b - a)} counts with no polygon, "
            f"{len(g) - g['GEO_MATCH'].nunique()} duplicate keys")
    print(f"ADM3 polygons {len(g)}, count rows {len(rel)}, GEO_MATCH identical on both "
          f"sides,\n   0 duplicates — §11h's free join, checked on the sixth country.")

    units = g.merge(rel[["GEO_MATCH", "census_pop"]], on="GEO_MATCH", how="left")
    units["unit"] = units["GEO_MATCH"].astype(str)
    units[["unit", "AREA_NAME", "ADM1_NAME", "ADM2_NAME", "census_pop", "geometry"]] \
        .to_file(OUT_UNITS, layer="communes", driver="GPKG")
    print(f"wrote {OUT_UNITS} ({len(units):,} communes)")

    eq = units.to_crs(3857)
    km2 = eq.geometry.area / 1e6
    top = km2.sort_values(ascending=False).head(50).index
    print(f"\n  the 50 largest communes are {100 * km2[top].sum() / km2.sum():.1f}% of "
          f"the land and "
          f"{100 * units.loc[top, 'census_pop'].sum() / units['census_pop'].sum():.1f}% "
          f"of the people — which is why a population grid is used rather than a uniform "
          f"fill.")

    # ---- 2. Kontur, joined on hex CENTROIDS so no hex is split across two communes
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
    print(f"  hexes whose centroid is in no commune: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%) — the border "
          f"overrun between\n     Kontur's coastline-free extract and LSIB; dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    # Every commune must get hexes, or its dots fall back to an equal share over nothing
    # and the commune silently empties (§9u).
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        names = units.set_index("unit")["AREA_NAME"]
        raise SystemExit("communes with no populated hex: "
                         + ", ".join(f"{u} ({names[u]})" for u in missing))
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"communes whose hexes sum to zero population: {zero}")
    print(f"  every one of the {len(units)} communes has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    # §8.2e / memory: a population grid must be FINER than the tier it weights, or it is
    # not weighting anything. Saint Vincent (§9ac) is where that floor was found.
    hex_km2 = 0.67
    smallest = float(km2.min())
    print(f"  the smallest commune is {smallest:,.1f} km² against a {hex_km2} km² hex "
          f"= {smallest / hex_km2:,.0f} hexes;\n     the grid is finer than the tier "
          f"everywhere (min hexes per commune {per['size'].min():,}).")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur 2023 {tot:,.0f} vs the 2003 religion universe "
          f"{CENSUS_POPULATION:,} — ratio {ratio:.3f}")
    if not KONTUR_RATIO_MIN <= ratio <= KONTUR_RATIO_MAX:
        raise SystemExit(
            f"ratio {ratio:.3f} outside [{KONTUR_RATIO_MIN}, {KONTUR_RATIO_MAX}] — "
            "twenty years of growth should put it near 1.4; check the download")
    print("     twenty years apart and a war in between, so the level is expected to "
          "differ\n     and only the SHAPE is used — Kontur is a within-commune weight, "
          "never a population.")

    out.to_file(OUT_HEXES, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT_HEXES} ({len(out):,} hexes)")

    # ---- 3. the lookup, which is also the per-commune sanity record
    per = per.rename(columns={"size": "hexes", "sum": "kontur_pop"})
    lut = units[["unit", "AREA_NAME", "ADM1_NAME", "ADM2_NAME", "census_pop"]] \
        .merge(per, left_on="unit", right_index=True, how="left")
    lut["kontur_over_census"] = lut["kontur_pop"] / lut["census_pop"]
    with open(OUT_LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "commune", "prefecture", "sous_prefecture",
                    "census_pop_2003", "kontur_pop_2023", "hexes",
                    "kontur_over_census"])
        for r in lut.sort_values("unit").itertuples(index=False):
            w.writerow([r.unit, r.AREA_NAME, r.ADM1_NAME, r.ADM2_NAME,
                        int(r.census_pop), round(r.kontur_pop, 1), int(r.hexes),
                        round(r.kontur_over_census, 3)])
    print(f"wrote {OUT_LOOKUP}")

    q = lut["kontur_over_census"].quantile([.01, .25, .5, .75, .99])
    print(f"\n  per-commune Kontur/census ratio: median {q[.5]:.2f}, "
          f"quartiles {q[.25]:.2f}–{q[.75]:.2f}, 1–99% {q[.01]:.2f}–{q[.99]:.2f}")

    # KONTUR IS NOT INDEPENDENT OF THIS CENSUS — measured, not asserted. CAR's last
    # enumeration is RGPH03, so Kontur had nothing newer to build on, and at commune level
    # its extract is close to a constant rescale of the table being drawn. Ethiopia, whose
    # Kontur IS independent, sits near 34% by this measure.
    near = float(((lut["kontur_over_census"] / q[.5] - 1).abs() <= 0.05).mean())
    print(f"     {100 * near:.1f}% of communes are within ±5% of the median ratio "
          f"(Ethiopia: 34.0%).")
    if near > 0.60:
        print("     THAT IS TOO TIGHT TO BE INDEPENDENT: Kontur's CAR extract is close to "
              "a\n     constant rescale of this same 2003 census, because CAR has had no "
              "census\n     since. So the ratio agreeing is NOT evidence about either "
              "source. The grid\n     is used only for WHERE INSIDE a commune people are, "
              "which is still its own\n     measurement; the level is never read.")
    print("     The check that matters is that every commune has hexes at all; the ratio"
          "\n     is reported, not asserted.")
    print("\n  the 8 communes where Kontur and the census disagree most:")
    for r in lut.reindex(lut["kontur_over_census"].sort_values(
            ascending=False).index).head(8).itertuples(index=False):
        print(f"     {r.kontur_over_census:6.2f}x  {str(r.AREA_NAME)[:30]:30s} "
              f"{str(r.ADM1_NAME)[:22]:22s} census {int(r.census_pop):>9,}")


if __name__ == "__main__":
    main()
