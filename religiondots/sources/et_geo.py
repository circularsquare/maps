"""Ethiopia — woreda boundaries and the placement grid.

Writes:
    data/geo/et/et_woredas.gpkg     the 738 counted woredas (`units`)
    data/geo/et/et_hexes.gpkg       Kontur H3 r8 hexes with `unit` and `pop` (`place`)
    data/geo/et/et_lookup.csv       unit -> names, census population, Kontur population

Usage:
    python sources/et_geo.py --fetch    one 31 MB gzipped gpkg from Kontur
    python sources/et_geo.py            rebuild from data/raw/et/

THE BOUNDARIES COME OUT OF THE SAME FILE AS THE COUNTS, which has not happened before on
this map. Every other country here needed a boundary source found separately and joined by
name or code — Chile's names, Guyana's ISO codes, Ghana's acronym collision, China's join
that cost the whole ingest (§9r). The USCB geodatabase carries `ET_GEOG_ADM3_2021` and
`ET_RELIGION_2007census` side by side, both keyed on `GEO_MATCH`, so the join is an identity
and there is nothing to verify about it. **That is the single biggest reason this country was
cheap**, and it is the argument for looking at the rest of the USCB series (sources.md §11h).

INLAND WATER IS SOLVED FOR FREE, AND SO ARE THE PARKS. 18 of the 756 ADM3 polygons carry no
counts, and 14 of those are named water or protected land: **Lake T'ana (3,214 km²), Abaya,
Chamo, Langano, Shala, Ziway, Awassa, Chomen, Afera, K'ok'a**, the Tezeké Dam, Mago and Nech
Sar National Parks and the Gambella Wildlife Reserve. They are cut OUT of the surrounding
woredas rather than swallowed by them, so nothing has to be clipped afterwards — §9n's Ghana
problem, absent. The other four are the woredas with no census data (see et.py).

KONTUR IS NEEDED HERE FOR KENYA'S REASON, AND MORE SHARPLY. 738 woredas sounds fine enough to
place uniformly inside, and it is not: **the 50 largest woredas by area are 38.4% of
Ethiopia's land, 5.3% of its people, and 75% Muslim on average.** Warder is 22,626 km² with
58,035 people, Bare 20,537 km² with 93,340, and the whole Somali region runs 98-99% Muslim.
Spread those uniformly and two fifths of the map fills with an even wash of one colour over
the Ogaden, where almost nobody lives — the same failure ke_grid.py exists to prevent, on a
country where the empty part is larger and the colour more uniform.

THE RATIO BAND IS WIDE ON PURPOSE, AND THAT IS THE INTERESTING DIFFERENCE FROM KENYA.
ke_grid.py asserts Kontur within 25% of the census, because Kenya's census is 2019 and
Kontur is 2023. **Ethiopia's census is 2007 and Kontur is 2023 — sixteen years apart, over
which Ethiopia grew from 73.8 million to something near 125 million.** So a ratio near 1.0
would be evidence of a BAD download, not a good one, and the band below is derived from that
growth rather than fitted to the answer. The weight is a within-woreda one, so the level does
not matter at all and only the shape does; what the check can still catch is a scrambled join
or a truncated file.
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
RAW = os.path.join(ROOT, "data", "raw", "et")
GEO = os.path.join(ROOT, "data", "geo", "et")
KONTUR = os.path.join(ROOT, "data", "geo", "kontur")

GDB = os.path.join(RAW, "Ethiopia.gdb")
XLSX = os.path.join(RAW, "ethiopia_uscb_202308.xlsx")
LAYER_ADM3 = "ET_GEOG_ADM3_2021_uscb_202308"

OUT_UNITS = os.path.join(GEO, "et_woredas.gpkg")
OUT_HEXES = os.path.join(GEO, "et_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "et_lookup.csv")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_ET_20231101.gpkg.gz")
GZ_NAME = "kontur_population_ET_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_ET_20231101.gpkg"

ADM3_FEATURES = 756           # polygons in the layer, water and parks included
WOREDAS = 738                 # of those, the ones with 2007 census counts
CENSUS_POPULATION = 73_750_932

# Kontur is 2023 and the census is 2007. Ethiopia roughly doubled in between (73.8M ->
# ~125M by every published estimate), so the expected ratio is well above 1 and a value
# NEAR 1 is the suspicious one. The band is that growth, loosely bracketed; it is here to
# catch a truncated download or a scrambled join, not to assert a population.
KONTUR_RATIO_MIN = 1.15
KONTUR_RATIO_MAX = 2.10

CATS = [f"RLG_{k}_B" for k in ("ORDX", "PROT", "CATH", "ISL", "TRAD", "OTHR")]


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
        raise SystemExit(f"missing {GDB} — run: python sources/et.py --fetch")
    gpkg = os.path.join(KONTUR, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} — run: python sources/et_geo.py --fetch")

    os.makedirs(GEO, exist_ok=True)

    # ---- 1. the woredas, and the polygons that are deliberately not woredas
    g = gpd.read_file(GDB, layer=LAYER_ADM3)
    if len(g) != ADM3_FEATURES:
        raise SystemExit(f"{LAYER_ADM3} has {len(g)} features, expected {ADM3_FEATURES}")

    rel = pd.read_excel(XLSX, sheet_name="Religion", header=0, skiprows=[1])
    for c in CATS:
        rel[c] = pd.to_numeric(rel[c], errors="coerce").mask(lambda s: s < 0)
    rel["census_pop"] = rel[CATS].sum(axis=1, min_count=1)
    rel = rel[rel["ADM_LEVEL"] == 3]

    g = g.merge(rel[["GEO_MATCH", "census_pop"]], on="GEO_MATCH", how="left")
    units = g[g["census_pop"].notna()].copy()
    dropped = g[g["census_pop"].isna()]
    if len(units) != WOREDAS:
        raise SystemExit(f"{len(units)} polygons carry counts, expected {WOREDAS}")
    print(f"ADM3 polygons {len(g)}: {len(units)} counted woredas, "
          f"{len(dropped)} carrying no counts —")
    for r in dropped.itertuples(index=False):
        print(f"     {r.AREA_NAME}")
    print("   the lakes and parks are cut out of the woredas, so inland water needs no clip.")

    units["unit"] = units["GEO_MATCH"].astype(str)
    units[["unit", "AREA_NAME", "ADM1_NAME", "ADM2_NAME", "census_pop", "geometry"]] \
        .to_file(OUT_UNITS, layer="woredas", driver="GPKG")
    print(f"\nwrote {OUT_UNITS} ({len(units):,} woredas)")

    # ---- 2. Kontur, joined on hex CENTROIDS so no hex is split across two woredas
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
    print(f"  hexes whose centroid is in no woreda: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     the border overrun, and the lakes and parks, which are holes in the "
          "woreda cover; dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    # Every woreda must get hexes, or its dots fall back to an equal share over nothing and
    # the woreda silently empties.
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"woredas with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"woredas whose hexes sum to zero population: {zero}")
    print(f"  every one of the {len(units)} woredas has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur 2023 {tot:,.0f} vs 2007 census {CENSUS_POPULATION:,} — "
          f"ratio {ratio:.3f}")
    if not KONTUR_RATIO_MIN <= ratio <= KONTUR_RATIO_MAX:
        raise SystemExit(
            f"ratio {ratio:.3f} outside [{KONTUR_RATIO_MIN}, {KONTUR_RATIO_MAX}] — "
            "sixteen years of Ethiopian growth should put it near 1.7; check the download")
    print("     sixteen years apart, so the level is expected to differ and only the "
          "SHAPE\n     is used — Kontur is a within-woreda weight and never a population.")

    out.to_file(OUT_HEXES, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT_HEXES} ({len(out):,} hexes)")

    # ---- 3. the lookup, which is also the per-woreda sanity record
    per = per.rename(columns={"size": "hexes", "sum": "kontur_pop"})
    lut = units[["unit", "AREA_NAME", "ADM1_NAME", "ADM2_NAME", "census_pop"]] \
        .merge(per, left_on="unit", right_index=True, how="left")
    lut["kontur_over_census"] = lut["kontur_pop"] / lut["census_pop"]
    with open(OUT_LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "woreda", "region", "zone", "census_pop_2007",
                    "kontur_pop_2023", "hexes", "kontur_over_census"])
        for r in lut.sort_values("unit").itertuples(index=False):
            w.writerow([r.unit, r.AREA_NAME, r.ADM1_NAME, r.ADM2_NAME,
                        int(r.census_pop), round(r.kontur_pop, 1), int(r.hexes),
                        round(r.kontur_over_census, 3)])
    print(f"wrote {OUT_LOOKUP}")

    q = lut["kontur_over_census"].quantile([.01, .25, .5, .75, .99])
    print(f"\n  per-woreda Kontur/census ratio: median {q[.5]:.2f}, "
          f"quartiles {q[.25]:.2f}–{q[.75]:.2f}, 1–99% {q[.01]:.2f}–{q[.99]:.2f}")
    print("     a wide spread is expected — sixteen years of very uneven growth, and Addis"
          "\n     and the regional capitals grew fastest. The check that matters is that "
          "every\n     woreda has hexes at all; the ratio is reported, not asserted.")
    print("\n  the 8 woredas where Kontur and the census disagree most:")
    for r in lut.reindex(lut["kontur_over_census"].abs().sort_values(
            ascending=False).index).head(8).itertuples(index=False):
        print(f"     {r.kontur_over_census:6.2f}x  {str(r.AREA_NAME)[:32]:32s} "
              f"{str(r.ADM1_NAME)[:22]:22s} census {int(r.census_pop):>9,}")


if __name__ == "__main__":
    main()
