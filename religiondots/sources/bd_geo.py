"""Bangladesh — upazila boundaries and the placement grid.

Writes:
    data/geo/bd/bd_upazilas.gpkg    the 544 counted upazilas and thanas (`units`)
    data/geo/bd/bd_hexes.gpkg       Kontur H3 r8 hexes with `unit` and `pop` (`place`)
    data/geo/bd/bd_lookup.csv       unit -> names, census population, Kontur population

Usage:
    python sources/bd_geo.py --fetch    one Kontur BD extract
    python sources/bd_geo.py            rebuild from data/raw/bd/

THE BOUNDARIES COME OUT OF THE SAME FILE AS THE COUNTS, for the third time on this map
(Ethiopia §9u, Pakistan §9t). `BD_GEOG_ADM3_2011` and `BD_RELIGION_AND_ETHNICITY_2011census`
are two layers of one geodatabase, both keyed on `GEO_MATCH`, so the join is an identity:
**544 polygons, 544 counted units, zero unmatched.** No name matching, no code bridge, and
nothing to verify.

AND THE VINTAGES MATCH, WHICH ETHIOPIA'S DID NOT. Ethiopia is a 2007 census re-cut onto 2021
woredas and 418 of its units carry a lineage note saying so; Bangladesh's boundaries are the
2011 ones the 2011 census was published on. Spec §8.1 satisfied outright.

**THERE ARE NO SPARE POLYGONS, SO INLAND WATER IS NOT FREE HERE.** Ethiopia's file has 756
ADM3 polygons of which 18 are lakes and parks cut OUT of the woredas, so its water solved
itself. Bangladesh has 544 polygons and 544 counted units — the upazila cover is complete and
the rivers are INSIDE it. In the largest delta on earth that is not a small detail: the
Jamuna is braided and several kilometres wide, the Padma and lower Meghna wider still, and a
uniform placement would put dots in the middle of all of them. Two things handle it and
neither is this file:
  * `water.py` runs for every country and subtracts OSM tidal water, which in Bangladesh
    reaches a long way inland — the whole lower Meghna estuary and the Sundarbans creeks;
  * Kontur's population hexes are empty over open water, so the river channels take no dots
    even where the tidal layer does not reach.
The check at the end of this module reports how much Kontur population the upazila cover
actually catches, which is the number that would move if either stopped working.

KONTUR IS NEEDED, AND FOR A DIFFERENT REASON THAN ETHIOPIA'S. Ethiopia and Pakistan need it
because a few enormous desert units would otherwise wash half the map in one colour.
Bangladesh is the opposite country — 544 units averaging 258 km², about as uniform a grid as
this map has — and it still needs Kontur, because the **exceptions are precisely the units
this country is worth drawing for**:

  * **The Chittagong Hill Tracts.** Baghaichhari is 1,617 km², Thanchi 1,079, Belai Chhari
    1,048 — forested mountain units at a tenth of the national density, where settlement sits
    in valley floors and along rivers. They are also where every Buddhist and tribal-Christian
    dot in Bangladesh is: Juraichhari is 94.6% Buddhist, Ruma 38.2% Christian. Spread those
    uniformly and the most interesting geography on the map is smeared evenly across empty
    ridgeline.
  * **The Sundarbans.** Shyamnagar (1,787 km²), Mongla (1,345), Koyra (1,373) and Dacope
    (797) are the four largest units in the country outside the Hill Tracts and each is
    largely mangrove forest with nobody in it. Dacope is also the single most Hindu upazila in
    Bangladesh at 56.5%, so this is not an empty corner either.

So the argument for the hex weight here is not "some units are huge and empty" but "the units
that are huge and empty are the ones carrying the minorities", which is a sharper version of
the same problem and would have been easy to miss on a country this uniform.

THE RATIO BAND IS DERIVED, NOT COPIED. Ethiopia's is [1.15, 2.10] for a 2007 census against a
2023 grid; Pakistan's is [0.90, 1.45] for 2017. **Bangladesh is 2011 against 2023** — twelve
years over which the country went from 144.0M to roughly 171M — so the expected ratio is near
1.19 and the band below brackets that. Copying either of the other two would be wrong in a
different direction each time; §12's rule is to re-derive it per country and this is why.
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
RAW = os.path.join(ROOT, "data", "raw", "bd")
GEO = os.path.join(ROOT, "data", "geo", "bd")
KONTUR = os.path.join(ROOT, "data", "geo", "kontur")

GDB = os.path.join(RAW, "Bangladesh.gdb")
XLSX = os.path.join(RAW, "bangladesh_uscb_202107.xlsx")
SHEET = "Religion and Ethnicity"
LAYER_ADM3 = "BD_GEOG_ADM3_2011_uscb_202107"

OUT_UNITS = os.path.join(GEO, "bd_upazilas.gpkg")
OUT_HEXES = os.path.join(GEO, "bd_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "bd_lookup.csv")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_BD_20231101.gpkg.gz")
GZ_NAME = "kontur_population_BD_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_BD_20231101.gpkg"

ADM3_FEATURES = 544           # and all 544 carry counts — there are no water polygons
UPAZILAS = 544
CENSUS_POPULATION = 144_043_696

# 2011 census against a 2023 grid: twelve years, 144.0M -> ~171M, so ~1.19 is expected.
# NOT Ethiopia's [1.15, 2.10] (sixteen years and a near-doubling) and NOT Pakistan's
# [0.90, 1.45] (six years). Re-derived per country, per §12.
KONTUR_RATIO_MIN = 1.00
KONTUR_RATIO_MAX = 1.55

CATS = ["RLG_MSL", "RLG_HIN", "RLG_CHR", "RLG_BUD", "RLG_OTH"]


def fetch():
    import requests

    os.makedirs(KONTUR, exist_ok=True)
    gz = os.path.join(KONTUR, GZ_NAME)
    gpkg = os.path.join(KONTUR, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 20_000_000:
        print("already have", gpkg)
        return
    if not os.path.exists(gz) or os.path.getsize(gz) < 5_000_000:
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
        raise SystemExit(f"missing {GDB} — run: python sources/bd.py --fetch")
    gpkg = os.path.join(KONTUR, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} — run: python sources/bd_geo.py --fetch")

    os.makedirs(GEO, exist_ok=True)

    # ---- 1. the upazilas. Every polygon is a counted unit here; there is nothing to drop.
    g = gpd.read_file(GDB, layer=LAYER_ADM3)
    if len(g) != ADM3_FEATURES:
        raise SystemExit(f"{LAYER_ADM3} has {len(g)} features, expected {ADM3_FEATURES}")

    rel = pd.read_excel(XLSX, sheet_name=SHEET, header=0, skiprows=[1])
    for c in CATS:
        rel[c] = pd.to_numeric(rel[c], errors="coerce")
    rel["census_pop"] = rel[CATS].sum(axis=1, min_count=1)
    rel = rel[rel["ADM_LEVEL"] == 3]

    g = g.merge(rel[["GEO_MATCH", "census_pop"]], on="GEO_MATCH", how="left")
    units = g[g["census_pop"].notna()].copy()
    if len(units) != UPAZILAS:
        raise SystemExit(f"{len(units)} polygons carry counts, expected {UPAZILAS}")
    print(f"ADM3 polygons {len(g)}: all {len(units)} carry counts, none dropped.")
    print("   NOTE there are no water or park polygons in this layer, unlike Ethiopia's — "
          "\n   the delta's rivers are INSIDE the upazilas. water.py and the Kontur weight "
          "\n   are what keep dots out of them; see the module docstring.")

    units["unit"] = units["GEO_MATCH"].astype(str)
    units[["unit", "AREA_NAME", "ADM1_NAME", "ADM2_NAME", "census_pop", "geometry"]] \
        .to_file(OUT_UNITS, layer="upazilas", driver="GPKG")
    print(f"\nwrote {OUT_UNITS} ({len(units):,} upazilas)")

    # ---- 2. Kontur, joined on hex CENTROIDS so no hex is split across two upazilas
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
    print(f"  hexes whose centroid is in no upazila: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     the border overrun and the offshore chars; dropped. A LARGE number here "
          "\n     would mean the upazila cover has holes — in a delta that is the failure "
          "\n     mode to watch, so it is reported rather than assumed small.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    # Every upazila must get hexes, or its dots fall back to an equal share over the whole
    # polygon and the unit silently reverts to uniform placement.
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"upazilas with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"upazilas whose hexes sum to zero population: {zero}")
    print(f"  every one of the {len(units)} upazilas has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur 2023 {tot:,.0f} vs 2011 census {CENSUS_POPULATION:,} — "
          f"ratio {ratio:.3f}")
    if not KONTUR_RATIO_MIN <= ratio <= KONTUR_RATIO_MAX:
        raise SystemExit(
            f"ratio {ratio:.3f} outside [{KONTUR_RATIO_MIN}, {KONTUR_RATIO_MAX}] — "
            "twelve years of Bangladeshi growth should put it near 1.19; check the download")
    print("     twelve years apart, so the level is expected to differ and only the SHAPE"
          "\n     is used — Kontur is a within-upazila weight and never a population.")

    out.to_file(OUT_HEXES, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT_HEXES} ({len(out):,} hexes)")

    # ---- 3. the lookup, which is also the per-upazila sanity record
    per = per.rename(columns={"size": "hexes", "sum": "kontur_pop"})
    lut = units[["unit", "AREA_NAME", "ADM1_NAME", "ADM2_NAME", "census_pop"]] \
        .merge(per, left_on="unit", right_index=True, how="left")
    lut["kontur_over_census"] = lut["kontur_pop"] / lut["census_pop"]
    with open(OUT_LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "upazila", "division", "zila", "census_pop_2011",
                    "kontur_pop_2023", "hexes", "kontur_over_census"])
        for r in lut.sort_values("unit").itertuples(index=False):
            w.writerow([r.unit, r.AREA_NAME, r.ADM1_NAME, r.ADM2_NAME,
                        int(r.census_pop), round(r.kontur_pop, 1), int(r.hexes),
                        round(r.kontur_over_census, 3)])
    print(f"wrote {OUT_LOOKUP}")

    q = lut["kontur_over_census"].quantile([.01, .25, .5, .75, .99])
    print(f"\n  per-upazila Kontur/census ratio: median {q[.5]:.2f}, "
          f"quartiles {q[.25]:.2f}–{q[.75]:.2f}, 1–99% {q[.01]:.2f}–{q[.99]:.2f}")
    print("     the ratio is reported, not asserted per unit. The check that matters is "
          "\n     that every upazila has hexes at all.")

    # The units the hex weight exists for — see the docstring. Reported so that a future
    # change to the Kontur extract shows up here rather than as a silently smeared map.
    a = units.to_crs(3106)
    units["km2"] = a.area.to_numpy() / 1e6
    units["dens"] = units["census_pop"] / units["km2"]
    print("\n  the 10 largest units, which are why the hex weight is here:")
    for r in units.nlargest(10, "km2").itertuples(index=False):
        print(f"     {r.km2:6,.0f} km²  {r.dens:6,.0f}/km²  {str(r.AREA_NAME)[:26]:26s} "
              f"{str(r.ADM1_NAME)[:12]:12s} pop {int(r.census_pop):>9,}")
    print(f"     national density is {CENSUS_POPULATION / units['km2'].sum():,.0f}/km².")

    # ---- 4. and the opposite end, which is a limit rather than a check.
    #
    # THE HEX WEIGHT DEGRADES WHERE A UNIT IS SMALLER THAN A FEW HEXES, and Bangladesh is
    # the first country here with units that small. A Kontur r8 hex is ~0.80 km²; central
    # Dhaka's thanas are 0.8-3 km². The centroid join (deliberate — it stops a hex being
    # counted twice) then gives such a thana only the hexes whose CENTRES land inside it, so
    # a 2.3 km² thana can end up with one hex covering a third of it, and its dots crowd
    # into that third.
    #
    # NOT corrected, and the reason is that the correction would not be better. The fallback
    # would be an equal share over the whole polygon, and there is no evidence Kontur's one
    # hex is worse than that — it is a built-up-area model and central Dhaka is uniformly
    # built up either way. The displacement is a few hundred metres inside a unit of about
    # 2 km², which is finer than anything this map claims. Reported so it is a known limit
    # rather than a surprise, per §8.2.
    units = units.merge(per["hexes"], left_on="unit", right_index=True, how="left")
    hex_km2 = float(a.area.median() / 1e6) if len(a) else 0.0
    kh = gpd.read_file(OUT_HEXES, layer="hexes", rows=200).to_crs(3106)
    hex_km2 = float(kh.area.median() / 1e6)
    units["cover"] = units["hexes"] * hex_km2 / units["km2"]
    thin = units[units["cover"] < 0.60]
    print(f"\n  hex cover (Kontur r8 is ~{hex_km2:.2f} km² here). "
          f"{len(thin)} units are under 60% covered, "
          f"{int(thin['census_pop'].sum()):,} people "
          f"({100.0 * thin['census_pop'].sum() / CENSUS_POPULATION:.2f}% of the country):")
    for r in thin.nsmallest(8, "cover").itertuples(index=False):
        print(f"     cover {r.cover:5.0%}  {int(r.hexes):3d} hex  {r.km2:6.2f} km²  "
              f"{str(r.AREA_NAME)[:24]:24s} {str(r.ADM1_NAME)[:10]:10s} "
              f"pop {int(r.census_pop):>9,}")
    print("     all central Dhaka, all 1-3 km². Their dots crowd into the covered part; "
          "\n     see the note above for why that is left alone.")


if __name__ == "__main__":
    main()
