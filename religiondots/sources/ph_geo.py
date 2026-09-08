"""Philippines — placement geography for the 117 census units.

Writes data/geo/ph/ph_barangays.gpkg and data/geo/ph/ph_lookup.csv.

Source: **U.S. Census Bureau, `Philippines.gdb`, edition 202402** — a geodatabase built to
carry the 2020 CPH tables, with boundaries for admin levels 0-4 aligned to the State
Department's Large Scale International Boundaries. On HDX as *Philippines Subnational
Population and Housing Data Tables with Administrative Boundaries*, 304 MB.

**Why this and not COD or geoBoundaries.** The Philippines needs a fine placement layer
(spec §8.2) and a unit tier that matches the census's own, and the two obvious sources give
neither:

- **OCHA COD-AB `cod-ab-phl`** carries 11,920 admin4 units against the country's ~42,000
  barangays — it is the Mindanao humanitarian subset, not a national layer. Its admin2 is
  the 88 plain provinces, which is the WRONG TIER: the census tabulates
  province-excluding-any-HUC-inside-it, so joining on plain provinces double counts every
  one of the 33 highly urbanised cities (sources/ph.md §2).
- **geoBoundaries PHL** stops at ADM3. There is no ADM4 at all.

The USCB file solves both at once, because it was cut for these tables rather than adapted
to them. Its **ADM2 is the census tabulation tier**, not the administrative one — the layer
is 116 features and their `NSO_NAME` values are the census's own row labels, parentheticals
and all: `Basilan (excluding the City of Isabela)`, `Maguindanao (including the City of
Cotabato)`. And its **ADM4 is 42,042 barangays**, nationally complete, each with the 10-digit
PSGC and a population.

## The join is by NAME, and that is normally a warning sign

sources/lk_geo.py is the standing lesson that a code join can be silently wrong, and here
the codes are no help anyway: ADM2 carries an old NSO-style `PH19007` while the census keys
on the 10-digit PSGC `1500700000`. So the join is `ADM2.NSO_NAME` -> `ph.csv geo_name`, and
it is exact on 116 of 117 with no folding, no fuzzy matching and no overrides — because both
strings are copies of the same PSA table.

**What makes that trustworthy is not the name agreement, it is check 3 below.** The
geodatabase carries PSA's religion table as its own layer, with `RLG_HPOP` = the household
population the religion figures are tabulated on. That is the same quantity `ph.csv` holds
and it is not implied by the name: on a scrambled join it would disagree everywhere.
**It agrees EXACTLY — to the person — on 115 of 116 units.**

## The 116th unit, and the one disagreement, are the same fact

The BARMM **Interim Province** — the 63 barangays the 2019 plebiscite moved out of six
Cotabato municipalities (sources/ph.md §2) — has no ADM2 polygon here, because the USCB put
those barangays back in Cotabato: its ADM1 layer says so outright, `63 Interim Province
Barangays moved to Soccsksargen`. So Cotabato is the one unit whose `RLG_HPOP` disagrees
with the census, and it disagrees by **exactly 215,348**, which is exactly the Interim
Province's household population. The two halves of the discrepancy are the same 63 barangays
seen from either side, and nothing else in the country moves.

They are recoverable to the barangay because the USCB tagged every one of them in
`USCBCMNT` with the BARMM cluster it came from — `From BARMM, Interim Province, Pikit
Cluster II`. Exactly 63 rows carry that prefix, in eight clusters, and they are moved back
to the Interim Province here. **This is the whole reason the Interim Province can be drawn
at all**: there is no polygon for it in any boundary set on the internet, and this file
reconstructs it out of its parts.

## Placement weight is barangay POPULATION, not equal shares

spec §8.2 leans on placement layers that were *designed* to a population target — US tracts,
Australian SA1s — so that an equal share per polygon is already a population weighting.
Philippine barangays are nothing of the kind: they run from a few hundred people to a few
hundred thousand, and Manila's 896 of them are not the same size as Palawan's. So the weight
is the barangay's own 2020 population, carried in `pop` and used by `_PhWeighter` in
countries.py. 42,042 barangays for 108.7M people is ~2,590 people each, comparable to a US
census tract.

The population here is TOTAL population (109,033,245) where the religion counts are
HOUSEHOLD population (108,667,043) — spec §3.7. The 0.34% difference is the institutional
population, and it does not matter for a weight that is normalised inside each unit. It is
visible in check 2 and is worth reading: the worst outlier in the country is Muntinlupa at
+4.7%, which is the New Bilibid Prison, and the tightest are the rural provinces.

Usage:
    python sources/ph_geo.py --fetch     304 MB geodatabase from HDX
    python sources/ph_geo.py
"""

import csv
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ph")
GDB = os.path.join(RAW, "philippines_uscb.gdb.zip")
OUT_DIR = os.path.join(ROOT, "data", "geo", "ph")
OUT = os.path.join(OUT_DIR, "ph_barangays.gpkg")
LOOKUP = os.path.join(OUT_DIR, "ph_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "ph.csv")

URL = ("https://data.humdata.org/dataset/809dfb22-77f4-482c-8560-79b07d20fc15/resource/"
       "f7835bf0-8168-4f4e-9961-108167c381a1/download/philippines.gdb.zip")

L_ADM2 = "PH_GEOG_ADM2_2020_uscb_202402"
L_ADM4 = "PH_GEOG_ADM4_2020_uscb_202402"
L_POP = "PH_AGE_SEX_2020census_uscb_202402"
L_RELIGION = "PH_RELIGION_2020census_uscb_202402"

EXPECTED_ADM2 = 116          # the census tier minus the Interim Province
EXPECTED_ADM4 = 42042
EXPECTED_UNITS = 117         # what ph.csv holds
SGU_PREFIX = "From BARMM, Interim Province"
SGU_BARANGAYS = 63
SGU_NAME = "Interim Province"
COTABATO = "Cotabato (North Cotabato)"


def fetch():
    import requests
    import zipfile

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(GDB) and os.path.getsize(GDB) > 250_000_000:
        print("already have", GDB)
        return
    print("GET", URL)
    with requests.get(URL, timeout=1800, stream=True,
                      headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        with open(GDB, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    if not zipfile.is_zipfile(GDB):
        raise SystemExit(f"{GDB} is not a zip")
    print(f"  {os.path.getsize(GDB):,} bytes -> {GDB}")


def _read(layer, columns=None, geometry=False):
    """Read one geodatabase layer and ASSERT IT HAS ROWS.

    Not a formality. sources/cl_geo.py hit a geodatabase of this same dataset family that
    opened cleanly, reported the right CRS and returned ZERO features — no exception, no
    warning. A read that succeeds is not a read that returned data (spec §12), so every
    layer this script touches goes through here.
    """
    import pyogrio

    df = pyogrio.read_dataframe(GDB, layer=layer, columns=columns,
                                read_geometry=geometry)
    if len(df) == 0:
        raise SystemExit(
            f"layer {layer} read cleanly and returned ZERO features -- that is the "
            "geodatabase failure mode in sources/cl_geo.py. Check the GDAL driver, not "
            "the file.")
    return df


def main():
    import pandas as pd

    if not os.path.exists(GDB):
        raise SystemExit(f"missing {GDB} -- run with --fetch first")
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/ph.py first")

    # ---- census side ------------------------------------------------------------------
    # keep_default_na=False because one of PSA's 129 categories is literally "None" and
    # pandas turns it into NaN, which would drop 43,931 irreligious Filipinos without a
    # word. countries.py reads this file the same way for the same reason.
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    fine = df[df["geo_level"].isin(["province", "city", "municipality"])]
    hh = fine[fine["source_category"] == "Household Population"]
    if len(hh) != EXPECTED_UNITS:
        raise SystemExit(f"{len(hh)} census units, expected {EXPECTED_UNITS}")
    cen_id = dict(zip(hh["geo_name"], hh["geo_id"]))
    cen_pop = dict(zip(hh["geo_name"], hh["count"]))
    print(f"census units: {len(hh)}  ({hh['count'].sum():,} people, household population)")

    # ---- geo side ---------------------------------------------------------------------
    a2 = _read(L_ADM2, ["GEO_MATCH", "NSO_NAME"])
    print(f"ADM2 (census tabulation tier): {len(a2)}")
    if len(a2) != EXPECTED_ADM2:
        raise SystemExit(f"expected {EXPECTED_ADM2} ADM2 features")

    print(f"reading {EXPECTED_ADM4:,} barangay polygons…")
    a4 = _read(L_ADM4, ["GEO_MATCH", "ADM4_NAME", "NSO_CODE", "USCBCMNT"], geometry=True)
    print(f"ADM4: {len(a4):,}  crs={a4.crs}")
    if len(a4) != EXPECTED_ADM4:
        raise SystemExit(f"expected {EXPECTED_ADM4} ADM4 features")
    if not a4["NSO_CODE"].astype(str).str.fullmatch(r"PH\d{10}").all():
        raise SystemExit("ADM4 NSO_CODE is not PH + 10 digits -- the USCB has changed it")
    if a4["NSO_CODE"].duplicated().any():
        raise SystemExit("duplicate barangay PSGC in ADM4")

    pop = _read(L_POP, ["GEO_MATCH", "BTOTL"])
    a4 = a4.merge(pop, on="GEO_MATCH", how="left")
    if a4["BTOTL"].isna().any():
        raise SystemExit(f"{int(a4['BTOTL'].isna().sum())} barangays have no population row")

    # ---- assign every barangay to a census unit ---------------------------------------
    # GEO_MATCH is the nesting key, PHL_<adm1>_<adm2>_<adm3>_<adm4>; its first three
    # segments name the ADM2 the barangay sits in.
    a4["a2key"] = a4["GEO_MATCH"].str.split("_").str[:3].str.join("_")
    orphan = set(a4["a2key"]) - set(a2["GEO_MATCH"])
    if orphan:
        raise SystemExit(f"barangays under unknown ADM2 keys: {sorted(orphan)[:5]}")
    a4["unit_name"] = a4["a2key"].map(dict(zip(a2["GEO_MATCH"], a2["NSO_NAME"])))

    # The Interim Province, put back. See the docstring: these 63 are inside Cotabato in
    # this file and inside their own census row in ph.csv.
    sgu = a4["USCBCMNT"].astype(str).str.startswith(SGU_PREFIX)
    print(f"\n  BARMM Special Geographic Area: {int(sgu.sum())} barangays tagged "
          f"{SGU_PREFIX!r}")
    if int(sgu.sum()) != SGU_BARANGAYS:
        raise SystemExit(
            f"expected exactly {SGU_BARANGAYS} Interim Province barangays, found "
            f"{int(sgu.sum())} -- the USCBCMNT tagging has changed and the Interim "
            "Province can no longer be reconstructed. Do NOT proceed: its 215,348 people "
            "would silently land in Cotabato.")
    if not (a4.loc[sgu, "unit_name"] == COTABATO).all():
        raise SystemExit("a tagged Interim Province barangay is not inside Cotabato")
    print(f"    in {a4.loc[sgu, 'USCBCMNT'].nunique()} clusters, "
          f"{a4.loc[sgu, 'BTOTL'].sum():,} people — moved out of {COTABATO!r}")
    a4.loc[sgu, "unit_name"] = SGU_NAME

    # ---- check 1: every census unit has polygons, and vice versa (§12, both ways) -----
    g, c = set(a4["unit_name"]), set(cen_pop)
    print(f"\n  check 1 — the join, both ways:")
    print(f"    matched                  {len(g & c):>5}")
    print(f"    census with no polygon   {len(c - g):>5}  {sorted(c - g)}")
    print(f"    polygons with no census  {len(g - c):>5}  {sorted(g - c)}")
    if g != c:
        raise SystemExit("join FAILED -- every census unit must have barangays")
    a4["unit"] = a4["unit_name"].map(cen_id)

    # ---- check 2: total / household population, per unit ------------------------------
    # A quantity the name join does not determine. Every unit should sit just above 1.0:
    # the gap is the institutional population (spec §3.7), which is 0.34% nationally and
    # concentrated where the prisons and dormitories are.
    per = a4.groupby("unit_name")["BTOTL"].sum()
    ratio = (per / pd.Series(cen_pop)).sort_values()
    print(f"\n  check 2 — total population / census household population, {len(ratio)} units:")
    print(f"    min {ratio.iloc[0]:.4f} ({ratio.index[0]})   "
          f"median {ratio.median():.4f}   max {ratio.iloc[-1]:.4f} ({ratio.index[-1]})")
    if ratio.min() < 0.97 or ratio.max() > 1.10:
        raise SystemExit("the total/household ratio is not systematic -- the join is "
                         "pairing the wrong units")
    print("    OK  every unit's institutional gap sits in a narrow band above 1, which a "
          "scrambled\n        join cannot produce. The max is the New Bilibid Prison.")

    # ---- check 3: the geodatabase's OWN religion table, against ph.csv ----------------
    # The strongest check available and entirely independent of the names: RLG_HPOP is the
    # denominator PSA tabulated the religion figures on, and ph.csv holds the same number
    # from the same table read separately. Exact agreement on a unit is not something a
    # wrong join produces.
    rel = _read(L_RELIGION, ["GEO_MATCH", "RLG_HPOP"])
    chk = a2.merge(rel, on="GEO_MATCH", how="left")
    chk["census"] = chk["NSO_NAME"].map(cen_pop)
    if chk["RLG_HPOP"].isna().any() or chk["census"].isna().any():
        raise SystemExit("check 3 could not be run -- a unit is missing on one side")
    chk["diff"] = chk["RLG_HPOP"] - chk["census"]
    exact = int((chk["diff"] == 0).sum())
    print(f"\n  check 3 — the gdb's own religion table vs ph.csv, household population:")
    print(f"    EXACT on {exact} of {len(chk)} units")
    off = chk[chk["diff"] != 0]
    for _, r in off.iterrows():
        print(f"      {r['NSO_NAME']}: gdb {r['RLG_HPOP']:,} vs census {r['census']:,}  "
              f"diff {r['diff']:+,}")
    if exact != len(chk) - 1 or list(off["NSO_NAME"]) != [COTABATO]:
        raise SystemExit("check 3 FAILED -- the only expected disagreement is Cotabato")
    if int(off["diff"].iloc[0]) != int(cen_pop[SGU_NAME]):
        raise SystemExit(
            f"Cotabato is off by {int(off['diff'].iloc[0]):,}, which is NOT the Interim "
            f"Province's {cen_pop[SGU_NAME]:,} -- the discrepancy is something else and "
            "the reconstruction above is not justified")
    print(f"    the one difference is exactly the Interim Province "
          f"({cen_pop[SGU_NAME]:,}), which the USCB folded into Cotabato and this script "
          "takes back out.")

    # ---- write ------------------------------------------------------------------------
    out = a4[["NSO_CODE", "unit", "ADM4_NAME", "BTOTL", "geometry"]].copy()
    out["NSO_CODE"] = out["NSO_CODE"].str[2:]
    out = out.rename(columns={"NSO_CODE": "bgy", "ADM4_NAME": "name", "BTOTL": "pop"})
    if out["unit"].isna().any():
        raise SystemExit("a barangay has no unit")
    if out.geometry.isna().any() or out.geometry.is_empty.any():
        n = int(out.geometry.isna().sum() + out.geometry.is_empty.sum())
        raise SystemExit(f"{n} barangays have empty geometry")
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_file(OUT, layer="barangays", driver="GPKG")
    print(f"\nwrote {OUT}\n  {len(out):,} barangays, {out['pop'].sum():,} people, "
          f"{out['unit'].nunique()} units")

    with open(LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "geo_name", "barangays", "gdb_pop", "census_household_pop"])
        for name in sorted(cen_pop):
            w.writerow([cen_id[name], name, int((a4["unit_name"] == name).sum()),
                        int(per[name]), int(cen_pop[name])])
    print(f"wrote {LOOKUP} ({len(cen_pop)} rows)")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    main()
