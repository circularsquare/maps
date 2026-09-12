"""Croatia — town/municipality boundaries for the Popis 2021 religion data.

Writes data/geo/hr/hr_opcine.gpkg (556 polygons) and data/geo/hr/hr_lookup.csv, the
(county|name) -> LAU code map that countries.py joins on.

Same shape as Romania, and for the same reason: **the DZS workbook has no geographic
codes**, only names, so the key is (županija, name) and the bridge is the Eurostat
LAU 2021 – NUTS 2021 correspondence table. See `sources/ro_geo.md` §2 for the method; this
file only records where Croatia differs.

## Where Croatia differs

**1. The dash.** Istria is officially bilingual and its 20 municipalities carry
Croatian–Italian double names. DZS writes them with an EN DASH and Eurostat with a
HYPHEN-MINUS:

    DZS       Bale – Valle          U+2013
    Eurostat  Bale - Valle          U+002D

That is 20 of 555 units, all in one county, failing for a reason that has nothing to do
with geography. `squash()` folds both to a hyphen and collapses the spaces around it,
which also fixes `Murter-Kornati` vs `Murter - Kornati`.

**2. Three names repeat across counties** — Otok, Privlaka and Sveta Nedelja are each two
different municipalities. The county half of the key resolves all three, which is why the
NUTS3↔županija pairing is derived before the join rather than after.

**3. GRAD ZAGREB IS IN THE BOUNDARIES BUT NOT IN THE CENSUS ROWS.** DZS publishes the
capital only as its **17 gradske četvrti**, so the census has 555 municipalities where
Croatia has 556. GISCO has all 556, including `01333 Grad Zagreb`, which is therefore the
one polygon with no census municipality — expected, not an error.

## Zagreb is drawn as one polygon, and it need not have been

The census DOES give religion for all 17 districts. What is missing is only their
BOUNDARIES: they are not in GISCO LAU (which stops at the municipality), and an Overpass
query for `admin_level` 9 and 10 inside Grad Zagreb returns nothing, so OSM does not carry
them either at the obvious tagging.

So `countries.py` sums the 17 districts back into Grad Zagreb, and **19.8% of Croatia is
one 641 km² polygon** — worse than Bucharest, better than Tallinn. This is the only place
in the project where the split is available in the DATA and blocked by the GEOMETRY, which
makes it the cheapest capital fix outstanding: one boundary file for 17 polygons would do
it. Candidates not yet tried: DGU's Registar prostornih jedinica, and Zagreb's own
geoportal.

Usage:
    python sources/hr_geo.py --fetch
"""

import os
import sys
import unicodedata
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

GEO = os.path.join(ROOT, "data", "geo", "lau2021")
OUTER_ZIP = os.path.join(GEO, "ref-lau-2021-01m.shp.zip")
INNER_NAME = "LAU_RG_01M_2021_4326.shp.zip"
SHP_DIR = os.path.join(GEO, "shp4326")
CORR = os.path.join(GEO, "EU-27-LAU-2021-NUTS-2021.xlsx")

OUT_DIR = os.path.join(ROOT, "data", "geo", "hr")
OUT = os.path.join(OUT_DIR, "hr_opcine.gpkg")
LOOKUP = os.path.join(OUT_DIR, "hr_lookup.csv")
NORMALIZED = os.path.join(ROOT, "data", "normalized", "hr.csv")

EXPECTED_UNITS = 556
EXPECTED_CENSUS = 555
EXPECTED_COUNTIES = 20          # census municipalities span 20; Zagreb is the 21st
ZAGREB_LAU = "01333"


def fold(s):
    s = " ".join(str(s).split())
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.casefold().strip()


def squash(s):
    """fold(), plus every kind of dash to a hyphen with no surrounding spaces.

    Istria's bilingual names are the reason: DZS writes `Bale – Valle` with an en dash,
    Eurostat writes `Bale - Valle` with a hyphen-minus, and `Murter-Kornati` appears both
    spaced and unspaced.
    """
    s = fold(s)
    for dash in ("–", "—", "−"):
        s = s.replace(dash, "-")
    parts = [p.strip() for p in s.split("-")]
    return "-".join(p for p in parts if p)


def unpack():
    if not os.path.exists(OUTER_ZIP):
        raise SystemExit(f"missing {OUTER_ZIP} -- run sources/pl_geo.py --fetch first")
    inner = os.path.join(GEO, INNER_NAME)
    if not os.path.exists(inner):
        with zipfile.ZipFile(OUTER_ZIP) as z:
            z.extract(INNER_NAME, GEO)
    if not os.path.isdir(SHP_DIR):
        with zipfile.ZipFile(inner) as z:
            z.extractall(SHP_DIR)
    return os.path.join(SHP_DIR,
                        [f for f in os.listdir(SHP_DIR) if f.endswith(".shp")][0])


def read_correspondence():
    import openpyxl
    if not os.path.exists(CORR):
        raise SystemExit(f"missing {CORR} -- run sources/ro_geo.py --fetch first")
    wb = openpyxl.load_workbook(CORR, read_only=True, data_only=True)
    rows = []
    for r in wb["HR"].iter_rows(min_row=2, values_only=True):
        if r[0] and r[1] is not None and r[2]:
            rows.append((str(r[0]).strip(), str(r[1]).strip(),
                         " ".join(str(r[2]).split())))
    return rows


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        from pl_geo import fetch as fetch_lau
        from ro_geo import fetch as fetch_corr
        fetch_lau()
        fetch_corr()

    shp = unpack()
    corr = read_correspondence()
    print(f"correspondence: {len(corr):,} HR rows, "
          f"{len({n for n, _, _ in corr})} NUTS3 regions")
    if len(corr) != EXPECTED_UNITS:
        raise SystemExit(f"correspondence has {len(corr)} HR rows, "
                         f"expected {EXPECTED_UNITS}")

    df = pd.read_csv(NORMALIZED, dtype={"geo_id": str}, low_memory=False)
    cen = df[df["geo_level"] == "municipality"][["geo_id"]].drop_duplicates().copy()
    cen[["county", "name"]] = cen["geo_id"].str.split("|", n=1, expand=True)
    print(f"census municipalities: {len(cen):,} in {cen['county'].nunique()} counties")
    if len(cen) != EXPECTED_CENSUS:
        raise SystemExit(f"{len(cen)} census municipalities, expected {EXPECTED_CENSUS}")

    # ---- derive NUTS3 <-> zupanija by name-set overlap, as ro_geo.py does
    corr_by_nuts = {}
    for n3, code, name in corr:
        corr_by_nuts.setdefault(n3, []).append((squash(name), code))
    cen_by_county = {}
    for _, r in cen.iterrows():
        cen_by_county.setdefault(r["county"], set()).add(squash(r["name"]))

    pairs, used = {}, set()
    for county, names in sorted(cen_by_county.items(), key=lambda kv: -len(kv[1])):
        best, score = None, -1
        for n3, lst in corr_by_nuts.items():
            if n3 in used:
                continue
            s = len(names & {nm for nm, _ in lst})
            if s > score:
                best, score = n3, s
        pairs[county] = (best, score, len(names))
        used.add(best)
    bad = {c: v for c, v in pairs.items() if v[1] < v[2]}
    print(f"\n  NUTS3 <-> zupanija: {len(pairs)} pairs, "
          f"{len(pairs) - len(bad)} matching every name")
    for c, (n, s, t) in sorted(bad.items(), key=lambda kv: kv[1][1] - kv[1][2])[:8]:
        print(f"      {c:<24} {n}  {s}/{t}")
    if len(pairs) != EXPECTED_COUNTIES:
        raise SystemExit(f"{len(pairs)} counties, expected {EXPECTED_COUNTIES}")

    # ---- resolve (county, name) -> LAU code
    resolved, leftovers = {}, []
    for county, sub in cen.groupby("county"):
        n3 = pairs[county][0]
        lut = dict(corr_by_nuts[n3])
        taken = set()
        for _, r in sub.iterrows():
            code = lut.get(squash(r["name"]))
            if code is not None and code not in taken:
                resolved[r["geo_id"]] = code
                taken.add(code)
            else:
                leftovers.append((county, r["name"]))
    cen["kod"] = cen["geo_id"].map(resolved)
    unresolved = cen[cen["kod"].isna()]
    print(f"\n  {'OK ' if unresolved.empty else 'BAD'} census municipalities with no LAU "
          f"code: {len(unresolved)}")
    for county, name in leftovers[:15]:
        print(f"      {county} | {name}")
    if not unresolved.empty:
        raise SystemExit("name resolution FAILED")
    if cen["kod"].nunique() != len(cen):
        raise SystemExit("two census municipalities resolved to the same LAU code")

    # ---- polygons
    gdf = gpd.read_file(shp)
    hr = gdf[gdf["CNTR_CODE"] == "HR"].copy()
    hr["kod"] = hr["LAU_ID"].astype(str).str.strip()
    if len(hr) != EXPECTED_UNITS:
        raise SystemExit(f"{len(hr)} HR polygons, expected {EXPECTED_UNITS}")

    geo_keys, cen_keys = set(hr["kod"]), set(cen["kod"])
    missing = cen_keys - geo_keys
    extra = geo_keys - cen_keys
    print(f"\n  polygons {len(geo_keys):,}  |  census units {len(cen_keys):,}")
    print(f"  {'OK ' if not missing else 'BAD'} census units with no polygon: "
          f"{len(missing)}")
    for k in sorted(missing)[:10]:
        print("      no polygon:", k)
    # Grad Zagreb is the expected extra: the census replaces it with 17 districts.
    if extra == {ZAGREB_LAU}:
        print(f"  OK  1 polygon with no census municipality: {ZAGREB_LAU} Grad Zagreb "
              "— expected, the census gives its 17 districts instead")
    else:
        print(f"  BAD polygons with no census unit: {len(extra)} -> {sorted(extra)[:10]}")
    if missing or extra != {ZAGREB_LAU}:
        raise SystemExit("join FAILED")

    out = hr[["kod", "LAU_NAME", "POP_2021", "AREA_KM2", "geometry"]].rename(
        columns={"LAU_NAME": "name", "POP_2021": "pop_2021", "AREA_KM2": "area_km2"})
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_file(OUT, layer="opcine", driver="GPKG")
    print("\nwrote", OUT, f"({len(out):,} polygons)")

    # the districts have no polygons of their own, so they are routed to Grad Zagreb
    dist = df[df["geo_level"] == "city_district"][["geo_id"]].drop_duplicates().copy()
    dist["kod"] = ZAGREB_LAU
    lut = pd.concat([cen[["geo_id", "kod"]], dist[["geo_id", "kod"]]], ignore_index=True)
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print("wrote", LOOKUP, f"({len(lut):,} rows; {len(dist)} Zagreb districts -> "
          f"{ZAGREB_LAU})")

    tot = out["pop_2021"].sum()
    zg = out[out["kod"] == ZAGREB_LAU].iloc[0]
    print(f"\n  Zagreb is {100 * zg['pop_2021'] / tot:.2f}% of Croatia in one "
          f"{zg['area_km2']:.0f} km² polygon — the census could split it into 17 and the "
          "boundaries to do so were not found (see the header).")


if __name__ == "__main__":
    main()
