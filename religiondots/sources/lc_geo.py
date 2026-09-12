"""Saint Lucia — boundaries for the 10 districts.

Writes data/geo/lc/lc_districts.gpkg and data/geo/lc/lc_lookup.csv.

OCHA COD-AB Saint Lucia, `lca_admbnda_adm1_gov_2019.shp`, which is the government's own
boundary set rather than a redraw. **COD'S ADM1 IS CSO'S DISTRICT TIER EXACTLY**: 10
polygons against Table D.2's 10 columns, and the districts are the only tier the census
publishes religion on.

**THERE IS A FINER TIER AND IT IS NOT USABLE HERE.** The same zip carries `adm2` — **547
settlements**, which would be one of the finest geographies on this map at ~310 people each.
Religion is not published on it: Table D.2 is districts and nothing in the report cuts
religion below them, so adm2 would be a placement grid with no counts to place. Kontur's
population hexes do that job (`sources/lc_grid.py`) and do it without implying a tier the
census does not have.

**COD'S DISTRICT AREAS DISAGREE WITH THE CENSUS'S OWN, BY UP TO 71%, AND IT DOES NOT MATTER
— WHICH HAD TO BE MEASURED RATHER THAN ASSUMED.** CSO's `Table A.6` publishes a land area per
district in square miles, summing to the island's official 238.2. COD's polygons do not
reproduce it: **Dennery is 1.71x the census's figure**, Soufriere 0.73x, Canaries 0.79x,
Micoud 0.83x. `main()` prints the whole ladder on every run.

The obvious suspicion is Cayman's (§9at), where a ratio table like that was a boundary error
that put dots in the wrong district. **Here it is not, and three tests say so:**

  * **geoBoundaries** (gbOpen LCA ADM1, from Wikimedia Commons, CC0) reproduces Table A.6 to
    within 8.3% on every district, so a boundary set matching the census does exist.
  * **The two sets agree on 528 of COD's own 547 ADM2 settlements.** The nineteen that move
    are the interior: `Central Forest Reserve`, `Forest Reserve`, `La Sorciere`, and hamlets
    on a shared edge.
  * **And they place the same people.** Summing Kontur's 400 m population grid inside each
    set and comparing with the census district by district gives an rms deviation of
    **0.320 under COD and 0.317 under geoBoundaries** — Dennery holds 10,581 people under
    COD and 10,571 under geoBoundaries, *despite COD's Dennery being 51 km² larger*. The
    extra land is the Central Forest Reserve and nobody lives in it.

So COD is kept: it is the government's own file (`..._gov_2019`), it is what every other
country here uses, and geoBoundaries has its own defects — it spells Anse La Raye
`Anse la Raya`, and it puts the populated **Babonneau** settlements in Gros Islet where
COD's government-sourced ADM2 puts them in Castries. **The area disagreement is real, is
recorded, and changes no dot.**

**OSM IS NOT A THIRD OPINION HERE**: it has no `admin_level=6` relations for Saint Lucia at
all — the country's quarters are simply not in it — so the two sets above are the only ones.

**THE ONLY NAME DIFFERENCE IS A HYPHEN**, on one of the ten: COD writes `Vieux-Fort` where
CSO writes `Vieux Fort`. `fold()` drops everything that is not alphanumeric rather than
carrying an alias table, because a frozen list of renames goes stale in silence at the next
release (§12).

**THE JOIN IS BY NAME AND THE INDEPENDENT CHECK IS THE P-CODE.** CSO publishes no code at
all, so `sources/lc.py` carries COD's `ADM1_PCODE` against each district name by hand and
this file asserts the same pairing from the boundary side. Nothing else would catch a
transposition, because every total in `lc.py` reconciles whichever polygon a district is
paired with (§9n's `TMA` lesson, and the same arrangement as Barbados, Trinidad and Cayman).

**COD's pcodes are neither alphabetical nor the census's order.** They run LC04 Anse La Raye,
LC05 Canaries, LC06 Soufriere, LC07 Choiseul, LC08 Laborie, LC09 Vieux-Fort, LC10 Micoud,
LC11 Dennery, LC12 Gros Islet, LC13 Castries — anticlockwise from the west coast, with the
capital last and LC01–LC03 unused. Worth knowing only because it means the numbering carries
no meaning and must not be assumed to match anything.

Usage:
    python sources/lc_geo.py --fetch    one ~5.1 MB shapefile zip from HDX
    python sources/lc_geo.py            rebuild from data/raw/lc/
"""

import os
import sys
import unicodedata
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "lc")
OUT_DIR = os.path.join(ROOT, "data", "geo", "lc")
OUT = os.path.join(OUT_DIR, "lc_districts.gpkg")
LOOKUP = os.path.join(OUT_DIR, "lc_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "lc.csv")

ZIP_URL = ("https://data.humdata.org/dataset/e6b16151-3fcf-4159-88fd-257b37775fe2/"
           "resource/73039d66-2055-4488-a60c-3932936e2bcb/download/"
           "lca_admbnda_gov_2019_shp.zip")
ZIP_NAME = "lca_admbnda_gov_2019_shp.zip"
SHP = "lca_admbnda_adm1_gov_2019.shp"
SHP_ADM2 = "lca_admbnda_adm2_gov_2019.shp"
EXPECTED = 10

UTM = 32620                     # UTM 20N covers Saint Lucia

# CSO's Table A.6, *Household Population: Districts by Land Area, Person Count, and
# Population Density*, in square miles. Transcribed from the report because it is a PDF
# and because it is **used ONLY for the diagnostic below and for no drawn value** — the
# same arrangement as `ESTIMATED_BY_PARISH` in sources/bb.py. It cannot affect the map.
CSO_LAND_AREA_SQMI = {
    "Castries": 39.6, "Anse La Raye": 14.5, "Canaries": 9.4, "Soufriere": 22.6,
    "Choiseul": 10.0, "Laborie": 13.1, "Vieux Fort": 19.2, "Micoud": 43.3,
    "Dennery": 27.9, "Gros Islet": 38.7,
}
SQMI_KM2 = 2.589988
CSO_TOTAL_SQMI = 238.2          # the island's official land area
# The whole island must still come out right even where the internal division does not.
NATIONAL_AREA_TOLERANCE = 0.05

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, ZIP_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print("already have", dest)
        return
    print("GET", ZIP_URL)
    r = requests.get(ZIP_URL, timeout=900, stream=True, headers={"User-Agent": UA})
    r.raise_for_status()
    with open(dest + ".part", "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    os.replace(dest + ".part", dest)
    with open(dest, "rb") as fh:
        magic = fh.read(2)
    if magic != b"PK":
        raise SystemExit(f"{dest} is not a zip -- starts {magic!r}")
    print(f"  {os.path.getsize(dest):,} bytes")


def fold(s):
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s.lower() if c.isalnum())


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    zp = os.path.join(RAW, ZIP_NAME)
    if not os.path.exists(zp):
        raise SystemExit(f"missing {zp} -- run with --fetch first")
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
    for needed in (SHP, SHP_ADM2):
        if not any(n.endswith(needed) for n in names):
            raise SystemExit(f"{zp} has no {needed} -- it holds {names[:8]}")

    g = gpd.read_file(f"zip://{zp}!{SHP}")
    # §12 (Chile): a read that succeeds is not a read that returned data.
    if len(g) != EXPECTED:
        raise SystemExit(f"{SHP} returned {len(g)} features, expected {EXPECTED}")
    for c in ("ADM1_EN", "ADM1_PCODE"):
        if c not in g.columns:
            raise SystemExit(f"{SHP} has no {c!r} -- columns are {list(g.columns)}")
    print(f"COD-AB ADM1: {len(g)} districts, crs={g.crs}")

    # The finer tier, read only so this file can say how fine it is and why it is not
    # used. See the module docstring.
    n_adm2 = len(gpd.read_file(f"zip://{zp}!{SHP_ADM2}"))
    print(f"COD-AB ADM2: {n_adm2} settlements — NOT used; religion is not published "
          "below\n             the district, so this tier would be placement with no "
          "counts (§8.2)")

    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/lc.py first")
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[])
    cen = (df[df["geo_level"] == "district"]
           .drop_duplicates("geo_id")[["geo_id", "geo_name"]])
    census = dict(zip(cen["geo_id"], cen["geo_name"]))
    if len(census) != EXPECTED:
        raise SystemExit(f"{len(census)} census districts, expected {EXPECTED}")

    poly = {}
    for nm, cd in zip(g["ADM1_EN"], g["ADM1_PCODE"]):
        k = fold(nm)
        if k in poly:
            raise SystemExit(f"COD name {nm!r} folds onto an existing one")
        poly[k] = (nm, str(cd).strip())

    pairs, missing = {}, []
    for code, nm in census.items():
        k = fold(nm)
        if k in poly:
            pairs[code] = poly[k]
        else:
            missing.append((code, nm))
    used = {v[1] for v in pairs.values()}
    spare = [(nm, cd) for nm, cd in poly.values() if cd not in used]

    print("\n  the join, both ways (§12):")
    print(f"    census districts        {len(census):>4}")
    print(f"    COD polygons            {len(poly):>4}")
    print(f"    matched                 {len(pairs):>4}")
    print(f"    census with no polygon  {len(missing):>4}  {missing}")
    print(f"    polygons with no census {len(spare):>4}  {spare}")
    if missing or spare:
        raise SystemExit("join FAILED")

    bad = [(c, census[c], p[1]) for c, p in pairs.items() if p[1] != c]
    print(f"\n    independent check — lc.py's name->pcode matches COD's ADM1_PCODE on "
          f"{len(pairs) - len(bad)}/{len(pairs)}")
    for c, nm, pc in bad:
        print(f"      {nm!r}: lc.py says {c}, COD says {pc}")
    if bad:
        raise SystemExit("lc.py's DISTRICTS pcodes disagree with COD -- every district's "
                         "dots would be placed in the wrong district")

    variants = [(census[c], pairs[c][0]) for c in pairs if census[c] != pairs[c][0]]
    print(f"    {len(variants)} name variant(s) resolved by fold(): {variants}")

    out = g[["ADM1_EN", "ADM1_PCODE", "geometry"]].rename(
        columns={"ADM1_EN": "name", "ADM1_PCODE": "pcode"})
    out["unit"] = out["pcode"].astype(str).str.strip()
    # Take the NAME from the statistical source, not the boundary file (§12, Chile).
    nso = {pairs[c][1]: census[c] for c in pairs}
    out["name"] = out["pcode"].map(lambda c: nso[str(c).strip()])

    areas = out.to_crs(UTM).area / 1e6
    pop = {r["geo_id"]: int(r["count"]) for _, r in
           df[(df["geo_level"] == "district") &
              (df["source_category"] == "Total")].iterrows()}
    ordered = out.sort_values("unit")
    print(f"\n  {len(out)} districts:")
    for (_, row), a in zip(ordered.iterrows(), areas[ordered.index]):
        p = pop[row["unit"]]
        print(f"    {row['unit']}  {row['name']:<14} {a:7.1f} km2   {p:>7,} people   "
              f"{p / a:6.0f}/km2")

    # ---- COD's areas against the census's own. A DIAGNOSTIC, not a gate: see the
    #      module docstring for the three tests that show the gap is empty forest.
    by_name = {row["name"]: a for (_, row), a in zip(ordered.iterrows(),
                                                     areas[ordered.index])}
    print("\n  COD's polygons against CSO's Table A.6 land areas — the internal division "
          "disagrees\n  and the island does not. The gap is the Central Forest Reserve "
          "and it holds nobody:")
    tot_cod = tot_cso = 0.0
    rows = []
    for nm, sqmi in CSO_LAND_AREA_SQMI.items():
        cso = sqmi * SQMI_KM2
        cod = by_name[nm]
        tot_cod += cod
        tot_cso += cso
        rows.append((cod / cso, nm, cod, cso))
    for r, nm, cod, cso in sorted(rows, reverse=True):
        print(f"      {nm:<14} COD {cod:7.1f} km2   CSO {cso:7.1f} km2   {r:5.2f}x")
    nat = tot_cod / (CSO_TOTAL_SQMI * SQMI_KM2)
    print(f"      {'SAINT LUCIA':<14} COD {tot_cod:7.1f} km2   CSO "
          f"{CSO_TOTAL_SQMI * SQMI_KM2:7.1f} km2   {nat:5.2f}x")
    if abs(nat - 1.0) > NATIONAL_AREA_TOLERANCE:
        raise SystemExit(
            f"COD's polygons cover {nat:.1%} of Saint Lucia's official land area. The "
            "per-district disagreement is known and documented; a NATIONAL one is not, "
            "and would mean the boundary file is no longer the country.")

    os.makedirs(OUT_DIR, exist_ok=True)
    out[["unit", "name", "pcode", "geometry"]].to_file(
        OUT, layer="districts", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    pd.DataFrame({"geo_id": sorted(pairs), "unit": sorted(pairs)}).to_csv(
        LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(pairs)} rows)")


if __name__ == "__main__":
    main()
