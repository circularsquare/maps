"""Haiti — boundaries and populations for the ten departments.

Writes data/geo/ht/ht_departements.gpkg and data/geo/ht/ht_lookup.csv.

OCHA COD-AB Haiti (`cod-ab-hti`), the **shapefile** bundle rather than the geodatabase on
§12's Chile rule, read with `engine="fiona"`. Ten ADM1 features, which is the tier every
Haitian source counts at.

## THE CODE JOIN IS ZERO FOR TEN, AND IT LOOKS PERFECTLY REASONABLE

ECVMAS numbers the departments **1 to 10 in French alphabetical order** — Artibonite 1,
Centre 2, Grand'Anse 3, Nippes 4 — and COD's pcodes run `HT01` to `HT10` in Haiti's own
**traditional order**, which starts at Ouest and works round the country. Both are dense
1..10 integer sequences over the same ten units, so `DEPT -> HTnn` runs without a warning
and pairs **none of the ten correctly**: it sends Artibonite's respondents to Ouest and
Nord's to Artibonite. Every national total survives it intact. `check_code_join()` asserts
the count is still zero, because a re-cut that made it *mostly* right is the dangerous
direction ([[reference_name_join_wrong_neighbour]]).

So the join is on the FRENCH name (`adm1_name1`; `adm1_name` is COD's English), and it needs
exactly one alias: ECVMAS writes **Grand'Anse** where COD writes **Grande'Anse**.

## THE POPULATION IS A PROJECTION AND THERE IS NO ALTERNATIVE

**Haiti has not counted since January 2003.** The fifth census was scheduled for 2018-19 and
has not been held, so every population figure for Haiti today is a projection off the 2003
base and there is no census to prefer to COD-PS the way `sources/do_geo.py` preferred the
Dominican Republic's 2022 count. This file therefore takes COD-PS 2024 (11,899,555 people)
and prints the comparison against the 2003 census department totals so the size of the
extrapolation is on the record rather than assumed. The direction is the surprising one:
**Ouest is 37.0% of Haiti in the census and 33.4% in the 2024 projection**, so the model has
Port-au-Prince's department growing more slowly than the country, and the departments that
gain are Centre, Sud and Sud-Est.

**The 2003 census is on NINE departments, because Nippes did not exist yet.** It was split
out of Grand'Anse in September 2003, eight months after enumeration, so the census's
Grand'Anse row of 626,928 covers both of today's units and the comparison above is run on
nine. The same thing is why `sources/ht.py`'s ECVH 2001 cross-check has nine departments and
its ECVMAS 2012 build has ten.

Usage:
    python sources/ht_geo.py --fetch    a ~5.7 MB zip from HDX and a 4 KB csv
    python sources/ht_geo.py            rebuild from data/raw/ht/
"""

import os
import sys
import unicodedata
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ht")
SHP_DIR = os.path.join(RAW, "shp")
OUT_DIR = os.path.join(ROOT, "data", "geo", "ht")
OUT = os.path.join(OUT_DIR, "ht_departements.gpkg")
LOOKUP = os.path.join(OUT_DIR, "ht_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

N_DEPARTMENTS = 10

DOWNLOADS = {
    # OCHA COD-AB, https://data.humdata.org/dataset/cod-ab-hti
    "hti_admin_boundaries.shp.zip":
        "https://data.humdata.org/dataset/777e8b06-337f-4295-80bc-ca1515244215/resource/"
        "1da7821f-a1ae-46f3-95f4-3794c33f0079/download/hti_admin_boundaries.shp.zip",
    # OCHA COD-PS 2024, https://data.humdata.org/dataset/cod-ps-hti
    "hti_admpop_adm1_2024.csv":
        "https://data.humdata.org/dataset/95ba5281-fd76-4d3a-ae29-247e1ff26447/resource/"
        "c2bbfc4a-d000-47fd-82d9-a74e44d7aa22/download/hti_admpop_adm1_2024.csv",
}

# ECVMAS's own `DEPT` value labels, verbatim from the .sav's label set. French alphabetical.
ECVMAS_DEPARTMENTS = {
    1: "Artibonite", 2: "Centre",  3: "Grand'Anse", 4: "Nippes", 5: "Nord",
    6: "Nord-Est",   7: "Nord-Ouest", 8: "Ouest",   9: "Sud",   10: "Sud-Est",
}

# ECVMAS name -> COD's French name, where the two spellings differ. Exactly one.
ALIAS = {"Grand'Anse": "Grande'Anse"}

# How many of the ten a naive `DEPT -> HTnn` join happens to get right. NONE of them.
CODE_JOIN_CORRECT = 0

# IHSI, 4eme RGPH 2003, resident population by department, read off the census's own
# summary sheet (`RGPH03_Resume_HAI.pdf`, the "POPULATION DEUX SEXES" column). NINE keys:
# Nippes was created out of Grand'Anse in September 2003 and the census's Grand'Anse row
# covers both. Used ONLY to print how far COD-PS 2024 has moved the departmental shares;
# nothing is drawn from it.
CENSUS_2003 = {
    ("HT01",): 3_096_967,            # Ouest
    ("HT02",): 484_675,              # Sud-Est
    ("HT03",): 823_043,              # Nord
    ("HT04",): 308_385,              # Nord-Est
    ("HT05",): 1_299_398,            # Artibonite
    ("HT06",): 581_505,              # Centre
    ("HT07",): 621_651,              # Sud
    ("HT08", "HT10"): 626_928,       # Grand'Anse, Nippes not yet split off
    ("HT09",): 531_198,              # Nord-Ouest
}
CENSUS_2003_TOTAL = 8_373_750


def fold(s):
    """Accent-, case- and punctuation-insensitive key for a department name."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s.lower() if ch.isalnum())


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for name, url in DOWNLOADS.items():
        dst = os.path.join(RAW, name)
        if os.path.exists(dst):
            print(f"  have {name} ({os.path.getsize(dst):,} bytes)")
            continue
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=600) as r, open(dst + ".part", "wb") as f:
            f.write(r.read())
        os.replace(dst + ".part", dst)
        print(f"  got  {name} ({os.path.getsize(dst):,} bytes)")


def check_code_join(by_name):
    """Report what a `DEPT -> HTnn` join would have done, and refuse to let it pass."""
    right, wrong = [], []
    for code, name in sorted(ECVMAS_DEPARTMENTS.items()):
        naive = f"HT{code:02d}"
        actual = by_name[fold(ALIAS.get(name, name))]
        (right if naive == actual else wrong).append((code, name, naive, actual))
    print(f"\n  witness 2 — the code join is NOT used. `DEPT -> HTnn` would pair "
          f"{len(right)} of {N_DEPARTMENTS} correctly and MISPAIR {len(wrong)}:")
    for code, name, naive, actual in wrong:
        other = next(n for n, p in by_name.items() if p == naive)
        print(f"      ECVMAS {code:>2} {name:<12} -> {naive}, which COD says is {other!r} "
              f"(the real one is {actual})")
    if len(right) != CODE_JOIN_CORRECT:
        raise SystemExit(
            f"the code join now gets {len(right)} of {N_DEPARTMENTS} right, not "
            f"{CODE_JOIN_CORRECT}. Somebody has re-cut Haiti's pcodes or ECVMAS's order — "
            "STOP and decide deliberately. Do not delete this assertion.")


def main():
    if "--fetch" in sys.argv:
        fetch()

    zpath = os.path.join(RAW, "hti_admin_boundaries.shp.zip")
    if not os.path.exists(zpath):
        raise SystemExit(f"{zpath} missing — run with --fetch")
    os.makedirs(SHP_DIR, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(SHP_DIR)

    # `hti_admin1.shp`, not `hti_admin1_em.shp`: the `_em` layer is the edge-matched variant
    # cut against the Dominican Republic's boundary, same ten features, different border.
    shp = os.path.join(SHP_DIR, "hti_admin1.shp")
    g = gpd.read_file(shp, engine="fiona")
    if len(g) != N_DEPARTMENTS:
        raise SystemExit(f"{len(g)} ADM1 features, expected {N_DEPARTMENTS} — COD has "
                         "re-cut Haiti")
    if g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"unexpected CRS {g.crs}")
    print(f"read {shp}: {len(g)} departments, {g.crs}")

    g["pcode"] = g["adm1_pcode"].astype(str).str.strip()
    # adm1_name is COD's ENGLISH ("North", "West"); adm1_name1 is the French every Haitian
    # source uses. Joining on the English one silently loses six of the ten.
    g["name_fr"] = g["adm1_name1"].astype(str).str.strip()
    by_name = dict(zip(g["name_fr"].map(fold), g["pcode"]))
    if len(by_name) != N_DEPARTMENTS:
        raise SystemExit("COD's French department names are not unique — the join is unsafe")

    # ---- witness 1: every ECVMAS department name is a COD French name ----
    missing = [n for n in ECVMAS_DEPARTMENTS.values()
               if fold(ALIAS.get(n, n)) not in by_name]
    spare = [n for n in g["name_fr"]
             if fold(n) not in {fold(ALIAS.get(v, v)) for v in ECVMAS_DEPARTMENTS.values()}]
    if missing or spare:
        print(f"    ECVMAS names with no polygon: {missing}")
        print(f"    polygons with no ECVMAS name: {spare}")
        raise SystemExit("the name join FAILED")
    print(f"  witness 1 — all {N_DEPARTMENTS} ECVMAS names match a COD French name, "
          f"with {len(ALIAS)} alias ({', '.join(f'{k} = {v}' for k, v in ALIAS.items())})")

    check_code_join(by_name)

    # ---- populations, joined on the pcode, which is COD's own key on both sides ----
    ppath = os.path.join(RAW, "hti_admpop_adm1_2024.csv")
    pop = pd.read_csv(ppath, encoding="utf-8-sig")
    pop["pcode"] = pop["ADM1_PCODE"].astype(str).str.strip()
    if len(pop) != N_DEPARTMENTS or set(pop["pcode"]) != set(g["pcode"]):
        raise SystemExit("COD-PS ADM1 does not cover the same ten pcodes as COD-AB")
    g = g.merge(pop[["pcode", "T_TL"]], on="pcode", how="left")
    if g["T_TL"].isna().any():
        raise SystemExit("a department came out of the population join with no total")
    g["pop"] = g["T_TL"].astype("int64")
    print(f"  witness 3 — COD-PS 2024 joins on the pcode: {g['pop'].sum():,} people")

    # Ouest holds Port-au-Prince and is by far the densest; Nippes and Grand'Anse, the
    # southern peninsula, are the emptiest. A permuted population join is what this catches.
    g["density"] = g["pop"] / g["area_sqkm"]
    hi = g.loc[g["density"].idxmax(), "name_fr"]
    lo = g.loc[g["density"].idxmin(), "name_fr"]
    print(f"    sparsest {lo!r} at {g['density'].min():.0f}/km², "
          f"densest {hi!r} at {g['density'].max():.0f}/km²")
    if fold(hi) != fold("Ouest"):
        raise SystemExit("Ouest is not the densest department — the population join is "
                         "permuted")

    # ---- how far the projection has moved since the only census Haiti has ----
    if sum(CENSUS_2003.values()) != CENSUS_2003_TOTAL:
        raise SystemExit(f"the 2003 department totals sum to {sum(CENSUS_2003.values()):,}, "
                         f"not the census's {CENSUS_2003_TOTAL:,}")
    print(f"\n  COD-PS 2024 against the 2003 census, which is the last time Haiti counted "
          f"({g['pop'].sum() / CENSUS_2003_TOTAL:.2f}x nationally over 21 years, on the "
          f"census's nine departments):")
    by_pcode = dict(zip(g["pcode"], g["pop"]))
    rows = []
    for pcodes, p03 in CENSUS_2003.items():
        name = " + ".join(g.loc[g["pcode"] == p, "name_fr"].iloc[0] for p in pcodes)
        s03 = p03 / CENSUS_2003_TOTAL
        s24 = sum(by_pcode[p] for p in pcodes) / g["pop"].sum()
        rows.append((name, s03, s24, (s24 - s03) * 100))
    for name, s03, s24, d in sorted(rows, key=lambda x: -x[3]):
        print(f"    {name:<24} {s03 * 100:5.1f}% of Haiti in 2003 -> {s24 * 100:5.1f}% "
              f"in 2024  ({d:+.1f} pt)")

    g["unit"] = g["pcode"]
    # geo_id IS the pcode. ECVMAS's own 1..10 never leaves sources/ht.py, so there is only
    # one department numbering anywhere under data/ and nobody downstream can pick the wrong
    # one. That is the whole defence against the zero-for-ten join above.
    g["geo_id"] = g["pcode"]
    g["name"] = g["name_fr"]

    os.makedirs(OUT_DIR, exist_ok=True)
    out = g[["unit", "name", "pcode", "geo_id", "pop", "geometry"]]
    out.to_file(OUT, layer="departements", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    order = sorted(g["pcode"])
    lut = pd.DataFrame({
        "geo_id": order,
        "unit": order,
        "name": [g.loc[g["pcode"] == p, "name"].iloc[0] for p in order],
        "pop_2024": [int(g.loc[g["pcode"] == p, "pop"].iloc[0]) for p in order],
        "ecvmas_dept": [next(c for c, n in ECVMAS_DEPARTMENTS.items()
                             if by_name[fold(ALIAS.get(n, n))] == p) for p in order],
    })
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows, with ECVMAS's DEPT code alongside)")


if __name__ == "__main__":
    main()
