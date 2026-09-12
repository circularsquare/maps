"""Suriname — boundaries for the 62 ressorten.

Writes data/geo/sr/sr_ressorten.gpkg and data/geo/sr/sr_lookup.csv.

OCHA COD-AB Suriname (`sur_adm_2017_SHP`), from HDX. **COD's ADM2 is the census's ressort
tier exactly** — 62 polygons against 62 census columns — and its ADM1 is the ten districts,
so the two-level structure the census file implies is present in the boundary file as well.

**THE JOIN IS ON (DISTRICT, RESSORT) AND IT HAS TO BE.** Ressort names are **not unique**:
there is a `Welgelegen` in both Paramaribo and Coronie, and a `Centrum` in both Paramaribo
and Brokopondo. A name-only join collides on four units and would place Coronie's 1,000
people in Paramaribo. The census file has no district column; `sources/sr.py` reconstructs
it from `district-profiel-census.xls`'s ressorten-per-district counts, which sum to 62 and
consume the columns in order.

**FOUR NAMES DO NOT FOLD, AND THE RESOLUTION IS FORCED RATHER THAN CHOSEN.**

    Wanica       census `Koewarasan`      COD `Kwarasan`
    Marowijne    census `Moengo Tapoe`    COD `Moengo Tapu`
    Brokopondo   census `Marchallkreeek`  COD `Marechallkreek`   (three e's, ABS's typo)
    Sipaliwini   census `Coeroeni`        COD `Coeroenie`

No alias table is written for these. Instead the exact fold runs first, and then **a
district with exactly one unmatched census ressort and exactly one unmatched polygon has
them paired by elimination** — which is a derivation, not a guess, and which stops the run
if a future vintage ever leaves two of either in one district. That is the §12-safe form of
what an alias list does: it cannot go stale, because it re-derives itself every run.

ABS's spellings are transcribed as printed, typo included, per §12.

Usage:
    python sources/sr_geo.py --fetch    one ~680 KB zip from HDX
    python sources/sr_geo.py            rebuild from data/raw/sr/
"""

import os
import re
import sys
import unicodedata
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "sr")
OUT_DIR = os.path.join(ROOT, "data", "geo", "sr")
OUT = os.path.join(OUT_DIR, "sr_ressorten.gpkg")
LOOKUP = os.path.join(OUT_DIR, "sr_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "sr.csv")

ZIP_URL = ("https://data.humdata.org/dataset/ab35e673-76f2-43e5-a6a4-a5b81f9e093c/"
           "resource/fd31ebfd-bf77-4b7d-902f-3ea3a9c2d7a2/download/sur_adm_2017_shp.zip")
ZIP_NAME = "sur_adm_2017_shp.zip"
SHP = "sur_admbnda_adm2_2017.shp"
EXPECTED = 62

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

PAREN = re.compile(r"\s*\([^)]*\)")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, ZIP_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 200_000:
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


def fold(s, district=None):
    """Drop parentheticals, accents, a leading district name, and punctuation.

    `Welgelegen (Par'bo)` -> `welgelegen`; `Brokopondo Centrum` -> `centrum`. The
    parenthetical and the district prefix are ABS's own disambiguators for the duplicate
    names, and COD carries neither, so both have to come off before comparing.
    """
    s = PAREN.sub("", str(s or ""))
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = " ".join(s.split())
    if district:
        s = re.sub(r"^" + re.escape(district) + r"\s+", "", s, flags=re.I)
    return "".join(c for c in s.lower() if c.isalnum())


def _read_adm2():
    import geopandas as gpd

    zp = os.path.join(RAW, ZIP_NAME)
    if not os.path.exists(zp):
        raise SystemExit(f"missing {zp} -- run with --fetch first")
    with zipfile.ZipFile(zp) as z:
        if SHP not in z.namelist():
            raise SystemExit(f"{zp} has no {SHP} -- it holds {z.namelist()[:8]}")
    g = gpd.read_file(f"zip://{zp}!{SHP}")
    # §12 (Chile): a read that succeeds is not a read that returned data.
    if len(g) != EXPECTED:
        raise SystemExit(f"{SHP} returned {len(g)} features, expected {EXPECTED}")
    for c in ("ADM1_NL", "ADM2_NL", "ADM2ALT1NL", "ADM2_PCODE"):
        if c not in g.columns:
            raise SystemExit(f"{SHP} has no {c!r} -- columns are {list(g.columns)}")
    return g


def main():
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    sys.path.insert(0, HERE)
    from sr import RESSORTEN

    g = _read_adm2()
    print(f"COD ADM2: {len(g)} polygons, crs={g.crs}")

    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/sr.py first")
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False)
    if df["geo_id"].nunique() != EXPECTED:
        raise SystemExit(f"{df['geo_id'].nunique()} census ressorten, expected {EXPECTED}")

    cod = list(zip(g["ADM1_NL"], g["ADM2_NL"], g["ADM2ALT1NL"], g["ADM2_PCODE"]))

    # ---- pass 1: the exact fold, on (district, name) ----
    by_key = {}
    for d, nm, alt, pc in cod:
        by_key.setdefault((fold(d), fold(nm, d)), pc)
        if alt:
            by_key.setdefault((fold(d), fold(alt, d)), pc)

    pairs, taken, leftover = {}, set(), []
    for dist, nm, _ in RESSORTEN:
        pc = by_key.get((fold(dist), fold(nm, dist)))
        if pc and pc not in taken:
            pairs[(dist, nm)] = pc
            taken.add(pc)
        else:
            leftover.append((dist, nm))
    print(f"\n  matched on an exact fold: {len(pairs)} of {EXPECTED}")

    # ---- pass 2: sole-remainder elimination, per district ----
    forced = []
    for dist, nm in leftover:
        rest = [pc for d, n2, a, pc in cod if fold(d) == fold(dist) and pc not in taken]
        if len(rest) != 1:
            raise SystemExit(
                f"{dist}/{nm!r}: {len(rest)} unmatched polygons left in that district, so "
                "the pairing is NOT forced. COD or ABS has changed a name; resolve it "
                "explicitly rather than letting this guess.")
        pairs[(dist, nm)] = rest[0]
        taken.add(rest[0])
        codname = next(n2 for d, n2, a, pc in cod if pc == rest[0])
        forced.append((dist, nm, codname, rest[0]))

    print(f"  resolved as the sole remainder in their district: {len(forced)}")
    for dist, nm, codname, pc in forced:
        print(f"      {dist:<12} census {nm!r} -> COD {codname!r} ({pc})")

    spare = [(d, nm, pc) for d, nm, a, pc in cod if pc not in taken]
    print("\n  the join, both ways (§12):")
    print(f"    census ressorten           {len(RESSORTEN):>4}")
    print(f"    COD polygons               {len(cod):>4}")
    print(f"    matched                    {len(pairs):>4}")
    print(f"    polygons with no census    {len(spare):>4}")
    for d, nm, pc in spare:
        print(f"      no census : {pc} {d} / {nm}")
    if len(pairs) != EXPECTED or spare:
        raise SystemExit("join FAILED")

    # ---- the independent check: sr.py's hardcoded pcode vs the derivation above ----
    bad = [(d, nm, pc, pairs[(d, nm)]) for d, nm, pc in RESSORTEN
           if pairs[(d, nm)] != pc]
    print(f"\n    independent check — sr.py's RESSORTEN pcodes match the name derivation "
          f"on {EXPECTED - len(bad)}/{EXPECTED}")
    for d, nm, pc, got in bad:
        print(f"      {d}/{nm!r}: sr.py says {pc}, the join says {got}")
    if bad:
        raise SystemExit("sr.py's RESSORTEN pcodes disagree with the boundary join -- "
                         "every affected ressort's dots would be in the wrong place")

    out = g[["ADM1_NL", "ADM2_NL", "ADM2_PCODE", "geometry"]].rename(
        columns={"ADM1_NL": "district", "ADM2_NL": "name", "ADM2_PCODE": "pcode"})
    out["unit"] = out["pcode"].astype(str).str.strip()
    # Take the NAME from the statistical source, not the boundary file (§12, Chile).
    nso = {pairs[(d, nm)]: nm for d, nm, _ in RESSORTEN}
    out["name"] = out["pcode"].map(lambda c: nso[str(c).strip()])

    os.makedirs(OUT_DIR, exist_ok=True)
    out[["unit", "name", "district", "pcode", "geometry"]].to_file(
        OUT, layer="ressorten", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    lut = pd.DataFrame({"geo_id": sorted(taken), "unit": sorted(taken)})
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows)")


if __name__ == "__main__":
    main()
