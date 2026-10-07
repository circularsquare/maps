"""Papua New Guinea: the 22 provinces, COD-AB polygons with the 2024 census count.

Writes data/geo/pg/pg_provinces.gpkg and data/geo/pg/pg_lookup.csv. `sources/pg.md` §5 is the record.

  * **boundaries**: COD-AB `cod-ab-png` v01 (OCHA ROAP; boundaries created 2019-05-10, reviewed
    2025-10-30), admin1: 22 features PG01-PG22, Hela (PG21) and Jiwaka (PG22) included.
  * **population**: the 2024 census, Final Figures Table 2 (read by `sources/pg.py`, which writes
    each province's 2024 count into data/normalized/pg.csv as the sum of its rows).

THE JOIN IS ON P-CODE, and `sources/pg.py::PROVINCES` names each p-code's province. Its witness is
independent of both: COD-PS 2011 (`png_admpop_adm1_2011_v2.csv`, the 2011 census by p-code) must
equal, province by province, the 2011 column of the 2024 Final Figures' Table 2, which pg.py reads
by name. A p-code paired with the wrong name fails it unless two provinces had the same 2011 count.

Usage:
    python sources/pg_geo.py --fetch    COD-AB geojson zip and COD-PS 2011 csv into data/raw/pg/
    python sources/pg_geo.py            rebuild from data/raw/pg/
"""

import io
import os
import re
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "6")

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "pg")
GEO = os.path.join(ROOT, "data", "geo", "pg")
OUT = os.path.join(GEO, "pg_provinces.gpkg")
LOOKUP = os.path.join(GEO, "pg_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "pg.csv")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
COD_AB_URL = ("https://data.humdata.org/dataset/f260301d-c32f-4e9e-a7d6-05bde2f70b6b/resource/"
              "3ef879e4-0896-4793-a0c2-6d0636395cf5/download/png_admin_boundaries.geojson.zip")
COD_AB = os.path.join(RAW, "png_admin_boundaries.geojson.zip")
COD_PS_URL = ("https://data.humdata.org/dataset/5f76d06a-70f6-4cc4-a0f2-d9da9342fc06/resource/"
              "8b4346b6-b0a4-4e65-8d15-b6c79dbacee2/download/png_admpop_adm1_2011_v2.csv")
COD_PS = os.path.join(RAW, "png_admpop_adm1_2011_v2.csv")
METRIC_AREA = "ESRI:54034"


def fetch():
    import requests
    os.makedirs(RAW, exist_ok=True)
    for url, dst in ((COD_AB_URL, COD_AB), (COD_PS_URL, COD_PS)):
        if os.path.exists(dst) and os.path.getsize(dst) > 1_000:
            print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
            continue
        r = requests.get(url, headers={"User-Agent": UA}, timeout=600)
        r.raise_for_status()
        with open(dst + ".part", "wb") as fh:
            fh.write(r.content)
        os.replace(dst + ".part", dst)
        print(f"  got  {os.path.basename(dst)} ({len(r.content):,} bytes)")


def fold(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


def main():
    import geopandas as gpd
    from pg import PROVINCES, read_2024

    if "--fetch" in sys.argv or not (os.path.exists(COD_AB) and os.path.exists(COD_PS)):
        fetch()

    with zipfile.ZipFile(COD_AB) as z:
        name = next(n for n in z.namelist() if re.search(r"admin1\.geojson$", n))
        g = gpd.read_file(io.BytesIO(z.read(name)))
    print(f"COD-AB {name}: {len(g)} features")
    want = {pc for pc, _, _ in PROVINCES}
    if set(g["adm1_pcode"]) != want or len(g) != 22:
        raise SystemExit(f"COD-AB admin1 is not PG01-PG22: {sorted(g['adm1_pcode'])}")
    if g.geometry.isna().any() or g.geometry.is_empty.any():
        raise SystemExit("COD-AB has an empty province geometry")

    # ---- witness: COD-PS 2011 by p-code against Table 2's 2011 column by name ----
    pop11, pop24 = read_2024()
    ps = pd.read_csv(COD_PS, encoding="utf-8-sig")
    ps11 = dict(zip(ps["ADM1_PCODE"], ps["T_TL"].astype(int)))
    bad = [(pc, ps11.get(pc), pop11[pc]) for pc in want if ps11.get(pc) != pop11[pc]]
    if bad:
        raise SystemExit(f"COD-PS 2011 by p-code disagrees with Table 2 by name: {bad}")
    print("  witness: COD-PS 2011 equals the Final Figures' 2011 column for all 22 p-codes")
    for pc, nm, _ in PROVINCES:
        cod = g.loc[g["adm1_pcode"] == pc, "adm1_name"].iloc[0]
        if fold(nm)[:5] not in fold(cod) and pc not in ("PG10", "PG15"):
            raise SystemExit(f"{pc}: COD-AB {cod!r} against the census's {nm!r}")
    print("  names: every COD-AB name contains the census's (Chimbu = Simbu, West Sepik = Sandaun "
          "are spelled differently and exempted)")

    # ---- the 2024 count each province's rows add up to ----
    n = pd.read_csv(NORM, dtype={"geo_id": str})
    tot = n.groupby("geo_id")["count"].sum().to_dict()
    if tot != pop24:
        raise SystemExit("data/normalized/pg.csv does not add to the 2024 counts; re-run sources/pg.py")

    g["unit"] = g["adm1_pcode"]
    g["name"] = g["unit"].map({pc: lab for pc, _, lab in PROVINCES})
    g["pop"] = g["unit"].map(pop24).astype(int)
    area = g.to_crs(METRIC_AREA).area / 1e6
    print("\n  province, 2024 census, area km2, people per km2:")
    for (_i, r), a in sorted(zip(g.iterrows(), area), key=lambda t: -t[0][1]["pop"]):
        print(f"      {r['unit']}  {r['name']:<22} {r['pop']:>9,}  {a:>9,.0f}  {r['pop'] / a:>7.1f}")

    os.makedirs(GEO, exist_ok=True)
    g[["unit", "name", "pop", "geometry"]].to_file(OUT, layer="provinces", driver="GPKG")
    lut = g[["unit", "name", "pop"]].rename(columns={"unit": "geo_id"})
    lut["unit"] = lut["geo_id"]
    lut.sort_values("geo_id").to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} and {LOOKUP} (22 provinces, {int(g['pop'].sum()):,} people)")


if __name__ == "__main__":
    main()
