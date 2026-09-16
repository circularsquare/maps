"""Sudan: the 18 states, COD-AB polygons with COD-PS 2022 populations.

Writes data/geo/sd/sd_states.gpkg and data/geo/sd/sd_lookup.csv. `sources/sd.md` §5 is the record.

  * **boundaries**: COD-AB `cod-ab-sdn` v03 (OCHA ROSEA; boundaries created 2013-08-13, valid from
    2020-08-31, reviewed 2025-10-30), `sdn_admin1.geojson`: 19 features, the 18 states SD01-SD18
    and `Abyei PCA` SD19.
  * **population**: COD-PS `cod-ps-sdn`, `sdn_admpop_2022.xlsx` (published 2022-11-22), the
    Central Bureau of Statistics' projection from the 2008 census for 2022, 18 states,
    46,934,433. It is the last edition before the war that began in April 2023: the 2024 and 2025
    editions fold in displacement since then (the 2025 technical note, p.3), and this map's
    surveys were all fielded between 2013 and 2022.

TWO PIECES OF COD'S POLYGONS ARE NOT DRAWN:

  * **Abyei** (SD19). The COD-PS has no projection for it (the 2025 technical note says so of its
    own edition, and the 2022 table has no row), neither survey samples it, and its final status
    between Sudan and South Sudan is not settled. Dropped, asserted by p-code.
  * **The Halaib triangle**, the part of COD's Red Sea state north of 22 degrees N. Egypt
    administers it, and spec §14.18 gives disputed land to its de facto administrator. Clipped from
    SD10 only: the Wadi Halfa salient on the Nile, which Sudan does administer, is in Northern state
    and is left alone. The area removed is printed and pinned.

The join is on p-code (COD-PS `ADM1_PCODE` = COD-AB `adm1_pcode`), with the English names on both
sides as the witness.

Usage:
    python sources/sd_geo.py --fetch    COD-AB geojson zip and COD-PS 2022 workbook into data/raw/sd/
    python sources/sd_geo.py            rebuild from data/raw/sd/
"""

import io
import os
import re
import sys
import unicodedata
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "6")

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "sd")
GEO = os.path.join(ROOT, "data", "geo", "sd")
OUT = os.path.join(GEO, "sd_states.gpkg")
LOOKUP = os.path.join(GEO, "sd_lookup.csv")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
COD_AB_URL = ("https://data.humdata.org/dataset/a66a4b6c-92de-4507-9546-aa1900474180/resource/"
              "018af991-4aa7-4043-a0d5-e429a55851fb/download/sdn_admin_boundaries.geojson.zip")
COD_AB = os.path.join(RAW, "sdn_admin_boundaries.geojson.zip")
COD_PS_URL = ("https://data.humdata.org/dataset/f4d30e65-a162-437a-a35d-d81b99723bd6/resource/"
              "5295c1d0-7788-4970-923f-b2c91cb4e18c/download/sdn_admpop_2022.xlsx")
COD_PS = os.path.join(RAW, "sdn_admpop_2022.xlsx")

TOTAL_2022 = 46_934_433          # the workbook's ADM0 T_TL, asserted against the 18 state rows
ABYEI = "SD19"
RED_SEA = "SD10"
HALAIB_LAT = 22.0
HALAIB_KM2 = (15_000, 25_000)    # area clipped from SD10, pinned; measured 2026-09-15 (printed)
METRIC_AREA = "ESRI:54034"       # world cylindrical equal area

# COD-PS ADM1_EN -> COD-AB adm1_name where the two spell a state differently (none measured; the
# witness below raises on any difference, and a new spelling goes here with the reason).
NAME_SAME = {}


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    return re.sub(r"[^a-z]", "", s)


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for url, dst in ((COD_AB_URL, COD_AB), (COD_PS_URL, COD_PS)):
        if os.path.exists(dst) and os.path.getsize(dst) > 50_000:
            print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
            continue
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = r.read()
        if data[:2] != b"PK":                      # a zip and an xlsx both start PK
            raise SystemExit(f"{url} did not return a zip or xlsx (starts {data[:16]!r})")
        with open(dst + ".part", "wb") as fh:
            fh.write(data)
        os.replace(dst + ".part", dst)
        print(f"  got  {os.path.basename(dst)} ({len(data):,} bytes)")


def main():
    import geopandas as gpd
    from shapely.geometry import box

    if "--fetch" in sys.argv or not (os.path.exists(COD_AB) and os.path.exists(COD_PS)):
        fetch()

    with zipfile.ZipFile(COD_AB) as z:
        g = gpd.read_file(io.BytesIO(z.read("sdn_admin1.geojson")))
    print(f"COD-AB admin1: {len(g)} features")
    if len(g) != 19 or set(g["adm1_pcode"]) != {f"SD{i:02d}" for i in range(1, 20)}:
        raise SystemExit(f"COD-AB admin1 is not SD01-SD19: {sorted(g['adm1_pcode'])}")
    if g.loc[g["adm1_pcode"] == ABYEI, "adm1_name"].iloc[0] != "Abyei PCA":
        raise SystemExit("SD19 is no longer Abyei PCA")
    g = g[g["adm1_pcode"] != ABYEI].copy()
    print("  dropped SD19 Abyei PCA: no COD-PS projection, never sampled, status not settled")

    # ---- the Halaib triangle: Red Sea state north of 22 N ----
    minx, miny, maxx, maxy = g.total_bounds
    south = box(minx - 1, miny - 1, maxx + 1, HALAIB_LAT)
    rs = g["adm1_pcode"] == RED_SEA
    before = g.loc[rs].to_crs(METRIC_AREA).area.iloc[0] / 1e6
    g.loc[rs, "geometry"] = g.loc[rs].geometry.intersection(south)
    after = g.loc[rs].to_crs(METRIC_AREA).area.iloc[0] / 1e6
    cut = before - after
    print(f"  Halaib triangle: {cut:,.0f} km2 of Red Sea state north of {HALAIB_LAT} N clipped "
          f"({before:,.0f} -> {after:,.0f} km2)")
    if not HALAIB_KM2[0] <= cut <= HALAIB_KM2[1]:
        raise SystemExit(f"the Halaib clip removed {cut:,.0f} km2, outside the pinned {HALAIB_KM2}")
    north = g[g["adm1_pcode"] != RED_SEA].geometry.bounds["maxy"].max()
    print(f"  northernmost point of the other states: {north:.3f} N (the Wadi Halfa salient stays)")

    # ---- COD-PS 2022 ----
    ps = pd.read_excel(COD_PS, sheet_name="sdn_admpop_adm1_2022")
    ps0 = pd.read_excel(COD_PS, sheet_name="sdn_admpop_adm0_2022")
    meta = pd.read_excel(COD_PS, sheet_name="Metadata")
    for _i, r in meta.iterrows():
        print(f"  COD-PS metadata  {r['Item']}: {str(r['Metadata'])[:160]}")
    if len(ps) != 18 or int(ps0["T_TL"].iloc[0]) != TOTAL_2022 or int(ps["T_TL"].sum()) != TOTAL_2022:
        raise SystemExit(f"COD-PS 2022 is not 18 states summing to {TOTAL_2022:,}: "
                         f"{len(ps)} rows, {int(ps['T_TL'].sum()):,}")
    if (ps["year"] != 2022).any():
        raise SystemExit("COD-PS rows are not all year 2022")
    pop = dict(zip(ps["ADM1_PCODE"], ps["T_TL"].astype(int)))
    psname = dict(zip(ps["ADM1_PCODE"], ps["ADM1_EN"]))
    if set(pop) != set(g["adm1_pcode"]):
        raise SystemExit(f"p-codes differ: COD-PS only {sorted(set(pop) - set(g['adm1_pcode']))}, "
                         f"COD-AB only {sorted(set(g['adm1_pcode']) - set(pop))}")
    for _i, r in g.iterrows():
        a, b = fold(r["adm1_name"]), fold(NAME_SAME.get(psname[r["adm1_pcode"]], psname[r["adm1_pcode"]]))
        if a != b:
            raise SystemExit(f"{r['adm1_pcode']}: COD-AB {r['adm1_name']!r} against COD-PS "
                             f"{psname[r['adm1_pcode']]!r}")
    print("  18 p-codes joined; every English name agrees on both sides")

    g["unit"] = g["adm1_pcode"]
    g["name"] = g["adm1_name"]
    g["pop"] = g["unit"].map(pop).astype(int)
    area = g.to_crs(METRIC_AREA).area / 1e6
    print("\n  state, COD-PS 2022, area km2 (after the clip), people per km2:")
    for (_i, r), a in sorted(zip(g.iterrows(), area), key=lambda t: -t[0][1]["pop"]):
        print(f"      {r['unit']}  {r['name']:<16} {r['pop']:>10,}  {a:>9,.0f}  {r['pop'] / a:>7.1f}")

    os.makedirs(GEO, exist_ok=True)
    g[["unit", "name", "pop", "geometry"]].to_file(OUT, layer="states", driver="GPKG")
    lut = g[["unit", "name", "pop"]].rename(columns={"unit": "geo_id"})
    lut["unit"] = lut["geo_id"]
    lut.sort_values("geo_id").to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} and {LOOKUP} (18 states, {int(g['pop'].sum()):,} people)")


if __name__ == "__main__":
    main()
