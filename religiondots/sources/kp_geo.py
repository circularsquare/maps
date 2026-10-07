"""North Korea: the 11 provinces and special cities of COD-AB, with the 2008 census's people in each.

Writes data/geo/kp/kp_provinces.gpkg and data/geo/kp/kp_lookup.csv. `sources/kp.md` §4 is the
record.

  * **boundaries**: COD-AB `cod-ab-prk` v01 (valid from 2019-06-24), `prk_admin1.geojson`: 11 units
    KP01-KP11, Nampo separate from South Pyongan.
  * **population**: Central Bureau of Statistics, *DPR Korea 2008 Population Census, National
    Report* (Pyongyang 2009, UNFPA-supported), **Table 2**, "Population by Sex and by Urban-Rural, by
    City/District/County and Province", pp.18-22 (PDF pp.24-28). 209 cities, districts and counties
    in 10 provinces, 23,349,859 people. Table 1's national total is 24,052,231 and is footnoted as
    including "military camps"; Table 2 carries no such note, and the 702,372 difference is given by
    no province (`sources/kp.md` §4).

WHY NOT COD-PS. OCHA's COD-PS admin 1 for North Korea (World Food Programme, reference year 2008)
is this same table re-cut to the 2019 units, and it carries two errors at county level, both
checked against Table 2 here: Samchon county (South Hwanghae, 86,042) is missing, and Sindo county's
11,810 is listed on its own and also added into Ryongchon's (135,634 printed, 147,444 in COD-PS). So
COD-PS's South Hwanghae is 86,042 short and its North Pyongan 11,810 long. This script rebuilds
the 11 units from Table 2 itself.

THE RE-CUT, pinned. Table 2 has 10 provinces; COD-AB has 11 on later boundaries. Two moves, both the
ones COD-PS also makes, and each asserted against Table 2's own rows:

  * **Nampo** (KP11): Nampho City and the five counties Kangso, Onchon, Ryonggang, Taean and Chollima
    come out of South Phyongan (983,660 people).
  * **North Hwanghae** (KP08) takes Kangnam, Junghwa and Sangwon from Pyongyang (239,477), the
    counties moved in 2010.

Every other unit is Table 2's printed province total. The join is on folded English names
(Table 2's `Phyongan` is COD's `Pyongan`).

Usage:
    python sources/kp_geo.py --fetch    COD-AB geojson zip and the census report into data/raw/kp/
    python sources/kp_geo.py            rebuild from data/raw/kp/
"""

import io
import os
import re
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "6")

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "kp")
GEO = os.path.join(ROOT, "data", "geo", "kp")
OUT = os.path.join(GEO, "kp_provinces.gpkg")
LOOKUP = os.path.join(GEO, "kp_lookup.csv")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
COD_AB_URL = ("https://data.humdata.org/dataset/0cd9579f-ca75-4af4-88e1-9c5fa0bbb7ab/resource/"
              "3481e99e-bf95-45d1-a057-46c7b2d46726/download/prk_admin_boundaries.geojson.zip")
COD_AB = os.path.join(RAW, "prk_admin_boundaries.geojson.zip")
CENSUS_URL = ("https://unstats.un.org/unsd/demographic/sources/census/wphc/North_Korea/"
              "Final%20national%20census%20report.pdf")
CENSUS = os.path.join(RAW, "dprk_census_2008_national_report.pdf")
TABLE2_PAGES = range(23, 28)            # 0-based PDF pages 24-28

CIVIL_TOTAL = 23_349_859                # Table 2, "DPR Korea", all areas
NATIONAL_TOTAL = 24_052_231             # Table 1, all ages, including military camps

# Table 2's printed province totals, read 2026-10-03 and asserted.
PRINTED = {"Ryanggang": 719_269, "North Hamgyong": 2_327_362, "South Hamgyong": 3_066_013,
           "Kangwon": 1_477_582, "Jagang": 1_299_830, "North Phyongan": 2_728_662,
           "South Phyongan": 4_051_696, "North Hwanghae": 2_113_672, "South Hwanghae": 2_310_485,
           "Pyongyang": 3_255_288}

# (from province, to COD unit): {county: people in Table 2}
MOVES = {("South Phyongan", "Nampo"): {"Nampho City": 366_815, "Kangso": 191_356,
                                       "Onchon": 149_851, "Ryonggang": 58_930, "Taean": 77_219,
                                       "Chollima": 139_489},
         ("Pyongyang", "North Hwanghae"): {"Kangnam": 69_279, "Junghwa": 77_367,
                                           "Sangwon": 92_831}}

NUM = re.compile(r"^(\d{1,3}( \d{3})*|-)$")


def fold(s):
    return re.sub(r"[^a-z]", "", str(s).lower()).replace("phyong", "pyong")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for url, dst, magic in ((COD_AB_URL, COD_AB, b"PK"), (CENSUS_URL, CENSUS, b"%PDF")):
        if os.path.exists(dst) and os.path.getsize(dst) > 400_000:
            print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
            continue
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = r.read()
        if not data.startswith(magic):
            raise SystemExit(f"{url} did not return the expected file (starts {data[:8]!r})")
        with open(dst + ".part", "wb") as fh:
            fh.write(data)
        os.replace(dst + ".part", dst)
        print(f"  got  {os.path.basename(dst)} ({len(data):,} bytes)")


def table2():
    """[(province, unit, people)] from Table 2's text layer; 'TOTAL' rows are the printed totals."""
    import fitz

    doc = fitz.open(CENSUS)
    lines = []
    for i in TABLE2_PAGES:
        t = doc[i].get_text()
        if "Table 2." not in t:
            raise SystemExit(f"PDF page {i + 1} is not Table 2")
        lines += [ln.strip() for ln in t.splitlines() if ln.strip()]
    rows, prov, k = [], None, 0
    while k < len(lines):
        ln = lines[k]
        if (not NUM.match(ln) and k + 9 < len(lines)
                and all(NUM.match(x) for x in lines[k + 1:k + 10])):
            both = int(lines[k + 1].replace(" ", ""))
            if ln == "Total":
                prov = lines[k - 1]
                rows.append((prov, "TOTAL", both))
            elif ln == "DPR Korea":
                if both != CIVIL_TOTAL:
                    raise SystemExit(f"Table 2's DPR Korea row is {both:,}, not {CIVIL_TOTAL:,}")
            else:
                rows.append((prov, ln, both))
            k += 10
            continue
        k += 1
    t = pd.DataFrame(rows, columns=["prov", "unit", "pop"])
    for p, g in t.groupby("prov", sort=False):
        printed = int(g.loc[g["unit"] == "TOTAL", "pop"].iloc[0])
        if printed != PRINTED.get(p) or int(g.loc[g["unit"] != "TOTAL", "pop"].sum()) != printed:
            raise SystemExit(f"Table 2 {p}: printed {printed:,}, pinned {PRINTED.get(p)}, "
                             f"rows sum {int(g.loc[g['unit'] != 'TOTAL', 'pop'].sum()):,}")
    if set(t["prov"]) != set(PRINTED) or sum(PRINTED.values()) != CIVIL_TOTAL:
        raise SystemExit(f"Table 2 provinces: {sorted(set(t['prov']))}")
    print(f"  Table 2: {int((t['unit'] != 'TOTAL').sum())} cities, districts and counties in 10 "
          f"provinces; every printed total equals its rows; {CIVIL_TOTAL:,} people")
    return t


def main():
    import geopandas as gpd

    if "--fetch" in sys.argv or not (os.path.exists(COD_AB) and os.path.exists(CENSUS)):
        fetch()

    with zipfile.ZipFile(COD_AB) as z:
        g = gpd.read_file(io.BytesIO(z.read("prk_admin1.geojson")))
    want = {f"KP{i:02d}" for i in range(1, 12)}
    if len(g) != 11 or set(g["adm1_pcode"]) != want:
        raise SystemExit(f"COD-AB admin1 is not KP01-KP11: {sorted(g['adm1_pcode'])}")
    print(f"COD-AB admin1: {len(g)} units, version {g['version'].iloc[0]}, valid {g['valid_on'].iloc[0]}")

    t = table2()
    pop = {fold(p): v for p, v in PRINTED.items()}
    for (src, dst), counties in MOVES.items():
        rows = t[(t["prov"] == src) & t["unit"].isin(counties)]
        got = dict(zip(rows["unit"], rows["pop"]))
        if got != counties:
            raise SystemExit(f"Table 2 {src} rows for the move to {dst}: {got}, pinned {counties}")
        moved = sum(counties.values())
        pop[fold(src)] -= moved
        pop[fold(dst)] = pop.get(fold(dst), 0) + moved
        print(f"  moved {moved:,} from {src} to {dst} ({', '.join(counties)})")

    g["key"] = g["adm1_name"].map(fold)
    if set(g["key"]) != set(pop):
        raise SystemExit(f"COD names {sorted(g['key'])} against census {sorted(pop)}")
    g["unit"] = g["adm1_pcode"]
    g["name"] = g["adm1_name"]
    g["pop"] = g["key"].map(pop).astype(int)
    if int(g["pop"].sum()) != CIVIL_TOTAL:
        raise SystemExit(f"the 11 units sum to {int(g['pop'].sum()):,}")

    area = g.to_crs("ESRI:54034").area / 1e6
    print("\n  unit, 2008 census (Table 2, re-cut), area km2, people per km2:")
    for (_i, r), a in sorted(zip(g.iterrows(), area), key=lambda x: -x[0][1]["pop"]):
        print(f"      {r['unit']}  {r['name']:<15} {r['pop']:>10,}  {a:>9,.0f}  {r['pop'] / a:>8.1f}")
    print(f"  not in any province: {NATIONAL_TOTAL - CIVIL_TOTAL:,} (Table 1's total, which "
          "includes military camps, less Table 2's)")

    os.makedirs(GEO, exist_ok=True)
    g[["unit", "name", "pop", "geometry"]].to_file(OUT, layer="provinces", driver="GPKG")
    lut = g[["unit", "name", "pop"]].rename(columns={"unit": "geo_id"})
    lut["unit"] = lut["geo_id"]
    lut.sort_values("geo_id").to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} and {LOOKUP} (11 units, {int(g['pop'].sum()):,} people)")


if __name__ == "__main__":
    main()
