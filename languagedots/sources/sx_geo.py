"""Sint Maarten placement layer: religiondots' Kontur hexes for SX (read only), cut along the
eight census regions and calibrated to their 2011 populations, with each region's 2011
birthplace shares -> data/geo/sx/sx_hexes.gpkg (unit = "SX" on every piece: the language counts
are national, sources/sx_census.py).

    python sources/sx_geo.py [--fetch]

SOURCES.
  * The eight regions: COD-AB for Sint Maarten (HDX cod-ab-sxm, sxm_admin1, VROMI boundaries
    valid 2024-01-22; SX1 Low Lands ... SX8 Upper Prince's Quarter), the same eight names the
    2011 census tabulates by.
  * Census 2011 report (STAT, reports.php?cat=CEN, download.php?type=rep&section=CEN&nummer=9),
    Table B-18, *Population by geographic region by country of birth* (PDF pages 48-51), and the
    region totals of Table B-21 (p.55), which B-22 (p.56) and B-23 (p.57) repeat.
  * religiondots/data/geo/sx/sx_hexes.gpkg: Kontur 2023 hexes for SX (93, read only).

B-18'S COLUMNS ARE PRINTED ONE PLACE OFF. Its header reads Colebay, Cul-de-sac, Little Bay, Low
Lands, Lower Princess Quarter, Philipsburg | Simpson Bay, Upper Princess Quarter, Not reported,
but the column totals in print order are 3,776 | 5,594 | 7,593 | 3,093 | 348 | 8,143 | 1,327 |
596 | 3,139, and B-21, B-22 and B-23 give Not reported 3,776, Colebay 5,594, Cul-de-sac 7,593,
Little Bay 3,093, Low Lands 348, Lower Princess Quarter 8,143, Philipsburg 1,327, Simpson Bay
596, Upper Princess Quarter 3,139. So the first column is Not reported and the rest follow in the
header's order. The script asserts the column order two ways: the Total population row against
B-21, and the Sint Maarten-born row against B-23's local-born (1,082 | 1,196 | 2,641 | 875 | 28 |
2,690 | 469 | 154 | 918). Low Lands' France- and USA-born (27 and 121 of 348) agree.

PLACEMENT. Each hex is intersected with the regions and each piece keeps its hex's people in
proportion to its share of the hex's area (hex area outside every region, sea or the French
side, is dropped); the pieces are then scaled so each region holds its 2011 population
(the 3,776 with no region recorded are left out: only shares within the island matter). Each
piece carries its region's share of people born in each of the BIRTH groups below, which
countries/sx.py multiplies into the weight of the matching languages. The counts stay national.
"""
import json
import math
import os
import random
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "sx"
REPORT = RAW / "census2011_report.pdf"
REPORT_URL = "https://stats.sintmaartengov.org/download.php?type=rep&section=CEN&nummer=9"
ADM_ZIP = RAW / "sxm_admin_boundaries.geojson.zip"
ADM_URL = ("https://data.humdata.org/dataset/b09efeaa-f9ee-44f2-a8b9-c99956be95ef/resource/"
           "aab6303e-1f46-4395-b49b-57028823e0c0/download/sxm_admin_boundaries.geojson.zip")
RD_HEX = HERE.parent / "religiondots" / "data" / "geo" / "sx" / "sx_hexes.gpkg"
OUT = HERE / "data" / "geo" / "sx" / "sx_hexes.gpkg"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
CRS_M = 32620           # UTM 20N

# B-18 column order as printed (see the docstring), mapped to COD-AB pcodes
COLS = ["NR", "SX3", "SX4", "SX5", "SX1", "SX7", "SX6", "SX2", "SX8"]
NAMES = {"SX1": "Low Lands", "SX2": "Simpson Bay", "SX3": "Cole Bay", "SX4": "Cul-de-Sac",
         "SX5": "Little Bay Area", "SX6": "Philipsburg", "SX7": "Lower Prince's Quarter",
         "SX8": "Upper Prince's Quarter"}
# Table B-21 (p.55) totals, repeated in B-22 and B-23
B21 = {"NR": 3_776, "SX3": 5_594, "SX4": 7_593, "SX5": 3_093, "SX1": 348, "SX7": 8_143,
       "SX6": 1_327, "SX2": 596, "SX8": 3_139}
# Table B-23 (p.57) local-born
B23_LOCAL = {"NR": 1_082, "SX3": 1_196, "SX4": 2_641, "SX5": 875, "SX1": 28, "SX7": 2_690,
             "SX6": 469, "SX2": 154, "SX8": 918}
TOTAL = 33_609
# birthplace groups (B-18 row labels) -> a column on the layer; countries/sx.py says which
# languages each one weights
BIRTH = {
    "b_hispanic": ["Dominican Republic", "Colombia", "Venezuela"],
    "b_haiti": ["Haiti"],
    "b_india": ["India"],
    "b_china": ["China"],
    "b_philippines": ["Phillipines"],
    "b_netherlands": ["Netherlands"],
    "b_abc": ["Aruba", "Bonaire", "Curacao"],
    "b_suriname": ["Surinam"],
    "b_french": ["France", "Guadeloupe"],
}
FIRST = {47: "Anguilla", 48: "Sint Maarten", 49: "Anguilla", 50: "Venezuela"}
NUM = re.compile(r"^[\d,\-\s]+$")


def get(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    data = urllib.request.urlopen(req, timeout=300).read()
    path.write_bytes(data)
    print(f"  {path.name}: {len(data):,} bytes")


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    get(REPORT_URL, REPORT)
    get(ADM_URL, ADM_ZIP)


def parse_page(text, first):
    lines = [l.strip() for l in text.split("\n")]
    i = lines.index(first)
    rows, label, nums = [], [], []
    for l in lines[i:]:
        if not l:
            continue
        if NUM.match(l):
            nums += [0 if t == "-" else int(t.replace(",", "")) for t in l.split()]
        else:
            if nums:
                rows.append((" ".join(label), nums))
                label, nums = [], []
            label.append(l)
    if nums:
        rows.append((" ".join(label), nums))
    return rows


def read_b18():
    import fitz
    doc = fitz.open(REPORT)
    if "Table B-18" not in doc[47].get_text():
        raise SystemExit("sx: Table B-18 is not on PDF page 48")
    left, right = {}, {}
    for pg, first in FIRST.items():
        for lab, nums in parse_page(doc[pg].get_text(), first):
            lab = lab.replace("’", "'")
            if pg < 49:
                if len(nums) != 18:
                    raise SystemExit(f"sx: B-18 p.{pg + 1} {lab}: {len(nums)} numbers")
                left[lab] = nums[2::3]
            else:
                if len(nums) != 10:
                    raise SystemExit(f"sx: B-18 p.{pg + 1} {lab}: {len(nums)} numbers")
                right[lab] = (nums[2:9:3], nums[9])
    if set(left) != set(right):
        raise SystemExit(f"sx: B-18 row labels differ: {set(left) ^ set(right)}")
    out = {}
    for lab in left:
        cols, total = left[lab] + right[lab][0], right[lab][1]
        if abs(sum(cols) - total) > 2:
            raise SystemExit(f"sx: B-18 {lab}: columns {sum(cols)} vs total {total}")
        out[lab] = dict(zip(COLS, cols))
    return out


def pear(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return num / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))


def main():
    import geopandas as gpd
    if "--fetch" in sys.argv or not REPORT.exists() or not ADM_ZIP.exists():
        fetch()
    b18 = read_b18()
    tot = b18.pop("Total population")
    if tot != B21:
        raise SystemExit(f"sx: B-18 column totals {tot} differ from B-21 {B21}")
    if b18["Sint Maarten"] != B23_LOCAL:
        raise SystemExit(f"sx: B-18 Sint Maarten-born {b18['Sint Maarten']} differ from B-23")
    if sum(tot.values()) != TOTAL:
        raise SystemExit("sx: B-18 does not sum to 33,609")
    print(f"  B-18: {len(b18)} birthplace rows; column totals equal B-21 and the Sint Maarten-born "
          f"row equals B-23's local-born, column by column")
    for cols in BIRTH.values():
        for c in cols:
            if c not in b18:
                raise SystemExit(f"sx: B-18 has no row {c!r}")

    with zipfile.ZipFile(ADM_ZIP) as z:
        adm = gpd.read_file(z.open("sxm_admin1.geojson"))
    if sorted(adm["adm1_pcode"]) != sorted(NAMES) or \
            dict(zip(adm["adm1_pcode"], adm["adm1_name"])) != NAMES:
        raise SystemExit(f"sx: COD-AB admin1 is not the eight expected regions: "
                         f"{dict(zip(adm['adm1_pcode'], adm['adm1_name']))}")
    print("  COD-AB admin1: the eight census regions, names as expected")

    h = gpd.read_file(RD_HEX)[["cellcode", "pop", "geometry"]].to_crs(CRS_M)
    zm = adm[["adm1_pcode", "geometry"]].rename(columns={"adm1_pcode": "region"}).to_crs(CRS_M)
    zm["geometry"] = zm.geometry.buffer(0)
    pieces = gpd.overlay(h, zm, how="intersection", keep_geom_type=True)
    pieces["area"] = pieces.geometry.area
    pieces = pieces[pieces["area"] > 1.0].copy()
    # a piece holds its own fraction of its hex's area (not of the hex's pieces): a hex mostly on
    # the French side or at sea would otherwise pour all its people into a sliver (SX:90, 84
    # people on 540 m^2). Regions are calibrated to the census below, so the dropped part costs
    # nothing but Kontur people the census did not count on this side.
    hex_area = h.set_index("cellcode").geometry.area
    pieces["kontur"] = pieces["pop"] * pieces["area"] / pieces["cellcode"].map(hex_area)
    lost = h["pop"].sum() - pieces["kontur"].sum()
    print(f"  Kontur: {len(h)} hexes, {h['pop'].sum():,.0f} people -> {len(pieces)} pieces in "
          f"{pieces['region'].nunique()} regions; {lost:,.0f} people on hex area outside every region")
    if pieces["region"].nunique() != 8:
        raise SystemExit("sx: the hexes do not cover the eight regions")

    per = pieces.groupby("region")["kontur"].sum()
    regs = sorted(NAMES)
    ratio = per.sum() / sum(B21[r] for r in regs)
    print(f"  Kontur / census 2011 by region (normalised, national {ratio:.2f}): " +
          ", ".join(f"{NAMES[r]} {per[r] / B21[r] / ratio:.2f}" for r in regs))
    lc, lk = [math.log(B21[r]) for r in regs], [math.log(per[r]) for r in regs]
    r = pear(lc, lk)
    rng = random.Random(0)
    shuf = sorted(abs(pear(lc, rng.sample(lk, len(lk)))) for _ in range(500))
    print(f"  log correlation r = {r:.3f}; shuffles: 95th percentile {shuf[474]:.3f}, "
          f"best {shuf[-1]:.3f}")
    if r <= shuf[474]:
        raise SystemExit("sx: the region join is not carrying information")

    pieces["pop"] = pieces["kontur"] * pieces["region"].map(lambda g: B21[g] / per[g])
    for col, labs in BIRTH.items():
        share_r = {g: sum(b18[l][g] for l in labs) / B21[g] for g in regs}
        pieces[col] = pieces["region"].map(share_r)
        top = max(regs, key=share_r.get)
        print(f"  {col}: {sum(b18[l][g] for l in labs for g in regs):,} born in the regions; "
              f"highest share {NAMES[top]} {share_r[top]:.1%}")
    foreign = {g: (B21[g] - b18["Sint Maarten"][g] - b18["Don't know"][g]
                   - b18["Not reported"][g]) / B21[g] for g in regs}
    pieces["b_foreign"] = pieces["region"].map(foreign)
    pieces["unit"] = "SX"
    cols = ["cellcode", "unit", "region", "pop", *BIRTH, "b_foreign", "geometry"]
    layer = gpd.GeoDataFrame(pieces[cols], geometry="geometry", crs=CRS_M).to_crs(4326)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    layer.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"  wrote {OUT.relative_to(HERE)} ({len(layer)} pieces, {layer['pop'].sum():,.0f} "
          f"people, the 2011 population of the eight regions)")


if __name__ == "__main__":
    main()
