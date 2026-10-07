"""Aruba, Fifth Population and Housing Census 2010 (CBS Aruba): language most spoken in the
household, by zone -> data/normalized/aw.csv.

    python sources/aw_census.py [--fetch]

THE TABLE. Census report "Fifth Population and Housing Census Aruba 2010", Table P-D.2,
*Population by language most spoken in the household by place of residence and sex* (report
pp.111-114, PDF pages 111-114): 8 regions, 55 zones (seven of them "<region> other", empty
or nearly so), nine columns: Papiamento, Spanish, Dutch, English, Chinese, Does not speak (yet),
Others, Not reported, All languages. One answer per household, applied to every member. Pages
111 and 113 hold regions 1-6 (four then five columns), 112 and 114 regions 7-8 and the island
total. The column order is read off the header words' positions (Dutch is left of English).

Why 2010 and not 2020: Census 2020 published language only as household combinations by region
("Papiamento only", "Papiamento and Spanish", ..., "Papiamento not one of the two"; Mapping
Census 2020 StoryMap, ArcGIS service Social_Atlas_Languages_by_Region), whose last group names no
language. sources/aw.md §1.

THE CHECKS, within rounding (each cell is rounded on its own; the largest deviations printed are
1, 3, 2 and 5): every row's Male + Female = Total for every column; the eight columns sum to All
languages; each region's zones sum to its TOTAL row; the zones sum to the printed island total; the island total is 101,484 (Table P-D.1, and the religion table religiondots drew); every
column within 10 of P-D.1's (by age) total; and each zone's All-languages total equals the 2010
population CBS carries on its zone polygons (ArcGIS Population_Tables_Census_2010_2020,
TotPop_10), which is the join check sources/aw_geo.py repeats.
"""
import csv
import re
import sys
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "aw"
OUT = HERE / "data" / "normalized" / "aw.csv"
PDF = RAW / "Fifth-Population-and-Housing-Census-Aruba.pdf"
# `id_` asks the Wayback Machine for the archived bytes (religiondots/sources/terr.py learned it);
# cbs.aw itself sits behind a Sucuri JavaScript challenge.
URL = ("https://web.archive.org/web/2016id_/http://cbs.aw/wp/wp-content/uploads/2012/07/"
       "Fifth-Population-and-Housing-Census-Aruba.pdf")
RD_COPY = (HERE.parent / "religiondots" / "data" / "raw" / "terr" /
           "Fifth-Population-and-Housing-Census-Aruba.pdf")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
SOURCE_ID = "aw_census_2010_tablePD2"

COLS_A = ["Papiamento", "Spanish", "Dutch", "English"]
COLS_B = ["Chinese", "Does not speak (yet)", "Others", "Not reported"]
# Table P-D.1 (by age), "All Ages" row, for the cross-check
PD1 = {"Papiamento": 69_354, "Spanish": 13_710, "Dutch": 6_110, "English": 7_129,
       "Chinese": 1_456, "Does not speak (yet)": 1_568, "Others": 1_725, "Not reported": 432}
TOTAL = 101_484
REGIONS = ["Noord/Tanki Leendert", "Oranjestad West", "Oranjestad East", "Paradera",
           "Santa Cruz", "Savaneta", "San Nicolas North", "San Nicolas South"]
# zone code (CBS GAC2, region digit + zone digit) for each printed zone row, in print order.
# The names differ from the ArcGIS layer's in spelling only (Pos Abao/Abou, Moko/Moco, Brasil/
# Brazil...); sources/aw_geo.py asserts the populations match code by code.
ZONES = {
    "Noord/Tanki Leendert": ["Palm Beach/Malmok", "Washington", "Alto Vista", "Moko/Tanki Flip",
                             "Tanki Leendert", "Noord other"],
    "Oranjestad West": ["Pos Abao/Cunucu Abao", "Eagle/Paardenbaai", "Madiki Kavel",
                        "Madiki/Rancho", "Paradijswijk/Santa Helena", "Socotoro/Rancho", "Ponton",
                        "Companashi/Solito"],
    "Oranjestad East": ["Nassaustraat", "Klip/Mon Plaisir", "Sividivi", "Seroe Blanco/Cumana",
                        "Dakota/Potrero", "Tarabana", "Sabana Blanco/Mahuma", "Simeon Antonio",
                        "Oranjestad East other"],
    "Paradera": ["Shiribana", "Paradera", "Ayo", "Piedra Plat", "Paradera other"],
    "Santa Cruz": ["Hooiberg", "Papilon", "Cashero", "Urataca", "Macuarima", "Balashi/Barcadera",
                   "Santa Cruz other"],
    "Savaneta": ["Pos Chiquito", "Jara/Seroe Alejandro", "De Bruynewijk", "Cura Cabai",
                 "Savaneta other"],
    "San Nicolas North": None,      # read from the page; checked against the ArcGIS names
    "San Nicolas South": None,
}
# The table's cells are rounded separately (the report's P-D.1 has 3,326 + 3,181 = 6,508 too), so
# sums are checked within a rounding tolerance and the largest deviation of each kind is printed.
DEV = {"m+f": 0, "columns": 0, "zones": 0, "regions": 0}
NUM = re.compile(r"^(-|\d{1,3}(,\d{3})*)$")


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    if RD_COPY.exists():                # the same bytes religiondots fetched; read, never written
        data = RD_COPY.read_bytes()
    else:
        req = urllib.request.Request(URL, headers={"User-Agent": UA})
        data = urllib.request.urlopen(req, timeout=600).read()
    if not data.startswith(b"%PDF") or b"%%EOF" not in data[-2048:]:
        raise SystemExit(f"aw: {URL} did not return a whole PDF ({len(data):,} bytes)")
    PDF.write_bytes(data)
    print(f"  {PDF.name}: {len(data):,} bytes")


def rows_of(page, ncols):
    """(label, [numbers]) for every printed line followed by numbers, in print order."""
    out, label, nums = [], None, []
    for line in page.get_text().split("\n"):
        toks = line.split()
        if not toks:
            continue
        if all(NUM.match(t) for t in toks):
            nums += [0 if t == "-" else int(t.replace(",", "")) for t in toks]
            continue
        if label is not None and nums:
            out.append((label, nums))
        label, nums = line.strip(), []
    if label is not None and nums:
        out.append((label, nums))
    for lab, ns in out:
        if len(ns) != ncols:
            raise SystemExit(f"aw: row {lab!r} has {len(ns)} numbers, expected {ncols}")
    return out


def read():
    import fitz
    doc = fitz.open(PDF)
    if doc.page_count != 290:
        raise SystemExit(f"aw: the census report has {doc.page_count} pages, expected 290")
    for i in (110, 111, 112, 113):
        if "Table P-D.2." not in doc[i].get_text():
            raise SystemExit(f"aw: PDF page {i + 1} is not Table P-D.2")
    # header order, by x position: Dutch left of English; Does not speak left of Others
    def xs(i):
        return {w[4]: w[0] for w in doc[i].get_text("words")}
    a, b = xs(110), xs(112)
    if not (a["Papiamento"] < a["Spanish"] < a["Dutch"] < a["English"]):
        raise SystemExit("aw: page 111's columns are not Papiamento, Spanish, Dutch, English")
    if not (b["Chinese"] < b["Does"] < b["Others"] < b["Not"] < b["All"]):
        raise SystemExit("aw: page 113's columns are not Chinese, Does not speak, Others, NR, All")

    left = rows_of(doc[110], 12) + rows_of(doc[111], 12)
    right = rows_of(doc[112], 15) + rows_of(doc[113], 15)
    if [l for l, _ in left] != [l for l, _ in right]:
        raise SystemExit("aw: the two halves of P-D.2 do not list the same rows")
    rows = []
    for (lab, na), (_, nb) in zip(left, right):
        cells = {}
        for j, c in enumerate(COLS_A + COLS_B + ["All"]):
            m, f, t = (na + nb)[3 * j: 3 * j + 3]
            if abs(m + f - t) > 1:
                raise SystemExit(f"aw: {lab} {c}: {m} + {f} != {t}")
            DEV["m+f"] = max(DEV["m+f"], abs(m + f - t))
            cells[c] = t
        d = abs(sum(cells[c] for c in COLS_A + COLS_B) - cells["All"])
        DEV["columns"] = max(DEV["columns"], d)
        if d > 4:
            raise SystemExit(f"aw: {lab}: languages sum to "
                             f"{sum(cells[c] for c in COLS_A + COLS_B)}, All says {cells['All']}")
        rows.append((lab, cells))
    return rows


def split(rows):
    """-> [(region, zone, code, cells)], checking zone sums against region TOTAL rows."""
    zones, region_i, cur, island = [], 0, [], None
    for lab, cells in rows:
        if lab == "TOTAL":
            reg = REGIONS[region_i]
            for c in cells:
                s = sum(z[1][c] for z in cur)
                DEV["zones"] = max(DEV["zones"], abs(s - cells[c]))
                if abs(s - cells[c]) > 1 + len(cur):
                    raise SystemExit(f"aw: {reg} zones sum to {s} {c}, TOTAL says {cells[c]}")
            want = ZONES[reg]
            if want is not None and [z[0] for z in cur] != want:
                raise SystemExit(f"aw: {reg} zones read as {[z[0] for z in cur]}")
            for k, (zn, zc) in enumerate(cur):
                zones.append((reg, zn, f"{region_i + 1}{k + 1}", zc))
            region_i, cur = region_i + 1, []
        elif lab == "Total population":
            island = cells
        elif lab == "Not reported":
            if cells["All"] != 0:
                raise SystemExit(f"aw: 'Miscellaneous / Not reported' holds {cells['All']} people")
        else:
            cur.append((lab, cells))
    if region_i != 8 or island is None or cur:
        raise SystemExit(f"aw: read {region_i} regions, island total {island is not None}")
    for c in island:
        s = sum(z[3][c] for z in zones)
        DEV["regions"] = max(DEV["regions"], abs(s - island[c]))
        if abs(s - island[c]) > 20:
            raise SystemExit(f"aw: zones sum to {s} {c}, the island row says {island[c]}")
    if island["All"] != TOTAL:
        raise SystemExit(f"aw: island total {island['All']}, expected {TOTAL}")
    for c, v in PD1.items():
        if abs(island[c] - v) > 10:
            raise SystemExit(f"aw: {c} {island[c]} in P-D.2 against {v} in P-D.1")
    return zones, island


def main():
    if "--fetch" in sys.argv or not PDF.exists():
        fetch()
    zones, island = split(read())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "geo_level", "geo_name", "source_category", "count", "tier",
                    "source_id", "year", "note"])
        for reg, zn, code, cells in zones:
            for c in COLS_A + COLS_B:
                if cells[c]:
                    w.writerow([f"AW{code}", "zone", zn, c, cells[c], "measured", SOURCE_ID, 2010,
                                reg])
                    n += 1
        for c in COLS_A + COLS_B:
            w.writerow(["AW", "national", "Aruba", c, island[c], "measured", SOURCE_ID, 2010,
                        "printed island total, Table P-D.2"])
    print(f"  {len(zones)} zones, {n} zone rows; island {island['All']:,}")
    for c in COLS_A + COLS_B:
        print(f"    {c:22s} {island[c]:>7,}  {100 * island[c] / island['All']:5.1f}%  "
              f"(P-D.1 {PD1[c]:,})")
    print(f"  largest rounding deviations: {DEV}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
