"""Timor-Leste, Population and Housing Census 2015: mother tongue by municipality
-> data/normalized/tl.csv.

    python sources/tl_census.py --fetch    download the workbooks and the 2010 witness
    python sources/tl_census.py            normalise from data/raw/tl/

THE DRAWN TABLE. Census 2015 Volume 2 (Population Distribution by Administrative Area), the
language workbook `4_2015-V2-Language.xls` on INETL's site, sheet 2.12: "Table 12 Population by
mother tongue, age, urban/rural location and Municipality". One answer per person, 32 Timorese
languages plus Portuguese, Indonesian, English, Malay, Chinese and Other, for the country, urban,
rural and the thirteen municipalities of 2015. Atauro was then an administrative post of Dili and
is inside Dili's column (it became a municipality in time for the 2022 census).

WHY 2015 AND NOT 2022. The 2022 census asked "What languages did <Name> learn as a child?" and
let each person give one or two (questionnaire item E55, main report p.192), so every 2022 figure
counts mentions: Ermera's chart prints 117,108 Tetun Prasa and 63,481 Mambai answers for 137,750
people. Nor is it published as a table: the only geography is a bar chart, an image, in each
municipality's `em Números 2022` volume, with no national table to reconcile Atauro against.
2015 is one answer per person, in a workbook, from the same office (AGENT_BRIEF §2: a single-
answer table beats a multi-answer one).

WHY NOT FINER. The 2015 suco tables (Volume 4) carry no language. The 2010 census printed mother
tongue per suco only as an unlabelled percentage bar chart, one 24-page `Sensus Fo Fila Fali`
PDF per suco (442 of them, mof.gov.tl, now only on the Wayback Machine); its Volume 2 has mother
tongue by district only (table 13), the same grain as here.

CHECKS, all asserted:
  1. per municipality, the 38 categories sum to the printed municipality total, and per category
     the thirteen municipalities sum to the national figure, and urban + rural = total;
  2. A SECOND TABLE OF THE SAME CENSUS: sheets 2.13a-2.13m, mother tongue by five-year age group,
     one per municipality, compiled separately; every (municipality, language) total there must
     equal table 12's cell, and sheet 2.13's national column the national one;
  3. the table's municipality totals against Table 1.a's population in private households
     (`1_2015-V2-Population-Household-Distribution.xls`), each within 0.5%;
  4. A SECOND CENSUS: 2010 Volume 2 table 13 (district by mother tongue, the 2010 Publication 2
     PDF from the Wayback Machine). Parsed off the PDF's text and checked internally (districts
     sum to the national figure), then: for every language over 2,000 speakers in both censuses,
     the municipality holding most of its speakers must be the same in 2010 and 2015. This is
     the join check: it would catch a swapped column (Manatuto and Manufahi have near-equal
     populations, so population shares cannot tell them apart; their languages can);
  5. Glottolog's point for each of seven regional languages falls inside the unit where 2015 has
     most of its speakers, on religiondots' p-coded hexes (read-only) - the p-code join itself.
"""
import os
import re
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "tl"
OUT = ROOT / "data" / "normalized" / "tl.csv"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

INETL = "https://inetl-ip.gov.tl/wp-content/uploads/"
FILES = {
    "tl_2015_v2_language.xls": INETL + "2023/03/4_2015-V2-Language.xls",
    "tl_2015_v2_population.xls": INETL + "2023/03/1_2015-V2-Population-Household-Distribution.xls",
    # the 2010 witness; mof.gov.tl is now a React app that answers every path with its shell
    "tl_2010_v2.pdf": "https://web.archive.org/web/20151224015626id_/https://www.mof.gov.tl/"
                      "wp-content/uploads/2011/06/Publication-2-English-Web.pdf",
}

# Table 12's municipality columns, in the order printed, -> COD-AB ADM1 p-code. The same codes
# religiondots' tl_hexes.gpkg carries (its sources/tl.py VOLUMES), where Atauro is split out of
# Dili as TL0604; countries/tl.py folds that back for 2015.
MUNIS = [("Aileu", "TL02"), ("Ainaro", "TL01"), ("Baucau", "TL03"), ("Bobonaro", "TL04"),
         ("Covalima", "TL05"), ("Dili", "TL06"), ("Ermera", "TL07"), ("Lautem", "TL09"),
         ("Liquiça", "TL08"), ("Manatuto", "TL11"), ("Manufahi", "TL10"),
         ("SAR1 of Oecusse", "TL12"), ("Viqueque", "TL13")]
NAME = {"SAR1 of Oecusse": "Oecusse"}
# sheet suffix of each municipality's age table (2.13a ... 2.13m), in the same order
AGE_SHEETS = dict(zip([m for m, _ in MUNIS], "abcdefghijklm"))
EXPECTED_CATS = 38
NATIONAL_TOTAL = 1_179_654

# 2010 table 13's district columns, in the order printed (Ainaro before Aileu)
D2010 = ["Ainaro", "Aileu", "Baucau", "Bobonaro", "Covalima", "Dili", "Ermera", "Liquiça",
         "Lautem", "Manufahi", "Manatuto", "SAR1 of Oecusse", "Viqueque"]
# 2010 spells two labels differently
SPELL_2010 = {"Tetum Prasa": "Tetun Prasa", "Tetum Terik": "Tetun Terik"}
# Dadu'a moved between censuses: 2010 has 1,656 in Dili (Atauro) and 1,400 in Manatuto, 2015 has
# 35 and 1,863, while Atauro's Rahesuk rose from 985 to 2,287. The Atauro answers drifted between
# the island's dialect names; the peak check leaves Dadu'a out for that reason.
PEAK_SKIP = {"Dadu’a"}

# Glottolog (data/raw/glottolog/languages.csv) points, lat lon, for languages whose point is well
# inside one municipality. Mambae's and Bunak's sit near municipal borders and are left out, and
# so is Naueti's (-8.707, 126.741), which lands a few km over the line in Lautem's Iliomar while
# its speakers are in Viqueque's Uatucarbau, 13,898 of 16,507 in 2015.
GLOTTO_POINTS = {"Fataluku": (-8.49464, 127.08), "Baikenu": (-9.32991, 124.256),
                 "Tokodede": (-8.6788, 125.283), "Galoli": (-8.61074, 125.959),
                 "Lakalei": (-8.86372, 125.716), "Waima’a": (-8.53416, 126.317),
                 "Idate": (-8.76258, 125.832)}


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        path = RAW / name
        if path.exists() and path.stat().st_size > 100_000:
            print("already have", path.name)
            continue
        print("GET", url)
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=300) as r:
            body = r.read()
        magic = {".xls": b"\xd0\xcf\x11\xe0", ".pdf": b"%PDF"}[path.suffix]
        if not body.startswith(magic):
            raise SystemExit(f"{name}: not a {path.suffix} file, first bytes {body[:60]!r}")
        path.write_bytes(body)
        print(f"  {len(body):,} bytes")


def _book(name):
    import xlrd
    path = RAW / name
    if not path.exists():
        raise SystemExit(f"missing {path} -- run with --fetch first")
    return xlrd.open_workbook(str(path))


def _num(v):
    return 0 if v in ("", "-") else int(round(float(v)))


def read_t12():
    sh = _book("tl_2015_v2_language.xls").sheet_by_name("2.12")
    head = [str(v).strip() for v in sh.row_values(3)]
    want = ["Total", "Urban", "Rural"] + [m for m, _ in MUNIS]
    if head[1:1 + len(want)] != want:
        raise SystemExit(f"table 12 header has changed: {head}")
    rows = {}
    for r in range(5, sh.nrows):
        vals = sh.row_values(r)
        label = str(vals[0]).strip()
        if not label or not any(str(v).strip() for v in vals[1:]):
            continue
        if label.startswith("1 Special"):
            break
        rows[label] = [_num(v) for v in vals[1:1 + len(want)]]
    total = rows.pop("TIMOR-LESTE")
    return total, rows


def read_age_totals():
    """{municipality: {language: total}} from sheets 2.13a-m, and the national sheet 2.13."""
    book = _book("tl_2015_v2_language.xls")
    out = {}
    for muni, suffix in [("Timor-Leste", "")] + list(AGE_SHEETS.items()):
        sh = book.sheet_by_name("2.13" + suffix)
        d = {}
        for r in range(5, sh.nrows):
            label = str(sh.cell_value(r, 0)).strip()
            if not label or label.startswith("1 Special"):
                continue
            v = sh.cell_value(r, 1)
            if str(v).strip() == "":
                continue
            d[label] = _num(v)
        out[muni] = d
    return out


def read_private_households():
    sh = _book("tl_2015_v2_population.xls").sheet_by_name("2.1.a")
    out = {}
    for r in range(sh.nrows):
        label = str(sh.cell_value(r, 0)).strip()
        if label:
            try:
                out[label.upper()] = _num(sh.cell_value(r, 5))   # private households, total
            except ValueError:
                pass
    return out


def read_2010():
    """Table 13 of the 2010 Volume 2, PDF pages 233-234, as {language: [17 values]}."""
    import fitz
    path = RAW / "tl_2010_v2.pdf"
    if not path.exists():
        raise SystemExit(f"missing {path} -- run with --fetch first")
    doc = fitz.open(str(path))
    rows = {}
    for p in (232, 233):
        text = doc[p].get_text()
        if "Table 13" not in text or "mother tongue" not in text:
            raise SystemExit(f"2010 PDF page {p + 1} is not table 13")
        body = text.split("(17)", 1)[1]
        label, vals = None, []
        for tok in [t.strip() for t in body.splitlines() if t.strip()]:
            if re.fullmatch(r"[\d,]+|-", tok):
                vals.append(0 if tok == "-" else int(tok.replace(",", "")))
            else:
                if label is not None:
                    rows[label] = vals
                label, vals = SPELL_2010.get(tok, tok), []
        rows[label] = vals
    rows = {k: v for k, v in rows.items() if not k.startswith("Timor-Leste Census")}
    bad = {k: len(v) for k, v in rows.items() if len(v) != 16}
    if bad:
        raise SystemExit(f"2010 table 13 rows without 16 values: {bad}")
    return rows


def check(total, rows):
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    n = len(MUNIS)
    say(len(rows) == EXPECTED_CATS, f"{len(rows)} categories (expected {EXPECTED_CATS})")
    say(total[0] == NATIONAL_TOTAL, f"national total {total[0]:,}")
    for j, (m, _) in enumerate(MUNIS):
        s = sum(v[3 + j] for v in rows.values())
        say(s == total[3 + j], f"{m}: categories sum to {s:,} against the printed {total[3 + j]:,}")
    bad = [k for k, v in rows.items() if sum(v[3:3 + n]) != v[0] or v[1] + v[2] != v[0]]
    say(not bad, f"every category: municipalities sum to the national figure, urban + rural = "
                 f"total {bad or ''}")
    say(sum(total[3:3 + n]) == total[0] and total[1] + total[2] == total[0],
        "the total row adds up across municipalities and urban/rural")

    # 2. the age tables, compiled separately
    ages = read_age_totals()
    miss = []
    for j, (m, _) in enumerate(MUNIS):
        a = ages[m]
        for k, v in rows.items():
            if a.get(k, 0) != v[3 + j]:
                miss.append((m, k, a.get(k), v[3 + j]))
        if a.get("Total") != total[3 + j]:
            miss.append((m, "Total", a.get("Total"), total[3 + j]))
    nat = ages["Timor-Leste"]
    miss += [("Timor-Leste", k, nat.get(k), v[0]) for k, v in rows.items() if nat.get(k, 0) != v[0]]
    say(not miss, f"tables 13 and 13.a-m agree with table 12 in all "
                  f"{(n + 1) * (len(rows) + 1) - 1} cells {miss[:6] or ''}")

    # 3. the private-household population
    ph = read_private_households()
    for j, (m, _) in enumerate(MUNIS):
        key = {"Lautem": "LAUTÉM", "Liquiça": "LIQUIÇA"}.get(m, m.upper())
        p = ph[key]
        say(abs(total[3 + j] / p - 1) < 0.005,
            f"{m}: table 12 {total[3 + j]:,} against {p:,} in private households "
            f"({total[3 + j] / p - 1:+.2%})")

    # 4. the 2010 census, district by district
    t10 = read_2010()
    tot10 = t10.pop("Total")
    bad10 = [k for k, v in t10.items() if sum(v[3:]) != v[0]]
    say(not bad10 and sum(tot10[3:]) == tot10[0],
        f"2010 table 13: {len(t10)} languages, districts sum to the national figure {bad10 or ''}")
    say(tot10[0] == 1_053_971, f"2010 national total {tot10[0]:,}")
    order15 = [m for m, _ in MUNIS]
    peaks = []
    for k, v in rows.items():
        if k in PEAK_SKIP or k not in t10 or v[0] < 2000 or t10[k][0] < 2000:
            continue
        p15 = order15[max(range(n), key=lambda j: v[3 + j])]
        p10 = D2010[max(range(n), key=lambda j: t10[k][3 + j])]
        peaks.append((k, p10, p15))
    swapped = [p for p in peaks if p[1] != p[2] and p[0] not in ("Tetun Prasa", "Portuguese",
                                                                 "Indonesian", "English")]
    say(len(peaks) >= 20 and not swapped,
        f"{len(peaks)} languages over 2,000 in both censuses peak in the same municipality "
        f"in 2010 and 2015 {swapped or ''}")

    # 5. Glottolog points on religiondots' p-coded hexes
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        sys.path.insert(0, str(ROOT))
        from rdlink import RD_GEO
        hexes = gpd.read_file(RD_GEO / "tl" / "tl_hexes.gpkg").to_crs(32751)
        hexes["unit"] = hexes["unit"].astype(str).replace({"TL0604": "TL06"})
        for lang, (lat, lon) in GLOTTO_POINTS.items():
            v = rows[lang]
            peak = MUNIS[max(range(n), key=lambda j: v[3 + j])][1]
            pt = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(32751).iloc[0]
            d = hexes.geometry.distance(pt)
            near = hexes.loc[d.idxmin(), "unit"]
            say(near == peak, f"Glottolog's {lang} point is in {near}, where 2015 has most of its "
                              f"speakers ({peak}, {max(v[3:3 + n]):,})")
    except ImportError as e:
        print(f"  -- Glottolog point check skipped: {e}")

    if not ok:
        raise SystemExit("checks failed")


def write(total, rows):
    import csv
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "geo_level", "geo_name", "source_category", "count"])
        for k, v in rows.items():
            w.writerow(["TL", "country", "Timor-Leste", k, v[0]])
        for j, (m, code) in enumerate(MUNIS):
            for k, v in rows.items():
                w.writerow([code, "municipality", NAME.get(m, m), k, v[3 + j]])
    os.replace(tmp, OUT)
    print(f"wrote {OUT} ({len(rows)} categories x {len(MUNIS)} municipalities)")


def main():
    if "--fetch" in sys.argv:
        fetch()
    total, rows = read_t12()
    print(f"table 12: {len(rows)} categories, {total[0]:,} people")
    check(total, rows)
    write(total, rows)


if __name__ == "__main__":
    main()
