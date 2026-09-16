"""The Gambia — 2013 Population and Housing Census, religion by Local Government Area.

Reads (or fetches) data/raw/gm/census_2013_spatial_distribution_report.pdf and writes
data/normalized/gm.csv. `sources/gm.md` is the write-up; `sources/gm_geo.py` builds the eight LGAs
and the grid.

## THE TABLES

Gambia Bureau of Statistics, *The Gambia 2013 Population and Housing Census: Spatial Distribution
Report* (373 pp), **Annex H, "Population by five-year age group, sex, religion and Local Government
Area"**, Tables H.1-H.63 on PDF pp.119-181. One table per page: the nation and each of the eight
LGAs, by both sexes, male and female, and for the six LGAs that have rural parts, urban and rural
too. Every table is age group x five columns, in counts:

    Islam  Christianity  Traditional  Other  Not stated

Banjul and Kanifing are wholly urban and have no urban or rural tables. Nothing finer than LGA
crosses religion anywhere in the report.

## H.28 IS A MISPRINT, AND THE TABLES AROUND IT SETTLE IT

H.28, captioned *Kerewan-Both sexes*, repeats H.31 (*Kerewan-Urban-Both sexes*) in every cell:
50,188 people. Kerewan is 220,080 in Table B.1 and B.3. H.29 (male) plus H.30 (female) give
220,080, and so do H.31 (urban) plus H.34 (rural), cell by cell over every age row, and with that
figure the eight LGAs sum to H.1 in every cell of the table. So Kerewan is read as H.29 + H.30.

## THE OTHER QUIRKS, ALL PINNED IN check()

- H.18 is captioned *Brikama-Rural-Male* and holds the female figures (H.17 + H.18 = H.16).
- H.39 prints the age label `10-15` and H.54 `4-9`, each in a row the other tables call `10-14`
  and `5-9`; the cells close, so they are label slips.
- Seven urban or small tables print no `Not stated` age row, and five Kuntaur tables (male, urban
  and rural male) print no `Traditional` column. Both are read as zero, and the sex and
  urban-rural identities, which hold cell by cell, confirm it.
- Five captions spell `Tradition`, and H.6 breaks `Age group` over two lines.

## THE QUESTIONNAIRE

Form A (household) Part 2, column 7, *What is your Religion?*: **1 Islam, 2 Christianity,
3 Traditional, 4 Other**; Form B (group quarters and floating population) has the same four codes.
The enumerator's manual (para 8.38): record the religion professed, "no need to probe to ascertain
the authenticity of the claim"; "Traditional" is "the traditional African religion"; for any other
religion "record code 4 for other religions and specify, e.g. Hindu". There is no code for no
religion and none for no answer, so `Not stated` is a blank field, and anyone who said they had no
religion was written under code 4 or left blank. All three documents are on the IHSN catalogue,
study 6065 (downloads 74247, 74248, 74250).

Usage:
    python sources/gm.py --fetch    one 8.7 MB PDF from gbosdata.org
    python sources/gm.py            normalise from data/raw/gm/
"""

import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "gm")
OUT = os.path.join(ROOT, "data", "normalized", "gm.csv")
sys.path.insert(0, HERE)

from fetch_checks import FetchCheckError, check_body, digest   # noqa: E402  shared, not copied

SOURCE_ID = "gm_phc2013_spatial_distribution_annex_h"
YEAR = 2013
BASIS = "self_id"
COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

_PATH = "www.gbosdata.org/downloads-file/213-census-2013-spatial-distribution-report"
# The live file, then a Wayback capture with the same digest (four captures, 2024-03-13 to
# 2026-01-11, share it; the 2025-03-27 capture has another digest and is not used).
URLS = ["https://" + _PATH, "https://web.archive.org/web/20260111230241id_/https://" + _PATH]
PDF = os.path.join(RAW, "census_2013_spatial_distribution_report.pdf")
# gbosdata.org on 2026-09-15: SHA-1 in CDX base32 form, and the size.
PIN_DIGEST = "R4BTYNYMIW5GCJ426JRHKDZFVVWKS43P"
PIN_SIZE = 8_702_036
PAGES = 373

PAGE_PROSE = 15             # 0-based; PDF p.16, section 2.1.4
PAGE_B1 = 33                # PDF p.34, Table B.1
PAGE_B3 = 34                # PDF p.35, Table B.3
PAGE_H = range(118, 181)    # PDF pp.119-181, Tables H.1-H.63

CATS = ["Islam", "Christianity", "Traditional", "Other", "Not stated"]
AGES = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49",
        "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85+", "Not stated"]

LGAS = ["Banjul", "Kanifing", "Brikama", "Mansakonko", "Kerewan", "Kuntaur", "Janjanbureh",
        "Basse"]
# LGA -> (both, male, female) table numbers; urban and rural triples where the LGA has them.
BOTH = {"Banjul": 4, "Kanifing": 7, "Brikama": 10, "Mansakonko": 19, "Kerewan": 28,
        "Kuntaur": 37, "Janjanbureh": 46, "Basse": 55}
URBAN = {"Brikama": 13, "Mansakonko": 22, "Kerewan": 31, "Kuntaur": 40, "Janjanbureh": 49,
         "Basse": 58}
RURAL = {k: v + 3 for k, v in URBAN.items()}

# Transcribed from the Total rows, and asserted equal to what is parsed off the pages.
H1_TOTAL = (1_782_859, 69_638, 1_028, 2_686, 970, 1_857_181)
LGA_TOTAL = {                      # both sexes; Kerewan is H.29 + H.30
    "Banjul":      (29_423, 1_504, 12, 64, 51, 31_054),
    "Kanifing":    (345_546, 29_036, 309, 1_996, 247, 377_134),
    "Brikama":     (654_198, 33_168, 602, 329, 447, 688_744),
    "Mansakonko":  (80_434, 540, 9, 39, 20, 81_042),
    "Kerewan":     (217_191, 2_763, 32, 32, 62, 220_080),
    "Kuntaur":     (95_992, 673, 3, 7, 28, 96_703),
    "Janjanbureh": (124_254, 817, 25, 34, 74, 125_204),
    "Basse":       (235_821, 1_137, 36, 185, 41, 237_220),
}
H28_PRINTED = (49_223, 897, 9, 10, 49, 50_188)

# Table B.1 / B.3, 2013 population, and B.3's area in km2 (used by sources/gm_geo.py).
B1_2013 = {"Banjul": 31_054, "Kanifing": 377_134, "Brikama": 688_744, "Mansakonko": 81_042,
           "Kerewan": 220_080, "Kuntaur": 96_703, "Janjanbureh": 125_204, "Basse": 237_220}
B3_AREA = {"Banjul": 12.23, "Kanifing": 75.55, "Brikama": 1_764.25, "Mansakonko": 1_608.00,
           "Kerewan": 2_255.50, "Kuntaur": 1_466.50, "Janjanbureh": 1_427.75, "Basse": 2_069.50}
B3_AREA_TOTAL = 10_679.28
TOTAL = 1_857_181

# The text-layer and print quirks, pinned so a different file announces itself.
NO_NS_AGE_ROW = {33, 40, 41, 42, 49, 50, 51}
NO_TRADITIONAL_COLUMN = {38, 40, 41, 42, 44}
AGE_LABEL_SLIPS = {39: ("10-15", "10-14"), 54: ("4-9", "5-9")}
CAPTION_SLIPS = {18: "Brikama-Rural-Male"}        # holds the female figures

CAP = re.compile(r"Table H\.(\d+): Population by five-year age ?group ?and religion \((.+)\)")
NUM = re.compile(r"^\d{1,3}(,\d{3})*$")
SPACES = dict.fromkeys([0x00A0, 0x2007, 0x2008, 0x2009, 0x202F, 0x205F], " ")


def despace(s):
    return re.sub(r"\s+", " ", str(s).translate(SPACES)).strip()


def fetch():
    import urllib.request

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(PDF) and os.path.getsize(PDF) == PIN_SIZE:
        print("already have", PDF)
        return
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}
    for url in URLS:
        try:
            req = urllib.request.Request(url, headers=ua)
            with urllib.request.urlopen(req, timeout=600) as r:
                body = r.read()
        except Exception as e:                       # noqa: BLE001 - try the next copy
            print(f"  {url[:70]}...: {e}")
            continue
        try:
            check_body(body, "pdf", where=f"{url[:70]}...", pin_size=PIN_SIZE,
                       pin_digest=PIN_DIGEST)
        except FetchCheckError as e:
            print(f"  {e}")
            continue
        with open(PDF + ".part", "wb") as fh:
            fh.write(body)
        os.replace(PDF + ".part", PDF)
        print(f"wrote {PDF} ({len(body):,} bytes) from {url[:40]}")
        return
    raise SystemExit("no copy of the Spatial Distribution Report matched the pin")


def _lines(doc, pno):
    return [ln for ln in (despace(x) for x in doc.load_page(pno).get_text().splitlines()) if ln]


def read_annex_h(doc):
    """Every Annex H table: {n: {"label", "page", "rows": {age: [5 cats + total]}, "quirks"}}.

    Columns are named from each table's own header, since five tables leave `Traditional` out;
    a missing column or a missing `Not stated` age row reads as zeros and is recorded."""
    tables = {}
    for pno in PAGE_H:
        lines = _lines(doc, pno)
        cap = next((m for m in map(CAP.search, lines) if m), None)
        if cap is None:
            raise SystemExit(f"no Annex H caption on PDF p.{pno + 1}")
        n, label = int(cap.group(1)), cap.group(2)
        i = lines.index("Islam")
        lead = lines[i - 2:i] if lines[i - 1] == "group" else lines[i - 1:i]
        k = lines.index("Total", i)
        cols = " | ".join(lines[i:k + 1]).replace("Not | stated", "Not stated").split(" | ")
        cols = ["Traditional" if c == "Tradition" else c for c in cols]
        if " ".join(lead) != "Age group" or cols[-1] != "Total" \
                or any(c not in CATS for c in cols[:-1]):
            raise SystemExit(f"H.{n} header reads {lead} {cols}")
        quirks = {"columns": cols[:-1], "labels": {}, "no_ns_row": False}
        rest, j, rows = lines[k + 1:], 0, {}
        w = len(cols)
        for age in AGES + ["Total"]:
            if age == "Not stated" and rest[j] == "Total":
                rows[age] = [0] * 6
                quirks["no_ns_row"] = True
                continue
            if NUM.match(rest[j]):
                raise SystemExit(f"H.{n}: expected the {age!r} label, found {rest[j:j + 3]}")
            if rest[j] != age:
                quirks["labels"][age] = rest[j]
            cells = rest[j + 1:j + 1 + w]
            if not all(NUM.match(c) for c in cells):
                raise SystemExit(f"H.{n} row {age!r}: {cells}")
            got = dict(zip(cols, (int(c.replace(",", "")) for c in cells)))
            rows[age] = [got.get(c, 0) for c in CATS + ["Total"]]
            j += 1 + w
        tables[n] = {"label": label, "page": pno + 1, "rows": rows, "quirks": quirks}
    return tables


def _add(a, b):
    return {age: [x + y for x, y in zip(a[age], b[age])] for age in a}


def check(doc):
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    print("The Gambia — Census 2013 Spatial Distribution Report, Annex H\n")
    say(doc.page_count == PAGES, f"the report is {doc.page_count} pages (expected {PAGES})")
    with open(PDF, "rb") as fh:
        d = digest(fh.read())
    say(d == PIN_DIGEST, f"the PDF's digest is the pinned {PIN_DIGEST} ({d})")

    # 1. every table parses and closes on itself
    T = read_annex_h(doc)
    say(sorted(T) == list(range(1, 64)), f"{len(T)} tables, H.1-H.63, one per page")
    bad = [(n, a) for n, t in T.items() for a, r in t["rows"].items() if sum(r[:5]) != r[5]]
    say(not bad, f"every row's five columns sum to its Total ({len(bad)} do not)")
    bad = [(n, c) for n, t in T.items() for c in range(6)
           if sum(t["rows"][a][c] for a in AGES) != t["rows"]["Total"][c]]
    say(not bad, f"every column's age rows sum to its Total row ({len(bad)} do not)")

    # 2. the quirks are the recorded ones
    say({n for n, t in T.items() if t["quirks"]["no_ns_row"]} == NO_NS_AGE_ROW,
        f"tables with no `Not stated` age row: H.{sorted(NO_NS_AGE_ROW)}")
    say({n for n, t in T.items() if "Traditional" not in t["quirks"]["columns"]}
        == NO_TRADITIONAL_COLUMN, f"tables with no Traditional column: H.{sorted(NO_TRADITIONAL_COLUMN)}")
    slips = {n: next(iter(t["quirks"]["labels"].items())) for n, t in T.items()
             if t["quirks"]["labels"]}
    say(slips == {n: (want, got) for n, (got, want) in AGE_LABEL_SLIPS.items()},
        f"age-label slips {slips}")
    say(all(T[n]["label"] == lab for n, lab in CAPTION_SLIPS.items()),
        f"H.18's caption reads {T[18]['label']!r}")

    # 3. both sexes = male + female, everywhere but H.28
    for n in range(1, 64, 3):
        same = _add(T[n + 1]["rows"], T[n + 2]["rows"]) == T[n]["rows"]
        if n == 28:
            say(not same and T[28]["rows"] == T[31]["rows"],
                "H.28 (Kerewan both sexes) is NOT H.29 + H.30, and equals H.31 (Kerewan urban) "
                "in every cell: the pinned misprint")
        else:
            say(same, f"H.{n} {T[n]['label']} = H.{n + 1} + H.{n + 2}, cell by cell")
    say(tuple(T[28]["rows"]["Total"]) == H28_PRINTED, f"H.28 prints {H28_PRINTED}")

    # 4. LGA = urban + rural, male and female too; Kerewan both from its sexes
    for lga, u in URBAN.items():
        r = RURAL[lga]
        for off, what in ((0, "both sexes"), (1, "male"), (2, "female")):
            if lga == "Kerewan" and off == 0:
                continue
            say(_add(T[u + off]["rows"], T[r + off]["rows"]) == T[BOTH[lga] + off]["rows"],
                f"{lga:<12}{what:<11} = urban H.{u + off} + rural H.{r + off}, cell by cell")
    ker = _add(T[29]["rows"], T[30]["rows"])
    say(ker == _add(T[31]["rows"], T[34]["rows"]),
        "Kerewan: male + female (H.29 + H.30) = urban + rural (H.31 + H.34), cell by cell")
    say(_add(T[17]["rows"], T[18]["rows"]) == T[16]["rows"]
        and _add(T[15]["rows"], T[18]["rows"]) == T[12]["rows"],
        "H.18 is Brikama rural FEMALE: H.17 + H.18 = H.16 and H.15 + H.18 = H.12")

    lga_rows = {lga: (ker if lga == "Kerewan" else T[BOTH[lga]]["rows"]) for lga in LGAS}

    # 5. the eight LGAs sum to the nation, cell by cell, and match the transcription
    nat = lga_rows["Banjul"]
    for lga in LGAS[1:]:
        nat = _add(nat, lga_rows[lga])
    say(nat == T[1]["rows"], "the eight LGAs (Kerewan as H.29 + H.30) sum to H.1 in all "
        f"{len(AGES) + 1} rows x 6 columns")
    say(tuple(T[1]["rows"]["Total"]) == H1_TOTAL, f"H.1 Total row = {H1_TOTAL}")
    say(all(tuple(lga_rows[g]["Total"]) == LGA_TOTAL[g] for g in LGAS),
        "each LGA's Total row = the transcription")

    # 6. Annex B: the same populations, and the areas gm_geo.py tests against
    b1 = despace(" ".join(_lines(doc, PAGE_B1)))
    b3 = despace(" ".join(_lines(doc, PAGE_B3)))
    say(all(f"{v:,}" in b1 and f"{v:,}" in b3 for v in B1_2013.values())
        and f"{TOTAL:,}" in b1 and f"{TOTAL:,}" in b3,
        "all eight 2013 LGA populations and 1,857,181 are printed in Tables B.1 and B.3")
    say(all(LGA_TOTAL[g][5] == B1_2013[g] for g in LGAS) and sum(B1_2013.values()) == TOTAL,
        "each LGA's religion total = its Table B.1 population; they sum to 1,857,181")
    say(all(f"{a:,.2f}" in b3 for a in B3_AREA.values()) and f"{B3_AREA_TOTAL:,.2f}" in b3
        and abs(sum(B3_AREA.values()) - B3_AREA_TOTAL) < 0.005,
        f"Table B.3's eight areas are on its page and sum to its {B3_AREA_TOTAL:,.2f} km2")

    # 7. section 2.1.4's shares are of the people who stated a religion
    prose = despace(" ".join(_lines(doc, PAGE_PROSE)))
    stated = H1_TOTAL[5] - H1_TOTAL[4]
    sh = [round(100.0 * x / stated, 1) for x in H1_TOTAL[:4]]
    say(sh == [96.0, 3.8, 0.1, 0.1] and "(96.0 per cent)" in prose and "(3.8 per cent)" in prose
        and "each accounted for 0.1 per cent" in prose,
        f"p.16's 96.0 / 3.8 / 0.1 / 0.1 per cent are shares of the {stated:,} who stated one "
        f"({', '.join(f'{x:.3f}' for x in (100.0 * v / stated for v in H1_TOTAL[:4]))})")

    # 8. printed for the record: where the age-not-stated people sit
    print("\n  age `Not stated` row by LGA (Islam, Christianity, Traditional, Other, Not stated, Total):")
    for g in LGAS:
        print(f"    {g:<12}{lga_rows[g]['Not stated']}")

    if not ok:
        raise SystemExit("reconciliation FAILED")
    return lga_rows


def emit(lga_rows):
    rows = []
    for g in LGAS:
        src = ("Annex H, H.29 + H.30 (H.28 misprints H.31)" if g == "Kerewan"
               else f"Annex H, Table H.{BOTH[g]}")
        for c, n in zip(CATS, lga_rows[g]["Total"]):
            if n <= 0:
                continue
            rows.append({
                "geo_id": g, "geo_level": "lga", "geo_name": g,
                "source_category": c, "count": n, "basis": BASIS, "year": YEAR,
                "source_id": SOURCE_ID, "note": f"{src}, Total row, both sexes",
            })
    return rows


def main():
    import fitz

    if "--fetch" in sys.argv:
        fetch()
    if not os.path.exists(PDF):
        raise SystemExit(f"{PDF} missing — run: python sources/gm.py --fetch")
    doc = fitz.open(PDF)
    lga_rows = check(doc)
    rows = emit(lga_rows)

    total = sum(r["count"] for r in rows)
    print(f"\n  8 LGAs, {total:,} people, {total / 8:,.0f} each")
    by_cat = {}
    for r in rows:
        by_cat[r["source_category"]] = by_cat.get(r["source_category"], 0) + r["count"]
    for c in CATS:
        print(f"    {by_cat.get(c, 0):>11,}  {100.0 * by_cat.get(c, 0) / total:6.3f}%  {c}")
    print(f"\n  {'LGA':<12}{'people':>9}  {'Christian':>9}  {'share':>6}  of all Christians")
    for g in LGAS:
        t = LGA_TOTAL[g]
        print(f"  {g:<12}{t[5]:>9,}  {t[1]:>9,}  {100.0 * t[1] / t[5]:5.2f}%  "
              f"{100.0 * t[1] / H1_TOTAL[1]:5.1f}%")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".part", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(OUT + ".part", OUT)
    print("\nwrote", OUT, f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
