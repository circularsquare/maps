"""Seychelles: Population and Housing Census 2022, Table B3.1a, first main language at home.

Writes data/normalized/sc.csv: one row per (unit, language) for the 25 districts of Mahé and
Praslin and the 18 islands the table prints inside `La Digue & Inner Islands` and `Outer
Islands`. geo_level `district` for the first, `island` for the second; regions and the
national row are checked, not written.

THE TABLE. NBS, *Seychelles Population and Housing Census 2022* (the full report, 301 pages),
Table B3.1a, page 92 of the print (PDF page 116): population aged 3 and over by "main first
language spoken at home", by region, district and island. Ten columns: Creole, English, French,
Gujarati, Hindi, Tamil, Other (Specify), Do Not Know, Refusal, Missing. The form (Q.01.13,
report page 265) asked each person aged 3+ "What main language does [Name] speak most often at
home?" with those boxes.

THE CHECKS, all asserted:
  1. every row's ten cells sum to its printed total;
  2. the districts and islands sum to their printed region rows, the regions to the national row;
  3. Table B3.1b (second language, PDF page 117) prints the same row totals for every row;
  4. Table B4.1 (religion, all households, all ages, PDF page 118) is a second witness on the
     Outer Islands' oddity: there the table has no Creole speaker at all and 707 Gujarati, and
     B4.1 has exactly 707 Hindus on the same islands, island by island (sc.md says why this
     is drawn as printed);
  5. B4.1's all-ages totals are at least B3.1a's 3+ totals in every row.

Usage:
    python sources/sc_census.py --fetch   the report PDF (~10 MB) from nbs.gov.sc
    python sources/sc_census.py           re-parse data/raw/sc/
"""
import csv
import os
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "sc"
PDF = RAW / "phc2022_report.pdf"
OUT = ROOT / "data" / "normalized" / "sc.csv"
URL = ("https://www.nbs.gov.sc/downloads/1555-seychelles-population-and-housing-census-2022/"
       "download")
PDF_SIZE = 10_525_457                 # as religiondots fetched it, 2026-09-14
RD_COPY = ROOT.parent / "religiondots" / "data" / "raw" / "sc" / "phc2022_report.pdf"

PAGE_B31A, PAGE_B31B, PAGE_B41 = 116, 117, 118      # 1-based PDF pages
COLS = ["Creole", "English", "French", "Gujarati", "Hindi", "Tamil", "Other (Specify)",
        "Do Not Know", "Refusal", "Missing"]
NATIONAL = 98_952

# The table's rows in print order: (label, level, unit id or None). Region rows are checked
# only. Unit ids: ISO 3166-2:SC numbers for the 2010 districts (as religiondots), SC-PI for Ile
# Perseverance (our own id: COD-AB's pcode is SC1127PI, and ISO's code for it was not checked),
# SC-I-<name> for islands.
REGIONS = {
    "Central": [("English River", "SC-16"), ("Mont Buxton", "SC-17"), ("Saint Louis", "SC-22"),
                ("Bel Air", "SC-09"), ("Mont Fleuri", "SC-18"), ("Plaisance", "SC-19"),
                ("Roche Caiman", "SC-25"), ("Les Mamelles", "SC-24"),
                ("Ile Perseverance", "SC-PI")],
    "East-South": [("Cascade", "SC-11"), ("Pointe Larue", "SC-20"), ("Anse Aux Pins", "SC-01"),
                   ("Anse Royale", "SC-05"), ("Takamaka", "SC-23"), ("Au Cap", "SC-04")],
    "West": [("Baie Lazare", "SC-06"), ("Anse Boileau", "SC-02"), ("Grand Anse Mahé", "SC-13"),
             ("Port Glaud", "SC-21")],
    "North": [("Belombre", "SC-10"), ("Beau Vallon", "SC-08"), ("Glacis", "SC-12"),
              ("Anse Etoile", "SC-03")],
    "Praslin": [("Baie Ste Anne", "SC-07"), ("Grand Anse Praslin", "SC-14")],
    "La Digue & Inner Islands": [(n, "SC-I-" + n.upper().replace(" ", "")) for n in
                                 ["Bird", "Denis", "Fregate", "La Digue", "North", "Silhouette"]],
    "Outer Islands": [(n, "SC-I-" + n.upper().replace("-", "")) for n in
                      ["Aldabra", "Alphonse", "Assumption", "Coetivy", "Darros", "Desroches",
                       "Farquhar", "Marie-Louise", "Platte", "Poivre", "Providence", "Remire"]],
}
ISLAND_REGIONS = {"La Digue & Inner Islands", "Outer Islands"}


def fetch():
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    if PDF.exists() and PDF.stat().st_size == PDF_SIZE:
        print("  have", PDF)
        return
    try:
        r = requests.get(URL, timeout=600, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; "
                         "Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"})
        r.raise_for_status()
        if not r.content.startswith(b"%PDF"):
            raise ValueError(f"not a PDF ({r.headers.get('content-type')})")
        PDF.with_suffix(".part").write_bytes(r.content)
        os.replace(PDF.with_suffix(".part"), PDF)
        print(f"  GET {URL}: {len(r.content):,} bytes")
    except Exception as e:                                  # noqa: BLE001
        if not RD_COPY.exists():
            raise SystemExit(f"fetch failed ({e}) and religiondots has no copy")
        print(f"  fetch failed ({e}); copying religiondots' copy of the same release")
        shutil.copyfile(RD_COPY, PDF)
    if PDF.stat().st_size != PDF_SIZE:
        print(f"  !! {PDF.stat().st_size:,} bytes, religiondots had {PDF_SIZE:,}; the checks decide")


def num(s):
    s = s.strip()
    if s == "-":
        return 0
    return int(s.replace(",", ""))


def parse_page(page_no, ncols):
    """-> list of (label, total, [cells]) in print order, national row first with label ''."""
    import fitz
    doc = fitz.open(PDF)
    lines = [l.strip() for l in doc[page_no - 1].get_text().splitlines() if l.strip()]
    # the national row is the first run of ncols+1 numbers
    rows, i = [], 0
    is_num = lambda s: s == "-" or s.replace(",", "").isdigit()         # noqa: E731
    while i < len(lines):
        if is_num(lines[i]) and all(is_num(x) for x in lines[i:i + ncols + 1]) and not rows:
            rows.append(("", num(lines[i]), [num(x) for x in lines[i + 1:i + ncols + 1]]))
            i += ncols + 1
            continue
        if rows and not is_num(lines[i]) and i + ncols + 1 < len(lines) + 1 \
                and all(is_num(x) for x in lines[i + 1:i + ncols + 2]):
            rows.append((lines[i], num(lines[i + 1]), [num(x) for x in lines[i + 2:i + ncols + 2]]))
            i += ncols + 2
            continue
        i += 1
    return rows


def expected_labels():
    out = [""]
    for reg, members in REGIONS.items():
        out.append(reg)
        out.extend(n for n, _ in members)
    return out


def check_shape(rows, name):
    labels = [r[0] for r in rows]
    if labels != expected_labels():
        raise SystemExit(f"{name}: rows are\n{labels}\nexpected\n{expected_labels()}")
    for lab, tot, cells in rows:
        if sum(cells) != tot:
            raise SystemExit(f"{name} {lab!r}: cells sum to {sum(cells)}, printed {tot}")
    # members -> regions -> national
    it = iter(rows)
    nat = next(it)
    reg_sum = [0] * len(nat[2])
    k = 1
    for reg, members in REGIONS.items():
        r = rows[k]
        mem = rows[k + 1:k + 1 + len(members)]
        s = [sum(m[2][j] for m in mem) for j in range(len(r[2]))]
        if s != r[2]:
            raise SystemExit(f"{name} {reg}: members sum to {s}, printed {r[2]}")
        reg_sum = [a + b for a, b in zip(reg_sum, r[2])]
        k += 1 + len(members)
    if reg_sum != nat[2]:
        raise SystemExit(f"{name}: regions sum to {reg_sum}, national row {nat[2]}")
    print(f"  {name}: {len(rows)} rows; every row closes, members -> regions -> national")


def main():
    if "--fetch" in sys.argv:
        fetch()
    if not PDF.exists():
        raise SystemExit(f"missing {PDF}; run with --fetch")

    a = parse_page(PAGE_B31A, len(COLS))
    check_shape(a, "B3.1a first language")
    if a[0][1] != NATIONAL:
        raise SystemExit(f"national {a[0][1]}, expected {NATIONAL:,}")

    b = parse_page(PAGE_B31B, 9)
    check_shape(b, "B3.1b second language")
    diff = [(x[0], x[1], y[1]) for x, y in zip(a, b) if x[1] != y[1]]
    if diff:
        raise SystemExit(f"B3.1a and B3.1b disagree on row totals: {diff}")
    print("  B3.1b prints the same total on all", len(a), "rows")

    c = parse_page(PAGE_B41, 8)
    check_shape(c, "B4.1 religion")
    small = [(x[0], x[1], y[1]) for x, y in zip(a, c) if y[1] < x[1]]
    if small:
        raise SystemExit(f"B4.1 all-ages total under B3.1a's 3+ total: {small}")
    gi, hi = COLS.index("Gujarati"), 3                      # B4.1: Catholic Anglican Islam Hindu
    k = 1 + sum(1 + len(m) for r, m in REGIONS.items() if r != "Outer Islands")
    outer = list(zip(a[k:k + 1 + len(REGIONS["Outer Islands"])],
                     c[k:k + 1 + len(REGIONS["Outer Islands"])]))
    for x, y in outer:
        if x[2][gi] != y[2][hi] or x[2][0] != 0:
            raise SystemExit(f"Outer Islands witness broke at {x[0]}: Gujarati {x[2][gi]}, "
                             f"Hindu {y[2][hi]}, Creole {x[2][0]}")
    print(f"  Outer Islands: no Creole, Gujarati = B4.1's Hindus on every island "
          f"({outer[0][0][2][gi]} on both)")
    print(f"  all-ages B4.1 {c[0][1]:,} against 3+ B3.1a {a[0][1]:,}")

    nat = dict(zip(COLS, a[0][2]))
    named = sum(v for kk, v in nat.items() if kk not in ("Do Not Know", "Refusal", "Missing"))
    print("  national: " + ", ".join(f"{kk} {v:,} ({v / a[0][1]:.1%})" for kk, v in nat.items()))
    print(f"  named a language: {named:,} ({named / a[0][1]:.1%})")

    by_label = {r[0]: r for r in a}
    out_rows = []
    for reg, members in REGIONS.items():
        level = "island" if reg in ISLAND_REGIONS else "district"
        for name, uid in members:
            # `North` is both a region and an island; the island is the later row
            lab, tot, cells = a[expected_labels().index(name, expected_labels().index(reg))]
            for col, v in zip(COLS, cells):
                if v:
                    out_rows.append([uid, level, name, col, v, "measured",
                                     "sc_phc2022_tableB3.1a", 2022, reg])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".part")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "geo_level", "geo_name", "source_category", "count", "tier",
                    "source_id", "year", "note"])
        w.writerows(out_rows)
    os.replace(tmp, OUT)
    tot = sum(r[4] for r in out_rows)
    if tot != NATIONAL:
        raise SystemExit(f"written rows sum to {tot}, expected {NATIONAL}")
    print(f"wrote {OUT} ({len(out_rows)} rows, {tot:,} people in "
          f"{len({r[0] for r in out_rows})} units)")


if __name__ == "__main__":
    main()
