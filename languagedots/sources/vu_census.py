"""Vanuatu, 2020 National Population and Housing Census: first language learnt to speak, by area
council -> data/normalized/vu.csv.

    python sources/vu_census.py [--fetch]

THE TABLE. *2020 National Population and Housing Census, Basic Tables Volume 1* (Vanuatu National
Statistics Office), Table 6.16, "Population 3 years and over living in private households with
first language learnt to speak by sex and region", PDF pages 188-191 (printed 178-181). Four
answers, English, French, Bislama and "Indigenous (Vernacular)", plus Not stated, each by sex,
for the nation, urban/rural, the six provinces, Port Vila, Luganville and the 64 rural area
councils. The ~110 indigenous languages are ONE category; the census does not name them.

WHO WAS ASKED. Not everyone aged 3+. The questionnaire (Volume 1's appendix, PDF page 374) asks
E9 "Can XYZ speak an indigenous (vernacular) language?" and then E10 "What is the first language
XYZ learned to SPEAK?" only where `speak_language!=3`, i.e. of people who can speak an indigenous
language. Volume 2 (Analytical Report, p65) says the same in words. So Table 6.16 covers 239,839
people against the 269,287 aged 3+ in private households that Table 6.17 (numeracy, same universe,
everyone asked) prints: 29,448 people, 10.9%, who speak no indigenous language were not asked,
and a quarter of the towns. This file carries Table 6.17's Total per unit as the
`Not asked (speaks no indigenous language)` category, computed as 6.17 Total - 6.16 Total, so
countries/vu.py can report it and, if Anita allows it, draw it.

THE NOT-STATED COLUMN is printed as a single column headed "Male" under "Not stated"; it is the
total (VANUATU: 4,955 + 1,942 + 34,723 + 198,216 + 3 = 239,839 exactly).

CHECKS, all asserted:
  1. each row's four answers and Not stated sum to its printed Total (6.16);
  2. Male + Female = Total for every answer on every row (6.16) and for 6.17's Total;
  3. Port Vila + Luganville = URBAN, URBAN + RURAL = VANUATU, the provinces sum to RURAL and each
     province's councils sum to it, column by column, in both tables;
  4. 66 drawable units whose names are exactly religiondots' vu_lookup.csv geo_ids (the Table 3.5
     names, same volume), both ways;
  5. 6.16 Total <= 6.17 Total <= Table 3.5 Total (all ages, religiondots' vu.csv) on all 66
     units, which is what catches 6.17's swapped town labels (see read());
  6. REPORTED, NOT ASSERTED: Volume 2 p65's shares of those asked (84.8% indigenous, 12.4%
     Bislama, 2.0% English, 0.8% French...). Every one is more vernacular than Table 6.16 (82.6%,
     14.5%, 2.1%, 0.8%), by more than rounding; sources/vu.md section 4.

TWO PRINTING FAULTS are corrected in read(): 6.16 has no Aneityum row (recovered as TAFEA minus
the other councils), and 6.17 swaps Port Vila's and Luganville's labels.

RAW FILES. The volume is religiondots' cached copy (../religiondots/data/raw/vu/), read-only; with
--fetch, and no copy there, it is downloaded to data/raw/vu/. vnso.gov.vu's certificate chain does
not verify from here (religiondots' sources/vu.py met the same), so the fetch skips verification
and checks the %PDF header and the %%EOF trailer instead.
"""
import csv
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RD_RAW = ROOT.parent / "religiondots" / "data" / "raw" / "vu"
RAW = ROOT / "data" / "raw" / "vu"
OUT = ROOT / "data" / "normalized" / "vu.csv"
RD_LOOKUP = ROOT.parent / "religiondots" / "data" / "geo" / "vu" / "vu_lookup.csv"

VOL1 = "vu_2020_basic_tables_vol1.pdf"
VOL1_URL = ("https://vnso.gov.vu/images/Public_Documents/Census_Surveys/Census/2020/Basic_Tables/"
            "2020NPHC_Volume_1_-_Version_2.pdf")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")

# Table 6.16: pages 188-189 carry Total, English, French (each Total/Male/Female); pages 190-191
# carry Bislama, Indigenous (each T/M/F) and Not stated (one column), the same rows repeated.
T616_A = ([188, 189], ["Total", "English", "French"])
T616_B = ([190, 191], ["Bislama", "Indigenous (Vernacular)"])
ANSWERS = ["English", "French", "Bislama", "Indigenous (Vernacular)", "Not stated"]
# Table 6.17 (numeracy): Total, Yes, No, Not stated, for Total / Male / Female.
T617_PAGES = [192, 193]
NOT_ASKED = "Not asked (speaks no indigenous language)"
MISSING_616 = "Aneityum"   # the one row Table 6.16 leaves out; see read()

PROVINCES = ["TORBA", "SANMA", "PENAMA", "MALAMPA", "SHEFA", "TAFEA"]
URBAN_UNITS = ["Port Vila", "Luganville"]
STRUCTURAL = ["VANUATU", "URBAN", "RURAL"] + PROVINCES
NUM = re.compile(r"^\d[\d,]*$")
TOL = 2   # VNSO rounds cells on their own in Table 3.5 (religiondots/sources/vu.py); allowed, counted
TOL_SUM = 3   # a sum of up to seventeen such cells
RD_NORM = ROOT.parent / "religiondots" / "data" / "normalized" / "vu.csv"

# Volume 2, p65: shares of those asked
VOL2 = {("VANUATU", "Indigenous (Vernacular)"): 84.8, ("VANUATU", "Bislama"): 12.4,
        ("VANUATU", "English"): 2.0, ("VANUATU", "French"): 0.8,
        ("URBAN", "Indigenous (Vernacular)"): 69.5, ("RURAL", "Indigenous (Vernacular)"): 89.3,
        ("TORBA", "Indigenous (Vernacular)"): 95.0, ("PENAMA", "Indigenous (Vernacular)"): 94.8,
        ("TAFEA", "Indigenous (Vernacular)"): 94.2, ("Port Vila", "Bislama"): 25.0,
        ("Luganville", "Bislama"): 30.7}


def vol1_path():
    for d in (RAW, RD_RAW):
        if (d / VOL1).exists():
            return d / VOL1
    return None


def fetch():
    if vol1_path():
        print("have", vol1_path())
        return
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    RAW.mkdir(parents=True, exist_ok=True)
    print("GET", VOL1_URL)
    r = requests.get(VOL1_URL, headers={"User-Agent": UA}, timeout=1800, verify=False)
    r.raise_for_status()
    if not r.content.startswith(b"%PDF") or b"%%EOF" not in r.content[-4096:]:
        raise SystemExit("vnso returned something that is not a whole PDF")
    (RAW / VOL1).write_bytes(r.content)


def _visual_rows(page, ytol=3.0):
    words = [w for w in page.get_text("words") if w[4].strip()]
    rows = []
    for x0, y0, x1, y1, txt, *_ in words:
        for r in rows:
            if abs(r[0] - y0) <= ytol:
                r[1].append((x0, x1, txt))
                break
        else:
            rows.append([y0, [(x0, x1, txt)]])
    for r in rows:
        r[1].sort(key=lambda t: t[0])
    rows.sort(key=lambda r: r[0])
    return rows


def _parse_page(doc, pno, ncols):
    """[(label, [values])] for one page. Numbers are right-aligned, so columns come from
    clustering right edges. The label column is found from the header: the first data column's
    right edge R is the first `Total` in the `Region` row, and a token ending more than 20pt left
    of R is label (row labels end 38pt or more short of it). Odd and even pages sit 39pt apart,
    so this is measured per page. A plain "is it numeric" test would put `Canal - Fanafo`'s
    hyphen (this table's zero) and `Central Pentecost 1`'s digit into the data."""
    page = doc[pno - 1]
    rows = _visual_rows(page)
    head = None
    for y, cells in rows:
        if cells and cells[0][2] == "Region":
            head = (y, cells)
    if head is None:
        raise SystemExit(f"page {pno}: no `Region` header row")
    head_y, head_cells = head
    totals = [x1 for x0, x1, t in head_cells if t == "Total"]
    if not totals:
        raise SystemExit(f"page {pno}: no `Total` in the header row")
    label_x1 = totals[0] - 20
    body = [(y, c) for y, c in rows if y > head_y + 3.0]
    data = []
    for y, cells in body:
        label = " ".join(t for x0, x1, t in cells if x1 < label_x1).strip()
        label = " ".join(re.sub(r"\s*-\s*", " - ", label).split())
        nums = [(x1, t) for x0, x1, t in cells if x1 >= label_x1 and (NUM.match(t) or t == "-")]
        if not label or not nums:
            continue
        data.append((label, nums))
    edges = sorted(x1 for _, nums in data for x1, _ in nums)
    clusters = [[edges[0]]]
    for e in edges[1:]:
        if e - clusters[-1][-1] <= 12:
            clusters[-1].append(e)
        else:
            clusters.append([e])
    centres = [sum(c) / len(c) for c in clusters]
    if len(centres) != ncols:
        raise SystemExit(f"page {pno}: {len(centres)} column clusters, expected {ncols}: "
                         f"{[round(c) for c in centres]}")
    out = []
    for label, nums in data:
        vals = [None] * ncols
        for x1, t in nums:
            j = min(range(ncols), key=lambda k: abs(centres[k] - x1))
            if vals[j] is not None:
                raise SystemExit(f"page {pno} row {label!r}: two values in column {j}")
            vals[j] = 0 if t == "-" else int(t.replace(",", ""))
        if any(v is None for v in vals):
            raise SystemExit(f"page {pno} row {label!r}: an empty column")
        out.append((label, vals))
    return out


def _block(doc, pages, ncols):
    rows = []
    for p in pages:
        rows += _parse_page(doc, p, ncols)
    return rows


def read():
    import fitz
    path = vol1_path()
    if path is None:
        raise SystemExit("no Volume 1 PDF -- run with --fetch")
    doc = fitz.open(path)
    a = _block(doc, T616_A[0], 9)
    b = _block(doc, T616_B[0], 7)
    c = _block(doc, T617_PAGES, 12)
    doc.close()
    if [lab for lab, _ in b] != [lab for lab, _ in a]:
        raise SystemExit("6.16's two column blocks list different rows")
    order = [lab for lab, _ in c]
    if len(set(order)) != len(order):
        raise SystemExit("a row label is printed twice")
    # ANEITYUM IS MISSING FROM TABLE 6.16. Every other table in the volume (3.5, 6.17) prints it
    # last under TAFEA; 6.16 stops at Futuna, yet its TAFEA row still includes it (TAFEA less
    # the ten printed councils is 1,261 people, against Aneityum's 1,338 aged 3+ in 6.17).
    # Its row is recovered as TAFEA minus the printed councils, column by column, and checked
    # below to be non-negative and to fit inside 6.17's universe.
    missing = [lab for lab in order if lab not in {x for x, _ in a}]
    if missing != [MISSING_616]:
        raise SystemExit(f"6.16 lacks {missing}; this file expects exactly [{MISSING_616!r}]")
    if [lab for lab in order if lab != MISSING_616] != [lab for lab, _ in a]:
        raise SystemExit("6.16 and 6.17 list their rows in different orders")
    t616, sexes, t617 = {}, {}, {}
    for (lab, va), (_, vb) in zip(a, b):
        # va: Total T/M/F, English T/M/F, French T/M/F; vb: Bislama T/M/F, Indig T/M/F, NS
        t616[lab] = {"Total": va[0], "English": va[3], "French": va[6], "Bislama": vb[0],
                     "Indigenous (Vernacular)": vb[3], "Not stated": vb[6]}
        sexes[lab] = {"Total": va[0:3], "English": va[3:6], "French": va[6:9],
                      "Bislama": vb[0:3], "Indigenous (Vernacular)": vb[3:6]}
    for lab, vc in c:
        t617[lab] = vc[0:12]
    # TABLE 6.17 SWAPS THE TWO TOWNS' LABELS. It prints Port Vila 15,978 and Luganville 44,856
    # aged 3+, but Port Vila is the capital: Table 3.5 has it at 48,461 people of all ages and
    # Luganville at 17,407, and 6.16 has 34,802 Port Vila residents ASKED, more than 15,978.
    # Swapped back here; check() then asserts 6.16 Total <= 6.17 Total <= Table 3.5 Total on
    # all 66 units, which the printed labels fail and the swapped ones pass.
    pv, lg = t617["Port Vila"], t617["Luganville"]
    if not (pv[0] < t616["Port Vila"]["Total"] and lg[0] > 40_000):
        raise SystemExit("Table 6.17's Port Vila/Luganville rows no longer look swapped; "
                         "remove the swap in read()")
    t617["Port Vila"], t617["Luganville"] = lg, pv
    prov = councils_of(order)["TAFEA"]
    others = [u for u in prov if u != MISSING_616]
    t616[MISSING_616] = {k: t616["TAFEA"][k] - sum(t616[u][k] for u in others)
                         for k in t616["TAFEA"]}
    sexes[MISSING_616] = {k: [t - sum(sexes[u][k][i] for u in others)
                              for i, t in enumerate(sexes["TAFEA"][k])]
                          for k in sexes["TAFEA"]}
    neg = {k: v for k, v in t616[MISSING_616].items() if v < 0}
    if neg:
        raise SystemExit(f"{MISSING_616} recovered by subtraction comes out negative: {neg}")
    return order, t616, sexes, t617


def councils_of(order):
    idx = {lab: i for i, lab in enumerate(order)}
    missing = [p for p in STRUCTURAL + URBAN_UNITS if p not in idx]
    if missing:
        raise SystemExit(f"missing structural rows {missing}")
    out = {}
    for p in PROVINCES:
        end = min([idx[q] for q in PROVINCES if idx[q] > idx[p]] + [len(order)])
        out[p] = order[idx[p] + 1:end]
    return out


def check(order, t616, sexes, t617):
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    councils = councils_of(order)
    units = [u for v in councils.values() for u in v] + URBAN_UNITS
    say(len(units) == 66 and sum(map(len, councils.values())) == 64,
        f"{len(units)} drawable units: 64 rural area councils + Port Vila, Luganville "
        f"({', '.join(f'{p} {len(councils[p])}' for p in PROVINCES)})")

    lut = {r["geo_id"] for r in csv.DictReader(open(RD_LOOKUP, encoding="utf-8"))}
    say(set(units) == lut, f"unit names equal religiondots' vu_lookup.csv geo_ids both ways "
        f"(only here: {sorted(set(units) - lut)}; only there: {sorted(lut - set(units))})")

    off = [(lab, t616[lab]["Total"], sum(t616[lab][a] for a in ANSWERS)) for lab in order]
    off = [x for x in off if x[1] != x[2]]
    say(all(abs(t - s) <= TOL for _, t, s in off),
        f"6.16: answers + Not stated = printed Total on {len(order) - len(off)}/{len(order)} rows"
        + (f"; off by <= {TOL} on {[(l, s - t) for l, t, s in off]}" if off else ""))

    # Aneityum's row is a difference of twelve rounded rows, so its slack is wider
    bad = [(lab, k, m + f - t) for lab in order for k, (t, m, f) in sexes[lab].items()
           if abs(m + f - t) > (TOL_SUM * 3 if lab == MISSING_616 else TOL)]
    nsex = sum(1 for lab in order for k, (t, m, f) in sexes[lab].items() if m + f != t)
    say(not bad, f"6.16: Male + Female = Total on every answer and row "
        f"({nsex} cells off by <= {TOL}, Aneityum's by <= {TOL_SUM * 3}; worse: {bad[:5]})")
    print(f"      Aneityum (recovered): {t616[MISSING_616]}")
    bad = [lab for lab in order
           if abs(t617[lab][4] + t617[lab][8] - t617[lab][0]) > TOL
           or abs(sum(t617[lab][1:4]) - t617[lab][0]) > TOL]
    say(not bad, f"6.17: Yes + No + Not stated = Total and Male + Female = Total ({bad[:5]})")

    def cmp(label, parts, whole):
        cols = ["Total"] + ANSWERS
        d616 = [sum(t616[p][c] for p in parts) - t616[whole][c] for c in cols]
        d617 = [sum(t617[p][i] for p in parts) - t617[whole][i] for i in range(12)]
        say(all(abs(x) <= TOL_SUM for x in d616 + d617),
            f"{label}: 6.16 max |diff| {max(map(abs, d616))}, 6.17 max |diff| {max(map(abs, d617))}")

    cmp("Port Vila + Luganville = URBAN", URBAN_UNITS, "URBAN")
    cmp("URBAN + RURAL = VANUATU", ["URBAN", "RURAL"], "VANUATU")
    cmp("the 6 provinces = RURAL", PROVINCES, "RURAL")
    for p in PROVINCES:
        cmp(f"{p}'s {len(councils[p])} councils = {p}"
            + (" (6.16 exact by construction: Aneityum is the residual)" if p == "TAFEA" else ""),
            councils[p], p)

    # Table 3.5 (religion, all ages, same volume) as religiondots normalised it, read-only
    t35 = {r["geo_id"]: int(r["count"]) for r in csv.DictReader(open(RD_NORM, encoding="utf-8"))
           if r["source_category"] == "Total"}
    over = [u for u in units if not (t616[u]["Total"] <= t617[u][0] <= t35[u])]
    say(not over, f"6.16 Total (asked) <= 6.17 Total (aged 3+) <= Table 3.5 Total (all ages) "
        f"on all {len(units)} units ({over})")
    r = sorted((t617[u][0] / t35[u], u) for u in units)
    print(f"      aged 3+ / all ages runs {r[0][0]:.3f} ({r[0][1]}) to {r[-1][0]:.3f} ({r[-1][1]})")

    # Volume 2's shares do NOT match Table 6.16, and not by rounding: every one is more
    # vernacular than the table (84.8% against 82.6% nationally, Port Vila Bislama 25.0% against
    # 30.2%). The table's own sums all hold, so this is reported, not asserted; Volume 2 does not
    # say what universe its shares are on. See sources/vu.md.
    print("\n  Volume 2 p65, shares of those asked (reported, not asserted):")
    for (lab, ans), want in VOL2.items():
        got = 100.0 * t616[lab][ans] / t616[lab]["Total"]
        print(f"    {lab:<10} {ans:<24} Vol 2 {want:5.1f}%  Table 6.16 {got:6.2f}%")

    nat, n17 = t616["VANUATU"], t617["VANUATU"][0]
    print(f"\n  national, aged 3+ in private households: {n17:,} (Table 6.17)")
    print(f"    asked (speak an indigenous language): {nat['Total']:,}  "
          f"{100 * nat['Total'] / n17:.1f}%")
    print(f"    not asked:                             {n17 - nat['Total']:,}  "
          f"{100 * (n17 - nat['Total']) / n17:.1f}%")
    for a in ANSWERS:
        print(f"      {a:<26} {nat[a]:>8,}  {100 * nat[a] / nat['Total']:5.1f}% of the asked")
    print("\n  not asked, by unit (largest share first):")
    sh = sorted(((t617[u][0] - t616[u]["Total"]) / t617[u][0], u) for u in units)[::-1]
    for s, u in sh[:8]:
        print(f"    {u:<22} {t617[u][0] - t616[u]['Total']:>6,} of {t617[u][0]:>6,}  {100 * s:5.1f}%")
    if not ok:
        raise SystemExit("reconciliation FAILED")
    return councils, units


def main():
    if "--fetch" in sys.argv:
        fetch()
    order, t616, sexes, t617 = read()
    councils, units = check(order, t616, sexes, t617)
    prov = {u: p for p, us in councils.items() for u in us}
    rows = []
    for u in units:
        note = f"province={prov[u]}" if u in prov else "urban municipality"
        for a in ANSWERS:
            rows.append([u, "area_council", u, a, t616[u][a], note])
        rows.append([u, "area_council", u, NOT_ASKED, t617[u][0] - t616[u]["Total"],
                     note + "; Table 6.17 Total (aged 3+) minus Table 6.16 Total"])
        rows.append([u, "area_council", u, "Total aged 3+", t617[u][0],
                     note + "; Table 6.17 Total, the universe, not a language"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "geo_level", "geo_name", "source_category", "count", "note"])
        w.writerows(rows)
    print(f"\nwrote {OUT} ({len(rows):,} rows)")


if __name__ == "__main__":
    main()
