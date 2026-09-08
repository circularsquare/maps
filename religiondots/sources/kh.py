"""Cambodia — NIS, General Population Census 2019, Tables 2.5.1 and 2.1.1.

Reads (or fetches) data/raw/kh/ and writes data/normalized/kh.csv.

**Four categories on 25 provinces for 15,552,211 people.** Cambodia is 97.1% Buddhist, and
the whole of the interesting map is in the remaining 3%: the Cham Muslim belt along the
Mekong and the Tonle Sap, and the highland provinces of the north-east where a fifth of the
population answers neither Buddhist nor Muslim nor Christian.

**THE SOURCE PUBLISHES PERCENTAGES AND NOT COUNTS, WHICH IS WHY TWO TABLES ARE READ.**
Table 2.5.1 gives the religious composition of each province to one decimal place and no
absolute figures anywhere. Table 2.1.1, nine pages earlier, gives each province's
population. The counts here are the product of the two, apportioned by largest remainder so
that a province's four categories sum to its published population exactly.

That is arithmetic on two published figures rather than an estimate (§14 rule 1 is about
inventing a magnitude the source does not publish; both factors are printed). What it costs
is stated plainly in `sources/kh.md` §3: **one decimal place on a percentage is ±0.05%, so
every cell carries a rounding band of ±0.0005 x the province population** — ±1,141 people in
Phnom Penh, ±21 in Kep. It matters most for the smallest category: `Other` is printed as
`0.0` in fifteen of the twenty-five provinces, and those are drawn as zero because zero is
what the table says (§3.5 — undercounting is marked, not filled).

**BOTH TABLES ARE READ IN FULL BECAUSE THE CROSS-TABLE IDENTITIES ARE THE ONLY REAL PARSE
CHECK.** Every identity available inside Table 2.5.1 — the four categories summing to 100 on
each row — survives a consistent column permutation, which is §12's Zimbabwe warning. What
does not survive it is the pair of checks that use the OTHER table:

  * `Male + Female == Total` on all 32 rows of Table 2.1.1, which catches a column landing
    in the wrong place there.
  * the province percentages, weighted by the province populations, reproducing the national
    percentages — for both census years, and separately for Urban and Rural. This pairs a
    number from each table and fails if either was read in the wrong order.

**THE ROW ORDER IS NIS's OWN PROVINCE-CODE ORDER**, which `sources/kh_geo.py` turns into a
free independent check on the boundary join: Banteay Meanchey is printed first and COD codes
it `KH01`, Kep is printed 23rd and coded `KH23`, Tbong Khmum printed last and coded `KH25`.
**The id minted here is `KH-01`..`KH-25` and deliberately NOT `KH01`..`KH25`** — §12's Benin
rule. It is a position in a printed table, the p-code is an official code, they happen to
agree, and nothing downstream should be able to assume that silently. Their agreement is
the check precisely because they are two different things.

**THE TABLE IS AN EXACT PARTITION.** NIS prints *"The sum of the four religion categories
amounts to 100 percent"* under it, and there is no `not stated` cell and no residual beyond
`Other`. So 100% of the census universe is drawn and there is no §3.5 gap — with the one
qualification that the universe itself excludes Cambodians working abroad, which both tables
say in a footnote and which `sources/kh.md` §2 sizes.

Usage:
    python sources/kh.py --fetch    one 26.6 MB PDF, seconds
    python sources/kh.py            normalise from data/raw/kh/
"""

import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "kh")
OUT = os.path.join(ROOT, "data", "normalized", "kh.csv")

SOURCE_ID = "kh_gpcc_2019"
YEAR = 2019
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# nis.gov.kh serves this with no wall and no redirect games — one GET, application/pdf.
# The WordPress site at nis.gov.kh/en/general-population-census-of-cambodia/ links a copy
# under /wp-content/uploads/2025/09/; this is the older static path and both serve.
PDF_URL = ("https://www.nis.gov.kh/nis/Census2019/"
           "Final%20General%20Population%20Census%202019-English.pdf")
PDF_NAME = "kh_gpcc2019_final_en.pdf"

POP_TITLE = re.compile(r"Table\s*2\.1\.1\.?\s*Distribution of total population by area,\s*"
                       r"region,\s*province and sex", re.I)
REL_TITLE = re.compile(r"Table\s*2\.5\.1\.?\s*Percentage distribution of population by "
                       r"religion,\s*area,\s*and province", re.I)

# Table 2.5.1's two panels. NIS spells the 2008 column `Christians` and the 2019 one
# `Christian`; both are asserted verbatim so a re-typeset stops the run.
CATEGORIES_2008 = ["Buddhist", "Muslims", "Christians", "Other"]
CATEGORIES_2019 = ["Buddhist", "Muslims", "Christian", "Other"]
CATEGORIES = CATEGORIES_2019          # the drawn panel
TOTAL_CAT = "Total"

# Table 2.1.1 carries three tiers above the provinces; Table 2.5.1 carries only the first.
AREAS = ["Total", "Urban", "Rural"]
REGIONS = ["Central Plain", "Tonle Sap", "Coastal and Sea", "Plateau and Mountains"]

# In the order NIS prints them, which is its own province-code order (01..25) — asserted
# against COD's `ADM1_PCODE` in sources/kh_geo.py rather than assumed here.
PROVINCES = [
    "Banteay Meanchey", "Battambang", "Kampong Cham", "Kampong Chhnang", "Kampong Speu",
    "Kampong Thom", "Kampot", "Kandal", "Koh Kong", "Kratie", "Mondul Kiri", "Phnom Penh",
    "Preah Vihear", "Prey Veng", "Pursat", "Ratanak Kiri", "Siem Reap", "Preah Sihanouk",
    "Stung Treng", "Svay Rieng", "Takeo", "Otdar Meanchey", "Kep", "Pailin", "Tbong Khmum",
]
EXPECTED_PROVINCES = 25
NATIONAL = 15_552_211          # Table 2.1.1's own Total row

# One decimal place on a percentage is +/-0.05%, so a province cell is +/-0.0005 x its
# population and the national total of a category is +/-0.0005 x 15,552,211 = +/-7,776 from
# each side of a comparison. Every band below is computed from that rather than chosen.
PCT_HALF_ULP = 0.05

INT = re.compile(r"^[\d,]+$")
PCT = re.compile(r"^\d+(?:\.\d+)?$")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, PDF_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 20_000_000:
        print("already have", dest)
        return
    print("GET", PDF_URL)
    r = requests.get(PDF_URL, timeout=900, stream=True,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    # §5a: HTTP 200 is not a download. Assert size AND type.
    with open(dest, "rb") as fh:
        magic = fh.read(5)
    if magic != b"%PDF-":
        raise SystemExit(f"{dest} is not a PDF -- starts {magic!r}, "
                         f"{os.path.getsize(dest):,} bytes")
    if os.path.getsize(dest) < 20_000_000:
        raise SystemExit(f"{dest} is {os.path.getsize(dest):,} bytes, expected ~26 MB -- "
                         "truncated")
    print(f"  {os.path.getsize(dest):,} bytes")


def _find_page(doc, rx, what):
    """The one page carrying a caption. Searching beats a pinned index and still asserts."""
    hits = [n for n in range(doc.page_count)
            if rx.search(" ".join(doc[n].get_text().split()))]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one page carrying {what}, found {hits} -- "
                         "NIS has re-typeset or re-paginated the report")
    return hits[0]


def _expect(lines, i, tokens, label):
    """Consume exactly `tokens` in order, skipping blanks. Returns the new index.

    Both tables put a header block between the caption and the first data row, and in
    Table 2.1.1 one of its cells is `Total` — which is also the first ROW label. So the
    header cannot be skipped by pattern and has to be walked, which is the stricter thing
    to do anyway: a changed column list stops the run here rather than shifting every
    figure by one.
    """
    for want in tokens:
        while i < len(lines) and not lines[i].strip():
            i += 1
        got = lines[i].strip() if i < len(lines) else "<eof>"
        if got.lower() != want.lower():
            raise SystemExit(f"{label}: expected the header cell {want!r} and read "
                             f"{got!r} -- NIS has changed the table's columns")
        i += 1
    return i


def _rows(lines, start, labels, n_values, cell_rx, kind, label):
    """Read `labels` in order, each followed by `n_values` value lines.

    The text layer of both tables emits the row label and then its figures one per line in
    column order, so this is a line read and not a geometry parse. The label is asserted
    before its figures are taken, which is what stops a row shifting silently.
    """
    i, out = start, {}
    for want in labels:
        while i < len(lines) and not lines[i].strip():
            i += 1
        got = lines[i].strip() if i < len(lines) else "<eof>"
        if got.lower() != want.lower():
            raise SystemExit(
                f"{label}: expected the {want!r} row and read {got!r} -- the row order of "
                f"{kind} has changed, and every figure after this point would be "
                "attributed to the wrong place")
        i += 1
        vals = []
        while len(vals) < n_values and i < len(lines):
            txt = lines[i].strip()
            i += 1
            if not txt:
                continue
            if not cell_rx.match(txt):
                raise SystemExit(f"{label} {want!r}: {txt!r} is not a figure -- the table "
                                 "has been re-typeset and the row would silently shift")
            vals.append(float(txt.replace(",", "")))
        if len(vals) != n_values:
            raise SystemExit(f"{label} {want!r}: read {len(vals)} of {n_values} figures")
        out[want] = vals
    return out, i


def read():
    import fitz

    p = os.path.join(RAW, PDF_NAME)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch first")
    doc = fitz.open(p)

    # ---- Table 2.1.1: Male / Female / Total, on areas + regions + provinces ----
    pno = _find_page(doc, POP_TITLE, "Table 2.1.1")
    lines = doc[pno].get_text().splitlines()
    head = next(j for j in range(len(lines))
                if POP_TITLE.search(" ".join(lines[j].split())))
    i = _expect(lines, head + 1,
                ["Area/Region/Province", "Male", "Female", "Total",
                 "(1)", "(2)", "(3)", "(4)"], "Table 2.1.1")
    pop, _ = _rows(lines, i, AREAS + REGIONS + PROVINCES, 3, INT,
                   "Table 2.1.1", "pop")
    print(f"  Table 2.1.1 on page {pno + 1}: "
          f"{len(AREAS) + len(REGIONS) + len(PROVINCES)} rows x 3 columns")

    # ---- Table 2.5.1: 2008 then 2019, four categories each, on areas + provinces ----
    pno = _find_page(doc, REL_TITLE, "Table 2.5.1")
    lines = doc[pno].get_text().splitlines()
    head = next(j for j in range(len(lines))
                if REL_TITLE.search(" ".join(lines[j].split())))

    # Assert the eight column labels in order before taking any figure. NIS spells the
    # 2008 column `Christians` and the 2019 one `Christian`, and both are required
    # verbatim: if the column list ever changes, taxonomy/kh2019.py must be revisited and
    # this is where that gets noticed.
    i = _expect(lines, head + 1,
                ["Area/Province", "2008", "2019"] + CATEGORIES_2008 + CATEGORIES_2019
                + [f"({n})" for n in range(1, 10)], "Table 2.5.1")

    rel, _ = _rows(lines, i, AREAS + PROVINCES, 8, PCT, "Table 2.5.1", "rel")
    print(f"  Table 2.5.1 on page {pno + 1}: "
          f"{len(AREAS) + len(PROVINCES)} rows x 8 columns")

    # ---- derive counts: percentage x population, largest remainder to the published total
    counts = {}
    for prov in PROVINCES:
        total = int(pop[prov][2])
        pct = dict(zip(CATEGORIES_2019, rel[prov][4:]))
        counts[prov] = _apportion(total, pct)

    rows = []
    for idx, prov in enumerate(PROVINCES, start=1):
        note = ("level=province; count = Table 2.5.1 percentage x Table 2.1.1 population, "
                "largest remainder; +/-0.0005 x population from the 1-dp rounding")
        for cat in CATEGORIES:
            rows.append({"geo_id": f"KH-{idx:02d}", "geo_level": "province",
                         "geo_name": prov, "source_category": cat,
                         "count": counts[prov][cat], "basis": BASIS, "year": YEAR,
                         "source_id": SOURCE_ID, "note": note})
        rows.append({"geo_id": f"KH-{idx:02d}", "geo_level": "province", "geo_name": prov,
                     "source_category": TOTAL_CAT, "count": int(pop[prov][2]),
                     "basis": BASIS, "year": YEAR, "source_id": SOURCE_ID,
                     "note": "level=province; universe total, not a religion category"})
    doc.close()
    return rows, pop, rel, counts


def _apportion(total, pct):
    """Split `total` across categories in proportion to their percentages.

    Largest remainder, so the four counts sum to the province's published population
    exactly — both factors are NIS's, and this only decides where the rounding residue of
    at most three people lands. A category printed as `0.0` gets zero, which is the
    published figure; it is NOT evidence that nobody is there (§3.5).
    """
    share = sum(pct.values())
    if share <= 0:
        raise SystemExit(f"a province's percentages sum to {share}")
    raw = {k: total * v / share for k, v in pct.items()}
    out = {k: int(v) for k, v in raw.items()}
    left = total - sum(out.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[:left]:
        out[k] += 1
    if sum(out.values()) != total:
        raise SystemExit(f"apportionment lost people: {sum(out.values())} vs {total}")
    return out


def check(rows, pop, rel, counts):
    ok = True

    units = {r["geo_id"] for r in rows}
    good = len(units) == EXPECTED_PROVINCES
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} province   {len(units):>4} units "
          f"(expected {EXPECTED_PROVINCES})")

    good = int(pop["Total"][2]) == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national universe {int(pop['Total'][2]):,} "
          f"(expected {NATIONAL:,})")

    # --- Table 2.1.1's own identities. Male+Female==Total is the one a consistent column
    # --- permutation cannot survive, which is why the sex columns are read at all (§12).
    bad = [k for k, v in pop.items() if int(v[0]) + int(v[1]) != int(v[2])]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} Male + Female == Total on all {len(pop)} rows "
          f"of Table 2.1.1 ({len(bad)} failures) {bad[:4]}")

    for label, group in (("Urban + Rural", ["Urban", "Rural"]),
                         ("the 4 regions", REGIONS),
                         ("the 25 provinces", PROVINCES)):
        s = sum(int(pop[k][2]) for k in group)
        good = s == NATIONAL
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {label} sum to the Total row "
              f"({s:,} vs {NATIONAL:,})")

    # --- Table 2.5.1's own identity, with the band computed from the rounding ---
    band = len(CATEGORIES) * PCT_HALF_ULP
    for year, cols in ((2008, slice(0, 4)), (2019, slice(4, 8))):
        bad = [(k, round(sum(v[cols]), 1)) for k, v in rel.items()
               if abs(sum(v[cols]) - 100.0) > band + 1e-9]
        ok &= not bad
        print(f"  {'OK ' if not bad else 'BAD'} the 4 categories sum to 100 +/-{band:.1f} "
              f"on all {len(rel)} rows, {year} ({len(bad)} failures) {bad[:4]}")
    print(f"      NIS prints \"The sum of the four religion categories amounts to 100 "
          f"percent\";\n      the band is 4 x {PCT_HALF_ULP} from the one decimal place, "
          "not a tolerance chosen to pass.")

    # --- THE CROSS-TABLE CHECKS. These pair a figure from each table, so they are the only
    # --- ones a consistent permutation of either table cannot survive.
    print("\n  cross-table: province percentages weighted by province populations, "
          "against\n  the national row of the OTHER table (§12 — the only check that "
          "crosses tables):")
    for year, cols in ((2019, 4),):
        for label, group in (("provinces", PROVINCES), ("urban+rural", ["Urban", "Rural"])):
            worst = 0.0
            for j, cat in enumerate(CATEGORIES_2019):
                got = sum(rel[k][cols + j] / 100.0 * int(pop[k][2]) for k in group)
                want = rel["Total"][cols + j] / 100.0 * NATIONAL
                worst = max(worst, abs(got - want))
            # each side is +/-0.0005 x its own population base, summed over the group
            tol = PCT_HALF_ULP / 100.0 * (sum(int(pop[k][2]) for k in group) + NATIONAL)
            good = worst <= tol
            ok &= good
            print(f"    {'OK ' if good else 'BAD'} {year} {label:<11} worst category off "
                  f"by {worst:>9,.0f} people (band +/-{tol:,.0f})")
    print("    The 2008 panel gets no such check and cannot: Table 2.1.1 publishes 2019 "
          "populations\n    only, and Cambodia grew 16.1% between the two censuses and "
          "unevenly by province, so\n    weighting 2008 percentages by 2019 populations "
          "is off by up to 27,000 people on a real\n    read. 2008 is parsed for its own "
          "sum-to-100 identity and is not drawn.")

    # --- what the derived counts came out as ---
    nat = {c: sum(counts[p][c] for p in PROVINCES) for c in CATEGORIES}
    drawn = sum(nat.values())
    good = drawn == NATIONAL
    ok &= good
    print(f"\n  {'OK ' if good else 'BAD'} the derived counts sum to {drawn:,} "
          f"(expected {NATIONAL:,}) — largest remainder, so this is exact by construction")

    print(f"\n  {len(rows):,} rows. Categories, national — derived count against NIS's own "
          "published national percentage:")
    for j, cat in enumerate(CATEGORIES):
        n = nat[cat]
        pub = rel["Total"][4 + j]
        print(f"    {n:>11,}  {100.0 * n / NATIONAL:6.2f}%  (NIS prints {pub:>4.1f}%)  {cat}")

    zeros = [(p, c) for p in PROVINCES for c in CATEGORIES if counts[p][c] == 0]
    print(f"\n  {len(zeros)} of {len(PROVINCES) * len(CATEGORIES)} cells are zero, every "
          "one of them a `0.0` in Table 2.5.1:")
    print("    " + ", ".join(f"{p}/{c}" for p, c in zeros[:8])
          + (" ..." if len(zeros) > 8 else ""))
    print("    Drawn as zero because zero is what NIS publishes. A `0.0` is anything under "
          "0.05%,\n    so the true figure could be a few hundred people — marked, not "
          "filled (§3.5).")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, pop, rel, counts = read()
    check(rows, pop, rel, counts)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
