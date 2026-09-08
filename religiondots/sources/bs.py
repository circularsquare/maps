"""The Bahamas — BNSI, 2022 Census of Population and Housing, All-Island Report, Table 12.x.

Reads (or fetches) data/raw/bs/ and writes data/normalized/bs.csv.

**Twenty-four named categories on 18 islands, for 398,165 people** — the whole census, with
no universe caveat at all. Table 12.1 to 12.18 is *Total Population by Sex, Age-Group and
Religion* for each island in turn, and the eighteen tables sum to the published national
count exactly.

**THE REPORT IS NOT ON THE PUBLISHER'S OWN PUBLICATIONS PAGE AND THAT IS WHY THE COUNTRY WAS
CALLED OPEN.** `sources.md` §11t swept Latin America and the Caribbean and left the Bahamas
as a lead, because BNSI's *first release* — the one `bnsistats.gov.bs/publications` lists,
and the one a search finds — carries religion **for All Bahamas only**, by age and sex. Its
own preface says so: *"The other Census topics (such as Religion, Marital Status, etc.) are
reported by Age group and Sex and are for All Bahamas."* The All-Island Report is a separate
579-page document published in June 2026, absent from that listing, and found on the CARICOM
regional mirror (§11v). It IS served by BNSI's own CDN, which is what `PDF_URL` points at —
the mirror is how the file becomes findable, not where it has to be cited from.

**THE CATEGORY LIST IS NOT FIXED ACROSS ISLANDS, WHICH IS THE ONE STRUCTURAL SURPRISE.**
Only New Providence prints all 24 named bodies. Every other island folds its smallest
answers into a single `OTHER RELIGION*` cell — and a starred footnote under each table names
exactly which bodies went in. So the parser cannot walk a fixed row sequence the way
`sources/tt.py` does; it validates each label against a known set instead, and reads the
footnotes so the collapsing is recorded rather than inferred.

    Ragged Island   4 categories   `Other Religion` includes Assemblies of God, Church of
                                   God, Church of God of Prophecy, Pentecostal, Roman
                                   Catholic, Other Christian Denomination and None

**IT COSTS ALMOST NOTHING AND IT IS NOT NOTHING.** The residual is **372 people, 0.093% of
the country**, across 17 islands. But on **two islands — Mayaguana and Ragged Island, 259
people between them — the footnote says `None` was folded in**, so those two islands report
no irreligion at all and their handful of non-religious people are drawn as `other.bs`.
Recorded in taxonomy/bs2022.py rather than corrected (§14.4).

**FOUR CHECKS, AND THE OUTER TWO ARE CROSS-DOCUMENT.**

  1. *Within a cell.* The seven age bands sum to the cell's own TOTAL column, exactly, on
     every cell read — which is a check on every number in the table rather than on its
     margins.
  2. *Within a cell again.* Male + Female == TOTAL, on all three sex rows.
  3. *Within an island.* The categories sum to the island's own TOTAL row.
  4. *Across the two reports.* The 18 islands sum to the **first release's** Table 6.0, the
     All-Bahamas religion table, category by category — and the shortfall on each category
     is exactly what the island footnotes say was folded into the residuals. That is an
     independent tabulation in a separate publication agreeing to the person.

**TWO SPELLINGS OF ONE CATEGORY, AND ONE OF THEM IS THE `tt.py` TRAP AGAIN.** `JEHOVAH'S
WITNESS` is printed with a curly apostrophe on five islands and a straight one on three.
`OTHER RELIGION` carries its footnote star on sixteen islands and not on Grand Bahama. Both
are folded before matching; nothing is matched on the raw string (§12).

Usage:
    python sources/bs.py --fetch    two PDFs, ~19 MB, seconds
    python sources/bs.py            normalise from data/raw/bs/
"""

import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bs")
OUT = os.path.join(ROOT, "data", "normalized", "bs.csv")

SOURCE_ID = "bs_phc_2022"
YEAR = 2022
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

# BNSI's own CDN. The CARICOM mirror carries a byte-identical copy at
# statistics.caricom.org/wp-content/uploads/2026/08/<same name>, which is where it was
# found; the publisher's copy is the citable one (§11t's rule, from Trinidad).
PDF_URL = ("https://cdn.bahamas.gov.bs/tenant/tenantbnsi/documents/"
           "2022-Census-of-Population-and-Housing-All-Island-Report-1-20260626112021.pdf")
PDF_NAME = "bs_all_island_report_2022.pdf"
PDF_PAGES = 579

# The first release, for the national cross-check ONLY. Nothing is drawn from it.
NAT_URL = ("https://cdn.bahamas.gov.bs/tenant/tenantbnsi/documents/"
           "2022-Census-Report-1st-Release-12-February-2025-FINAL-20250526040559.pdf")
NAT_NAME = "bs_first_release_2022.pdf"
NAT_PAGES = 163
NAT_TABLE_PAGES = range(91, 95)          # 1-based; Table 6.0 and its three continuations

# **THE TABLE NUMBER IS THE CENSUS'S OWN ISLAND CODE.** The 2022 questionnaire (first
# release, p93) pre-fills `Name of Island` from a numbered list, and Tables 12.1-12.18 come
# in exactly that order. So the id below is BNSI's, not one invented here.
ISLANDS = {
    1:  "New Providence",
    2:  "Grand Bahama",
    3:  "Abaco",
    4:  "Acklins",
    5:  "Andros",
    6:  "Berry Islands",
    7:  "Bimini",
    8:  "Cat Island",
    9:  "Crooked Island",
    10: "Eleuthera",
    11: "Exuma",
    12: "Harbour Island",
    13: "Inagua",
    14: "Long Island",
    15: "Mayaguana",
    16: "Ragged Island",
    17: "San Salvador and Rum Cay",
    18: "Spanish Wells",
}

# Every named body the census offers, in Table 12.1's printed order — New Providence is the
# only island that prints the whole list. Membership of this set is what the parser
# validates each label against, since the SEQUENCE is not stable across islands.
CATEGORIES = [
    "ANGLICAN",
    "ASSEMBLIES OF GOD",
    "BAPTIST",
    "BRETHREN",
    "CHURCH OF GOD AND CHURCH OF GOD OF PROPHECY",
    "GREEK ORTHODOX",
    "JEHOVAH'S WITNESS",
    "LUTHERAN",
    "METHODIST",
    "PENTECOSTAL",
    "PRESBYTERIAN",
    "ROMAN CATHOLIC",
    "SEVENTH DAY ADVENTIST",
    "MORMON",
    "OTHER CHRISTIAN DENOMINATION (INCLUDING NON-DENOMINATIONAL GROUPS)",
    "BAHAI FAITH",
    "HINDU",
    "ISLAM (MUSLIM)",
    "JUDAISM (JEWISH)",
    "RASTAFARIAN",
    "OTHER NON-CHRISTIAN RELIGION",
    "NONE",
    "AFRICAN METHODIST EPISCOPAL (AME)",
    "ATHEIST",
    "NOT STATED",
]
RESIDUAL = "OTHER RELIGION"     # printed `OTHER RELIGION*` on 16 of the 17 islands with one
TOTAL_CAT = "TOTAL"             # the island's own population row

NATIONAL_TOTAL = 398_165        # first release Table 6.0, All Bahamas

N_COLS = 8                      # TOTAL + seven age bands
SEX_ROWS = ("TOTAL", "Male", "Female")

FIG = re.compile(r"^-?[\d,]+$")
FOOTNOTE = re.compile(r'\*\s*["“]?Other Religion["”]?\s*includes the following:'
                      r'\s*(.+?)\s*\.\s*$', re.S | re.I)


# ---------------------------------------------------------------- fetch

def _get(url, name, pages, min_bytes):
    import requests

    dest = os.path.join(RAW, name)
    if os.path.exists(dest) and os.path.getsize(dest) > min_bytes:
        print("already have", dest)
        return
    print("GET", url)
    r = requests.get(url, timeout=1800, stream=True, headers={"User-Agent": UA})
    r.raise_for_status()
    with open(dest + ".part", "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    os.replace(dest + ".part", dest)

    # §5a and tt.py's sharpening of it: a complete download is not an intact file. BNSI
    # serves these as `application/octet-stream`, so the content-type says nothing either.
    with open(dest, "rb") as fh:
        head = fh.read(5)
        fh.seek(max(0, os.path.getsize(dest) - 4096))
        tail = fh.read()
    if head != b"%PDF-":
        raise SystemExit(f"{dest} is not a PDF -- starts {head!r}")
    if b"%%EOF" not in tail:
        raise SystemExit(f"{dest} has no %%EOF -- it is TRUNCATED at source")
    import fitz
    doc = fitz.open(dest)
    if doc.page_count != pages:
        raise SystemExit(f"{dest} has {doc.page_count} pages, expected {pages} -- "
                         "BNSI has reissued the report; re-check the table pages")
    print(f"  {os.path.getsize(dest):,} bytes, {doc.page_count} pages")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    _get(PDF_URL, PDF_NAME, PDF_PAGES, 10_000_000)
    _get(NAT_URL, NAT_NAME, NAT_PAGES, 3_000_000)


# ---------------------------------------------------------------- parsing

def fold(s):
    """Match key for a category label.

    Uppercase alphanumerics only, which absorbs all four ways this source varies a name:
    the curly/straight apostrophe in `JEHOVAH'S WITNESS`, the footnote star on `OTHER
    RELIGION*`, the line-break hyphen in `NON- DENOMINATIONAL`, and the first release's
    title case (`Jehovahs Witness`) against the All-Island Report's caps. §12: never match
    a category on its raw string.
    """
    return "".join(c for c in str(s).upper() if c.isalnum())


KNOWN = {fold(c): c for c in CATEGORIES}
KNOWN[fold(RESIDUAL)] = RESIDUAL
KNOWN[fold(TOTAL_CAT)] = TOTAL_CAT

# **THE TWO REPORTS NAME TWO CELLS DIFFERENTLY, AND NO FOLD RECONCILES EITHER.** These are
# cross-document aliases and nothing more: the All-Island Report's spelling is the canonical
# one because it is what gets drawn, and Table 6.0 is only ever used as a check. Kept
# explicit and tiny rather than solved by fuzzy matching — a near-match rule here would
# happily pair `Church of God` with `Church of God of Prophecy` if BNSI ever split them.
#
#   * `Athiest` is BNSI's own misspelling, in the first release only. Left as the source
#     wrote it on the left-hand side; the drawn category is the All-Island Report's
#     correctly spelled `ATHEIST`.
NATIONAL_ALIAS = {
    "Church of God (including Church of God of Prophecy)":
        "CHURCH OF GOD AND CHURCH OF GOD OF PROPHECY",
    "Athiest": "ATHEIST",
}
KNOWN_NATIONAL = dict(KNOWN)
KNOWN_NATIONAL.update({fold(k): v for k, v in NATIONAL_ALIAS.items()})


def _num(tok, where):
    if not FIG.match(tok):
        raise SystemExit(f"{where}: {tok!r} is not a figure")
    return int(tok.replace(",", ""))


def _blocks(toks, where):
    """Walk `label, (TOTAL|Male|Female) + 8 figures` triples out of a token stream.

    Returns [(label, {sex: [8 figures]})] and the tokens left over after the last block,
    which is where the starred footnote lives.
    """
    out, i, label = [], 0, []
    cur = None
    while i < len(toks):
        if toks[i] in SEX_ROWS:
            figs = toks[i + 1:i + 1 + N_COLS]
            if len(figs) == N_COLS and all(FIG.match(f) for f in figs):
                vals = [_num(f, where) for f in figs]
                if toks[i] == "TOTAL":
                    cur = {}
                    out.append((" ".join(label).strip(), cur))
                    label = []
                if cur is None:
                    raise SystemExit(f"{where}: a {toks[i]!r} row before any TOTAL row")
                cur[toks[i]] = vals
                i += 1 + N_COLS
                continue
        label.append(toks[i])
        i += 1
    return out, " ".join(label)


def _panel_tokens(doc, pages, where):
    toks = []
    for p in pages:
        lines = [l.strip() for l in doc[p - 1].get_text().splitlines() if l.strip()]
        # The column header band ends on the line carrying `OVER` (`65 AND / OVER`).
        head = [j for j, l in enumerate(lines[:26]) if "OVER" in l.upper()]
        if not head:
            raise SystemExit(f"{where} p{p}: no `OVER` column header -- the table has been "
                             "re-typeset")
        for l in lines[max(head) + 1:]:
            toks.extend(l.split())
    return toks


def read_islands():
    """{island code: {'name', 'cells': {category: {sex: [8]}}, 'footnote': str}}"""
    import fitz

    path = os.path.join(RAW, PDF_NAME)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run with --fetch first")
    doc = fitz.open(path)
    if doc.page_count != PDF_PAGES:
        raise SystemExit(f"{path} has {doc.page_count} pages, expected {PDF_PAGES}")

    # Find each table's pages and its printed island name, from the caption itself. The
    # contents pages carry the same caption, so the body is identified by ALSO requiring
    # the `RELIGION OR` column header — and the front matter is skipped outright.
    spans, names = {}, {}
    for i in range(doc.page_count):
        if i < 20:
            continue
        flat = " ".join(doc[i].get_text().split())
        m = re.search(r"Table 12\.(\d+)\.", flat)
        if not m or "RELIGION OR" not in flat:
            continue
        tab = int(m.group(1))
        spans.setdefault(tab, []).append(i + 1)
        cap = re.search(r"Table 12\.\d+\.\s*Total Population by Sex, Age-Group and "
                        r"Religion,\s*(.+?)\s*2022", flat)
        if cap:
            names[tab] = " ".join(cap.group(1).split())

    if sorted(spans) != sorted(ISLANDS):
        raise SystemExit(f"found Table 12.x for {sorted(spans)}, expected "
                         f"{sorted(ISLANDS)} -- the report's island list has changed")

    out = {}
    for tab in sorted(spans):
        where = f"Table 12.{tab}"
        blocks, trailing = _blocks(_panel_tokens(doc, spans[tab], where), where)

        cells = {}
        for label, sexes in blocks:
            key = fold(label)
            if key not in KNOWN:
                raise SystemExit(
                    f"{where}: unknown category {label!r}. BNSI has added a religion to "
                    "the form; add it to CATEGORIES and to taxonomy/bs2022.py.")
            cat = KNOWN[key]
            if cat in cells:
                raise SystemExit(f"{where}: {cat!r} appears twice")
            if set(sexes) != set(SEX_ROWS):
                raise SystemExit(f"{where}/{cat}: sex rows are {sorted(sexes)}, "
                                 f"expected {sorted(SEX_ROWS)}")
            cells[cat] = sexes

        if TOTAL_CAT not in cells:
            raise SystemExit(f"{where}: no TOTAL row")

        # The printed name is checked against ISLANDS rather than trusted, because the
        # code->island pairing is the whole join and nothing downstream would catch a
        # table appearing out of order.
        printed = names.get(tab, "")
        if fold(printed) != fold(ISLANDS[tab]):
            raise SystemExit(f"{where} is captioned {printed!r} but island code {tab} is "
                             f"{ISLANDS[tab]!r} -- the tables are not in questionnaire "
                             "order and ISLANDS must be rechecked")

        foot = FOOTNOTE.search(" ".join(trailing.split()))
        out[tab] = {"name": ISLANDS[tab], "cells": cells,
                    "footnote": foot.group(1) if foot else ""}
    doc.close()
    return out


def read_national():
    """First release Table 6.0, All Bahamas — {category: total}. Used only as a check.

    Table 6.0's row shape is NOT Table 12.x's: the category label is followed straight away
    by its eight figures, with no `TOTAL` marker, and then by `Male` and `Female` rows. So
    it needs its own walk.
    """
    import fitz

    path = os.path.join(RAW, NAT_NAME)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run with --fetch first")
    doc = fitz.open(path)
    if doc.page_count != NAT_PAGES:
        raise SystemExit(f"{path} has {doc.page_count} pages, expected {NAT_PAGES}")

    toks = _panel_tokens(doc, NAT_TABLE_PAGES, "Table 6.0")
    doc.close()

    out, i, label = {}, 0, []
    while i < len(toks):
        figs = toks[i:i + N_COLS]
        if label and len(figs) == N_COLS and all(FIG.match(f) for f in figs):
            name = " ".join(label).strip()
            key = fold(name)
            if key not in KNOWN_NATIONAL:
                raise SystemExit(
                    f"Table 6.0: unknown category {name!r}. If this is the All-Island "
                    "Report's category under another name, add it to NATIONAL_ALIAS.")
            out[KNOWN_NATIONAL[key]] = [_num(f, "Table 6.0") for f in figs]
            label = []
            i += N_COLS
            # The two sex rows that follow are not needed; skip them by name.
            for sex in ("Male", "Female"):
                if i < len(toks) and toks[i] == sex:
                    i += 1 + N_COLS
            continue
        if toks[i] in ("Male", "Female"):
            i += 1
            continue
        label.append(toks[i])
        i += 1
    return out


# ---------------------------------------------------------------- output

def rows_from(islands):
    rows = []
    for code in sorted(islands):
        isl = islands[code]
        for cat, sexes in isl["cells"].items():
            note = "level=island"
            if cat == TOTAL_CAT:
                note += "; island total, not a religion category"
            elif cat == RESIDUAL:
                note += ("; per-island residual, contents named by the table's own "
                         f"footnote: {isl['footnote']}")
            rows.append({"geo_id": f"{code:02d}", "geo_level": "island",
                         "geo_name": isl["name"], "source_category": cat,
                         "count": sexes["TOTAL"][0], "basis": BASIS, "year": YEAR,
                         "source_id": SOURCE_ID, "note": note})
    return rows


def check(rows, islands, national):
    ok = True

    def result(label, bad, n, extra=""):
        nonlocal ok
        good = not bad
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {label} ({n} checks){extra}")
        for b in bad[:6]:
            print(f"        {b}")

    n_units = len({r["geo_id"] for r in rows})
    result(f"island {n_units} units (expected {len(ISLANDS)})",
           [] if n_units == len(ISLANDS) else ["wrong unit count"], n_units)

    # ---- 1. the seven age bands sum to the cell's own TOTAL column ----
    bad, n = [], 0
    for code, isl in islands.items():
        for cat, sexes in isl["cells"].items():
            for sex, vals in sexes.items():
                n += 1
                if sum(vals[1:]) != vals[0]:
                    bad.append(f"{isl['name']}/{cat}/{sex}: bands {sum(vals[1:]):,} "
                               f"vs total {vals[0]:,}")
    result("the seven age bands sum to each cell's own TOTAL column", bad, n,
           "  <- a check on every figure read, not just the margins")

    # ---- 2. Male + Female == TOTAL ----
    bad, n = [], 0
    for code, isl in islands.items():
        for cat, sexes in isl["cells"].items():
            n += 1
            if sexes["Male"][0] + sexes["Female"][0] != sexes["TOTAL"][0]:
                bad.append(f"{isl['name']}/{cat}")
    result("Male + Female == TOTAL", bad, n)

    # ---- 3. the categories sum to the island's own TOTAL row ----
    bad = []
    for code, isl in islands.items():
        got = sum(s["TOTAL"][0] for c, s in isl["cells"].items() if c != TOTAL_CAT)
        want = isl["cells"][TOTAL_CAT]["TOTAL"][0]
        if got != want:
            bad.append(f"{isl['name']}: {got:,} vs {want:,}")
    result("the categories sum to the island's own TOTAL row", bad, len(islands))

    # ---- 4. the 18 islands sum to the FIRST RELEASE's national table ----
    grand = sum(i["cells"][TOTAL_CAT]["TOTAL"][0] for i in islands.values())
    result(f"the 18 islands sum to the published national count {NATIONAL_TOTAL:,}",
           [] if grand == NATIONAL_TOTAL else [f"got {grand:,}"], 1)

    per_cat = {}
    for isl in islands.values():
        for cat, sexes in isl["cells"].items():
            if cat == TOTAL_CAT:
                continue
            per_cat[cat] = per_cat.get(cat, 0) + sexes["TOTAL"][0]

    missing = sorted(set(national) - set(per_cat) - {TOTAL_CAT})
    extra = sorted(set(per_cat) - set(national) - {RESIDUAL})
    result("every national category appears on at least one island",
           [f"national-only: {missing}"] if missing else [], len(national))
    result("no island category is absent from the national table",
           [f"island-only: {extra}"] if extra else [], len(per_cat))

    over = [f"{c}: islands {per_cat[c]:,} > national {national[c][0]:,}"
            for c in per_cat if c != RESIDUAL and per_cat[c] > national[c][0]]
    result("no category sums HIGHER across islands than the national table says",
           over, len(per_cat))

    shortfall = sum(national[c][0] - per_cat.get(c, 0)
                    for c in national if c != TOTAL_CAT)
    residual = per_cat.get(RESIDUAL, 0)
    result("the national shortfall equals the islands' residual exactly",
           [] if shortfall == residual else
           [f"shortfall {shortfall:,} vs residual {residual:,}"], 1,
           f"  <- {residual:,} people")

    # ---- what the collapsing costs, per island ----
    print(f"\n  {len(rows):,} rows on {len(ISLANDS)} islands. New Providence is the only "
          f"one with no residual —\n  every other island folds its smallest answers into "
          "OTHER RELIGION and names them in a footnote:")
    folds_none = []
    for code in sorted(islands):
        isl = islands[code]
        tot = isl["cells"][TOTAL_CAT]["TOTAL"][0]
        res = isl["cells"].get(RESIDUAL, {}).get("TOTAL", [0])[0]
        n_cat = len(isl["cells"]) - 1
        if re.search(r"\bNone\b", isl["footnote"]):
            folds_none.append((isl["name"], tot, res))
        print(f"    {code:>2} {isl['name']:<26} {tot:>8,} people  {n_cat:>2} cats  "
              f"residual {res:>4,} ({100.0 * res / tot if tot else 0:5.2f}%)")

    print(f"\n  the residual is {residual:,} people, "
          f"{100.0 * residual / NATIONAL_TOTAL:.3f}% of the country, "
          f"and every island's footnote names its contents.")
    if folds_none:
        n_pop = sum(t for _, t, _ in folds_none)
        n_res = sum(r for _, _, r in folds_none)
        print(f"  {len(folds_none)} island(s) fold `None` into it — "
              f"{', '.join(n for n, _, _ in folds_none)} — so those "
              f"{n_pop:,} people ({100.0 * n_pop / NATIONAL_TOTAL:.2f}% of the country) "
              f"report no irreligion at all.\n  Their residual is {n_res:,} people; "
              "nothing here redistributes it (§14.4).")

    print("\n  categories, national (from the islands, checked against Table 6.0):")
    for cat in CATEGORIES + [RESIDUAL]:
        v = per_cat.get(cat, 0)
        n_isl = sum(1 for i in islands.values() if cat in i["cells"])
        print(f"    {v:>9,}  {100.0 * v / NATIONAL_TOTAL:6.2f}%  {cat:<66} "
              f"on {n_isl:>2}/18")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    islands = read_islands()
    national = read_national()
    rows = rows_from(islands)
    check(rows, islands, national)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
