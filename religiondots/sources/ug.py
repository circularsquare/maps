"""Uganda — religion by district, 2002 census Table B7.

Writes data/normalized/ug.csv.

**THIS IS THE ONLY RELIGION-BY-GEOGRAPHY TABLE UGANDA HAS EVER PUBLISHED, and it is
filed away from every census report.** `Table B7: Religion by District for the
Population` is Annex 2 of the 2002 census, published as its own two-page PDF in
`ubos.org/onlinefiles/uploads/ubos/census_tabulations/`, a directory of fifteen loose
annex tables that no current UBOS page links to. Seven categories on the 56 districts of
2002. Everything else Uganda has released puts religion at the nation and stops:

  * **NPHC 2024 Final Report Volume 1** (434 pages) — Table 3.1 is religion for 2014 and
    2024, ten categories, national only. Religion appears on fourteen of its pages and
    not once beside a geography, while Table 3.4 in the same chapter breaks birth
    registration down by sub-region.
  * **The NPHC 2024 statistics portal** (`statistics.ubos.org/nphc`) is a real query
    engine — `api/get_profile_data.php`, `api/get_geospatial_data.php`, and four
    location endpoints — reaching PARISH level. It serves fifteen tables: sex, household
    size, age groups, birth registration, mobile phones, internet, ICT devices,
    information sources, health, mental health, unemployment, water and sanitation,
    lighting, subsistence and PDM, children out of school. `format=all` and
    `format=summary` return the same fifteen. Religion is in none of them and the map's
    indicator list has no religion option.
  * **The seventeen NPHC 2024 sub-region profile reports** (Acholi is 412 pages, district
    by district, down to sub-county) carry the same fifteen tables. No religion.
  * **NPHC-2024-Subcounty-Profiles-Excel-Tables.xlsx**, 13,527 rows to parish. No religion.
  * **NPHC 2014 Main Report** — Table 4.1, religion for 2002 and 2014, national only.
  * **The 2014 Area Specific Profiles**, one per district. No religion.
  * **The 2002 Population Composition analytical report** reaches four REGIONS plus
    urban/rural (its Table 3.7) and no further.

**THE UNSD DEMOGRAPHIC YEARBOOK IS THE OUTSIDE WITNESS.** Its Uganda 2002 row is seven
categories totalling 24,433,132, and every one of the seven equals this table's national
column to the person — a separate publication, UBOS's own return to the UN, reproducing
the file being parsed here. `python tools/oracle.py Uganda` prints them.

**AND THE TABLE RECONCILES AGAINST A DIFFERENT UBOS TABLE WITH MORE CATEGORIES.** The
2002 Population Composition report's Table 3.6 gives NINE national categories and
excludes Kotido district. Take Kotido out of B7 and the five shared cells agree to the
hundred (Catholic 9,921,398 against 9,921.4 thousand, Anglican 8,753,811 against 8,753.8,
Moslem 2,953,808 against 2,953.8, Pentecostal 1,128,020 against 1,128.0, SDA 367,596
against 367.6), and B7's `Other` plus `None` equals 3.6's Orthodox plus Other Christian
plus Bahai plus Others, 716,629 against 716.6 thousand. That is what `Other` contains and
where it came from, checked rather than assumed.

**KOTIDO IS THE ONE DISTRICT THAT NEEDS A DECISION AND IT IS TAKEN IN countries.py.**
Table B1 of the same annex series gives Kotido 591,889 people in 2002. Table A3 of the
2014 census Main Report gives the 1991 and 2002 censuses redistributed onto 2014
boundaries, and summing its 2002 column over the districts that came out of 2002 Kotido
gives 377,102 — 214,787 fewer, 36% of the district. For the other 55 districts the two
publications agree TO THE PERSON on both the 1991 and the 2002 column, and Kotido's 1991
figure agrees as well, so this is a revision of one count and not a boundary artefact.
`sources/ug.md` has the full record.

Usage:
    python sources/ug.py --fetch     four PDFs, ~2.5 MB
    python sources/ug.py             rebuild from data/raw/ug/
"""

import os
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")     # [[reference_scipy_eats_all_cores]]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ug")
OUT = os.path.join(ROOT, "data", "normalized", "ug.csv")

SOURCE_ID = "ug_phc_2002_tableB7"

# The 2002 annex tables live only on the retired ubos.org tree; the live site 301s the
# path away and serves nothing at the new one. The Wayback Machine's `id_` replay returns
# the original bytes, not a rewritten page.
OLD = ("http://www.ubos.org/onlinefiles/uploads/ubos/census_tabulations/%s.pdf")
WB = "https://web.archive.org/web/2018id_/" + OLD
MAIN_2014 = ("https://www.ubos.org/wp-content/uploads/publications/"
             "03_20182014_National_Census_Main_Report.pdf")

FILES = [
    ("centableB7.pdf", WB % "centableB7", 40_000),      # religion by district
    ("centableB1.pdf", WB % "centableB1", 20_000),      # population by district, 3 censuses
    ("centableC1.pdf", WB % "centableC1", 100_000),     # district/county/sub-county tree
    ("2014_NPHC_Main_Report.pdf", MAIN_2014, 1_000_000),
]

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

NUM = re.compile(r"^[\d,]+$")

# Table B7's column order, left to right. `None` is an answer ("no religion"), not a
# missing value, and is the last column before the row total.
CATEGORIES = ["Catholic", "Anglican", "SDA", "Pentecostal", "Moslem", "Other", "None"]

# The four regions in the order B7 prints them, with the district count under each. Used
# to give every district a stable id and to check that no row was dropped.
REGION_SIZES = [("Central", 13), ("Eastern", 15), ("Northern", 13), ("Western", 15)]

NATIONAL = {          # B7's own UGANDA row, which is also UNSD Demographic Yearbook's
    "Catholic": 10_242_594,
    "Anglican": 8_782_821,
    "SDA": 367_972,
    "Pentecostal": 1_129_647,
    "Moslem": 2_956_121,
    "Other": 741_589,
    "None": 212_388,
}
NATIONAL_TOTAL = 24_433_132

# Table 3.6 of the 2002 Population Composition analytical report, in thousands, EXCLUDING
# Kotido district. Five of its nine categories are shared with B7 and are checked against
# B7-minus-Kotido below; the other four are what B7 folds into `Other`.
COMPOSITION_3_6 = {
    "Catholic": 9_921.4, "Anglican": 8_753.8, "Moslem": 2_953.8,
    "Pentecostal": 1_128.0, "SDA": 367.6,
}
COMPOSITION_RESIDUAL = 35.4 + 282.3 + 18.5 + 380.4      # Orthodox, Other Christian, Bahai, Others


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for name, url, floor in FILES:
        path = os.path.join(RAW, name)
        if os.path.exists(path) and os.path.getsize(path) >= floor:
            print("already have", name)
            continue
        print("GET", url)
        r = requests.get(url, timeout=900, headers={"User-Agent": UA}, verify=False)
        r.raise_for_status()
        body = r.content
        # [[reference_pdf_truncated_at_source]]: a 200 and a plausible length are not a
        # readable PDF. Check the header and the trailer before writing.
        if body[:4] != b"%PDF":
            raise SystemExit(f"{name}: not a PDF, starts {body[:16]!r}")
        if b"%%EOF" not in body[-2048:]:
            raise SystemExit(f"{name}: no %%EOF trailer, truncated at source "
                             f"({len(body):,} bytes)")
        if len(body) < floor:
            raise SystemExit(f"{name}: {len(body):,} bytes, expected at least {floor:,}")
        with open(path + ".part", "wb") as fh:      # [[reference_wb_truncates]]
            fh.write(body)
        os.replace(path + ".part", path)
        print(f"  {len(body):,} bytes -> {path}")


def rows_of(page):
    """Visual lines of a page, as (leftmost x, name, [numbers])."""
    lines = {}
    for x0, y0, x1, y1, w, *_ in page.get_text("words"):
        lines.setdefault(round(y0 / 3.0), []).append((x0, w))
    out = []
    for key in sorted(lines):
        ws = sorted(lines[key])
        nums = [t for _, t in ws if NUM.match(t)]
        namew = [(x, t) for x, t in ws if not NUM.match(t)]
        name = " ".join(t for _, t in namew).strip()
        out.append((namew[0][0] if namew else None, name,
                    [int(t.replace(",", "")) for t in nums]))
    return out


def parse_b7(path):
    """-> [(region, district, {category: count}, printed_total)] in print order."""
    import fitz

    doc = fitz.open(path)
    if doc.page_count != 2:
        raise SystemExit(f"{path} has {doc.page_count} pages, expected 2")
    districts, totals = [], {}
    for pi in range(doc.page_count):
        for _x, name, nums in rows_of(doc[pi]):
            if len(nums) != 8 or not name:
                continue
            counts = dict(zip(CATEGORIES, nums[:7]))
            if name in ("Regional Total", "Region", "Total"):
                totals.setdefault("regions", []).append((counts, nums[7]))
            elif name == "UGANDA":
                totals["uganda"] = (counts, nums[7])
            else:
                districts.append((name, counts, nums[7]))
    return districts, totals


def main():
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    b7 = os.path.join(RAW, "centableB7.pdf")
    if not os.path.exists(b7):
        raise SystemExit(f"missing {b7} — run with --fetch first")

    districts, totals = parse_b7(b7)
    want = sum(n for _, n in REGION_SIZES)
    if len(districts) != want:
        raise SystemExit(f"Table B7 gave {len(districts)} district rows, expected {want}: "
                         f"{[d for d, _, _ in districts]}")
    if len(totals.get("regions", [])) != len(REGION_SIZES):
        raise SystemExit(f"expected {len(REGION_SIZES)} regional total rows, got "
                         f"{len(totals.get('regions', []))}")
    if "uganda" not in totals:
        raise SystemExit("no UGANDA row in Table B7")
    print(f"Table B7: {len(districts)} districts, {len(CATEGORIES)} categories")

    # 1. every row's seven cells sum to the total the table prints beside them.
    for name, counts, tot in districts:
        if sum(counts.values()) != tot:
            raise SystemExit(f"{name}: cells sum to {sum(counts.values()):,}, "
                             f"printed total {tot:,}")

    # 2. the districts under each region sum to that region's printed row.
    i = 0
    region_of = {}
    for (region, n), (rcounts, rtot) in zip(REGION_SIZES, totals["regions"]):
        block = districts[i:i + n]
        i += n
        for cat in CATEGORIES:
            got = sum(c[cat] for _, c, _ in block)
            if got != rcounts[cat]:
                raise SystemExit(f"{region} {cat}: districts sum to {got:,}, "
                                 f"printed {rcounts[cat]:,}")
        for name, _, _ in block:
            region_of[name] = region
    print("  every district row sums to its printed total; every region's districts sum "
          "to its printed row")

    # 3. the national row equals the sum, and equals UNSD's Uganda 2002 return.
    nat = {cat: sum(c[cat] for _, c, _ in districts) for cat in CATEGORIES}
    for cat in CATEGORIES:
        if nat[cat] != totals["uganda"][0][cat]:
            raise SystemExit(f"{cat}: districts sum to {nat[cat]:,}, UGANDA row "
                             f"{totals['uganda'][0][cat]:,}")
        if nat[cat] != NATIONAL[cat]:
            raise SystemExit(f"{cat}: parsed {nat[cat]:,}, UNSD Demographic Yearbook "
                             f"{NATIONAL[cat]:,}")
    if sum(nat.values()) != NATIONAL_TOTAL:
        raise SystemExit(f"national total {sum(nat.values()):,}, expected {NATIONAL_TOTAL:,}")
    print(f"  all seven national figures equal the UNSD Demographic Yearbook's Uganda 2002 "
          f"return to the person ({NATIONAL_TOTAL:,} people)")

    # 4. drop Kotido and the five shared cells must equal the analytical report's Table
    #    3.6, which is a different publication with nine categories and no Kotido.
    kot = next((c for n, c, _ in districts if n == "Kotido"), None)
    if kot is None:
        raise SystemExit("no Kotido row — the Table 3.6 reconciliation cannot run")
    for cat, thousands in COMPOSITION_3_6.items():
        got = (nat[cat] - kot[cat]) / 1000.0
        if abs(got - thousands) > 0.05:
            raise SystemExit(f"{cat} without Kotido is {got:,.1f}k, Table 3.6 prints "
                             f"{thousands:,.1f}k")
    resid = (nat["Other"] - kot["Other"] + nat["None"] - kot["None"]) / 1000.0
    if abs(resid - COMPOSITION_RESIDUAL) > 0.05:
        raise SystemExit(f"`Other` plus `None` without Kotido is {resid:,.1f}k, Table 3.6's "
                         f"Orthodox+Other Christian+Bahai+Others is "
                         f"{COMPOSITION_RESIDUAL:,.1f}k")
    print("  minus Kotido, the five shared cells and the residual reproduce the 2002 "
          "Population\n  Composition report's Table 3.6, which is nine categories and a "
          "different publication")

    # --- write ------------------------------------------------------------------
    recs = []
    for idx, (name, counts, tot) in enumerate(districts, start=1):
        gid = "UG%02d" % idx
        for cat in CATEGORIES:
            recs.append({
                "geo_id": gid, "geo_level": "district_2002", "geo_name": name,
                "source_category": cat, "count": counts[cat], "basis": "self_id",
                "year": "2002", "source_id": SOURCE_ID,
                "note": f"level=district_2002; region={region_of[name]}",
            })
        recs.append({
            "geo_id": gid, "geo_level": "district_2002", "geo_name": name,
            "source_category": "Total", "count": tot, "basis": "self_id",
            "year": "2002", "source_id": SOURCE_ID,
            "note": (f"level=district_2002; region={region_of[name]}; universe total, "
                     "not a religion category"),
        })
    df = pd.DataFrame(recs)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(df)} rows, {sum(nat.values()):,} people, "
          f"{len(CATEGORIES)} categories, {len(districts)} units)")

    print("\n  national, as published:")
    for cat, v in sorted(nat.items(), key=lambda kv: -kv[1]):
        print(f"    {100.0 * v / NATIONAL_TOTAL:6.2f}%  {cat:<12} {v:>10,}")

    print("\n  the four districts where `Other` is largest, which is what that cell is:")
    share = sorted(((c["Other"] / t, n, c["Other"], t) for n, c, t in districts),
                   reverse=True)[:4]
    for sh, n, v, t in share:
        print(f"    {n:<14} {sh * 100:5.1f}%  ({v:,} of {t:,})")


if __name__ == "__main__":
    main()
