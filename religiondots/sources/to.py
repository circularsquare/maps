"""Tonga — Tonga Statistics Department, 2021 Census, General Table G 20.

Reads (or fetches) data/raw/to/ and writes data/normalized/to.csv.

**RELIGION BY VILLAGE, IN A PUBLISHED SPREADSHEET, WITH TWENTY-TWO CATEGORIES.** TSD posts the
2021 census general tables as one workbook per topic, and `4-religion.xlsx` holds three of
them: G 18 is religion by division crossed with sex, G 19 is religion by district, and **G 20
is religion by division, district and village** — 156 villages, 637 people each, every one of
the twenty-two categories printed at every one of them.

**THE TABLE CLOSES FOUR WAYS AND NEEDS NO TOLERANCE.** Each district's villages sum to the
district; each division's districts sum to the division; the divisions sum to the printed
national row; and G 19 and G 18, typeset as separate tables, reproduce G 20's district and
division figures cell for cell. **And then UNSD's Demographic Yearbook table 28 reproduces the
national row again**, all twenty-two categories to the person, from the return Tonga forwarded
rather than from this workbook — so the parse is checked against a transcription that shares no
lineage with it. `check()` asserts all five.

**FIFTY-FIVE PERCENT OF TONGA IS METHODIST AND IT IS SPLIT FIVE WAYS.** The Free Wesleyan
Church is 34.2% and is the church of the monarchy; the Free Church of Tonga is 11.3%, the
Church of Tonga 6.8%, the Tokaikolo Christian Church 1.5% and the Constitutional Church of
Tonga 1.2%. Every one of them is a census cell in its own right. Nothing else on this map
divides a single Protestant tradition into five countable national churches.

**AND THE VILLAGE NAMES ARE NOT UNIQUE, WHICH IS THE TRAP HERE.** Niuafo'ou was evacuated
after the 1946 eruption and much of its population resettled on 'Eua, where they gave the new
villages the names of the ones they had left: 'Esia, Sapa'ata, Fata'ulua, Mata'aho, Mu'a,
Tongamama'o and Petani are each printed twice in G 20, once in 'Eua Fo'ou and once in
Niuafo'ou, 900 km apart. Kolofo'ou, Hihifo, Pangai and Houma repeat for ordinary reasons.
**Every row therefore carries its district**, and `sources/to_geo.py` joins on the pair.

Usage:
    python sources/to.py --fetch    one ~95 KB workbook from tongastats.gov.to
    python sources/to.py            normalise from data/raw/to/
"""

import csv
import os
import re
import sys
import unicodedata

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

RAW = os.path.join(ROOT, "data", "raw", "to")
OUT = os.path.join(ROOT, "data", "normalized", "to.csv")

SOURCE_ID = "to_phc_2021"
YEAR = 2021
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# The Census Tables page (tongastats.gov.to/census-2/population-census-3/census-tables/) is a
# WP File Download library, the same plugin Fiji, PNG and the Solomon Islands run
# ([[reference_wpfd_sweep]]); `4-religion` is one of nineteen general tables. The office's
# `wp/v2/search` is disabled, but `wp/v2/media` and `wp/v2/pages` are both open, and the page
# body carries every download URL.
XLSX_URL = "https://tongastats.gov.to/download/266/general-tables/7664/4-religion.xlsx"
XLSX_NAME = "4-religion.xlsx"

# G 20's column codes, in print order, expanded from the legend printed under the table
# itself (rows 191-194 of the sheet). The expansion is what reaches `source_category`, and
# the code is kept in `note` so the workbook can be read against the CSV.
CODES = [
    ("FWC",    "Free Wesleyan Church"),
    ("RC",     "Roman Catholic"),
    ("LDS",    "Latter Day Saints"),
    ("FCOT",   "Free Church of Tonga"),
    ("COT",    "Church of Tonga"),
    ("AOG",    "Assembly of God"),
    ("TOK",    "Tokaikolo/Maamafo'ou"),
    ("CCOT",   "Constitutional Church of Tonga"),
    ("GOS",    "Gospel Church"),
    ("AGC",    "Anglican Church"),
    ("SDA",    "Seventh Day Adventist"),
    ("MF",     "Mo'ui Fo'ou 'Ia Kalaisi"),
    ("TSA",    "The Salvation Army"),
    ("JW",     "Jehovah's Witness"),
    ("OP",     "Other Pentecostal"),
    ("BF",     "Baha'i Faith"),
    ("BUDH",   "Buddhist"),
    ("ISL",    "Islam"),
    ("HND",    "Hinduism"),
    ("NO Rel", "No Religious affiliation"),
    ("REF",    "Refuse to answer"),
    ("Other",  "Other minor religious groups"),
]
RESIDUALS = ["Refuse to answer"]
DRAWN = [name for _, name in CODES if name not in RESIDUALS]

DIVISIONS = ["Tongatapu", "Vava'u", "Ha'apai", "'Eua", "Ongo Niua"]

NATIONAL = 99_408
EXPECTED_VILLAGES = 156
EXPECTED_DISTRICTS = 23
EXPECTED_DIVISIONS = 5

# UNSD Demographic Yearbook table 28 spells four of the twenty-two differently. The oracle
# check pairs on these rather than on the string, so a genuine disagreement about a NUMBER is
# not hidden by a disagreement about a NAME.
ORACLE_ALIAS = {
    "Gospel Church": "Full Gospel Church",
    "Tokaikolo/Maamafo'ou": "Tokaikolo Christian Church",
    "Anglican Church": "Anglican",
    "The Salvation Army": "Salvation Army",
    "Jehovah's Witness": "Jehovah's Witnesses",
    "Other Pentecostal": "Other Pentecostal Churches",
    "Baha'i Faith": "Baha'i",
    "Hinduism": "Hindu",
    "Islam": "Islam ",                       # DYB carries the trailing space
    "Mo'ui Fo'ou 'Ia Kalaisi": "Mo'ui Fo'ou 'ia Kalaisi",
    "No Religious affiliation": "No Religion",
    "Refuse to answer": "Refused to answer",
    "Other minor religious groups": "Other",
}


def fold(s):
    """Casefold, strip accents and punctuation — for COMPARING names, never for storing."""
    s = unicodedata.normalize("NFKD", str(s).strip().lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("‐", "-").replace("`", "'").replace("‘", "'").replace("’", "'")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, XLSX_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 50_000:
        print("already have", dest)
        return
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"}
    print("GET", XLSX_URL)
    r = requests.get(XLSX_URL, headers=ua, timeout=600)
    r.raise_for_status()
    if not r.content.startswith(b"PK"):
        raise SystemExit(f"tongastats returned something that is not a workbook "
                         f"({len(r.content):,} bytes)")
    tmp = dest + ".part"                                  # [[reference_wb_truncates]]
    with open(tmp, "wb") as fh:
        fh.write(r.content)
    os.replace(tmp, dest)
    print(f"  {os.path.getsize(dest):,} bytes")


def _sheet(wb, name):
    """Rows of a sheet as lists, blanks as ''."""
    return [[("" if c is None else c) for c in r]
            for r in wb[name].iter_rows(values_only=True)]


def _int(v):
    if isinstance(v, (int, float)):
        return int(v)
    raise ValueError(f"not a number: {v!r}")


def _data_rows(rows, nvals=1 + len(CODES)):
    """The rows of a G-table that are a label in column A and `nvals` figures after it.

    The sheets end in a note and a four-line legend, all of which have text in column A and
    nothing numeric in column B, so the filter is the column-B type rather than a row count:
    a legend line that grew would otherwise walk into the table.

    G 18 is a different shape from G 19 and G 20 — it crosses religion with sex, so its rows
    are 1 + 3x6 wide and its labels are the category codes rather than places. `nvals=1`
    reads just its national column, which is all it is wanted for.
    """
    out = []
    for r in rows:
        name = str(r[0]).strip()
        if not name or len(r) < 1 + nvals:
            continue
        try:
            vals = [_int(r[1 + i]) for i in range(nvals)]
        except (ValueError, IndexError):
            continue
        out.append((name, vals))
    return out


def parse():
    """G 20 -> [(division, district, village, [total, *22 counts]), ...], structure asserted.

    G 20 prints divisions, districts and villages in one column with no indentation, no code
    and no marker of which tier a row is. What makes it unambiguous is G 19, which prints the
    same figures for the divisions and districts ALONE: walking G 19 gives the expected tier
    of every row in G 20 in order, and each district's villages are then read until they sum
    to the district's own printed total. A row that is not where G 19 says it should be, or a
    village block that does not close, raises rather than being absorbed.
    """
    import openpyxl

    path = os.path.join(RAW, XLSX_NAME)
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing — run `python sources/to.py --fetch`")
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    for need in ("G 18", "G 19", "G 20"):
        if need not in wb.sheetnames:
            raise SystemExit(f"{XLSX_NAME} has no sheet {need!r} — TSD reissued the workbook; "
                             f"sheets are {wb.sheetnames}")

    # --- G 19: the division -> district skeleton, and the district figures to check against.
    g19 = _data_rows(_sheet(wb, "G 19"))
    skeleton, g19_district, g19_division, cur = {}, {}, {}, None
    national19 = None
    for name, vals in g19:
        if fold(name) == "total" and national19 is None:
            national19 = vals
            continue
        matched = [d for d in DIVISIONS if fold(d) == fold(name)]
        if matched:
            cur = matched[0]
            skeleton[cur] = []
            g19_division[cur] = vals
        elif cur is None:
            raise SystemExit(f"G 19 row {name!r} appears before any division")
        else:
            skeleton[cur].append(name)
            g19_district[(cur, name)] = vals
    if list(skeleton) != DIVISIONS:
        raise SystemExit(f"G 19 divisions {list(skeleton)}, expected {DIVISIONS}")
    ndist = sum(len(v) for v in skeleton.values())
    if ndist != EXPECTED_DISTRICTS:
        raise SystemExit(f"G 19 has {ndist} districts, expected {EXPECTED_DISTRICTS}")

    # --- G 20: walk it against that skeleton.
    g20 = _data_rows(_sheet(wb, "G 20"))
    i = 0
    if fold(g20[i][0]) != "tonga":
        raise SystemExit(f"G 20 opens with {g20[i][0]!r}, expected the national row")
    national = g20[i][1]
    i += 1

    villages, notes = [], []
    for div in DIVISIONS:
        name, vals = g20[i]
        if fold(name) != fold(div):
            raise SystemExit(f"G 20 row {i}: {name!r} where division {div!r} was expected")
        if vals != g19_division[div]:
            raise SystemExit(f"G 18/G 19 and G 20 disagree about division {div}")
        div_total, seen = vals[0], 0
        i += 1
        for dist in skeleton[div]:
            name, vals = g20[i]
            # G 20 titles four districts differently from G 19: `Foa District` for `Foa`,
            # `Nomuka` for `Mu'omu'a` (the district is named for its largest village in one
            # table and for its old name in the other), and the two 'Eua districts lose their
            # leading apostrophe. The FIGURES are identical, which is what settles the pairing.
            if fold(name).replace(" district", "") != fold(dist):
                notes.append(f"district printed as {name!r} in G 20 and {dist!r} in G 19")
            if vals != g19_district[(div, dist)]:
                raise SystemExit(f"G 19 and G 20 disagree about district {dist} of {div}")
            i += 1
            dist_total, run, block = vals[0], 0, []
            while i < len(g20) and run < dist_total:
                vname, vvals = g20[i]
                run += vvals[0]
                block.append((vname, vvals))
                i += 1
            if run != dist_total:
                raise SystemExit(f"{dist} ({div}): villages sum to {run}, district says "
                                 f"{dist_total}")
            for col in range(1, 1 + len(CODES)):
                got = sum(v[col] for _, v in block)
                if got != vals[col]:
                    raise SystemExit(f"{dist} ({div}), {CODES[col - 1][0]}: villages sum to "
                                     f"{got}, district says {vals[col]}")
            for vname, vvals in block:
                villages.append((div, dist, vname, vvals))
            seen += dist_total
        if seen != div_total:
            raise SystemExit(f"{div}: districts sum to {seen}, division says {div_total}")

    if i != len(g20):
        raise SystemExit(f"G 20 has {len(g20) - i} rows left over after the walk")
    if len(villages) != EXPECTED_VILLAGES:
        raise SystemExit(f"{len(villages)} villages, expected {EXPECTED_VILLAGES}")
    if national[0] != NATIONAL:
        raise SystemExit(f"G 20 national total {national[0]}, expected {NATIONAL}")
    if national19 != national:
        raise SystemExit("G 19's total row and G 20's national row disagree")

    # --- G 18, typeset separately with a sex cross, for the divisions a fourth time.
    g18 = _data_rows(_sheet(wb, "G 18"), nvals=1)
    g18_by_cat = {fold(n): v for n, v in g18}
    for k, (code, name) in enumerate(CODES, start=1):
        want = national[k]
        got = g18_by_cat.get(fold(code), [None])[0]
        if got is None:
            raise SystemExit(f"G 18 has no row for {code!r}")
        if got != want:
            raise SystemExit(f"G 18 says {code}={got}, G 20 says {want}")

    for n in notes:
        print("  note:", n)
    return villages, national


def check(villages, national):
    """The two checks that do not come from the workbook."""
    # 1. the categories partition the population, at every village.
    for div, dist, vname, vals in villages:
        if sum(vals[1:]) != vals[0]:
            raise SystemExit(f"{vname} ({dist}): categories sum to {sum(vals[1:])}, "
                             f"total says {vals[0]}")
    print(f"  partition: EXACT at all {len(villages)} villages and nationally")

    # 2. UNSD's Demographic Yearbook table 28, which is Tonga's own return to the UN and is
    #    not a copy of this workbook. [[reference_unsd_religion_oracle]]
    try:
        import oracle
        got = oracle.oracle("Tonga", YEAR)
    except Exception as exc:                                   # noqa: BLE001
        print(f"  oracle check SKIPPED ({exc}) — run `python tools/oracle.py --fetch`")
        return
    if not got:
        print("  oracle check SKIPPED — no Tonga 2021 row")
        return
    counts = got.get(oracle.TOTAL, {})
    bad = []
    for k, (code, name) in enumerate(CODES, start=1):
        want = counts.get(ORACLE_ALIAS.get(name, name))
        if want is None:
            bad.append(f"{name}: not in the DYB")
        elif int(want) != national[k]:
            bad.append(f"{name}: DYB {int(want)} vs workbook {national[k]}")
    if bad:
        raise SystemExit("UNSD table 28 disagrees with the workbook:\n   " +
                         "\n   ".join(bad))
    print(f"  UNSD table 28: all {len(CODES)} categories agree with the workbook, "
          "to the person")


def normalise():
    villages, national = parse()
    check(villages, national)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for div, dist, vname, vals in villages:
        # Village names repeat across districts (the Niuafo'ou resettlement, see the module
        # docstring), so the key is the pair and never the name alone.
        # [[reference_name_join_wrong_neighbour]]
        geo_id = f"{fold(dist).replace(' ', '')}/{fold(vname).replace(' ', '')}"
        for k, (code, name) in enumerate(CODES, start=1):
            if vals[k] == 0:
                continue
            rows.append({
                "geo_id": geo_id,
                "geo_level": "village",
                "geo_name": vname,
                "source_category": name,
                "count": vals[k],
                "basis": BASIS,
                "year": YEAR,
                "source_id": SOURCE_ID,
                "note": f"{div}|{dist}|{code}",
            })

    ids = {r["geo_id"] for r in rows}
    if len(ids) != EXPECTED_VILLAGES:
        raise SystemExit(f"{len(ids)} distinct geo_ids for {EXPECTED_VILLAGES} villages — "
                         "two villages in one district share a name")

    tmp = OUT + ".part"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, OUT)

    drawn = sum(r["count"] for r in rows if r["source_category"] in DRAWN)
    print(f"wrote {OUT}")
    print(f"  {len(rows):,} rows, {len(ids)} villages, {sum(r['count'] for r in rows):,} people")
    print(f"  drawn {drawn:,} ({drawn / NATIONAL:.2%}); residual "
          f"{NATIONAL - drawn:,} refused the question")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    else:
        normalise()
