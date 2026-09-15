"""Pakistan — 2023 Digital Census, religion by district, read from PBS's own Table 9 PDFs.

Reads (or fetches) data/raw/pk2023/ and writes data/normalized/pk.csv (the 2017 build's file is
pk2017.csv). Named `pk_2023.py` and not `pk2023.py` because a sources/ module with a taxonomy
module's name shadows it on coverage.py's sys.path (spec §12).

Replaces the 2017 build (`sources/pk.py`, USCB's transcription). Anita approved the rebuild
2026-09-14, at district: *"ok we can rebuild pakistan at district."* `sources/pk.md` §7b is
where the tables were found and §9 is what 2023 changed.

THE SOURCE IS PBS ITSELF, one PDF per province, under a path that is not a WordPress media
item (which is why §7a's exhaustive media-library sweep never saw it):

    https://www.pbs.gov.pk/wp-content/uploads/census_tables/tables/table_9_<prov>_districts.pdf
    ...and table_9_islamabad.pdf for the capital territory, which has no `_districts`

*Table 9: Population by sex, religion and rural/urban, Census-2023.* Each PDF runs province,
then each DISTRICT, then that district's tehsils/talukas; every block carries ALL LOCALITIES,
RURAL and URBAN, each by ALL SEXES / MALE / FEMALE / TRANSGENDER. Nine columns: Total, Muslim,
Christian, Hindu Jati, Qadiani/Ahmadi, Scheduled Castes, Sikh, Parsi, Others.

THE PARSER READS BY GEOMETRY, NOT BY TEXT ORDER. The text layer is real, but its stream order
puts each block's bold header after its rows, page 1 prints thousands separators and later
pages do not, and the Muslim column is set in bold on some pages. So every span is placed by
its bounding box: a label at the left margin makes a row, numbers at the same height fill it,
and a column is whichever of the printed column numbers `1`..`10` the number's centre is
nearest to between midpoints. A row with anything other than nine cells fails loudly.

WHAT IS ASSERTED, because a PDF parse is the classic silent failure (spec §12, shapes 1 and 4):
  * inside every block, on all 12 rows: Total = the sum of the eight religions, and
    ALL SEXES = MALE + FEMALE + TRANSGENDER; on every column, ALL LOCALITIES = RURAL + URBAN;
  * tehsils sum to their district and districts to their province, on every column;
  * provinces equal the National Census Report's Table 4.13, cell by cell (its `Others` is
    Table 9's Sikh + Parsi + Others);
  * Lahore and Islamabad equal the two records §7a recovered from the dead census23 portal's
    archived AJAX response, which is a second publication path sharing no text with these PDFs.

THE DRAWN TIER IS STILL DISTRICT. The 2023 state prints religion by tehsil, and Anita chose to
keep district anyway (spec §14.4's ceiling is now a choice below it, not the state's limit;
`sources/pk.md` §3 and §9). The tehsil rows are normalised for the sum checks and not drawn.

Usage:
    python sources/pk_2023.py --fetch    five GETs, ~10 MB
    python sources/pk_2023.py            normalise from data/raw/pk2023/
"""

import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "pk2023")
# `pk.csv`, not `pk2023.csv`: tools/check_mapping.py and tools/gap_share.py read
# data/normalized/<cc>.csv, so the drawn vintage owns the plain name. The 2017 file is pk2017.csv.
OUT = os.path.join(ROOT, "data", "normalized", "pk.csv")

SOURCE_ID = "pk_census_2023_pbs_table9"
YEAR = 2023
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

BASE = "https://www.pbs.gov.pk/wp-content/uploads/census_tables/tables/"
# (file, province as Table 4.13 names it, least bytes, pages)
FILES = [
    ("table_9_kp_districts.pdf", "Khyber Pakhtunkhwa", 2_000_000, 46),
    ("table_9_punjab_districts.pdf", "Punjab", 2_000_000, None),
    ("table_9_sindh_districts.pdf", "Sindh", 2_000_000, None),
    ("table_9_balochistan_districts.pdf", "Balochistan", 2_000_000, 47),
    ("table_9_islamabad.pdf", "Islamabad", 30_000, 1),
]

CATS = ["Muslim", "Christian", "Hindu Jati", "Qadiani/Ahmadi", "Scheduled Castes",
        "Sikh", "Parsi", "Others"]
NCOL = 1 + len(CATS)          # Total + eight

SECTIONS = ["ALL LOCALITIES", "RURAL", "URBAN"]
SEXES = ["ALL SEXES", "MALE", "FEMALE", "TRANSGENDER"]
LABELS = set(SECTIONS) | set(SEXES)

# National Census Report 2023, Table 4.13 (p.137). Columns: Total, Muslim, Christian, Hindu,
# Qadiani/Ahmadi, Scheduled Castes, Others -- where Others = Table 9's Sikh + Parsi + Others.
# **PUNJAB'S MUSLIM CELL IS MISPRINTED IN THE REPORT AS 24,462,897**, a dropped leading 1; the
# row's own total less its other five cells is 124,462,897, which is what is held here, and
# the national Muslim cell (231,686,709) only adds up with the corrected figure.
T413 = {
    "Khyber Pakhtunkhwa": (40_641_120, 40_486_153, 134_884, 5_473, 951, 629, 13_030),
    "Punjab": (127_333_305, 124_462_897, 2_458_924, 228_559, 140_512, 21_157, 21_256),
    "Sindh": (55_638_409, 50_126_428, 546_968, 3_575_848, 18_266, 1_325_559, 45_340),
    "Balochistan": (14_562_011, 14_429_568, 62_731, 57_010, 557, 2_097, 10_048),
    "Islamabad": (2_283_244, 2_181_663, 97_281, 839, 2_398, 45, 1_018),
}
T413_PAKISTAN = (240_458_089, 231_686_709, 3_300_788, 3_867_729, 162_684, 1_349_487, 90_692)

# The census headline. The religion universe is 1,041,342 smaller: NCR p.124, "includes
# individuals from restricted areas for whom only headcounts are available. Consequently,
# detailed demographic characteristics such as ... religion ... are available for only
# 240,458,089". Those people are in no religion table at any level.
CENSUS_HEADLINE = 241_499_431

# sources/pk.md §7a: the two records in the one archived census23.pbos.gov.pk AJAX response,
# r1..r8 in Table 9's column order. An independent publication path for the same census.
PORTAL = {
    "LAHORE": (12_978_661, 12_363_149, 602_431, 2_487, 7_139, 324, 715, 77, 2_339),
    "ISLAMABAD": (2_283_244, 2_181_663, 97_281, 839, 2_398, 45, 60, 10, 948),
}

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for name, _, least, _ in FILES:
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > least:
            print("already have", dest)
            continue
        r = requests.get(BASE + name, timeout=600, headers=UA)
        r.raise_for_status()
        # spec §12 shape 4: the magic bytes and the trailer, not the status code
        # ([[reference_pdf_truncated_at_source]]).
        if r.content[:5] != b"%PDF-" or b"%%EOF" not in r.content[-2048:]:
            raise SystemExit(f"{name}: not a whole PDF (starts {r.content[:16]!r})")
        if len(r.content) < least:
            raise SystemExit(f"{name}: only {len(r.content):,} bytes")
        with open(dest, "wb") as fh:
            fh.write(r.content)
        print(f"wrote {dest} ({len(r.content):,} bytes)")


# ---------------------------------------------------------------- parsing

_NUM = re.compile(r"^-$|^\d{1,3}(,\d{3})+$|^\d+$")


def _num(t):
    return 0 if t == "-" else int(t.replace(",", ""))


def _spans(page):
    out = []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            for s in ln["spans"]:
                t = " ".join(s["text"].split())
                if t:
                    x0, y0, x1, y1 = s["bbox"]
                    out.append({"t": t, "x0": x0, "x1": x1, "y": (y0 + y1) / 2,
                                "bold": "Bold" in s["font"]})
    return out


def _column_edges(page):
    """The table's own vertical rules, off the header box the page draws.

    NOT the midpoints between the printed column numbers `1`..`10`, which was the first
    version and failed on KP page 1: the columns are different widths (Scheduled Castes is
    wide, Sikh is narrow) and every number is RIGHT-aligned against its cell's border, so a
    one-digit Scheduled Castes cell sits 3 pt past the 7|8 midpoint and read as Sikh. The rules
    are drawn as thin rectangles, a pair of x's ~0.8 pt apart per rule; there are 11.
    Returns the 9 internal borders, col1|col2 ... col9|col10, plus the right edge.
    """
    xs = []
    for d in page.get_drawings():
        for it in d["items"]:
            if it[0] == "l" and abs(it[1].x - it[2].x) < 0.5 and min(it[1].y, it[2].y) < 70:
                xs.append(it[1].x)
            elif it[0] == "re" and it[1].y0 < 70:
                xs += [it[1].x0, it[1].x1]
    xs.sort()
    rules = []
    for x in xs:
        if rules and x - rules[-1][-1] < 1.5:
            rules[-1].append(x)
        else:
            rules.append([x])
    rules = [sum(r) / len(r) for r in rules]
    if len(rules) != 11:
        return None
    return rules[1:10], rules[10]


def events(path):
    """The PDF as an ordered stream of ('head', name) and ('row', label, cells|None)."""
    import fitz

    doc = fitz.open(path)
    evs, pages = [], doc.page_count
    edges = right = None
    for pno in range(pages):
        sp = _spans(doc[pno])
        e = _column_edges(doc[pno])
        if e is not None:
            edges, right = e
        if edges is None:
            raise SystemExit(f"{os.path.basename(path)} p{pno + 1}: no column header")
        header_y = max(s["y"] for s in sp if s["t"] in {"1", "10"} and s["bold"] and s["y"] < 60)

        labels = [s for s in sp if s["t"] in LABELS and s["x0"] < edges[0]]
        nums = [s for s in sp if _NUM.match(s["t"]) and s["y"] > header_y + 2]
        heads = [s for s in sp if s["bold"] and s["y"] > header_y + 2 and s["t"] not in LABELS
                 and not _NUM.match(s["t"]) and not s["t"].startswith("TABLE 9")]
        used = set()
        items = []
        for lab in labels:
            row = [s for s in nums if abs(s["y"] - lab["y"]) < 2.5]
            for s in row:
                used.add(id(s))
            items.append((lab["y"], 1, "row", lab["t"], row))
        stray = [s for s in nums if id(s) not in used]
        if stray:
            raise SystemExit(f"{os.path.basename(path)} p{pno + 1}: {len(stray)} numbers on no "
                             f"labelled row, first {stray[0]['t']!r} at y={stray[0]['y']:.1f}")
        # a header that wraps onto a second line is merged into one name
        heads.sort(key=lambda s: (s["y"], s["x0"]))
        merged = []
        for h in heads:
            if merged and h["y"] - merged[-1][0] < 8 and not any(
                    merged[-1][0] < it[0] < h["y"] for it in items):
                merged[-1] = (merged[-1][0], merged[-1][1] + " " + h["t"])
            else:
                merged.append((h["y"], h["t"]))
        for y, name in merged:
            items.append((y, 0, "head", name, None))

        for y, _, kind, text, row in sorted(items, key=lambda i: (i[0], i[1])):
            if kind == "head":
                evs.append(("head", " ".join(text.split()), pno + 1))
                continue
            if not row:
                evs.append(("row", text, None, pno + 1))
                continue
            cells = [None] * NCOL
            for s in row:
                if s["x1"] > right + 1.0:
                    raise SystemExit(f"{os.path.basename(path)} p{pno + 1} y={y:.1f}: cell "
                                     f"{s['t']!r} ends at x={s['x1']:.1f}, past the table's "
                                     f"right rule at {right:.1f}")
                # right-aligned, so the RIGHT edge says which cell; 1 pt absorbs a spill
                cx = s["x1"] - 1.0
                col = sum(cx > ed for ed in edges)      # 0 = AREA/SEX, 1 = Total, ... 9 = Others
                k = col - 1
                if not 0 <= k < NCOL or cells[k] is not None:
                    raise SystemExit(f"{os.path.basename(path)} p{pno + 1} y={y:.1f}: cell "
                                     f"{s['t']!r} lands in column {col}, which is "
                                     f"{'taken' if 0 <= k < NCOL else 'not a data column'}")
                cells[k] = _num(s["t"])
            if None in cells:
                raise SystemExit(f"{os.path.basename(path)} p{pno + 1} y={y:.1f} {text}: "
                                 f"{sum(c is None for c in cells)} of {NCOL} cells empty")
            evs.append(("row", text, cells, pno + 1))
    return evs, pages


def blocks(evs, fname, default_name):
    """Group the stream into blocks: name -> {(section, sex): [9 ints]}."""
    out, cur, section = [], None, None
    for ev in evs:
        if ev[0] == "head":
            cur = {"name": ev[1], "rows": {}, "page": ev[2]}
            out.append(cur)
            section = None
            continue
        _, label, cells, page = ev
        if cells is None:
            if label not in SECTIONS:
                raise SystemExit(f"{fname} p{page}: `{label}` with no numbers")
            section = label
            continue
        if cur is None:
            cur = {"name": default_name, "rows": {}, "page": page}
            out.append(cur)
        if section is None or label not in SEXES:
            raise SystemExit(f"{fname} p{page}: row `{label}` outside a section in {cur['name']}")
        key = (section, label)
        if key in cur["rows"]:
            raise SystemExit(f"{fname} p{page}: {cur['name']} has {key} twice -- a header was "
                             f"missed and two blocks ran together")
        cur["rows"][key] = cells
    return out


def level_of(name, first):
    if first:
        return "province"
    if name.endswith(" DISTRICT"):
        return "district"
    return "tehsil"


def check_block(b, fname):
    bad = []
    rows = b["rows"]
    want = {(s, x) for s in SECTIONS for x in SEXES}
    if set(rows) != want:
        return [f"{b['name']}: rows {sorted(want - set(rows))} missing"]
    for k, c in rows.items():
        if c[0] != sum(c[1:]):
            bad.append(f"{b['name']} {k}: total {c[0]:,} != religions {sum(c[1:]):,}")
    for s in SECTIONS:
        for i in range(NCOL):
            if rows[(s, "ALL SEXES")][i] != sum(rows[(s, x)][i] for x in SEXES[1:]):
                bad.append(f"{b['name']} {s} col{i}: sexes do not add")
    for x in SEXES:
        for i in range(NCOL):
            if rows[("ALL LOCALITIES", x)][i] != rows[("RURAL", x)][i] + rows[("URBAN", x)][i]:
                bad.append(f"{b['name']} {x} col{i}: rural + urban != all")
    return bad


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def read():
    ok = True
    units = []           # dicts: level, province, district, name, cells
    for fname, prov, least, pages in FILES:
        path = os.path.join(RAW, fname)
        with open(path, "rb") as fh:
            raw = fh.read()
        good = raw[:5] == b"%PDF-" and b"%%EOF" in raw[-2048:] and len(raw) >= least
        evs, npages = events(path)
        good &= pages is None or npages == pages
        bl = blocks(evs, fname, "ISLAMABAD" if prov == "Islamabad" else prov.upper())
        errs = []
        for b in bl:
            errs += check_block(b, fname)
        n_d = sum(1 for i, b in enumerate(bl) if i > 0 and (
            level_of(b["name"], False) == "district" or b["name"] in DISTRICT_HEADERS))
        print(f"  {'OK ' if good and not errs else 'BAD'} {fname}: {npages} pages, {len(bl)} "
              f"blocks ({n_d} districts), every block's 12 rows add across religions, sexes "
              f"and rural/urban ({len(errs)} failures)")
        for e in errs[:10]:
            print("        ", e)
        ok &= good and not errs

        district = None
        for i, b in enumerate(bl):
            lv = level_of(b["name"], i == 0)
            if b["name"] in DISTRICT_HEADERS:
                lv = "district"
            if prov == "Islamabad" and i == 0:
                # the capital's first block is headed ISLAMABAD DISTRICT and is ALSO the
                # territory row Table 4.13 prints; the district copy is added after the loop
                lv = "province"
                district = b["name"]
            if lv == "district":
                district = b["name"]
            units.append({"level": lv, "province": prov, "district": district if lv != "province" else None,
                          "name": b["name"], "cells": b["rows"][("ALL LOCALITIES", "ALL SEXES")],
                          "page": b["page"], "file": fname})
        if prov == "Islamabad":
            if len(bl) != 2:
                raise SystemExit(f"Islamabad has {len(bl)} blocks, expected the territory "
                                 f"and its one tehsil")
            u = dict(next(x for x in units if x["province"] == "Islamabad"))
            u.update(level="district", district="ISLAMABAD DISTRICT", name="ISLAMABAD DISTRICT")
            units.append(u)

    # PBS "List of Administrative Districts by Division & Province (as on 01-03-2023)",
    # pbs.gov.pk/wp-content/uploads/2020/07/List-of-Administrative-Districts-2023.pdf
    for prov, want in DISTRICTS_2023.items():
        got = sum(1 for u in units if u["province"] == prov and u["level"] == "district")
        good = got == want
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {prov}: {got} district blocks, and PBS's "
              f"01-03-2023 district list has {want}")
    return units, ok


# Block headers that are districts without saying DISTRICT. Malakand's legal name is the
# Malakand Protected Area, and a header rule on the suffix alone silently filed it, and its two
# sub-divisions, under Lower Kohistan as three tehsils; the district count below is what caught it.
DISTRICT_HEADERS = {"MALAKAND PROTECTED AREA"}

DISTRICTS_2023 = {"Khyber Pakhtunkhwa": 35, "Punjab": 36, "Sindh": 30, "Balochistan": 34,
                  "Islamabad": 1}


def check(units, ok):
    print()
    provs = {u["province"]: u for u in units if u["level"] == "province"}

    # 1. tehsils -> district, districts -> province, on every column
    for prov, pu in provs.items():
        ds = [u for u in units if u["province"] == prov and u["level"] == "district"]
        s = [sum(u["cells"][i] for u in ds) for i in range(NCOL)]
        good = s == pu["cells"]
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {prov}: {len(ds)} districts sum to the province "
              f"on all 9 columns ({pu['cells'][0]:,})")
        if not good:
            print("        ", s, "\n        ", pu["cells"])
        nt_bad, n_t, lone = [], 0, []
        for d in ds:
            ts = [u for u in units if u["level"] == "tehsil" and u["province"] == prov
                  and u["district"] == d["name"]]
            n_t += len(ts)
            if not ts:
                lone.append(d["name"])
                continue
            s = [sum(u["cells"][i] for u in ts) for i in range(NCOL)]
            if s != d["cells"]:
                nt_bad.append(d["name"])
        if prov != "Islamabad":
            ok &= not nt_bad
            print(f"  {'OK ' if not nt_bad else 'BAD'}    and {n_t} tehsils sum to their "
                  f"district, all columns ({len(nt_bad)} failures: {nt_bad[:5]}; "
                  f"{len(lone)} districts print no tehsil: {lone[:6]})")

    # 2. provinces against Table 4.13
    nat = [0] * 7
    for prov, want in T413.items():
        c = provs[prov]["cells"]
        got = (c[0], c[1], c[2], c[3], c[4], c[5], c[6] + c[7] + c[8])
        nat = [a + b for a, b in zip(nat, got)]
        good = got == want
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {prov} equals NCR Table 4.13 cell by cell"
              + ("" if good else f"\n        got  {got}\n        want {want}"))
    good = tuple(nat) == T413_PAKISTAN
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the five sum to Table 4.13's Pakistan row, "
          f"{T413_PAKISTAN[0]:,} (so the Punjab Muslim misprint is the report's, not ours)")
    print(f"      the census headline is {CENSUS_HEADLINE:,}; {CENSUS_HEADLINE - nat[0]:,} people "
          f"in restricted areas were counted by head only and are in no religion table")

    # 3. the portal's two records, an independent path
    for name, want in PORTAL.items():
        hit = [u for u in units if u["level"] == "district" and u["name"] == f"{name} DISTRICT"]
        good = len(hit) == 1 and tuple(hit[0]["cells"]) == want
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {name.title()} district equals the archived "
              f"census23 portal record (sources/pk.md §7a) on all 9 columns")

    # ---- what is drawn
    ds = [u for u in units if u["level"] == "district"]
    tot = sum(u["cells"][0] for u in ds)
    print(f"\n  {len(ds)} districts, {tot:,} people, {tot / len(ds):,.0f} each. National:")
    for i, cat in enumerate(CATS, start=1):
        n = sum(u["cells"][i] for u in ds)
        print(f"    {n:>12,}  {100.0 * n / tot:7.4f}%  {cat}")
    sfx = {}
    for u in units:
        if u["level"] == "tehsil":
            k = u["name"].split()[-1]
            sfx[k] = sfx.get(k, 0) + 1
    print(f"  tehsil-tier header suffixes: {sfx}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def rows_out(units):
    out = []
    for u in units:
        pid = "PK23-" + slug(u["province"])
        if u["level"] == "province":
            gid = pid
        elif u["level"] == "district":
            gid = f"{pid}/{slug(u['name'])}"
        else:
            gid = f"{pid}/{slug(u['district'])}/{slug(u['name'])}"
        note = f"level={u['level']}; province={u['province']}; file={u['file']}; page={u['page']}"
        if u["level"] == "tehsil":
            note += f"; district={u['district']}"
        for i, cat in enumerate(CATS, start=1):
            out.append({"geo_id": gid, "geo_level": u["level"], "geo_name": u["name"],
                        "source_category": cat, "count": u["cells"][i], "basis": BASIS,
                        "year": YEAR, "source_id": SOURCE_ID, "note": note})
    return out


def main():
    if "--fetch" in sys.argv:
        fetch()
    for name, *_ in FILES:
        if not os.path.exists(os.path.join(RAW, name)):
            raise SystemExit(f"{name} missing -- run: python sources/pk_2023.py --fetch")
    print("Pakistan -- 2023 census religion, PBS Table 9\n")
    units, ok = read()
    check(units, ok)
    rows = rows_out(units)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
