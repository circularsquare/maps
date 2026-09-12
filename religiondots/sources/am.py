"""Armenia — Statistical Committee (Armstat), 2022 Population Census, religion by marz.

Reads (or fetches) data/raw/am/ and writes data/normalized/am.csv.

Eleven marzes (ten provinces and the city of Yerevan), 2,932,731 people, 16 named religions
plus `Other religious groups`, `No religion` and `Refused to answer`.

**THE SWEEP CLOSED THIS COUNTRY ON THE WRONG PAGE.** sources.md §11o recorded *"armstat.am
census pages carry no religion table"*. The 2022 census results page (`?nid=944`) really does
look empty: it is a clickable GIF map of Armenia whose eleven `<area>` polygons ALL point at
the same national volume, so clicking any marz returns the same file. The marz volumes exist
and are one nid each, reachable only from the left nav.

**AND THE ENGLISH PAGES ARE THE EMPTY ONES.** `armstat.am/en/?nid=945` through `?nid=957` are
the eleven marz volumes and every one of them is a bare `<h1>` over the sentence
*"Information is not available in English"*. The same nids under `/am/` carry the nine section
archives each. So the English tree is a strict subset here, and a sweep that reads only `/en/`
sees a published census as an unpublished one. Check the office's own language before
concluding anything is missing.

**THE 2022 CENSUS IS THE NEWER ONE AND IT IS NOT THE FINER ONE FOR RELIGION.** Both 2011 and
2022 stop at marz. 2022 is taken because it is newer, has more categories, and reconciles; the
2011 volumes are the same eleven units (`?nid=533`-`543`, also Armenian-only).

**THE MARZ TABLES ARE FINER IN CATEGORIES THAN THE NATIONAL TABLE THEY SUM TO.** National
table 5.5 has thirteen named religions; the marz tables between them name fifteen, breaking
out `Protestant` (Ararat, Lori, Kotayk, Tavush) and `TM (Transcendental meditation)` (Ararat
alone) that national 5.5 folds into `Other religious groups`. Nationally those two are only
visible in table 5.7, which is the 6-and-over universe. Each marz prints only the columns it
has people in, so the header is a DIFFERENT SET OF COLUMNS IN EVERY FILE and must be read per
file rather than assumed.

**THE LABELS CARRY LINE-BREAK HYPHENS.** Armstat typeset these for a printed page, so the
same category is `Ավետարանական` in one marz and `Ավետարանա-կան` in the next, and
`Շարֆադինա-կան`, `Հեթանոսա-կան`, `Կրիշնյա գիտակ-ցության` likewise. Every label is folded
(hyphens, soft hyphens and whitespace removed) before lookup, and an unrecognised label is a
hard error rather than a dropped column.

**IT RECONCILES TO THE PERSON ON THE TOTALS AND NOT ON THE CATEGORIES, AND THAT IS CORRECT.**
Every marz's own columns sum to its own population exactly, and the eleven marz totals sum to
2,932,731, the published national total, exactly. The per-category sums do NOT match the
national table, and demanding that they should would reject a correct read: because a marz
prints only the columns it has people in, a rare answer in a marz that gives it no column sits
inside THAT MARZ's `Other religious groups`. Islam is the legible case, and it was checked by
hand rather than assumed: Yerevan 320, Shirak 120, Armavir 26 and Ararat 17 are printed and
sum to 483, and the other 32 of the country's 515 Muslims are in the seven marzes with no
Muslim column. So a small category's marz sum is a FLOOR, and the whole shortfall (169 people)
reappears in Other.

**AND THE SAME COMPARISON FINDS EIGHT PEOPLE THAT FOLDING CANNOT EXPLAIN.** Three categories
come out of the marz tables slightly LARGER than the national table: `Refused to answer` by 6,
`Evangelical` by 1, `Jehovah's witness` by 1. Folding can only make a category smaller, so
these are edits between two Armstat publications of one census. 177 people in 2.93 million,
0.0060%, bounded by `EDIT_CAP` so a real divergence would still fail the check.

**ONE FILE HAS A SECOND SHEET OF SCRATCH WORKING.** Shirak's `table 5.5 .xlsx` (note the
space before the extension) carries a 99-row `Лист1` of intermediate arithmetic beside the
published `Sheet1`. Only the first sheet is read. The English national archive has the
matching trap in a filename: `table 5.6Е..xlsx` spells its `Е` in Cyrillic.

Usage:
    python sources/am.py --fetch    12 archives, about 1 MB
    python sources/am.py            normalise from data/raw/am/
"""

import csv
import io
import os
import re
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "am")
OUT = os.path.join(ROOT, "data", "normalized", "am.csv")

SOURCE_ID = "am_census_2022"
YEAR = 2022
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
BASE = "https://www.armstat.am"

# The national volume, English. `section_N` is English, `sector_N` Armenian, `Раздел_N`
# Russian; all three are on ?nid=82&id=2623.
NATIONAL_URL = BASE + "/file/article/section_5.7z"
NATIONAL_7Z = os.path.join(RAW, "section_5_national_en.7z")

# The eleven marz volumes. Armenian only. The section-5 archive inside each is found by its
# link text rather than hard-coded, because Armstat re-uploads under a fresh numeric doc id.
SECTION_5_PREFIX = "ԲԱԺԻՆ 5"

# nid -> (ISO 3166-2 code, English name, the marz's own label in its table's stub column)
MARZES = {
    945: ("AM-ER", "Yerevan", "ք. Երևան"),
    946: ("AM-AG", "Aragatsotn", "Արագածոտնի մարզ"),
    947: ("AM-AR", "Ararat", "Արարատի մարզ"),
    948: ("AM-AV", "Armavir", "Արմավիրի մարզ"),
    949: ("AM-GR", "Gegharkunik", "Գեղարքունիքի մարզ"),
    950: ("AM-LO", "Lori", "Լոռու մարզ"),
    951: ("AM-KT", "Kotayk", "Կոտայքի մարզ"),
    952: ("AM-SH", "Shirak", "Շիրակի մարզ"),
    953: ("AM-SU", "Syunik", "Սյունիքի մարզ"),
    956: ("AM-VD", "Vayots Dzor", "Վայոց ձորի մարզ"),
    957: ("AM-TV", "Tavush", "Տավուշի մարզ"),
}

# Armstat's own English wording, taken from its national tables 5.5 and 5.7 so that the
# category strings in am.csv are the office's and not a translation of ours. `Molokai` is
# Armstat's spelling of Molokan and `Shar-fadinian` of Sharfadin; both are kept as printed.
HY_TO_EN = {
    "Հայառաքելական": "Armenian apostolic",
    "Կաթոլիկ": "Catholic",
    "Ուղղափառ": "Orthodox",
    "Նեստորական": "Nestorian",
    "Ավետարանական": "Evangelical",
    "Եհովայիվկա": "Jehovah's witness",
    "Բողոքական": "Protestant",
    "Մորմոն": "Mormon",
    "Մոլոկան": "Molokai",
    "Շարֆադինական": "Shar-fadinian",
    "Հեթանոսական": "Pagan",
    "Մահմեդական": "Islam",
    "Հուդայական": "Judaism",
    "Կրիշնյա գիտակցության կամ Հարե կրիշնյա": "Krishna consciousness or Hare Krishna",
    "Տրանսցենդետալմեդիտացիա": "TM (Transcendental meditation)",
    "Այլ": "Other religious groups",
    "Չունենկրոնականդավանանք": "No religion",
    "Հրաժարվելենպատասխանել": "Refused to answer",
}
# Read and checked against, but not emitted: they are universes, not answers.
TOTAL_HY = "Բնակչություն"                    # the marz's whole population
FOLLOWER_HY = "Ունեն կրոնական դավանանք"      # has a religious belief: the sub-total
TOTAL_CAT = "Total"                          # what the total is called in am.csv

NATIONAL = 2_932_731
EXPECTED_MARZES = 11

# Three categories come out of the marz tables slightly LARGER than the national table:
# `Refused to answer` by 6, `Evangelical` by 1 and `Jehovah's witness` by 1. Folding cannot
# make a category grow, so these are edits between two Armstat publications of one census
# rather than anything structural. Bounded here so that a real divergence would still fail.
EDIT_CAP = 10

# National table 5.5 (all ages), for the reconciliation. Read from the file rather than
# typed; these are the published figures the read is checked against.
NATIONAL_TABLE = "table 5.5E."


def _fold(s):
    """Category labels, comparable across files.

    Armstat hyphenates for the printed column width, so the same answer is `Ավետարանական`
    in one marz and `Ավետարանա-կան` in the next. Drop every hyphen, space and case
    distinction and normalise the Unicode; what is left is stable.
    """
    s = unicodedata.normalize("NFKC", str(s or ""))
    s = s.replace("­", "")                      # soft hyphen
    s = re.sub(r"[\s\-‐-―−]+", "", s)
    return s.lower()


def _stub(s):
    """The stub column, folded for the marz-name assertion.

    Yerevan writes `ք․ Երևան` with U+2024 ONE DOT LEADER rather than a full stop, and the
    marz files vary between `Վայոց ձորի մարզ` and `Վայոց Ձորի մարզ`. Same fold, plus the
    dots.
    """
    return _fold(s).replace(".", "").replace("․", "").replace("։", "")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)

    def get(url, dest, what):
        if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
            print(f"  have {what}: {os.path.basename(dest)}")
            return
        r = requests.get(url, headers=UA, timeout=300)
        r.raise_for_status()
        # §5a: HTTP 200 is not a download. Armstat serves these as text/plain.
        if r.content[:6] != b"7z\xbc\xaf\x27\x1c":
            raise SystemExit(f"{url} is not a 7z archive "
                             f"({len(r.content)} bytes, starts {r.content[:16]!r})")
        with open(dest, "wb") as fh:
            fh.write(r.content)
        print(f"  {what}: {os.path.getsize(dest):,} bytes  <- {url}")

    print("national volume (English), ?nid=82&id=2623")
    get(NATIONAL_URL, NATIONAL_7Z, "national section 5")

    print(f"\n{EXPECTED_MARZES} marz volumes (Armenian only; the /en/ pages are empty)")
    for nid, (code, name, _) in MARZES.items():
        dest = os.path.join(RAW, f"section_5_{code}.7z")
        if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
            print(f"  have {name}: {os.path.basename(dest)}")
            continue
        page = requests.get(f"{BASE}/am/?nid={nid}", headers=UA, timeout=180)
        page.raise_for_status()
        hits = [m for m in re.finditer(
            r'<a[^>]*href="(\.\./file/[^"]+\.7z)"[^>]*>(.*?)</a>', page.text, re.S | re.I)
            if re.sub(r"<[^>]+>", "", m.group(2)).strip().startswith(SECTION_5_PREFIX)]
        if len(hits) != 1:
            raise SystemExit(f"?nid={nid} ({name}): {len(hits)} links whose text starts "
                             f"{SECTION_5_PREFIX!r}, expected exactly one")
        get(hits[0].group(1).replace("../", BASE + "/"), dest, name)


def _sheets(path_7z):
    """The first worksheet of every .xlsx in a 7z, as (filename, list-of-rows).

    py7zr 1.0 has no in-memory read, only `extractall`, so each archive is unpacked once
    beside itself into `_x/` and read from there. `~$...` entries are Excel's own lock files
    and are left in the archive; Armstat shipped four of them.
    """
    import py7zr
    import openpyxl

    dest = os.path.join(RAW, "_x", os.path.splitext(os.path.basename(path_7z))[0])
    if not os.path.isdir(dest):
        with py7zr.SevenZipFile(path_7z, "r") as z:
            z.extractall(path=dest)
    out = []
    for dirpath, _, names in os.walk(dest):
        for base in sorted(names):
            if not base.lower().endswith(".xlsx") or base.startswith("~$"):
                continue
            wb = openpyxl.load_workbook(os.path.join(dirpath, base),
                                        read_only=True, data_only=True)
            # Shirak's file has a second sheet of scratch working beside the published one.
            ws = wb.worksheets[0]
            out.append((base, [list(r) for r in ws.iter_rows(values_only=True)]))
            wb.close()
    return sorted(out)


def _read_table(rows, where):
    """One 5.5 table -> {category label: count}, off its own header row.

    The header is the row whose second cell is the population column; the marz total is the
    row after it. Both are found rather than indexed, because the preamble rows differ by
    one between files.
    """
    hdr = None
    for i, row in enumerate(rows):
        if len(row) > 1 and _fold(row[1]) == _fold(TOTAL_HY):
            hdr = i
            break
        # The English national table names the same column `Population`.
        if len(row) > 1 and _fold(row[1]) == "population":
            hdr = i
            break
    if hdr is None:
        raise SystemExit(f"{where}: no header row (no cell {TOTAL_HY!r} or 'Population' "
                         "in column B)")
    # THE HEADER IS TWO ROWS AND THE TOP ONE IS NOT DECORATION. The religions sit under a
    # spanning `Religious belief` title in the lower row, but `No religion` and `Refused to
    # answer` are outside that span and are printed one row HIGHER, at the far right. Read
    # only the lower row and every marz silently loses its irreligious and its non-answers,
    # which is 66,854 people nationally and looks like a clean sub-total instead of a hole.
    above = rows[hdr - 1] if hdr else []
    header = list(rows[hdr])
    for j, cell in enumerate(above):
        if j < len(header) and (header[j] is None or not str(header[j]).strip()):
            if cell is not None and str(cell).strip():
                header[j] = cell
    body = rows[hdr + 1]
    if len(body) < 3 or body[0] in (None, ""):
        raise SystemExit(f"{where}: the row under the header has no stub label")
    cells = {}
    for j in range(1, len(header)):
        label = header[j]
        if label in (None, ""):
            continue
        v = body[j] if j < len(body) else None
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        cells[str(label)] = int(float(v))
    return str(body[0]), cells


def _resolve(cells, where):
    """Folded Armenian (or Armstat English) labels -> the office's English wording."""
    hy_by_fold = {_fold(k): v for k, v in HY_TO_EN.items()}
    en_by_fold = {_fold(v): v for v in HY_TO_EN.values()}
    out, total, follower = {}, None, None
    for label, n in cells.items():
        f = _fold(label)
        if f == _fold(TOTAL_HY) or f == "population":
            total = n
            continue
        if f == _fold(FOLLOWER_HY) or f == _fold("Follower of religious belief"):
            follower = n
            continue
        name = hy_by_fold.get(f) or en_by_fold.get(f)
        if name is None:
            raise SystemExit(f"{where}: unrecognised category {label!r} (folded {f!r}). "
                             "Add it to HY_TO_EN rather than letting it be dropped.")
        if name in out:
            raise SystemExit(f"{where}: {name!r} appears twice")
        out[name] = n
    if total is None:
        raise SystemExit(f"{where}: no population column")
    return out, total, follower


def _ethnicity_rows(rows, where):
    """National table 5.5's FIRST block: religion crossed with the twelve ethnicities.

    Not drawn, and not a geography: `geo_level` is `country_by_ethnicity` and nothing in the
    pipeline reads it (every consumer filters on an explicit level). It is carried because it
    is the only place the census says WHO is in a category, and because two of Armenia's
    interesting facts live only here — that 9,939 of the country's 31,079 Yezidis answered
    `Armenian apostolic`, and that 3,246 of them are inside `Other religious groups`.

    The sheet repeats the whole block for Man, Woman, Urban and Rural, so the block ends at
    the first stub that is one of those. Reading past it double-counts the country.
    """
    hdr = None
    for i, row in enumerate(rows):
        if len(row) > 1 and _fold(row[1]) in (_fold(TOTAL_HY), "population"):
            hdr = i
            break
    header = list(rows[hdr])
    above = rows[hdr - 1] if hdr else []
    for j, cell in enumerate(above):
        if j < len(header) and (header[j] is None or not str(header[j]).strip()):
            if cell is not None and str(cell).strip():
                header[j] = cell

    BLOCK_ENDS = {"man", "woman", "urban", "rural"}
    out, seen = [], []
    for row in rows[hdr + 2:]:                      # +2 skips the RA total row
        stub = str(row[0] or "").strip()
        if not stub:
            continue
        if _fold(stub) in BLOCK_ENDS:
            break
        cells = {str(header[j]): row[j] for j in range(1, len(header))
                 if header[j] not in (None, "") and j < len(row) and row[j] is not None}
        cats, total, _ = _resolve({k: int(float(v)) for k, v in cells.items()},
                                  f"{where} / {stub}")
        seen.append(stub)
        slug = re.sub(r"[^a-z]+", "", stub.lower())
        for cat, n in cats.items():
            out.append({"geo_id": f"AM-eth-{slug}", "geo_level": "country_by_ethnicity",
                        "geo_name": stub, "source_category": cat, "count": n,
                        "basis": BASIS, "year": YEAR, "source_id": SOURCE_ID,
                        "note": "national volume; table 5.5; ethnicity block, not a place"})
        out.append({"geo_id": f"AM-eth-{slug}", "geo_level": "country_by_ethnicity",
                    "geo_name": stub, "source_category": TOTAL_CAT, "count": total,
                    "basis": BASIS, "year": YEAR, "source_id": SOURCE_ID,
                    "note": "national volume; table 5.5; ethnicity block, not a place"})
    if len(seen) < 5:
        raise SystemExit(f"{where}: only {len(seen)} ethnicity rows ({seen})")
    got = sum(r["count"] for r in out if r["source_category"] == TOTAL_CAT)
    if got != NATIONAL:
        raise SystemExit(f"{where}: the {len(seen)} ethnicities sum to {got:,}, not "
                         f"{NATIONAL:,} -- the block boundary is wrong and the country is "
                         "being counted more than once")
    print(f"  OK  {len(seen)} ethnicities sum to {got:,}, so the block ends where it should")
    return out


def read():
    marz_rows, marz_totals, marz_follower = [], {}, {}
    for nid, (code, name, stub) in MARZES.items():
        path = os.path.join(RAW, f"section_5_{code}.7z")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} -- run with --fetch first")
        books = [b for b in _sheets(path) if "5.5" in b[0]]
        if len(books) != 1:
            raise SystemExit(f"{name}: {len(books)} table-5.5 workbooks in {path}, "
                             f"expected one ({[b[0] for b in books]})")
        base, rows = books[0]
        where = f"{name} ({base})"
        got_stub, cells = _read_table(rows, where)
        if _stub(got_stub) != _stub(stub) and _fold(name) not in _stub(got_stub):
            raise SystemExit(f"{where}: total row is labelled {got_stub!r}, expected "
                             f"{stub!r} -- the marz volumes may have been renumbered")
        cats, total, follower = _resolve(cells, where)
        marz_totals[code] = total
        marz_follower[code] = follower
        # The published population of the marz, carried so that am_geo.py's Kontur check
        # reads Armstat's own total rather than re-adding the columns it is checking.
        # EXCLUDED in am2022.py, like Georgia's.
        marz_rows.append({"geo_id": code, "geo_level": "marz", "geo_name": name,
                          "source_category": TOTAL_CAT, "count": total, "basis": BASIS,
                          "year": YEAR, "source_id": SOURCE_ID,
                          "note": f"nid={nid}; table 5.5; universe total, not a religion "
                                  "category"})
        for cat, n in cats.items():
            marz_rows.append({"geo_id": code, "geo_level": "marz", "geo_name": name,
                              "source_category": cat, "count": n, "basis": BASIS,
                              "year": YEAR, "source_id": SOURCE_ID,
                              "note": f"nid={nid}; table 5.5; marz total row"})

    # The national volume, English: the published figures the marz sums are checked against.
    books = [b for b in _sheets(NATIONAL_7Z) if NATIONAL_TABLE in b[0]]
    if len(books) != 1:
        raise SystemExit(f"{len(books)} national table-5.5 workbooks in {NATIONAL_7Z}")
    base, rows = books[0]
    _, cells = _read_table(rows, f"national ({base})")
    nat, nat_total, nat_follower = _resolve(cells, f"national ({base})")
    nat_rows = [{"geo_id": "AM", "geo_level": "country", "geo_name": "Armenia",
                 "source_category": cat, "count": n, "basis": BASIS, "year": YEAR,
                 "source_id": SOURCE_ID, "note": "national volume; table 5.5"}
                for cat, n in nat.items()]
    nat_rows.append({"geo_id": "AM", "geo_level": "country", "geo_name": "Armenia",
                     "source_category": TOTAL_CAT, "count": nat_total, "basis": BASIS,
                     "year": YEAR, "source_id": SOURCE_ID,
                     "note": "national volume; table 5.5; universe total, not a religion "
                             "category"})
    nat_rows += _ethnicity_rows(rows, f"national ({base})")

    return marz_rows, nat_rows, marz_totals, marz_follower, nat, nat_total, nat_follower


def check(marz_rows, nat_rows, marz_totals, marz_follower, nat, nat_total, nat_follower):
    ok = True

    good = len(marz_totals) == EXPECTED_MARZES
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {len(marz_totals)} marzes "
          f"(expected {EXPECTED_MARZES}: ten provinces and the city of Yerevan)")

    good = nat_total == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national volume total {nat_total:,} "
          f"(published {NATIONAL:,})")

    s = sum(marz_totals.values())
    good = s == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the eleven marz totals sum to {s:,}"
          + ("" if good else f", off by {s - NATIONAL:+,}"))

    # Every marz: its own columns must account for its own people, to the person.
    print("\n  each marz: religions + no religion + refused == the marz population")
    for code in sorted(marz_totals):
        parts = sum(r["count"] for r in marz_rows
                    if r["geo_id"] == code and r["source_category"] != TOTAL_CAT)
        good = parts == marz_totals[code]
        ok &= good
        name = next(r["geo_name"] for r in marz_rows if r["geo_id"] == code)
        print(f"    {'OK ' if good else 'BAD'} {name:<13} {marz_totals[code]:>9,}"
              + ("" if good else f"  parts {parts:,}, off by {parts - marz_totals[code]:+,}"))
        # And the printed `has a religion` sub-total must equal the named religions.
        if marz_follower[code] is not None:
            named = sum(r["count"] for r in marz_rows if r["geo_id"] == code
                        and r["source_category"] not in ("No religion", "Refused to answer",
                                                         TOTAL_CAT))
            g2 = named == marz_follower[code]
            ok &= g2
            if not g2:
                print(f"        BAD 'has a religious belief' {marz_follower[code]:,} != "
                      f"named religions {named:,}")

    # ---- category by category against the national table, and it is NOT an equality ----
    #
    # A marz table prints only the columns that marz has people in, so a rare answer in a
    # marz that gives it no column is inside THAT MARZ's `Other religious groups`. Armenia's
    # 515 Muslims are the clear case: Yerevan 320, Shirak 120, Armavir 26 and Ararat 17 are
    # printed and sum to 483, and the remaining 32 live in the seven marzes with no Muslim
    # column. So the marz sum of a small category is a FLOOR, the shortfall is in Other, and
    # demanding equality here would reject a correct read.
    #
    # What must hold: every shortfall is non-negative and adds up to Other's excess.
    OTHER = "Other religious groups"
    NAT_FOLDS_INTO_OTHER = ("Protestant", "Mormon", "TM (Transcendental meditation)")
    print("\n  each category: marz sum vs the national table. The marz sum of a rare\n"
          "  answer is a floor, because a marz without a column for it puts it in Other:")
    marz_cat = {}
    for r in marz_rows:
        if r["source_category"] == TOTAL_CAT:
            continue
        marz_cat[r["source_category"]] = marz_cat.get(r["source_category"], 0) + r["count"]

    shortfall, excess = 0, 0
    for cat in sorted(set(marz_cat) | set(nat), key=lambda c: -marz_cat.get(c, 0)):
        if cat == OTHER or cat in NAT_FOLDS_INTO_OTHER:
            continue
        got, want = marz_cat.get(cat, 0), nat.get(cat, 0)
        d = want - got
        (shortfall, excess) = (shortfall + d, excess) if d >= 0 else (shortfall, excess - d)
        flag = "OK " if abs(d) <= EDIT_CAP or d >= 0 else "BAD"
        if d < 0 and abs(d) > EDIT_CAP:
            ok = False
        print(f"    {flag} {cat:<38} {got:>9,} vs {want:>9,}   {d:+,}")

    # `Other` runs the other way: it is bigger in the marz tables by exactly what the named
    # categories are short, less the handful the national table is short.
    got = marz_cat.get(OTHER, 0) + sum(marz_cat.get(c, 0) for c in NAT_FOLDS_INTO_OTHER)
    want = nat.get(OTHER, 0)
    good = (got - want) == (shortfall - excess)
    ok &= good
    extra = ", ".join(f"{c.split(' (')[0]} {marz_cat[c]:,}"
                      for c in NAT_FOLDS_INTO_OTHER if c in marz_cat)
    print(f"    {'OK ' if good else 'BAD'} {OTHER:<38} {got:>9,} vs {want:>9,}   "
          f"{got - want:+,} == {shortfall:,} short - {excess:,} over"
          f"   (marz Other {marz_cat.get(OTHER, 0):,} + {extra})")

    # And the whole disagreement between the two publications must stay negligible.
    frac = (shortfall + excess) / NATIONAL
    good = frac < 0.0005
    ok &= good
    print(f"    {'OK ' if good else 'BAD'} the two Armstat publications differ by "
          f"{shortfall + excess:,} people in total, {frac:.4%} of the country")

    print(f"\n  {len(marz_rows):,} marz rows + {len(nat_rows)} national rows. "
          "Answers, national, from the marz tables:")
    for cat, n in sorted(marz_cat.items(), key=lambda kv: -kv[1]):
        print(f"    {n:>10,}  {100.0 * n / NATIONAL:6.3f}%  {cat}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    marz_rows, nat_rows, *rest = read()
    check(marz_rows, nat_rows, *rest)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(marz_rows + nat_rows)
    print("\nwrote", OUT, f"({len(marz_rows) + len(nat_rows):,} rows)")


if __name__ == "__main__":
    main()
