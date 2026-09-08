"""Vietnam — GSO, religion by province, from the 2009 census; 2019 for the national check.

Reads (or fetches) data/raw/vn/ and writes data/normalized/vn.csv.

**Biểu 7** of *Kết quả toàn bộ Tổng điều tra dân số và nhà ở Việt Nam năm 2009* —
*Population by urban/rural residence, sex, religion, socio-economic region and province/city*
— 13 named religions plus a not-stated row, over the 6 socio-economic regions and the **63
provinces**, 32 pages of a 901-page volume. 15,651,467 people.

THE 2019 CENSUS ASKS THE SAME QUESTION AND PUBLISHES THE ANSWER NATIONALLY ONLY. Its own
*Kết quả toàn bộ* gives religion **one page** — Biểu 3, no geography at all — while giving
ethnicity a full province × 54-group tabulation over 167 pages. So the newer census is the
better measurement of Vietnam's magnitudes and says nothing whatever about where anybody is,
and the older one is the only source of geography that exists. Both volumes are one GET from
the same host. See sources/vn.md §2 and sources.md §11i.

**WHY 2009 IS DRAWN AS IT STANDS RATHER THAN RESCALED TO 2019 — spec §3.4's escape clause.**
§3.4's Brazil case had structure and totals *at the same geography*, so the rescale was per
município and kept each place's own shape. Here the recent source has no geography at all, so
any rescale is one national factor per religion applied to all 63 provinces — which preserves
2009's spatial pattern exactly and only relabels the year. And the factors are not credible as
history: **Buddhism −32.3%, Hòa Hảo −31.4% and Cao Đài −31.2% between the two censuses**, three
unrelated traditions moving together to within one percentage point while the population grows
12%. That is an instrument change, and spreading it uniformly over every province would assert
something nobody measured. §3.4: *"Where there is no recent total to rescale to, the old figure
is used as it stands"* — the reason there is the same one, and India (2011) and Russia (2012)
are already drawn this way. The 2019 national table is emitted here at `geo_level=country` so
the comparison stays in the data and is checked on every run; countries.py reads `province`
and never sees it.

WHAT IS AND IS NOT IN THIS TABLE. Biểu 7's own `Tổng số` row is the **religious** population,
not the population: An Giang's is 2,025,015 against a census population of 2,142,709. The 70.2M
Vietnamese who answered `Không theo tôn giáo` are not in the table at all, so this source covers
**18.2% of the country** and the rest of the map is blank by construction (spec §6.12). Biểu 1
of the same volume is read for each province's population, emitted as its own universe row, so
the denominator is in the data rather than in a comment.

THE NOT-STATED ROW HAS NO CODE, AND THAT IS THE PARSER TRAP. Every religion row is
`NN <label> <9 numbers>`; `Không xác định tôn giáo - Not stated` is the same shape with the two
digits missing. A pattern that requires the code parses the whole table perfectly and is short
by exactly the national not-stated count — 30 people, which reads as rounding at this scale.
The check below reconciles each province against its own printed total, which is what makes
that visible.

Usage:
    python sources/vn.py --fetch    two GETs, 13.3 MB
    python sources/vn.py            normalise from data/raw/vn/
"""

import csv
import os
import re
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "vn")
OUT = os.path.join(ROOT, "data", "normalized", "vn.csv")

SOURCE_2009 = "vn_census_2009"
SOURCE_2019 = "vn_census_2019"
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# The office renamed from gso.gov.vn to nso.gov.vn and kept the WordPress install; the old
# host now times out on connect while the new one serves the identical /wp-content/ paths.
# sources.md §9q's sibling-host rule, fifth sighting.
URL_2009 = "https://www.nso.gov.vn/wp-content/uploads/2019/03/KQ-toan-bo.pdf"
URL_2019 = ("https://www.nso.gov.vn/wp-content/uploads/2019/12/"
            "Ket-qua-toan-bo-Tong-dieu-tra-dan-so-va-nha-o-2019.pdf")
PDF_2009 = os.path.join(RAW, "KQ-toan-bo-2009.pdf")
PDF_2019 = os.path.join(RAW, "Ket-qua-toan-bo-2019.pdf")

PROVINCES = 63
REGIONS = 6
POP_2009 = 85_846_997          # Biểu 1's own national total
POP_2019 = 96_208_984          # Biểu 3's own national total
RELIGIOUS_2009 = 15_651_467    # Biểu 7's national Tổng số

# Table titles, asserted before anything is read off a page, so a repagination fails loudly
# rather than parsing the wrong table (ke.py's rule).
T1_TITLE = re.compile(r"DÂN SỐ CHIA THEO THÀNH THỊ/NÔNG THÔN, GIỚI TÍNH, "
                      r"CÁC VÙNG KINH TẾ - XÃ HỘI VÀ TỈNH/THÀNH PHỐ")
T7_TITLE = re.compile(r"DÂN SỐ CHIA THEO THÀNH THỊ/NÔNG THÔN, GIỚI TÍNH, TÔN GIÁO, "
                      r"CÁC VÙNG KINH TẾ - XÃ HỘI VÀ TỈNH/THÀNH PHỐ")
T3_TITLE = re.compile(r"DÂN SỐ THEO TÔN GIÁO, THÀNH THỊ/NÔNG THÔN, GIỚI TÍNH")
CONT = "tiếp theo - continued"

# Biểu 7's thirteen religion rows, by the source's own two-digit code, with the label exactly
# as it prints it — misspellings and all (`Buddish`, `Bửi Sơn` on the 2019 sheet, lower-case
# `ấn` in `Tứ ấn hiếu nghĩa`), because the normalised CSV reproduces the source verbatim
# (spec §2.4) and taxonomy/vn2009.py keys on these strings.
CAT_2009 = {
    "01": "Phật Giáo - Buddish",
    "02": "Công Giáo - Catholics",
    "03": "Phật Giáo - Buddish  Hòa Hảo",
    "04": "Hồi Giáo - Muslim",
    "05": "Cao Đài",
    "06": "Minh Sư Đạo",
    "07": "Minh Lý Đạo",
    "08": "Tin Lành - Protestantism",
    "09": "Tịnh độ cư sĩ Phật hội Việt Nam",
    "10": "Đạo Tứ ấn hiếu nghĩa",
    "11": "Bửu sơn Kỳ hương",
    "12": "Ba Ha'i",
    "13": "Bà La Môn",
}
NOT_STATED = "Không xác định tôn giáo - Not stated"

# Universe rows. Neither is a religion; both are EXCLUDED in taxonomy/vn2009.py and both are
# here so that coverage is a fact in the data rather than a number in a docstring.
UNIV_RELIGIOUS = "Tổng số - Total"       # Biểu 7's own row: people WITH a religion
UNIV_POP = "Dân số - Population"         # Biểu 1: the province's whole population

# The 2019 national table's first row, and — since 2026-09-06 — a row this file COMPUTES for
# 2009 at province level as well. Biểu 7 has no such row at any geography: its universe is
# people with a religion. But Biểu 1 of the same volume publishes each province's population,
# the two tables are the same census on the same universe, and the difference is exactly the
# people who answered this. So the residual is the complement of a published partition rather
# than a model, and it is `measured` for the same reason every other figure here is.
#
# **It is emitted because the alternative is worse.** Left out, 81.8% of Vietnam is absent from
# the map and the country reads as almost empty — which spec §6.12 can label but cannot fix.
# Drawn, the people are visible as people, and the node they land on says outright that we do
# not know what they practise: see `unknown` in taxonomy/branches.py. Anita's call 2026-09-06,
# and it is spec §14.7's decision for China applied to the country that got there first.
NO_RELIGION_2019 = "Không theo tôn giáo - No religion"

# THE REGIONS ARE NOT PRINTED AROUND THEIR PROVINCES, THEY ARE PRINTED BEFORE THEM. Biểu 1
# and Biểu 7 both run national, then all six socio-economic regions, then all 63 provinces in
# code order — three flat blocks, no nesting and no marker of which province is in which
# region. So the parent/child check §12 asks for is not free here: the composition has to
# come from the published standard (GSO's six vùng kinh tế - xã hội, 2009 vintage) and is
# written out below. **It is then VERIFIED by the arithmetic rather than trusted** — the check
# requires every region's provinces to sum to its printed row in all fourteen categories, so
# a province in the wrong region fails 28 equations at once. That is the point of writing it
# out: a transcription that reconciles to the person is evidence, and one that does not is a
# loud failure instead of a silent regional map that is subtly wrong.
REGION_OF = {}
for _rid, _codes in {
    "V1": "02 04 06 08 10 11 12 14 15 17 19 20 24 25",       # Northern Midlands and Mountains
    "V2": "01 22 26 27 30 31 33 34 35 36 37",                # Red River Delta
    "V3": "38 40 42 44 45 46 48 49 51 52 54 56 58 60",       # North and South Central Coast
    "V4": "62 64 66 67 68",                                  # Central Highlands
    "V5": "70 72 74 75 77 79",                               # Southeast
    "V6": "80 82 83 84 86 87 89 91 92 93 94 95 96",          # Mekong River Delta
}.items():
    for _c in _codes.split():
        REGION_OF[_c] = _rid

NUM = r"(-|[\d\.]+)"
NINE = r"\s+".join([NUM] * 9)

RE_PROV_T1 = re.compile(r"^\s*(\d{2})\s+([^\d]+?)\s+" + NINE + r"\s*$")
RE_PROV_T7 = re.compile(r"^\s*(\d{1,2})\.\s+([^\d]+?)\s+" + NINE + r"\s*$")
RE_REGION = re.compile(r"^\s*(V\d)\.\s+(.+?)\s*$")
RE_CAT = re.compile(r"^\s*(\d{2})\s+(.+?)\s+" + NINE + r"\s*$")
RE_TOTAL = re.compile(r"^\s*Tổng số - Total\s+" + NINE + r"\s*$")
RE_NOTSTATED = re.compile(r"^\s*Không xác định tôn giáo - Not stated\s+" + NINE + r"\s*$")
RE_NATIONAL = re.compile(r"^\s*TOÀN QUỐC - ENTIRE COUNTRY")


def _n(tok):
    return 0 if tok == "-" else int(tok.replace(".", ""))


def _txt(s):
    """Collapse whitespace and normalise to NFC.

    A PDF'S TEXT LAYER NEED NOT BE IN THE SAME UNICODE NORMAL FORM AS YOUR SOURCE FILE, AND
    ONLY SOME WORDS WILL SHOW IT. The 2019 volume returns `Giá o hội Cơ đố c Phục lâm Việt
    Nam` with `á` and `ố` DECOMPOSED — a base letter plus a combining acute — while every
    other Vietnamese label on the same page comes back precomposed. The two affected
    syllables are exactly the two the typesetter split across glyph runs. Compared as bytes
    the string is unequal to a visually identical literal in taxonomy/vn2009.py, so the
    category resolves to nothing and countries.py drops it in silence; compared by eye, in a
    terminal or a diff, the two are the same word. Everything written to the normalised CSV
    goes through here so the file is canonical NFC and the mapping can be a plain literal.
    """
    return unicodedata.normalize("NFC", " ".join(str(s).split()))


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for url, path, least in ((URL_2009, PDF_2009, 3_000_000),
                             (URL_2019, PDF_2019, 8_000_000)):
        if os.path.exists(path) and os.path.getsize(path) >= least:
            print("already have", path)
            continue
        print("GET", url)
        r = requests.get(url, timeout=600, headers={"User-Agent": "religiondots/1.0"})
        r.raise_for_status()
        # §5a: HTTP 200 is not a download. Assert type and size.
        if r.content[:4] != b"%PDF":
            raise SystemExit(f"not a PDF -- starts {r.content[:16]!r}")
        if len(r.content) < least:
            raise SystemExit(f"only {len(r.content):,} bytes, expected >= {least:,}")
        with open(path, "wb") as fh:
            fh.write(r.content)
        print(f"  wrote {path} ({len(r.content):,} bytes)")


def _pages(doc):
    return [(p.extract_text() or "") for p in doc.pages]


def _span(pages, title, what):
    """The contiguous run of pages holding one table: its title page plus continuations."""
    flat = [" ".join(t.split()) for t in pages]
    starts = [i for i, t in enumerate(flat) if title.search(t)]
    if len(starts) != 1:
        raise SystemExit(f"found {len(starts)} pages carrying the {what} title, expected 1 "
                         f"(pages {[i + 1 for i in starts]}). The volume has been "
                         "re-typeset; check the title regex before trusting anything.")
    i = starts[0]
    end = i
    while end + 1 < len(pages) and CONT in flat[end + 1]:
        end += 1
    return i, end


def read_2009():
    from pypdf import PdfReader

    pages = _pages(PdfReader(PDF_2009))

    # ---- Biểu 1: population by province, the denominator ------------------------------
    lo, hi = _span(pages, T1_TITLE, "Biểu 1")
    print(f"  Biểu 1 on PDF pages {lo + 1}-{hi + 1}")
    pop, name = {}, {}
    for i in range(lo, hi + 1):
        for ln in pages[i].splitlines():
            m = RE_PROV_T1.match(ln)
            if m:
                code = m.group(1)
                if code in pop:
                    raise SystemExit(f"province {code} appears twice in Biểu 1")
                name[code] = " ".join(m.group(2).split())
                pop[code] = _n(m.group(3))
    if len(pop) != PROVINCES:
        raise SystemExit(f"Biểu 1 gave {len(pop)} provinces, expected {PROVINCES}")

    # ---- Biểu 7: religion by province -------------------------------------------------
    lo, hi = _span(pages, T7_TITLE, "Biểu 7")
    print(f"  Biểu 7 on PDF pages {lo + 1}-{hi + 1}")

    national, regions, provs = {}, {}, {}
    cur = None                       # the block being filled

    def blank():
        return {"_total": None, "rows": {}, "notstated": 0}

    for i in range(lo, hi + 1):
        for ln in pages[i].splitlines():
            if RE_NATIONAL.match(ln):
                national.update(blank())
                cur = national
                continue
            m = RE_REGION.match(ln)
            if m and not RE_CAT.match(ln):
                regions[m.group(1)] = blank()
                cur = regions[m.group(1)]
                continue
            m = RE_PROV_T7.match(ln)
            if m:
                code = f"{int(m.group(1)):02d}"
                if code in provs:
                    raise SystemExit(f"province {code} appears twice in Biểu 7")
                provs[code] = blank()
                provs[code]["_total"] = _n(m.group(3))
                cur = provs[code]
                continue
            m = RE_TOTAL.match(ln)
            if m and cur is not None:
                cur["_total"] = _n(m.group(1))
                continue
            m = RE_NOTSTATED.match(ln)
            if m and cur is not None:
                cur["notstated"] = _n(m.group(1))
                continue
            m = RE_CAT.match(ln)
            if m and cur is not None:
                code, label = m.group(1), " ".join(m.group(2).split())
                if code not in CAT_2009:
                    raise SystemExit(f"unknown religion code {code!r} ({label!r}) on PDF "
                                     f"page {i + 1}")
                want = CAT_2009[code]
                if " ".join(label.split()) != " ".join(want.split()):
                    raise SystemExit(f"code {code} is labelled {label!r} on PDF page "
                                     f"{i + 1}, expected {want!r}")
                cur["rows"][code] = _n(m.group(3))

    if len(provs) != PROVINCES:
        raise SystemExit(f"Biểu 7 gave {len(provs)} provinces, expected {PROVINCES}")
    if len(regions) != REGIONS:
        raise SystemExit(f"Biểu 7 gave {len(regions)} regions, expected {REGIONS}")
    if set(provs) != set(pop):
        raise SystemExit("Biểu 1 and Biểu 7 disagree about which provinces exist:\n"
                         f"  only in Biểu 1: {sorted(set(pop) - set(provs))}\n"
                         f"  only in Biểu 7: {sorted(set(provs) - set(pop))}")
    return pop, name, national, regions, provs


# --- the 2019 volume, and why it needs a different reader ----------------------------------
# THE TWO VOLUMES OF ONE SERIES USE DIFFERENT THOUSANDS SEPARATORS. 2009 writes `85.846.997`
# and 2019 writes `96 208 984` — same office, same publication name, same layout, and the
# space-separated one CANNOT be parsed from the text layer, because the digit groups are
# indistinguishable from separate numbers. `Tôn giáo Baha'i 2 153 1 089 1 064 841 419 422 ...`
# has a second valid reading in which 841, 419 and 422 are three values rather than one, and
# an anchored nine-number regex finds it. So 2019 is read by GEOMETRY: the nine columns are
# right-aligned on a fixed grid, digit groups inside one number sit 2.6pt apart and adjacent
# columns 11.8pt apart, and the grid is calibrated off the national row rather than hard-coded.
# 2009 keeps the plain text reader, which is exact there and simpler.
COL_GAP = 6.0        # pt; within-number 2.6, between-column >= 11.8
# pt; everything left of this is label. The split has to be exact in both directions: the
# leftmost digit group on the page starts at 282 and the rightmost label word — `đạo`, the
# last syllable of `Giáo hội Phật đường Nam Tông Minh Sư đạo` — starts at 253, so a cutoff
# of 250 silently truncates that one category's name and 275 does not.
LABEL_X = 275.0


def _fitz_rows(page):
    """Cluster a page's words into visual rows, each sorted left to right."""
    out = []
    for w in sorted(page.get_text("words"), key=lambda w: (w[1], w[0])):
        yc = (w[1] + w[3]) / 2.0
        for r in out:
            if abs(r["y"] - yc) < 4.0:
                r["w"].append(w)
                break
        else:
            out.append({"y": yc, "w": [w]})
    for r in out:
        r["w"].sort(key=lambda w: w[0])
    return sorted(out, key=lambda r: r["y"])


def _numbers(row):
    """[(value, RIGHT edge of the number)] for the numeric part of a row.

    The right edge, not the left: the columns are right-aligned, so a two-digit cell starts
    5pt further right than a three-digit one in the same column (`36` at 545 against `105`
    at 540) and a left-edge grid misses it.
    """
    words = [w for w in row["w"] if w[0] >= LABEL_X and re.fullmatch(r"\d+", w[4])]
    out, cur, prev = [], [], None
    for w in words:
        if prev is not None and w[0] - prev[2] > COL_GAP:
            out.append((int("".join(t[4] for t in cur)), cur[-1][2]))
            cur = []
        cur.append(w)
        prev = w
    if cur:
        out.append((int("".join(t[4] for t in cur)), cur[-1][2]))
    return out


def read_2019():
    """Biểu 3 of the 2019 volume: the national table, for the check and the note."""
    import fitz

    doc = fitz.open(PDF_2019)
    pages = [(p.get_text() or "") for p in doc]
    lo, hi = _span(pages, T3_TITLE, "Biểu 3")
    print(f"  Biểu 3 (2019) on PDF pages {lo + 1}-{hi + 1}")
    if hi != lo:
        raise SystemExit(f"Biểu 3 spans pages {lo + 1}-{hi + 1}; it is one page and the "
                         "reader assumes so")

    rows = _fitz_rows(doc[lo])

    # Calibrate the column grid on the national row, whose nine values are known.
    anchors = None
    for r in rows:
        if any(w[4] == "QUỐC" for w in r["w"]):
            nums = _numbers(r)
            if len(nums) != 9:
                raise SystemExit(f"the national row parsed as {len(nums)} values, not 9: "
                                 f"{[v for v, _ in nums]}")
            if nums[0][0] != POP_2019:
                raise SystemExit(f"the national row's first value is {nums[0][0]:,}, "
                                 f"expected {POP_2019:,}")
            anchors = [x for _, x in nums]
            break
    if anchors is None:
        raise SystemExit("no TOÀN QUỐC row found on the 2019 religion page")
    step = [round(b - a, 1) for a, b in zip(anchors, anchors[1:])]
    if len(set(step)) != 1:
        raise SystemExit(f"the nine columns are not evenly spaced: {step}. The page has been "
                         "re-typeset and the geometric reader cannot be trusted.")
    print(f"    column grid calibrated: 9 columns {step[0]}pt apart")

    def place(r):
        """Row -> nine values, by which anchor each number's last group lands on."""
        cols = [None] * 9
        for v, x in _numbers(r):
            hit = [i for i, a in enumerate(anchors) if abs(x - a) < 3.0]
            if len(hit) != 1:
                raise SystemExit(f"a value {v:,} at x={x:.0f} matches {len(hit)} of the "
                                 "nine columns; the grid is wrong")
            if cols[hit[0]] is not None:
                raise SystemExit(f"two values in column {hit[0]} of one row: "
                                 f"{cols[hit[0]]:,} and {v:,}")
            cols[hit[0]] = v
        return cols

    def label_of(r):
        return " ".join(w[4] for w in r["w"]
                        if w[0] < LABEL_X and not re.fullmatch(r"\d+", w[4]))

    # Everything above the column header is title and units. `Biểu - Table 3` puts a bare
    # `3` in the numeric half of the page and would otherwise read as a one-value data row.
    head = [r["y"] for r in rows if any(w[4] == "Female" for w in r["w"])]
    if not head:
        raise SystemExit("no `Female` column header found; the page layout has changed")
    body = max(head) + 1.0

    # A record is a row carrying all nine values. THE LDS ROW'S LABEL WRAPS AROUND ITS
    # FIGURES — `Giáo hội Các thành hữu Ngày sau của` above them and `Chúa Giê su Ky tô Việt
    # Nam (Mormon)` below — so a label fragment can belong to the record before it or the one
    # after it, and taking "the one after" for all of them steals the closing fragment for
    # the next category and the opening fragment for the previous one. Each fragment goes to
    # the NEAREST record by y; the wrapped label's two halves are 5pt from their own figures
    # and 18pt from their neighbours', so the assignment is not close.
    body_rows = [(i, r) for i, r in enumerate(rows) if r["y"] > body]
    recs = [i for i, r in body_rows if _numbers(r)]
    frags = [i for i, r in body_rows if not _numbers(r) and label_of(r)]
    attached = {i: [] for i in recs}
    for f in frags:
        near = min(recs, key=lambda i: (abs(rows[i]["y"] - rows[f]["y"]), i))
        attached[near].append(f)

    out, seen_total = {}, None
    for i in recs:
        r = rows[i]
        nums = _numbers(r)
        if len(nums) != 9:
            raise SystemExit(f"row {label_of(r)!r} parsed as {len(nums)} values, not 9")
        cols = place(r)
        parts = [label_of(rows[j]) for j in sorted(attached[i] + [i], key=lambda j: rows[j]["y"])]
        label = " ".join(" ".join(parts).split())
        if not label:
            raise SystemExit(f"a row of nine values at y={r['y']:.0f} has no label")
        if label.startswith("TOÀN QUỐC"):
            seen_total = cols[0]
            continue
        # Total = Male + Female and Total = Urban + Rural, in each of the three blocks.
        # 34 identities over the table, and any misplaced digit group breaks one.
        for a, b, c in ((0, 1, 2), (3, 4, 5), (6, 7, 8)):
            if cols[a] != cols[b] + cols[c]:
                raise SystemExit(f"{label!r}: {cols[a]:,} != {cols[b]:,} + {cols[c]:,}")
        if cols[0] != cols[3] + cols[6]:
            raise SystemExit(f"{label!r}: total {cols[0]:,} != urban {cols[3]:,} + rural "
                             f"{cols[6]:,}")
        if label in out:
            raise SystemExit(f"row {label!r} appears twice")
        out[label] = cols[0]
    doc.close()

    if seen_total != POP_2019:
        raise SystemExit(f"2019 national total parsed as {seen_total}, expected {POP_2019:,}")
    if len(out) != 17:
        raise SystemExit(f"2019 table gave {len(out)} rows, expected 17 "
                         f"(no religion + 16 religions):\n  " + "\n  ".join(sorted(out)))
    return out, seen_total


def build(pop, name, national, provs, nat_2019, total_2019):
    rows = []

    def add(gid, level, gname, cat, n, year, sid, note):
        rows.append({"geo_id": gid, "geo_level": level, "geo_name": _txt(gname),
                     "source_category": _txt(cat), "count": n, "basis": BASIS,
                     "year": year, "source_id": sid, "note": note})

    for code in sorted(provs):
        blk = provs[code]
        note = "level=province"
        for cc in sorted(CAT_2009):
            add(code, "province", name[code], CAT_2009[cc], blk["rows"].get(cc, 0),
                2009, SOURCE_2009, note)
        add(code, "province", name[code], NOT_STATED, blk["notstated"], 2009, SOURCE_2009,
            note + "; religion not stated, spec §3.5")
        add(code, "province", name[code], UNIV_RELIGIOUS, blk["_total"], 2009, SOURCE_2009,
            note + "; Biểu 7's own total, i.e. people WITH a religion -- not the population")
        add(code, "province", name[code], UNIV_POP, pop[code], 2009, SOURCE_2009,
            note + "; Biểu 1 census population, the denominator -- not a religion category")
        add(code, "province", name[code], NO_RELIGION_2019,
            pop[code] - blk["_total"], 2009, SOURCE_2009,
            note + "; COMPUTED, not a Biểu 7 row: Biểu 1's population minus Biểu 7's "
                   "religious total, which is exactly the people who answered this")

    # The 2009 national row, and the 2019 national table beside it. countries.py reads
    # `province` only, so nothing here is drawn; it is here to be checked and cited.
    for cc in sorted(CAT_2009):
        add("0", "country", "Việt Nam", CAT_2009[cc], national["rows"][cc], 2009,
            SOURCE_2009, "level=country")
    add("0", "country", "Việt Nam", NOT_STATED, national["notstated"], 2009, SOURCE_2009,
        "level=country; religion not stated")
    add("0", "country", "Việt Nam", UNIV_RELIGIOUS, national["_total"], 2009, SOURCE_2009,
        "level=country; people WITH a religion")
    add("0", "country", "Việt Nam", NO_RELIGION_2019, POP_2009 - national["_total"], 2009,
        SOURCE_2009, "level=country; COMPUTED — Biểu 1's population minus Biểu 7's total")
    add("0", "country", "Việt Nam", UNIV_POP, POP_2009, 2009, SOURCE_2009,
        "level=country; census population")
    for label, n in sorted(nat_2019.items()):
        add("0", "country", "Việt Nam", label, n, 2019, SOURCE_2019,
            "level=country; Biểu 3 of the 2019 volume, national only -- there is no 2019 "
            "religion table at any geography")
    add("0", "country", "Việt Nam", UNIV_POP, total_2019, 2019, SOURCE_2019,
        "level=country; 2019 census population")
    return rows


def check(pop, name, national, regions, provs, nat_2019):
    ok = True

    def flag(good, msg):
        nonlocal ok
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {msg}")
        return good

    flag(len(provs) == PROVINCES, f"{len(provs)} provinces (expected {PROVINCES})")
    flag(sum(pop.values()) == POP_2009,
         f"Biểu 1's provinces sum to {sum(pop.values()):,} (census {POP_2009:,})")
    flag(national["_total"] == RELIGIOUS_2009,
         f"Biểu 7's national total is {national['_total']:,} (expected {RELIGIOUS_2009:,})")

    # 1. every province's own rows sum to its own printed total. This is the check that
    #    catches a dropped row, and it is what makes the un-coded not-stated row visible.
    bad = []
    for code, blk in provs.items():
        s = sum(blk["rows"].values()) + blk["notstated"]
        if s != blk["_total"]:
            bad.append((code, name[code], s, blk["_total"]))
    flag(not bad, f"all {PROVINCES} provinces' categories sum to their own printed total "
                  f"({len(bad)} failures)")
    for c, nm, s, t in bad[:6]:
        print(f"        {c} {nm}: rows {s:,} vs printed {t:,} (diff {s - t:+,})")

    # 2. the provinces sum to the national row, CATEGORY BY CATEGORY. A per-category check,
    #    not a per-total one: a swapped pair of religions passes the total and fails this.
    bad = []
    for cc in sorted(CAT_2009):
        s = sum(p["rows"].get(cc, 0) for p in provs.values())
        if s != national["rows"][cc]:
            bad.append((CAT_2009[cc], s, national["rows"][cc]))
    s = sum(p["notstated"] for p in provs.values())
    if s != national["notstated"]:
        bad.append((NOT_STATED, s, national["notstated"]))
    flag(not bad, f"the {PROVINCES} provinces sum to the national row in all "
                  f"{len(CAT_2009) + 1} categories ({len(bad)} failures)")
    for c, s, n in bad[:6]:
        print(f"        {c}: {s:,} vs {n:,}")

    # 3. THE PARENT/CHILD CHECK §12 demands, and it is free here: each socio-economic
    #    region's provinces must sum to the region's own row, in every category. This is
    #    what would catch a province block attached to the wrong region, or a missing one.
    bad = []
    for rid, reg in sorted(regions.items()):
        kids = [c for c in provs if REGION_OF.get(c) == rid]
        for cc in sorted(CAT_2009):
            s = sum(provs[c]["rows"].get(cc, 0) for c in kids)
            if s != reg["rows"][cc]:
                bad.append((rid, CAT_2009[cc], s, reg["rows"][cc]))
        s = sum(provs[c]["_total"] for c in kids)
        if s != reg["_total"]:
            bad.append((rid, "TOTAL", s, reg["_total"]))
    sizes = {rid: sum(1 for c in provs if REGION_OF.get(c) == rid) for rid in regions}
    flag(not bad, f"every socio-economic region's provinces sum to it, category by "
                  f"category ({len(bad)} failures); sizes " +
                  " ".join(f"{k}={v}" for k, v in sorted(sizes.items())))
    for rid, c, s, n in bad[:6]:
        print(f"        {rid} {c}: {s:,} vs {n:,}")
    flag(sum(sizes.values()) == PROVINCES,
         f"the six regions between them hold all {PROVINCES} provinces")

    # 4. nobody has more religious people than people. This is also what makes the computed
    #    `no religion` residual non-negative in every province, so it is asserted before the
    #    residual is emitted rather than after.
    bad = [(c, name[c], provs[c]["_total"], pop[c])
           for c in provs if provs[c]["_total"] > pop[c]]
    flag(not bad, f"no province's religious total exceeds its population ({len(bad)} do)")
    for c, nm, r, p in bad[:6]:
        print(f"        {c} {nm}: {r:,} religious vs {p:,} people")

    # 4a. THE RESIDUAL IS A PARTITION, WHICH IS THE WHOLE LICENCE FOR DRAWING IT. Biểu 7's
    #     religions plus the computed residual must be the population, in every province and
    #     nationally, to the person — if it were not, the residual would be a model rather
    #     than the complement of a published one.
    res = {c: pop[c] - provs[c]["_total"] for c in provs}
    bad = [(c, name[c]) for c in provs
           if sum(provs[c]["rows"].values()) + provs[c]["notstated"] + res[c] != pop[c]]
    flag(not bad, f"religions + not-stated + residual = population in all {PROVINCES} "
                  f"provinces ({len(bad)} failures)")
    nat_res = POP_2009 - national["_total"]
    flag(sum(res.values()) == nat_res,
         f"the province residuals sum to {sum(res.values()):,} (national {nat_res:,})")
    flag(min(res.values()) >= 0, f"every province's residual is non-negative "
                                 f"(smallest {min(res.values()):,})")

    # 5. the 2019 table, which is a different volume, a different year and a different
    #    layout, and is not determined by anything above.
    s = sum(nat_2019.values())
    flag(s == POP_2019, f"the 2019 table's 17 rows sum to {s:,} (census {POP_2019:,})")

    print(f"\n  Vietnam 2009, and 2019 beside it. 2009 shares are of the {POP_2009:,} "
          f"population,\n  not of the religious total, because that is what the map's "
          "blank means:")
    print(f"    {'2009':>11} {'%':>6}   {'2019':>11} {'%':>6}   category")
    n19 = dict(nat_2019)
    pairs = [
        ("01", "Phật Giáo - Buddish"), ("02", "Công giáo - Catholics"),
        ("03", "Phật giáo Hòa Hảo - Buddish Hòa Hảo"), ("08", "Tin lành - Protestantism"),
        ("05", "Cao Đài"), ("04", "Hồi giáo - Muslim"), ("13", "Chăm Bà la môn"),
        ("10", "Đạo Tứ Ân Hiếu nghĩa"), ("11", "Bửi Sơn Kỳ hương"),
        ("09", "Tịnh độ Cư sỹ Phật hội Việt Nam"), ("12", "Tôn giáo Baha'i"),
        ("06", "Giáo hội Phật đường Nam Tông Minh Sư đạo"),
        ("07", "Hội thánh Minh lý đạo - Tam Tông Miếu"),
    ]
    for cc, lbl19 in pairs:
        a = national["rows"][cc]
        b = n19.get(lbl19)
        bs = f"{b:>11,} {100.0 * b / POP_2019:6.2f}" if b is not None else f"{'—':>11} {'':>6}"
        print(f"    {a:>11,} {100.0 * a / POP_2009:6.2f}   {bs}   {CAT_2009[cc]}")
    seen = {lbl for _, lbl in pairs} | {NO_RELIGION_2019}
    for lbl in sorted(set(n19) - seen):
        print(f"    {'—':>11} {'':>6}   {n19[lbl]:>11,} {100.0 * n19[lbl] / POP_2019:6.2f}"
              f"   {lbl}   (2019 only)")
    print(f"    {'-' * 11}        {'-' * 11}")
    print(f"    {national['_total']:>11,} {100.0 * national['_total'] / POP_2009:6.2f}   "
          f"{POP_2019 - n19[NO_RELIGION_2019]:>11,} "
          f"{100.0 * (POP_2019 - n19[NO_RELIGION_2019]) / POP_2019:6.2f}   with a religion")
    print(f"    {POP_2009 - national['_total']:>11,} "
          f"{100.0 * (POP_2009 - national['_total']) / POP_2009:6.2f}   "
          f"{n19[NO_RELIGION_2019]:>11,} "
          f"{100.0 * n19[NO_RELIGION_2019] / POP_2019:6.2f}   NOT in Biểu 7 at any geography")

    print("\n  The three that move together, and the reason 2009 is not rescaled to 2019:")
    for cc, lbl19 in (("01", "Phật Giáo - Buddish"),
                      ("03", "Phật giáo Hòa Hảo - Buddish Hòa Hảo"),
                      ("05", "Cao Đài")):
        a, b = national["rows"][cc], n19[lbl19]
        print(f"    {CAT_2009[cc]:<34} {a:>10,} -> {b:>10,}  {100.0 * (b - a) / a:+6.1f}%")
    print(f"    {'population':<34} {POP_2009:>10,} -> {POP_2019:>10,}  "
          f"{100.0 * (POP_2019 - POP_2009) / POP_2009:+6.1f}%")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    for p in (PDF_2009, PDF_2019):
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing -- run: python sources/vn.py --fetch")

    pop, name, national, regions, provs = read_2009()
    nat_2019, total_2019 = read_2019()
    check(pop, name, national, regions, provs, nat_2019)

    rows = build(pop, name, national, provs, nat_2019, total_2019)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
