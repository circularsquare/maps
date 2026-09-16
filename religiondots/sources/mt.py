"""Malta: Census of Population and Housing 2021, Final Report Volume 1, Chapter 5.

Reads data/raw/mt/Census-of-Population-2021-volume1-final.pdf and writes

    data/normalized/mt.csv          ten answers by the 68 localities, geo_level `locality`

`sources/mt_geo.py` builds the units and the placement layer; `taxonomy/mt2021.py` is the
mapping; `sources/mt.md` is the write-up.

## THE TABLE

National Statistics Office (NSO), *Census of Population and Housing 2021: Final Report:
Population, migration and other social characteristics (Volume 1)*, 175 pages. Chapter 5,
*Religious Affiliation*: **Table 5.3**, *Population aged 15 and over by religious affiliation
and locality*, printed pp.162-164, ten answers by the 68 localities (the census's LAU list,
grouped into six districts). Malta's first census to ask religion. The question was put to
residents aged 15 and over (p.7), "irrespective of the level of religious attendance or
observance, or formal membership" (p.159). The ten answers sum to the total in every row:
there is no not-stated column anywhere in the chapter.

## THE DOWNLOAD

`nso.gov.mt` answers 403 to every non-browser client (curl with a browser user agent and
WebFetch, 2026-09-15). Anita downloaded the PDF in a browser (ask 027), so `--fetch` only
explains that. The file is pinned by size, page count and trailer.

## THE CHECKS

The district and national rows of Table 5.2 (p.161) are transcribed below as constants; the
68 locality rows are not, because a hand transcription of a born-digital text layer would only
repeat the parse. They are checked by arithmetic against three other printings instead:
  1. Table 5.3's locality rows close (ten answers = total) and sum to its own district rows
     in all eleven columns; its district rows equal Table 5.2's, which equal the constants.
  2. Table 5.2's persons = males + females; Table 5.1's age rows sum to its total, and its
     total row is the national row (so the universe starts at 15).
  3. Table 5.4 (citizenship): Maltese + non-Maltese = the national row, and the chapter's
     sentence "Almost a third of those with no religious affiliation (7,254) were Maltese
     citizens" matches it.
  4. **Every locality's religion total equals its population aged 15 and over in Table 1.5**
     (population by single year of age and locality, one page per locality), and Table 1.5's
     totals equal Table 1.2's. So nobody aged 15 or over is missing from the religion table,
     and the under-15s, 67,816 people, 13.1% of residents, are the whole of the gap.

Usage:
    python sources/mt.py --fetch    says where the file comes from (a browser download)
    python sources/mt.py            parse, check and write data/normalized/mt.csv
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
RAW = os.path.join(ROOT, "data", "raw", "mt")
OUT = os.path.join(ROOT, "data", "normalized", "mt.csv")
sys.path.insert(0, HERE)

import micro                                                   # noqa: E402  COLUMNS

SOURCE_ID = "nso_census2021_vol1_t5_3"
YEAR = 2021
BASIS = "self_id"
COLUMNS = micro.COLUMNS

URL = "https://nso.gov.mt/wp-content/uploads/Census-of-Population-2021-volume1-final.pdf"
PDF = os.path.join(RAW, "Census-of-Population-2021-volume1-final.pdf")
SIZE = 6_820_290
PAGES = 175

# 0-based page indices; printed page = index + 1 throughout this volume.
PAGE_INTRO = 6          # p.7, "asked to respondents aged 15 and over"
PAGES_T12 = (22, 23)    # pp.23-24, total population by sex, average age and locality
T15_RANGE = range(24, 158)   # Table 1.5 is found by its caption inside this range
PAGE_CH5 = 158          # p.159, chapter text
PAGE_T51 = 159          # p.160
PAGE_T52 = 160          # p.161
PAGES_T53 = (161, 162, 163)  # pp.162-164
PAGE_T54 = 164          # p.165
PAGE_DEF = 173          # p.174, definition of religious affiliation

CATS = ["Roman Catholicism", "Islam", "Orthodoxy", "Hinduism", "Church of England",
        "Protestantism", "Buddhism", "Judaism", "Other religious groups",
        "No religious affiliation"]

# The six districts and their localities, in the order Tables 1.2 and 5.3 print them. Hyphens
# are U+2010 on the page and ASCII here (lines_of() folds them).
DISTRICTS = {
    "Southern Harbour": [
        "Bormla", "Floriana", "Ħal Luqa", "Ħal Tarxien", "Ħaż-Żabbar", "Il-Birgu", "Il-Fgura",
        "Il-Kalkara", "Il-Marsa", "Ix-Xgħajra", "L-Isla", "Raħal Ġdid", "Santa Luċija",
        "Valletta"],
    "Northern Harbour": [
        "Birkirkara", "Ħal Qormi", "Il-Gżira", "Il-Ħamrun", "Is-Swieqi", "L-Imsida", "Pembroke",
        "San Ġiljan", "San Ġwann", "Santa Venera", "Ta' Xbiex", "Tal-Pieta'", "Tas-Sliema"],
    "South Eastern": [
        "Birżebbuġa", "Ħal Għaxaq", "Ħal Kirkop", "Ħal Safi", "Il-Gudja", "Il-Qrendi",
        "Iż-Żejtun", "Iż-Żurrieq", "L-Imqabba", "Marsaskala", "Marsaxlokk"],
    "Western": [
        "Ħad-Dingli", "Ħal Balzan", "Ħal Lija", "Ħ'Attard", "Ħaż-Żebbuġ", "Ir-Rabat",
        "Is-Siġġiewi", "L-Iklin", "L-Imdina", "L-Imtarfa"],
    "Northern": [
        "Ħal Għargħur", "Il-Mellieħa", "Il-Mosta", "In-Naxxar", "L-Imġarr",
        "San Pawl Il-Baħar"],
    "Gozo and Comino": [
        "Għajnsielem and Comino", "Il-Fontana", "Il-Munxar", "Il-Qala", "In-Nadur",
        "Ir-Rabat, Għawdex", "Ix-Xagħra", "Ix-Xewkija", "Iż-Żebbuġ", "L-Għarb", "L-Għasri",
        "San Lawrenz", "Ta' Kerċem", "Ta' Sannat"],
}
LOCALITIES = [(d, loc) for d, locs in DISTRICTS.items() for loc in locs]
N_LOCALITIES = 68
ISLANDS = {"Malta": ["Southern Harbour", "Northern Harbour", "South Eastern", "Western",
                     "Northern"],
           "Gozo and Comino": ["Gozo and Comino"]}
# The row sequence both locality tables print: nation, two islands, then each district row
# followed by its localities.
SEQUENCE = (["MALTA", "Malta", "Gozo and Comino"]
            + [x for d, locs in DISTRICTS.items() for x in [d] + locs])

# Table 5.2 (p.161), persons, transcribed: ten answers in CATS order, then total.
T52 = {
    "MALTA": (373304, 17454, 16457, 6411, 5706, 4516, 2495, 1249, 911, 23243, 451746),
    "Malta": (344174, 16709, 15865, 6184, 4757, 4153, 2080, 1178, 846, 21504, 417450),
    "Gozo and Comino": (29130, 745, 592, 227, 949, 363, 415, 71, 65, 1739, 34296),
    "Southern Harbour": (67175, 2892, 943, 749, 376, 370, 212, 107, 113, 2173, 75110),
    "Northern Harbour": (105309, 7779, 6904, 3224, 1582, 2199, 1039, 623, 398, 9895, 138952),
    "South Eastern": (58448, 2424, 1453, 395, 560, 423, 236, 147, 94, 2452, 66632),
    "Western": (51695, 874, 752, 259, 357, 256, 102, 59, 50, 1898, 56302),
    "Northern": (61547, 2740, 5813, 1557, 1882, 905, 491, 242, 191, 5086, 80454),
}
# Table 1.2 (pp.23-24), total population, transcribed for the nation, islands and districts.
T12_TOTALS = {"MALTA": 519562, "Malta": 480275, "Gozo and Comino": 39287,
              "Southern Harbour": 86009, "Northern Harbour": 157297, "South Eastern": 77948,
              "Western": 65266, "Northern": 93755}
# Table 1.5 heads Gozo's Żebbuġ page (p.73) `Iż-Żebbuġ, Għawdex`, the way Table 1.1 and the
# prose on p.15 name it; Tables 1.2, 1.10 and 5.3 print `Iż-Żebbuġ`. Malta's Żebbuġ is
# `Ħaż-Żebbuġ` everywhere, so there is no twin to confuse, and the page's totals then match.
T15_ALIAS = {"Iż-Żebbuġ, Għawdex": "Iż-Żebbuġ"}
RESIDENTS = 519_562
AGED_15_PLUS = 451_746
NO_RELIGION_MALTESE = 7_254

SPACES = dict.fromkeys([0x00A0, 0x2007, 0x2008, 0x2009, 0x202F, 0x205F], " ")
COUNT = re.compile(r"\d{1,3}(?:,\d{3})*|-")
DECIMAL = re.compile(r"\d{1,2}\.\d")


def despace(s):
    """Copied from sources/gw.py (not shared yet)."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s)).translate(SPACES)).strip()


def num(tok):
    if not COUNT.fullmatch(tok):
        raise ValueError(f"not a count: {tok!r}")
    return 0 if tok == "-" else int(tok.replace(",", ""))


def lines_of(doc, pno):
    """The page's text-layer lines, NFC, spaces folded, U+2010/U+2011 hyphens made ASCII."""
    out = []
    for ln in doc.load_page(pno).get_text().split("\n"):
        s = despace(ln).replace("‐", "-").replace("‑", "-")
        if s:
            out.append(s)
    return out


def page_text(doc, pno):
    return " ".join(lines_of(doc, pno))


def read_rows(lines, n, is_value, where):
    """[(label, [n tokens])] from a stream of label line, n value lines, label line, ..."""
    out, i = [], 0
    while i < len(lines):
        lab = lines[i]
        vals = lines[i + 1:i + 1 + n]
        if is_value(lab) or len(vals) < n or not all(is_value(v) for v in vals):
            raise SystemExit(f"{where}: bad row at line {i}: {lab!r} {vals}")
        out.append((lab, vals))
        i += 1 + n
    return out


def _after_header(lines, caption, where):
    if not caption in lines[0]:
        raise SystemExit(f"{where}: page does not open with {caption!r}: {lines[0]!r}")
    return lines.index("Total") + 1         # the column header's last label


def _strip_page_number(lines, pno, where):
    if lines[-1] != str(pno + 1):
        raise SystemExit(f"{where}: last line {lines[-1]!r} is not printed page {pno + 1}")
    return lines[:-1]


def read_locality_table(doc):
    """Table 5.3 -> {label: 11-tuple}, labels in SEQUENCE order (asserted)."""
    rows = []
    for pno in PAGES_T53:
        ls = lines_of(doc, pno)
        start = _after_header(ls, "TABLE 5.3. Population aged 15 and over by religious "
                                  "affiliation and locality", f"p.{pno + 1}")
        body = _strip_page_number(ls[start:], pno, f"p.{pno + 1}")
        rows += read_rows(body, 11, COUNT.fullmatch, f"Table 5.3 p.{pno + 1}")
    labels = [r[0] for r in rows]
    if labels != SEQUENCE:
        diff = [(i, a, b) for i, (a, b) in enumerate(zip(labels, SEQUENCE)) if a != b][:5]
        raise SystemExit(f"Table 5.3: {len(labels)} rows, expected {len(SEQUENCE)}; first "
                         f"differences {diff}")
    return rows


def read_t52(doc):
    """Table 5.2 -> [(label, 11-tuple)] for all 24 rows (nation, islands, Total/Males/Females,
    then each district and its Males and Females)."""
    ls = lines_of(doc, PAGE_T52)
    start = _after_header(ls, "TABLE 5.2. Population aged 15 and over by religious affiliation, "
                              "district and sex", "p.161")
    body = _strip_page_number(ls[start:], PAGE_T52, "p.161")
    return [(lab, tuple(map(num, v))) for lab, v in read_rows(body, 11, COUNT.fullmatch,
                                                              "Table 5.2")]


def read_t51(doc):
    """Table 5.1 -> [(label, 11-tuple)]: Total, nine age rows, Males and its nine, Females."""
    ls = lines_of(doc, PAGE_T51)
    start = _after_header(ls, "TABLE 5.1. Population aged 15 and over by religious affiliation, "
                              "sex and age", "p.160")
    body = _strip_page_number(ls[start:], PAGE_T51, "p.160")
    return [(lab, tuple(map(num, v))) for lab, v in read_rows(body, 11, COUNT.fullmatch,
                                                              "Table 5.1")]


def read_t54(doc):
    """Table 5.4 -> three blocks of 11 rows x (males, females, total), in print order."""
    ls = lines_of(doc, PAGE_T54)
    start = _after_header(ls, "TABLE 5.4. Population aged 15 and over by type of citizenship, "
                              "sex and religious affiliation", "p.165")
    body = ls[start:start + 33 * 4]
    tail = ls[start + 33 * 4:]
    if tail != ["Total", "Maltese", "Non-Maltese", str(PAGE_T54 + 1)]:
        raise SystemExit(f"Table 5.4: unexpected tail {tail}")
    rows = [(lab, tuple(map(num, v))) for lab, v in read_rows(body, 3, COUNT.fullmatch,
                                                              "Table 5.4")]
    return [rows[0:11], rows[11:22], rows[22:33]]


def read_population(doc):
    """Table 1.2 -> {label: total population}, labels in SEQUENCE order (asserted), and the
    male/female/total triples for the arithmetic check."""
    rows = []
    for pno in PAGES_T12:
        ls = lines_of(doc, pno)
        if "TABLE 1.2. Total population by sex, average age and locality" not in ls[0]:
            raise SystemExit(f"p.{pno + 1} is not Table 1.2: {ls[0]!r}")
        start = ls.index("Total") + 4        # Males Females Total | Males Females Total
        stop = ls.index("Average age")
        rows += read_rows(ls[start:stop], 6,
                          lambda t: bool(COUNT.fullmatch(t) or DECIMAL.fullmatch(t)),
                          f"Table 1.2 p.{pno + 1}")
    labels = [r[0] for r in rows]
    if labels != SEQUENCE:
        raise SystemExit(f"Table 1.2: {len(labels)} rows, expected {len(SEQUENCE)}")
    return [(lab, tuple(num(x) for x in v[:3])) for lab, v in rows]


def read_age_pages(doc):
    """Table 1.5, one page per locality -> {locality: (total, aged under 15)}.

    Each page is a stream of (age label, males, females, total) groups, two columns of ages
    interleaved (0-9 beside 50-59, Less than 1 beside 50, ...), with the locality's name and the
    printed page number as the last two lines. Age labels such as `12` look like counts, so the
    stream is read in fours from a known start rather than by token type, and every group must
    close (males + females = total).
    """
    out = {}
    for pno in T15_RANGE:
        ls = lines_of(doc, pno)
        if not ls or "TABLE 1.5. Total population by sex, age and locality" not in ls[0]:
            continue
        head = ls[1:9]
        if head != ["Age", "Males", "Females", "Total", "Age", "Males", "Females", "Total"]:
            raise SystemExit(f"Table 1.5 p.{pno + 1}: unexpected header {head}")
        if ls[-1] != str(pno + 1):
            raise SystemExit(f"Table 1.5 p.{pno + 1}: last line {ls[-1]!r}")
        name, data = ls[-2], ls[9:-2]
        if len(data) % 4:
            raise SystemExit(f"Table 1.5 p.{pno + 1} ({name}): {len(data)} tokens, not fours")
        groups = {}
        for i in range(0, len(data), 4):
            lab, m, f, t = data[i], num(data[i + 1]), num(data[i + 2]), num(data[i + 3])
            if m + f != t:
                raise SystemExit(f"Table 1.5 {name} {lab}: {m} + {f} != {t}")
            if lab in groups:
                raise SystemExit(f"Table 1.5 {name}: age label {lab!r} twice")
            groups[lab] = t
        decades = ["0-9"] + [f"{a}-{a + 9}" for a in range(10, 90, 10)] + ["Over 89"]
        if sum(groups[d] for d in decades) != groups["Total"]:
            raise SystemExit(f"Table 1.5 {name}: the age groups do not sum to the total")
        singles_0_9 = groups["Less than 1"] + sum(groups[str(a)] for a in range(1, 10))
        if singles_0_9 != groups["0-9"]:
            raise SystemExit(f"Table 1.5 {name}: single years 0-9 do not sum to 0-9")
        name = T15_ALIAS.get(name, name)
        if name in out:
            raise SystemExit(f"Table 1.5: {name} on two pages")
        out[name] = (groups["Total"], groups["0-9"] + sum(groups[str(a)] for a in range(10, 15)))
    return out


def check(doc):
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    print("Malta: Census of Population and Housing 2021, Volume 1, Table 5.3\n")
    with open(PDF, "rb") as fh:
        body = fh.read()
    say(len(body) == SIZE and body[:5] == b"%PDF-" and b"%%EOF" in body[-16:],
        f"the file is {len(body):,} bytes (expected {SIZE:,}), a PDF with its trailer")
    say(doc.page_count == PAGES, f"the volume is {doc.page_count} pages (expected {PAGES})")

    # 1. who was asked, and what the answer means
    intro = page_text(doc, PAGE_INTRO)
    say("The question on religion/religious denomination was asked to respondents aged 15 and "
        "over." in intro, "p.7: the religion question was asked of respondents aged 15 and over")
    defin = page_text(doc, PAGE_DEF)
    say("The set of beliefs and practices with which respondents identify themselves, "
        "regardless of the level of religious attendance or observance, or formal membership "
        "to a church or religious community." in defin,
        "p.174 defines religious affiliation as self-identification, not membership")
    ch5 = page_text(doc, PAGE_CH5)
    say("A total of 373,304 residents (or 82.6 per cent) identified themselves as Roman "
        "Catholic." in ch5 and "A total of 23,243 residents (5.1 per cent) stated that they did "
        "not belong to any religion" in ch5,
        "p.159's prose gives Roman Catholic 373,304 and no religion 23,243")

    # 2. Table 5.2 against the transcription, and its own arithmetic
    t52 = read_t52(doc)
    persons = {lab: v for lab, v in t52 if lab in T52 and lab != "Gozo and Comino"}
    # `Gozo and Comino` is both an island row (third) and a district row (last) with the same
    # figures; take the district row's triple for the sex check below.
    labels52 = [lab for lab, _ in t52]
    persons["Gozo and Comino"] = t52[2][1]
    say(persons == T52, "Table 5.2's nation, island and district rows equal the transcription")
    say(labels52[:6] == ["MALTA", "Malta", "Gozo and Comino", "Total", "Males", "Females"]
        and t52[3][1] == T52["MALTA"], "Table 5.2's Total row is the national row")
    sex_ok = True
    for k in range(3, len(t52), 3):
        (lt, t), (lm, m), (lf, f) = t52[k], t52[k + 1], t52[k + 2]
        sex_ok &= lm == "Males" and lf == "Females" and all(a == b + c for a, b, c in zip(t, m, f))
    say(sex_ok and len(t52) == 24, "Table 5.2: persons = males + females in all 7 x 11 blocks")
    say(all(sum(v[:10]) == v[10] for v in T52.values()),
        "every Table 5.2 row's ten answers sum to its total")
    say(all(tuple(sum(T52[d][i] for d in ds) for i in range(11)) == T52[isl]
            for isl, ds in ISLANDS.items())
        and tuple(T52["Malta"][i] + T52["Gozo and Comino"][i] for i in range(11)) == T52["MALTA"],
        "the districts sum to their islands and the islands to the nation, in all 11 columns")

    # 3. Table 5.1: the universe starts at 15 and the total row is the national row
    t51 = read_t51(doc)
    ages = ["15-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "Over 89"]
    blocks = {t51[i][0]: [t51[i]] + t51[i + 1:i + 10] for i in (0, 10, 20)}
    say(list(blocks) == ["Total", "Males", "Females"]
        and all([lab for lab, _ in blk[1:]] == ages for blk in blocks.values()),
        "Table 5.1 has Total, Males and Females blocks, each with ages 15-19 to over 89")
    say(all(tuple(sum(v[i] for _, v in blk[1:]) for i in range(11)) == blk[0][1]
            for blk in blocks.values()),
        "Table 5.1: the nine age rows sum to each block's total in all 11 columns")
    say(blocks["Total"][0][1] == T52["MALTA"]
        and all(a == b + c for a, b, c in zip(blocks["Total"][0][1], blocks["Males"][0][1],
                                              blocks["Females"][0][1])),
        "Table 5.1's total row is the national row, and males + females close")

    # 4. Table 5.4: citizenship blocks
    t54 = read_t54(doc)
    names54 = [[lab for lab, _ in blk] for blk in t54]
    say(all(n == ["Total"] + CATS for n in names54),
        "Table 5.4 prints the ten answers with the same labels as Table 5.3's header")
    tot = [tuple(v[2] for _, v in blk) for blk in t54]
    nat = (T52["MALTA"][10],) + T52["MALTA"][:10]
    say(tot[2] == nat and all(a + b == c for a, b, c in zip(tot[0], tot[1], tot[2])),
        f"Table 5.4: Maltese {tot[0][0]:,} + non-Maltese {tot[1][0]:,} = the national row")
    say(tot[0][10] == NO_RELIGION_MALTESE
        and "Almost a third of those with no religious affiliation (7,254) were Maltese "
            "citizens." in ch5,
        "Table 5.4's Maltese no-religion cell is p.159's 7,254")

    # 5. Table 5.3: the locality rows
    rows = read_locality_table(doc)
    t53 = {}
    for lab, v in rows:
        t53.setdefault(lab, []).append(tuple(map(num, v)))
    say(all(sum(v[:10]) == v[10] for vs in t53.values() for v in vs),
        "every Table 5.3 row's ten answers sum to its total")
    head = {"MALTA": t53["MALTA"][0], "Malta": t53["Malta"][0],
            "Gozo and Comino": t53["Gozo and Comino"][-1]}
    head.update({d: t53[d][0] for d in DISTRICTS if d != "Gozo and Comino"})
    say(head == T52 and t53["Gozo and Comino"][0] == t53["Gozo and Comino"][-1],
        "Table 5.3's nation, island and district rows equal Table 5.2's")
    sums = {d: tuple(sum(t53[loc][0][i] for loc in locs) for i in range(11))
            for d, locs in DISTRICTS.items()}
    say(all(sums[d] == T52[d] for d in DISTRICTS),
        "the 68 localities sum to their district rows in all 11 columns")
    say(len(LOCALITIES) == N_LOCALITIES and len({loc for _, loc in LOCALITIES}) == N_LOCALITIES,
        f"{N_LOCALITIES} distinct localities")

    # 6. Table 1.2 and Table 1.5: every locality's religion total is its population aged 15+
    pop = read_population(doc)
    popd = {}
    for lab, (m, f, t) in pop:
        popd.setdefault(lab, []).append(t)
    say(all(m + f == t for _, (m, f, t) in pop), "Table 1.2: males + females = total, every row")
    say(all(popd[k][-1] == v for k, v in T12_TOTALS.items())
        and all(sum(popd[loc][0] for loc in locs) == T12_TOTALS[d]
                for d, locs in DISTRICTS.items()),
        f"Table 1.2's localities sum to its districts; nation {T12_TOTALS['MALTA']:,}")
    age = read_age_pages(doc)
    want = {loc for _, loc in LOCALITIES}
    say(set(age) == want,
        f"Table 1.5 has one page for each of the 68 localities ({len(age)} found)"
        + (f"; pages not in the list: {sorted(set(age) - want)}; list not on a page: "
           f"{sorted(want - set(age))}" if set(age) != want else ""))
    bad_tot = [(loc, age.get(loc, (None,))[0], popd[loc][0]) for _, loc in LOCALITIES
               if age.get(loc, (None,))[0] != popd[loc][0]]
    say(not bad_tot, "Table 1.5's locality totals equal Table 1.2's"
        + (f"; differ (locality, Table 1.5, Table 1.2): {bad_tot[:6]}" if bad_tot else ""))
    bad_15 = [(loc, age[loc][0] - age[loc][1], t53[loc][0][10]) for _, loc in LOCALITIES
              if loc in age and age[loc][0] - age[loc][1] != t53[loc][0][10]]
    say(not bad_15, "every locality's religion total equals its population aged 15 and over in "
        "Table 1.5: nobody 15+ is missing from the religion table"
        + (f"; differ (locality, 15+, religion total): {bad_15[:6]}" if bad_15 else ""))
    under15 = sum(age[loc][1] for _, loc in LOCALITIES if loc in age)
    say(under15 == RESIDENTS - AGED_15_PLUS,
        f"under-15s in Table 1.5 sum to {under15:,} = {RESIDENTS:,} - {AGED_15_PLUS:,}, "
        f"{100.0 * under15 / RESIDENTS:.2f}% of residents")

    if not ok:
        raise SystemExit("reconciliation FAILED")
    return {loc: t53[loc][0] for _, loc in LOCALITIES}, {loc: popd[loc][0] for _, loc in LOCALITIES}


def emit(t53):
    district = {loc: d for d, loc in LOCALITIES}
    rows = []
    for _, loc in LOCALITIES:
        v = t53[loc]
        for c, n in zip(CATS, v[:10]):
            if n > 0:
                rows.append(dict(geo_id=loc, geo_level="locality", geo_name=loc,
                                 source_category=c, count=n, basis=BASIS, year=YEAR,
                                 source_id=SOURCE_ID,
                                 note=f"{district[loc]} district; NSO Census 2021 Vol. 1 Table "
                                      f"5.3 (printed pp.162-164); aged 15 and over"))
    return rows


def report(t53, popd):
    nat = T52["MALTA"]
    print(f"\n  national shares of the {nat[10]:,} aged 15+: "
          + ", ".join(f"{c} {100.0 * n / nat[10]:.2f}%" for c, n in zip(CATS, nat[:10])))
    for i, c in enumerate(CATS):
        top = sorted(t53, key=lambda loc: -t53[loc][i] / t53[loc][10])[:4]
        most = max(t53, key=lambda loc: t53[loc][i])
        print(f"  {c}: highest share " + ", ".join(
            f"{loc} {100.0 * t53[loc][i] / t53[loc][10]:.1f}% ({t53[loc][i]:,})" for loc in top)
            + f"; most people {most} {t53[most][i]:,} of {nat[i]:,}")
    print(f"  localities: {len(t53)}, residents per locality {RESIDENTS / len(t53):,.0f}")


def main():
    import fitz

    if "--fetch" in sys.argv:
        if os.path.exists(PDF):
            print("already have", PDF)
        else:
            raise SystemExit(f"{URL} answers 403 to non-browser clients; download it in a browser "
                             f"into {RAW} (ask 027)")
    if not os.path.exists(PDF):
        raise SystemExit(f"{PDF} missing; see --fetch")
    doc = fitz.open(PDF)
    t53, popd = check(doc)
    report(t53, popd)
    rows = emit(t53)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".part", "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLUMNS)
        wr.writeheader()
        wr.writerows(rows)
    os.replace(OUT + ".part", OUT)                                 # [[reference_wb_truncates]]
    print(f"\nwrote {OUT} ({len(rows)} rows, {sum(r['count'] for r in rows):,} people)")


if __name__ == "__main__":
    main()
