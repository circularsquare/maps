"""Austria — STATISTIK AUSTRIA, Volkszählung 2001, Tabelle 4 "Bevölkerung nach Religion".

Reads (or fetches) data/raw/at/ and writes data/normalized/at.csv.

**Ten religion categories on 2,358 Gemeinden for 8,032,926 people** — ~3,400 people per unit,
which puts Austria between Poland's gminy (15,400) and Ireland's Small Areas. The table is an
exact partition: the ten categories sum to each unit's `Insgesamt` on every row, at all three
printed tiers, in all nine volumes.

**THE 31 CATEGORIES IN THE QUEUE ROW ARE NATIONAL AND DO NOT EXIST AT GEMEINDE.** UNSD's
oracle row for Austria lists 31 named bodies because that is what Statistik Austria forwarded
to the Demographic Yearbook at country level. Every subnational table in the 2001 publications
carries **ten**, and Tabelle 15 — the Bundesland-level cross-tab, where more detail would live
if it lived anywhere — carries the same ten cross-tabbed by age and citizenship instead. The
depth and the geography are not both available, which is §3.9 in its usual shape.

**THE COLUMN NUMBERS ARE PRINTED OUT OF ORDER AND THE VOLUMES DISAGREE WITH EACH OTHER.**
In the eight Länder volumes the header numbers read `1 2 3 5 4 6 7 8 9 10 11` left to right:
Orthodox is printed fourth and numbered **5**, Evangelisch is printed fifth and numbered **4**.
In the Wien volume the same eleven columns are numbered `1 2 3 4 5 6 7 8 9 10 11` in print
order. So the printed number means different things in different volumes, and keying on it
swaps Orthodoxy and Protestantism in eight of nine — silently, because both are plausible
sizes and every total still reconciles. **Columns are identified from the header LABELS by
x-position and the resulting order is asserted**, which is the only reading that cannot be got
wrong this way.

Three facts in the Vorarlberg volume's own prose pin the label order independently: its text
says 274,000 people gave `römisch-katholisch`, 78.0% (the parse gives 273,978 / 351,095 =
78.03%), and that Vorarlberg has the **highest Muslim share of any Bundesland** (the parse
gives 8.36%, and no other Land exceeds it).

**THE GEOMETRY IS THE PARSE.** Every figure is right-aligned on one of eleven fixed x1
anchors, identical in all nine volumes, and the Kennziffer is right-aligned in its own column
at x1 ≈ 57.85. Reading the page as text lines does not work: PyMuPDF emits the stacked header
fragments in an order that is neither print order nor column order (this is how the `5 4`
anomaly first looked like an extraction bug rather than a fact about the source).

**`-` IS AN IN-BAND ZERO**, in 7,259 cells. Dropping it shifts every figure in the row left by
one column, which no total would catch.

**THE TIER IS THE CODE LENGTH.** The Vorspalte carries Statistik Austria's Topographische
Kennziffer: 1 digit = Bundesland, 3 = Politischer Bezirk, 5 = Gemeinde. Tabelle 15 shares the
header with Tabelle 4 but has no Kennziffer column at all, so requiring a code is also what
keeps the age cross-tab out of the drawn rows.

**WIEN IS THE EXCEPTION AND IT IS AN UPGRADE.** The Wien volume's Tabelle 4 is not by Gemeinde
— Vienna is one Gemeinde — but by **Zählbezirk**, 245 of them, with the 23 Gemeindebezirke as
the tier above. Both are emitted. Vienna is 19.3% of the country and by a distance its most
mixed part (25.6% ohne Bekenntnis against 12.0% nationally, 7.8% Muslim against 4.2%), so
drawing it as a single polygon would flatten the one place the map most wants to resolve.

**THE ÖSTERREICH VOLUME IS THE CROSS-CHECK, NOT A SOURCE.** Its Tabelle 4 stops at Politischer
Bezirk. It is parsed anyway because 99 district rows read out of a tenth file are an
independent check on the eight that were read out of the others, and because it carries the
national row this file asserts against.

Checks, all equalities:
  * the ten categories sum to `Insgesamt` on every row of every tier (24,948 rows);
  * Gemeinden sum to their Politischer Bezirk, Bezirke to their Bundesland (Wien: Zählbezirke
    to their Gemeindebezirk, those to Wien);
  * the nine Bundesland totals sum to 8,032,926, the published census count;
  * every Bezirk row in the Österreich volume equals the same Bezirk assembled from Gemeinden
    in its Land volume — 99 rows × 11 columns across two independently parsed files;
  * four columns are checked against UNSD's independently forwarded national figures:
    Islamisch 338,988, Ohne Bekenntnis 963,263, Unbekannt 160,662, Israelitisch 8,140.

Usage:
    python sources/at.py --fetch    ten PDFs, ~50 MB, seconds
    python sources/at.py            normalise from data/raw/at/
"""

import csv
import os
import re
import ssl
import sys
import urllib.request
from collections import defaultdict

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "at")
OUT = os.path.join(ROOT, "data", "normalized", "at.csv")

SOURCE_ID = "at_vz_2001"
YEAR = 2001

BASE = ("https://www.statistik.at/fileadmin/publications/"
        "Volkszaehlung_2001__Hauptergebnisse_I_-_%s.pdf")

# site slug -> (our code, Bundesland Kennziffer, name).  Wien's volume is by Zählbezirk.
VOLUMES = [
    ("Burgenland",       "bgld",  "1", "Burgenland"),
    ("Kaernten",         "ktn",   "2", "Kärnten"),
    ("Niederoesterreich", "noe",  "3", "Niederösterreich"),
    ("Oberoesterreich",  "ooe",   "4", "Oberösterreich"),
    ("Salzburg",         "sbg",   "5", "Salzburg"),
    ("Steiermark",       "stmk",  "6", "Steiermark"),
    ("Tirol",            "tirol", "7", "Tirol"),
    ("Vorarlberg",       "vbg",   "8", "Vorarlberg"),
    ("Wien",             "wien",  "9", "Wien"),
]
NATIONAL_VOLUME = ("OEsterreich", "oesterreich")

# The eleven columns, left to right as PRINTED.  Never as numbered — see the module docstring.
# Each entry is (canonical label, [fragments that must appear in the header text above the
# column], [fragments that must NOT]).  Fragments are chosen to survive the volumes' broken
# text encoding, which returns U+FFFD for every umlaut.
COLUMNS = [
    ("Insgesamt",                                  ["Ins", "gesamt"],                []),
    ("Römisch-katholisch",                         ["misch", "katho"],               ["chisch"]),
    ("Griechisch-katholisch",                      ["chisch", "katho"],              []),
    ("Orthodox",                                   ["Ortho"],                        []),
    ("Evangelisch",                                ["Evan"],                         []),
    ("Andere christliche Gemeinschaften",          ["Andere", "christl", "Gemein"],  ["nicht"]),
    ("Israelitisch",                               ["Israe"],                        []),
    ("Islamisch",                                  ["Islamisch"],                    []),
    ("Andere nichtchristliche Gemeinschaften",     ["Andere", "nicht", "christl"],   []),
    ("Ohne Bekenntnis",                            ["Ohne", "kenntnis"],             []),
    ("Unbekannt",                                  ["bekannt"],                      ["kenntnis"]),
]
CATEGORIES = [c[0] for c in COLUMNS[1:]]          # the ten religion columns
TOTAL_LABEL = COLUMNS[0][0]

# UNSD Demographic Yearbook table 28, Austria 2001 — forwarded by Statistik Austria to the UN
# and therefore independent of these PDFs.  Only the four categories whose UNSD label maps to
# exactly one column here are usable; the rest of UNSD's 31 sit inside the two `Andere` cells.
UNSD_NATIONAL = {
    "Römisch-katholisch": 5915421,
    "Evangelisch": 376150,
    "Islamisch": 338988,
    "Ohne Bekenntnis": 963263,
    "Unbekannt": 160662,
    "Israelitisch": 8140,
}
CENSUS_TOTAL = 8032926

# ---------------------------------------------------------------------------------------
# THE CROSSWALK BETWEEN THE TEN DRAWN COLUMNS AND UNSD'S THIRTY-ONE NATIONAL ROWS.
#
# UNSD's Demographic Yearbook carries 31 named bodies for Austria 2001 and this table carries
# ten.  The four columns that are not a single UNSD row are set out below, and each is
# ASSERTED to equal the sum of its members exactly.  Nothing here is a judgement about what a
# body is: it is arithmetic over 25 published figures, and it closes to the person on all four
# with no remainder and no cell used twice.
#
# It does two jobs.  It is much the sharpest check on the parse — four exact sums over 25
# independently published numbers would not survive a single misread column — and it is what
# licenses `taxonomy/at2001.py` to split the two `Andere` cells at all.  Without it the split
# would be a guess about which bodies Statistik Austria put in which bucket; with it, it is a
# published decomposition.
#
# TWO OF UNSD'S ENGLISH LABELS DO NOT MEAN WHAT THEY SAY.  `Greek Oriental` (1,089) and
# `Catholic` (764) sum to Griechisch-katholisch and belong to nothing else: 1,853 has no other
# decomposition among the 25.  `Greek Oriental` is a translation of *griechisch-orientalisch*,
# the old Austrian term for the Orthodox, and it lands here anyway — which is the reason this
# file never maps a UNSD row by its name.
CROSSWALK = {
    "Griechisch-katholisch": {
        "Greek Oriental": 1089, "Catholic": 764,
    },
    "Orthodox": {
        "Orthodox": 159115, "Greek Orthodox": 18533, "Armenian Apostolic": 1824,
    },
    "Andere christliche Gemeinschaften": {
        "Jehovah Witness": 23206, "Old Catholic Church": 14621,
        "Free Christian Community": 7186, "Evangelical": 4892,
        "Seventh Day Adventist": 4220, "New Apostolic": 4217,
        "Church of England": 2317, "Latter Day Saints": 2236, "Baptist": 2108,
        "Christian Community": 1428, "Methodist": 1263, "Christengmeinschaft": 1152,
        "Mennonite": 381,
    },
    "Andere nichtchristliche Gemeinschaften": {
        "Buddhist": 10402, "Hindu": 3629, "Sikh": 2794, "Other Religions": 1745,
        "Baha'i": 760, "Unification": 297, "Shinto": 123,
    },
}

# Page geometry.  Constant across all ten volumes.
CODE_X1_MAX = 62.0        # the Kennziffer column is right-aligned at x1 ~ 57.85
NAME_X0_MIN = 63.0
NAME_X1_MAX = 195.0       # the first figure column starts at x0 ~ 195
ROW_Y_TOL = 2.0
ANCHOR_TOL = 1.6
N_COLS = 11


# ---------------------------------------------------------------- fetch

def _fetch():
    """Download the ten volumes.  statistik.at omits a TLS intermediate, as stat.gov.pl and
    statsbank.gh do; every ordinary client fails identically with 'unable to get local issuer
    certificate' and the fix is the same."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    os.makedirs(RAW, exist_ok=True)
    want = [(slug, code) for slug, code, _, _ in VOLUMES] + [NATIONAL_VOLUME]
    for slug, code in want:
        dest = os.path.join(RAW, code + ".pdf")
        if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
            print("  have %-12s %9d" % (code, os.path.getsize(dest)))
            continue
        url = BASE % slug
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=300) as r:
            body = r.read()
        if not body.startswith(b"%PDF"):
            raise SystemExit("%s did not return a PDF (%d bytes)" % (url, len(body)))
        tmp = dest + ".part"
        with open(tmp, "wb") as f:
            f.write(body)
        os.replace(tmp, dest)
        print("  got  %-12s %9d  %s" % (code, len(body), url))

    if not os.path.exists(VBG_XLS):
        req = urllib.request.Request(VBG_XLS_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=300) as r:
            body = r.read()
        with open(VBG_XLS + ".part", "wb") as f:
            f.write(body)
        os.replace(VBG_XLS + ".part", VBG_XLS)
        print("  got  %-12s %9d  vorarlberg.at (the .xls cross-check)" % ("vbg xls", len(body)))


# ---------------------------------------------------------------- parse

DASHES = [0]


def _num(tok, count=False):
    """A figure cell.  `-` is an in-band zero and appears in thousands of cells."""
    if tok == "-":
        if count:
            DASHES[0] += 1
        return 0
    if re.fullmatch(r"\d+", tok):
        return int(tok)
    return None


def _column_anchors(words):
    """The eleven right-aligned figure columns, derived from the page's own figures.

    Clusters x1 over every numeric-looking token right of the name column and keeps the
    eleven most populated clusters.  Deriving them from the data rather than from the header
    is what makes the same code read all ten volumes.
    """
    xs = defaultdict(int)
    for x0, y0, x1, y1, txt, *_ in words:
        if x1 <= NAME_X1_MAX:
            continue
        if _num(txt) is None:
            continue
        xs[round(x1, 1)] += 1
    if not xs:
        return None
    clusters = []
    for x in sorted(xs):
        if clusters and x - clusters[-1][-1] <= ANCHOR_TOL:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    scored = sorted(clusters, key=lambda c: -sum(xs[x] for x in c))[:N_COLS]
    if len(scored) < N_COLS:
        return None
    return sorted(max(c) for c in scored)


def _header_for(anchors, words, first_data_y):
    """Read the stacked header label sitting above each figure column."""
    out = []
    for i, a in enumerate(anchors):
        left = anchors[i - 1] if i else NAME_X1_MAX - 5
        frags = [w[4] for w in words
                 if w[3] < first_data_y - 12 and w[0] > left - 4 and w[2] <= a + 6]
        out.append(" ".join(frags))
    return out


def _identify(headers):
    """Match each column's header text to a canonical label, and assert the order."""
    got = []
    for text in headers:
        hit = None
        for label, need, forbid in COLUMNS:
            if all(f in text for f in need) and not any(f in text for f in forbid):
                hit = label
                break
        got.append(hit)
    return got


def _rows_on_page(page, volume_code):
    words = page.get_text("words")
    anchors = _column_anchors(words)
    if anchors is None:
        return None, None

    # The header band ends where the first row carrying a Kennziffer begins.
    coded_ys = [w[1] for w in words
                if w[2] <= CODE_X1_MAX and w[0] > 30 and re.fullmatch(r"\d+", w[4])]
    if not coded_ys:
        return None, None          # Tabelle 15 and every other page: no Kennziffer column
    first_data_y = min(coded_ys)

    headers = _header_for(anchors, words, first_data_y)
    labels = _identify(headers)
    if labels != [c[0] for c in COLUMNS]:
        # Tabellen 1-3 also carry eleven right-aligned figure columns and a Kennziffer, so a
        # page that matches nothing is simply a different table.  A page that matches MOST of
        # the religion header and not all of it is the thing worth failing on: that is the
        # layout having changed under the parse.
        if sum(l is not None for l in labels) >= 6:
            return "BADHEADER", (headers, labels)
        return None, None

    # group words into rows
    rows = defaultdict(list)
    for w in words:
        if w[1] < first_data_y - 1:
            continue
        rows[round(w[1] / ROW_Y_TOL)].append(w)

    out = []
    for key in sorted(rows):
        ws = sorted(rows[key], key=lambda w: w[0])
        code = "".join(w[4] for w in ws if w[2] <= CODE_X1_MAX and w[0] > 30
                       and re.fullmatch(r"\d+", w[4]))
        if not code:
            continue
        name = " ".join(w[4] for w in ws
                        if NAME_X0_MIN <= w[0] and w[2] <= NAME_X1_MAX)
        cells = {}
        for w in ws:
            v = _num(w[4], count=True)
            if v is None or w[2] <= NAME_X1_MAX:
                continue
            for i, a in enumerate(anchors):
                if abs(w[2] - a) <= ANCHOR_TOL:
                    if i in cells:
                        raise SystemExit(
                            "%s: two figures on anchor %d in row %r" % (volume_code, i, name))
                    cells[i] = v
                    break
        if len(cells) != N_COLS:
            continue
        out.append((code, name.strip(), [cells[i] for i in range(N_COLS)]))
    return out, anchors


TITLE_RE = re.compile(r"^(\d+):$")


def _table_page_range(doc, want=4):
    """The pages of `Tabelle <want>`, bounded by the printed titles either side.

    Bounding the table by its own title is not decoration.  Tabelle 15 is *also* religion by
    these same eleven columns, cross-tabbed by age and citizenship instead of by place, and in
    the Burgenland volume its stub carries figures that read as a Kennziffer.  Geometry alone
    cannot tell the two apart; the title can.
    """
    titles = {}
    for i in range(doc.page_count):
        for x0, y0, x1, y1, txt, *_ in doc[i].get_text("words"):
            if txt != "Tabelle" or y0 > 70 or x0 > 45:
                continue
            nxt = [w for w in doc[i].get_text("words")
                   if abs(w[1] - y0) < 2 and w[0] > x1 and w[0] < x1 + 12]
            if nxt and TITLE_RE.match(nxt[0][4]):
                titles[i] = int(TITLE_RE.match(nxt[0][4]).group(1))
            break
    if want not in titles.values():
        raise SystemExit("no page carries the title 'Tabelle %d:'" % want)
    start = min(i for i, n in titles.items() if n == want)
    # Continuation pages REPEAT the same title, so the table ends at the next page carrying a
    # DIFFERENT number.  Bounding on "the next title of any kind" silently returns two pages
    # of a fifty-page table, and every check downstream still passes on the subset.
    later = [i for i in titles if i > start and titles[i] != want]
    end = min(later) if later else doc.page_count
    return list(range(start, end))


def _parse_volume(path, volume_code):
    doc = fitz.open(path)
    rows, pages = [], []
    bad = []
    for i in _table_page_range(doc, 4):
        got, extra = _rows_on_page(doc[i], volume_code)
        if got == "BADHEADER":
            bad.append((i, extra))
            continue
        if got:
            rows.extend(got)
            pages.append(i)
    doc.close()
    return rows, pages, bad


# ---------------------------------------------------------------- checks

# The fourteen Statutarstädte outside Vienna.  Each is a Politischer Bezirk that consists of
# exactly one Gemeinde, so Tabelle 4 prints it ONCE, at the Bezirk tier, and it never appears
# with a five-digit Kennziffer.  Left alone that drops 1,044,429 people — 13.0% of Austria and
# every one of its large cities — out of the drawn Gemeinde layer, with no total disagreeing,
# because at the Bezirk tier they are all present and correct.  The Gemeinde code is the
# Bezirk code plus `01`, which GISCO's 2001 commune file confirms for all fourteen.
STATUTARSTAEDTE = {
    "101": "Eisenstadt", "102": "Rust", "201": "Klagenfurt", "202": "Villach",
    "301": "Krems an der Donau", "302": "Sankt Pölten", "303": "Waidhofen an der Ybbs",
    "304": "Wiener Neustadt", "401": "Linz", "402": "Steyr", "403": "Wels",
    "501": "Salzburg", "601": "Graz", "701": "Innsbruck",
}


def _add_statutarstaedte(rows, volume_code):
    """Mint the Gemeinde row a Statutarstadt is never printed with."""
    have5 = {c for c, _, _ in rows if len(c) == 5}
    have3 = {c: (n, v) for c, n, v in rows if len(c) == 3}
    childless = sorted(b for b in have3 if not any(g.startswith(b) for g in have5))
    minted = []
    for b in childless:
        if b not in STATUTARSTAEDTE:
            raise SystemExit("%s: Bezirk %s %r has no Gemeinden and is not a known "
                             "Statutarstadt" % (volume_code, b, have3[b][0]))
        gid = b + "01"
        if gid in have5:
            raise SystemExit("%s: %s already exists as a Gemeinde" % (volume_code, gid))
        name, vals = have3[b]
        rows.append((gid, STATUTARSTAEDTE[b], list(vals)))
        minted.append(gid)
    return minted


VBG_XLS = os.path.join(RAW, "vbg_table4.xls")
VBG_XLS_URL = ("https://vorarlberg.at/documents/302033/472657/"
               "Volksz%C3%A4hlung+2001+-+Bev%C3%B6lkerung+nach+Religion.xls/"
               "736910d9-36a3-51b2-e5cd-8c9c2395091f?t=1616166395157")


def _vorarlberg_xls_check(rows):
    """Vorarlberg's Tabelle 4, as a spreadsheet, against Vorarlberg's Tabelle 4 as a PDF.

    The Land of Vorarlberg republishes this one table as .xls — the only Bundesland that
    does; the other eight exist as PDF only, the Excel editions having shipped on a CD-ROM
    with the print run. So one ninth of the country can be checked against a machine-readable
    file that shares no code path, no font, and no layout with the parse: 101 rows and 1,111
    figures, every one an equality.

    It also settles the column-numbering anomaly independently. The spreadsheet's own header
    row reads `1 2 3 5 4 6 7 8 9 10 11` over Insgesamt, Römisch-katholisch,
    Griechisch-katholisch, **Orthodox**, **Evangelisch**, ... — so Orthodox really is numbered
    5 and printed fourth, in a file where nothing about extraction could have reordered it.
    """
    try:
        import xlrd
    except ImportError:
        print("  (skipped the Vorarlberg .xls check — no xlrd)")
        return
    if not os.path.exists(VBG_XLS):
        print("  (skipped the Vorarlberg .xls check — run with --fetch)")
        return
    sh = xlrd.open_workbook(VBG_XLS).sheet_by_index(0)

    # the header's own column numbers, which are the point of the check
    nums = [sh.cell_value(2, c) for c in range(3, 14)]
    if [int(n) for n in nums] != [1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]:
        raise SystemExit("  the Vorarlberg .xls numbers its columns %s; this parse is built "
                         "on 1 2 3 5 4 6 7 8 9 10 11 and on the labels, not the numbers"
                         % [int(n) for n in nums])

    want = {}
    for r in range(sh.nrows):
        v = sh.cell_value(r, 0)
        if not isinstance(v, float):
            continue
        vals = []
        for c in range(3, 14):
            x = sh.cell_value(r, c)
            vals.append(0 if x == "-" else int(x))
        want[str(int(v))] = vals

    got = {c: v for c, _, v in rows}
    if not want:
        raise SystemExit("  the Vorarlberg .xls yielded no rows")
    missing = sorted(set(want) - set(got))
    if missing:
        raise SystemExit("  the Vorarlberg .xls has %d codes the PDF parse does not: %s"
                         % (len(missing), missing[:8]))
    for code, vals in sorted(want.items()):
        if got[code] != vals:
            raise SystemExit("  Vorarlberg %s: PDF %s, .xls %s" % (code, got[code], vals))
    print("  Vorarlberg checks against an independently published .xls of the same table: "
          "%d rows, %d figures, all equal" % (len(want), len(want) * N_COLS))


def _partition_check(rows, what):
    for code, name, vals in rows:
        if sum(vals[1:]) != vals[0]:
            raise SystemExit("%s: %s %r categories sum to %d, Insgesamt is %d"
                             % (what, code, name, sum(vals[1:]), vals[0]))


def _nest_check(rows, what):
    """Gemeinden sum to their Bezirk, Bezirke to the Bundesland."""
    by_len = defaultdict(dict)
    for code, name, vals in rows:
        if code in by_len[len(code)]:
            raise SystemExit("%s: duplicate code %s (%r)" % (what, code, name))
        by_len[len(code)][code] = (name, vals)
    for child_len, parent_len in ((5, 3), (3, 1)):
        if child_len not in by_len or parent_len not in by_len:
            continue
        agg = defaultdict(lambda: [0] * N_COLS)
        for code, (_, vals) in by_len[child_len].items():
            p = code[:parent_len]
            for i, v in enumerate(vals):
                agg[p][i] += v
        for p, (name, vals) in by_len[parent_len].items():
            if p not in agg:
                raise SystemExit("%s: %s %r has no children" % (what, p, name))
            if agg[p] != vals:
                raise SystemExit("%s: %s %r children sum to %s, printed %s"
                                 % (what, p, name, agg[p], vals))
    return by_len


# ---------------------------------------------------------------- main

def build():
    if fitz is None:
        raise SystemExit("PyMuPDF is required: pip install pymupdf")

    per_volume = {}
    print("parsing:")
    for slug, code, bl, land in VOLUMES:
        path = os.path.join(RAW, code + ".pdf")
        if not os.path.exists(path):
            raise SystemExit("missing %s — run with --fetch" % path)
        rows, pages, bad = _parse_volume(path, code)
        if bad:
            raise SystemExit("%s: header did not match on pages %s\n  %r"
                             % (code, [b[0] for b in bad], bad[0][1]))
        if not rows:
            raise SystemExit("%s: found no Tabelle 4 rows" % code)
        minted = [] if code == "wien" else _add_statutarstaedte(rows, code)
        _partition_check(rows, code)
        by_len = _nest_check(rows, code)
        bl_rows = by_len.get(1, {})
        if list(bl_rows) != [bl]:
            raise SystemExit("%s: expected one Bundesland row %s, got %s"
                             % (code, bl, list(bl_rows)))
        per_volume[code] = (land, bl, by_len, pages, set(minted))
        print("  %-6s pages %-9s  %3d Bezirk/e  %5d fine units  %9d people%s"
              % (code, "%d-%d" % (min(pages), max(pages)),
                 len(by_len.get(3, {})), len(by_len.get(5, {})), bl_rows[bl][1][0],
                 ("  (+%d Statutarstadt/-städte)" % len(minted)) if minted else ""))

    # national volume — Bezirk tier only, parsed as an independent check
    nat_path = os.path.join(RAW, NATIONAL_VOLUME[1] + ".pdf")
    nat_rows, nat_pages, nat_bad = _parse_volume(nat_path, "oesterreich")
    if nat_bad:
        raise SystemExit("oesterreich: header did not match on pages %s"
                         % [b[0] for b in nat_bad])
    _partition_check(nat_rows, "oesterreich")
    nat_by_len = _nest_check(nat_rows, "oesterreich")
    print("  %-6s pages %-9s  %4d Bezirk/e  %5d fine units"
          % ("at", "%d-%d" % (min(nat_pages), max(nat_pages)),
             len(nat_by_len.get(3, {})), len(nat_by_len.get(5, {}))))

    print("\nchecks:")

    # 1. the nine Bundesland totals against the published census count
    grand = [0] * N_COLS
    for code, (land, bl, by_len, _, minted) in per_volume.items():
        for i, v in enumerate(by_len[1][bl][1]):
            grand[i] += v
    if grand[0] != CENSUS_TOTAL:
        raise SystemExit("  nine Bundesländer sum to %d, census is %d" % (grand[0], CENSUS_TOTAL))
    print("  the nine Bundesland totals sum to %s — the published census count" % f"{grand[0]:,}")

    # 2. every Bezirk row, two independently parsed files
    n = 0
    for code, (land, bl, by_len, _, minted) in per_volume.items():
        for bez, (name, vals) in by_len.get(3, {}).items():
            if bez not in nat_by_len.get(3, {}):
                raise SystemExit("  Bezirk %s (%s) is in %s and not in the Österreich volume"
                                 % (bez, name, code))
            if nat_by_len[3][bez][1] != vals:
                raise SystemExit("  Bezirk %s %r: Land volume %s, Österreich volume %s"
                                 % (bez, name, vals, nat_by_len[3][bez][1]))
            n += 1
    if n != len(nat_by_len.get(3, {})):
        raise SystemExit("  %d Bezirke matched but the Österreich volume has %d"
                         % (n, len(nat_by_len[3])))
    print("  all %d Politische Bezirke agree cell for cell across two separately parsed "
          "volumes (%d equalities)" % (n, n * N_COLS))

    # 3. UNSD's independently forwarded national figures
    for label, want in UNSD_NATIONAL.items():
        got = grand[1 + CATEGORIES.index(label)]
        if got != want:
            raise SystemExit("  %s: parsed %d, UNSD forwarded %d" % (label, got, want))
    print("  %d of UNSD's national figures reproduced exactly: %s"
          % (len(UNSD_NATIONAL), ", ".join("%s %s" % (k, f"{v:,}")
                                           for k, v in UNSD_NATIONAL.items())))

    # 3b. the four columns that are not a single UNSD row, against their members
    seen = dict(UNSD_NATIONAL)
    for label, members in CROSSWALK.items():
        got = grand[1 + CATEGORIES.index(label)]
        if sum(members.values()) != got:
            raise SystemExit("  %s: parsed %d, UNSD members sum to %d"
                             % (label, got, sum(members.values())))
        for m, v in members.items():
            if m in seen:
                raise SystemExit("  UNSD row %r used in two columns" % m)
            seen[m] = v
    if len(seen) != 31:
        raise SystemExit("  the crosswalk accounts for %d UNSD rows, not 31" % len(seen))
    if sum(seen.values()) != CENSUS_TOTAL:
        raise SystemExit("  the 31 UNSD rows sum to %d, not %d"
                         % (sum(seen.values()), CENSUS_TOTAL))
    print("  and its other 25 rows decompose the remaining four columns exactly "
          "(%s), each row used once, all 31 summing to the census"
          % ", ".join("%s=%d" % (k.split()[0].rstrip(","), len(v))
                      for k, v in CROSSWALK.items()))

    # 4. the two prose facts in the Vorarlberg volume
    vbg = per_volume["vbg"][2][1]["8"][1]
    rk = vbg[1 + CATEGORIES.index("Römisch-katholisch")] / vbg[0] * 100
    if not 77.9 <= rk <= 78.1:
        raise SystemExit("  Vorarlberg römisch-katholisch is %.2f%%, its own text says 78.0%%" % rk)
    shares = {}
    for code, (land, bl, by_len, _, minted) in per_volume.items():
        v = by_len[1][bl][1]
        shares[land] = v[1 + CATEGORIES.index("Islamisch")] / v[0] * 100
    top = max(shares, key=shares.get)
    if top != "Vorarlberg":
        raise SystemExit("  the Vorarlberg volume says it has the highest Muslim share; "
                         "the parse says %s (%.2f%%)" % (top, shares[top]))
    print("  Vorarlberg reads 78.0%% römisch-katholisch and the highest Muslim share "
          "(%.2f%%) — both stated in its own text" % shares["Vorarlberg"])

    # 5. one whole Bundesland against a machine-readable edition of the same table
    _vorarlberg_xls_check([(c, n, v) for c, (n, v) in
                           list(per_volume["vbg"][2].get(1, {}).items())
                           + list(per_volume["vbg"][2].get(3, {}).items())
                           + list(per_volume["vbg"][2].get(5, {}).items())])

    # ------------------------------------------------------------ write
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n_rows = 0
    dashes = 0
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["geo_id", "geo_level", "geo_name", "source_category", "count",
                    "basis", "year", "source_id", "note"])
        for code, (land, bl, by_len, _, minted) in sorted(per_volume.items(),
                                                          key=lambda kv: kv[1][1]):
            wien = code == "wien"
            for clen, level in ((1, "bundesland"),
                                (3, "gemeindebezirk" if wien else "bezirk"),
                                (5, "zaehlbezirk" if wien else "gemeinde")):
                for gcode, (name, vals) in sorted(by_len.get(clen, {}).items()):
                    note = "level=%s; bundesland=%s" % (level, land)
                    if gcode in minted:
                        note += "; Statutarstadt, printed only at the Bezirk tier"
                    w.writerow(["AT" + gcode, level, name, TOTAL_LABEL, vals[0],
                                "self_id", YEAR, SOURCE_ID,
                                note + "; universe total, not a religion category"])
                    n_rows += 1
                    for i, cat in enumerate(CATEGORIES):
                        w.writerow(["AT" + gcode, level, name, cat, vals[1 + i],
                                    "self_id", YEAR, SOURCE_ID, note])
                        n_rows += 1
                        dashes += 1 if vals[1 + i] == 0 else 0

    print("\nwrote %s" % os.path.relpath(OUT, ROOT))
    print("  %d rows; %d cells printed `-`, an in-band zero" % (n_rows, DASHES[0]))
    print("\nnational composition, 2001:")
    for i, cat in enumerate(CATEGORIES):
        print("  %-42s %10s  %6.2f%%"
              % (cat, f"{grand[1 + i]:,}", grand[1 + i] / grand[0] * 100))
    print("  %-42s %10s  %6.2f%%" % ("TOTAL", f"{grand[0]:,}", 100.0))


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        _fetch()
    build()
