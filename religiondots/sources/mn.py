"""Mongolia — NSO, 2020 Population and Housing Census, the twenty-two per-aimag volumes.

Reads (or fetches) data/raw/mn/aimags/ and writes data/normalized/mn.csv.

**THE RELIGION QUESTION IS A 10% CENSUS LONG FORM, AND THAT IS THE FIRST THING TO KNOW ABOUT
THIS COUNTRY.** Every aimag volume prints the same sentence, and it is the source's own:

    "Хүн ам, орон сууцны тооллогод 'Та шашин шүтдэг үү, шүтдэг бол ямар шашин шүтдэг вэ?'
     гэсэн асуултыг анх удаагаа 2010 оны тооллогоор асууж байсан бөгөөд ээлжит тооллогод
     хүн амын 10 хувийн түүвэрт сонгогдсон 15, түүнээс дээш насны хүн амаас асуусан болно."

— the question was first asked at the 2010 census, and at this one it was put to the population
aged 15 and over *selected into a 10 per cent sample*. So the universe is 15+, not everybody,
and within that it is a designed subsample of roughly 240,000 people nationally. That is an
order of magnitude more respondents than any survey on this map and it is enumerated by the
census, but it is not a full count and the small cells are not precise. See §5 of sources/mn.md.

**THE TABLES ARE PERCENTAGES AND THERE ARE TWO OF THEM, CHAINED.** No volume prints a single
absolute religion figure. Each prints:

  * `ХҮСНЭГТ 3.4` — the 15+ population split `Шүтдэггүй` / `Шүтдэг`, per cent, one decimal;
  * `ХҮСНЭГТ 3.5` — the *religious* population split `Будда` / `Христ` / `Ислам` / `Бөө` /
    `Бусад`, per cent of the religious, one decimal.

So a type's share of the 15+ population is the product of the two, and the absolute count needs
a third table from the same volume, `ХҮСНЭГТ 2.4`, the resident population by age group, which
is the only one of the three printed in people rather than per cent.

**NOTHING IS FOUND BY ITS CAPTION, AND THAT IS THE MAIN LESSON OF THIS COUNTRY.** Twenty-two
statistics departments typeset twenty-two volumes and agreed on almost nothing. The religion
tables are numbered 3.4/3.5 in Bayan-Ölgii, 3.5/3.6 in Arkhangai, 3.7 in Khövsgöl, 3.9/3.10 in
Dornod and 3.10/3.11 in Bayankhongor. The age table is `СУУРИН ХҮН АМЫН ТОО...` in some volumes
and `ХҮН АМЫН ТОО...` in others, `ХҮЙС` in some and `ХҮЙСЭЭР` in others, prefixed with the
aimag's own name in Dornod, and Govi-Altai spells `ХҮН` as `ХУН`. Six volumes MERGE the
religiosity table into the type-of-religion table and print all seven rows together.
Ulaanbaatar TRANSPOSES the type table, putting the religions down the side. Selenge prints
2020 only where everyone else prints 2010 and 2020 side by side.

So each table is found by an IDENTITY THAT ONLY IT SATISFIES, checked on every page in turn:

  * the age table is the page whose own age bands sum to its own printed total;
  * the religiosity table is the page with `Шүтдэггүй` and `Шүтдэг` rows summing to 100.0;
  * the type table is the page with a 2020 distribution over the five religions summing
    to 100.0, in either layout.

That is `[[reference_pdf_table_geometry]]`'s advice with the anchor moved from the header to
the arithmetic, and it is what a caption-matched parser cannot do: a wrong page fails the sum,
where a wrong caption match returns numbers.

**THE COLUMN ORDER IS READ, NEVER ASSUMED.** Bayan-Ölgii prints Будда, Христ, Ислам, Бөө,
Бусад and Govisumber prints Будда, Бөө, Христ, Ислам, Бусад, so the categories are keyed off
the header row's own x-coordinates. The 2010/2020 column group is read the same way, off the
`Бүгд Эрэгтэй Эмэгтэй` header, because reading 2010 as 2020 is the one error no total here
would catch: Mongolia's religion shares barely moved between the two censuses.

Usage:
    python sources/mn.py --fetch    22 PDFs, ~330 MB, a few minutes
    python sources/mn.py            normalise from data/raw/mn/aimags/
"""

import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "mn", "aimags")
OUT = os.path.join(ROOT, "data", "normalized", "mn.csv")

SOURCE_ID = "mn_phc_2020_aimag"
YEAR = 2020
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

BASE = "https://downloads.1212.mn/"

# pcode -> (English name, the filename on downloads.1212.mn).
#
# THE FILENAMES ARE INCONSISTENT AND THAT IS THE HOST'S DOING, not a transcription slip:
# `Dun` and `dun` both occur, Darkhan-Uul has a SPACE where the others have an underscore,
# and Tuv has a space AND a doubled dot before the extension. Every one below was confirmed
# to return 200 with a `%PDF-` magic. The names are the NSO's romanisations, which do not
# always match COD's (`Khuvsgul` vs COD's `Khovsgol`, `Umnugovi` vs `Omnogovi`), which is
# exactly why this table is keyed on the PCODE and no name is ever joined.
AIMAGS = {
    "MN11": ("Ulaanbaatar", "Ulaanbaatar_XAOCT_Negdsen_dun.pdf"),
    "MN21": ("Dornod", "Dornod_XAOCT_Negdsen_Dun.pdf"),
    "MN22": ("Sukhbaatar", "Sukhbaatar_XAOCT_Negdsen_dun.pdf"),
    # THE PATTERN BREAKS COMPLETELY FOR THREE AIMAGS and they were nearly written off. They
    # are on the same host under names sharing nothing with the other nineteen, recovered
    # from the Wayback CDX of the whole 1212.mn domain: the retired
    # `BookLibraryDownload.ashx?url=<filename>` links are still archived, and that `url=`
    # parameter is exactly the filename on downloads.1212.mn. 273 spellings of the
    # `<Aimag>_XAOCT_Negdsen_dun.pdf` pattern were tried first and every one 404s.
    #
    # `Khentii.pdf` ALSO EXISTS, RETURNS 200 AND IS THE WRONG CENSUS. It is the 2010 volume,
    # published 2012, and its text layer is a legacy non-Unicode Cyrillic font that extracts
    # as mojibake. The 2020 Khentii volume is the one named below.
    "MN23": ("Khentii", "CENSUS-2020_KHENTII_MAIN_REPORT.pdf"),
    "MN84": ("Khovd", "Khovd.pdf"),
    "MN41": ("Tuv", "Tuv_XAOCT_Negdsen%20dun..pdf"),
    "MN42": ("Govisumber", "Govisumber_XAOCT_Negdsen_dun.pdf"),
    "MN43": ("Selenge", "Selenge_XAOCT_Negdsen_dun.pdf"),
    "MN44": ("Dornogovi", "Dornogovi_XAOCT_Negdsen_Dun.pdf"),
    "MN46": ("Umnugovi", "Umnugovi_XAOCT_Negdsen_Dun.pdf"),
    "MN61": ("Orkhon", "Orkhon_XAOCT_Negdsen_Dun.pdf"),
    "MN62": ("Uvurkhangai", "Uvurkhangai_XAOCT_Negdsen_dun.pdf"),
    "MN63": ("Bulgan", "Bulgan_XAOCT_Negdsen_dun.pdf"),
    "MN64": ("Bayankhongor", "Bayankhongor_XAOCT_Negdsen_dun.pdf"),
    "MN65": ("Arkhangai", "Arkhangai_XAOCT_Negdsen_dun.pdf"),
    "MN67": ("Khuvsgul", "Khuvsgul_XAOCT_Negdsen_Dun.pdf"),
    "MN81": ("Zavkhan", "Zavkhan_XAOCT_Negdsen_dun.pdf"),
    "MN82": ("Govi-Altai", "Govi-Altai_XAOCT_Negdsen_Dun.pdf"),
    "MN83": ("Bayan-Ulgii", "Bayan-Ulgii_XAOCT_Negdsen_Dun.pdf"),
    "MN85": ("Uvs", "Uvs_XAOCT_Negdsen_Dun.pdf"),
}

# KHOVD PUBLISHES NO RELIGIOSITY TABLE AT ALL, only the type-of-religion one, and states
# the split in a SENTENCE instead (page index 55):
#
#   "Арван тав, түүнээс дээш насны нийт хүний 39.3 хувь нь ямар нэг шашингүйчүүд, 60.6
#    хувь нь хүн шашинтан байгаагийн 49.3 хувь нь Буддын шашинтан байна."
#
# — of the population aged 15 and over, 39.3 per cent have no religion and 60.6 per cent
# have one, of whom 49.3 per cent are Buddhist. Those two figures are taken here rather than
# dropping the aimag, because Khovd is one of only two with a substantial Muslim population
# and losing it would misstate where Mongolia's Muslims live.
#
# **THE SENTENCE CHECKS ITSELF, which is why this is a transcription and not a guess.** Its
# third figure, 49.3, is the Buddhist share of ALL adults; the type table on the next page
# gives Buddhists as 81.3% of the religious; and 0.606 x 0.813 = 0.4927. A row read from the
# wrong year or the wrong line would not reproduce that. `check()` re-asserts it below.
OVERRIDE = {
    "MN84": (39.3, 60.6),
}

# DUNDGOVI IS PUBLISHED AND IS STILL NOT DRAWN, and the reason is not that it is missing.
# `https://downloads.1212.mn/Dundgovi.pdf` is the right volume, 2020, 211 pages, 34.8 MB --
# and it is a SCAN. Its producer is `iLovePDF`, and the only extractable text on any page is
# the running header and the page number; the tables are pixels. Nothing in this file can
# read it, and OCR of Mongolian Cyrillic tables is not something to bolt on here. See
# sources/mn.md §7 for the two ways it could be brought in.
#
# DARKHAN-UUL IS THE SAME PROBLEM ONE STEP FURTHER IN. Its volume has a text layer for the
# prose and the captions, and its TABLES are pasted-in images: the two religion pages carry
# nine picture objects between them and two extractable data rows. Nothing here can read it.
MISSING = {
    "MN48": "Dundgovi",
    "MN45": "Darkhan-Uul",
}

ALL_AIMAG_NAMES = ({n for n, _ in AIMAGS.values()} | set(MISSING.values()))

CATEGORIES = ["Будда", "Христ", "Ислам", "Бөө", "Бусад"]
NOT_RELIGIOUS = "Шүтдэггүй"
RELIGIOUS = "Шүтдэг"
TOTAL_CAT = "Total"

# Captions, as printed in the body, in capitals. NOT case-insensitive -- see the module
# docstring. The table number is matched as `[\d.]+` because it is not stable across volumes.
# Pages before this are front matter -- covers, editorial credits and the contents list,
# which repeats every caption and every page number and would otherwise be parsed as data.
FRONT_MATTER = 15

# The national report, which supplies the DENOMINATOR only. Its religion chapter has no
# geography at all (sex, age group and ethnicity, nationally), which is why the twenty-two
# aimag volumes have to be read for the shares.
NATIONAL_REPORT = "Census2020_Main_report_Eng.pdf"
NATIONAL_REPORT_URL = BASE + NATIONAL_REPORT
APPENDIX_1_1 = re.compile(
    r"TABLE\s*1\.1\.\s*NUMBER OF RESIDENT POPULATION OF MONGOLIA,\s*BY AGE GROUP,"
    r"\s*AIMAGS AND THE CAPITAL", re.I)

DASHES = {"-", "–", "—", "−"}
INT = re.compile(r"^\d{1,3}$")
GRP = re.compile(r"^\d{3}$")
DEC = re.compile(r"^\d+[.,]\d$")
# `15-19` ... `65-69` and `70+`. Bayan-Olgii's table 3.5 misprints `50-59` as `40-59`; that
# is the source's typo and is harmless here because only the БҮГД row is ever read.
BAND = re.compile(r"^(\d{1,2})\s*[-–]\s*\d{1,2}$")
OPEN_BAND = re.compile(r"^(\d{1,2})\s*\+$")

ADULT_AGE = 15          # the question's own universe


def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    os.makedirs(RAW, exist_ok=True)
    todo = [(p, n, f, os.path.join(RAW, _local(f)))
            for p, (n, f) in sorted(AIMAGS.items())]
    todo.append(("--", "national report", NATIONAL_REPORT,
                 os.path.join(os.path.dirname(RAW), NATIONAL_REPORT)))
    for pcode, name, fname, dest in todo:
        if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
            print(f"have  {pcode} {name}")
            continue
        url = BASE + fname
        print("GET", url)
        # downloads.1212.mn serves a broken TLS chain (a missing intermediate, the same
        # failure as stat.gov.pl and Ghana's statsbank). A cert error here is not the host
        # being down. sources/mn.md §4.
        r = requests.get(url, timeout=900, verify=False, stream=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        body = r.content
        # §5a: HTTP 200 is not a download. This host answers a wrong filename with a 404
        # page, but assert the magic anyway.
        if not body.startswith(b"%PDF-"):
            raise SystemExit(f"{url} is not a PDF -- {len(body):,} bytes, "
                             f"starts {body[:8]!r}")
        tmp = dest + ".part"
        with open(tmp, "wb") as fh:
            fh.write(body)
        os.replace(tmp, dest)
        print(f"  {len(body):,} bytes")


def _local(fname):
    """The on-disk name: the URL's %20 and doubled dot are not put on the filesystem."""
    return fname.replace("%20", "_").replace("..pdf", ".pdf")


def _rows(page):
    """[(label, [(value, x)])] per printed line.

    A figure printed `85 232` reaches PyMuPDF as two words, so a 3-digit token that follows
    a number within a few points is a thousands group and is joined back on. Anything that
    is not a number goes to the label, which is why the year separator rows (`2010`, `2020`,
    four digits) survive as labels and can be used to tell the two blocks apart.
    """
    lines = {}
    for w in page.get_text("words"):
        lines.setdefault(round(w[1] / 3.0), []).append(w)

    out = []
    for y in sorted(lines):
        ws = sorted(lines[y], key=lambda w: w[0])
        vals, label, i = [], [], 0
        while i < len(ws):
            t = ws[i][4]
            if t in DASHES:
                # `-` IS AN IN-BAND ZERO, exactly as in Sri Lanka's GN table, and dropping
                # it into the label instead of the value list shifts every figure to its
                # left by one column. Govi-Altai's `Ислам` row is printed with dashes for
                # the cells it has none in.
                vals.append((0.0, ws[i][0]))
            elif INT.match(t) or DEC.match(t):
                text, x1 = t, ws[i][2]
                while (i + 1 < len(ws) and GRP.match(ws[i + 1][4])
                       and ws[i + 1][0] - x1 < 6.0):
                    text += ws[i + 1][4]
                    x1 = ws[i + 1][2]
                    i += 1
                vals.append((float(text.replace(",", ".")), ws[i][0]))
            else:
                label.append(t)
            i += 1
        out.append((" ".join(label).strip(), vals, ws))
    return out


def _col_2020(rows, n_values):
    """Index, among a data row's values, of the 2020 `Бүгд` column. None if not that shape.

    **THE WIDTH IS TAKEN FROM THE DATA ROW, NOT FROM THE HEADER.** Counting `Бүгд` tokens in
    the header was the obvious way and it is wrong: several volumes set the header over two
    baselines, so one physical line carries only the first group's `Бүгд Эрэгтэй Эмэгтэй`
    and the table reads as though it had a single year. A data row cannot lie about how many
    columns it has.

    Getting this wrong reports 2010 as 2020, and that is the one error no total downstream
    would catch, because Mongolia's religion shares barely moved between the two censuses.
    """
    if n_values >= 6:
        col = 3          # 2010 Бүгд/Эр/Эм then 2020 Бүгд/Эр/Эм
    elif n_values >= 3:
        col = 0          # a 2020-only table; Selenge prints one
    else:
        return None

    if col and not any("2020" in l.split() for l, _, _ in rows):
        return None

    # Every volume prints 2010 on the left and 2020 on the right, so that is the default.
    # It is confirmed against the spanner row where there is an unambiguous one: a row whose
    # WHOLE label is the two years and which carries no figures.
    #
    # **THE SPANNER MUST NOT BE LOOKED FOR ANYWHERE ELSE ON THE PAGE**, which was the first
    # attempt and was wrong in a way that produced numbers rather than an error. Govi-Altai
    # captions its table `... ДҮНД ЭЗЛЭХ ХУВИАР, 2010 ОН, 2020 ОН`, the caption wraps, and
    # `2020` therefore begins a line further LEFT than the `2010` above it -- so a page-wide
    # x comparison concluded the columns were reversed and read Govi-Altai's 2010 figures as
    # its 2020 ones. Sükhbaatar sets the two year labels on separate baselines, which is
    # also not a spanner, and is likewise left to the default.
    if col:
        for label, vals, ws in rows:
            if not vals and set(label.split()) == {"2010", "2020"}:
                xs = {w[4]: w[0] for w in ws if w[4] in ("2010", "2020")}
                return 0 if xs["2020"] < xs["2010"] else 3
    return col


def read_population():
    """{aimag English name: (resident population, population aged 15+)}, 2020.

    **THE DENOMINATOR DOES NOT COME OUT OF THE AIMAG VOLUMES, AND THAT IS DELIBERATE.**
    Each volume does print its own population by age group, but not comparably: Bayan-Ölgii
    prints it in people and Dornod prints the very same table in per cent, so a parser that
    read all twenty-two would be reading two different quantities and would have to find a
    total elsewhere for half of them anyway.

    `Census2020_Main_report_Eng.pdf` APPENDIX TABLE 1.1 gives resident population by age
    group for all twenty-two units in one English table, and everything needed is on its
    first page: the row's `Total` and its `0-4`, `5-9` and `10-14` columns, so 15+ is a
    subtraction and the continuation pages are never touched. It also covers Dundgovi, whose
    religion shares are not readable (see MISSING), so the denominator is already here if
    that volume is ever transcribed.

    Checked against Bayan-Ölgii's own volume, which prints 103,908 and the same three child
    bands: identical.
    """
    import fitz

    p = os.path.join(os.path.dirname(RAW), NATIONAL_REPORT)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch first")
    doc = fitz.open(p)

    for i in range(100, doc.page_count):
        flat = " ".join(doc[i].get_text().split())
        if not APPENDIX_1_1.search(flat):
            continue
        out = {}
        for label, vals, _ in _rows(doc[i]):
            if len(vals) < 4:
                continue
            name = label.strip()
            if name == "TOTAL" or name in ALL_AIMAG_NAMES:
                total = vals[0][0]
                kids = sum(v for v, _ in vals[1:4])
                # **setdefault, NOT assignment.** The page carries the table's continuation
                # block underneath the first, so every aimag appears TWICE on it: once with
                # `Total, 0-4, 5-9, 10-14, ...` and once with `35-39, 40-44, ...`. Taking
                # the last occurrence read the second block's age columns as though they
                # were a total and three child bands, which made every 15+ figure NEGATIVE
                # -- caught here only because the sign was absurd. The block below now
                # rejects that shape outright as well.
                out.setdefault(name, (int(total), int(round(total - kids))))
        missing = ALL_AIMAG_NAMES - set(out)
        if missing or "TOTAL" not in out:
            raise SystemExit(f"APPENDIX TABLE 1.1 on page {i} is missing rows for "
                             f"{sorted(missing) or ['TOTAL']}")
        # Every 15+ figure must be a POSITIVE MINORITY SHARE of its own total -- between a
        # half and nine tenths of it. Mongolia is a young country, so the floor is low.
        for n, (tot, p15) in out.items():
            if not 0.5 * tot <= p15 <= 0.9 * tot:
                raise SystemExit(f"APPENDIX TABLE 1.1: {n} reads {p15:,} aged 15+ out of "
                                 f"{tot:,}, which is not a possible age structure -- the "
                                 "wrong columns are being read")
        # The 22 rows must add up to the printed TOTAL row, on both quantities. This is the
        # whole check on the name join: a row read twice or skipped breaks it.
        for k, idx in (("resident population", 0), ("population 15+", 1)):
            s = sum(v[idx] for n, v in out.items() if n != "TOTAL")
            if abs(s - out["TOTAL"][idx]) > 1:
                raise SystemExit(f"APPENDIX TABLE 1.1: the 22 units' {k} sums to {s:,} "
                                 f"against a printed TOTAL of {out['TOTAL'][idx]:,}")
        return out
    raise SystemExit(f"{NATIONAL_REPORT}: APPENDIX TABLE 1.1 not found")


def _read_status(doc, path):
    """(not-religious %, religious %) for 2020, out of ХҮСНЭГТ 3.4."""
    for i in range(FRONT_MATTER, doc.page_count):
        rows = _rows(doc[i])
        # THE LABEL IS NOT SPELLED THE SAME EVERYWHERE. Selenge's volume prints the
        # not-religious row as `Шүтлэггүй`, with an л where every other volume has a д.
        # That is a typo in the published table, so the two rows are found by their shape
        # instead: both begin `Шүт`, and the one that also carries the negative suffix
        # `гүй` is the not-religious one.
        cand = [(l, v) for l, v, _ in rows if l.startswith("Шүт") and len(v) >= 3]
        if len(cand) < 2:
            continue
        col = _col_2020(rows, min(len(v) for _, v in cand))
        if col is None:
            continue
        # Övörkhangai prints the religiosity rows TWICE on one page, once in its own table
        # and again as the first two lines of the type table below, so the first of each
        # kind is taken rather than insisting there be exactly one.
        neg = [v for l, v in cand if "гүй" in l]
        pos = [v for l, v in cand if "гүй" not in l]
        if not neg or not pos:
            continue
        a, b = neg[0][col][0], pos[0][col][0]
        # Again the identity is the finder: the two shares are a distribution and must close.
        # Six volumes merge this table into the type-of-religion one and print all seven rows
        # together, which this reads without caring.
        if abs(a + b - 100.0) < 0.15:
            return a, b
    raise SystemExit(f"{os.path.basename(path)}: no page holds a `Шүт...гүй` and a `Шүт...` "
                     "row whose 2020 shares sum to 100.0")


def _closes(d):
    """Is `d` a plausible 2020 distribution over the religion types?

    **NOT EVERY VOLUME PRINTS ALL FIVE ROWS.** Sükhbaatar has no `Ислам` row at all, because
    it has essentially no Muslims and the statistics department dropped the line rather than
    printing a zero. A missing category is therefore read as 0.0, which is what it means, and
    the guard against reading some unrelated table instead is that `Будда` must be present
    (no aimag lacks Buddhists), at least three of the five must be, and the whole thing must
    add to 100 within the rounding of one decimal place. The age-by-religion table on the
    next page, whose `Бүгд` row is five 100.0s, fails that last test at 500.
    """
    return (len(d) >= 3 and "Будда" in d and abs(sum(d.values()) - 100.0) <= 0.5)


def _read_type(doc, path):
    """({category: % of the religious} for 2020, the layout it came from).

    **THERE ARE TWO LAYOUTS AND THE CAPITAL USES THE OTHER ONE.** The twenty-one aimag
    volumes print the religions ACROSS the top and the age groups down the side, in two
    stacked blocks headed by a bare `2010` and `2020` row. Ulaanbaatar's volume transposes
    it: the religions are the ROWS and the columns are 2010 Бүгд/Эр/Эм then 2020 Бүгд/Эр/Эм,
    exactly like table 3.4 beside it. Neither layout is detectable from the caption, so both
    are tried and the one that yields all five categories wins.

    In the wide layout the category ORDER is read off the header row's x-coordinates rather
    than assumed, because the volumes do not agree on it either.
    """
    for i in range(FRONT_MATTER, doc.page_count):
        rows = _rows(doc[i])

        # ---- layout B: one row per religion, year groups across the top.
        cand = [(l, v) for l, v, _ in rows if l in CATEGORIES and len(v) >= 3]
        if len(cand) >= 3:
            col = _col_2020(rows, min(len(v) for _, v in cand))
            if col is not None:
                tall = {}
                for label, vals in cand:
                    tall.setdefault(label, vals[col][0])
                if _closes(tall):
                    return {c: tall.get(c, 0.0) for c in CATEGORIES}, sorted(tall)

        # ---- layout A: religions across the top, two stacked blocks headed `2010`/`2020`.
        order = None
        for label, _, ws in rows:
            seen = [(w[4], w[0]) for w in ws if w[4] in CATEGORIES]
            if len(seen) >= 3:
                order = [t for t, _ in sorted(seen, key=lambda p: p[1])]
                break
        if order is None:
            continue
        year, out = None, None
        for label, vals, _ in rows:
            if label in ("2010", "2020") and not vals:
                year = label
            elif (label.upper() == "БҮГД" and len(vals) == len(order) + 1
                  and vals[0][0] == 100.0):
                cand = {c: v for c, (v, _) in zip(order, vals[1:])}
                # A 2020-only table (Selenge, Sukhbaatar's third one) has no year marker.
                if (year == "2020" or year is None) and _closes(cand):
                    out = cand
        if out:
            return {c: out.get(c, 0.0) for c in CATEGORIES}, order
    raise SystemExit(f"{os.path.basename(path)}: no page holds a 2020 type-of-religion "
                     f"distribution over {CATEGORIES} summing to 100.0")


def read():
    import fitz

    pop = read_population()

    out, report = [], []
    for pcode, (name, fname) in sorted(AIMAGS.items()):
        path = os.path.join(RAW, _local(fname))
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} -- run with --fetch first")
        doc = fitz.open(path)

        pop_all, pop15 = pop[name]
        if pcode in OVERRIDE:
            none_pc, rel_pc = OVERRIDE[pcode]
        else:
            none_pc, rel_pc = _read_status(doc, path)
        types, order = _read_type(doc, path)

        counts = {NOT_RELIGIOUS: pop15 * none_pc / 100.0}
        for cat in CATEGORIES:
            counts[cat] = pop15 * rel_pc / 100.0 * types[cat] / 100.0

        for cat, n in counts.items():
            out.append({"geo_id": pcode, "geo_level": "aimag", "geo_name": name,
                        "source_category": cat, "count": int(round(n)),
                        "basis": BASIS, "year": YEAR, "source_id": SOURCE_ID,
                        "note": f"pop15={pop15}; religious_pc={rel_pc}"})
        out.append({"geo_id": pcode, "geo_level": "aimag", "geo_name": name,
                    "source_category": TOTAL_CAT, "count": pop15,
                    "basis": BASIS, "year": YEAR, "source_id": SOURCE_ID,
                    "note": "universe total (population aged 15 and over), not a category"})
        report.append((pcode, name, pop_all, pop15, none_pc, rel_pc, types, order))
    return out, report


def check(rows, report):
    ok = True

    good = len(report) == len(AIMAGS)
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {len(report)} aimag volumes parsed "
          f"(of Mongolia's {len(AIMAGS) + len(MISSING)}); "
          f"{len(MISSING)} have no published volume: "
          f"{', '.join(sorted(MISSING.values()))}")

    # the two published percentage tables are each a distribution and must close
    bad = [(p, n, a + b) for p, n, _, _, a, b, _, _ in report if abs(a + b - 100.0) > 0.15]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} `Шүтдэггүй` + `Шүтдэг` = 100.0 in all "
          f"{len(report)} volumes ({len(bad)} failures)")
    for p, n, s in bad[:5]:
        print(f"        {p} {n}: {s}")

    bad = [(p, n, sum(t.values())) for p, n, _, _, _, _, t, _ in report
           if abs(sum(t.values()) - 100.0) > 0.35]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the five religion types sum to 100.0 in all "
          f"{len(report)} volumes ({len(bad)} failures)")
    for p, n, s in bad[:5]:
        print(f"        {p} {n}: {s}")

    # the 15+ population must be a plausible slice of the aimag's own total
    bad = [(p, n, pop15 / pa) for p, n, pa, pop15, _, _, _, _ in report
           if not 0.55 <= pop15 / pa <= 0.85]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} 15+ is between 55% and 85% of each aimag's "
          f"resident population ({len(bad)} outside)")
    for p, n, r in bad[:5]:
        print(f"        {p} {n}: {r:.1%}")

    # rounding: the six category counts must reconstruct the universe
    by_unit = {}
    for r in rows:
        by_unit.setdefault(r["geo_id"], {})[r["source_category"]] = r["count"]
    bad = []
    for g, d in by_unit.items():
        s = sum(v for c, v in d.items() if c != TOTAL_CAT)
        if abs(s - d[TOTAL_CAT]) > max(50, 0.004 * d[TOTAL_CAT]):
            bad.append((g, s, d[TOTAL_CAT]))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the six categories reconstruct the 15+ "
          f"universe to within 0.4% in every aimag ({len(bad)} failures)")
    for g, s, t in bad[:5]:
        print(f"        {g}: {s:,} vs {t:,}")

    orders = {tuple(o) for _, _, _, _, _, _, _, o in report}
    print(f"\n  {len(orders)} distinct column orders across the volumes, which is why the "
          f"header is read:")
    for o in sorted(orders):
        who = [n for _, n, _, _, _, _, _, oo in report if tuple(oo) == o]
        print(f"      {' '.join(o)}   <- {len(who)} volume(s), e.g. {who[0]}")

    nat = {}
    for r in rows:
        nat[r["source_category"]] = nat.get(r["source_category"], 0) + r["count"]
    universe = nat[TOTAL_CAT]
    print(f"\n  {len(rows):,} rows. Reconstructed national totals over the "
          f"{len(report)} aimags drawn ({universe:,} people aged 15+):")
    for cat in [NOT_RELIGIOUS] + CATEGORIES:
        n = nat[cat]
        print(f"    {n:>10,}  {100.0 * n / universe:6.2f}%  {cat}")

    print("\n  For comparison, NSO's published NATIONAL figures for the 2020 census are "
          "51.7% Buddhist,\n  40.6% no religion, 3.2% Muslim, 2.5% Shamanist, 1.3% "
          "Christian, 0.7% other. This map's\n  reconstruction covers "
          f"{len(report)} of {len(AIMAGS) + len(MISSING)} aimags, so it is NOT expected to "
          "match those exactly;\n  it is expected to be close, and a large gap means the "
          "parse is wrong rather than the\n  coverage being partial.")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, report = read()
    check(rows, report)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
