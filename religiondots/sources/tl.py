"""Timor-Leste — religion by municipality, 2022 census, out of thirteen yearbook PDFs.

Writes data/normalized/tl.csv.

**THE 2022 CENSUS PUBLISHES RELIGION NATIONALLY AND NOWHERE ELSE.** INETL's *Population and
Housing Census 2022 Main Report* devotes section 3.2.2 to religion and prints exactly one
table for it, `4.07`, which is religion by five-year age group and sex for the whole country.
None of the twenty-four basic tables crosses religion with geography, and neither does any of
the eight thematic reports. A session that read the main report and stopped would record
Timor-Leste as national-only.

**THE GEOGRAPHY IS IN A SEPARATE PUBLICATION SERIES, ONE VOLUME PER MUNICIPALITY, ISSUED
THREE YEARS LATER BY A DIFFERENT DIRECTORATE.** Each of the thirteen Serviços de Estatística
Municipais publishes an annual `<Municipality> em Números`, and the 2022 edition of every one
of them carries, under *Proteção Social*, a table headed `Distribuição População por Religião`
with the 2010, 2015 and 2022 censuses side by side. Thirteen volumes, one table each, on
`inetl-ip.gov.tl/wp-content/uploads/`. That series is the only place the 2022 religion
question exists below the national total.

**THE 2015 COLUMN IS WHAT MAKES THE 2022 COLUMN TRUSTWORTHY.** These volumes are typed by
hand in the municipal offices and are not extracts of a central file; their layouts, their
category spellings and even their column headings disagree with each other. But every one of
them reprints 2015, and 2015 was published centrally, as table 11 of *Census 2015 Volume 2:
Nationality, Citizenship and Religion*. So this file reads both and checks all 252 cells of
the 2015 column, male, female and total, against the official workbook. Four are wrong, in
three volumes, and they are two different mistakes: Ainaro and Manatuto mistype a TOTAL cell
whose own male and female figures are exactly right, and Oecusse mistypes a female cell and
carries the error into its total. Everything else reproduces to the person.

**AND THE 2022 COLUMN RECONCILES AGAINST TABLE 4.07 CATEGORY BY CATEGORY**, which is the
check that matters, because it is the one the transcription could not have been copied from.
What is left over after the thirteen is Atauro, the fourteenth municipality, which has no
volume of its own; see `ATAURO` below.

**THE UNSD DEMOGRAPHIC YEARBOOK IS THE OUTSIDE WITNESS, AND IT IS THE ONLY CHECK HERE THAT
DOES NOT SHARE A PUBLISHER WITH WHAT IT IS CHECKING.** Table 28 carries Timor-Leste for 2004,
2015 and 2022, and both of the tables this file leans on reproduce from it to the person: all
seven of the 2015 figures against Volume 2's thirteen municipalities, and all five of the
named 2022 figures against main report table 4.07. Ask `tools/oracle.py` **by name**, because
it matches UNSD's own country string and a bare `tl` comes back as a miss that reads like an
absence (the `fm` review, 2026-09-08).

**ITS 2022 ROW IS A WITNESS AND NOT A SOURCE.** Table 28 names five categories and then puts
everything else in one `Other` cell of 95,328 against a total of 1,341,737. That total is the
whole resident population; the census's own religion universe is 1,248,705, the population
aged 3 and over in private households. So the oracle's residual is mostly the under-threes
rather than a religious category, and drawing it would put 7.1% of Timor-Leste in a cell that
does not exist.

Usage:
    python sources/tl.py --fetch     the thirteen volumes and the two official workbooks
    python sources/tl.py             rebuild from data/raw/tl/
"""

import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "tl")
OUT = os.path.join(ROOT, "data", "normalized", "tl.csv")

BASE = "https://inetl-ip.gov.tl/wp-content/uploads/"

# key -> (upload path, ADM1 p-code, the name COD-AB prints, the volume's own page layout)
#
# `wide` is the ordinary layout: one table, nine value columns, 2010 | 2015 | 2022 each split
# male / female / total. `viqueque` is the one volume that breaks the table up: it prints
# 2010 and 2015 BY ADMINISTRATIVE POST on one page and 2022 by municipality on the next, with
# only three value columns. Viqueque is also the ONLY volume, in any edition, that publishes
# religion below the municipality at all, and only for 2010 and 2015, which is why this map is
# drawn at ADM1 (see sources/tl.md §3). The Covalima volume prints a `Posto Administrativo`
# table on the same page and it is Bolsa da Mãe recipients, not religion; the two tables are
# told apart on their column count, which is why `_table` requires exactly `width` values.
VOLUMES = {
    "aileu":    ("2025/06/Aileu-em-Numeros-2022.pdf",     "TL02", "Aileu",    "wide"),
    "ainaro":   ("2025/07/Ainaro-em-Numeros-2022.pdf",    "TL01", "Ainaro",   "wide"),
    "baucau":   ("2025/06/Baucau-em-Numeros-2022.pdf",    "TL03", "Baucau",   "wide"),
    "bobonaro": ("2025/06/Bobonaro-em-Numeros-2022.pdf",  "TL04", "Bobonaro", "wide"),
    "covalima": ("2025/06/Covalima-em-Numeros-2022.pdf",  "TL05", "Covalima", "wide"),
    "dili":     ("2026/03/Dili-em-Numeros-2022-revisi.pdf", "TL06", "Dili",   "wide"),
    "ermera":   ("2025/06/Ermera-em-Numeros-2022.pdf",    "TL07", "Ermera",   "wide"),
    "lautem":   ("2025/06/Lautem-em-Numeros-2022.pdf",    "TL09", "Lautém",   "wide"),
    "liquica":  ("2025/06/Liquica-em-Numeros-2022.pdf",   "TL08", "Liquiçá",  "wide"),
    "manatuto": ("2025/06/Manatuto-em-Numeros-2022.pdf",  "TL11", "Manatuto", "wide"),
    "manufahi": ("2025/06/Manufahi-em-Numeros-2022.pdf",  "TL10", "Manufahi", "wide"),
    "oecusse":  ("2025/06/Oecusse-em-Numeros-2022.pdf",   "TL12", "Oecussi",  "wide"),
    "viqueque": ("2025/07/Viqueque-em-Numeros-2022.pdf",  "TL13", "Viqueque", "viqueque"),
}

# The 2015 census workbook, for the cell-by-cell check on the 2015 column, and the 2022 main
# report's basic tables, for the national reconciliation of the 2022 column.
WORKBOOKS = {
    "tl_2015_v2_religion.xls":
        "2023/03/3_2015-V2-Nationality-Citizenship-Religion.xls",
    "tl_2022_ch4_basic.xlsx":
        "2023/05/Chapter-4-TLPHC-Census-report-Basic-tables.xlsx",
}

# The nine rows, in the order every volume prints them. The labels do not agree across the
# thirteen: Baucau writes the last two in Tetum (`Laiha Rligiaun`, `La hatan`), Manufahi
# hyphenates `Musul-mano` across two lines, and Aileu's `No religion` loses its second word
# in the PDF's own text layer. So a row is classified by SEARCHING its label for any of
# these, and the resulting sequence is asserted against ORDER as well.
CATEGORY = [
    ("Catholicism", r"cat[oó]lic"),
    ("Protestantism/Evangelicalism", r"protestan"),
    ("Islam", r"musul|mu[cç]ulman|islam"),
    ("Buddhism", r"budist|buddhis"),
    ("Hinduism", r"hindu"),
    ("Indigenous religion", r"tradicion|tradision|ind[ií]gen"),
    ("Other", r"seluk|outr|other"),
    ("No religion", r"la\s*iha|laiha|no\s*religi|sem\s*religi|^no$"),
    ("No answer", r"la\s*hatan|no\s*answer|sem\s*respost|n[aã]o\s*respond"),
]
TOTAL_ROW = re.compile(r"^\s*total\s*$", re.I)

# THE OUTSIDE WITNESS. UNSD Demographic Yearbook table 28 is INETL's own return to New York
# and is a different publication from anything read here, so it is the one check on this file
# that does not share a source with the thing it is checking. `python tools/oracle.py
# "Timor-Leste"` prints it; ASK IT BY NAME, because oracle.py matches UNSD's country name and
# a bare `tl` comes back as a miss that reads like an absence (the `fm` review, 2026-09-08).
#
# It carries 2004, 2015 and 2022. Its 2015 row is asserted below against Volume 2's table 11,
# municipality by municipality, and its 2022 row against main report table 4.07 category by
# category. The 2022 row's own TOTAL is not usable and `ORACLE_2022_TOTAL` says why.
ORACLE_2015 = {
    "Catholicism": 1150990, "Protestantism/Evangelicalism": 23100, "Islam": 2824,
    "Buddhism": 560, "Hinduism": 272, "Indigenous religion": 918, "Other": 990,
}
# Table 28's 2022 row names only these five, and then puts everything else in one `Other`
# cell of 95,328 against a total of 1,341,737. That total is the whole resident population,
# not the religion universe, so its residual is mostly the under-threes; the five named
# figures are the census's own and are checked, the sixth is not a category.
ORACLE_2022 = {
    "Catholicism": 1217157, "Protestantism/Evangelicalism": 25511, "Islam": 3202,
    "Buddhism": 378, "Hinduism": 161,
}
ORACLE_2022_TOTAL = 1341737
ORACLE_2022_OTHER = 95328

# Main report table 4.07, population in private households aged 3 and over.
NATIONAL_2022 = {
    "Catholicism": 1217157, "Protestantism/Evangelicalism": 25511, "Islam": 3202,
    "Buddhism": 378, "Hinduism": 161, "Indigenous religion": 240, "Other": 1020,
    "No religion": 797, "No answer": 239,
}
UNIVERSE_2022 = 1248705          # table 4.07's own total
POPULATION_2022 = 1341737        # table 4.03: everybody, which the religion table is not

# ATAURO IS THE FOURTEENTH MUNICIPALITY AND HAS NO VOLUME. The island was a posto of Dili
# until 2022 and became a municipality of its own in time for the census, which lists it
# separately in table 4.03 at 10,295 people. No `em Números` was ever published for it, and
# the Dili volume's 2022 column excludes it. So its religion row is the national table minus
# the thirteen, which is exact arithmetic on two published tables rather than a model, and it
# is the only reason this map is not missing the most Protestant place in the country.
ATAURO = ("TL0604", "Atauro")
ATAURO_POPULATION_2022 = 10295          # table 4.03
ATAURO_BAND = (0.85, 1.00)              # its universe as a share of its population

VALUE = re.compile(r"^-?[\d,]+$")


# --------------------------------------------------------------------------------------
def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    want = [(f"{k}_2022.pdf", rel) for k, (rel, _, _, _) in VOLUMES.items()]
    want += list(WORKBOOKS.items())
    for name, rel in want:
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
            print("already have", name)
            continue
        url = BASE + rel
        print("GET", url)
        r = requests.get(url, timeout=900, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(dest, "wb") as fh:
            fh.write(r.content)
        print(f"  {len(r.content):,} bytes")
    for name in [w for w, _ in want if w.endswith(".pdf")]:
        with open(os.path.join(RAW, name), "rb") as fh:
            fh.seek(-2048, os.SEEK_END)
            if b"%%EOF" not in fh.read():
                raise SystemExit(f"{name} has no %%EOF trailer -- truncated at source, "
                                 "[[reference_pdf_truncated_at_source]]")


# --------------------------------------------------------------------------------------
def _rows(page, tol=3.0):
    """The page's words clustered into visual rows, each sorted left to right.

    A row is `(label, values)`. Everything before the first numeric token is the label and
    everything from it on is a value; a row with no numbers at all is a label fragment and is
    carried forward, which is how `Musul-` / `mano` on three separate baselines in the
    Manufahi volume comes back as one Islam row.
    """
    buckets = {}
    for x0, y0, x1, y1, w, *_ in page.get_text("words"):
        for k in buckets:
            if abs(k - y0) <= tol:
                buckets[k].append((x0, w))
                break
        else:
            buckets[y0] = [(x0, w)]

    out, pending = [], []
    for y in sorted(buckets):
        words = [w for _, w in sorted(buckets[y])]
        label, values = [], []
        for w in words:
            if VALUE.match(w) or w == "-":
                values.append(w)
            elif values:
                values = []             # text after numbers: not a data row
                break
            else:
                label.append(w)
        if not values:
            if label and len(" ".join(label)) < 40:
                pending.append(" ".join(label))
            continue
        out.append((" ".join(pending + label).strip(), values))
        pending = []
    return out


def _classify(label):
    for name, pat in CATEGORY:
        if re.search(pat, label.strip(), re.I):
            return name
    return None


def _table(page, width):
    """The religion table on `page`, as `{category: [values]}` with `width` columns each."""
    got, seen_total = {}, None
    for label, values in _rows(page):
        if TOTAL_ROW.match(label) and len(values) == width and seen_total is None:
            seen_total = [_int(v) for v in values]
            continue
        if len(values) != width:
            continue
        cat = _classify(label)
        if cat is None or cat in got:
            continue
        got[cat] = [_int(v) for v in values]
    return got, seen_total


def _int(v):
    return 0 if v == "-" else int(v.replace(",", ""))


def _page_with_table(doc, want_2022_only=False):
    for i in range(doc.page_count):
        t = doc[i].get_text()
        if not re.search(r"Religi", t, re.I) or not re.search(r"cat[oó]lic", t, re.I):
            continue
        if want_2022_only and not re.search(r"Censo\s*2022", t):
            continue
        if want_2022_only and re.search(r"Posto", t, re.I):
            continue
        return i
    return None


def read_volume(key):
    """One volume -> `{category: {2010: n, 2015: n, 2022: n}}` of TOTAL (both sexes)."""
    import fitz

    rel, pcode, name, layout = VOLUMES[key]
    path = os.path.join(RAW, f"{key}_2022.pdf")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run with --fetch first")
    doc = fitz.open(path)

    if layout == "viqueque":
        # 2022 only, on its own page, three columns: male | female | total.
        i = _page_with_table(doc, want_2022_only=True)
        if i is None:
            raise SystemExit(f"{key}: no 2022-only religion page found")
        got, total = _table(doc[i], 3)
        cells = {c: {2022: tuple(v[0:3])} for c, v in got.items()}
        printed = total[2] if total else None
        page = i + 1
    else:
        i = _page_with_table(doc)
        if i is None:
            raise SystemExit(f"{key}: no religion page found")
        got, total = _table(doc[i], 9)
        cells = {c: {2010: tuple(v[0:3]), 2015: tuple(v[3:6]), 2022: tuple(v[6:9])}
                 for c, v in got.items()}
        printed = total[8] if total else None
        page = i + 1

    if "Catholicism" not in cells:
        raise SystemExit(f"{key}: no Catholic row on page {page}")
    # Every volume prints the seven original categories; only 2022 added the last two.
    for cat in [c for c, _ in CATEGORY[:7]]:
        if cat not in cells:
            raise SystemExit(f"{key}: no {cat!r} row on page {page} -- layout changed")
    # THE 2022 COLUMN HAS TO ADD UP ROW BY ROW, because the 2015 check below proves the
    # volumes mistype total cells: Ainaro's 2015 Catholic total is the municipality's whole
    # population and Manatuto's 2015 `Seluk` total is four people too many, while in both
    # cases the male and female cells beside them are exactly right. So a 2022 row whose
    # sexes do not sum to its own total is refused here rather than carried into the file.
    for cat, per in cells.items():
        m, f, t = per[2022]
        if m + f != t:
            raise SystemExit(f"{key}: 2022 {cat} prints {m:,} + {f:,} = {m + f:,} against "
                             f"a row total of {t:,} on page {page}")
    # A printed row total, where the volume has one, must equal the categories.
    if printed is not None:
        s = sum(v[2022][2] for v in cells.values())
        if s != printed:
            raise SystemExit(f"{key}: 2022 categories sum to {s:,}, volume prints "
                             f"{printed:,} on page {page}")
    return cells, page


# --------------------------------------------------------------------------------------
def official_2015():
    """Table 11 of Census 2015 Volume 2 -> `{municipality: {category: (male, female, both)}}`.

    The table prints each municipality as three consecutive rows, the name then `Male` then
    `Female`, so all three cells of every volume's 2015 column can be checked and not just
    the total. That matters: two of the three errors the check finds are total cells whose
    own male and female figures are right.
    """
    import xlrd

    path = os.path.join(RAW, "tl_2015_v2_religion.xls")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run with --fetch first")
    sh = xlrd.open_workbook(path).sheet_by_name("2.11")
    cats = ["Catholicism", "Protestantism/Evangelicalism", "Islam", "Buddhism",
            "Hinduism", "Indigenous religion", "Other"]

    def numbers(r):
        vals = []
        for c in range(1, 9):
            v = sh.cell_value(r, c)
            if not isinstance(v, float):
                return None
            vals.append(int(v))
        return vals

    out, current = {}, None
    for r in range(sh.nrows):
        lab = str(sh.cell_value(r, 0)).strip()
        vals = numbers(r)
        if vals is None:
            continue
        low = lab.lower()
        if low == "male" and current:
            out[current]["_male"] = vals
        elif low == "female" and current:
            out[current]["_female"] = vals
        elif lab:
            current = lab.upper()
            out[current] = {"_both": vals}
    return {k: {c: (v["_male"][i + 1], v["_female"][i + 1], v["_both"][i + 1])
                for i, c in enumerate(cats)}
            for k, v in out.items() if "_male" in v and "_female" in v}


# The workbook's printed municipality strings, which are neither COD-AB's nor the volumes'.
PRINTED_2015 = {
    "aileu": "AILEU", "ainaro": "AINARO", "baucau": "BAUCAU", "bobonaro": "BOBONARO",
    "covalima": "COVALIMA", "dili": "DILI", "ermera": "ERMERA", "lautem": "LAUTÉM",
    "liquica": "LIQUIÇA", "manatuto": "MANATUTO", "manufahi": "MANUFAHI",
    "oecusse": "SAR1 OF OECUSSE", "viqueque": "VIQUEQUE",
}
# THE FOUR CELLS IN THE WHOLE 2015 COLUMN THAT THE VOLUMES GET WRONG, keyed
# `(volume, category, male|female|both) -> (what the volume prints, what the workbook says)`.
# They are recorded rather than corrected, because nothing is drawn from the 2015 column;
# what they buy is a calibrated answer to "how carefully were these volumes typed", which is
# the only question the 2022 column raises and the only one nothing else can answer.
#
# Two of them are TOTAL cells whose own male and female figures are exactly right, which is
# why `read_volume` refuses a 2022 row that does not add up. Oecusse's is the other kind: it
# mistypes a female cell as 31,962 and carries the 2,000 into its total, so the row is
# internally consistent and only the workbook catches it.
KNOWN_2015_MISMATCH = {
    ("ainaro", "Catholicism", "both"): (62988, 62388),
    ("manatuto", "Other", "both"): (97, 93),
    ("oecusse", "Catholicism", "female"): (31962, 33962),
    ("oecusse", "Catholicism", "both"): (66402, 68402),
}


def main():
    if "--fetch" in sys.argv:
        fetch()

    vols, pages = {}, {}
    for key in VOLUMES:
        vols[key], pages[key] = read_volume(key)
        print(f"read {key:<9} page {pages[key]:>3}  "
              f"{len(vols[key])} categories")

    # ---- check 0: the outside witness, UNSD table 28, against both official tables ----
    off = official_2015()
    print("\nUNSD Demographic Yearbook table 28, INETL's own return to New York:")
    for cat, want in ORACLE_2015.items():
        got = sum(v[cat][2] for k, v in off.items() if k != "TIMOR LESTE"
                  and not k.startswith("URBAN") and not k.startswith("RURAL"))
        print(f"  2015 {cat:<30} municipalities {got:>10,}   UNSD {want:>10,}")
        if got != want:
            raise SystemExit(f"Volume 2 table 11's municipalities sum to {got:,} for {cat} "
                             f"and UNSD says {want:,}")
    for cat, want in ORACLE_2022.items():
        print(f"  2022 {cat:<30} table 4.07    {NATIONAL_2022[cat]:>10,}   "
              f"UNSD {want:>10,}")
        if NATIONAL_2022[cat] != want:
            raise SystemExit(f"table 4.07 says {NATIONAL_2022[cat]:,} for {cat} and UNSD "
                             f"says {want:,}")
    resid = ORACLE_2022_TOTAL - sum(ORACLE_2022.values())
    if resid != ORACLE_2022_OTHER:
        raise SystemExit(f"UNSD's 2022 row no longer leaves {ORACLE_2022_OTHER:,} in "
                         f"`Other`; it leaves {resid:,}")
    print(f"  2022 UNSD's sixth cell is {ORACLE_2022_OTHER:,} against a total of "
          f"{ORACLE_2022_TOTAL:,}, which is the population and not the religion universe; "
          f"{POPULATION_2022 - UNIVERSE_2022:,} of it is people the question never reached")

    # ---- check 1: the 2015 column, cell by cell, against the official workbook ----
    checked = bad = known = 0
    for key, cells in vols.items():
        if VOLUMES[key][3] == "viqueque":
            continue                      # its 2015 table is by administrative post
        ref = off[PRINTED_2015[key]]
        for cat, want in ref.items():
            if 2015 not in cells.get(cat, {}):
                continue
            for i, which in enumerate(("male", "female", "both")):
                checked += 1
                got = cells[cat][2015][i]
                if got == want[i]:
                    continue
                if KNOWN_2015_MISMATCH.get((key, cat, which)) == (got, want[i]):
                    known += 1
                    print(f"  KNOWN: {key} 2015 {cat} {which} prints {got:,}, "
                          f"workbook {want[i]:,}")
                    continue
                bad += 1
                print(f"  MISMATCH: {key} 2015 {cat} {which} prints {got:,}, "
                      f"workbook {want[i]:,}")
    print(f"\n2015 cross-check: {checked} cells against Census 2015 Volume 2 table 11, "
          f"{known} known errors, {bad} unexplained")
    if bad or known != len(KNOWN_2015_MISMATCH):
        raise SystemExit("the volumes and the 2015 workbook disagree in a way this file "
                         "does not already know about -- do not write")

    # ---- check 2: the 2022 column against main report table 4.07, category by category ----
    # WHAT IS LEFT OVER IS ATAURO, and this is where the fourteenth municipality is
    # recovered. The residual is required to be non-negative in every category and to sit
    # inside a band around Atauro's own published population; the reason to trust it beyond
    # that is that the three categories Atauro plainly has none of -- Islam, Buddhism,
    # indigenous religion -- come out at exactly zero without being told to.
    print("\n2022 reconciliation against Census 2022 main report table 4.07:")
    residual, tot = {}, 0
    for cat, want in NATIONAL_2022.items():
        got = sum(v.get(cat, {}).get(2022, (0, 0, 0))[2] for v in vols.values())
        tot += got
        residual[cat] = want - got
        print(f"  {cat:<30} {got:>10,}   national {want:>10,}"
              f"   Atauro {residual[cat]:>7,}")
    print(f"  {'TOTAL':<30} {tot:>10,}   national {UNIVERSE_2022:>10,}"
          f"   Atauro {UNIVERSE_2022 - tot:>7,}")
    neg = {c: n for c, n in residual.items() if n < 0}
    if neg:
        raise SystemExit(f"the thirteen volumes OVERSHOOT table 4.07 in {neg} -- the "
                         "residual cannot be Atauro, do not write")
    at = UNIVERSE_2022 - tot
    share = at / ATAURO_POPULATION_2022
    lo, hi = ATAURO_BAND
    print(f"\nAtauro: {at:,} people in the religion universe against a census population of "
          f"{ATAURO_POPULATION_2022:,} ({share:.3f} of it; the country is "
          f"{UNIVERSE_2022 / POPULATION_2022:.3f})")
    if not lo <= share <= hi:
        raise SystemExit(f"the residual is {share:.3f} of Atauro's population, outside "
                         f"{lo}-{hi} -- it is not Atauro, do not write")

    print(f"\nuniverse: {UNIVERSE_2022:,} of {POPULATION_2022:,} people "
          f"({100.0 * UNIVERSE_2022 / POPULATION_2022:.2f}%); the rest are the under-threes "
          "and the population outside private households, whom the question never reached")

    # ---- write ----
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for key, cells in vols.items():
        rel, pcode, name, _ = VOLUMES[key]
        for cat, _pat in CATEGORY:
            n = cells.get(cat, {}).get(2022, (0, 0, 0))[2]
            if n <= 0:
                continue
            rows.append({
                "geo_id": pcode,
                "geo_level": "municipality",
                "geo_name": name,
                "source_category": cat,
                "count": n,
                "basis": "self_id",
                "year": 2022,
                "source_id": "tl_phc_2022_emnumeros",
                "note": f"volume={key} em Numeros 2022; page={pages[key]}; adm1={pcode}",
            })
    for cat, n in residual.items():
        if n <= 0:
            continue
        rows.append({
            "geo_id": ATAURO[0],
            "geo_level": "municipality",
            "geo_name": ATAURO[1],
            "source_category": cat,
            "count": n,
            "basis": "self_id",
            "year": 2022,
            "source_id": "tl_phc_2022_t407_residual",
            "note": f"residual=yes; main report table 4.07 minus the thirteen volumes; "
                    f"adm1={ATAURO[0]}",
        })
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["geo_id", "geo_level", "geo_name",
                                           "source_category", "count", "basis", "year",
                                           "source_id", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}: {len(rows)} rows, "
          f"{len({r['geo_id'] for r in rows})} municipalities, "
          f"{sum(r['count'] for r in rows):,} people")


if __name__ == "__main__":
    main()
