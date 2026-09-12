"""Benin — INStaD, RGPH-4 (2013), the twelve departmental *Principaux indicateurs*.

Reads (or fetches) data/raw/bj/ and writes data/normalized/bj.csv.

**Ten religion categories on 77 communes for 10,008,749 people** — about 130,000 per unit,
finer per head than Malawi's districts and eight times finer than Kenya's counties. It is
**the first Vodun mapping on this project**: `Vodoun` has its own cell, separate from
`Autres traditionnelles`, and it runs from 56.5% of the Couffo department to 0.4% in the
Alibori. `Chrétien céleste` — the Celestial Church of Christ, founded in Porto-Novo in 1947
— has its own cell too, and is only the second source here to feed
`christianity.africaninstituted`.

**THE TABLE IS PERCENTAGES AND THE COUNTS ARE THIS MODULE'S ARITHMETIC.** Tableau 8 prints
shares to one decimal and no counts at all. sources.md §11p recorded that as a cost needing
a join to the *Résultats définitifs*; it does not. **Tableau 2 of the same booklet prints the
commune's population**, so each booklet is self-contained and `count = pct/100 x total`
with no cross-document join anywhere. The precision cost is bounded and stated: one decimal
on a share is +/-0.05% of the unit, which is +/-34 people in a 68,000-person commune and
+/-340 in Cotonou. Nothing is inferred — every person in this file was counted by INStaD in
the commune they are drawn in — so the rows are `measured` (§7a) and the caveat is
arithmetic precision, not confidence.

**THE TEN SHARES DO NOT SUM TO 100 AND THE REMAINDER IS NON-RESPONSE.** Nationally they sum
to 98.81%, and the gap is 1.19% — 119,094 people. RGPH-4's own religion tabulations carry
exactly these ten categories wherever they appear (the fertility tables in *Synthèse des
analyses* TOME 2 use the same ten), so the undeclared are excluded from the categories and
present in the denominator. That is far too large to be rounding: ten values each rounded to
0.1 have a standard error of 0.09pp against an observed 0.6-2.0pp per commune. The residual
is emitted as `Non déclaré (calculé)` so the arithmetic is visible, and taxonomy/bj2013.py
excludes it — §3.5 marks non-response, it does not fill it.

**PARSED BY COLUMN GEOMETRY, NOT BY LINES.** Read as text lines the table returns the first
value of a row on its own and the rest run together, so the column identity is lost. This
module takes the words with their coordinates, groups them into rows by y, finds the data
rows by counting percentage-shaped tokens, and takes each column's x-centre as the median
over those rows. **The header is then reconstructed by assigning every header word to its
nearest column**, which is what handles the wrapped names — `Abomey-` above `Calavi`, and
`Akpro-` above `Missérété` — without any special case, and produces a name per column that
is checked against the published commune list.

**`(*)` IS A SENTINEL AND IT MEANS "UNDER 0.1%", NOT "MISSING".** The booklets say so in a
footnote — *NB: (*) Valeurs inférieures à 0,1%*. **As it happens no religion cell in any of
the twelve uses it**: it appears in Tableau 2's age distribution, and on the Tableau 8 pages
of Alibori, Borgou and Couffo only inside that footnote. The parser handles it anyway and
`check()` reports the count, because the failure it prevents is silent and total — a
numeric-only regex drops the token, the row then holds one value too few, and **every
remaining figure in that row shifts one column left**, which is Malawi's Likoma trap (§9bb)
with a whole row's blast radius instead of a cell's. Read as 0, which is the sentinel's
lower bound and not an invention.

**FOUR CHECKS, AND THE STRONGEST ONE IS A DIFFERENT FILE.** INStaD also publishes
`Doc_principaux_indicateurs.xlsx`, whose `ETHNIE_RELIGION` sheet carries the same shares at
**full float precision** for the nation and all twelve departments. Every booklet's rounded
department column is checked against it — 120 equations, in a different format, from a
different page of the office's site — and it is the only check here that would catch a whole
department's column being read into the wrong place. Beside it: the communes sum to their
department, the departments sum to the nation, and Cotonou's thirteen arrondissements sum to
Cotonou. All four are bands computed from the rounding, never equalities (§12).

**INStaD MISSPELLS ITS OWN RELEASE IN THREE PLACES AND ALL THREE ARE LOAD-BEARING.**
`Principaux` is spelled four different ways across the twelve filenames (`Princiapux`,
`Principaunx`, `Principaux-idnicateurs`, `Principaux-idndicateurs`); the Ouémé booklet's
table title reads `PAR COMMMUNE` with three Ms; and the workbook's row label is
`Autres Réligion`. The filenames are transcribed verbatim in `BOOKLETS`, the title regex
tolerates the extra M, and the workbook labels have their own mapping. None of this is
repaired — the source's spelling is the key (§12).

Usage:
    python sources/bj.py --fetch    twelve ~1 MB PDFs and one 115 KB xlsx, seconds
    python sources/bj.py            normalise from data/raw/bj/
"""

import csv
import os
import re
import statistics
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bj")
OUT = os.path.join(ROOT, "data", "normalized", "bj.csv")

SOURCE_ID = "bj_rgph4_2013"
YEAR = 2013
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

BASE = "https://rgph5.instad.bj/wp-content/uploads/2023/03/"

# The workbook lives on the office's OTHER site — instad.bj, a Joomla install — under a path
# with spaces in it. rgph5.instad.bj is the RGPH portal and does not carry it.
XLSX_URL = ("https://instad.bj/images/docs/insae-statistiques/enquetes-recensements/RGPH/"
            "1.RGPH_4/resultats%20finaux/Resultats%20globaux/Doc_principaux_indicateurs.xlsx")
XLSX_NAME = "bj_doc_principaux_indicateurs.xlsx"

# INStaD's own filenames, verbatim. Four spellings of `Principaux` in one release; do not
# "fix" them, they are the URLs.
BOOKLETS = {
    "Alibori": "Principaux-indicateurs-Alibori_final.pdf",
    "Atacora": "Principaux-indicateurs-Atacora_final.pdf",
    "Atlantique": "Principaux-indicateurs-Atlantique_Final.pdf",
    "Borgou": "Principaux-indicateurs-du-Borgou_Final.pdf",
    "Collines": "Princiapux-indicateurs-des-Collines_Final.pdf",
    "Couffo": "Principaux-idnicateurs-Couffo_Final.pdf",
    "Donga": "Principaux-indicateurs-Donga_Final.pdf",
    "Littoral": "Principaux-indicateurs-Littoral_Final.pdf",
    "Mono": "Princiapux-indicateurs-Mono_Final.pdf",
    "Ouémé": "Principaux-idndicateurs-Oueme_Final.pdf",
    "Plateau": "Principaunx-indicateurs-Plateau_Final.pdf",
    "Zou": "Principaux-indicateurs-Zou_Final.pdf",
}

# Department -> (code, communes in the order INStaD prints them, which is alphabetical and
# is also the order of COD's adm2_pcode — asserted in sources/bj_geo.py, not assumed here).
# The names are INStaD's own spellings as they come off Tableau 8; `bj_geo.py` folds them.
DEPARTMENTS = {
    "Alibori": ("01", ["Banikoara", "Gogounou", "Kandi", "Karimama", "Malanville",
                       "Ségbana"]),
    "Atacora": ("02", ["Boukoumbé", "Cobly", "Kérou", "Kouandé", "Matéri", "Natitingou",
                       "Péhunco", "Tanguiéta", "Toucountouna"]),
    "Atlantique": ("03", ["Abomey-Calavi", "Allada", "Kpomassè", "Ouidah", "So-Ava",
                          "Toffo", "Torri-Bossito", "Zè"]),
    "Borgou": ("04", ["Bembèrèkè", "Kalalé", "N'dali", "Nikki", "Parakou", "Pèrèrè",
                      "Sinendé", "Tchaourou"]),
    "Collines": ("05", ["Bantè", "Dassa", "Glazoué", "Ouessè", "Savalou", "Savè"]),
    "Couffo": ("06", ["Aplahoué", "Djakotomey", "Dogbo", "Klouekanmè", "Lalo",
                      "Toviklin"]),
    "Donga": ("07", ["Bassila", "Copargo", "Djougou", "Ouaké"]),
    # Littoral is one commune and Tableau 8's first column IS Cotonou; there is no separate
    # department column. The thirteen that follow are its arrondissements, not communes.
    "Littoral": ("08", ["Cotonou"]),
    "Mono": ("09", ["Athiémé", "Bopa", "Comè", "Grand-Popo", "Houéyogbé", "Lokossa"]),
    "Ouémé": ("10", ["Adjarra", "Adjohoun", "Aguégués", "Akpro-Missérété", "Avrankou",
                     "Bonou", "Dangbo", "Porto-Novo", "Sèmè-Kpodji"]),
    "Plateau": ("11", ["Adja-Ouèrè", "Ifangni", "Kétou", "Pobè", "Sakété"]),
    "Zou": ("12", ["Abomey", "Agbangnizoun", "Bohicon", "Covè", "Djidja", "Ouinhi",
                   "Zagnanado", "Za-Kpota", "Zogbodomey"]),
}
EXPECTED_COMMUNES = 77
# geo_ids are POSITIONAL and the hyphen is there to say so: `BJ12-07` is the seventh commune
# printed in the Zou booklet and is deliberately not shaped like COD's `BJ1207`, because the
# two orders are alphabetical under different spellings and disagree on three pairs. See
# sources/bj_geo.py, which pairs by name and treats the order as evidence rather than a key.
COTONOU = "BJ08-01"                 # the Littoral's single commune
COTONOU_ARRONDISSEMENTS = 13

# Down Tableau 8's religion block, in the order INStaD prints them.
CATEGORIES = [
    "Vodoun",
    "Catholique",
    "Protestant méthodiste",
    "Autres protestants",
    "Chrétien céleste",
    "Islam",
    "Autres chrétiens",
    "Autres traditionnelles",
    "Autres religions",
    "Aucune",
]
# Computed here, never printed by INStaD: the complement of the ten published shares.
RESIDUAL = "Non déclaré (calculé)"

# The workbook spells four of the ten differently, and misspells `Religion`.
XLSX_LABELS = {
    "Vodoun": "Vodoun",
    "Catholique": "Catholique",
    "Protestant méthodiste": "Protestants Méthodistes",
    "Autres protestants": "Autres Protestants",
    "Chrétien céleste": "Céleste",
    "Islam": "Islam",
    "Autres chrétiens": "Autres Chrétiens",
    "Autres traditionnelles": "Autres Traditionnelle",
    "Autres religions": "Autres Réligion",
    "Aucune": "Aucune",
}

NATIONAL = 10_008_749               # RGPH-4's published resident population

# Ouémé's Tableau 8 is titled `PAR COMMMUNE`. The extra M is INStaD's.
T2_RE = re.compile(r"Tableau\s*2\s*:\s*STRUCTURE DE LA POPULATION PAR "
                   r"(COMM?MUNE|ARRONDISSEMENT)", re.I)
T8_RE = re.compile(r"Tableau\s*8\s*:\s*ETHNIE ET RELIGION PAR "
                   r"(COMM?MUNE|ARRONDISSEMENT)", re.I)

PCT = re.compile(r"^\d{1,3},\d$")
DIGITS = re.compile(r"^\d{1,3}$")
STAR = "(*)"

# A percentage printed to one decimal is this far from the truth, as a share.
ROUNDING = 0.0005


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for dep, fn in BOOKLETS.items():
        dest = os.path.join(RAW, _pdf_name(dep))
        if os.path.exists(dest) and os.path.getsize(dest) > 200_000:
            print(f"  have {dep}")
            continue
        url = BASE + fn
        print("GET", url)
        r = requests.get(url, timeout=600, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        # §5a: HTTP 200 is not a download. Assert the type, not the absence of an exception.
        if r.content[:5] != b"%PDF-":
            raise SystemExit(f"{dep}: not a PDF -- starts {r.content[:20]!r}, "
                             f"{len(r.content):,} bytes")
        with open(dest, "wb") as fh:
            fh.write(r.content)
        print(f"  {dep:<12} {len(r.content):>9,} bytes")

    dest = os.path.join(RAW, XLSX_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 50_000:
        print("  have the workbook")
        return
    print("GET", XLSX_URL)
    r = requests.get(XLSX_URL, timeout=600, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    if r.content[:4] != b"PK\x03\x04":
        raise SystemExit(f"the workbook is not an xlsx -- starts {r.content[:20]!r}")
    with open(dest, "wb") as fh:
        fh.write(r.content)
    print(f"  workbook     {len(r.content):>9,} bytes")


def _pdf_name(dep):
    return "pi_" + _ascii(dep).lower() + ".pdf"


def _ascii(s):
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def fold(s):
    """Loose key for comparing a printed name with a published one."""
    return re.sub(r"[^a-z0-9]+", "", _ascii(s).lower())


def _find_page(doc, pat, what, path):
    for i in range(doc.page_count):
        if pat.search(" ".join(doc[i].get_text().split())):
            return i
    raise SystemExit(f"{path}: no page matching {what} -- INStaD has re-issued the "
                     "booklet, or re-titled the table")


def _words(page):
    return [(w[0], w[1], w[2], w[3], w[4]) for w in page.get_text("words")]


def _data_rows(words, is_value, min_values):
    """Rows of words that carry at least `min_values` value-shaped tokens.

    Clustered on the vertical CENTRE with a tolerance, swept in order, rather than binned
    on the top edge. A label and its figures do not share a baseline — the label is often a
    point smaller and sits a little lower — and any fixed binning splits some rows in two,
    which loses the label from the row and makes the table look as though it were missing
    seven of its ten categories. Atacora and the Littoral both do this.
    """
    rows = []
    for w in sorted(words, key=lambda w: ((w[1] + w[3]) / 2.0, w[0])):
        mid = (w[1] + w[3]) / 2.0
        if rows and mid - rows[-1][0] <= 5.0:
            rows[-1][1].append(w)
        else:
            rows.append((mid, [w]))
    out = []
    for _, ws in rows:
        vals = sorted((w for w in ws if is_value(w[4])), key=lambda w: w[0])
        if len(vals) >= min_values:
            out.append((sorted(ws, key=lambda w: w[0]), vals))
    return out


def _columns(rows, path, what):
    """Column x-centres, from the rows that carry the full complement of values."""
    n = max(len(v) for _, v in rows)
    full = [v for _, v in rows if len(v) == n]
    if len(full) < 3:
        raise SystemExit(f"{path}: {what} has only {len(full)} full rows of {n} values")
    return n, [statistics.median((v[j][0] + v[j][2]) / 2 for v in full) for j in range(n)]


def _band(words, rows):
    """The y range holding the header: below the title, above the first data row.

    **The title's own number is a digit token and it sits inside the column band.** `Tableau
    2 :` puts a bare `2` at x=115, which is 14pt from the Littoral's and the Plateau's first
    column and 190pt from everybody else's — so reading the population row without cutting
    the title off gives Cotonou 2,679,012 people instead of 679,012 and the Plateau
    2,622,372 instead of 622,372, in two booklets of twelve, with every other department
    correct. Both figures are plausible, both parse, and the only thing that sees it is the
    national sum.
    """
    y_first = min(w[1] for ws, _ in rows for w in ws)
    y_title = max((w[1] for w in words if w[4] in ("Tableau", "ETHNIE", "STRUCTURE")),
                  default=0.0)
    return y_title + 2, y_first - 2


def _header(words, rows, cols, path):
    """One name per column, built by giving every header word to its nearest column.

    This is what absorbs the wrapped names — `Abomey-` sits above `Calavi` and both are
    nearest the same column, so they join in reading order with no special case.
    """
    lo, hi = _band(words, rows)
    names = {j: [] for j in range(len(cols))}
    for w in sorted((w for w in words if lo < w[1] < hi), key=lambda w: (w[1], w[0])):
        if DIGITS.match(w[4]):
            continue                      # Tableau 2's population row, handled separately
        cx = (w[0] + w[2]) / 2.0
        j = min(range(len(cols)), key=lambda k: abs(cols[k] - cx))
        if abs(cols[j] - cx) < 42:
            names[j].append(w[4])
    out = [" ".join(names[j]) for j in range(len(cols))]
    if not all(out):
        raise SystemExit(f"{path}: header column(s) {[j for j, s in enumerate(out) if not s]}"
                         " came back empty -- the table has been re-laid out")
    return out


def _cell(tok, where):
    """One printed share -> a fraction. `(*)` means under 0.1% and must not be dropped."""
    if tok == STAR:
        return None
    if not PCT.match(tok):
        raise SystemExit(f"{where}: {tok!r} is neither a percentage nor {STAR!r} -- "
                         "Tableau 8 has been re-typeset and the row would silently shift")
    return float(tok.replace(",", ".")) / 100.0


def _read_populations(doc, path):
    """Tableau 2: the population of every column, as ints."""
    page = doc[_find_page(doc, T2_RE, "Tableau 2", path)]
    words = _words(page)
    rows = _data_rows(words, PCT.match, 4)
    n, cols = _columns(rows, path, "Tableau 2")

    # The population row is the digit row between the title and the first percentage row.
    # Its figures are split into thousands groups by the printed space, so the tokens of
    # one number arrive as separate words and are rejoined by column.
    lo, hi = _band(words, rows)
    groups = {j: [] for j in range(n)}
    for w in sorted((w for w in words if DIGITS.match(w[4]) and lo < w[1] < hi),
                    key=lambda w: (w[1], w[0])):
        cx = (w[0] + w[2]) / 2.0
        j = min(range(n), key=lambda k: abs(cols[k] - cx))
        if abs(cols[j] - cx) < 46:
            groups[j].append((w[1], w[0], w[4]))
    pops = []
    for j in range(n):
        toks = [t for _, _, t in sorted(groups[j])]
        if not toks:
            raise SystemExit(f"{path}: Tableau 2 column {j} has no population figure")
        pops.append(int("".join(toks)))
    return _header(words, rows, cols, path), pops


def _read_shares(doc, path):
    """Tableau 8: the ten religion shares for every column."""
    page = doc[_find_page(doc, T8_RE, "Tableau 8", path)]
    words = _words(page)
    rows = _data_rows(words, lambda t: bool(PCT.match(t)) or t == STAR, 4)
    n, cols = _columns(rows, path, "Tableau 8")
    names = _header(words, rows, cols, path)

    # The label is everything left of the row's first value. Match the religion block by
    # its labels rather than by position, so an added ethnicity row cannot shift it.
    want = {fold(c): c for c in CATEGORIES}
    found = {}
    for ws, vals in rows:
        if len(vals) != n:
            continue
        label = " ".join(w[4] for w in ws if w[2] <= vals[0][0] - 1)
        key = fold(label.replace("(%)", ""))
        cat = want.get(key)
        if cat is None:
            continue
        if cat in found:
            raise SystemExit(f"{path}: Tableau 8 prints {cat!r} twice")
        found[cat] = [_cell(v[4], f"{path} {cat!r}") for v in vals]
    missing = [c for c in CATEGORIES if c not in found]
    if missing:
        raise SystemExit(f"{path}: Tableau 8 is missing {missing} -- INStaD has changed "
                         "the category list and the mapping must be revisited")
    return names, found


def read():
    import fitz

    rows, meta = [], {}
    for dep, (code, communes) in DEPARTMENTS.items():
        path = os.path.join(RAW, _pdf_name(dep))
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} -- run with --fetch first")
        doc = fitz.open(path)

        h2, pops = _read_populations(doc, path)
        h8, shares = _read_shares(doc, path)
        if len(h2) != len(h8):
            raise SystemExit(f"{dep}: Tableau 2 has {len(h2)} columns and Tableau 8 has "
                             f"{len(h8)} -- they are supposed to be the same units")
        # The two tables of one booklet are independent reads of the same column order.
        # Couffo writes `Klouékanmè` in Tableau 2 and `Klouekanmè` in Tableau 8, so the
        # comparison is folded; a transposed column is not survivable by folding.
        bad = [(a, b) for a, b in zip(h2, h8) if fold(a) != fold(b)]
        if bad:
            raise SystemExit(f"{dep}: Tableau 2 and Tableau 8 disagree on the column "
                             f"order/names: {bad}")

        # Column 0 is the department, except in the Littoral where the department IS
        # Cotonou and column 0 is the commune itself.
        is_littoral = dep == "Littoral"
        expected = communes if is_littoral else [dep] + communes
        if not is_littoral and len(h8) != 1 + len(communes):
            raise SystemExit(f"{dep}: {len(h8)} columns, expected "
                             f"{1 + len(communes)} (the department and its "
                             f"{len(communes)} communes)")
        if is_littoral and len(h8) != 1 + COTONOU_ARRONDISSEMENTS:
            raise SystemExit(f"Littoral: {len(h8)} columns, expected "
                             f"{1 + COTONOU_ARRONDISSEMENTS} (Cotonou and its "
                             f"{COTONOU_ARRONDISSEMENTS} arrondissements)")
        head = h8[:len(expected)]
        if [fold(x) for x in head] != [fold(x) for x in expected]:
            first = next((a, b) for a, b in zip(head, expected) if fold(a) != fold(b))
            raise SystemExit(f"{dep}: printed column {first[0]!r} where the published "
                             f"commune list says {first[1]!r} -- the order has changed")

        for j, name in enumerate(h8):
            # The Littoral IS Cotonou, so its one column is both the department and the
            # commune — but the department row is named for the department, or bj_geo.py's
            # department-scoped join finds a `Cotonou` where COD has a `Littoral`.
            if is_littoral:
                if j == 0:
                    ids = [(f"BJ{code}", "department", dep), (COTONOU, "commune", name)]
                else:
                    ids = [(f"{COTONOU}-A{j:02d}", "arrondissement", name)]
            elif j == 0:
                ids = [(f"BJ{code}", "department", name)]
            else:
                ids = [(f"BJ{code}-{j:02d}", "commune", name)]

            total = pops[j]
            drawn = 0
            cells = []
            for cat in CATEGORIES:
                share = shares[cat][j]
                if share is None:
                    cells.append((cat, 0, True))
                else:
                    n = int(round(share * total))
                    drawn += n
                    cells.append((cat, n, False))
            cells.append((RESIDUAL, total - drawn, False))

            for geo_id, level, geo_name in ids:
                for cat, n, starred in cells:
                    note = f"level={level}; from a share printed to 0.1% x a population " \
                           f"of {total:,}"
                    if starred:
                        note += f"; the share is `{STAR}`, under 0.1%, so this is a floor"
                    if cat == RESIDUAL:
                        note = (f"level={level}; computed as {total:,} minus the ten "
                                "published categories; INStaD prints no such row")
                    rows.append({"geo_id": geo_id, "geo_level": level,
                                 "geo_name": geo_name,
                                 "source_category": cat, "count": n, "basis": BASIS,
                                 "year": YEAR, "source_id": SOURCE_ID, "note": note})
                meta[geo_id] = dict(level=level, name=geo_name, dep=dep, code=code,
                                    total=total, shares={c: shares[c][j]
                                                         for c in CATEGORIES})
    return rows, meta


def _read_workbook():
    """The full-precision national and departmental shares, from the office's workbook."""
    import openpyxl

    path = os.path.join(RAW, XLSX_NAME)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run with --fetch first")
    ws = openpyxl.load_workbook(path, data_only=True)["ETHNIE_RELIGION"]
    grid = [[c for c in row] for row in ws.iter_rows(values_only=True)]
    head = [("" if v is None else str(v)).strip() for v in grid[0]]
    cols = {}
    for want in ["Bénin"] + list(DEPARTMENTS):
        matches = [i for i, h in enumerate(head) if fold(h) == fold(want)]
        if len(matches) != 1:
            raise SystemExit(f"the workbook has {len(matches)} columns for {want!r}")
        cols[want] = matches[0]

    out = {}
    for cat, label in XLSX_LABELS.items():
        hits = [r for r in grid if r and fold(str(r[0] or "")) == fold(label)]
        if len(hits) != 1:
            raise SystemExit(f"the workbook has {len(hits)} rows for {label!r} "
                             f"(the {cat!r} row)")
        out[cat] = {k: float(hits[0][i]) / 100.0 for k, i in cols.items()}
    return out


def check(rows, meta, book):
    ok = True

    levels = {}
    for r in rows:
        levels.setdefault(r["geo_level"], set()).add(r["geo_id"])
    for lv, want in (("department", len(DEPARTMENTS)),
                     ("commune", EXPECTED_COMMUNES),
                     ("arrondissement", COTONOU_ARRONDISSEMENTS)):
        got = len(levels.get(lv, ()))
        good = got == want
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {lv:<15} {got:>4} units (expected {want})")

    dep_total = sum(m["total"] for m in meta.values() if m["level"] == "department")
    com_total = sum(m["total"] for m in meta.values() if m["level"] == "commune")
    arr_total = sum(m["total"] for m in meta.values() if m["level"] == "arrondissement")
    for what, got in (("the 12 departments", dep_total),
                      ("the 77 communes", com_total)):
        good = got == NATIONAL
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} Tableau 2: {what} sum to {got:,} "
              f"(expected {NATIONAL:,})")
    good = arr_total == meta[COTONOU]["total"]
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} Tableau 2: Cotonou's {COTONOU_ARRONDISSEMENTS} "
          f"arrondissements sum to {arr_total:,} (Cotonou is "
          f"{meta[COTONOU]['total']:,})")

    # ---- the workbook, which is the only check from outside this document family ----
    print(f"\n  the office's own workbook, at full float precision — 120 equations "
          f"a booklet's column\n  cannot satisfy unless it was read into the right "
          f"department:")
    bad = []
    for dep, (code, _) in DEPARTMENTS.items():
        m = meta[f"BJ{code}"]
        for cat in CATEGORIES:
            got, want = m["shares"][cat], book[cat][dep]
            if got is None:
                if want >= 0.001:
                    bad.append((dep, cat, "(*)", f"{want:.5%}"))
            elif abs(got - want) > ROUNDING + 1e-9:
                bad.append((dep, cat, f"{got:.3%}", f"{want:.5%}"))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} all {len(DEPARTMENTS) * len(CATEGORIES)} "
          f"departmental shares agree to within the printed rounding ({len(bad)} failures)")
    for dep, cat, got, want in bad[:6]:
        print(f"        {dep}/{cat}: booklet {got} vs workbook {want}")

    nat_share = {c: book[c]["Bénin"] for c in CATEGORIES}
    s = sum(nat_share.values())
    print(f"\n  the ten published shares sum to {s:.4%} nationally, so "
          f"{(1 - s) * NATIONAL:,.0f} people\n  ({1 - s:.4%}) are undeclared and are not "
          f"drawn — §3.5. Ten values rounded to 0.1pp\n  have a standard error of 0.09pp, "
          f"so this is a category and not the rounding.")

    # ---- the communes against their department, in the band the rounding allows ----
    by_unit = {}
    for r in rows:
        by_unit.setdefault((r["geo_level"], r["geo_id"]), {})[r["source_category"]] = \
            r["count"]
    bad = []
    for dep, (code, communes) in DEPARTMENTS.items():
        kids = [k for lv, k in by_unit
                if lv == "commune" and (k.startswith(f"BJ{code}")
                                        or (dep == "Littoral" and k == COTONOU))]
        # +/-0.05% of each child's own population, plus the same on the parent's figure.
        band = ROUNDING * (sum(meta[k]["total"] for k in kids)
                           + meta[f"BJ{code}"]["total"])
        for cat in CATEGORIES:
            got = sum(by_unit[("commune", k)][cat] for k in kids)
            want = by_unit[("department", f"BJ{code}")][cat]
            if abs(got - want) > band + 1:
                bad.append((dep, cat, got, want, band))
    ok &= not bad
    print(f"\n  {'OK ' if not bad else 'BAD'} the communes sum to their department on all "
          f"{len(CATEGORIES)} categories, within the\n      band the printed rounding "
          f"allows ({len(bad)} failures)")
    for dep, cat, got, want, band in bad[:6]:
        print(f"        {dep}/{cat}: {got:,} vs {want:,} (band +/-{band:,.0f})")

    bad = []
    arr = [k for lv, k in by_unit if lv == "arrondissement"]
    band = ROUNDING * (sum(meta[k]["total"] for k in arr) + meta[COTONOU]["total"])
    for cat in CATEGORIES:
        got = sum(by_unit[("arrondissement", k)][cat] for k in arr)
        want = by_unit[("commune", COTONOU)][cat]
        if abs(got - want) > band + 1:
            bad.append((cat, got, want))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} Cotonou's {COTONOU_ARRONDISSEMENTS} "
          f"arrondissements sum to Cotonou on all {len(CATEGORIES)} categories\n"
          f"      (band +/-{band:,.0f}; they are parsed and kept but NOT drawn — "
          f"sources/bj.md §5)")
    for cat, got, want in bad[:6]:
        print(f"        {cat}: {got:,} vs {want:,}")

    # ---- the `(*)` sentinel: how many, and the most they could hide ----
    starred = [(m["name"], c) for gid, m in meta.items() if m["level"] == "commune"
               for c in CATEGORIES if m["shares"][c] is None]
    bound = sum(0.001 * meta[gid]["total"] for gid, m in meta.items()
                if m["level"] == "commune" for c in CATEGORIES if m["shares"][c] is None)
    print(f"\n  {len(starred)} commune cells print `{STAR}` (under 0.1%), which can hide at "
          f"most {bound:,.0f} people\n      ({bound / NATIONAL:.4%}). ZERO is the expected "
          f"figure and the guard is kept anyway: the\n      sentinel is real — Tableau 2's "
          f"age rows use it and Tableau 8's footnote defines it —\n      and dropping the "
          f"token would shift a whole row one column left in silence.")
    for name, cat in starred[:8]:
        print(f"        {name} / {cat}")

    drawn = sum(r["count"] for r in rows
                if r["geo_level"] == "commune" and r["source_category"] != RESIDUAL)
    resid = sum(r["count"] for r in rows
                if r["geo_level"] == "commune" and r["source_category"] == RESIDUAL)
    print(f"\n  {len(rows):,} rows. Drawn at commune: {drawn:,} people, "
          f"{drawn / NATIONAL:.2%} of the census;\n      residual {resid:,} "
          f"({resid / NATIONAL:.2%}), not drawn.")
    print("\n  Categories, national (from the workbook's full precision):")
    for cat in CATEGORIES:
        print(f"    {nat_share[cat] * NATIONAL:>11,.0f}  {nat_share[cat]:7.2%}  {cat}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, meta = read()
    check(rows, meta, _read_workbook())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
