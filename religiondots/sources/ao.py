"""Angola — INE, RGPH 2024, religion by municipality, from the 21 provincial volumes.

Reads (or fetches) data/raw/ao/ and writes data/normalized/ao.csv.

**Twenty-one bodies on 326 municipalities for 34.5 million people**, about 106,000 per
unit, from a census taken 19 September 2024 and published between November 2025 and
February 2026. The category list is the deepest in Africa on this map and four of its
entries are drawn nowhere else: **Tocoísta** (the Igreja de Nosso Senhor Jesus Cristo no
Mundo, founded by Simão Toco in 1949 and 353,885 strong), **Kimbanguista**, **Bom Deus**
and **Josafat**, all of them African-initiated churches that INE names individually rather
than folding into a Protestant catch-all.

**THE COUNTRY WAS REDRAWN BETWEEN THE FIELDWORK AND THE PUBLICATION.** The census was
collected on the division of Lei 18/16 -- 18 provinces, 164 municipalities, 562 communes --
and Lei 14/24 of 5 September 2024 replaced it with **21 provinces, 326 municipalities and
378 communes** while the data was being processed. INE retabulated to the new division, so
the printed tables are the ONLY published figures on it and no earlier boundary set
matches them. `sources/ao_geo.py` has where the polygons come from.

**THE UNIVERSE IS THE POPULATION AGED 2 AND OVER**, not everybody: 34,492,888 of the
country's 36,175,745. The question was not asked about infants, so the missing 1.68 million
are not a refusal and not an undercount of any body. `ao.py` carries the 2+ figure as the
row's own total and `countries.py` states the universe; the shares are of those asked.

**THE PANEL SPLIT AND THE COLUMN ORDER BOTH CHANGE FROM VOLUME TO VOLUME.** Bengo prints
the 21 bodies as 12 + 9 over two quadros on one page, Benguela as 7 + 7 + 7 over three,
Moxico as 10 + 11, and Moxico puts `Bom Deus` in the second panel where every other
province puts it sixth in the first. A parser that trusted a position would transpose
provinces against each other and nothing would disagree. So the columns are read from the
printed HEADER and matched to the canonical 21 by a global assignment across all of a
province's panels, under the constraint that each body is used exactly once.

**A BLANK CELL IS NOT A ZERO AND MUST NOT SHIFT THE ROW.** Bié's Belo Horizonte, Luando
and Umpulo rows print 11, 10 and 11 figures where the panel has 12 columns; Lunda Sul does
the same. Read left to right, Belo Horizonte's 10,588 Protestants land in the Universal do
Reino de Deus column and every figure after them moves one place, with no total anywhere
disagreeing -- §12's shape 2. Figures are therefore assigned to columns by the **right edge
of the printed number**, which is where INE aligns them, and a column with nothing in it is
recorded as zero and counted in `blanks`.

**UÍGE'S TABLE IS TRUNCATED IN THE PUBLISHED PDF.** Page 111 prints the universe and 11
bodies for all 23 municipalities; page 112, where the remaining 10 belong, is a blank page
-- no text, no image, no drawing. The words `Metodista` and `Adventista` occur nowhere in
the volume outside its list of tables. So 1.90 million people, 5.5% of the country, have
Catholic, Protestant, Tocoísta, Kimbanguista, Josafat, Bom Deus, Islamic, Animist, Judaic,
Universal and Nova Apostólica counts and no Methodist, Baptist, Adventist, Evangelical,
Pentecostal, Jehovah's Witness, Mensagem, no-religion or other-religion counts. The parser
does not invent them: Uíge's municipalities are drawn on the 11 bodies that were printed,
which reaches 62.9% of its 2+ population, and `countries.py` says so in `gap`. Whether INE
issues an errata is worth a look; it has issued one for Luanda and Icolo e Bengo already.

Usage:
    python sources/ao.py --fetch     22 PDFs, about 250 MB
    python sources/ao.py             normalise from data/raw/ao/
    python sources/ao.py --report    the parse, province by province, with the residuals
    python sources/ao.py --no-fill   leave Uíge and Moxico Leste on the eleven bodies their
                                     volumes printed, instead of filling the other ten from
                                     the national volume's province row (see `fill()`).
                                     Diagnostic: it makes the file smaller and the country
                                     wronger, and `countries.py`'s `fill=` would then be a
                                     lie, so do not build from it.
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
sys.path.insert(0, HERE)

import ao_pdfs as P                                              # noqa: E402

RAW = os.path.join(ROOT, "data", "raw", "ao")
OUT = os.path.join(ROOT, "data", "normalized", "ao.csv")

SOURCE_ID = "ao_rgph_2024"
YEAR = 2024
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# INE's 21 bodies, in the order the national volume prints them. `Não sabe/Não respondeu`
# is a non-answer rather than a religion; it is read so that a row reconciles against its
# own universe, and EXCLUDED in taxonomy/ao2024.py.
CATEGORIES = [
    "Católica", "Bom Deus", "Islâmica/Muçulmana", "Animista", "Judaica", "Protestante",
    "Universal do reino de Deus", "Nova apostólica", "Tocoísta", "Kimbanguista",
    "Josafat", "Assembleia de Deus pentecostal", "Testemunha de Jeová", "Metodista",
    "Evangélica", "Adventista", "Baptista", "Mensagem dos ultimos Tempos", "Sem religião",
    "Outra religião", "Não sabe/Não respondeu",
]
UNIVERSE = "População com 2 ou mais anos"

# What a column's header must contain for it to be that body. Stems and not whole words,
# because the spelling moves between volumes: Malanje heads its column `Outro` where every
# other province writes `Outra religião`, Benguela writes `Testemunhas` for Bengo's
# `Testemunha`, and `Últimos` is capitalised in some and not in others. Matched against the
# header's TOKENS rather than its text, because `assembleia` contains `sem`.
STEMS = {
    "Católica": ("catolic",), "Bom Deus": ("bom",),
    "Islâmica/Muçulmana": ("islamic", "muculman"), "Animista": ("animist",),
    "Judaica": ("judaic",), "Protestante": ("protestant",),
    "Universal do reino de Deus": ("universal",), "Nova apostólica": ("apostolic",),
    "Tocoísta": ("tocoist",), "Kimbanguista": ("kimbanguist",),
    "Josafat": ("josafat",), "Assembleia de Deus pentecostal": ("assembleia",),
    "Testemunha de Jeová": ("jeova", "testemunha"), "Metodista": ("metodist",),
    "Evangélica": ("evangelic",), "Adventista": ("adventist",), "Baptista": ("baptist",),
    "Mensagem dos ultimos Tempos": ("mensagem",), "Sem religião": ("sem",),
    "Outra religião": ("outr",), "Não sabe/Não respondeu": ("respond",),
}
assert set(STEMS) == set(CATEGORIES)

TITLE_RE = re.compile(r"^Quadro\s*7\.\s*\d+\s*-?\s*"
                      r"(Popula|Total\s+da\s+popula)", re.I)
NUMTOK = re.compile(r"^\d{1,3}$|^\d{4,}$")

# Row labels that are furniture rather than a municipality. INE varies the stub wording by
# volume, so this is matched folded and by prefix.
FURNITURE = ("areaderesidencia", "urbana", "rural", "municipio", "municipios",
             "municipioscomunas", "comunas", "total", "provinciae", "continua")


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def tokens(text):
    return [fold(t) for t in re.split(r"[\s/,.()]+", text) if fold(t)]


def is_furniture(label):
    f = fold(label)
    return (not f) or f in FURNITURE or f.startswith("continua")


# ---------------------------------------------------------------- fetch


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    todo = [("national", P.NATIONAL)] + sorted(P.PROVINCES.items())
    for name, fn in todo:
        dest = os.path.join(RAW, fn)
        if os.path.exists(dest) and _looks_like_pdf(dest):
            print(f"have {name}")
            continue
        url = P.BASE + fn
        print("GET", url)
        r = requests.get(url, timeout=1800, stream=True,
                         headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        if not _looks_like_pdf(dest):
            raise SystemExit(f"{dest} is not a whole PDF -- "
                             f"{os.path.getsize(dest):,} bytes, no %%EOF trailer. "
                             "INE truncates on a slow connection; delete it and retry.")
        print(f"  {name}: {os.path.getsize(dest):,} bytes")


def _looks_like_pdf(path):
    """§5a, and [[reference_pdf_truncated_at_source]]: 200 and a size are not a download."""
    if os.path.getsize(path) < 1_000_000:
        return False
    with open(path, "rb") as fh:
        if fh.read(5) != b"%PDF-":
            return False
        fh.seek(-2048, os.SEEK_END)
        return b"%%EOF" in fh.read()


# ---------------------------------------------------------------- the page


def page_rows(page):
    """Words on the page, grouped into printed rows and ordered left to right."""
    rows = {}
    for x0, y0, x1, y1, w, *_ in page.get_text("words"):
        key = next((k for k in rows if abs(k - y0) <= 3.0), None)
        if key is None:
            key = y0
            rows[key] = []
        rows[key].append((x0, x1, w))
    return [(y, sorted(v)) for y, v in sorted(rows.items())]


def split_row(items):
    """(label, [(value, x_left, x_right)]) for a data row, else (label, None).

    INE prints a thousands separator as a space, so `679 348` arrives as two words that
    touch; they are merged on the x-gap. The RIGHT edge is what a figure is placed by,
    because that is the edge INE aligns a column on and it is what lets a row with a blank
    cell be read; the left edge is kept only to cut the header into columns.
    """
    first = next((i for i, (x0, x1, w) in enumerate(items) if NUMTOK.match(w)), None)
    if first is None:
        return " ".join(w for _, _, w in items), None
    label = " ".join(w for _, _, w in items[:first])
    out, cur, lo, hi = [], "", None, None
    for x0, x1, w in items[first:]:
        if not NUMTOK.match(w):
            return label, None
        if cur and x0 - hi < 3.0:
            cur += w
        else:
            if cur:
                out.append((int(cur), lo, hi))
            cur, lo = w, x0
        hi = x1
    if cur:
        out.append((int(cur), lo, hi))
    return label, out


def _block(rows, pno, where):
    """One printed table -> (centres, header words, placed rows), or None.

    Columns come from the rows that print a figure in every one of them -- the province
    row and the urban/rural pair always do -- and every other row is dropped into those
    columns by the right edge of each figure, which is the edge INE aligns on.
    """
    data, header = [], []
    for y, items in rows:
        lbl, nums = split_row(items)
        if nums and len(nums) > 3:
            data.append((y, lbl, nums))
        else:
            header.append((y, items))
    if len(data) < 4:
        return None
    ncol = max(len(n) for _, _, n in data)
    full = [n for _, _, n in data if len(n) == ncol]
    if len(full) < 3:
        raise SystemExit(f"{where} p{pno}: only {len(full)} rows of the block's {ncol} "
                         "columns are complete; the column edges cannot be trusted")
    edges = [sum(r[c][2] for r in full) / len(full) for c in range(ncol)]
    lefts = [min(r[c][1] for r in full) for c in range(ncol)]
    rights = [max(r[c][2] for r in full) for c in range(ncol)]
    placed = []
    for y, lbl, nums in data:
        slot = [None] * ncol
        for v, lo, hi in nums:
            c = min(range(ncol), key=lambda k: abs(edges[k] - hi))
            if slot[c] is not None:
                raise SystemExit(f"{where} p{pno} row {lbl!r}: two figures land in "
                                 f"column {c}; the block is not right-aligned")
            slot[c] = v
        placed.append((y, lbl, slot))
    placed = _attach_wrapped_labels(placed, header)
    centres = [(lefts[c] + rights[c]) / 2 for c in range(ncol)]
    hwords = [((x0 + x1) / 2, w) for y, items in header for x0, x1, w in items]
    return centres, hwords, placed


def _stems_present(hwords):
    """Which bodies this block's header names. The fingerprint that spots a continuation."""
    out = set()
    for _, w in hwords:
        f = fold(w)
        for cat, stems in STEMS.items():
            if any(f.startswith(s) for s in stems):
                out.add(cat)
    return frozenset(out)


def panels(doc, where, anchor=None):
    """Every religion panel in the volume, as (page, column centres, header words, rows).

    **A PANEL DOES NOT ALWAYS BEGIN WITH A TITLE.** The national volume runs its first
    nine bodies over pages 117 and 118 and starts the next twelve at the top of page 119
    with no `Quadro` line at all; Huambo, Huíla and Moxico Leste do the same. Anchoring on
    the title alone reads a fifth of the country's bodies as absent and says nothing.

    So the walk starts at a titled page and continues onto each following page for as long
    as that page still prints a table whose first row is the province. A page that repeats
    the SAME column headings is the same panel continued -- pages 117 and 118 are one
    table broken over two -- and its rows are appended rather than treated as a new panel
    needing nine more bodies. Sameness is decided on which bodies the heading names, which
    is stable, and not on the header text, which is re-wrapped on every page.
    """
    anchor = anchor or where
    titled = set()
    for pno in range(doc.page_count):
        rows = page_rows(doc[pno])
        if any(TITLE_RE.match(" ".join(w for _, _, w in it)) for _, it in rows):
            if any(fold(split_row(it)[0]) == fold(anchor) for _, it in rows):
                titled.add(pno)
    if not titled:
        return []

    pages, pno = [], min(titled)
    while pno < doc.page_count:
        rows = page_rows(doc[pno])
        if pno not in titled:
            has = any(split_row(it)[1] and fold(split_row(it)[0]) == fold(anchor)
                      for _, it in rows)
            if not has:
                break
        pages.append((pno, rows))
        pno += 1

    blocks = []
    for pno, rows in pages:
        marks = [i for i, (y, it) in enumerate(rows)
                 if TITLE_RE.match(" ".join(w for _, _, w in it))]
        segs = []
        if marks:
            for j, i in enumerate(marks):
                end = marks[j + 1] if j + 1 < len(marks) else len(rows)
                segs.append(rows[i + 1:end])
            if marks[0] > 0:
                segs.insert(0, rows[:marks[0]])
        else:
            segs = [rows]
        for seg in segs:
            b = _block(seg, pno + 1, where)
            if b:
                blocks.append((pno + 1, b))

    out = []
    for pno, (centres, hwords, placed) in blocks:
        fp = _stems_present(hwords)
        prev = out[-1] if out else None
        same = (prev is not None and len(prev[1]) == len(centres)
                and (fp == prev[4] or not fp))
        if same:
            prev[3].extend(placed)
            continue
        out.append([pno, centres, hwords, list(placed), fp])

    # **HUAMBO PRINTS THE SAME PANEL TWICE**, on pages 110 and 111, figure for figure,
    # under a `Continua na página seguinte` that continues nothing. Appending both gives
    # the province 34 municipalities. An exact repeat of a row already in the panel is
    # dropped; a row that repeats a LABEL with different figures is not, because that is
    # how the national volume prints its urban and rural blocks.
    for panel in out:
        seen, keep = set(), []
        for y, lbl, slot in panel[3]:
            key = (fold(lbl), tuple(slot))
            if key in seen:
                continue
            seen.add(key)
            keep.append((y, lbl, slot))
        panel[3] = keep

    # **HUÍLA FOLLOWS ITS TABLE WITH A SECOND, COARSER ONE.** Page 117 prints the same 23
    # municipalities against a nine-column summary that groups the bodies differently and
    # heads two adjacent columns `Protestante`. It is not part of the 21 and reading it
    # would ask a province for 30 bodies. A panel that names a body an earlier panel has
    # already named is not a continuation of this table and is dropped.
    kept, claimed = [], set()
    for panel in out:
        if panel[4] & claimed:
            continue
        claimed |= panel[4]
        kept.append(panel)
    return [(p, c, h, r) for p, c, h, r, _ in kept]


def _attach_wrapped_labels(placed, header):
    """`Maquela do` / figures / `Zombo` -> one row labelled `Maquela do Zombo`."""
    stubs = [(y, " ".join(w for _, _, w in items)) for y, items in header]
    fixed = []
    for y, lbl, slot in placed:
        if lbl.strip():
            fixed.append((y, lbl, slot))
            continue
        above = [(abs(sy - y), sy, s) for sy, s in stubs if sy < y]
        below = [(abs(sy - y), sy, s) for sy, s in stubs if sy > y]
        parts = []
        if above:
            parts.append(min(above)[2])
        if below:
            parts.append(min(below)[2])
        fixed.append((y, " ".join(p for p in parts if p).strip(), slot))
    return fixed


# ---------------------------------------------------------------- the columns


BIG = 1e6


def universe_columns(found, where):
    """Which column of each panel is the universe, and what the province's universe is.

    **NOT ALWAYS COLUMN ZERO, AND NOT ALWAYS THERE.** Every volume repeats
    `População com 2 ou mais anos` beside the stub of each continuation panel except
    Namibe's second, which goes straight into Kimbanguista. Assuming column zero reads
    Namibe's 660 Kimbanguistas as its universe and turns the province into a rounding
    error. The universe is instead the LARGEST figure in the province row anywhere in the
    volume, which it must be: the bodies partition it, so no single one can reach it.
    """
    biggest = max(row[0][2][c] or 0
                  for _, centres, _, row in found for c in range(len(centres)))
    cols = []
    for pno, centres, hwords, rows in found:
        hit = [c for c in range(len(centres)) if rows[0][2][c] == biggest]
        if len(hit) > 1:
            raise SystemExit(f"{where} p{pno}: {len(hit)} columns hold the universe "
                             f"{biggest:,}")
        cols.append(hit[0] if hit else None)
    return cols, biggest


def assign(found, unicols, where):
    """Every category column of a volume -> one canonical body, decided together.

    A column is NOT cut out of the header band first. INE wraps a long heading over three
    lines and lets it overhang its neighbours, so `Universal do reino de Deus` reaches
    Cabinda's page as `Universal` sitting over the Protestante column and `do reino de
    Deus` over its own; any rule that slices the band at a boundary gets one of the two
    wrong. Instead each header WORD that carries a body's stem votes for the column whose
    figures are printed nearest it, and the whole volume is assigned at once under the
    constraint that a body is used ONCE. That constraint is what settles Cuanza Norte,
    where `Mensagem` sits closer to the Baptista column than to its own: Baptista's own
    word is closer still, so the two land where they belong.
    """
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    slots = []                       # (panel index, column index, x centre)
    for p, (pno, centres, hwords, placed) in enumerate(found):
        for c in range(len(centres)):
            if c != unicols[p]:
                slots.append((p, c, centres[c]))
    cost = np.full((len(slots), len(CATEGORIES)), BIG)
    for i, (p, c, x) in enumerate(slots):
        hwords = found[p][2]
        span = max(1.0, found[p][1][-1] - found[p][1][0])
        for j, cat in enumerate(CATEGORIES):
            near = [abs(wx - x) for wx, w in hwords
                    if any(fold(w).startswith(s) for s in STEMS[cat])]
            if near:
                cost[i, j] = min(near) / span
    rows, cols = linear_sum_assignment(cost)
    out = {}
    for i, j in zip(rows, cols):
        if cost[i, j] >= BIG:
            p, c, x = slots[i]
            raise SystemExit(f"{where} p{found[p][0]} column {c}: no header word carries "
                             f"a body's stem near x={x:.0f}")
        out[(slots[i][0], slots[i][1])] = CATEGORIES[j]
    if len(set(out.values())) != len(out):
        raise SystemExit(f"{where}: a body was assigned twice")
    return out


# ---------------------------------------------------------------- one province


def read_province(path, prov, expect):
    import fitz

    doc = fitz.open(path)
    found = panels(doc, prov)
    if not found:
        raise SystemExit(f"{prov}: no religion panel found in {os.path.basename(path)}")

    unicols, prov_universe = universe_columns(found, prov)
    mapping = assign(found, unicols, prov)

    universe, values, order, blanks = {}, {}, [], []

    def rows_of(p):
        """A panel's rows, with the furniture and the repeated province row dropped.

        **SEVEN PROVINCES CONTAIN A MUNICIPALITY OF THEIR OWN NAME** -- Benguela, Huambo,
        Malanje, Uíge, Cabinda, Bengo's Dande is the exception rather than the rule -- and
        a panel that runs over two pages prints the province row again at the top of the
        second. Dropping every row labelled with the province loses a municipality of
        788,380 people; keeping them all makes Huambo's universe disagree with itself. The
        province row is the one whose universe IS the province's.
        """
        pno, centres, hwords, rows = found[p]
        uc = unicols[p]
        out, seen_province = [], False
        for _, label, slot in rows:
            name = re.sub(r"\s+", " ", label).strip()
            if is_furniture(name):
                continue
            if fold(name) == fold(prov):
                if uc is not None and slot[uc] == prov_universe:
                    continue
                if uc is None and not seen_province:
                    seen_province = True
                    continue
            out.append((name, slot))
        return out

    # pass one: the municipality list and its universe, from the panels that carry one
    for p in range(len(found)):
        uc = unicols[p]
        if uc is None:
            continue
        for name, slot in rows_of(p):
            if slot[uc] is None:
                raise SystemExit(f"{prov} p{found[p][0]}: {name!r} has no universe")
            if name in universe and universe[name] != slot[uc]:
                raise SystemExit(f"{prov}: {name!r} universe {universe[name]:,} on one "
                                 f"panel and {slot[uc]:,} on another")
            universe[name] = slot[uc]
            if name not in order:
                order.append(name)
    if len(order) != expect:
        raise SystemExit(f"{prov}: {len(order)} municipality rows, expected {expect}\n"
                         f"  {order}")

    # pass two: the figures. A panel with no universe column is matched to the list by
    # name, and asserted to carry the same municipalities in the same order.
    for p, (pno, centres, hwords, rows) in enumerate(found):
        uc = unicols[p]
        got = rows_of(p)
        names = [n for n, _ in got]
        if names != order:
            raise SystemExit(f"{prov} p{pno}: municipalities are {names}, "
                             f"not the province's {order}")
        for name, slot in got:
            for c in range(len(centres)):
                if c == uc:
                    continue
                cat = mapping[(p, c)]
                if slot[c] is None:
                    blanks.append((name, cat))
                values[(name, cat)] = 0 if slot[c] is None else slot[c]
        prow = found[p][3][0][2]
        for c in range(len(centres)):
            if c != uc:
                values[("__province__", mapping[(p, c)])] = prow[c] or 0

    if len(order) != expect:
        raise SystemExit(f"{prov}: {len(order)} municipality rows, expected {expect}\n"
                         f"  {order}")
    drawn = sorted({c for c in CATEGORIES if (order[0], c) in values},
                   key=CATEGORIES.index)
    return {
        "order": order, "universe": universe, "values": values,
        "province_universe": prov_universe, "categories": drawn, "blanks": blanks,
    }


def read():
    out = {}
    for prov in sorted(P.PROVINCES):
        path = os.path.join(RAW, P.PROVINCES[prov])
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} -- run with --fetch first")
        out[prov] = read_province(path, prov, P.MUNICIPALITIES[prov])
    return out


# ---------------------------------------------------------------- the truncated volumes


def fill(data, national):
    """Uíge and Moxico Leste's ten unprinted bodies, from margins that ARE published.

    **Not filling is not the neutral choice, which is why this exists.** Uíge's volume
    prints eleven bodies covering 61.3% of its 2+ population, so a map that draws only
    what was printed shows the province 39% short on people AND compositionally wrong:
    Catholicism comes out at 55% of Uíge's drawn dots against a true 33.8%, and the
    province appears to contain no Evangelicals (347,084 of them, its second largest body)
    and nobody with no religion. Omission makes a stronger false claim here than the fill
    does.

    **BOTH MARGINS ARE PUBLISHED NUMBERS, which is the whole case for doing it.**

      * the ROW margin is each municipality's own unexplained remainder, `universe` minus
        the eleven bodies its volume printed, straight off the provincial table;
      * the COLUMN margin is the province's total for each unprinted body, straight off
        Quadro 7 of the NATIONAL volume, which does print all twenty-one by province.

    They very nearly agree without being made to. Moxico Leste's municipal remainders sum
    to 202,180 and the national volume's ten missing bodies sum to **202,180 exactly**;
    Uíge's are 733,662 against 734,842, a ratio of 0.9984, the residue of the same
    Cabinda/Uíge revision `check()` reports. The column margins are scaled to the row
    total, so each municipality's remainder is consumed exactly.

    **What is assumed, stated plainly: the MIX is uniform inside a province.** With two
    margins and nothing in the interior, the maximum-entropy fill is the outer product,
    so every municipality of Uíge receives the province's own proportions among the ten.
    The AMOUNT each gets is measured and varies (Lucunga's remainder is 29.3% of its
    people, the city of Uíge's 44.2%); the SPLIT of that amount between Methodist and
    Adventist is not, and is identical everywhere. That is why these rows are `derived`,
    never ring, and vanish under `inferred dots: not shown`.

    This is [[feedback_proxy_residual_nameable]]'s test passed rather than dodged: the
    non-matching part is a published number to weight by, not a correlation.
    """
    out = {}
    for prov, d in data.items():
        missing = [c for c in CATEGORIES if c not in d["categories"]]
        if not missing:
            continue
        col = {}
        for c in missing:
            v = national.get((prov, c))
            if v is None:
                v = next((national[k] for k in national
                          if k[1] == c and fold(k[0]) == fold(prov)), None)
            if v is None:
                raise SystemExit(f"{prov}: the national volume has no {c!r} either, so "
                                 "there is no column margin to fill from")
            col[c] = v
        colsum = sum(col.values())
        rows = {n: d["universe"][n] - sum(d["values"].get((n, c), 0)
                                          for c in d["categories"])
                for n in d["order"]}
        bad = {n: v for n, v in rows.items() if v < 0}
        if bad:
            raise SystemExit(f"{prov}: negative remainder in {bad}, so the printed bodies "
                             "already exceed the universe and the fill is meaningless")
        rowsum = sum(rows.values())
        ratio = rowsum / colsum if colsum else 0
        if not 0.95 < ratio < 1.05:
            raise SystemExit(f"{prov}: municipal remainders sum to {rowsum:,} but the "
                             f"national volume's missing bodies to {colsum:,} "
                             f"(ratio {ratio:.4f}); too far apart to fill")
        filled = {}
        for n in d["order"]:
            share = rows[n]
            # largest remainder, so the municipality's own measured remainder is consumed
            # to the person and the province's proportions are held as closely as integers
            # allow.
            exact = {c: share * col[c] / colsum for c in missing}
            base = {c: int(v) for c, v in exact.items()}
            short = share - sum(base.values())
            for c in sorted(missing, key=lambda k: -(exact[k] - base[k]))[:short]:
                base[c] += 1
            if sum(base.values()) != share:
                raise SystemExit(f"{prov}/{n}: fill sums to {sum(base.values()):,}, "
                                 f"not the remainder {share:,}")
            filled[n] = base
        out[prov] = {"missing": missing, "cells": filled, "rows": rows,
                     "col": col, "ratio": ratio}
    return out


# ---------------------------------------------------------------- the national check


def read_national():
    """Quadro 7.1/7.2 of the national volume: the same 21 bodies by PROVINCE.

    Printed from the same tabulation, so it cannot catch a mistake INE made; it catches
    a mistake THIS parser makes, which is the point. A province whose 326-municipality
    sum disagrees with the national volume's row for it has been read wrong.
    """
    import fitz

    path = os.path.join(RAW, P.NATIONAL)
    doc = fitz.open(path)
    found = panels(doc, "national", anchor="Angola")
    unicols, _ = universe_columns(found, "national")
    mapping = assign(found, unicols, "national")
    out = {}
    for p, (pno, centres, hwords, rows) in enumerate(found):
        uc = unicols[p]
        seen = set()
        for _, label, slot in rows:
            name = re.sub(r"\s+", " ", label).strip()
            if is_furniture(name) or fold(name) in seen:
                continue          # the urban and rural blocks repeat every province name
            seen.add(fold(name))
            for c in range(len(centres)):
                if c != uc:
                    out[(name, mapping[(p, c)])] = slot[c] or 0
            if uc is not None and slot[uc] is not None:
                out[(name, UNIVERSE)] = slot[uc]
    return out


# ---------------------------------------------------------------- output


def geo_id(prov, i):
    """`AO` + the province's index in INE's printed order + the municipality's.

    Positional, like `sources/mw.py`'s, because the printed tables carry no code at all.
    `ao_geo.py` joins by NAME and asserts that the two sides agree on every one of the
    326, which is the check this id cannot make for itself.
    """
    return f"AO{PROV_INDEX[prov]:02d}{i + 1:02d}"


# INE's own order, from Quadro 9 of the national volume; it is the order the provinces are
# printed in throughout, and it starts in the north and works south.
PROV_ORDER = ["Cabinda", "Zaire", "Uíge", "Bengo", "Luanda", "Cuanza Norte", "Cuanza Sul",
              "Malanje", "Lunda Norte", "Lunda Sul", "Moxico", "Bié", "Huambo",
              "Benguela", "Namibe", "Huíla", "Cunene", "Cubango", "Icolo e Bengo",
              "Moxico Leste", "Cuando"]
assert sorted(PROV_ORDER) == sorted(P.PROVINCES)
PROV_INDEX = {p: i + 1 for i, p in enumerate(PROV_ORDER)}


def write(data, filled=None):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    filled = filled or {}
    rows = []
    for prov in PROV_ORDER:
        d = data[prov]
        f = filled.get(prov)
        for i, name in enumerate(d["order"]):
            gid = geo_id(prov, i)
            rows.append([gid, "municipality", name, UNIVERSE, d["universe"][name],
                         BASIS, YEAR, SOURCE_ID,
                         f"province={prov}; universe total, not a religion category"])
            for cat in d["categories"]:
                rows.append([gid, "municipality", name, cat, d["values"][(name, cat)],
                             BASIS, YEAR, SOURCE_ID, f"province={prov}"])
            if not f:
                continue
            for cat in f["missing"]:
                rows.append([gid, "municipality", name, cat, f["cells"][name][cat],
                             BASIS, YEAR, SOURCE_ID,
                             f"province={prov}; tier=derived; this volume did not print "
                             f"{cat}, so it is the national volume's {prov} total split "
                             "by each municipality's unexplained remainder"])
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        w.writerows(rows)
    n_derived = sum(len(f["missing"]) * len(data[p]["order"])
                    for p, f in filled.items())
    print(f"wrote {OUT}  {len(rows):,} rows ({n_derived:,} derived)")
    return rows


# INE's own arithmetic is a person or two out in most volumes: a province row and the sum
# of its municipalities differ by 1 or 2 in eleven of the twenty-one, and `Outra religião`
# by as much as 95 in Bié. Nothing about that is a parse error, and a parser that demanded
# an exact match would be asserting something INE never claimed.
SLACK = 120


def check(data):
    """Two checks, and only the first of them is about this parser.

    **THE MUNICIPALITIES AGAINST THEIR OWN VOLUME'S PROVINCE ROW, BODY BY BODY.** This is
    the one that can fail. A figure read into the wrong column, a row transposed, a body
    assigned to the wrong header: all of them break this and nothing else would notice.

    **THE PROVINCES AGAINST THE NATIONAL VOLUME.** This one is reported and never raises,
    because the two disagree for a reason that is INE's and not ours. The national volume
    was published 20 November 2025 and the provincial volumes in January and February
    2026, and the later ones carry revised figures: Luanda hands 144,745 people to Icolo e
    Bengo, which is the published `Errata das Províncias de Luanda e Icolo e Bengo`;
    Cabinda gains 7,050 from Uíge; Benguela and Namibe swap 29. Inside those provinces the
    revision moved people INTO `Sem religião` and out of almost everything else, so
    Cabinda's no-religion count rises 19,316 while each of its bodies falls a little. The
    provincial volumes are the later word and are what is drawn.
    """
    national = read_national()
    natkey = {(fold(n), c): v for (n, c), v in national.items()}
    print(f"{'province':16s} {'munis':>5s} {'cats':>5s} {'universe':>12s} "
          f"{'sum of rows':>12s} {'vs national':>12s} {'blanks':>6s}")
    total, bad = 0, []
    for prov in PROV_ORDER:
        d = data[prov]
        s = sum(d["universe"].values())
        total += s
        for cat in d["categories"]:
            mine = sum(d["values"].get((n, cat), 0) for n in d["order"])
            prow = d["values"].get(("__province__", cat))
            if prow is not None and abs(mine - prow) > SLACK:
                bad.append(f"{prov} {cat}: municipalities {mine:,} vs the volume's own "
                           f"province row {prow:,}")
        nat = natkey.get((fold(prov), UNIVERSE))
        delta = "" if nat is None else f"{s - nat:+,}"
        print(f"{prov:16s} {len(d['order']):5d} {len(d['categories']):5d} "
              f"{d['province_universe']:12,} {s:12,} {delta:>12s} "
              f"{len(d['blanks']):6d}")
    print(f"{'ANGOLA':16s} {sum(len(d['order']) for d in data.values()):5d} "
          f"{'':5s} {'':12s} {total:12,}")
    print()
    for cat in CATEGORIES:
        mine = sum(d["values"].get((n, cat), 0)
                   for d in data.values() for n in d["order"])
        nat = natkey.get((fold("Angola"), cat))
        flag = "" if nat is None else f"   national volume {nat:>12,}  {mine - nat:+,}"
        print(f"  {cat:32s} {mine:12,}{flag}")
    if bad:
        raise SystemExit("\n".join(["the parse does not reconcile:"] + bad))
    print("\nevery province's municipalities reconcile with its own volume, body by body")


def report(data):
    for prov in PROV_ORDER:
        d = data[prov]
        print(f"=== {prov}  ({len(d['order'])} municipalities, "
              f"{len(d['categories'])}/21 bodies)")
        for name in d["order"]:
            u = d["universe"][name]
            s = sum(d["values"].get((name, c), 0) for c in d["categories"])
            print(f"   {name:26s} {u:10,}  bodies {s:10,}  residual {u - s:+7,}")
        if d["blanks"]:
            print(f"   blank cells: {d['blanks']}")


def report_fill(filled):
    for prov, f in filled.items():
        print(f"=== {prov}: filling {len(f['missing'])} bodies the volume did not print")
        print(f"    municipal remainders {sum(f['rows'].values()):,}  vs the national "
              f"volume's {sum(f['col'].values()):,}  (ratio {f['ratio']:.4f})")
        tot = sum(f["col"].values())
        for c in f["missing"]:
            got = sum(f["cells"][n][c] for n in f["cells"])
            print(f"    {c:32s} {f['col'][c]:>10,}  ->  {got:>10,}  "
                  f"({100.0 * f['col'][c] / tot:5.1f}% of every municipality's remainder)")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
        sys.exit()
    data = read()
    if "--report" in sys.argv:
        report(data)
    check(data)
    filled = {} if "--no-fill" in sys.argv else fill(data, read_national())
    if filled:
        report_fill(filled)
    write(data, filled)
