"""Angola, RGPH 2024, mother tongue by municipality -> data/normalized/ao.csv.

    python sources/ao_rgph.py [--fetch] [--report]

SOURCE. INE Angola, Recenseamento Geral da População e Habitação 2024, Resultados
Definitivos: Quadro 6.2 `População por município e área de residência, segundo a língua
materna, 2024` in each of the 21 provincial volumes (January and February 2026), and Quadro
6.2 of the national volume (20 November 2025), which prints the same table by PROVINCE.
One answer per person, the population aged 2 and over (34,492,888 of 36,175,745).

THE SAME PDFs RELIGIONDOTS ALREADY HOLDS. religiondots/sources/ao.py parsed Quadro 7
(religion) out of these 22 volumes, so they are read from religiondots/data/raw/ao/ when
they are there (read-only) and fetched into languagedots/data/raw/ao/ only when they are
not. The table-reading machinery below is that parser's, adapted: the column layout moves
between volumes, so columns are matched to the printed header by a global assignment and
figures are placed by their right edge (religiondots/sources/ao.py's docstring has the
traps that taught it).

WHAT THE TWO LEVELS PRINT.
  * provincial volumes, by municipality: Português, Kimbundu, Umbundu, Cokue, Kikongo,
    Olunyaneka, Nganguela, Oxikwanhama, Ifyoti, Muhumbi, Luvale, Khoisan, `Línguas
    estrangeiras`, `Outras línguas`, `Não sabe`.
  * national volume, by province: the same, except that `Línguas estrangeiras` is printed as
    nine columns (Mandarim, Inglês, Francês, Espanhol, Alemão, Russo, Árabe, Lingala,
    Criolo).

So a municipality's `Línguas estrangeiras` is split into the nine by its province's own mix
from the national volume (largest remainder, so the municipality's measured total is kept
to the person), and those rows are `tier=derived`. Both margins are INE's; only the split
inside a province is assumed. 84,914 of the 122,390 foreign-language speakers are Lingala.

INE GROUPED SOME LANGUAGES BEFORE PRINTING (national volume p.54, footnote 5): Kimbundu
includes Mbongala, Songo and Ngoya; Umbundu includes Mukubale; Cokue includes Lunda;
Olunyaneka includes Humbi, Handa, Mucilengue, Sela and Kimbali; Nganguela includes Luchazi,
Mbunda and Ukumbi; Kwanhama includes Herero and Mudimba; the rest went to `Outras` (Gestual,
Kisumbe, Kiswahili, Mwalabi, Sikabunda, Muko, Lucumai...). Muhumbi has a column of its own
all the same. taxonomy/ao2024.py maps each printed label; sources/ao.md says what it means.

GEOGRAPHY. Each municipality is joined by (province, name) to religiondots'
data/normalized/ao.csv, from the same 21 volumes, to take its geo_id (and through
religiondots' ao_lookup.csv its polygon). The join is asserted both ways, and each
municipality's language universe must EQUAL its religion universe: the same population aged
2+, printed twice. That is what catches a wrong twin.

CHECKS (all must pass):
  1. every municipality's categories sum to its universe (INE's arithmetic, within SLACK;
     Bengo's doubled column and Uíge's swapped ones repaired first, see DOUBLED, TAIL_SWAP)
  2. every province's municipalities sum to its own volume's province row, per category
  3. 326 municipalities, joined one-to-one to religiondots', universe equal for each but the
     four in UNIVERSE_DIFFERS, where the ethnic table sides with the language table
  4. the foreign split re-aggregates to each municipality's measured total exactly
Reported, not asserted: each province against the national volume (the provincial volumes
are the later, revised word; religiondots/sources/ao.py describes the revision).
"""
import argparse
import csv
import os
import re
import sys
import unicodedata
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RD = HERE.parent / "religiondots"
RAW = HERE / "data" / "raw" / "ao"
RD_RAW = RD / "data" / "raw" / "ao"
RD_NORM = RD / "data" / "normalized" / "ao.csv"
OUT = HERE / "data" / "normalized" / "ao.csv"

BASE = "https://www.ine.gov.ao/Arquivos/arquivosCarregados/Carregados/"
# The file index, from religiondots/sources/ao_pdfs.py (INE's names carry no meaning).
NATIONAL = "Publicacao_638996687409619846.pdf"
PROVINCES = {
    "Bengo": "Publicacao_639070145024903075.pdf",
    "Benguela": "Publicacao_639064071318925115.pdf",
    "Bié": "Publicacao_639071861895047488.pdf",
    "Cabinda": "Publicacao_639086316474352489.pdf",
    "Cuando": "Publicacao_639175197254207294.pdf",
    "Cuanza Norte": "Publicacao_639074613297362596.pdf",
    "Cuanza Sul": "Publicacao_639065784580611643.pdf",
    "Cubango": "Publicacao_639175237509463838.pdf",
    "Cunene": "Publicacao_639050104455150531.pdf",
    "Huambo": "Publicacao_639064975879258682.pdf",
    "Huíla": "Publicacao_639051372856668517.pdf",
    "Icolo e Bengo": "Publicacao_639081910852232827.pdf",
    "Luanda": "Publicacao_639175196705255655.pdf",
    "Lunda Norte": "Publicacao_639062413002295499.pdf",
    "Lunda Sul": "Publicacao_639064347606305071.pdf",
    "Malanje": "Publicacao_639071901376985768.pdf",
    "Moxico": "Publicacao_639064111089603097.pdf",
    "Moxico Leste": "Publicacao_639065854702898637.pdf",
    "Namibe": "Publicacao_639051805987160989.pdf",
    "Uíge": "Publicacao_639120325078684256.pdf",
    "Zaire": "Publicacao_639175220682196239.pdf",
}
MUNICIPALITIES = {
    "Cabinda": 10, "Zaire": 11, "Uíge": 23, "Bengo": 12, "Luanda": 16,
    "Cuanza Norte": 17, "Cuanza Sul": 24, "Malanje": 27, "Lunda Norte": 19,
    "Lunda Sul": 14, "Moxico": 12, "Bié": 19, "Huambo": 17, "Benguela": 23,
    "Namibe": 9, "Huíla": 23, "Cunene": 14, "Cubango": 11, "Icolo e Bengo": 7,
    "Moxico Leste": 9, "Cuando": 9,
}
# The label the volume's Quadro 6.2 prints on its PROVINCE row, where it is not the province.
# Uíge's says `Cunene` and Lunda Sul's `Benguela` (template slips: the universes, 1,895,001
# and 840,585, are their own).
ANCHOR = {"Uíge": "Cunene", "Lunda Sul": "Benguela"}
# Provinces whose volume has no language table (from_national() says why).
NO_TABLE = {"Moxico Leste"}
assert sorted(PROVINCES) == sorted(MUNICIPALITIES)
assert sum(MUNICIPALITIES.values()) == 326

UNIVERSE = "Número de pessoas com 2 ou mais anos"
FOREIGN_TOTAL = "Línguas estrangeiras"
FOREIGN = ["Mandarim", "Inglês", "Francês", "Espanhol", "Alemão", "Russo", "Árabe",
           "Lingala", "Criolo"]
# INE's labels, in the national volume's order. What a column header must contain to be that
# category: stems, matched against the header's folded tokens.
STEMS = {
    "Português": ("portugu",),
    "Kimbundu": ("kimbundu",),
    "Umbundu": ("umbundu",),
    "Cokue (Chokwe/Kioko)": ("cokue", "chokwe", "kioko"),
    "Kikongo": ("kikongo",),
    "Olunyaneka (Nhaneka)": ("olunyaneka", "nhaneka"),
    "Nganguela": ("nganguela", "ngangela"),
    "Oxikwanhama (Kwanhama)": ("oxikwanhama", "kwanhama"),
    "Ifyoti (Fiote)": ("ifyoti", "fiote"),
    "Muhumbi": ("muhumbi",),
    "Luvale": ("luvale",),
    "Khoisan": ("khoisan",),
    FOREIGN_TOTAL: ("estrangeir", "estangeir"),     # Zaire prints `Estangeiras`
    "Mandarim": ("mandarim",),
    "Inglês": ("ingles",),
    "Francês": ("frances",),
    "Espanhol": ("espanhol",),
    "Alemão": ("alemao",),
    "Russo": ("russo",),
    "Árabe": ("arabe",),
    "Lingala": ("lingala",),
    "Criolo": ("criolo", "crioulo"),
    "Outras línguas": ("outra",),
    "Não sabe": ("sabe",),
}
CATEGORIES = list(STEMS)

# Any Quadro 6.x whose title names the mother tongue: Namibe prints its second panel as
# `Quadro 6. 3`, Cubango's second page has an en dash, Lunda Norte's no dash at all. Moxico
# Leste's 6.2 is titled (and is) the ethnic table, so it does not match.
TITLE_RE = re.compile(r"^Quadro\s*6\.\s*\d\s*[-–—]?\s*Popula.*l[ií]nguas?\s+materna", re.I)
ANY_TITLE = re.compile(r"^Quadro\s*\d", re.I)
NUMTOK = re.compile(r"^\d{1,3}$|^\d{4,}$")
DASHES = ("-", "–", "—")
FURNITURE = ("areaderesidencia", "urbana", "rural", "municipio", "municipios",
             "municipioscomunas", "comunas", "total", "provinciae", "continua",
             "provincias", "areaderesidenciamunicipiosecomunas")
SLACK = 120
# BENGO PRINTS ITS `Línguas estrangeiras` COLUMN DOUBLED. Every row sums over its universe by
# half its foreign figure (Dande 405 foreign, 203 over; Panguila 666 and 332; Barra do Dande
# 236 and 118), and the column totals 1,658 against the national volume's 829 for Bengo,
# exactly twice. So the column is halved (rounded) before any check.
DOUBLED = {"Bengo": FOREIGN_TOTAL}

# UÍGE PRINTS THE LAST THREE COLUMNS OF THREE PAIRS OF ROWS SWAPPED. Its Quadro 6.2 rows sum
# to their universes within ~90 except six adjacent pairs, off by opposite amounts:
# Ambuíla -467 / Negage +324, Puri -5,594 / Maquela do Zombo +5,660, Cangola -2,295 /
# Uíge +2,327. Swapping `Línguas estrangeiras`, `Outras línguas` and `Não sabe` within each
# pair brings Negage to +1, Maquela to 0 and Uíge to 0 (and Puri +66, Cangola +32), and
# makes the figures plausible: the provincial capital gets 1,450 foreign speakers instead of
# 5, and Maquela do Zombo on the DRC border 5,566 instead of 131. Column sums are untouched.
TAIL = ("Línguas estrangeiras", "Outras línguas", "Não sabe")
TAIL_SWAP = {"Uíge": [("Ambuíla", "Negage"), ("Puri", "Maquela do Zombo"),
                      ("Cangola", "Uíge")]}
# Rows left off by more than the slack after every repair, accepted as INE's own: Ambuíla
# stays 144 over (0.7%) after its swap; nothing printed accounts for the rest.
ACCEPT_OFF = {("Uíge", "Ambuíla")}
# Municipalities whose language universe differs from religiondots' (the religion table's).
# The ethnic table of the same volume (Quadro 6.1) prints the language table's figure, so the
# religion table is the odd one out: Malanje p109 Cacuso 56,342 and Ngola Luiji 18,601 (religion
# p119: 51,218 and 23,725), Moxico p97 Luena 340,536 and Lucusse 16,469 (religion p101: 337,755
# and 19,250). Each pair's total agrees. The language table's own figures are kept.
UNIVERSE_DIFFERS = {("Malanje", "Cacuso"), ("Malanje", "Ngola Luiji"),
                    ("Moxico", "Luena"), ("Moxico", "Lucusse")}


# MALANJE PRINTS NO FOREIGN COLUMN. Its rows sum to their universes without one, and its
# `Outras línguas` equals the national volume's Malanje other + foreign (split_foreign()
# asserts it), so its few hundred foreign speakers stay inside `Outras línguas` there.
FOREIGN_IN_OTHER = {"Malanje"}
OTHER_WITH_FOREIGN = "Outras línguas (Malanje: com as línguas estrangeiras)"


def halve_doubled(data):
    for prov, cat in DOUBLED.items():
        d = data[prov]
        for n in d["order"] + ["__province__"]:
            d["values"][(n, cat)] = int(round(d["values"][(n, cat)] / 2))
        print(f"  halved {prov}'s {cat} column, printed doubled")


def swap_tail(data):
    for prov, pairs in TAIL_SWAP.items():
        d = data[prov]
        for a, b in pairs:
            def off(n):
                return d["universe"][n] - sum(d["values"][(n, c)] for c in d["categories"])
            before = (off(a), off(b))
            for c in TAIL:
                d["values"][(a, c)], d["values"][(b, c)] = d["values"][(b, c)], d["values"][(a, c)]
            after = (off(a), off(b))
            if not abs(after[0]) + abs(after[1]) < (abs(before[0]) + abs(before[1])) / 3:
                raise SystemExit(f"{prov}: swapping {a}/{b} does not help: {before} -> {after}")
            print(f"  swapped the last three columns, {prov}/{a} and {b}: off by {before} -> "
                  f"{after}")


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def is_furniture(label):
    f = fold(label)
    return (not f) or f in FURNITURE or f.startswith("continua")


def pdf_path(fn):
    for d in (RAW, RD_RAW):
        p = d / fn
        if p.exists():
            return p
    raise SystemExit(f"{fn} is in neither {RAW} nor {RD_RAW}: run with --fetch")


def _looks_like_pdf(path):
    if os.path.getsize(path) < 1_000_000:
        return False
    with open(path, "rb") as fh:
        if fh.read(5) != b"%PDF-":
            return False
        fh.seek(-2048, os.SEEK_END)
        return b"%%EOF" in fh.read()


def fetch():
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    for name, fn in [("national", NATIONAL)] + sorted(PROVINCES.items()):
        if (RD_RAW / fn).exists() or ((RAW / fn).exists() and _looks_like_pdf(RAW / fn)):
            print(f"have {name}")
            continue
        print("GET", BASE + fn)
        r = requests.get(BASE + fn, timeout=1800, stream=True,
                         headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        r.raise_for_status()
        with open(RAW / fn, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        if not _looks_like_pdf(RAW / fn):
            raise SystemExit(f"{fn}: not a whole PDF; delete it and retry")


# ---------------------------------------------------------------- the page (religiondots' ao.py)


def page_rows(page):
    rows = {}
    for x0, y0, x1, y1, w, *_ in page.get_text("words"):
        key = next((k for k in rows if abs(k - y0) <= 3.0), None)
        if key is None:
            key = y0
            rows[key] = []
        rows[key].append((x0, x1, w))
    return [(y, sorted(v)) for y, v in sorted(rows.items())]


def split_row(items):
    first = next((i for i, (x0, x1, w) in enumerate(items) if NUMTOK.match(w)), None)
    if first is None:
        return " ".join(w for _, _, w in items), None
    label = " ".join(w for _, _, w in items[:first])
    out, cur, lo, hi = [], "", None, None
    for x0, x1, w in items[first:]:
        if w in DASHES:
            # Cunene prints a zero as `-` (Otchinjau, Nehone, Cafima): a figure of its own
            if cur:
                out.append((int(cur), lo, hi))
            out.append((0, x0, x1))
            cur, lo, hi = "", None, x1
            continue
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


def _attach_wrapped_labels(placed, header):
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


def _block(rows, pno, where):
    data, header = [], []
    for y, items in rows:
        lbl, nums = split_row(items)
        # >= 2, not > 3: Lunda Sul prints Alto Chicapa in its second panel as two figures
        # and eight blanks
        if nums and len(nums) >= 2:
            data.append((y, lbl, nums))
        else:
            header.append((y, items))
    if len(data) < 4:
        return None
    ncol = max(len(n) for _, _, n in data)
    full = [n for _, _, n in data if len(n) == ncol]
    if len(full) < 3:
        raise SystemExit(f"{where} p{pno}: only {len(full)} complete rows of {ncol} columns")
    edges = [sum(r[c][2] for r in full) / len(full) for c in range(ncol)]
    lefts = [min(r[c][1] for r in full) for c in range(ncol)]
    rights = [max(r[c][2] for r in full) for c in range(ncol)]
    placed = []
    for y, lbl, nums in data:
        slot = [None] * ncol
        for v, lo, hi in nums:
            c = min(range(ncol), key=lambda k: abs(edges[k] - hi))
            if slot[c] is not None:
                raise SystemExit(f"{where} p{pno} row {lbl!r}: two figures in column {c}")
            slot[c] = v
        placed.append((y, lbl, slot))
    placed = _attach_wrapped_labels(placed, header)
    centres = [(lefts[c] + rights[c]) / 2 for c in range(ncol)]
    hwords = [((x0 + x1) / 2, w) for y, items in header for x0, x1, w in items]
    return centres, hwords, placed


def _stems_present(hwords):
    out = set()
    for _, w in hwords:
        f = fold(w)
        for cat, stems in STEMS.items():
            if any(f.startswith(s) for s in stems):
                out.add(cat)
    return frozenset(out)


def panels(doc, where, anchor=None):
    """Every Quadro 6.2 panel: from the titled page, onward while a page still prints a row
    for the anchor and no other Quadro's title."""
    anchor = anchor or where
    titled = []
    for pno in range(doc.page_count):
        rows = page_rows(doc[pno])
        if any(TITLE_RE.match(" ".join(w for _, _, w in it)) for _, it in rows):
            if any(split_row(it)[1] and fold(split_row(it)[0]) == fold(anchor)
                   for _, it in rows):
                titled.append(pno)
    if not titled:
        return []
    # from EVERY titled page, not just the first: Bengo prints its two panels on pages 101
    # and 104 with two other tables between them
    pages, done = [], set()
    for start in titled:
        pno = start
        while pno < doc.page_count and pno not in done:
            rows = page_rows(doc[pno])
            if pno != start:
                other = any(ANY_TITLE.match(" ".join(w for _, _, w in it))
                            and not TITLE_RE.match(" ".join(w for _, _, w in it))
                            for _, it in rows)
                has = any(split_row(it)[1] and fold(split_row(it)[0]) == fold(anchor)
                          for _, it in rows)
                # Luanda's middle page repeats the title but not the province row, and
                # Huambo's prints neither, only the column headings
                ours = any(TITLE_RE.match(" ".join(w for _, _, w in it))
                           or "maternaagrupada" in fold(" ".join(w for _, _, w in it))
                           for _, it in rows)
                if other or not (has or ours):
                    break
            pages.append((pno, rows))
            done.add(pno)
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
            # what sits above the title continues our table from the page before, unless it
            # is another table (Moxico Leste prints Quadro 6.1 above 6.2 on one page)
            pre = rows[:marks[0]]
            if marks[0] > 0 and not any(ANY_TITLE.match(" ".join(w for _, _, w in it))
                                        for _, it in pre):
                segs.insert(0, pre)
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
            # a continuation that only repeats rows the panel already has is the same page
            # printed twice (Huambo does this in its religion table); drop the BLOCK. Not
            # row by row: Cabinda prints a one-commune municipality and its commune as two
            # identical rows, and both are needed for the commune walk below.
            have = {(fold(l), tuple(s)) for _, l, s in prev[3]}
            if all((fold(l), tuple(s)) in have for _, l, s in placed):
                continue
            prev[3].extend(placed)
            continue
        if not fp:
            raise SystemExit(f"{where} p{pno}: a table with no language header")
        out.append([pno, centres, hwords, list(placed), fp])
    # A panel that names nothing new is a repeat or a summary and is dropped. One that names
    # SOME columns already read is kept: Cabinda repeats `Português` in its second panel.
    kept, claimed = [], set()
    for panel in out:
        if panel[4] <= claimed:
            continue
        claimed |= panel[4]
        kept.append(panel)
    return [(p, c, h, r) for p, c, h, r, _ in kept]


BIG = 1e6


def universe_columns(found, where):
    biggest = max(row[0][2][c] or 0
                  for _, centres, _, row in found for c in range(len(centres)))
    cols = []
    for pno, centres, hwords, rows in found:
        hit = [c for c in range(len(centres)) if rows[0][2][c] == biggest]
        if len(hit) > 1:
            raise SystemExit(f"{where} p{pno}: {len(hit)} columns hold the universe")
        cols.append(hit[0] if hit else None)
    return cols, biggest


def assign(found, unicols, where):
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    slots = []
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
    # each PANEL's columns to distinct categories; a category may recur across panels
    # (Cabinda's repeated `Português`), and read_province() asserts the figures agree
    out = {}
    for p in range(len(found)):
        idx = [i for i, sl in enumerate(slots) if sl[0] == p]
        rows, cols = linear_sum_assignment(cost[idx])
        for r, j in zip(rows, cols):
            i = idx[r]
            if cost[i, j] >= BIG:
                _, c, x = slots[i]
                raise SystemExit(f"{where} p{found[p][0]} column {c}: no header word near "
                                 f"x={x:.0f}")
            out[(slots[i][0], slots[i][1])] = CATEGORIES[j]
    if len(out) != len(slots):
        raise SystemExit(f"{where}: {len(slots)} columns, {len(out)} assigned")
    return out


# ---------------------------------------------------------------- one province


def read_province(prov, expect):
    import fitz
    doc = fitz.open(pdf_path(PROVINCES[prov]))
    anchor = ANCHOR.get(prov, prov)
    found = panels(doc, prov, anchor=anchor)
    if not found:
        raise SystemExit(f"{prov}: no Quadro 6.2 found")
    unicols, prov_universe = universe_columns(found, prov)
    mapping = assign(found, unicols, prov)
    values, blanks = {}, []

    def rows_of(p):
        pno, centres, hwords, rows = found[p]
        uc = unicols[p]
        out, seen_province = [], False
        for _, label, slot in rows:
            name = re.sub(r"\s+", " ", label).strip()
            if is_furniture(name):
                continue
            if fold(name) == fold(anchor):
                if uc is not None and slot[uc] == prov_universe:
                    continue
                if uc is None and not seen_province:
                    seen_province = True
                    continue
            out.append((name, slot))
        return out

    # WHICH ROWS ARE MUNICIPALITIES. Some volumes print the communes too, each under its
    # municipality (Cabinda: `Belize` 18,473, then its communes `Belize` 16,708 and `Luali`
    # 1,765). The municipality list is known: religiondots' ao.csv has the 326, by province,
    # read from the religion table of these same volumes. So a row whose name is a
    # municipality of this province not yet seen opens it, and the rows after it are its
    # communes. A commune that shares its municipality's name comes after it, when that name
    # is already used up.
    first = next(p for p in range(len(found)) if unicols[p] is not None)
    lead = rows_of(first)
    uc0 = unicols[first]
    labels = [n for n, _ in lead]
    for p in range(len(found)):
        got = [n for n, _ in rows_of(p)]
        if got != labels:
            raise SystemExit(f"{prov} p{found[p][0]}: rows {got} are not {labels}")
    want = {fold(n): (n, u) for n, u, _ in expect}
    groups = []                                   # (row index, [commune row indices])
    for i, name in enumerate(labels):
        if fold(name) in want and fold(name) not in {fold(labels[g]) for g, _ in groups}:
            groups.append((i, []))
        elif groups:
            groups[-1][1].append(i)
        else:
            raise SystemExit(f"{prov}: row {name!r} comes before any municipality")
    if len(groups) != len(expect):
        missing = sorted(set(want) - {fold(labels[g]) for g, _ in groups})
        raise SystemExit(f"{prov}: {len(groups)} municipalities found, {len(expect)} "
                         f"expected; missing {missing}")
    rowvals = [rows_of(p) for p in range(len(found))]

    # A MUNICIPALITY ROW CAN BE WRONG WHERE ITS COMMUNES ARE RIGHT. Malanje prints Kunda dya
    # Baze as 494,052 people (157,403 Portuguese, 335,552 Kimbundu) above communes of 8,889
    # and 4,852. Each municipality must equal its religion universe; where the printed row
    # does not and the sum of its communes does, the communes are used and it is reported.
    order, universe, repaired, mismatched, n_communes = [], {}, [], [], 0
    for i, comm in groups:
        name = labels[i]
        n_communes += len(comm)
        rd_u = want[fold(name)][1]
        u_row = lead[i][1][uc0]
        u_com = sum(lead[j][1][uc0] or 0 for j in comm) if comm else None
        if u_row is not None and abs(u_row - rd_u) <= 3:
            use = [i]
        elif u_com is not None and abs(u_com - rd_u) <= 3:
            use = comm
            repaired.append(f"{name}: row says {u_row:,}, communes {u_com:,}, "
                            f"religion table {rd_u:,}")
        else:
            use = [i]
            mismatched.append(f"{name}: row {u_row}, communes {u_com}, religion {rd_u:,}")
        order.append(name)
        universe[name] = sum(lead[j][1][uc0] for j in use)
        for p, (pno, centres, hwords, rows) in enumerate(found):
            uc = unicols[p]
            for c in range(len(centres)):
                if c == uc:
                    continue
                cat = mapping[(p, c)]
                v = 0
                for j in use:
                    x = rowvals[p][j][1][c]
                    if x is None:
                        blanks.append((rowvals[p][j][0], cat))
                    v += x or 0
                if (name, cat) in values and values[(name, cat)] != v:
                    raise SystemExit(f"{prov}/{name} {cat}: {values[(name, cat)]:,} on one "
                                     f"panel, {v:,} on another")
                values[(name, cat)] = v
    for p, (pno, centres, hwords, rows) in enumerate(found):
        uc = unicols[p]
        prow = rows[0][2]
        for c in range(len(centres)):
            if c != uc:
                values[("__province__", mapping[(p, c)])] = prow[c] or 0
    cats = [c for c in CATEGORIES if (order[0], c) in values]
    ids = {n: g for n, _, g in expect}
    ids = {n: ids[want[fold(n)][0]] for n in order}
    return {"order": order, "universe": universe, "values": values, "ids": ids,
            "province_universe": prov_universe, "categories": cats, "blanks": blanks,
            "pages": [f[0] for f in found], "communes": n_communes, "repaired": repaired,
            "mismatched": mismatched}


def read_national():
    """Quadro 6.2 of the national volume by province: {(province, category): count}."""
    import fitz
    doc = fitz.open(pdf_path(NATIONAL))
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
                continue
            seen.add(fold(name))
            for c in range(len(centres)):
                if c != uc:
                    out[(fold(name), mapping[(p, c)])] = slot[c] or 0
            if uc is not None and slot[uc] is not None:
                out[(fold(name), UNIVERSE)] = slot[uc]
    return out


def nat_key(prov):
    # the national volume writes `Cuanza-Norte`; fold() drops the hyphen
    return fold(prov)


# ---------------------------------------------------------------- checks, split, join


def check(data, national):
    bad = []
    print(f"{'province':16s} {'munis':>5s} {'cats':>4s} {'universe':>11s} {'vs national':>11s}"
          f" {'blanks':>6s}  pages")
    for prov in MUNICIPALITIES:
        d = data[prov]
        for n in d["order"]:
            s = sum(d["values"][(n, c)] for c in d["categories"])
            if abs(s - d["universe"][n]) > SLACK and (prov, n) not in ACCEPT_OFF:
                bad.append(f"{prov}/{n}: categories sum to {s:,}, universe {d['universe'][n]:,}")
        for c in d["categories"]:
            mine = sum(d["values"][(n, c)] for n in d["order"])
            prow = d["values"].get(("__province__", c))
            if prow is not None and abs(mine - prow) > SLACK:
                bad.append(f"{prov} {c}: municipalities {mine:,}, province row {prow:,}")
        s = sum(d["universe"].values())
        nat = national.get((nat_key(prov), UNIVERSE))
        print(f"{prov:16s} {len(d['order']):5d} {len(d['categories']):4d} {s:11,} "
              f"{'' if nat is None else f'{s - nat:+,}':>11s} {len(d['blanks']):6d}  "
              f"{d['pages']} communes {d['communes']}")
    print()
    for c in CATEGORIES:
        mine = sum(d["values"].get((n, c), 0) for d in data.values() for n in d["order"])
        nat = national.get(("angola", c))
        print(f"  {c:26s} {mine:12,}   national volume {'' if nat is None else f'{nat:,}':>12}")
    if bad:
        raise SystemExit("the parse does not reconcile:\n  " + "\n  ".join(bad))
    print("\ncheck 1-2: every municipality sums to its universe and every province's "
          "municipalities to its own province row")


def split_foreign(data, national):
    """Each municipality's `Línguas estrangeiras` -> the nine, by its province's mix."""
    out = {}
    print("\nforeign languages: municipal total vs the national volume's nine, per province")
    for prov in MUNICIPALITIES:
        d = data[prov]
        if prov in FOREIGN_IN_OTHER:
            mine = sum(d["values"][(n, "Outras línguas")] for n in d["order"])
            nat_o = national[(nat_key(prov), "Outras línguas")]
            nat_f = sum(national[(nat_key(prov), c)] for c in FOREIGN)
            print(f"  {prov:16s} no foreign column: its `Outras línguas` {mine:,}, the national "
                  f"volume's {nat_o:,} other + {nat_f:,} foreign = {nat_o + nat_f:,}")
            if abs(mine - nat_o - nat_f) > SLACK:
                raise SystemExit(f"{prov}: `Outras línguas` is not other + foreign")
            continue
        if FOREIGN_TOTAL not in d["categories"]:
            if not all(c in d["categories"] for c in FOREIGN):
                raise SystemExit(f"{prov}: neither a foreign total nor the nine")
            continue
        col = {c: national[(nat_key(prov), c)] for c in FOREIGN}
        colsum = sum(col.values())
        rowsum = sum(d["values"][(n, FOREIGN_TOTAL)] for n in d["order"])
        ratio = rowsum / colsum if colsum else float("nan")
        print(f"  {prov:16s} municipalities {rowsum:7,}  national {colsum:7,}  "
              f"ratio {ratio:.3f}  Lingala {100 * col['Lingala'] / colsum:5.1f}%")
        if colsum == 0 or (colsum > 500 and not 0.8 < ratio < 1.25):
            raise SystemExit(f"{prov}: foreign totals too far apart to split by")
        cells = {}
        for n in d["order"]:
            share = d["values"][(n, FOREIGN_TOTAL)]
            exact = {c: share * col[c] / colsum for c in FOREIGN}
            base = {c: int(v) for c, v in exact.items()}
            short = share - sum(base.values())
            for c in sorted(FOREIGN, key=lambda k: -(exact[k] - base[k]))[:short]:
                base[c] += 1
            assert sum(base.values()) == share, (prov, n)
            cells[n] = base
        out[prov] = cells
    print("check 4: every municipality's split sums to its measured foreign total")
    return out


def from_national(prov, expect, national):
    """A province whose volume has no language table, spread over its municipalities.

    MOXICO LESTE'S VOLUME PRINTS THE ETHNIC TABLE TWICE. Its list of tables gives Quadro 6.1
    and 6.2 the same title (`segundo os grupos étnicos ou tribos`) and page 81 prints exactly
    that: the ethnic groups in two panels, and no language table anywhere in the volume.
    The national volume does print Moxico Leste's languages, as one province row. So each
    municipality gets its own measured population aged 2+ (from the religion table, the
    same universe) shared out in the province's mix, largest remainder. Every row is
    `derived`: the province's language counts are INE's, their spread between its nine
    municipalities is not.
    """
    key = nat_key(prov)
    cats = [c for c in CATEGORIES if c != FOREIGN_TOTAL and (key, c) in national]
    col = {c: national[(key, c)] for c in cats}
    colsum = sum(col.values())
    total = sum(u for _, u, _ in expect)
    nat_u = national[(key, UNIVERSE)]
    print(f"  {prov}: no language table in its volume; the national row ({nat_u:,}, "
          f"categories {colsum:,}) shared over {len(expect)} municipalities ({total:,})")
    if abs(colsum - nat_u) > SLACK or abs(total - nat_u) > SLACK:
        raise SystemExit(f"{prov}: national row and municipal universes disagree")
    values = {}
    for n, u, _ in expect:
        exact = {c: u * col[c] / colsum for c in cats}
        base = {c: int(v) for c, v in exact.items()}
        for c in sorted(cats, key=lambda k: -(exact[k] - base[k]))[:u - sum(base.values())]:
            base[c] += 1
        assert sum(base.values()) == u
        for c in cats:
            values[(n, c)] = base[c]
    for c in cats:
        values[("__province__", c)] = col[c]
    return {"order": [n for n, _, _ in expect], "universe": {n: u for n, u, _ in expect},
            "values": values, "ids": {n: g for n, _, g in expect},
            "province_universe": nat_u, "categories": cats, "blanks": [], "pages": [],
            "communes": 0, "repaired": [], "mismatched": [], "tier": "derived"}


def rd_municipalities():
    """religiondots' 326 municipalities: {province: [(name, universe 2+, geo_id)]}."""
    import pandas as pd
    rd = pd.read_csv(RD_NORM, dtype={"geo_id": str}, keep_default_na=False)
    rd = rd[(rd["geo_level"] == "municipality")
            & (rd["source_category"] == "População com 2 ou mais anos")].copy()
    rd["province"] = rd["note"].str.extract(r"province=([^;]+)")[0].str.strip()
    out = {}
    for p, n, g, c in zip(rd["province"], rd["geo_name"], rd["geo_id"], rd["count"]):
        out.setdefault(p, []).append((n, int(c), g))
    if sorted(out) != sorted(MUNICIPALITIES):
        raise SystemExit(f"religiondots' ao.csv provinces: {sorted(out)}")
    for p, v in out.items():
        if len(v) != MUNICIPALITIES[p] or len({fold(n) for n, _, _ in v}) != len(v):
            raise SystemExit(f"religiondots' ao.csv: {p} has {len(v)} municipalities")
        if len({g for _, _, g in v}) != len(v):
            raise SystemExit(f"religiondots' ao.csv: {p} repeats a geo_id")
    return out


def write(data, foreign):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for prov in MUNICIPALITIES:
        d = data[prov]
        for n in d["order"]:
            gid = d["ids"][n]
            rows.append([gid, "municipality", prov, n, UNIVERSE, d["universe"][n], "universe"])
            for c in d["categories"]:
                if c == FOREIGN_TOTAL:
                    continue
                # Malanje's `Outras` holds its foreign speakers too, so it gets a label of its
                # own, which taxonomy/ao2024.py maps to a wider node
                label = OTHER_WITH_FOREIGN if (prov in FOREIGN_IN_OTHER
                                               and c == "Outras línguas") else c
                rows.append([gid, "municipality", prov, n, label, d["values"][(n, c)],
                             d.get("tier", "measured")])
            if prov in foreign:
                for c in FOREIGN:
                    rows.append([gid, "municipality", prov, n, c, foreign[prov][n][c],
                                 "derived"])
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "geo_level", "province", "geo_name", "source_category", "count",
                    "tier"])
        w.writerows(rows)
    print(f"wrote {OUT}  {len(rows):,} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    expect = rd_municipalities()
    national = read_national()
    data = {prov: (from_national(prov, expect[prov], national) if prov in NO_TABLE
                   else read_province(prov, expect[prov])) for prov in MUNICIPALITIES}
    n = sum(len(d["order"]) for d in data.values())
    ids = [g for d in data.values() for g in d["ids"].values()]
    if n != 326 or len(set(ids)) != 326:
        raise SystemExit(f"{n} municipalities, {len(set(ids))} distinct ids")
    print("check 3: 326 municipalities, each on religiondots' id, each language universe "
          "equal to its religion universe")
    differs = set()
    for prov, d in data.items():
        for r in d["repaired"]:
            print(f"  repaired from communes, {prov}/{r}")
        for r in d["mismatched"]:
            print(f"  universe differs from the religion table (known), {prov}/{r}")
            differs.add((prov, r.split(":")[0]))
    if differs != UNIVERSE_DIFFERS:
        raise SystemExit(f"universe differs from religiondots' for {sorted(differs)}, "
                         f"expected {sorted(UNIVERSE_DIFFERS)}")
    halve_doubled(data)
    swap_tail(data)
    if a.report:
        for prov, d in data.items():
            print(f"=== {prov}: {d['categories']}")
            for n in d["order"]:
                print(f"   {n:26s} {d['universe'][n]:10,}")
    check(data, national)
    foreign = split_foreign(data, national)
    write(data, foreign)
