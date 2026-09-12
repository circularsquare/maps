"""São Tomé and Príncipe — INE, IV RGPH 2012, Quadro 8 of the seven district reports.

Reads (or fetches) data/raw/st/ and writes data/normalized/st.csv.

**THE OFFICE PUBLISHES A WHOLE CENSUS REPORT PER DISTRICT, AND EVERY ONE OF THEM CARRIES THE
FULL THIRTEEN-CATEGORY RELIGION TABLE.** `queue.md` priced São Tomé off the UNSD oracle, whose
rows are national and urban/rural, and §11w closed the row as *not chased*. INE's own site has
a folder called `Dados Distritais e Nacional Recenseamento 2012` holding eight PDFs: one per
district and one national. Each district report reprints Quadros 1-26 for that district alone,
so the religion table exists seven times over at the tier below the one the oracle shows. That
is `AGENT_BRIEF`'s highest-yield question asked of a fourth lusophone office.

**THE ROUTE IS AN OPEN AUTOINDEX AND NOT A CMS.** `www.ine.st` is Joomla with Phoca Download,
but the download tree is served by LiteSpeed with directory listing left on, so
`/phocadownload/userupload/Documentos/` walks recursively and prints all 223 files with their
sizes and dates. No plugin id sweep, no REST base, no search: just `Index of`.

**THE NATIONAL ROW MATCHES UNSD TABLE 28 IN ALL THIRTEEN CATEGORIES, TO THE PERSON**, and the
Yearbook return is a transcription INE forwarded to New York rather than a copy of these PDFs,
so the parse is checked against a lineage it shares nothing with. `check()` asserts it.

**WHY 2012 AND NOT 2024.** The V RGPH of November-December 2024 was published in July 2025 and
does tabulate religion by district, in sixteen categories, adding Islam and a separate Ateu
row; `sources/st.md` §4 records it in full. It is not drawn because **56,200 of its 209,161
people, 26.9%, are `ND`** against 1,756 non-responses in 2012, and unlike Cabo Verde's
28% the residual cannot be shown to be an age cut: it runs 1.026 to 1.037 times each
district's under-10 population, close but never exact, and the report never says who was
asked. Its category labels have also been round-tripped through machine translation, which
turns `Messiânica Mundial` into `Copa do Mundo` and `Quadro` into `Pintura`. So 2024 is used
as an independent witness on direction and 2012 is what is drawn.

**THE LOCALITY TABLE EXISTS AND IS COARSER, WHICH IS THE OPPOSITE OF THE USUAL TRADE.** INE's
2016 *Publicação dos Resultados sobre Localidades* prints religion for every locality in the
country (Tabela 3), which is far finer than seven districts, but only in seven columns:
Adventista, Assembleia de Deus, Católica, Nova Apostólica, Igreja Universal, `Outras
religiões` and `Não tem`. Its two catch-alls fold this table's other seven categories away,
exactly: 8.990 + 4.191 + 2.202 + 1.432 + 688 = 17.503 and 37.935 + 1.268 + 488 = 39.691.
`check()` asserts both identities, so the locality publication and the district reports are
each other's witnesses even though only the district reports are drawn.

Usage:
    python sources/st.py --fetch    eight PDFs, ~36 MB, from www.ine.st
    python sources/st.py            normalise from data/raw/st/
"""

import csv
import os
import re
import sys
import unicodedata
import urllib.parse

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

RAW = os.path.join(ROOT, "data", "raw", "st")
OUT = os.path.join(ROOT, "data", "normalized", "st.csv")

SOURCE_ID = "st_rgph_2012"
YEAR = 2012
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

BASE = "https://www.ine.st"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126 Safari/537.36"}

DIR_2012 = ("/phocadownload/userupload/Documentos/Recenseamentos/2012/"
            "Dados Distritais e Nacional Recenseamento 2012/")
DIR_LOC = "/phocadownload/userupload/Documentos/DADOS_LOCALIDADE_PROJECOES/"
DIR_2024 = "/phocadownload/userupload/Documentos/Recenseamentos/2024/"

# COD-AB's pcode -> (local filename, INE's own spelling of the district, remote path).
# The pcodes are `stp_admin1.shp`'s, which is the tier COD-PS calls ADM2; see
# `sources/st_geo.py`, which is where that disagreement is handled.
DISTRICTS = {
    "ST21": ("d_agua_grande.pdf", "Água-Grande",
             DIR_2012 + "Resultado_Distrital__ÁGUA-GRANDE.pdf"),
    "ST22": ("d_cantagalo.pdf", "Cantagalo",
             DIR_2012 + "Resultado Distrital_CANTAGALO.pdf"),
    "ST23": ("d_caue.pdf", "Caué",
             DIR_2012 + "Resultado_Distrital_CAUÉ.pdf"),
    "ST24": ("d_lemba.pdf", "Lembá",
             DIR_2012 + "Resultado_Distrital_LEMBÁ.pdf"),
    "ST25": ("d_lobata.pdf", "Lobata",
             DIR_2012 + "Resultado Distrital_LOBATA.pdf"),
    "ST26": ("d_me_zochi.pdf", "Mé-Zóchi",
             DIR_2012 + "Resultado_Distrital_MÉ-ZÓCHI.pdf"),
    "ST11": ("d_principe.pdf", "Região Autónoma do Príncipe",
             DIR_2012 + "Resultado Distrital_REGIÃO AUTÓNOMA DO PRÍNCIPE.pdf"),
}

NATIONAL_PDF = ("nacional2012.pdf",
                DIR_2012 + "Resultados Nacionais do IV RGPH 2012.pdf")
LOCALIDADES_PDF = ("localidades2012.pdf",
                   DIR_LOC + "Publicação dos Resultados sobre Localidades - IV RGPH 2012.pdf")
# Kept because §4 of sources/st.md reads it and because it is the newest count of the
# country; nothing in this file's output comes from it.
CENSUS_2024_PDF = ("resultado_vrgph2024.pdf", DIR_2024 + "Resultado_VRGPH 2024.pdf")

EXTRA = [NATIONAL_PDF, LOCALIDADES_PDF, CENSUS_2024_PDF]

# Quadro 8's thirteen religion columns, in print order, split as INE typesets them: the
# table is too wide for the page, so it is broken after `Maná` and the second block repeats
# the row stubs. `BLOCK1` is preceded by a `Total` column and `BLOCK2` is not.
BLOCK1 = ["Adventista", "Assembléia de Deus", "Católica Apostólica Romana",
          "Deus é amor", "Jeová", "Maná"]
BLOCK2 = ["Nova Apostólica", "Messiânica Mundial", "Igreja Universal do Reino de Deus",
          "Outras", "Não declarou", "Não sabe", "Não tem"]
CATEGORIES = BLOCK1 + BLOCK2

# UNSD Demographic Yearbook table 28's English for each. INE forwarded the return; the
# spellings are the Yearbook's and `Maná` keeps its accent there.
ORACLE_ALIAS = {
    "Adventista": "Adventist",
    "Assembléia de Deus": "Assembly of God",
    "Católica Apostólica Romana": "Roman Apostolic Catholic",
    "Deus é amor": "God is Love",
    "Jeová": "Jehovah Witness",
    "Maná": "Maná",
    "Nova Apostólica": "New Apostolic",
    "Messiânica Mundial": "Messianica",
    "Igreja Universal do Reino de Deus": "Universal of the Kingdom of God",
    "Outras": "Other",
    "Não declarou": "Not Specified",
    "Não sabe": "Unknown",
    "Não tem": "No Religion",
}

NATIONAL = 178_739

# The locality publication's Tabela 3 folds the thirteen into seven. These are the two
# folds, and `check()` asserts them against the district reports rather than describing
# them. [[reference_pdf_table_geometry]] is not needed for either: both are one addition.
LOC_OUTRAS = ["Outras", "Maná", "Jeová", "Deus é amor", "Messiânica Mundial"]
LOC_NAO_TEM = ["Não tem", "Não sabe", "Não declarou"]
LOC_TOTALS = {                    # TOTAL DO PAÍS, Tabela 3, page 41 of the 2016 publication
    "Total": 178_739,
    "Adventista": 7_239,
    "Assembléia de Deus": 5_991,
    "Católica Apostólica Romana": 99_570,
    "Nova Apostólica": 5_177,
    "Igreja Universal do Reino de Deus": 3_568,
    "Outras religiões": 17_503,
    "Não tem": 39_691,
}

# What the census counted but this map does not draw as a religion.
RESIDUALS = ["Não declarou", "Não sabe"]
DRAWN = [c for c in CATEGORIES if c not in RESIDUALS]

NUM = re.compile(r"^\d{1,3}(?:\.\d{3})*$|^\d+$")


def fold(s):
    """Casefold and strip accents — for COMPARING names, never for storing."""
    s = unicodedata.normalize("NFKD", str(s).strip().lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    wanted = [(f, p) for f, _, p in DISTRICTS.values()] + list(EXTRA)
    for name, path in wanted:
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 500_000:
            print(f"  have {name} ({os.path.getsize(dest):,} bytes)")
            continue
        url = BASE + urllib.parse.quote(path)
        print("GET", url)
        r = requests.get(url, headers=UA, timeout=900)
        r.raise_for_status()
        body = r.content
        # [[reference_pdf_truncated_at_source]] — Content-Length can match a damaged file,
        # and PyMuPDF then reports page_count=0 without raising.
        if not body.startswith(b"%PDF") or b"%%EOF" not in body[-4096:]:
            raise SystemExit(f"{name}: {len(body):,} bytes with no %%EOF trailer")
        tmp = dest + ".part"                              # [[reference_wb_truncates]]
        with open(tmp, "wb") as fh:
            fh.write(body)
        os.replace(tmp, dest)
        print(f"  {os.path.getsize(dest):,} bytes")


def _lines(doc, i):
    return [l.strip() for l in doc[i].get_text().split("\n") if l.strip()]


def _numbers(lines, start, want):
    """The first `want` Portuguese-formatted integers at or after `start`.

    Tokenised on whitespace rather than taken a line at a time: PyMuPDF usually gives one
    cell per line, but the locality publication's widest rows put the last two cells on one
    line (`17.503 39.691`), and a line-at-a-time reader silently walks past them into the
    next row. Nothing in either header is numeric, so over-reading is the only failure mode
    and it is the one this guards.
    """
    vals, j = [], start
    while j < len(lines) and len(vals) < want:
        for tok in lines[j].split():
            if NUM.match(tok):
                vals.append(int(tok.replace(".", "")))
        j += 1
    if len(vals) < want:
        raise SystemExit(f"only found {len(vals)} of {want} figures")
    if len(vals) > want:
        raise SystemExit(f"a line carried more figures than the row has cells: {vals}")
    return vals, j


def read_quadro8(path, label):
    """Quadro 8's first data row — the whole district, across both header blocks.

    Quadro 8 is `population by sex and nationality, by religion`. Its opening row is the
    district total and its remaining rows split that by nationality and sex, so one row of it
    is the district's religion table. It is used rather than Quadro 6 (religion by age and
    sex) because Quadro 6 runs to eight pages and puts its total row in a different place in
    each report, while Quadro 8 is two pages, or one for Água-Grande, and always opens with
    the total.

    The parse anchors on the LAST CELL OF EACH HEADER — `Maná` closes the first block and
    `Não tem` the second — and then takes the next seven figures in reading order. Nothing in
    either header is numeric, so the first figure after the anchor is the first cell of the
    total row. [[reference_pdf_table_geometry]] in its cheapest form: anchor the header,
    then read.
    """
    import fitz

    doc = fitz.open(path)
    if doc.page_count == 0:
        raise SystemExit(f"{path}: PyMuPDF reports zero pages")
    pages = [i for i in range(doc.page_count) if "Quadro 8" in doc[i].get_text()]
    if not pages:
        raise SystemExit(f"{path}: no page mentions Quadro 8")
    lines = []
    for i in pages:
        lines.extend(_lines(doc, i))
    doc.close()

    try:
        a = lines.index("Maná")
    except ValueError:
        raise SystemExit(f"{label}: Quadro 8's first header does not end in 'Maná'")
    block1, j = _numbers(lines, a + 1, 1 + len(BLOCK1))

    rest = lines[j:]
    b = None
    for k, tok in enumerate(rest):
        if tok in ("Não tem", "tem"):
            b = k
            break
    if b is None:
        raise SystemExit(f"{label}: Quadro 8's second header does not end in 'Não tem'")
    block2, _ = _numbers(rest, b + 1, len(BLOCK2))

    total = block1[0]
    counts = dict(zip(BLOCK1, block1[1:]))
    counts.update(zip(BLOCK2, block2))
    if sorted(counts) != sorted(CATEGORIES):
        raise SystemExit(f"{label}: parsed {sorted(counts)}")
    got = sum(counts.values())
    if got != total:
        raise SystemExit(f"{label}: the thirteen categories sum to {got:,}, Quadro 8's "
                         f"total column says {total:,}")
    return total, counts


def parse():
    """-> {pcode: (name, total, {category: count})}, plus the national row."""
    out = {}
    for pcode, (fname, name, _) in DISTRICTS.items():
        path = os.path.join(RAW, fname)
        if not os.path.exists(path):
            raise SystemExit(f"{path} is missing — run `python sources/st.py --fetch`")
        total, counts = read_quadro8(path, name)
        out[pcode] = (name, total, counts)
        print(f"  {pcode}  {name:<30}{total:>8,}  "
              f"catholic {counts['Católica Apostólica Romana'] / total:6.1%}")

    nat_path = os.path.join(RAW, NATIONAL_PDF[0])
    if not os.path.exists(nat_path):
        raise SystemExit(f"{nat_path} is missing — run `python sources/st.py --fetch`")
    nat_total, nat = read_quadro8(nat_path, "São Tomé e Príncipe")
    return out, (nat_total, nat)


def check_localities(national):
    """The 2016 locality publication's national row, folded, against the district reports.

    Tabela 3 is religion by locality with seven columns rather than thirteen. It is a
    separate publication, typeset three years later, and its two catch-alls are exact sums
    of this table's categories — so agreement is a check on the parse and on the census's
    own arithmetic at once. Parsed from the PDF rather than trusted from `LOC_TOTALS`, which
    is only what to expect.
    """
    import fitz

    path = os.path.join(RAW, LOCALIDADES_PDF[0])
    if not os.path.exists(path):
        print("  locality check SKIPPED — localidades2012.pdf is not on disk")
        return
    doc = fitz.open(path)
    row = None
    for i in range(doc.page_count):
        lines = _lines(doc, i)
        if "TOTAL DO PAÍS" not in lines or "Tabela 3:" not in " ".join(lines[:8]):
            continue
        vals, _ = _numbers(lines, lines.index("TOTAL DO PAÍS") + 1, 8)
        row = vals
        break
    doc.close()
    if row is None:
        raise SystemExit("localidades2012.pdf: no Tabela 3 page with a TOTAL DO PAÍS row")

    got = dict(zip(["Total", "Adventista", "Assembléia de Deus",
                    "Católica Apostólica Romana", "Nova Apostólica",
                    "Igreja Universal do Reino de Deus", "Outras religiões",
                    "Não tem"], row))
    if got != LOC_TOTALS:
        raise SystemExit(f"Tabela 3's national row reads {got}, expected {LOC_TOTALS}")

    want_outras = sum(national[c] for c in LOC_OUTRAS)
    want_nao = sum(national[c] for c in LOC_NAO_TEM)
    for k in ("Adventista", "Assembléia de Deus", "Católica Apostólica Romana",
              "Nova Apostólica", "Igreja Universal do Reino de Deus"):
        if got[k] != national[k]:
            raise SystemExit(f"Tabela 3 says {k}={got[k]:,}, Quadro 8 says {national[k]:,}")
    if got["Outras religiões"] != want_outras:
        raise SystemExit(f"Tabela 3's `Outras religiões` is {got['Outras religiões']:,}, "
                         f"the five it folds sum to {want_outras:,}")
    if got["Não tem"] != want_nao:
        raise SystemExit(f"Tabela 3's `Não tem` is {got['Não tem']:,}, the three it folds "
                         f"sum to {want_nao:,}")
    print(f"  locality publication: Tabela 3's seven columns reproduce Quadro 8 exactly, "
          f"with\n    Outras religiões = {want_outras:,} over five categories and "
          f"Não tem = {want_nao:,} over three")


def check_oracle(national):
    """UNSD Demographic Yearbook table 28 — INE's own return, not a copy of these PDFs."""
    try:
        import oracle
        got = oracle.oracle("Sao Tome and Principe", YEAR)
    except Exception as exc:                                   # noqa: BLE001
        print(f"  oracle check SKIPPED ({exc}) — run `python tools/oracle.py --fetch`")
        return
    if not got:
        print("  oracle check SKIPPED — no São Tomé 2012 row")
        return
    counts = got.get(oracle.TOTAL, {})
    bad = []
    for c in CATEGORIES:
        want = counts.get(ORACLE_ALIAS[c])
        if want is None:
            bad.append(f"{c}: not in the DYB as {ORACLE_ALIAS[c]!r}")
        elif int(want) != national[c]:
            bad.append(f"{c}: DYB {int(want):,} vs Quadro 8 {national[c]:,}")
    if bad:
        raise SystemExit("UNSD table 28 disagrees with the district reports:\n   " +
                         "\n   ".join(bad))
    print(f"  UNSD table 28: all {len(CATEGORIES)} categories agree with Quadro 8, "
          "to the person")


def check(districts, national_row):
    nat_total, national = national_row

    if nat_total != NATIONAL:
        raise SystemExit(f"the national report's Quadro 8 totals {nat_total:,}, "
                         f"expected {NATIONAL:,}")
    summed = sum(t for _, t, _ in districts.values())
    if summed != NATIONAL:
        raise SystemExit(f"the seven district reports sum to {summed:,} people, the "
                         f"national report says {NATIONAL:,}")
    for c in CATEGORIES:
        got = sum(cs[c] for _, _, cs in districts.values())
        if got != national[c]:
            raise SystemExit(f"{c}: the districts sum to {got:,}, the national report "
                             f"says {national[c]:,}")
    print(f"  the seven districts sum to the national report in all {len(CATEGORIES)} "
          "categories and in total")
    print(f"  partition: EXACT at all {len(districts)} districts and nationally")

    check_localities(national)
    check_oracle(national)


def read():
    """{pcode: (name, total, {category: count})} — for st_geo.py and st_grid.py."""
    districts, _ = parse()
    return districts


# Tabela 1.3 of the 2024 report, `Distribuição da população, áreas e densidades por
# distrito`, keyed by the district names that table prints. `RAP` is Região Autónoma do
# Príncipe and `POR FAVOR` is the total row: the report was round-tripped through machine
# translation, so `Junto` became `Together` became `POR FAVOR`, `Quadro` became `Pintura`
# and `idade de união` became `idade sindical`. The FIGURES are unaffected.
NAMES_2024 = {
    "RAP": "ST11", "Água-Grande": "ST21", "Cantagalo": "ST22", "Caué": "ST23",
    "Lemba": "ST24", "Lobata": "ST25", "Mé-Zochi": "ST26",
}
TOTAL_2024 = 209_607


def read_2024():
    """{pcode: (population, area_km2)} from the V RGPH 2024, for `sources/st_geo.py`.

    This is the witness the join actually rests on. It is the same office counting the same
    seven districts twelve years later, and it prints each district's AREA beside its
    population, so it pairs a polygon with a name geometrically as well as by size. Seven
    units is too few for a rank correlation to mean much; 16.5 km² against 267 km² is not.
    """
    import fitz

    path = os.path.join(RAW, CENSUS_2024_PDF[0])
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing — run `python sources/st.py --fetch`")
    doc = fitz.open(path)
    out, total = {}, None
    for i in range(doc.page_count):
        lines = _lines(doc, i)
        if not (any(l.startswith("Tabela 1.3") for l in lines)
                and set(NAMES_2024) <= set(lines)):
            continue
        # Each row is three consecutive lines: population, area, density. The area carries
        # a decimal comma for three of the seven (`16,5`), which is why it is read straight
        # off the line rather than through `_numbers`.
        for j, l in enumerate(lines):
            if l in NAMES_2024 or l == "POR FAVOR":
                pop = _numbers(lines, j + 1, 1)[0][0]
                area = float(lines[j + 2].replace(".", "").replace(",", "."))
                if not 5.0 <= area <= 1100.0:
                    raise SystemExit(f"Tabela 1.3: {l} has area {area}, which is not a "
                                     "São Tomé district in km²")
                if l == "POR FAVOR":
                    total = pop
                else:
                    out[NAMES_2024[l]] = (pop, area)
        break
    doc.close()
    if sorted(out) != sorted(DISTRICTS):
        raise SystemExit(f"Tabela 1.3 of the 2024 report gave {sorted(out)}")
    if total != TOTAL_2024 or sum(p for p, _ in out.values()) != TOTAL_2024:
        raise SystemExit(f"Tabela 1.3 sums to {sum(p for p, _ in out.values()):,} against a "
                         f"printed total of {total} and an expected {TOTAL_2024:,}")
    return out


def normalise():
    districts, national_row = parse()
    check(districts, national_row)

    rows = []
    for pcode, (name, _, counts) in districts.items():
        for c in CATEGORIES:
            if counts[c] == 0:
                continue
            rows.append({
                "geo_id": pcode,
                "geo_level": "district",
                "geo_name": name,
                "source_category": c,
                "count": counts[c],
                "basis": BASIS,
                "year": YEAR,
                "source_id": SOURCE_ID,
                "note": "IV RGPH 2012, Quadro 8",
            })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".part"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, OUT)

    drawn = sum(r["count"] for r in rows if r["source_category"] in DRAWN)
    print(f"\nwrote {OUT}")
    print(f"  {len(rows)} rows, {len(districts)} districts, "
          f"{sum(r['count'] for r in rows):,} people")
    print(f"  drawn {drawn:,} ({drawn / NATIONAL:.2%}); residual {NATIONAL - drawn:,} "
          f"({(NATIONAL - drawn) / NATIONAL:.2%}) did not declare or did not know")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    else:
        normalise()
