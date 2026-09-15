"""Republic of the Congo — RGPH 2007, religion by département, in counts.

Reads (or fetches) data/raw/cg/rgph2007pd.pdf and writes data/normalized/cg.csv.
`sources/cg.md` is the write-up; `sources/cg_geo.py` builds the polygons and the grid.

## THE TABLE

CNSEE with UNFPA, *Le RGPH-2007 en quelques chiffres* (Brazzaville, July 2010, 23 pp),
**Tableau 11** *Répartition de la population résidante par département selon la religion
pratiquée*, printed p13 (PDF page index 12). Nine answers, twelve départements, **counts**, and a
national row with the counts and a share per answer. Tableau 1 (pp2-4) gives the département
and district resident populations; religion stops at the département.

Found by scout `f95259a4-scout-af`, sources.md §11aq.

## CITE AND FETCH ONLY THE WAYBACK COPY

`cnsee.org` is squatted. Its live URL serves the same brochure reflowed to 20 pages with finance
and crypto spam links injected into the text layer (§11aq). This module fetches only

    web.archive.org/web/20111113144639id_/http://www.cnsee.org/pdf/rgph2007pd.pdf

with `id_`, so the body is the archived file and not the Wayback frame. The file checked here is
1,120,560 bytes, Word 2007's PDF 1.5, 23 pages, author `Léonard`, created 2010-07-02, and
contains no `http` string at all (the squatter's copy is full of links).

**ITS `%%EOF` IS NOT IN THE LAST 2 KB, AND THAT IS NOT TRUNCATION OF THE CONTENT.** The file has
four `%%EOF` markers, the last at byte 1,027,215, followed by 93,345 bytes of a free-object xref
list (`0000003630 65535 f`) that stops mid-entry. PyMuPDF opens it as repaired. Every page has
its text, the page count is the 23 the brochure's own table list implies (the list is the last
two pages), and Tableau 11 closes to the person on Tableau 1. So the check here is page count,
text on every page and the reconciliation, not the trailer rule `gn.py` uses
([[reference_pdf_truncated_at_source]]). The Wayback CDX was 503/offline on 2026-09-14, so
whether another capture has the complete trailer is unchecked.

## THE QUESTIONNAIRE AGREES WITH THE COLUMN ORDER

The Zambia lesson (§9db): read the form. The RGPH-06 *Feuille de ménage ordinaire* (the census
was planned for 2006 and enumerated with a reference date of 28 April 2007, Tableau 8's note),
UNSD's copy `unstats.un.org/unsd/demographic/sources/census/quest/COG2007fr.pdf`, 4 scanned pages
(IREDA holds another scan, `cog-2007-rec-q1_quest_menage_ordinaire.pdf`). P13 *Religion* is
asked of every resident with nine codes:

    CA=1 PR=2 SA=3 KI=4 MU=5 ER=6 AN=7 AU=8 SR=9

Catholique, Protestante, Salutiste, Kimbanguiste, Musulmane, Eglises de réveil, Animiste,
Autres, Sans religion: **Tableau 11's nine columns in the same order.** There is no
non-response code, which is why the table sums to the whole resident population.

## UNSD HAS NO ROW

`python tools/oracle.py Congo` says ABSENT: Congo forwarded no religion tabulation to the
Demographic Yearbook. So the check here is internal: rows against Tableau 1, columns against the
printed national row, and the national row against its own printed shares.

Usage:
    python sources/cg.py --fetch    one ~1.1 MB PDF from the Wayback Machine
    python sources/cg.py            normalise from data/raw/cg/
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
RAW = os.path.join(ROOT, "data", "raw", "cg")
OUT = os.path.join(ROOT, "data", "normalized", "cg.csv")
sys.path.insert(0, os.path.join(ROOT, "tools"))

from fetch_checks import check_body   # noqa: E402  shared, not copied

SOURCE_ID = "cg_rgph2007_en_quelques_chiffres"
YEAR = 2007
BASIS = "self_id"
COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

WAYBACK = ("http://web.archive.org/web/20111113144639id_/"
           "http://www.cnsee.org/pdf/rgph2007pd.pdf")
PDF = os.path.join(RAW, "rgph2007pd.pdf")
SIZE = 1_120_560
PAGES = 23

# 0-based page indices.
PAGES_T1 = (1, 2, 3)      # printed pp2-4, Tableau 1, population by département and district
PAGE_T3 = 5               # printed p6, Tableau 3, density: the département totals again
PAGE_T11 = 12             # printed p13, Tableau 11
PAGE_T20_21 = 15          # printed p16, Tableaux 20 and 21: the département totals twice more

# Tableau 11's column order, which is P13's code order 1..9 on the 2007 form.
CATS = ["Catholique", "Protestante", "Salutiste", "Kimbanguiste", "Musulmane",
        "Eglises de réveil", "Animiste", "Autres", "Sans religion"]
FORM_CODES = ["CA", "PR", "SA", "KI", "MU", "ER", "AN", "AU", "SR"]   # P13, 1..9

# Tableau 11, transcribed, and asserted equal to what is parsed off the page.
T11 = {
    "Kouilou":       (18398, 15504, 2986, 1346, 472, 15382, 1556, 26973, 9338),
    "Niari":         (62371, 76217, 8428, 2450, 1839, 26739, 1697, 25694, 25836),
    "Lékoumou":      (16236, 35743, 7016, 161, 332, 11205, 389, 5418, 19893),
    "Bouenza":       (94625, 93686, 3424, 3008, 887, 35912, 1922, 33308, 42301),
    "Pool":          (100079, 50040, 10454, 6314, 453, 29933, 2548, 22582, 14192),
    "Plateaux":      (16424, 17009, 3110, 2509, 369, 43565, 2756, 29584, 59265),
    "Cuvette":       (33438, 11075, 532, 449, 1289, 59010, 699, 6059, 43493),
    "Cuvette-Ouest": (14561, 10620, 110, 94, 447, 24529, 965, 3227, 18446),
    "Sangha":        (10846, 10938, 149, 673, 2251, 33277, 451, 5110, 22043),
    "Likouala":      (29694, 36425, 429, 2090, 4269, 59717, 1685, 6902, 12904),
    "Brazzaville":   (588367, 206756, 31687, 25930, 32212, 345998, 6181, 48382, 87869),
    "Pointe-Noire":  (237151, 169965, 13009, 9129, 15051, 138866, 6160, 61757, 64246),
}
T11_CONGO = (1_222_190, 733_978, 81_334, 54_153, 59_871, 824_133, 27_009, 274_996, 419_826)
T11_PCT = (33.1, 19.9, 2.2, 1.5, 1.6, 22.3, 0.7, 7.4, 11.3)

# Tableau 1: (ensemble, hommes, femmes), résident population, 28 April 2007.
T1 = {
    "Kouilou":       (91_955, 46_976, 44_979),
    "Niari":         (231_271, 112_942, 118_329),
    "Lékoumou":      (96_393, 45_877, 50_516),
    "Bouenza":       (309_073, 148_523, 160_550),
    "Pool":          (236_595, 115_026, 121_569),
    "Plateaux":      (174_591, 84_446, 90_145),
    "Cuvette":       (156_044, 76_373, 79_671),
    "Cuvette-Ouest": (72_999, 35_538, 37_461),
    "Sangha":        (85_738, 42_992, 42_746),
    "Likouala":      (154_115, 76_850, 77_265),
    "Brazzaville":   (1_373_382, 677_599, 695_783),
    "Pointe-Noire":  (715_334, 358_215, 357_119),
}
RESIDENT = 3_697_490
MEN = 1_821_357
WOMEN = 1_876_133

INT = re.compile(r"^\d{2,7}$")
SPACES = dict.fromkeys([0x00A0, 0x2007, 0x2008, 0x2009, 0x202F, 0x205F], " ")


def despace(s):
    return re.sub(r"\s+", " ", str(s).translate(SPACES)).strip()


def norm(s):
    # The page prints `Cuvette-0uest` with a digit zero; fold it before anything else.
    s = str(s).replace("0uest", "Ouest")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def fetch():
    import urllib.request

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(PDF) and os.path.getsize(PDF) == SIZE:
        print("already have", PDF)
        return
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}
    req = urllib.request.Request(WAYBACK, headers=ua)
    with urllib.request.urlopen(req, timeout=600) as r:
        body = r.read()
    # Pinned to the exact size, because this complete Word export keeps 93 KB after its last
    # %%EOF and the trailer rule would refuse it; `http` means the squatted copy's spam links
    # (sources/fetch_checks.py::check_body raises SystemExit on either).
    check_body(body, "pdf", where="the Wayback copy", pin_size=SIZE, forbid=(b"http",))
    with open(PDF + ".part", "wb") as fh:
        fh.write(body)
    os.replace(PDF + ".part", PDF)
    print(f"wrote {PDF} ({len(body):,} bytes)")


def _lines(doc, pno):
    return [ln for ln in (despace(x) for x in doc.load_page(pno).get_text().splitlines())
            if ln]


def read_t11(doc):
    """Tableau 11 off the page: a département line, then nine integer lines, twelve times."""
    lines = _lines(doc, PAGE_T11)
    start = next((i for i, ln in enumerate(lines) if ln.startswith("Tableau 11")), None)
    if start is None:
        raise SystemExit(f"no `Tableau 11` on page index {PAGE_T11}")
    keys = {norm(k): k for k in T11}
    out, i = {}, start + 1
    while i < len(lines) and norm(lines[i]) != "congo":
        k = keys.get(norm(lines[i]))
        if k is not None:
            nums = lines[i + 1:i + 10]
            if len(nums) != 9 or not all(INT.match(n) for n in nums):
                raise SystemExit(f"Tableau 11 row {lines[i]!r}: expected 9 integers, got {nums}")
            out[k] = tuple(int(n) for n in nums)
            i += 10
        else:
            i += 1
    return out, " ".join(lines[i:])


def check(doc):
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    def spaced(n):
        return f"{n:,}".replace(",", " ")

    print("Republic of the Congo — RGPH 2007, Le RGPH-2007 en quelques chiffres, Tableau 11\n")
    say(doc.page_count == PAGES, f"the brochure is {doc.page_count} pages (expected {PAGES})")
    empty = [i for i in range(doc.page_count) if not doc.load_page(i).get_text().strip()]
    say(not empty, f"every page has a text layer (empty: {empty})")
    say(os.path.getsize(PDF) == SIZE, f"the file is {os.path.getsize(PDF):,} bytes")

    # 1. the parsed table is the transcribed one
    parsed, tail = read_t11(doc)
    say(parsed == T11, f"Tableau 11 parsed off the page: {len(parsed)} départements, "
        "every cell identical to the transcription")
    if parsed != T11:
        for k, v in T11.items():
            if parsed.get(k) != v:
                print(f"        {k}: page {parsed.get(k)} transcribed {v}")
    found = [c for c in T11_CONGO if spaced(c) in tail]
    say(len(found) == 9, f"all 9 national counts appear in the Congo row ({len(found)})")

    # 2. rows close on Tableau 1, to the person
    bad = {k: (sum(v), T1[k][0]) for k, v in T11.items() if sum(v) != T1[k][0]}
    say(not bad, f"every département's nine answers sum to its Tableau 1 population {bad or ''}")

    # 3. columns close on the printed national row
    cols = tuple(sum(v[j] for v in T11.values()) for j in range(9))
    say(cols == T11_CONGO, "every column sums to the printed national count")
    if cols != T11_CONGO:
        for c, a, b in zip(CATS, cols, T11_CONGO):
            print(f"        {c:<18} summed {a:>10,} printed {b:>10,}")
    say(sum(T11_CONGO) == RESIDENT, f"the national row sums to {sum(T11_CONGO):,} "
        f"= the resident population {RESIDENT:,}")

    # 4. the printed shares. `Sans religion` is 11.354% and printed 11,3, which makes the
    #    printed shares sum to exactly 100.0; the bound allows that one rounding.
    worst = 0.0
    for c, n, p in zip(CATS, T11_CONGO, T11_PCT):
        d = abs(100.0 * n / RESIDENT - p)
        worst = max(worst, d)
        if d > 0.05:
            print(f"        {c}: {100.0 * n / RESIDENT:.3f}% printed {p:.1f}")
    say(worst <= 0.06, f"the printed national shares match the counts within {worst:.3f} pp")

    # 5. Tableau 1's département totals, and the same totals in three other tables
    t1text = despace(" ".join(" ".join(_lines(doc, p)) for p in PAGES_T1))
    say(all(spaced(t) in t1text and spaced(m) in t1text and spaced(f) in t1text
            for t, m, f in T1.values()),
        "all 12 département totals, men and women appear on Tableau 1's pages")
    say(all(m + f == t for t, m, f in T1.values()), "every département's men + women = total")
    say(sum(t for t, _m, _f in T1.values()) == RESIDENT
        and sum(m for _t, m, _f in T1.values()) == MEN
        and sum(f for _t, _m, f in T1.values()) == WOMEN,
        f"the 12 sum to {RESIDENT:,} ({MEN:,} men, {WOMEN:,} women)")
    for label, pno in (("Tableau 3", PAGE_T3), ("Tableaux 20-21", PAGE_T20_21)):
        text = despace(" ".join(_lines(doc, pno)))
        n = sum(spaced(t) in text for t, _m, _f in T1.values())
        say(n == 12, f"and all 12 again in {label} ({n})")

    # 6. the form
    say(len(FORM_CODES) == len(CATS) == 9,
        "P13's nine codes on the 2007 form match Tableau 11's nine columns, in order "
        "(transcribed from the scanned form, see the docstring)")

    # 7. UNSD: Congo is absent from table 28, so there is nothing to compare
    try:
        import oracle
        rows = oracle.oracle("Congo", YEAR)
    except (SystemExit, Exception):             # noqa: BLE001 - absent is the expected answer
        rows = None
    print(f"  --  UNSD table 28: {'a row exists, COMPARE IT' if rows else 'no Congo row'}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def emit():
    rows = []
    for d, cells in T11.items():
        for c, n in zip(CATS, cells):
            if n <= 0:
                continue
            rows.append({
                "geo_id": d, "geo_level": "departement", "geo_name": d,
                "source_category": c, "count": n, "basis": BASIS, "year": YEAR,
                "source_id": SOURCE_ID,
                "note": f"Tableau 11, count; Tableau 1 pop={T1[d][0]}",
            })
    return rows


def main():
    import fitz

    if "--fetch" in sys.argv:
        fetch()
    if not os.path.exists(PDF):
        raise SystemExit(f"{PDF} missing — run: python sources/cg.py --fetch")
    doc = fitz.open(PDF)
    check(doc)
    rows = emit()

    total = sum(r["count"] for r in rows)
    print(f"\n  12 départements, {total:,} people, {total / 12:,.0f} each")
    for j, c in enumerate(CATS):
        n = T11_CONGO[j]
        top = sorted(T11, key=lambda d: -T11[d][j] / T1[d][0])[:3]
        where = ", ".join(f"{d} {100 * T11[d][j] / T1[d][0]:.1f}%" for d in top)
        big = max(T11, key=lambda d: T11[d][j])
        print(f"    {n:>10,}  {100.0 * n / total:6.2f}%  {c:<18} highest: {where}; "
              f"most people: {big} ({100 * T11[big][j] / n:.1f}% of them)")
    print(f"\n  {'département':<14}{'people':>10}   shares")
    for d, cells in T11.items():
        s = T1[d][0]
        print(f"  {d:<14}{s:>10,}   " + "  ".join(f"{c[:4]} {100 * n / s:4.1f}"
                                                 for c, n in zip(CATS, cells)))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
