"""Niger — RGP/H 2012, religion by région.

Reads (or fetches) data/raw/ne/ETAT_STRUCTURE_POPULATION.pdf and writes data/normalized/ne.csv.
`sources/ne.md` is the write-up; `sources/ne_geo.py` builds the eight régions and the grid.

## THE TABLE

Institut National de la Statistique, *État et structure de la population du Niger en 2012*
(RGP/H 2012 thematic volume, 88 pp), **Tableau A 11, "Répartition de la population par région
selon la religion"**, PDF p.88. Eight régions and a Total row, in counts:

    Sans religion  Musulman  Chrétien  Animiste  Autre à préciser  ND  Total

The column order was read off the rendered page; the text layer gives the same order with `Total`
moved to the front of the header, which is pinned as a string. Nothing finer than région crosses
religion anywhere in the volume. Every column of the Total row equals UNSD Demographic Yearbook
table 28's 2012 row for Niger, `ND` being its `Unknown`.

## THE QUESTIONNAIRE

Household form (17 December 2012; UNSD's copy, `unstats.un.org/unsd/demographic/sources/census/
quest/NER2012frHh.pdf`, read 2026-09-15), column C07, *Quelle est la religion de [PRENOM] ?*:
**0 Sans religion, 1 Musulmane, 2 Chrétienne, 3 Animiste, 9 Autre à préciser**. "Pour les enfants
en bas âge, prendre le code correspondant à la religion du père ou de la mère." There is no code
for no answer, so `ND` is the processing's own cell, and animist is offered beside no religion.
The volume (p.17) quotes the same question.

## THE OTHER TABLES IT IS CHECKED AGAINST, ALL IN THE SAME VOLUME

- Tableau 3 (p.23), population by région 1977-2012: the 2012 column is A11's Total column.
- Tableau 20 (p.59), the same table as one-decimal shares. They are shares of the people who
  stated a religion (ND left out), not of everyone, and one cell is forced so the row closes
  (`T20_FORCED`).
- Section V.1 (pp.57-58): the national counts, and each région's share of the country's Christians
  and animists, in prose.
- Tableau 7 (p.32), density by région; `ne_geo.py` turns it into an area band per région.
- Tableau 23 (pp.61-62), foreigners by région, printed for the record: Niamey and Tillabéri hold
  half of the 113,647 foreigners, and the report puts Tillabéri's Christians down partly to the
  refugee camps from the war in Mali (p.58).

Usage:
    python sources/ne.py --fetch    one 1.9 MB PDF from stat-niger.org
    python sources/ne.py            normalise from data/raw/ne/
"""

import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ne")
OUT = os.path.join(ROOT, "data", "normalized", "ne.csv")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from fetch_checks import FetchCheckError, check_body, digest   # noqa: E402  shared, not copied

SOURCE_ID = "ne_rgph2012_etat_structure_tableau_a11"
YEAR = 2012
BASIS = "self_id"
COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

_PATH = "stat-niger.org/wp-content/uploads/2020/05/ETAT_STRUCTURE_POPULATION.pdf"
URLS = ["https://" + _PATH, "https://www." + _PATH,
        "https://web.archive.org/web/2026id_/https://" + _PATH]
PDF = os.path.join(RAW, "ETAT_STRUCTURE_POPULATION.pdf")
# stat-niger.org on 2026-09-15: SHA-1 in CDX base32 form, and the size.
PIN_DIGEST = "N5DQIOCM7DAYQGQ3E5HKDNDC4LPJJWKO"
PIN_SIZE = 1_883_934
PAGES = 88

PAGE_T3 = 22        # 0-based; PDF p.23, Tableau 3
PAGE_T7 = 31        # PDF p.32, Tableau 7
PAGE_V1 = (56, 57)  # PDF pp.57-58, section V.1 prose
PAGE_T20 = 58       # PDF p.59, Tableau 20
PAGE_T23 = (60, 61)  # PDF pp.61-62, Tableau 23
PAGE_A11 = 87       # PDF p.88, Tableau A 11

CATS = ["Sans religion", "Musulman", "Chrétien", "Animiste", "Autre à préciser", "ND"]
A11_HEADER = "REGION Religion Total Sans religion Musulman Chrétien Animiste Autre à préciser ND"

# The table's own row labels -> the unit id used downstream (ASCII, as printed, title case).
LABELS = {"AGADEZ": "Agadez", "DIFFA": "Diffa", "DOSSO": "Dosso", "MARADI": "Maradi",
          "TAHOUA": "Tahoua", "TILLABERI": "Tillaberi", "ZINDER": "Zinder", "NIAMEY": "Niamey"}
REGIONS = list(LABELS.values())

# Transcribed from the rendered page, and asserted equal to what is parsed off it.
# (Sans religion, Musulman, Chrétien, Animiste, Autre à préciser, ND, Total)
A11 = {
    "Agadez":    (106, 483_921, 1_015, 433, 30, 2_115, 487_620),
    "Diffa":     (125, 589_946, 1_180, 651, 22, 1_897, 593_821),
    "Dosso":     (3_774, 2_016_264, 4_948, 6_863, 193, 5_671, 2_037_713),
    "Maradi":    (3_791, 3_379_780, 6_420, 3_743, 360, 8_000, 3_402_094),
    "Tahoua":    (4_603, 3_307_027, 4_245, 4_898, 284, 7_308, 3_328_365),
    "Tillaberi": (4_979, 2_685_707, 21_292, 5_656, 175, 4_673, 2_722_482),
    "Zinder":    (4_643, 3_515_602, 3_403, 9_053, 412, 6_651, 3_539_764),
    "Niamey":    (1_027, 1_000_642, 14_353, 3_489, 1_044, 6_293, 1_026_848),
}
A11_TOTAL = (23_048, 16_978_889, 56_856, 34_786, 2_520, 42_608, 17_138_707)
TOTAL = 17_138_707

# UNSD table 28 labels for the same six columns.
UNSD = {"No Religion": "Sans religion", "Muslim": "Musulman", "Christian": "Chrétien",
        "Animist": "Animiste", "Other": "Autre à préciser", "Unknown": "ND"}

# Tableau 7, persons per km2 in 2012, one decimal; ne_geo.py bands each région's area with it.
DENSITY_2012 = {"Agadez": 0.7, "Diffa": 3.8, "Dosso": 60.2, "Maradi": 81.4, "Tahoua": 29.4,
                "Tillaberi": 28.0, "Zinder": 22.7, "Niamey": 4026.9}
DENSITY_2012_NIGER = 13.5

# Tableau 20 cells that are not the stated-religion share rounded to one decimal, as (région,
# column): printed value. In each of the three rows the rounded shares sum to 99.9 and the printed
# cell is 0.1 above its rounding, so the row was forced to 100.0 (Mali's Tableau 6.13 did the same).
T20_FORCED = {("Dosso", "Animiste"): 0.4, ("Tahoua", "Chrétien"): 0.2, ("Niamey", "Chrétien"): 1.5}

# Tableau 23, foreigners by région (total of both sexes).
FOREIGN_2012 = {"Agadez": 2_346, "Diffa": 7_927, "Dosso": 9_170, "Maradi": 10_857,
                "Tahoua": 14_806, "Tillaberi": 27_585, "Zinder": 10_297, "Niamey": 30_659}
FOREIGN_TOTAL = 113_647

SPACES = dict.fromkeys([0x00A0, 0x2007, 0x2008, 0x2009, 0x202F, 0x205F], " ")
INT = re.compile(r"^\d+$")


def despace(s):
    return re.sub(r"\s+", " ", str(s).translate(SPACES)).strip()


def fold(s):
    import unicodedata

    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).upper()


def fetch():
    import urllib.request

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(PDF) and os.path.getsize(PDF) == PIN_SIZE:
        print("already have", PDF)
        return
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}
    for url in URLS:
        try:
            req = urllib.request.Request(url, headers=ua)
            with urllib.request.urlopen(req, timeout=600) as r:
                body = r.read()
        except Exception as e:                       # noqa: BLE001 - try the next copy
            print(f"  {url[:70]}...: {e}")
            continue
        try:
            check_body(body, "pdf", where=f"{url[:70]}...", pin_size=PIN_SIZE,
                       pin_digest=PIN_DIGEST)
        except FetchCheckError as e:
            print(f"  {e}")
            continue
        with open(PDF + ".part", "wb") as fh:
            fh.write(body)
        os.replace(PDF + ".part", PDF)
        print(f"wrote {PDF} ({len(body):,} bytes) from {url[:40]}")
        return
    raise SystemExit("no copy of État et structure de la population du Niger en 2012 matched the pin")


def _text(doc, pno):
    return despace(doc.load_page(pno).get_text())


def read_a11(doc):
    """-> (header string, {unit: 7-tuple}, total 7-tuple), off PDF p.88's text layer."""
    t = _text(doc, PAGE_A11)
    cap = "Tableau A 11 : Répartition de la population par région selon la religion"
    i, k = t.find(cap), t.find("Tableau A 12")
    if i < 0 or k < 0:
        raise SystemExit("Tableau A 11's caption or the next caption is not on PDF p.88")
    body = t[i + len(cap):k]
    j = body.find("AGADEZ")
    header, toks = body[:j].strip(), body[j:].split()
    rows, n = {}, 0
    for label in list(LABELS) + ["Total"]:
        if toks[n] != label:
            raise SystemExit(f"A11: expected row {label!r}, found {toks[n:n + 3]}")
        cells = toks[n + 1:n + 8]
        if len(cells) != 7 or not all(INT.match(c) for c in cells):
            raise SystemExit(f"A11 row {label}: {cells}")
        rows[LABELS.get(label, label)] = tuple(int(c) for c in cells)
        n += 8
    if n != len(toks):
        raise SystemExit(f"A11: {len(toks) - n} tokens left after the Total row: {toks[n:]}")
    total = rows.pop("Total")
    return header, rows, total


def _rows_by_label(tokens, labels):
    """Split a token list at the given row labels -> {label: [tokens up to the next label]}."""
    idx = [(n, t) for n, t in enumerate(tokens) if t in labels]
    return {t: tokens[n + 1:(idx[m + 1][0] if m + 1 < len(idx) else len(tokens))]
            for m, (n, t) in enumerate(idx)}


def _num(s):
    return float(s.replace(",", "."))


def check(doc):
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    print("Niger — RGP/H 2012, État et structure de la population, Tableau A 11\n")
    say(doc.page_count == PAGES, f"the volume is {doc.page_count} pages (expected {PAGES})")
    with open(PDF, "rb") as fh:
        d = digest(fh.read())
    say(d == PIN_DIGEST, f"the PDF's digest is the pinned {PIN_DIGEST} ({d})")
    say(not doc.load_page(PAGE_A11).get_images(), "PDF p.88 has no images: the table is text")

    # 1. the table parses, matches the transcription, and closes
    header, rows, total = read_a11(doc)
    say(header == A11_HEADER, f"A11's header reads {header!r}")
    say(rows == A11 and total == A11_TOTAL, "every A11 cell parsed = the transcription (8 rows + Total)")
    say(all(sum(r[:6]) == r[6] for r in list(rows.values()) + [total]),
        "each row's six columns sum to its Total")
    say(all(sum(r[c] for r in rows.values()) == total[c] for c in range(7)),
        "the eight régions sum to the Total row in all 7 columns")
    say(total[6] == TOTAL, f"the Total is {TOTAL:,}")

    # 2. UNSD table 28
    try:
        import oracle
        u = oracle.oracle("Niger", YEAR)
    except SystemExit:
        u = None
    if not u:
        print("  -- oracle cache not present; UNSD check skipped")
    else:
        tot = next((v for k, v in u.items() if "total" in k.lower()), {})
        got = {UNSD.get(k, k): v for k, v in tot.items() if k != oracle.TOTAL}
        say(got == dict(zip(CATS, total[:6])) and tot.get(oracle.TOTAL) == TOTAL,
            "UNSD table 28, Niger 2012, equals A11's Total row in all six columns and the total")

    # 3. Tableau 3's 2012 column is A11's Total column
    t3 = _text(doc, PAGE_T3)
    toks = t3[t3.find("Agadez"):].split()
    t3rows = _rows_by_label(toks, {"Agadez", "Diffa", "Dosso", "Maradi", "Tahoua", "Tillabéry",
                                   "Zinder", "Niamey", "Niger"})
    got = {("Tillaberi" if k == "Tillabéry" else k): int(v[3].replace(".", ""))
           for k, v in t3rows.items()}
    want = {**{u: r[6] for u, r in A11.items()}, "Niger": TOTAL}
    say(got == want, f"Tableau 3 (p.23), 2012 column = A11's Total column, région by région ({got})")

    # 4. Tableau 20: one-decimal shares of the people who stated a religion
    t20 = _text(doc, PAGE_T20)
    toks = t20[t20.find("AGADEZ"):t20.find("V.2.")].split()
    t20rows = _rows_by_label(toks, set(LABELS) | {"Total"})
    say(len(t20rows) == 9 and all(len(v) == 6 for v in t20rows.values()),
        "Tableau 20 (p.59) has 9 rows of five shares and a total")
    off = {"stated": {}, "everyone": {}}
    for label, vals in t20rows.items():
        u = LABELS.get(label, "Total")
        r = A11.get(u, A11_TOTAL)
        for base, denom in (("stated", r[6] - r[5]), ("everyone", r[6])):
            for c, v in zip(CATS[:5], vals[:5]):
                share = round(100.0 * r[CATS.index(c)] / denom + 1e-9, 1)
                if abs(share - _num(v)) > 1e-9:
                    off[base][(u, c)] = _num(v)
    say(off["stated"] == T20_FORCED,
        f"Tableau 20 is shares of those who stated a religion: every cell but {T20_FORCED} "
        f"rounds from A11 ({len(off['stated'])} cells off; against everyone, "
        f"{len(off['everyone'])} cells off)")
    forced_ok = True
    for (u, c) in T20_FORCED:
        r = A11[u]
        rounded = [round(100.0 * r[k] / (r[6] - r[5]) + 1e-9, 1) for k in range(5)]
        forced_ok &= (abs(sum(rounded) - 99.9) < 1e-9
                      and abs(T20_FORCED[(u, c)] - rounded[CATS.index(c)] - 0.1) < 1e-9)
    say(forced_ok, "each forced cell sits in a row whose rounded shares sum to 99.9, and is "
        "printed 0.1 above its rounding")
    say(all(_num(v[5]) == 100.0 for v in t20rows.values()), "every Tableau 20 row prints 100,0")

    # 5. section V.1's prose
    prose = despace(_text(doc, PAGE_V1[0]) + " " + _text(doc, PAGE_V1[1]))
    chr_sh = {u: 100.0 * r[2] / A11_TOTAL[2] for u, r in A11.items()}
    ani_sh = {u: 100.0 * r[3] / A11_TOTAL[3] for u, r in A11.items()}
    say(all(s in prose for s in ("16 978 889 (soit 99,3%", "56 856 adeptes (soit 0,3%",
                                 "34 786 adeptes (0,2%", "23 048 (0,1%)")),
        "p.57 prints the national counts of Muslims, Christians, animists and no religion")
    say("37,4%, 25,2% et 11,3%" in prose and "Agadez (1,8%) et Diffa (2,1%)" in prose
        and [round(chr_sh[u], 1) for u in ("Tillaberi", "Niamey", "Maradi", "Agadez", "Diffa")]
        == [37.4, 25.2, 11.3, 1.8, 2.1],
        "p.58's shares of all Christians (Tillabéri 37.4, Niamey 25.2, Maradi 11.3, Agadez 1.8, "
        "Diffa 2.1) are A11's")
    say("Zinder (26%), Dosso (19,7%) et Tillabéry (16,3%)" in prose
        and [round(ani_sh[u], 1) for u in ("Zinder", "Dosso", "Tillaberi")] == [26.0, 19.7, 16.3],
        "p.58's shares of all animists (Zinder 26, Dosso 19.7, Tillabéri 16.3) are A11's")

    # 6. Tableau 7, density: printed for ne_geo.py's area band
    t7 = _text(doc, PAGE_T7)
    toks = t7[t7.find("Agadez"):t7.find("III.3.")].split()
    t7rows = _rows_by_label(toks, {"Agadez", "Diffa", "Dosso", "Maradi", "Tahoua", "Tillabéry",
                                   "Zinder", "Niamey", "Ensemble", "Niger"})
    got = {("Tillaberi" if k == "Tillabéry" else k): _num(v[-1]) for k, v in t7rows.items() if v}
    say(t7rows.get("Ensemble") == [] and len(t7rows["Niamey"]) == 3,
        "Tableau 7's national row is labelled `Ensemble Niger`, and Niamey has no 1977 density")
    say(got == {**DENSITY_2012, "Niger": DENSITY_2012_NIGER},
        "Tableau 7 (p.32), 2012 density = the transcription")

    # 7. Tableau 23, foreigners by région
    t23 = _text(doc, PAGE_T23[0])
    t23 = t23[t23.find("Tableau 23"):] + " " + _text(doc, PAGE_T23[1])
    toks = t23.split()
    t23rows = _rows_by_label(toks, set(LABELS) | {"Total"})
    got = {LABELS[k]: int(v[4]) for k, v in t23rows.items() if k in LABELS}
    say(got == FOREIGN_2012 and sum(got.values()) == FOREIGN_TOTAL
        and all(int(v[0]) + int(v[1]) == int(v[4]) for v in t23rows.values()),
        f"Tableau 23 (pp.61-62), foreigners by région = the transcription, summing to {FOREIGN_TOTAL:,}")

    if not ok:
        raise SystemExit("reconciliation FAILED")
    return rows


def emit(rows):
    out = []
    for u in REGIONS:
        for c, n in zip(CATS, rows[u][:6]):
            if n <= 0:
                continue
            out.append({
                "geo_id": u, "geo_level": "region", "geo_name": u,
                "source_category": c, "count": n, "basis": BASIS, "year": YEAR,
                "source_id": SOURCE_ID, "note": "Tableau A 11, PDF p.88",
            })
    return out


def main():
    import fitz

    if "--fetch" in sys.argv:
        fetch()
    if not os.path.exists(PDF):
        raise SystemExit(f"{PDF} missing — run: python sources/ne.py --fetch")
    doc = fitz.open(PDF)
    rows = check(doc)
    out = emit(rows)

    total = sum(r["count"] for r in out)
    print(f"\n  8 régions, {total:,} people, {total / 8:,.0f} each")
    for c, n in zip(CATS, A11_TOTAL):
        print(f"    {n:>11,}  {100.0 * n / total:7.3f}%  {c}")
    print(f"\n  {'région':<11}{'people':>11}{'Chrétien':>10}{'share':>8}{'Animiste':>10}{'share':>8}"
          f"{'none':>7}{'share':>8}{'foreign':>9}")
    for u in REGIONS:
        r = A11[u]
        print(f"  {u:<11}{r[6]:>11,}{r[2]:>10,}{100.0 * r[2] / r[6]:7.2f}%{r[3]:>10,}"
              f"{100.0 * r[3] / r[6]:7.2f}%{r[0]:>7,}{100.0 * r[0] / r[6]:7.2f}%{FOREIGN_2012[u]:>9,}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".part", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out)
    os.replace(OUT + ".part", OUT)
    print("\nwrote", OUT, f"({len(out)} rows)")


if __name__ == "__main__":
    main()
