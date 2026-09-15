"""Chad — RGPH2 2009, religion by région (INSEED, État et structures de la population).

Reads (or fetches) data/raw/td/rgph2_etat_structures.pdf and writes data/normalized/td.csv.
`sources/td.md` is the write-up; `sources/td_geo.py` builds the polygons and the grid.

## THE TABLE IS IN THE STRUCTURE VOLUME, AND THE VOLUME IS ONLY ON INSEED'S RETIRED STORE

INSEED, *Deuxième Recensement Général de la Population et de l'Habitat (RGPH2, 2009): Analyse
thématique des résultats définitifs, État et structures de la population* (November 2013,
210 pp). Chapter 5 carries religion as three tables:

    Tableau 5.06  religion x milieu x sex, COUNTS                   printed p130 (index 129)
    Tableau 5.07  région x religion, % to one decimal, + population  printed p131 (index 130)
    Tableau 5.08  religion, 1993 and 2009, %                         printed p132 (index 131)

The file was served by the jdownloads component of the old `inseed.td` Joomla site and is not
on the rebuilt one. The only copy found is Wayback's 2020-06-08 capture (§11aq), fetched with
the `id_` form so the archive does not rewrite it. It is checked for `%PDF-`, for `%%EOF` in its
last 2 KB and for 210 pages, because a redirect page has been saved under a PDF's name before
and a Congo PDF once carried its end marker early.

## THE CONSTRUCTION: SHARES FITTED TO TWO PRINTED MARGINS

Chad is ABSENT from UNSD table 28 (`python tools/oracle.py Chad`), so unlike Guinea there is no
outside national count. The report itself prints both margins of the same universe:

    seed          Tableau 5.07's one-decimal shares x Tableau 5.07's région populations
    row margins   Tableau 5.07's `Effectif`, 22 régions, summing to 10,941,682
    column margins Tableau 5.06's `Ensemble` counts, six religions, summing to 10,941,682

The seed is raked (iterative proportional fitting) until every région sums to its printed
population and every religion to its printed national count. A printed `0,0` stays zero. That
removes the one-decimal rounding from both margins at once and leaves it only inside the
table, where it cannot be helped; `check_fit()` asserts how far any cell moved.

## THE UNIVERSE IS THE CENSUSED POPULATION, COLLECTIVE HOUSEHOLDS AND REFUGEE CAMPS INCLUDED

`Population totale recensée` is 10,941,682 (Tableau 2.02): ordinary households 10,621,672 and
collective households 320,010 (Tableau 2.04), which the report says are mostly refugee camps,
military camps and prisons. Tableau 5.01's text puts 235,183 foreigners in refugee camps, nearly
all Sudanese in the east. All of them are in Tableaux 5.06 and 5.07 and all of them are drawn.

**Not in any religion table: 98,191 people `estimées`** (Tableau 2.02), the populations of zones
enumerators could not reach, whose headcount was estimated. The footnote names rural Sila (printed
91,011) and urban Tibesti, the commune of Zouar (4,180), which add to 95,191 and not 98,191.
**The footnote's Sila figure is a misprint for 94,011.** Tableau 2.13 (printed p54) gives the
total population of Sila as 387,461 and of Tibesti as 25,483, against 293,450 and 21,303
censused in Tableau 5.07; the differences are 94,011 and 4,180, and those add to 98,191. So
**24.3% of Sila's people are in no religion table**, and Sila is drawn on the 293,450 counted.

## THE QUESTIONNAIRE AGREES WITH THE COLUMNS

The Zambia lesson (§9db) is that a header can be wrong. The RGPH2 household form (NADA
`anad.inseed.td` catalog 26, download 160, p3) asks **B12 RELIGION**, every age, codes
**ANI 1 animiste, CAT 2 catholique, MUS 3 musulman, PRO 4 protestant, AUT 5 autres, SAN 6
sans**. Six boxes, the table's six columns in the same order. Animists had their own box, so
`Sans religion` is not the Mozambique case (spec §6.3a-ii, 2026-09-14): nothing on the form
routes traditional religion into it.

Usage:
    python sources/td.py --fetch    one ~5.5 MB PDF from the Wayback Machine
    python sources/td.py            normalise from data/raw/td/
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
RAW = os.path.join(ROOT, "data", "raw", "td")
OUT = os.path.join(ROOT, "data", "normalized", "td.csv")

from fetch_checks import check_body   # noqa: E402  shared, not copied

SOURCE_ID = "td_rgph2_2009_etat_structures"
YEAR = 2009
BASIS = "self_id"
COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

WAYBACK = ("https://web.archive.org/web/20200608144533id_/https://www.inseed.td/index.php/"
           "component/jdownloads/send/7-documents-et-publications-demographique/"
           "318-etat-structures-population-rgph2009")
PDF = os.path.join(RAW, "rgph2_etat_structures.pdf")
PAGES = 210

# 0-based page indices.
PAGE_T202 = 40        # printed p41, Tableaux 2.01 and 2.02
PAGE_T204 = 42        # printed p43, Tableaux 2.04 and 2.05
PAGE_T506 = 129       # printed p130
PAGE_T507 = 130       # printed p131

CATS = ["Animiste", "Catholique", "Musulmane", "Protestante", "Autres religions",
        "Sans religion"]

# Tableau 5.07, transcribed in the printed row order, and asserted equal to what is parsed off
# the page. (Animiste, Catholique, Musulmane, Protestante, Autres, Sans religion), Effectif.
T507 = {
    "Batha":             ((0.2, 0.8, 98.8, 0.2, 0.0, 0.0), 488_458),
    "Borkou":            ((0.2, 0.3, 99.3, 0.2, 0.0, 0.0), 93_584),
    "Chari Baguirmi":    ((0.9, 9.8, 84.1, 4.3, 0.1, 0.9), 578_425),
    "Guéra":             ((1.3, 1.3, 95.6, 1.4, 0.0, 0.4), 538_359),
    "Hadjer Lamis":      ((0.3, 0.8, 98.2, 0.6, 0.0, 0.0), 566_858),
    "Kanem":             ((0.1, 0.2, 99.4, 0.1, 0.0, 0.1), 333_387),
    "Lac":               ((0.2, 0.5, 98.8, 0.4, 0.0, 0.0), 433_790),
    "Logone Occidental": ((0.4, 47.8, 7.7, 43.5, 0.2, 0.5), 689_044),
    "Logone Oriental":   ((0.5, 48.1, 8.6, 41.9, 0.4, 0.4), 779_339),
    "Mandoul":           ((4.3, 47.9, 8.7, 28.1, 3.6, 7.4), 628_065),
    "Mayo Kebbi Est":    ((32.0, 17.6, 17.9, 25.4, 1.0, 6.2), 774_782),
    "Mayo Kebbi Ouest":  ((12.5, 29.5, 11.1, 30.3, 1.4, 15.2), 564_470),
    "Moyen Chari":       ((6.4, 36.8, 25.4, 23.2, 1.1, 7.1), 588_008),
    "Ouaddaï":           ((0.2, 0.4, 98.8, 0.4, 0.0, 0.2), 721_166),
    "Salamat":           ((0.2, 0.7, 98.6, 0.5, 0.0, 0.0), 302_301),
    "Tandjilé":          ((3.7, 46.1, 8.4, 38.2, 0.4, 3.2), 661_906),
    "Wadi Fira":         ((0.2, 0.2, 99.3, 0.2, 0.1, 0.1), 508_383),
    "N'Djaména":         ((0.4, 11.7, 70.7, 16.2, 0.4, 0.6), 951_418),
    "Barh El Gazal":     ((0.2, 0.3, 99.3, 0.1, 0.0, 0.1), 257_267),
    "Ennedi":            ((0.1, 0.3, 99.4, 0.2, 0.0, 0.0), 167_919),
    "Sila":              ((0.2, 0.4, 99.0, 0.3, 0.0, 0.1), 293_450),
    "Tibesti":           ((0.5, 1.0, 98.0, 0.5, 0.0, 0.0), 21_303),
}
T507_TCHAD = ((4.0, 18.5, 58.4, 16.1, 0.5, 2.4), 10_941_682)

# Tableau 5.06, (Masculin, Féminin, Ensemble) per religion, per milieu, same category order.
T506 = {
    "Urbain": [(18_893, 16_381, 35_274), (172_955, 183_392, 356_349),
               (816_549, 731_628, 1_548_174), (207_680, 214_473, 422_155),
               (5_051, 4_440, 9_491), (17_445, 11_079, 28_523)],
    "Rural": [(201_435, 202_152, 403_587), (782_954, 886_890, 1_669_847),
              (2_388_471, 2_455_326, 4_843_794), (639_928, 699_414, 1_339_344),
              (23_782, 23_386, 47_168), (127_931, 110_048, 237_978)],
    "Ensemble": [(220_314, 218_517, 438_831), (955_887, 1_070_260, 2_026_152),
                 (3_205_059, 3_186_986, 6_392_040), (847_615, 913_899, 1_761_516),
                 (28_832, 27_825, 56_657), (145_368, 121_119, 266_486)],
}
T506_TOTAL = {"Urbain": 2_399_965, "Rural": 8_541_717, "Ensemble": 10_941_682}

CENSUSED = 10_941_682            # Tableau 2.02, recensée
ESTIMATED = 98_191               # Tableau 2.02, estimée: in no religion table
TOTAL = 11_039_873               # Tableau 2.01/2.02
FOOTNOTE_PARTS = (91_011, 4_180)  # Sila rural, Zouar: the footnote's own figures
PAGE_T213 = 53                   # printed p54, Tableau 2.13, total population by région
T213 = {"Sila": 387_461, "Tibesti": 25_483}   # the two régions whose total != censused
ORDINARY, COLLECTIVE = 10_621_672, 320_010   # Tableau 2.04

PCT = re.compile(r"^\d{1,3},\d$")
INT = re.compile(r"^\d[\d ]*$")
SPACES = dict.fromkeys([0x00A0, 0x2007, 0x2008, 0x2009, 0x202F, 0x205F], " ")


def despace(s):
    return re.sub(r"\s+", " ", str(s).translate(SPACES)).strip()


def norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def fetch():
    import urllib.request

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(PDF) and os.path.getsize(PDF) > 5_000_000:
        print("already have", PDF)
        return
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}
    req = urllib.request.Request(WAYBACK, headers=ua)
    with urllib.request.urlopen(req, timeout=900) as r:
        body = r.read()
    # A 200 is not a download: the archive has served a redirect page under a PDF's name.
    # Not pinned, so the trailer rule applies (sources/fetch_checks.py::check_body).
    check_body(body, "pdf", where="the Wayback capture")
    with open(PDF + ".part", "wb") as fh:
        fh.write(body)
    os.replace(PDF + ".part", PDF)
    print(f"wrote {PDF} ({len(body):,} bytes)")


def _lines(doc, pno):
    return [ln for ln in (despace(x) for x in doc.load_page(pno).get_text().splitlines())
            if ln]


def _int(s):
    return int(s.replace(" ", ""))


def read_t507(doc):
    """A name line, six percentages, `100,0`, the région's population."""
    lines = _lines(doc, PAGE_T507)
    out = {}
    for i in range(len(lines) - 8):
        if PCT.match(lines[i]) or INT.match(lines[i]):
            continue
        cells = lines[i + 1:i + 7]
        if (all(PCT.match(c) for c in cells) and lines[i + 7] == "100,0"
                and INT.match(lines[i + 8])):
            out[lines[i]] = (tuple(float(c.replace(",", ".")) for c in cells),
                             _int(lines[i + 8]))
    return out


def read_t506(doc):
    """Three blocks (Urbain, Rural, Ensemble); in each, a religion's M, F and total counts."""
    lines = _lines(doc, PAGE_T506)
    iu = lines.index("Urbain")
    ir = lines.index("Rural", iu)
    ie = lines.index("Ensemble", ir)
    stop = next(i for i in range(ie, len(lines)) if lines[i].startswith("5.2.2"))
    out = {}
    for block, (a, b) in (("Urbain", (iu, ir)), ("Rural", (ir, ie)), ("Ensemble", (ie, stop))):
        seg = lines[a:b]
        rows = []
        for want in CATS + ["Total"]:
            j = next(k for k, ln in enumerate(seg) if norm(ln) == norm(want))
            nums = [_int(x) for x in seg[j + 1:j + 4] if INT.match(x)]
            rows.append(tuple(nums))
        out[block] = rows
    return out


def rake(seed, rows, cols, tol=1e-11, iters=10_000):
    """Iterative proportional fitting of a région x religion seed to both margins."""
    m = [list(r) for r in seed]
    for _ in range(iters):
        for j, target in enumerate(cols):
            s = sum(m[i][j] for i in range(len(m)))
            for i in range(len(m)):
                m[i][j] *= target / s
        worst = 0.0
        for i, target in enumerate(rows):
            s = sum(m[i])
            worst = max(worst, abs(s - target) / target)
            m[i] = [v * target / s for v in m[i]]
        if worst < tol:
            return m
    raise SystemExit("raking did not converge")


def integerise(m, rows):
    """Largest remainder within each région, so every région total is exact."""
    out = []
    for r, target in zip(m, rows):
        base = [int(v) for v in r]
        short = target - sum(base)
        order = sorted(range(len(r)), key=lambda j: -(r[j] - base[j]))
        for j in order[:short]:
            base[j] += 1
        out.append(base)
    return out


def check(doc):
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    print("Chad — RGPH2 2009, État et structures de la population, Tableaux 5.06 and 5.07\n")
    say(doc.page_count == PAGES, f"the volume is {doc.page_count} pages (expected {PAGES})")

    # 1. the parsed tables are the transcribed ones
    parsed = read_t507(doc)
    by_key = {norm(k): v for k, v in parsed.items()}
    bad = [k for k, v in T507.items() if by_key.get(norm(k)) != v]
    say(not bad and len(parsed) == 23,
        f"Tableau 5.07 parsed off the page: {len(parsed)} rows (22 régions + TCHAD), every "
        "région's six shares and population identical to the transcription")
    for k in bad:
        print(f"        {k}: page {by_key.get(norm(k))} transcribed {T507[k]}")
    say(by_key.get("tchad") == T507_TCHAD, f"its TCHAD row is {by_key.get('tchad')}")

    t506 = read_t506(doc)
    for block in ("Urbain", "Rural", "Ensemble"):
        want = T506[block] + [(None, None, T506_TOTAL[block])]
        got = t506[block]
        same = all(g == w for g, w in zip(got[:6], want[:6])) and got[6][2] == want[6][2]
        say(same, f"Tableau 5.06 {block} parsed off the page = the transcription")

    # 2. internal arithmetic
    worst = max(abs(sum(s) - 100.0) for s, _p in list(T507.values()) + [T507_TCHAD])
    say(worst <= 0.30 + 1e-9, f"every 5.07 row's six cells sum to 100 within {worst:.2f} pp "
        "(bound 0.30 = 6 cells x 0.05)")
    pops = {k: p for k, (_s, p) in T507.items()}
    say(sum(pops.values()) == CENSUSED,
        f"the 22 région populations sum to {sum(pops.values()):,} (censused {CENSUSED:,})")
    ens = [e for _m, _f, e in T506["Ensemble"]]
    say(sum(ens) == CENSUSED, f"5.06's six Ensemble counts sum to {sum(ens):,}")
    for block in ("Urbain", "Rural"):
        s = sum(e for _m, _f, e in T506[block])
        say(abs(s - T506_TOTAL[block]) <= 2,
            f"5.06 {block}: six counts sum to {s:,} against the printed {T506_TOTAL[block]:,} "
            f"({s - T506_TOTAL[block]:+d})")
    say(T506_TOTAL["Urbain"] + T506_TOTAL["Rural"] == CENSUSED, "urban + rural = censused")
    for i, c in enumerate(CATS):
        u, r, e = T506["Urbain"][i][2], T506["Rural"][i][2], T506["Ensemble"][i][2]
        mf = T506["Ensemble"][i][0] + T506["Ensemble"][i][1]
        # The blocks do not add to the person: separately rounded tabulations. Bounded, and
        # printed so the size is on the record.
        say(abs(u + r - e) <= 100 and abs(mf - e) <= 10,
            f"5.06 {c:<17} urban+rural {u + r - e:+4d}, men+women {mf - e:+3d} against "
            f"Ensemble {e:,}")

    # 3. the shares reproduce the national row and 5.06's counts
    for i, c in enumerate(CATS):
        w = sum(T507[k][0][i] * pops[k] for k in T507) / CENSUSED
        implied = sum(T507[k][0][i] / 100.0 * pops[k] for k in T507)
        rel = implied / ens[i] - 1
        good = abs(w - T507_TCHAD[0][i]) <= 0.10 and (ens[i] / CENSUSED < 0.01
                                                      or abs(rel) <= 0.005)
        say(good, f"{c:<17} population-weighted {w:6.2f} against TCHAD {T507_TCHAD[0][i]:.1f}; "
            f"implied {implied:>11,.0f} against 5.06 {ens[i]:>10,} ({100 * rel:+.2f}%)")
        say(abs(100.0 * ens[i] / CENSUSED - T507_TCHAD[0][i]) <= 0.05,
            f"    5.06's {c} is {100.0 * ens[i] / CENSUSED:.2f}% of the censused, printed "
            f"{T507_TCHAD[0][i]:.1f}")

    # 4. the population tables the universe rests on
    t202 = re.sub(r"(?<=\d) (?=\d)", "", despace(" ".join(_lines(doc, PAGE_T202))))
    for n in (CENSUSED, ESTIMATED, TOTAL) + FOOTNOTE_PARTS:
        say(str(n) in t202, f"{n:,} is printed on Tableau 2.02's page")
    say(CENSUSED + ESTIMATED == TOTAL, "censused + estimated = total")
    t213 = re.sub(r"(?<=\d) (?=\d)", "", despace(" ".join(_lines(doc, PAGE_T213))))
    others = [k for k in T507 if k not in T213]
    say(all(str(n) in t213 for n in T213.values())
        and all(str(T507[k][1]) in t213 for k in others),
        f"Tableau 2.13 prints Sila {T213['Sila']:,} and Tibesti {T213['Tibesti']:,} in the "
        f"total, and the other {len(others)} régions at their censused 5.07 populations")
    est = {k: T213[k] - T507[k][1] for k in T213}
    say(est["Tibesti"] == FOOTNOTE_PARTS[1] and sum(est.values()) == ESTIMATED,
        f"total minus censused: Sila {est['Sila']:,} + Tibesti {est['Tibesti']:,} = "
        f"{sum(est.values()):,}, the estimated; the footnote's Sila {FOOTNOTE_PARTS[0]:,} is a "
        "misprint")
    t204 = re.sub(r"(?<=\d) (?=\d)", "", despace(" ".join(_lines(doc, PAGE_T204))))
    say(str(ORDINARY) in t204 and str(COLLECTIVE) in t204 and ORDINARY + COLLECTIVE == CENSUSED,
        f"Tableau 2.04: ordinary {ORDINARY:,} + collective {COLLECTIVE:,} = censused")

    if not ok:
        raise SystemExit("reconciliation FAILED")
    return pops, ens


def check_fit(counts, pops, ens):
    """How far raking moved each printed share. Returns the worst move in points."""
    names = list(T507)
    worst, moved = 0.0, 0
    for i, k in enumerate(names):
        for j in range(len(CATS)):
            drawn = 100.0 * counts[i][j] / pops[k]
            d = drawn - T507[k][0][j]
            worst = max(worst, abs(d))
            if round(drawn + 1e-9, 1) != T507[k][0][j]:
                moved += 1
    col = [sum(counts[i][j] for i in range(len(names))) for j in range(len(CATS))]
    print(f"\n  raked: {moved} of {len(names) * len(CATS)} cells no longer round to the printed "
          f"share; the largest move is {worst:.3f} points")
    print("  religion totals after integerising, against 5.06: "
          + ", ".join(f"{c} {col[j] - ens[j]:+d}" for j, c in enumerate(CATS)))
    # One-decimal shares are +-0.05 each; a move far past that means a misread cell.
    if worst > 0.25:
        raise SystemExit(f"raking moved a share by {worst:.3f} points, more than rounding")
    return worst


def emit(pops, ens):
    names = list(T507)
    seed = [[T507[k][0][j] / 100.0 * pops[k] for j in range(len(CATS))] for k in names]
    raked = rake(seed, [pops[k] for k in names], ens)
    counts = integerise(raked, [pops[k] for k in names])
    check_fit(counts, pops, ens)

    rows = []
    for i, k in enumerate(names):
        for j, c in enumerate(CATS):
            n = counts[i][j]
            if n <= 0:
                continue
            rows.append({
                "geo_id": k, "geo_level": "region", "geo_name": k,
                "source_category": c, "count": n, "basis": BASIS, "year": YEAR,
                "source_id": SOURCE_ID,
                "note": (f"Tableau 5.07 pct={T507[k][0][j]:.1f}; pop={pops[k]}; raked to "
                         "Tableau 5.06's national counts and 5.07's région populations"),
            })
    return rows


def main():
    import fitz

    if "--fetch" in sys.argv:
        fetch()
    if not os.path.exists(PDF):
        raise SystemExit(f"{PDF} missing — run: python sources/td.py --fetch")
    with open(PDF, "rb") as fh:
        body = fh.read()
    check_body(body, "pdf", where=PDF)
    doc = fitz.open(PDF)
    pops, ens = check(doc)
    rows = emit(pops, ens)

    total = sum(r["count"] for r in rows)
    print(f"\n  22 régions, {total:,} people, {total / 22:,.0f} each")
    by_cat, by_reg = {}, {}
    for r in rows:
        by_cat[r["source_category"]] = by_cat.get(r["source_category"], 0) + r["count"]
        by_reg.setdefault(r["geo_name"], {})[r["source_category"]] = r["count"]
    for c, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"    {n:>11,}  {100.0 * n / total:6.2f}%  {c}")
    south = ["Logone Occidental", "Logone Oriental", "Mandoul", "Mayo Kebbi Est",
             "Mayo Kebbi Ouest", "Moyen Chari", "Tandjilé"]
    print("\n  the seven southern régions' share of each religion: " + ", ".join(
        f"{c} {100 * sum(by_reg[s].get(c, 0) for s in south) / by_cat[c]:.1f}%" for c in CATS))
    print("  N'Djaména's: " + ", ".join(
        f"{c} {100 * by_reg[chr(78) + chr(39) + 'Djaména'].get(c, 0) / by_cat[c]:.1f}%"
        for c in CATS))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".part", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(OUT + ".part", OUT)
    print("\nwrote", OUT, f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
