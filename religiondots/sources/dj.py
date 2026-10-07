"""Djibouti — RGPH-3 (2024), religion by région.

Reads (or fetches) two volumes of the 2024 census into data/raw/dj/rgph3/ and writes
data/normalized/dj.csv. `sources/dj.md` is the write-up; `sources/dj_geo.py` builds the six
régions and the grid.

## WHERE THE VOLUMES ARE

INSTAD (Institut National de la Statistique de Djibouti) published the RGPH-3's final report and
21 thematic volumes on 18 November 2025. `instad.dj` is a Nuxt app that renders nothing without
JavaScript (so earlier scouts saw an empty page, sources.md §11aq, §scout-2026-09-15-negatives);
its bundle's `baseURL` is a Heroku API, and `GET <API>/fichiers/RGPH/undefined` lists every volume
with a Firebase Storage link (`LISTING`, read 2026-10-03). No login.

## THE TABLE

*Tome 4: Caractéristiques socioculturelles de la population*, chapter 4, **Tableau n°42,
"Répartition (%) de la population résidente par région et milieu de résidence selon la pratique
religieuse"**, PDF p.130 (printed 129). Six régions, urban and rural, and a national row, each as
a count and a column share, for four answers:

    Islam  Christianisme  Sans religion  Autre religion  Ensemble

Its universe is the population of ordinary and nomadic households, 1,003,800 (Tome 4 Tableau 2:
P12_RELIGION, 1,003,800 expected, 1,003,800 valid, none missing). The 30,351 homeless and the
32,658 in collective households (barracks, boarding schools, hospitals; final report Tableau 7)
are in no religion table.

## THE QUESTION

Tome 4, p.124: the question was put to every resident of the household, with eight codes
(musulmane, catholique, protestante, orthodoxe, animiste, athée, sans religion, autres
religions); "compte tenu des réponses", only four were kept for the tables. Which code went into
which of the four is not printed; the natural reading is athée into `Sans religion` and animiste
into `Autre religion`. The questionnaire itself is not published with the volumes.

## THE OTHER TABLES IT IS CHECKED AGAINST

- Tome 4 Tableau 41 (p.126): the national counts, = Tableau 42's Ensemble row.
- Tome 4 Tableau 43 (p.132): religion by sex, age, wealth and nationality; its Ensemble row and
  its sex and nationality blocks close on the same column totals.
- Final report Tableau 14 (pp.53-54): population of ordinary and nomadic households by région
  and sub-unit; each région = Tableau 42's Ensemble column.
- Final report Tableau 7 (p.38): 1,003,800 + 30,351 homeless + 32,658 collective = 1,066,809.
- Final report Tableau 12 (p.46): de jure population by région, which gives each région's
  people outside ordinary households (the `gap`).

The volume's prose reads the column shares as row shares in several places (p.126: "La région
d'Ali-Sabieh se distingue ... par une plus grande présence de chrétiens (7,9%)", which is
Ali-Sabieh's share of the country's Christians; p.130: Ethiopians "40,6%" Christian, which is
Ethiopians' share of all Christians, 1,807 of 4,455). The counts are used, never the prose.

Usage:
    python sources/dj.py --fetch    two PDFs (8.3 MB) from INSTAD's Firebase storage
    python sources/dj.py            normalise from data/raw/dj/rgph3/
"""

import csv
import json
import os
import re
import sys
import urllib.parse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "dj", "rgph3")
OUT = os.path.join(ROOT, "data", "normalized", "dj.csv")
sys.path.insert(0, HERE)

from fetch_checks import FetchCheckError, check_body, digest   # noqa: E402  shared, not copied

SOURCE_ID = "dj_rgph3_2024_tome4_tableau42"
YEAR = 2024
BASIS = "self_id"
COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

LISTING = "https://instad-dj-6abc7b0eb612.herokuapp.com/fichiers/RGPH/undefined"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}

# file name in Firebase Storage (and on disk) -> (title in the listing, size, SHA-1 base32, pages)
TOME4 = "Tome 4_Caractéristiques socioculturelles de la population_18112025.pdf"
RAPPORT = "RGPH3_Rapport Résultats définitifs_18112025.pdf"
DOCS = {
    TOME4: ("Caractéristiques socioculturelles de la population", 4_444_117,
            "AARA7BBEO4JCRMESPTWAO7NIZGUEGNOM", 164),
    RAPPORT: ("Rapport des Résultats du RGPH3", 3_876_004,
              "LZQDU536ND3WSP5VE6LJWBL2K4FAUJNP", 74),
}

# 0-based page indices
T4_TABLEAU2 = 30     # PDF p.31, missing-value rates per variable
T4_TABLEAU41 = 125   # PDF p.126
T4_TABLEAU42 = 129   # PDF p.130
T4_TABLEAU43 = 131   # PDF p.132
R_TABLEAU7 = 37      # PDF p.38
R_TABLEAU12 = 45     # PDF p.46
R_TABLEAU14 = (52, 53)  # PDF pp.53-54

CATS = ["Islam", "Christianisme", "Sans religion", "Autre religion"]

# Tableau 42's row labels -> unit id (ASCII, the final report's spelling).
LABELS = {"Djibouti Ville": "Djibouti-Ville", "Ali-Sabieh": "Ali-Sabieh", "Dikhil": "Dikhil",
          "Tadjourah": "Tadjourah", "Obock": "Obock", "Arta": "Arta"}
REGIONS = list(LABELS.values())

# Transcribed from the rendered page; asserted equal to the text layer, cell by cell, with the
# printed column share beside each count exactly as printed ("1" for 1.0, "0" for none).
# (Islam, %, Christianisme, %, Sans religion, %, Autre religion, %, Ensemble, %)
T42 = {
    "Djibouti Ville": (723_127, "72,4", 3_860, "86,6", 841, "96,9", 182, "89,2", 728_010, "72,5"),
    "Ali-Sabieh":     (73_026, "7,3", 350, "7,9", 9, "1", 17, "8,3", 73_402, "7,3"),
    "Dikhil":         (63_454, "6,4", 74, "1,7", 1, "0,1", 0, "0", 63_529, "6,3"),
    "Tadjourah":      (57_230, "5,7", 94, "2,1", 10, "1,2", 4, "2", 57_338, "5,7"),
    "Obock":          (35_625, "3,6", 23, "0,5", 0, "0", 0, "0", 35_648, "3,6"),
    "Arta":           (45_811, "4,6", 54, "1,2", 7, "0,8", 1, "0,5", 45_873, "4,6"),
}
T42_MILIEU = {
    "Urbain": (837_596, "83,9", 4_058, "91,1", 860, "99,1", 185, "90,7", 842_699, "84"),
    "Rural":  (160_677, "16,1", 397, "8,9", 8, "0,9", 19, "9,3", 161_101, "16"),
}
T42_TOTAL = (998_273, "100", 4_455, "100", 868, "100", 204, "100", 1_003_800, "100")
TOTAL = 1_003_800

# Final report Tableau 7: the three sub-populations.
HOMELESS = 30_351
COLLECTIVE = 32_658
DE_JURE = 1_066_809
# Final report Tableau 12: de jure population by région (Ensemble column).
DE_JURE_REGION = {"Djibouti-Ville": 767_250, "Ali-Sabieh": 76_414, "Dikhil": 66_196,
                  "Tadjourah": 60_645, "Obock": 47_382, "Arta": 48_922}
# Final report Tableau 14: population of ordinary and nomadic households, région rows.
T14_REGION_LABELS = {"Djibouti-ville": "Djibouti-Ville", "Ali-Sabieh": "Ali-Sabieh",
                     "Dikhil": "Dikhil", "Tadjourah": "Tadjourah", "Obock": "Obock",
                     "Arta": "Arta"}

# Tome 4 Tableau 43, nationality block: (Islam, Christianisme, Sans religion, Autre, Ensemble).
T43_NATIONALITY = {
    "Djibouti": (923_107, 1_934, 416, 46, 925_503),
    "Érythrée": (1_534, 132, 1, 12, 1_679),
    "Éthiopie": (49_461, 1_807, 159, 54, 51_481),
    "Somalie": (12_496, 33, 8, 0, 12_537),
    "Yémen": (3_771, 10, 0, 0, 3_781),
    "Étranger": (1_796, 475, 282, 92, 2_645),
    "Pas de nationalité": (6_108, 64, 2, 0, 6_174),
}
T43_SEX = {"Masculin": (494_776, 1_941, 434, 101, 497_252),
           "Féminin": (503_497, 2_514, 434, 103, 506_548)}

SPACES = dict.fromkeys([0x00A0, 0x2007, 0x2008, 0x2009, 0x202F, 0x205F], " ")


def despace(s):
    return re.sub(r"\s+", " ", str(s).translate(SPACES)).strip()


def fr(n):
    """723127 -> '723 127', the volumes' thousands separator."""
    return f"{n:,}".replace(",", " ")


def _text(doc, pno):
    return despace(doc.load_page(pno).get_text())


def fetch():
    import urllib.request

    os.makedirs(RAW, exist_ok=True)
    need = [f for f, (_t, size, _d, _p) in DOCS.items()
            if not (os.path.exists(os.path.join(RAW, f))
                    and os.path.getsize(os.path.join(RAW, f)) == size)]
    if not need:
        print("already have both volumes")
        return
    req = urllib.request.Request(LISTING, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        items = json.load(r)
    with open(os.path.join(RAW, "_listing.json.part"), "w", encoding="utf-8") as fh:
        json.dump(items, fh, ensure_ascii=False, indent=1)
    os.replace(os.path.join(RAW, "_listing.json.part"), os.path.join(RAW, "_listing.json"))
    by_file = {}
    for it in items:
        name = urllib.parse.unquote(it["imgUrl"].split("/o/")[1].split("?")[0]).split("/")[-1]
        by_file[name] = it
    for f in need:
        title, size, dig, _pages = DOCS[f]
        it = by_file.get(f)
        if it is None or it.get("title") != title:
            raise SystemExit(f"INSTAD's listing has no {f!r} titled {title!r}; "
                             f"titles now: {sorted(i.get('title') for i in items)}")
        with urllib.request.urlopen(urllib.request.Request(it["imgUrl"], headers=UA),
                                    timeout=600) as r:
            body = r.read()
        try:
            check_body(body, "pdf", where=f, pin_size=size, pin_digest=dig)
        except FetchCheckError as e:
            raise SystemExit(str(e))
        dst = os.path.join(RAW, f)
        with open(dst + ".part", "wb") as fh:
            fh.write(body)
        os.replace(dst + ".part", dst)
        print(f"wrote {dst} ({len(body):,} bytes)")


def _row_string(label, cells):
    return label + " " + " ".join(fr(c) if isinstance(c, int) else c for c in cells)


def check(t4, rp):
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    print("Djibouti — RGPH-3 (2024), Tome 4 Tableau 42\n")
    for f, doc in ((TOME4, t4), (RAPPORT, rp)):
        _title, size, dig, pages = DOCS[f]
        with open(os.path.join(RAW, f), "rb") as fh:
            body = fh.read()
        say(len(body) == size and digest(body) == dig and doc.page_count == pages,
            f"{f[:40]}...: {len(body):,} bytes, digest pinned, {doc.page_count} pages")

    # 1. Tableau 42 on the page = the transcription, row by row, in order
    t = _text(t4, T4_TABLEAU42)
    cap = ("Tableau n°42. Répartition (%) de la population résidente par région et milieu de "
           "résidence selon la pratique religieuse")
    i, k = t.find(cap), t.find("Source : INSTAD (2024) RGPH-3", t.find(cap))
    say(i >= 0 and k > i, "Tableau 42's caption and its source line are on PDF p.130")
    body = t[i + len(cap):k]
    # (the page carries a running-header image, so an image count is no test here; the exact
    # text match below is the test that the numbers are text)
    want = ("Pratique religieuse Islam Christianisme Sans religion Autre religion Ensemble "
            "Effectif % Effectif % Effectif % Effectif % Effectif % Région "
            + " ".join(_row_string(lb, r) for lb, r in T42.items())
            + " Milieu de résidence "
            + " ".join(_row_string(lb, r) for lb, r in T42_MILIEU.items())
            + " " + _row_string("Ensemble", T42_TOTAL))
    say(body.strip() == want, "Tableau 42's text layer is exactly the transcription "
        "(header, 6 régions, urban, rural, Ensemble)")

    # 2. it closes, every way
    cnt = {lb: r[0::2] for lb, r in T42.items()}
    tot = T42_TOTAL[0::2]
    say(all(sum(r[:4]) == r[4] for r in list(cnt.values()) + [tot]
            + [m[0::2] for m in T42_MILIEU.values()]),
        "each row's four answers sum to its Ensemble")
    say(all(sum(r[c] for r in cnt.values()) == tot[c] for c in range(5)),
        "the six régions sum to the Ensemble row in all five columns")
    say(all(T42_MILIEU["Urbain"][2 * c] + T42_MILIEU["Rural"][2 * c] == tot[c] for c in range(5)),
        "urban + rural = the Ensemble row in all five columns")
    say(tot[4] == TOTAL, f"the universe is {TOTAL:,}")

    # 3. the printed shares are column shares (a share of the country's people of that answer)
    bad = []
    for lb, r in list(T42.items()) + list(T42_MILIEU.items()):
        for c in range(5):
            n, printed = r[2 * c], r[2 * c + 1]
            v = 100.0 * n / tot[c]
            if abs(v - float(printed.replace(",", "."))) > 0.051:
                bad.append((lb, CATS[c] if c < 4 else "Ensemble", printed, round(v, 2)))
    say(not bad, f"every printed % is the column share of its count, to the printed decimal {bad}")

    # 4. Tableau 41, the national counts
    t41 = _text(t4, T4_TABLEAU41)
    say("Groupe religieuse Effectif % Islam 998 273 99,4 Christianisme 4 455 0,5 Sans religion "
        "868 0,1 Autre religion5 204 - Ensemble 1 003 800 100" in t41,
        "Tableau 41 (p.126) prints the same national counts")

    # 5. Tableau 2: the religion variable's universe, no missing values
    say("P12_RELIGION Religion 1 003 800 1 003 800 0 0" in _text(t4, T4_TABLEAU2),
        "Tableau 2 (p.31): P12_RELIGION 1,003,800 expected, 1,003,800 valid, 0 missing")

    # 6. Tableau 43: sex and nationality blocks close on the same totals
    t43 = _text(t4, T4_TABLEAU43)
    for block in (T43_SEX, T43_NATIONALITY):
        sums = tuple(sum(r[c] for r in block.values()) for c in range(5))
        say(sums == tot and all(sum(r[:4]) == r[4] for r in block.values()),
            f"Tableau 43's {'sex' if block is T43_SEX else 'nationality'} block sums to Tableau "
            f"42's totals, each row closing")
    found =all(fr(r[0]) in t43 and fr(r[4]) in t43 and lb in t43
                for lb, r in T43_NATIONALITY.items())
    say(found and _row_string("Ensemble", T42_TOTAL) in t43,
        "Tableau 43 (p.132) prints each nationality's counts and the same Ensemble row")

    # 7. final report Tableau 14: région populations of ordinary and nomadic households
    t14 = _text(rp, R_TABLEAU14[0]) + " " + _text(rp, R_TABLEAU14[1])
    # Space-separated thousands make "728 010 118 778 6,1" ambiguous as text, so each row is
    # matched with Tableau 42's figure in place and the household count and size after it.
    want14 = {LABELS[lb]: r[8] for lb, r in T42.items()}
    hh = {}
    for lb, u in T14_REGION_LABELS.items():
        m = re.search(re.escape(lb) + " " + fr(want14[u]) + r" (\d{1,3}(?: \d{3})?) (\d,\d) ", t14)
        if m:
            hh[u] = (int(m.group(1).replace(" ", "")), float(m.group(2).replace(",", ".")))
    say(len(hh) == 6 and all(abs(want14[u] / h - s) <= 0.051 for u, (h, s) in hh.items())
        and sum(h for h, _s in hh.values()) == 172_097,
        f"final report Tableau 14 (pp.53-54): each région's ordinary and nomadic household "
        f"population = Tableau 42's Ensemble, its households sum to 172,097 and population over "
        f"households is the printed mean size ({hh})")
    say("Ensemble 1 003 800 172 097 5,8" in t14, "Tableau 14's Ensemble is 1,003,800 in 172,097 households")

    # 8. final report Tableau 7: the three sub-populations
    t7 = _text(rp, R_TABLEAU7)
    say(f"Ménages ordinaires/nomades 497 252 506 548 {fr(TOTAL)}" in t7
        and f"Sans-abris 16 059 14 292 {fr(HOMELESS)}" in t7
        and f"Ménages collectifs 17 051 15 607 {fr(COLLECTIVE)}" in t7
        and f"Ensemble 530 362 536 447 {fr(DE_JURE)}" in t7
        and TOTAL + HOMELESS + COLLECTIVE == DE_JURE,
        f"Tableau 7 (p.38): {TOTAL:,} + {HOMELESS:,} homeless + {COLLECTIVE:,} collective = "
        f"{DE_JURE:,}")
    say(T43_SEX["Masculin"][4] == 497_252 and T43_SEX["Féminin"][4] == 506_548,
        "Tableau 43's sexes are Tableau 7's ordinary-household sexes")

    # 9. final report Tableau 12: de jure by région, so the people outside the table per région
    t12 = _text(rp, R_TABLEAU12)
    say(all(re.search(re.escape(u) + r" \d{1,3}(?: \d{3})* \d{1,3}(?: \d{3})* " + fr(n) + " ", t12)
            for u, n in DE_JURE_REGION.items())
        and sum(DE_JURE_REGION.values()) == DE_JURE,
        "Tableau 12 (p.46): de jure population by région, summing to 1,066,809")
    outside = {u: DE_JURE_REGION[u] - want14[u] for u in REGIONS}
    say(all(v > 0 for v in outside.values()) and sum(outside.values()) == HOMELESS + COLLECTIVE,
        f"de jure minus the table, per région, is positive and sums to the homeless and "
        f"collective {HOMELESS + COLLECTIVE:,}")

    if not ok:
        raise SystemExit("reconciliation FAILED")
    return cnt, outside


def emit(cnt):
    out = []
    for lb, u in LABELS.items():
        for c, n in zip(CATS, cnt[lb][:4]):
            if n <= 0:
                continue
            out.append({
                "geo_id": u, "geo_level": "region", "geo_name": u,
                "source_category": c, "count": n, "basis": BASIS, "year": YEAR,
                "source_id": SOURCE_ID, "note": "Tome 4 Tableau 42, PDF p.130",
            })
    return out


def main():
    import fitz

    if "--fetch" in sys.argv:
        fetch()
    for f in DOCS:
        if not os.path.exists(os.path.join(RAW, f)):
            raise SystemExit(f"{f} missing — run: python sources/dj.py --fetch")
    t4 = fitz.open(os.path.join(RAW, TOME4))
    rp = fitz.open(os.path.join(RAW, RAPPORT))
    cnt, outside = check(t4, rp)
    out = emit(cnt)

    total = sum(r["count"] for r in out)
    print(f"\n  6 régions, {total:,} people in ordinary and nomadic households")
    for c, n in zip(CATS, T42_TOTAL[0:8:2]):
        print(f"    {n:>9,}  {100.0 * n / total:7.3f}%  {c}")
    print(f"\n  {'région':<16}{'people':>9}{'Christian':>10}{'share':>8}{'none':>6}{'other':>6}"
          f"{'outside':>9}{'de jure':>9}{'out %':>7}")
    for lb, u in LABELS.items():
        r = cnt[lb]
        print(f"  {u:<16}{r[4]:>9,}{r[1]:>10,}{100.0 * r[1] / r[4]:7.2f}%{r[2]:>6}{r[3]:>6}"
              f"{outside[u]:>9,}{DE_JURE_REGION[u]:>9,}{100.0 * outside[u] / DE_JURE_REGION[u]:6.1f}%")
    print(f"\n  outside every religion table: {HOMELESS + COLLECTIVE:,} of {DE_JURE:,} "
          f"({(HOMELESS + COLLECTIVE) / DE_JURE:.5f})")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".part", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out)
    os.replace(OUT + ".part", OUT)
    print("\nwrote", OUT, f"({len(out)} rows)")


if __name__ == "__main__":
    main()
