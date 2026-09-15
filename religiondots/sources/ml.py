"""Mali — RGPH5 (2022), religion by région, residents of ordinary households.

Reads (or fetches) three INSTAT files into data/raw/ml/ and writes data/normalized/ml.csv.
`sources/ml.md` is the write-up; `sources/ml_geo.py` builds the polygons and the grid.

## TWO VOLUMES PRINT THE RÉGION TABLE, AND EACH HAS WHAT THE OTHER LACKS

    Tableau 6.13  État et structure de la population, PDF p132      20 régions x 7 answers,
                  one decimal, with each région's population            <- the Christian split
    Tableau 2.03  Caractéristiques culturelles de la population, PDF p31
                  the same 20 régions and populations, TWO decimals,
                  Christians as one column                              <- the five groups
    Tableau 2.9   État et structure, PDF p64: ordinary-household
                  population by région and sex, counts                  <- the denominators
    Annex A01     Caractéristiques culturelles, PDF p52: national
                  counts for all seven codes AND `Non Déclaré` 48,746   <- the national margins
    Tableau 2.01  Caractéristiques culturelles, PDF p30: national
                  counts for the five groups, non-response spread in

**Tableau 6.13's rows are forced to 100.0.** All 21 sum to exactly 100.0, and where a cell
disagrees with 2.03 by more than rounding it is `Sans religion` or `Autre religion`, the cells
last in the row (Koulikoro's `Autre religion` is 0.5 in 6.13 and 0.36 in 2.03; Dioïla's `Sans
religion` 0.5 against 0.39). So 2.03's two decimals are used for the five groups and 6.13 only
for how each région's Christians divide.

**Every percentage table has the non-response prorated in.** A01 counts 48,746 `Non Déclaré`
(0.23%); Tableau 2.01's five counts are A01's with that row spread over the answers in
proportion, to within two people in every column, and 2.03's national row is 2.01's shares.
The dots therefore include it, as the office published them. Spec §12: record a proration,
never undo it.

**The construction.** Stage 1: 2.03's shares on 2.9's populations, raked to 2.9's région totals
and to A01's national counts with the non-response prorated (which are 2.01's). Stage 2: each
région's Christians split in 6.13's Catholic : Protestant : other-Christian proportions, raked
to A01's three Christian counts, prorated the same way. Both rakes move the large categories by
a rounding correction only; `check()` bounds it.

## THE UNIVERSE: ORDINARY HOUSEHOLDS, 95.3% OF THE COUNTED POPULATION

Tableau 2.1 (PDF p54): 21,347,587 residents of ordinary households, 105,416 in collective
households, 1,151 homeless, and **941,335 in areas the enumeration did not reach for insecurity**,
estimated with GRID3 from building footprints (PDF p44-45, Tableau 1.2), making the population de
droit 22,395,489. The religion tables cover the first group only.

The counts were adjusted for omission measured by the post-enumeration survey, by region
coefficients of 1.029 to 1.068, and 1.068 everywhere on paper forms (Tableau 1.4, PDF p47).
Recorded, not undone.

**Tableau 1.2 and Tableaux 2.3 minus 2.9 disagree about Douentza and Bandiagara.** 2.3 minus 2.9
leaves 22,960 in Douentza and 148,637 in Bandiagara outside ordinary households; 1.2 puts the
unenumerated at 154,979 and 986. The two régions' sums agree to within a plausible
collective-household count (15,632). Every other région's residual is non-negative. Printed, not
resolved; it does not touch a drawn count.

## THE FORM

P10 of the ordinary-household questionnaire (`IMP_RGPH5_Questionnaire_Menag_Ordinaire_VF_SMAP_20MARS
2021.pdf`, NADA catalog 95, the paper version): *"Quelle est la religion de [NOM] ?"* with 1
Musulman, 2 Catholique, 3 Protestant, 4 Autre religion chrétienne, 5 Animiste, 6 Sans religion. The
paper form prints no seventh code; Tableau 1.01 of the cultural volume (PDF p26) lists 7 Autre
religion, and the tables carry it. There is no code for no answer, and A01 counts 48,746 anyway.
Animism has its own code beside `Sans religion`, so the no-religion box is the `separate` case.

Usage:
    python sources/ml.py --fetch    two INSTAT reports (38 MB) and the form (0.4 MB)
    python sources/ml.py            normalise from data/raw/ml/
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
RAW = os.path.join(ROOT, "data", "raw", "ml")
OUT = os.path.join(ROOT, "data", "normalized", "ml.csv")
sys.path.insert(0, os.path.join(ROOT, "tools"))

from fetch_checks import FetchCheckError, check_body, check_pdf_doc, digest  # noqa: E402

SOURCE_ID = "ml_rgph5_2022"
YEAR = 2022
BASIS = "self_id"
COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

INSTAT = "https://www.instat-mali.org/laravel-filemanager/files/shares/rgph/"
NADA = "https://microdata.instat.ml/index.php/catalog/95/download/"
FILES = {
    "es": dict(name="rapport-etat-structure-population-rgph5-rgph.pdf",
               urls=[INSTAT + "rapport-etat-structure-population-rgph5-rgph.pdf", NADA + "708"],
               size=20_745_926, digest="THF6DE3WDJTEI65N4YF6PGW5VCDHIPOB", pages=168,
               empty=[1, 2, 3, 165, 166, 167]),
    "cc": dict(name="rapport-caracteristiques-culturelles-population-rgph5_rpgh.pdf",
               urls=[INSTAT + "rapport-caracteristiques-culturelles-population-rgph5_rpgh.pdf",
                     NADA + "722"],
               size=17_708_018, digest="2NM3KCJYQT2WAF3ZRZWGCBSPH4ASM6S7", pages=70,
               empty=[1, 2, 3, 68, 69]),
    "qf": dict(name="IMP_RGPH5_Questionnaire_Menag_Ordinaire_VF_SMAP_20MARS2021.pdf",
               urls=[NADA + "585"],
               size=364_040, digest="5ZYSUAJILMQTYGI2HLTRSL2X4IOEE5WS", pages=8),
}

# 0-based page indices.
ES_T11, ES_T12, ES_T21, ES_T23, ES_T29, ES_T612, ES_T613 = 35, 44, 53, 55, 63, 130, 131
CC_T101, CC_T201, CC_T203, CC_A01 = 25, 29, 30, 51
QF_P10 = 1

CATS = ["Musulman", "Catholique", "Protestant", "Autre religion chrétienne", "Animiste",
        "Sans religion", "Autre religion"]
FIVE = ["Musulman", "Chrétien", "Animiste", "Sans religion", "Autre religion"]
CHRISTIAN = CATS[1:4]

REGIONS = ["Kayes", "Koulikoro", "Sikasso", "Ségou", "Mopti", "Tombouctou", "Gao", "Kidal",
           "Taoudenni", "Ménaka", "Nioro", "Kita", "Dioïla", "Nara", "Bougouni", "Koutiala",
           "San", "Douentza", "Bandiagara", "Bamako"]

# Tableau 6.13, CATS order, then the région's population. Transcribed and asserted equal to the
# page.
T613 = {
    "Kayes":      (98.8, 0.5, 0.2, 0.0, 0.0, 0.4, 0.1, 1826564),
    "Koulikoro":  (95.3, 1.4, 0.6, 0.1, 0.7, 1.4, 0.5, 2246154),
    "Sikasso":    (96.3, 1.0, 0.4, 0.1, 0.9, 1.0, 0.3, 1528398),
    "Ségou":      (98.5, 0.8, 0.5, 0.0, 0.0, 0.1, 0.1, 2208847),
    "Mopti":      (99.1, 0.5, 0.3, 0.0, 0.0, 0.1, 0.0, 842209),
    "Tombouctou": (99.7, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 693719),
    "Gao":        (99.5, 0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 679911),
    "Kidal":      (99.7, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 79324),
    "Taoudenni":  (99.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 99499),
    "Ménaka":     (99.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 225223),
    "Nioro":      (99.7, 0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 668966),
    "Kita":       (98.8, 0.6, 0.2, 0.1, 0.0, 0.3, 0.0, 680380),
    "Dioïla":     (98.5, 0.8, 0.2, 0.0, 0.0, 0.5, 0.0, 674419),
    "Nara":       (99.6, 0.1, 0.1, 0.0, 0.2, 0.0, 0.0, 278904),
    "Bougouni":   (98.9, 0.6, 0.3, 0.0, 0.0, 0.2, 0.0, 1567533),
    "Koutiala":   (91.6, 1.8, 2.4, 0.1, 3.1, 0.7, 0.3, 1153375),
    "San":        (70.1, 9.6, 7.2, 0.2, 8.3, 3.7, 0.9, 815185),
    "Douentza":   (99.6, 0.1, 0.2, 0.0, 0.0, 0.1, 0.0, 147229),
    "Bandiagara": (94.0, 4.1, 1.9, 0.0, 0.0, 0.0, 0.0, 720279),
    "Bamako":     (97.6, 1.5, 0.7, 0.1, 0.0, 0.1, 0.0, 4211468),
    "Ensemble":   (96.4, 1.4, 0.8, 0.1, 0.7, 0.5, 0.1, 21347587),
}

# Tableau 2.03, FIVE order, then the population. The cultural volume prints the régions in
# another order (grouped under their pre-2023 parents); a dict does not care.
T203 = {
    "Kayes":      (98.75, 0.70, 0.03, 0.42, 0.10, 1826564),
    "Kita":       (98.80, 0.86, 0.04, 0.27, 0.03, 680380),
    "Nioro":      (99.70, 0.11, 0.01, 0.14, 0.04, 668966),
    "Koulikoro":  (95.32, 2.14, 0.74, 1.44, 0.36, 2246154),
    "Dioïla":     (98.48, 1.08, 0.03, 0.39, 0.02, 674419),
    "Nara":       (99.65, 0.19, 0.15, 0.01, 0.00, 278904),
    "Sikasso":    (96.30, 1.50, 0.93, 1.04, 0.23, 1528398),
    "Bougouni":   (98.89, 0.93, 0.04, 0.11, 0.03, 1567533),
    "Koutiala":   (91.57, 4.32, 3.14, 0.62, 0.35, 1153375),
    "Ségou":      (98.52, 1.29, 0.04, 0.13, 0.02, 2208847),
    "San":        (70.10, 16.99, 8.33, 3.65, 0.93, 815185),
    "Mopti":      (99.10, 0.84, 0.01, 0.04, 0.01, 842209),
    "Bandiagara": (93.96, 5.99, 0.04, 0.00, 0.01, 720279),
    "Douentza":   (99.64, 0.34, 0.01, 0.00, 0.01, 147229),
    "Tombouctou": (99.67, 0.31, 0.00, 0.02, 0.00, 693719),
    "Gao":        (99.51, 0.47, 0.01, 0.00, 0.01, 679911),
    "Kidal":      (99.64, 0.29, 0.01, 0.02, 0.04, 79324),
    "Taoudenni":  (99.81, 0.18, 0.00, 0.00, 0.01, 99499),
    "Ménaka":     (99.67, 0.30, 0.02, 0.00, 0.01, 225223),
    "Bamako":     (97.58, 2.32, 0.01, 0.06, 0.03, 4211468),
    "Ensemble":   (96.45, 2.27, 0.65, 0.50, 0.13, 21347587),
}

# Tableau 2.9: (masculin, féminin, ensemble), ordinary-household residents.
T29 = {
    "Kayes": (909052, 917512, 1826564), "Koulikoro": (1135295, 1110859, 2246154),
    "Sikasso": (756575, 771823, 1528398), "Ségou": (1101581, 1107266, 2208847),
    "Mopti": (430450, 411759, 842209), "Tombouctou": (363823, 329896, 693719),
    "Gao": (349928, 329983, 679911), "Kidal": (43253, 36071, 79324),
    "Taoudenni": (54854, 44645, 99499), "Ménaka": (123285, 101938, 225223),
    "Nioro": (325287, 343679, 668966), "Kita": (339768, 340612, 680380),
    "Dioïla": (337000, 337419, 674419), "Nara": (139615, 139289, 278904),
    "Bougouni": (791810, 775723, 1567533), "Koutiala": (563184, 590191, 1153375),
    "San": (413389, 401796, 815185), "Douentza": (74979, 72250, 147229),
    "Bandiagara": (357410, 362869, 720279), "Bamako": (2082278, 2129190, 4211468),
    "Ensemble": (10692816, 10654771, 21347587),
}

# Tableau 2.3: (masculin, féminin, ensemble), every resident (population de droit).
T23 = {
    "Kayes": (921044, 919285, 1840329), "Koulikoro": (1142015, 1113142, 2255157),
    "Sikasso": (760239, 772884, 1533123), "Ségou": (1231931, 1223332, 2455263),
    "Mopti": (483089, 452490, 935579), "Tombouctou": (504864, 469414, 974278),
    "Gao": (373996, 353521, 727517), "Kidal": (45229, 37963, 83192),
    "Taoudenni": (55284, 45074, 100358), "Ménaka": (170205, 148671, 318876),
    "Nioro": (333731, 344330, 678061), "Kita": (340853, 340818, 681671),
    "Dioïla": (338396, 337569, 675965), "Nara": (154080, 153697, 307777),
    "Bougouni": (794189, 776790, 1570979), "Koutiala": (573069, 596813, 1169882),
    "San": (418353, 402454, 820807), "Douentza": (87186, 83003, 170189),
    "Bandiagara": (433502, 435414, 868916), "Bamako": (2095300, 2132269, 4227569),
    "Ensemble": (11256555, 11138934, 22395489),
}

# Tableau 1.2: estimated population of the areas not enumerated, (urbain, rural, ensemble); a
# printed `-` is None.
T12 = {
    "Koulikoro": (None, 1786, 1786), "Nara": (None, 28873, 28873),
    "Bougouni": (None, 1078, 1078), "Koutiala": (None, 12508, 12508),
    "Ségou": (None, 228421, 228421), "Mopti": (2151, 87969, 90120),
    "Douentza": (620, 154359, 154979), "Bandiagara": (None, 986, 986),
    "Tombouctou": (556, 279836, 280392), "Gao": (None, 44795, 44795),
    "Kidal": (2048, 1696, 3744), "Ménaka": (None, 93653, 93653),
    "Ensemble": (5375, 935960, 941335),
}

# Tableau 2.1, counts by kind of population.
T21 = {"ordinary": 21_347_587, "collective": 105_416, "homeless": 1_151,
       "not_enumerated": 941_335, "total": 22_395_489}

# Tableau 1.1: cercles per région under the law of 13 March 2023 (Bamako prints `-`). Read by
# sources/ml_geo.py as a witness for the polygons.
T11 = {
    "Kayes": 10, "Koulikoro": 8, "Sikasso": 8, "Ségou": 11, "Mopti": 8, "Tombouctou": 13,
    "Gao": 16, "Kidal": 9, "Taoudenni": 6, "Ménaka": 6, "Nioro": 6, "Kita": 6, "Dioïla": 6,
    "Nara": 6, "Bougouni": 10, "Koutiala": 8, "San": 7, "Douentza": 6, "Bandiagara": 9,
    "Bamako": None, "Total": 159,
}

# Tableau 2.01: (masculin, féminin, ensemble) in FIVE order, then Ensemble.
T201 = {
    "Musulman": (10308680, 10280239, 20588918), "Chrétien": (243604, 241251, 484857),
    "Animiste": (72254, 66864, 139118), "Sans religion": (53836, 52391, 106227),
    "Autre religion": (14443, 14025, 28468), "Ensemble": (10692816, 10654771, 21347587),
}
T201_LABEL = {"Sans religion": "Sans", "Autre religion": "Autre"}   # split over two lines

# Annex A01: (masculin, féminin, ensemble), CATS order, then Non Déclaré and Total.
A01 = {
    "Musulman": (10283688, 10258217, 20541904), "Catholique": (146354, 144597, 290952),
    "Protestant": (87949, 87741, 175691), "Autre religion chrétienne": (8710, 8397, 17107),
    "Animiste": (72079, 66721, 138800), "Sans religion": (53705, 52279, 105984),
    "Autre religion": (14408, 13995, 28403), "Non Déclaré": (25923, 22824, 48746),
    "Total": (10692816, 10654771, 21347587),
}

NUM = re.compile(r"^\d{1,3}(?: \d{3})+(?:,\d+)?$|^\d+(?:,\d+)?$")


def norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def path(key):
    return os.path.join(RAW, FILES[key]["name"])


def fetch():
    import urllib.request

    os.makedirs(RAW, exist_ok=True)
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}
    for key, f in FILES.items():
        dst = path(key)
        if os.path.exists(dst) and os.path.getsize(dst) == f["size"]:
            print("already have", dst)
            continue
        for url in f["urls"]:
            try:
                req = urllib.request.Request(url, headers=ua)
                with urllib.request.urlopen(req, timeout=900) as r:
                    body = r.read()
                print("  " + check_body(body, "pdf", where=url[:80], pin_digest=f["digest"]))
            except (FetchCheckError, OSError) as e:
                print(f"  {url[:80]}: {e}")
                continue
            with open(dst + ".part", "wb") as fh:
                fh.write(body)
            os.replace(dst + ".part", dst)
            print(f"wrote {dst} ({len(body):,} bytes)")
            break
        else:
            raise SystemExit(f"no URL returned the pinned {f['name']}")


def _lines(doc, pno):
    return [unicodedata.normalize("NFC", " ".join(x.split()))
            for x in doc.load_page(pno).get_text().splitlines()]


def _num(s):
    if s == "-":
        return None
    t = s.replace(" ", "")
    return float(t.replace(",", ".")) if "," in t else int(t)


def rows_after(lines, caption, labels, k, stop=None):
    """{label: its k cells}. Labels are found in order, each on its own line, starting at the
    line that begins with `caption`; a row's cells are the next k numeric (or `-`) lines, and a
    text line after the first cell means the row is broken."""
    i = next((n for n, ln in enumerate(lines) if ln.startswith(caption)), None)
    if i is None:
        raise SystemExit(f"no {caption!r} on the page")
    end = len(lines)
    if stop:
        end = next((n for n in range(i + 1, len(lines)) if lines[n].startswith(stop)), end)
    out = {}
    for lab in labels:
        j = next((n for n in range(i + 1, end) if lines[n] == lab), None)
        if j is None:
            raise SystemExit(f"{caption}: no row {lab!r}")
        cells, n = [], j + 1
        while len(cells) < k and n < end:
            if NUM.match(lines[n]) or lines[n] == "-":
                cells.append(_num(lines[n]))
            elif cells:
                raise SystemExit(f"{caption}: row {lab!r} broken by {lines[n]!r}")
            n += 1
        if len(cells) < k:
            raise SystemExit(f"{caption}: row {lab!r} has {len(cells)} cells, expected {k}")
        out[lab] = tuple(cells)
        i = n - 1
    return out


def check(docs):
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    print("Mali — RGPH5 2022, Tableaux 6.13 and 2.03\n")
    for key, f in FILES.items():
        with open(path(key), "rb") as fh:
            body = fh.read()
        say(digest(body) == f["digest"], f"{f['name'][:48]}: digest {digest(body)}")
        for good, msg in check_pdf_doc(docs[key], f["pages"]):
            # The reports' blank inside covers and back pages have no text. The digest is pinned,
            # so a truncated copy cannot pass; the pin here is on WHICH pages are empty.
            if "text layer" in msg:
                good = msg.endswith(f"{f.get('empty', [])})")
                msg += " (expected: the blank cover and back pages)" if f.get("empty") else ""
            say(good, "  " + msg)

    es, cc, qf = docs["es"], docs["cc"], docs["qf"]
    rows = REGIONS + ["Ensemble"]

    # 1. every transcription equals its page
    def same(name, parsed, want):
        bad = [k for k in want if parsed[k] != want[k]]
        say(not bad, f"{name}: all {len(want)} rows parsed off the page equal the transcription"
            + (f"; differ: {[(k, parsed[k], want[k]) for k in bad]}" if bad else ""))

    same("Tableau 6.13", rows_after(_lines(es, ES_T613), "Tableau 6.13", rows, 9),
         {r: v[:7] + (100.0, v[7]) for r, v in T613.items()})
    p203 = rows_after(_lines(cc, CC_T203), "Tableau 2.03", list(T203), 7)
    same("Tableau 2.03", {r: v[:5] + (v[6],) for r, v in p203.items()}, T203)
    say(all(v[5] == 100.0 for v in p203.values()), "Tableau 2.03: every row's printed % is 100,00")
    same("Tableau 2.9", {r: v[:3] for r, v in rows_after(
        _lines(es, ES_T29), "Tableau 2.9", list(T29), 6).items()}, T29)
    same("Tableau 2.3", {r: v[:3] for r, v in rows_after(
        _lines(es, ES_T23), "Tableau 2.3", list(T23), 6).items()}, T23)
    same("Tableau 1.2", rows_after(_lines(es, ES_T12), "Tableau 1.2", list(T12), 3), T12)
    p11 = rows_after(_lines(es, ES_T11), "Tableau 1.1", list(T11), 5)
    same("Tableau 1.1 (cercles)", {r: v[0] for r, v in p11.items()}, T11)
    p201 = rows_after(_lines(cc, CC_T201), "Tableau 2.01",
                      [T201_LABEL.get(k, k) for k in T201], 6)
    same("Tableau 2.01", {k: p201[T201_LABEL.get(k, k)][0::2] for k in T201}, T201)
    same("Annex A01", rows_after(_lines(cc, CC_A01), "A01", list(A01), 3, stop="A02"), A01)
    p612 = rows_after(_lines(es, ES_T612), "Tableau 6.12",
                      ["Musulmane", "Catholique", "Protestant", "Autres religion", "Animiste",
                       "Sans religion", "Autre religion"], 9)
    say(tuple(v[8] for v in p612.values()) == T613["Ensemble"][:7],
        "Tableau 6.12's Ensemble column equals 6.13's Ensemble row")

    # 2. rows close
    say(all(round(sum(v[:7]), 1) == 100.0 for v in T613.values()),
        "Tableau 6.13: all 21 rows sum to exactly 100.0 (forced; see the docstring)")
    worst = max(abs(sum(v[:5]) - 100) for v in T203.values())
    say(worst <= 0.025 + 1e-9, f"Tableau 2.03: every row sums to 100 within {worst:.3f}")

    # 3. one population per région, three times printed
    say(all(T613[r][7] == T203[r][5] == T29[r][2] for r in rows),
        "6.13, 2.03 and 2.9 print the same 20 région populations and the same national total")
    # Weighted counts (Tableau 1.4), rounded per région: both tables' régions come to one person
    # under their printed national row.
    s29 = sum(T29[r][2] for r in REGIONS)
    say(T29["Ensemble"][2] == T21["ordinary"] and abs(s29 - T21["ordinary"]) <= 1,
        f"Tableau 2.9's 20 régions sum to {s29:,} against the printed {T21['ordinary']:,} "
        "(Tableau 2.1's ordinary households), within one person")
    say(all(m + f == t for m, f, t in list(T29.values()) + list(T23.values())),
        "Tableaux 2.9 and 2.3: masculin + féminin = ensemble on every row")
    s23 = sum(T23[r][2] for r in REGIONS)
    say(T23["Ensemble"][2] == T21["total"] and abs(s23 - T21["total"]) <= 1,
        f"Tableau 2.3's régions sum to {s23:,} against the printed {T21['total']:,}, within one")

    # 4. the universe
    t21 = " ".join(_lines(es, ES_T21))
    spaced = [f"{v:,}".replace(",", " ") for v in T21.values()]
    say(all(s in t21 for s in spaced), "Tableau 2.1 prints all five population counts")
    say(sum(v for k, v in T21.items() if k != "total") == T21["total"],
        "ordinary + collective + homeless + not enumerated = 22,395,489")
    say(sum(T12[r][2] for r in T12 if r != "Ensemble") == T12["Ensemble"][2] == T21["not_enumerated"]
        and all((u or 0) + r_ == t for u, r_, t in T12.values()),
        "Tableau 1.2's régions sum to 941,335, and urbain + rural = ensemble")
    resid = {r: T23[r][2] - T29[r][2] - T12.get(r, (0, 0, 0))[2] for r in REGIONS}
    neg = sorted(r for r, v in resid.items() if v < 0)
    say(sum(resid.values()) == T21["collective"] + T21["homeless"],
        f"2.3 - 2.9 - 1.2 sums to the collective and homeless population, "
        f"{T21['collective'] + T21['homeless']:,}")
    say(neg == ["Douentza"] and resid["Douentza"] + resid["Bandiagara"] >= 0,
        f"that residual is negative only in Douentza ({resid['Douentza']:+,}), and Douentza + "
        f"Bandiagara is {resid['Douentza'] + resid['Bandiagara']:+,}: Tableau 1.2 disagrees with "
        "2.3 minus 2.9 about which of the two holds the unenumerated (not resolved)")

    # 5. the two volumes agree, cell by cell, to their rounding
    dmus = max(abs(T613[r][0] - T203[r][0]) for r in rows)
    dani = max(abs(T613[r][4] - T203[r][2]) for r in rows)
    dchr = max(abs(sum(T613[r][1:4]) - T203[r][1]) for r in rows)
    say(dmus <= 0.10 + 1e-9 and dani <= 0.05 + 1e-9,
        f"6.13 against 2.03: Muslim within {dmus:.2f} (Kidal's closure, below), animist within "
        f"{dani:.2f}")
    say(dchr <= 0.155 + 1e-9, f"6.13's three Christian cells against 2.03's one: within {dchr:.2f}")
    loose = [(r, c, T613[r][i6], T203[r][i3]) for r in rows
             for c, i6, i3 in (("Musulman", 0, 0), ("Sans religion", 5, 3), ("Autre religion", 6, 4))
             if abs(T613[r][i6] - T203[r][i3]) > 0.05 + 1e-9]
    say(all(abs(a - b) <= 0.15 + 1e-9 for _r, _c, a, b in loose),
        f"Muslim, `Sans religion` and `Autre religion`: {len(loose)} cells beyond rounding, all "
        "within 0.15, where 6.13 closed its rows")
    for r, c, a, b in loose:
        print(f"        {r:<11} {c:<15} 6.13 {a:.1f}   2.03 {b:.2f}")

    pop = {r: T29[r][2] for r in REGIONS}
    tot = T21["ordinary"]
    for i, c in enumerate(FIVE):
        w = sum(T203[r][i] * pop[r] for r in REGIONS) / tot
        say(abs(w - T203["Ensemble"][i]) <= 0.006, f"2.03 {c:<15} population-weighted "
            f"{w:.3f} against the printed {T203['Ensemble'][i]:.2f}")

    # 6. A01, and the proration that turns it into 2.01
    for j, sex in enumerate(("masculin", "féminin", "ensemble")):
        say(sum(A01[c][j] for c in A01 if c != "Total") == A01["Total"][j] == T29["Ensemble"][j],
            f"A01 {sex}: the eight rows sum to the Total, which is Tableau 2.9's")
    say(all(abs(m + f - t) <= 2 for m, f, t in A01.values()),
        "A01: masculin + féminin within 2 of ensemble on every row (weighted counts, rounded)")
    worst = 0
    for j in range(3):
        k = A01["Total"][j] / (A01["Total"][j] - A01["Non Déclaré"][j])
        for c in FIVE:
            src = CHRISTIAN if c == "Chrétien" else [c]
            worst = max(worst, abs(k * sum(A01[s][j] for s in src) - T201[c][j]))
    say(worst <= 3, f"Tableau 2.01 is A01 with `Non Déclaré` prorated over the answers: every "
        f"cell within {worst:.1f} people")
    say(all(abs(round(100 * T201[c][2] / tot, 2) - T203["Ensemble"][i]) <= 0.005
            for i, c in enumerate(FIVE)), "and 2.01's shares are 2.03's national row")

    # 7. the form
    q = " ".join(_lines(qf, QF_P10))
    codes = [r"1 = Musulman", r"2 = Catholique", r"3 = Protestant", r"4 = Autre r[ée]ligion",
             r"5 = Animiste", r"6 = Sans r[ée]ligion"]
    pos = [m.start() if (m := re.search(c, q)) else -1 for c in codes]
    # `(?<!\d)`: the relationship codes print "17 = Autre parent du CM" on the same page.
    say(all(p >= 0 for p in pos) and pos == sorted(pos)
        and re.search(r"(?<!\d)7 = Autre r", q) is None,
        "the ordinary-household form's P10 prints codes 1-6 in the tables' order, and no 7")
    t101 = " ".join(_lines(cc, CC_T101))
    say(re.search(r"6= Sans religion 7= Autre religion", t101) is not None,
        "Tableau 1.01 of the cultural volume lists 7= Autre religion")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def _lr(values, total):
    """Largest-remainder rounding of non-negative floats to integers summing to `total`."""
    base = [int(v) for v in values]
    short = total - sum(base)
    order = sorted(range(len(values)), key=lambda i: -(values[i] - base[i]))
    for i in order[:max(short, 0)]:
        base[i] += 1
    if short < 0:
        raise SystemExit("largest remainder asked to remove people")
    return base


def _rake(seed, rows, cols, label):
    """Iterative proportional fitting of a {row: [cells]} seed to row and column margins."""
    m = {r: list(v) for r, v in seed.items()}
    ncol = len(cols)
    for _ in range(500):
        for j in range(ncol):
            s = sum(m[r][j] for r in m)
            if s > 0:
                for r in m:
                    m[r][j] *= cols[j] / s
        for r in m:
            s = sum(m[r])
            if s > 0:
                m[r] = [x * rows[r] / s for x in m[r]]
        err = max(abs(sum(m[r][j] for r in m) - cols[j]) for j in range(ncol))
        if err < 0.01:
            break
    else:
        raise SystemExit(f"{label}: the rake did not converge (column error {err:.2f})")
    out = {r: _lr(m[r], rows[r]) for r in m}
    return out, m


def emit():
    pop = {r: T29[r][2] for r in REGIONS}
    tot = T21["ordinary"]

    # national margins: A01 with its non-response prorated, to whole people. They are rounded to
    # the 20 régions' own sum, 21,347,586, one under the printed national 21,347,587, or the two
    # margins of the rake disagree by one person and it never settles.
    k = tot / (tot - A01["Non Déclaré"][2])
    nat = dict(zip(CATS, _lr([A01[c][2] * k * sum(pop.values()) / tot for c in CATS],
                             sum(pop.values()))))
    five_nat = [nat["Musulman"], sum(nat[c] for c in CHRISTIAN), nat["Animiste"],
                nat["Sans religion"], nat["Autre religion"]]
    bad = [(c, a, T201[c][2]) for c, a in zip(FIVE, five_nat) if abs(a - T201[c][2]) > 2]
    if bad:
        raise SystemExit(f"prorated A01 does not reproduce Tableau 2.01: {bad}")

    # stage 1: the five groups
    seed = {r: [T203[r][i] / 100.0 * pop[r] for i in range(5)] for r in REGIONS}
    s1, _ = _rake(seed, pop, five_nat, "stage 1")
    print("\n  stage 1, Tableau 2.03 on Tableau 2.9, raked to A01 prorated:")
    for i, c in enumerate(FIVE):
        raw = sum(seed[r][i] for r in REGIONS)
        got = sum(s1[r][i] for r in REGIONS)
        print(f"    {c:<15} seed {raw:>12,.0f}  x{five_nat[i] / raw:.5f}  -> {got:>10,} "
              f"(margin {five_nat[i]:,})")
        if five_nat[i] / tot > 0.01 and abs(five_nat[i] / raw - 1) > 0.005:
            raise SystemExit(f"{c} needs x{five_nat[i] / raw:.4f}, too far from 1 for rounding")

    # stage 2: each région's Christians, split by Tableau 6.13
    chr_rows = {r: s1[r][1] for r in REGIONS}
    empty = [r for r in REGIONS if chr_rows[r] > 0 and sum(T613[r][1:4]) == 0]
    if empty:
        raise SystemExit(f"Christians with no split in 6.13: {empty}")
    cols = _lr([nat[c] * sum(chr_rows.values()) / sum(nat[c] for c in CHRISTIAN) for c in CHRISTIAN],
               sum(chr_rows.values()))
    seed2 = {r: [chr_rows[r] * T613[r][1 + i] / sum(T613[r][1:4]) for i in range(3)]
             for r in REGIONS}
    s2, _ = _rake(seed2, chr_rows, cols, "stage 2")
    print("  stage 2, each région's Christians split by Tableau 6.13, raked to A01 prorated:")
    for i, c in enumerate(CHRISTIAN):
        raw = sum(seed2[r][i] for r in REGIONS)
        print(f"    {c:<26} seed {raw:>9,.0f}  x{cols[i] / raw:.4f}  -> {cols[i]:>8,}")

    counts = {r: {"Musulman": s1[r][0], "Catholique": s2[r][0], "Protestant": s2[r][1],
                  "Autre religion chrétienne": s2[r][2], "Animiste": s1[r][2],
                  "Sans religion": s1[r][3], "Autre religion": s1[r][4]} for r in REGIONS}

    # the drawn table, read back as shares, against both printed tables
    # 6.13's one-decimal Catholic cells, population-weighted, overstate A01's Catholics by 1.5%
    # (Bamako's 1.5 carries 4.2 million people), so the rake takes Catholics down by that much
    # everywhere; with the 0.05 of printed rounding that is up to about 0.15 pp in San.
    diffs = [(abs(100 * counts[r][c] / pop[r] - T613[r][CATS.index(c)]), r, c)
             for r in REGIONS for c in ("Musulman", "Catholique", "Protestant")]
    worst6 = max(diffs)
    worst2 = max(abs(100 * sum(counts[r][c] for c in CHRISTIAN) / pop[r] - T203[r][1])
                 for r in REGIONS)
    print(f"  drawn shares against 6.13 (Muslim, Catholic, Protestant): within {worst6[0]:.3f} pp "
          f"(worst {worst6[1]} {worst6[2]}); Christians against 2.03: within {worst2:.3f} pp")
    if worst6[0] > 0.15 or worst2 > 0.02:
        raise SystemExit("the rake moved a région's share beyond the printed rounding")

    rows = []
    for r in REGIONS:
        for c in ["Total"] + CATS:
            n = pop[r] if c == "Total" else counts[r][c]
            if n <= 0:
                continue
            note = (f"Tableau 2.9 ordinary-household residents" if c == "Total" else
                    f"{100.0 * n / pop[r]:.3f}% of {pop[r]:,}; Tableau 2.03 "
                    + (f"Chrétien {T203[r][1]:.2f} split by Tableau 6.13 "
                       f"{T613[r][1]:.1f}/{T613[r][2]:.1f}/{T613[r][3]:.1f}"
                       if c in CHRISTIAN else f"{T203[r][FIVE.index(c)]:.2f}")
                    + "; raked to annex A01 with non-response prorated")
            rows.append({"geo_id": r, "geo_level": "region", "geo_name": r,
                         "source_category": c, "count": n, "basis": BASIS, "year": YEAR,
                         "source_id": SOURCE_ID, "note": note})
    return rows, counts, nat


def main():
    import fitz

    if "--fetch" in sys.argv:
        fetch()
    for key in FILES:
        if not os.path.exists(path(key)):
            raise SystemExit(f"{path(key)} missing — run: python sources/ml.py --fetch")
    docs = {key: fitz.open(path(key)) for key in FILES}
    check(docs)
    rows, counts, nat = emit()

    tot = T21["ordinary"]
    print("\n  national: " + ", ".join(f"{c} {100 * nat[c] / tot:.2f}%" for c in CATS))
    for c in CATS[1:]:
        top = sorted(REGIONS, key=lambda r: -counts[r][c] / T29[r][2])[:4]
        big = max(REGIONS, key=lambda r: counts[r][c])
        print(f"  {c:<26} highest share: " + ", ".join(
            f"{r} {100 * counts[r][c] / T29[r][2]:.2f}%" for r in top)
            + f"; largest count {big}, {100 * counts[big][c] / nat[c]:.1f}% of the national")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".part", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(OUT + ".part", OUT)
    print("\nwrote", OUT, f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
