"""Burkina Faso — RGPH 2006, religion by province, the whole resident population.

Reads (or fetches) data/raw/bf/Theme2-Etat_et_structure_de_la_population.pdf and writes
data/normalized/bf.csv. `sources/bf.md` is the write-up; `sources/bf_geo.py` builds the
polygons and the grid.

## THE PROVINCE TABLE IS IN THE STRUCTURE VOLUME'S ANNEX, ON A RETIRED TREE

INSD, *Recensement général de la population et de l'habitation de 2006, Thème 2: État et
structure de la population* (181 pp). Chapter 5 prints religion by région as shares (Tableau
5.3, PDF p95); the annex prints it in counts, and only the annex goes down to province:

    Tableau A 3.1 bis   province x sex, with area and density         PDF p121    witness
    Tableau A5.4        nation x milieu x sex x religion, counts      PDF p163    witness
    Tableau A5.5        région x sex x religion, counts               PDF p163    witness
    Tableau A5.6        province x religion, counts                   PDF p164-5  <- drawn
    Tableau A5.7        age x religion, counts                        PDF p165    the ND row

The file is not on insd.bf's current site. Three Wayback captures on three retired paths have
one SHA-1 (sources.md §11aq found them by listing CDX directories), and `fetch()` tries them in
turn against that digest.

**The drawn numbers are counts, exact: no share is multiplied by anything.** The 45 provinces'
six columns sum to A5.4's national row and to UNSD table 28's Burkina Faso 2006 row to the
person; grouped by région they equal A5.5 cell for cell, which also proves the province ->
région assignment below; and their totals equal A 3.1 bis, a table of population by sex.

## THE FORM, AND WHY THERE IS NO GAP

Column P15 of the household form (IPUMS International's copy, `enum_form_bf2006a.pdf` p11, read
2026-09-14) asks *"Quelle est la religion de (Nom) ?"* and has the enumerator circle one of six
printed codes: 1 Animiste, 2 Musulman, 3 Catholique, 4 Protestant, 5 Autre, 6 Sans religion.
The table's columns are those codes in that order, and animism has its own box beside `Sans
religion` (so the no-religion procedure's `separate` case, not Mozambique's lumped box).

There is no code for no answer, and Tableau 1.3 (PDF p45) prints 0.00% non-déclarés for
religion in every cell, against 0.53% for age and 1.53% for language. The universe is every
resident, 14,017,262, the census's headline population. Nothing is left out, so no `gap`.

Children under six were given their mother's religion, or that of the person caring for them
if she was not in the household (p37, and the enumerator manual's column P15).

## `Autre` PROBABLY HOLDS PART OF THE NON-RESPONSE

A form with no non-response code and a table with 0.00% non-response has put the missing
answers somewhere. A5.7's `ND` row (people whose age was not recorded) points at `Autre`: of
its 74,487 people, 13,919 (18.7%) are `Autre`, against 0.57% of the population. Records
missing one item tend to miss others, so `Autre` very probably includes records whose religion
was blank. How much of its 79,485 that is, nothing printed says. `check()` asserts the row;
taxonomy/bf2006.py's REVIEW carries the call.

Usage:
    python sources/bf.py --fetch    one 2.3 MB PDF from the Wayback Machine
    python sources/bf.py            normalise from data/raw/bf/
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
RAW = os.path.join(ROOT, "data", "raw", "bf")
OUT = os.path.join(ROOT, "data", "normalized", "bf.csv")
sys.path.insert(0, os.path.join(ROOT, "tools"))

from fetch_checks import FetchCheckError, check_body, check_pdf_doc, digest  # noqa: E402

SOURCE_ID = "bf_rgph2006_theme2"
YEAR = 2006
BASIS = "self_id"
COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

_TAIL = "Theme2-Etat_et_structure_de_la_population.pdf"
WAYBACK = [
    ("https://web.archive.org/web/20190203182046id_/http://www.insd.bf:80/documents/"
     "publications/insd/publications/resultats_enquetes/RGPH2006/" + _TAIL),
    ("https://web.archive.org/web/20210824213728id_/http://www.insd.bf/contenu/"
     "enquetes_recensements/rgph-bf/themes_en_demographie/" + _TAIL),
    "https://web.archive.org/web/20101113152623id_/http://www.insd.bf/fr/IMG/pdf/" + _TAIL,
]
PDF = os.path.join(RAW, _TAIL)
SIZE = 2_348_972
DIGEST = "TEQWCJXT64ODTIJH47FGJFD4KECGRVTU"   # SHA-1, base32, as the Wayback CDX writes it
PAGES = 181

# 0-based page indices.
PAGE_T13 = 44      # PDF p45, Tableau 1.3, non-response by variable
PAGE_A31 = 120     # PDF p121, Tableau A 3.1 bis, province x sex, area, density
PAGE_A54 = 162     # PDF p163, Tableau A5.4 national and A5.5 région (Ensemble block)
PAGE_A56 = 163     # PDF p164, Tableau A5.6 first 22 provinces
PAGE_A56B = 164    # PDF p165, Tableau A5.6 the other 23 and Total; Tableau A5.7

CATS = ["Animiste", "Musulman", "Catholique", "Protestant", "Autre", "Sans religion"]

# Tableau A5.6 as printed, in its own row order: the 30 provinces of 1985, then the 15 cut in
# 1996 (A 3.1 bis's second page lists the splits). Values in CATS order, then Total. Generated
# from the page by solving each row's cell boundaries against its own total, then checked here
# digit for digit against the page and by every sum below.
A56 = {
    "Bam": (9461, 206105, 56218, 2622, 730, 55, 275191),
    "Bazèga": (30771, 113848, 82293, 10132, 1055, 326, 238425),
    "Bougouriba": (62747, 22624, 11117, 3807, 640, 544, 101479),
    "Boulgou": (33774, 411939, 81446, 12849, 2389, 1173, 543570),
    "Boulkiemdé": (108718, 163665, 189852, 39062, 2533, 1376, 505206),
    "Comoé": (67839, 294039, 33562, 6354, 2247, 3487, 407528),
    "Ganzourgou": (22083, 212931, 72500, 10023, 1666, 177, 319380),
    "Gnagna": (79341, 166468, 79596, 76224, 2835, 4205, 408669),
    "Gourma": (55236, 146882, 84238, 14651, 2461, 2468, 305936),
    "Houet": (54947, 712836, 153317, 24879, 6116, 3356, 955451),
    "Kadiogo": (13193, 966141, 625034, 104412, 16887, 1723, 1727390),
    "Kénédougou": (39000, 223451, 16440, 3058, 1157, 2589, 285695),
    "Kossi": (23772, 173957, 59387, 18901, 1297, 1232, 278546),
    "Kouritenga": (3135, 205766, 115913, 3378, 1377, 210, 329779),
    "Mouhoun": (50436, 180323, 49150, 14575, 1460, 1406, 297350),
    "Nahouri": (72028, 44422, 19765, 18882, 1201, 773, 157071),
    "Namentenga": (121769, 164452, 31775, 8813, 1501, 510, 328820),
    "Oubritenga": (16647, 154464, 59771, 6524, 1263, 106, 238775),
    "Oudalan": (2138, 192088, 647, 206, 870, 15, 195964),
    "Passoré": (106196, 136184, 62387, 16816, 1216, 423, 323222),
    "Poni": (192070, 35860, 13901, 11592, 1803, 1705, 256931),
    "Sanguié": (71988, 61661, 131961, 23052, 2366, 6008, 297036),
    "Sanmatenga": (152828, 346515, 79537, 15186, 2749, 1199, 598014),
    "Séno": (4022, 256600, 2092, 880, 1234, 163, 264991),
    "Sissili": (33155, 137101, 26482, 10232, 812, 627, 208409),
    "Soum": (5408, 334582, 2826, 1348, 2895, 276, 347335),
    "Sourou": (22657, 153206, 34846, 8953, 811, 149, 220622),
    "Tapoa": (195655, 66059, 37617, 33443, 3028, 6503, 342305),
    "Yatenga": (5604, 530121, 11902, 3319, 2141, 77, 553164),
    "Zoundwéogo": (13594, 153511, 71144, 6451, 964, 283, 245947),
    "Bale": (64462, 112752, 25555, 8664, 777, 1213, 213423),
    "Banwa": (45573, 182921, 30979, 8758, 898, 246, 269375),
    "Ioba": (97137, 13174, 77450, 2142, 988, 1430, 192321),
    "Komandjoari": (22821, 46404, 3968, 5733, 308, 273, 79507),
    "Kompienga": (18657, 36725, 12061, 5366, 367, 2691, 75867),
    "Koulpelogo": (19960, 190505, 42675, 3875, 947, 705, 258667),
    "Kourwéogo": (38207, 47926, 44045, 6915, 764, 360, 138217),
    "Léraba": (7516, 113073, 2264, 607, 529, 291, 124280),
    "Loroum": (506, 138524, 2533, 873, 378, 39, 142853),
    "Nayala": (9091, 93798, 55432, 4369, 512, 231, 163433),
    "Noumbiel": (50760, 8634, 7330, 2719, 115, 478, 70036),
    "Tuy": (66779, 125682, 22444, 11930, 972, 651, 228458),
    "Yagha": (4706, 150694, 763, 2345, 878, 766, 160152),
    "Ziro": (26099, 114445, 26152, 8028, 830, 361, 175915),
    "Zondoma": (7823, 142091, 13869, 2206, 518, 50, 166557),
}
A56_TOTAL = (2150309, 8485149, 2664236, 585154, 79485, 52929, 14017262)

# The drawn table's spelling -> the name this project writes. A5.6 prints `Bale` in its second
# block; A 3.1 bis and every other table print `Balé`.
CANON = {"Bale": "Balé"}

# Tableau A5.5, the Ensemble block (PDF p163), CATS order then Total.
A55 = {
    "Boucle du Mouhoun": (215991, 896957, 255349, 64220, 5755, 4477, 1442749),
    "Cascades": (75355, 407112, 35826, 6961, 2776, 3778, 531808),
    "Centre": (13193, 966141, 625034, 104412, 16887, 1723, 1727390),
    "Centre-est": (56869, 808210, 240034, 20102, 4713, 2088, 1132016),
    "Centre-nord": (284058, 717072, 167530, 26621, 4980, 1764, 1202025),
    "Centre-ouest": (239960, 476872, 374447, 80374, 6541, 8372, 1186566),
    "Centre-sud": (116393, 311781, 173202, 35465, 3220, 1382, 641443),
    "Est": (371710, 462538, 217480, 135417, 8999, 16140, 1212284),
    "Hauts-bassins": (160726, 1061969, 192201, 39867, 8245, 6596, 1469604),
    "Nord": (120129, 946920, 90691, 23214, 4253, 589, 1185796),
    "Plateau central": (76937, 415321, 176316, 23462, 3693, 643, 696372),
    "Sahel": (16274, 933964, 6328, 4779, 5877, 1220, 968442),
    "Sud-ouest": (402714, 80292, 109798, 20260, 3546, 4157, 620767),
}

# The 13 régions of 2001-2025 and their 45 provinces, keyed by A5.6's spelling. Asserted: every
# religion column of A5.6 summed over each région's provinces equals A5.5. (The 2025 reform made
# 17 régions and 47 provinces; the census predates it.)
REGION_OF = {
    **dict.fromkeys(["Bale", "Banwa", "Kossi", "Mouhoun", "Nayala", "Sourou"], "Boucle du Mouhoun"),
    **dict.fromkeys(["Comoé", "Léraba"], "Cascades"),
    "Kadiogo": "Centre",
    **dict.fromkeys(["Boulgou", "Koulpelogo", "Kouritenga"], "Centre-est"),
    **dict.fromkeys(["Bam", "Namentenga", "Sanmatenga"], "Centre-nord"),
    **dict.fromkeys(["Boulkiemdé", "Sanguié", "Sissili", "Ziro"], "Centre-ouest"),
    **dict.fromkeys(["Bazèga", "Nahouri", "Zoundwéogo"], "Centre-sud"),
    **dict.fromkeys(["Gnagna", "Gourma", "Komandjoari", "Kompienga", "Tapoa"], "Est"),
    **dict.fromkeys(["Houet", "Kénédougou", "Tuy"], "Hauts-bassins"),
    **dict.fromkeys(["Loroum", "Passoré", "Yatenga", "Zondoma"], "Nord"),
    **dict.fromkeys(["Ganzourgou", "Kourwéogo", "Oubritenga"], "Plateau central"),
    **dict.fromkeys(["Oudalan", "Séno", "Soum", "Yagha"], "Sahel"),
    **dict.fromkeys(["Bougouriba", "Ioba", "Noumbiel", "Poni"], "Sud-ouest"),
}

# Tableau A 3.1 bis (PDF p121): (masculin, féminin, total, superficie km2, densité), keyed by
# its own spelling. The areas are also bf_geo.py's witness for the polygons.
A31 = {
    "Burkina Faso": (6768739, 7248523, 14017262, 272967, 51.4),
    "Balé": (105582, 107841, 213423, 4539, 47.0),
    "Bam": (130228, 144963, 275191, 4010, 68.6),
    "Banwa": (132052, 137323, 269375, 5802, 46.4),
    "Bazèga": (111459, 126966, 238425, 3947, 60.4),
    "Bougouriba": (49440, 52039, 101479, 2774, 36.6),
    "Boulgou": (250908, 292662, 543570, 6520, 83.4),
    "Boulkiemdé": (223195, 282011, 505206, 4275, 118.2),
    "Comoé": (201453, 206075, 407528, 15405, 26.5),
    "Ganzourgou": (149969, 169411, 319380, 4169, 76.6),
    "Gnagna": (199252, 209417, 408669, 8544, 47.8),
    "Gourma": (148270, 157666, 305936, 11212, 27.3),
    "Houet": (474086, 481365, 955451, 11548, 82.7),
    "Ioba": (93245, 99076, 192321, 3261, 59.0),
    "Kadiogo": (867010, 860380, 1727390, 2869, 602.2),
    "Kénédougou": (140950, 144745, 285695, 8404, 34.0),
    "Komandjari": (39419, 40088, 79507, 5125, 15.5),
    "Kompienga": (38357, 37510, 75867, 6967, 10.9),
    "Kossi": (138459, 140087, 278546, 7426, 37.5),
    "Koulpelogo": (125276, 133391, 258667, 5392, 48.0),
    "Kouritenga": (153149, 176630, 329779, 2798, 117.8),
    "Kourwéogo": (62157, 76060, 138217, 1595, 86.6),
    "Léraba": (59915, 64365, 124280, 3019, 41.2),
    "Loroum": (67590, 75263, 142853, 3685, 38.8),
    "Mouhoun": (148089, 149261, 297350, 6872, 43.3),
    "Nahouri": (76152, 80919, 157071, 3842, 40.9),
    "Namentenga": (157079, 171741, 328820, 6391, 51.5),
    "Nayala": (81208, 82225, 163433, 3718, 44.0),
    "Noumbiel": (34241, 35795, 70036, 2810, 24.9),
    "Oubritenga": (112462, 126313, 238775, 2841, 84.0),
    "Oudalan": (97563, 98401, 195964, 10069, 19.5),
    "Passoré": (149146, 174076, 323222, 3978, 81.2),
    "Poni": (122338, 134593, 256931, 7472, 34.4),
    "Sanguié": (137548, 159488, 297036, 5107, 58.2),
    "Sanmatenga": (278679, 319335, 598014, 9276, 64.5),
    "Séno": (131754, 133237, 264991, 6997, 37.9),
    "Sissili": (101297, 107112, 208409, 7080, 29.4),
    "Soum": (171505, 175830, 347335, 12540, 27.7),
    "Sourou": (108952, 111670, 220622, 5976, 36.9),
    "Tapoa": (169570, 172735, 342305, 14846, 23.1),
    "Tuy": (111193, 117265, 228458, 5622, 40.6),
    "Yagha": (80553, 79599, 160152, 6536, 24.5),
    "Yatenga": (261272, 291892, 553164, 6770, 81.7),
    "Ziro": (84785, 91130, 175915, 5291, 33.2),
    "Zondoma": (76684, 89873, 166557, 1980, 84.1),
    "Zoundwéogo": (115248, 130699, 245947, 3668, 67.1),
}
# A5.6 spelling -> A 3.1 bis spelling, where they differ.
A31_NAME = {"Bale": "Balé", "Komandjoari": "Komandjari"}

# Tableau A5.7's `ND` row (age not recorded), CATS order then Total: the `Autre` evidence.
AGE_ND = (10157, 38254, 9669, 2182, 13919, 306, 74487)

UNSD = {"Animist": "Animiste", "Muslim": "Musulman", "Catholic": "Catholique",
        "Protestant": "Protestant", "Other": "Autre", "No Religion": "Sans religion"}


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def fetch():
    import urllib.request

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(PDF) and os.path.getsize(PDF) == SIZE:
        print("already have", PDF)
        return
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}
    for url in WAYBACK:
        try:
            req = urllib.request.Request(url, headers=ua)
            with urllib.request.urlopen(req, timeout=600) as r:
                body = r.read()
        except Exception as e:                       # noqa: BLE001 - try the next capture
            print(f"  {url[:80]}...: {e}")
            continue
        try:
            print("  " + check_body(body, "pdf", where=url[:80], pin_digest=DIGEST))
        except FetchCheckError as e:
            print(f"  {e}")
            continue
        with open(PDF + ".part", "wb") as fh:
            fh.write(body)
        os.replace(PDF + ".part", PDF)
        print(f"wrote {PDF} ({len(body):,} bytes)")
        return
    raise SystemExit("none of the three Wayback captures returned the pinned PDF")


def _text(doc, pno):
    return unicodedata.normalize("NFC", doc.load_page(pno).get_text())


def _label(name):
    words = [re.escape(w) for w in name.split()]
    return re.compile(r"(?<![\w-])" + r"\s+".join(words) + r"(?![\w-])")


def _digits(values):
    return "".join(f"{v:.1f}".replace(".", ",") if isinstance(v, float) else str(v)
                   for v in values)


def segments(text, labels, start=None, ordered=True):
    """{label: the digits and commas printed between it and the next label}.

    `ordered` finds the labels one after another from `start`; otherwise each is found on its
    own and the page order is taken from where they fall (A 3.1 bis runs two columns side by
    side, so its text interleaves them).
    """
    pos = 0 if start is None else text.find(start)
    if pos < 0:
        raise SystemExit(f"no {start!r} on the page")
    found = []
    for name in labels:
        m = _label(name).search(text, pos)
        if not m:
            raise SystemExit(f"no row {name!r} after {start!r}")
        found.append((name, m.start(), m.end()))
        if ordered:
            pos = m.end()
    found.sort(key=lambda f: f[1])
    out = {}
    for k, (name, _s, e) in enumerate(found):
        end = found[k + 1][1] if k + 1 < len(found) else len(text)
        out[name] = re.sub(r"[^\d,]", "", text[e:end])
    return out


def unsd_counts():
    try:
        import oracle
        rows = oracle.oracle("Burkina Faso", 2006)
    except SystemExit:
        rows = None
    if not rows:
        return None
    tot = next((v for k, v in rows.items() if "total" in k.lower()), None)
    return {c: n for c, n in tot.items() if c != oracle.TOTAL} if tot else None


def check(doc):
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    print("Burkina Faso — RGPH 2006, Thème 2, Tableau A5.6\n")
    with open(PDF, "rb") as fh:
        body = fh.read()
    say(digest(body) == DIGEST, f"file digest {digest(body)} (expected {DIGEST})")
    for good, msg in check_pdf_doc(doc, PAGES):
        say(good, msg)

    # 1. the page equals the transcription, digit for digit, row by row
    names = list(A56)
    t1 = _text(doc, PAGE_A56)
    t1 = t1[t1.find("Tableau A5.6"):]
    t2 = _text(doc, PAGE_A56B)
    t2 = t2[:t2.find("Tableau A5.7")]
    seg = {**segments(t1, names[:22], names[0]), **segments(t2, names[22:] + ["Total"], names[22])}
    want = {**{p: _digits(v) for p, v in A56.items()}, "Total": _digits(A56_TOTAL)}
    bad = [p for p in want if seg[p] != want[p]]
    say(not bad, f"Tableau A5.6: all {len(want)} rows (45 provinces and Total) equal the "
        "transcription digit for digit" + (f"; differ: {bad}" if bad else ""))

    t3 = _text(doc, PAGE_A54)
    a55t = t3[t3.find("Tableau A5.5"):]
    a55t = a55t[:a55t.find("Masculin", a55t.find("Ensemble"))]
    seg55 = segments(a55t, list(A55) + ["Total"], "Boucle")
    bad = [r for r in A55 if seg55[r] != _digits(A55[r])] + (
        [] if seg55["Total"] == _digits(A56_TOTAL) else ["Total"])
    say(not bad, "Tableau A5.5 (Ensemble): all 13 régions and Total equal the transcription"
        + (f"; differ: {bad}" if bad else ""))

    a31t = _text(doc, PAGE_A31)
    a31t = a31t[a31t.find("Densité"):]
    seg31 = segments(a31t, list(A31), ordered=False)
    bad = [p for p in A31 if seg31[p] != _digits(A31[p])]
    say(not bad, "Tableau A 3.1 bis: all 46 rows (sex, total, area, density) equal the "
        "transcription" + (f"; differ: {bad}" if bad else ""))

    # 2. A5.6 closes on itself
    say(len(A56) == 45 and set(A56) == set(REGION_OF), "45 provinces, each assigned one région")
    say(all(sum(v[:6]) == v[6] for v in A56.values()),
        "every province's six answers sum to its printed total")
    say(all(sum(v[c] for v in A56.values()) == A56_TOTAL[c] for c in range(7)),
        "every column over the 45 provinces sums to the printed Total row, 14,017,262")

    # 3. grouped by région, A5.6 is A5.5 (this also proves REGION_OF)
    bad = []
    for r, v in A55.items():
        got = tuple(sum(A56[p][c] for p in A56 if REGION_OF[p] == r) for c in range(7))
        if got != v:
            bad.append((r, got, v))
    say(not bad, "A5.6's provinces summed by région equal Tableau A5.5 in all 7 columns for "
        "all 13 régions" + (f"; differ: {bad}" if bad else ""))
    say(all(sum(v[c] for v in A55.values()) == A56_TOTAL[c] for c in range(7)),
        "and A5.5's régions sum to the same national row")

    # 4. A 3.1 bis, a table of population by sex, has the same province totals
    bad = [p for p in A56 if A31[A31_NAME.get(p, p)][2] != A56[p][6]]
    say(not bad and set(A31_NAME.get(p, p) for p in A56) | {"Burkina Faso"} == set(A31),
        "Tableau A 3.1 bis prints the same 45 province totals" + (f"; differ: {bad}" if bad else ""))
    say(all(m + f == t for m, f, t, _a, _d in A31.values()),
        "A 3.1 bis: masculin + féminin = total on all 46 rows")
    say(all(t / (a + 0.5) - 0.0501 <= d <= t / (a - 0.5) + 0.0501
            for _m, _f, t, a, d in A31.values()),
        "A 3.1 bis: every density is its total over its whole-km2 area, to rounding")
    # Not asserted: the office prints a national area that is not the sum of its provinces.
    area45 = sum(v[3] for k, v in A31.items() if k != "Burkina Faso")
    print(f"  --  A 3.1 bis: the 45 printed areas sum to {area45:,} km2 against the printed "
          f"national {A31['Burkina Faso'][3]:,} ({area45 - A31['Burkina Faso'][3]:+,})")

    # 5. the national row: A5.4 and UNSD
    t54 = t3[t3.find("Tableau A5.4"):t3.find("Tableau A5.5")]
    seg54 = segments(t54, CATS, "Animiste")
    say(all(seg54[c].startswith(str(A56_TOTAL[i])) for i, c in enumerate(CATS)),
        "Tableau A5.4's Ensemble column opens with the same six national counts")
    u = unsd_counts()
    if u is None:
        print("  -- oracle cache not present; UNSD check skipped")
    else:
        say({UNSD[k]: v for k, v in u.items()} == dict(zip(CATS, A56_TOTAL[:6])),
            f"UNSD table 28 Burkina Faso 2006 equals the table to the person: {u}")

    # 6. no non-response for religion, as printed
    t13 = _text(doc, PAGE_T13)
    say(re.search(r"Religion\s+(?:0,00\s+){9}", t13) is not None,
        "Tableau 1.3 prints 0,00% religion non-déclarés in all nine milieu x sex cells")

    # 7. the `Autre` evidence (A5.7's ND row)
    t57 = t2_full = _text(doc, PAGE_A56B)
    t57 = t2_full[t2_full.find("Tableau A5.7"):]
    nd = segments(t57, ["ND", "Total"], "95+")
    say(nd["ND"] == _digits(AGE_ND) and sum(AGE_ND[:6]) == AGE_ND[6],
        f"Tableau A5.7: the {AGE_ND[6]:,} people with no age recorded are "
        f"{100 * AGE_ND[4] / AGE_ND[6]:.1f}% `Autre`, against "
        f"{100 * A56_TOTAL[4] / A56_TOTAL[6]:.2f}% of the population")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def emit():
    rows = []
    for p, v in A56.items():
        name = CANON.get(p, p)
        for c, n in zip(["Total"] + CATS, (v[6],) + v[:6]):
            if n <= 0:
                continue
            rows.append({
                "geo_id": name, "geo_level": "province", "geo_name": name,
                "source_category": c, "count": n, "basis": BASIS, "year": YEAR,
                "source_id": SOURCE_ID,
                "note": (f"Tableau A5.6 count; {100.0 * n / v[6]:.2f}% of the province's "
                         f"{v[6]:,} residents; région {REGION_OF[p]}"),
            })
    return rows


def main():
    import fitz

    if "--fetch" in sys.argv:
        fetch()
    if not os.path.exists(PDF):
        raise SystemExit(f"{PDF} missing — run: python sources/bf.py --fetch")
    doc = fitz.open(PDF)
    check(doc)
    rows = emit()

    tot = A56_TOTAL[6]
    print("\n  national: " + ", ".join(f"{c} {100 * A56_TOTAL[i] / tot:.2f}%"
                                      for i, c in enumerate(CATS)))
    for i, c in enumerate(CATS):
        top = sorted(A56, key=lambda p: -A56[p][i] / A56[p][6])[:4]
        print(f"  {c:<14} highest share: " + ", ".join(
            f"{CANON.get(p, p)} {100 * A56[p][i] / A56[p][6]:.1f}%" for p in top))
    for c in ("Catholique", "Protestant", "Animiste"):
        i = CATS.index(c)
        top = max(A56, key=lambda p: A56[p][i])
        print(f"  {c}: largest count in {CANON.get(top, top)}, "
              f"{100 * A56[top][i] / A56_TOTAL[i]:.1f}% of the national count")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".part", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(OUT + ".part", OUT)
    print("\nwrote", OUT, f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
