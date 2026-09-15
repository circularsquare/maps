"""Senegal — RGPH 1988, religion and Sufi order by région, with Diourbel by département.

Reads (or fetches) three PDFs into data/raw/sn/ and writes data/normalized/sn.csv.
`sources/sn.md` is the write-up; `sources/sn_geo.py` builds the units and the grid.

## THE SOURCES, ALL FROM IREDA (CEPED's inventory of African censuses)

    national   Direction de la Prévision et de la Statistique, *Recensement général de la
               population et de l'habitat de 1988 (résultats définitifs)*, June 1993, 76 pp
               Tableau 1.1  (p8)   resident population 6,896,808
               Tableau 1.2  (p9)   population and density by région
               Tableau 1.15 (p28, PDF p30)  région x religion, one decimal, % of residents
               the household form (PDF p75): P11 RELIGION, eight codes
    diourbel   the same office's Diourbel regional analysis report of the 1988 results
               Tableau 1.2 and 1.2A (p9, PDF p11)   area and population by département
               Tableau 1.12 (p30, PDF p32)          religion by milieu and département, COUNTS
    manuel     *Manuel de l'agent recenseur*, 1988: P11's instructions (p26)

`ansd.sn/sites/default/files/recensements/rapport/Chapitre 1 - ETAT DE LA POPULATION.pdf` is the
national report's chapter 1 alone and prints the same Tableau 1.15; the whole report is used here
because it carries the questionnaire and Tableau 1.1.

## WHAT THE FORM ASKED (PDF p75, and the manual p26)

P11 asked every person, with the brotherhood as the answer for Muslims: *"Pour les musulmans,
encerclez la confrérie déclarée"*, 1 KH Khadr, 2 LA Layène, 3 MO Mouride, 4 TI Tidiane, 5 AM
Muslims who belong to none of these; 6 CA Catholic, 7 AC other Christians (*protestants,
luthériens, témoins de Jéhovah etc.*); 8 AR others (*Juifs, Bouddhistes, Animistes etc.*). There is
no code for no religion and none for no answer. The national table prints the five Muslim codes,
Christians together and `Autres`; the report's own reading of `Autres` is *animisme
principalement*.

## THE TABLE IS SHARES OF THE WHOLE RESIDENT POPULATION, NOT OF MUSLIMS

The five order columns sum to `Musulmans` in every row (Dakar 6.9 + 2.1 + 23.4 + 51.5 + 8.8 =
92.7), and `Musulmans` + `Chrétiens` + `Autres` is 100. So every order is a share of all residents.
Counts are each row's seven leaf shares, divided by their printed sum (99.9 to 100.1), times
Tableau 1.2's population; Diourbel's are the regional report's counts (below).

## THE NATIONAL TABLE PRINTS DIOURBEL'S KHADRIYA IN THE LAYÈNE COLUMN

Tableau 1.15's Diourbel row reads Khadriya `-` and Layène 3.7. The Diourbel report's Tableau 1.12
counts 22,886 Khadria (3.70%) and 265 Layène (0.04%). The print is wrong, not the text layer: the
rendered page shows the same. The swap is also the only reading under which the table's own
Ensemble Layène of 0.6 comes back (0.60 swapped, 0.93 as printed). `check()` asserts both.

## THE ENSEMBLE ROW DOES NOT COME BACK FROM THE RÉGION ROWS

Weighted by Tableau 1.2's populations, the ten rows give Musulmans 94.4 against the printed 93.8,
Khadriya 11.7 against 10.9, Chrétiens 4.5 against 4.3 and Autres 1.2 against 1.6; only Layène and
Tidiane agree within 0.15. The Ensemble row does not sum to 100 either (99.7). The report does not
say where it came from; the foreword says the provisional results used a 10% sample, and the
Ensemble may be from that or from another tabulation. The région rows are what is drawn, because
Diourbel's row is the regional report's full count to the rounding. There is no rescale to the
Ensemble. `check()` pins which columns agree, so a transcription change shows.

## DIOURBEL IS DRAWN AT ITS THREE DÉPARTEMENTS

The Diourbel report's Tableau 1.12 is the only 1988 religion table below the région found
anywhere (the other nine regional reports are in East View's gated archive; sources.md §11aq). Its
département columns sum 1,387 people short of its région column (268 in Bambey, 491 in Diourbel,
628 in Mbacké against Tableau 1.2), spread over every category. Each département's eight
categories are raked to two printed margins: Tableau 1.2's département populations and Tableau
1.12's région column. Catholics and other Christians are kept as the report prints them.

Usage:
    python sources/sn.py --fetch    three PDFs from ireda.ceped.org (~9 MB)
    python sources/sn.py            normalise from data/raw/sn/
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
RAW = os.path.join(ROOT, "data", "raw", "sn")
OUT = os.path.join(ROOT, "data", "normalized", "sn.csv")
sys.path.insert(0, HERE)

from fetch_checks import FetchCheckError, check_body   # noqa: E402  shared, not copied

SOURCE_ID = "sn_rgph1988"
YEAR = 1988
BASIS = "self_id"
COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

IREDA = "https://ireda.ceped.org/inventaire/ressources/"
FILES = {   # key: (file name, bytes, SHA-1 in CDX base32, pages); fetched 2026-09-15
    "national": ("sen-1988-rec-o1_rapport_resultats_definitifs.pdf", 3_794_341,
                 "KOBVKWE7PFQBHNON3OKMWGNQ2G2NUHYL", 76),
    "diourbel": ("sen-1988-rec-o2_diourbel.pdf", 3_192_222,
                 "S7QZ763272HTG2X65A6ZFE632N7PNN2Z", 69),
    "manuel": ("sen-1988-rec-m1_manuel_recenseur.pdf", 1_983_511,
               "NTJS2ZPMA34CIUABACZULSHA3S6DXFCO", 46),
}

# 0-based page indices
NAT_T11, NAT_T12, NAT_T115, NAT_FORM = 9, 10, 29, 74
DIO_T12, DIO_T112 = 10, 31
MAN_P11 = 25

COLS = ["Musulmans", "Khadriya", "Layène", "Mouride", "Tidiane", "Autres mus.", "Chrétiens",
        "Autres"]
LEAVES = COLS[1:]
ORDERS = COLS[1:6]

# Tableau 1.15 as printed, `None` for a printed "-". Transcribed from the rendered page and
# asserted equal to the text layer.
T115 = {
    "Dakar":       (92.7,  6.9,  2.1, 23.4, 51.5,  8.8,  6.7, 0.7),
    "Ziguinchor":  (75.2, 32.0,  0.3,  4.0, 22.9, 16.0, 17.1, 7.7),
    "Diourbel":    (99.0, None,  3.7, 85.3,  9.5,  0.4,  0.6, 0.3),
    "Saint-Louis": (98.7,  8.4,  0.2,  6.4, 80.2,  3.5,  0.4, 0.9),
    "Tambacounda": (96.3, 25.2,  0.1,  7.5, 54.0,  9.4,  2.4, 1.3),
    "Kaolack":     (98.4,  4.9, None, 27.2, 65.3,  0.9,  1.0, 0.6),
    "Thiès":       (94.4,  7.4,  0.5, 44.7, 40.3,  1.5,  4.9, 0.7),
    "Louga":       (99.5, 15.1,  0.3, 45.9, 37.3,  0.9,  0.1, 0.4),
    "Fatick":      (91.8, 12.4,  0.1, 38.6, 39.6,  1.1,  7.8, 0.5),
    "Kolda":       (93.4, 26.0,  0.1,  3.6, 52.7, 11.0,  5.0, 1.6),
}
T115_ENSEMBLE = (93.8, 10.9, 0.6, 30.1, 47.4, 4.8, 4.3, 1.6)
NAME_ALIASES = {"stlouis": "Saint-Louis", "koida": "Kolda"}   # the text layer's spellings
# The misprint, corrected: Khadriya is the 3.7 printed under Layène, Layène is below 0.05.
DIOURBEL_CORRECTED = (99.0, 3.7, None, 85.3, 9.5, 0.4, 0.6, 0.3)
# Ensemble columns the population-weighted région rows reproduce within 0.15 (measured).
ENSEMBLE_AGREES = {"Layène", "Tidiane"}

# National Tableau 1.2, 1988 résident population by région.
T12 = {
    "Dakar": 1_488_941, "Ziguinchor": 398_337, "Diourbel": 619_245, "Saint-Louis": 660_282,
    "Tambacounda": 385_982, "Kaolack": 811_258, "Thiès": 941_151, "Louga": 490_077,
    "Fatick": 509_702, "Kolda": 591_833,
}
RESIDENT = 6_896_808                 # Tableau 1.1 and 1.2
COMPTEE_A_PART = 35_000              # p8: "auxquels s'ajoutent 35000 personnes ... comptée à part"

# Diourbel report Tableau 1.12, counts: (Région, Rural, Urbain, Bambey, Diourbel, Mbacké).
T112_ROWS = ["KHADRIA", "LAYENNE", "MOURIDE", "TIDIANE", "AUTRES", "TOTAL MUSULMAN",
             "CATHOLIQUE", "AUTRES CHRETIENS", "TOTAL CHRETIEN", "AUTRE RELIGION", "TOTAL"]
T112 = {
    "KHADRIA":          (22_886, 17_041, 5_845, 5_803, 8_359, 8_605),
    "LAYENNE":          (265, 120, 145, 81, 149, 35),
    "MOURIDE":          (528_118, 428_629, 99_489, 169_991, 141_686, 215_967),
    "TIDIANE":          (59_152, 34_535, 24_617, 19_435, 29_385, 10_033),
    "AUTRES":           (2_727, 1_625, 1_102, 1_128, 1_146, 305),
    "TOTAL MUSULMAN":   (613_148, 481_950, 131_198, 196_438, 180_725, 234_945),
    "CATHOLIQUE":       (3_775, 2_328, 1_447, 1_462, 1_994, 200),
    "AUTRES CHRETIENS": (222, 99, 123, 32, 154, 5),
    "TOTAL CHRETIEN":   (3_997, 2_427, 1_570, 1_494, 2_148, 205),
    "AUTRE RELIGION":   (2_100, 1_428, 672, 690, 619, 594),
    "TOTAL":            (619_245, 485_805, 133_440, 198_622, 183_492, 235_744),
}
T112_LEAVES = ["KHADRIA", "LAYENNE", "MOURIDE", "TIDIANE", "AUTRES", "CATHOLIQUE",
               "AUTRES CHRETIENS", "AUTRE RELIGION"]
DEPTS = ["Bambey", "Diourbel", "Mbacké"]
# Tableau 1.12's leaf -> Tableau 1.15's column, for comparing the two tables.
T112_AS_T115 = {"KHADRIA": "Khadriya", "LAYENNE": "Layène", "MOURIDE": "Mouride",
                "TIDIANE": "Tidiane", "AUTRES": "Autres mus.", "TOTAL CHRETIEN": "Chrétiens",
                "AUTRE RELIGION": "Autres", "TOTAL MUSULMAN": "Musulmans"}
# Diourbel report Tableau 1.2 (population) and 1.2A (men, women).
DEPT_POP = {"Bambey": 198_890, "Diourbel": 183_983, "Mbacké": 236_372}
DEPT_SEX = {"Bambey": (95_485, 103_405), "Diourbel": (87_339, 96_644),
            "Mbacké": (107_842, 128_530)}
DEPT_SHORTFALL = {"Bambey": 268, "Diourbel": 491, "Mbacké": 628}   # DEPT_POP - Tableau 1.12 TOTAL

SPACES = dict.fromkeys([0x00A0, 0x2007, 0x2008, 0x2009, 0x202F, 0x205F], " ")


def norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def lines(doc, pno):
    out = []
    for x in doc.load_page(pno).get_text().splitlines():
        x = re.sub(r"\s+", " ", x.translate(SPACES)).strip()
        if x:
            out.append(x)
    return out


def fetch():
    import urllib.request

    os.makedirs(RAW, exist_ok=True)
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}
    for key, (name, size, dig, _pages) in FILES.items():
        path = os.path.join(RAW, name)
        if os.path.exists(path) and os.path.getsize(path) == size:
            print(f"  have {name}")
            continue
        req = urllib.request.Request(IREDA + name, headers=ua)
        with urllib.request.urlopen(req, timeout=600) as r:
            body = r.read()
        try:
            check_body(body, "pdf", where=name, pin_size=size, pin_digest=dig)
        except FetchCheckError as e:
            raise SystemExit(str(e))
        with open(path + ".part", "wb") as fh:
            fh.write(body)
        os.replace(path + ".part", path)
        print(f"  wrote {name} ({len(body):,} bytes)")


def _num(tok):
    """A Tableau 1.15 cell off the text layer: a float, '-' or None if not a cell."""
    t = tok.replace(" ", "")
    if t == "-":
        return "-"
    if re.fullmatch(r"[\dO]+(?:[,'][\dO])?", t):
        return float(t.replace("O", "0").replace("'", ".").replace(",", "."))
    return None


def read_t115(doc):
    ls = lines(doc, NAT_T115)
    start = next(i for i, x in enumerate(ls) if x.startswith("Tableau 1.15"))
    end = next(i for i, x in enumerate(ls) if x.startswith("1.8"))
    keys = {norm(k): k for k in T115} | NAME_ALIASES | {"ensemble": "Ensemble"}
    out, i = {}, start
    while i < end:
        name = keys.get(norm(ls[i]))
        if name is None:
            i += 1
            continue
        vals, j = [], i + 1
        while j < end and len(vals) < 9:
            v = _num(ls[j])
            if v is None:
                break
            vals.append(None if v == "-" else v)
            j += 1
        out[name] = tuple(vals)
        i = j
    return out


def read_t112(doc):
    ls = lines(doc, DIO_T112)
    start = next(i for i, x in enumerate(ls) if x.startswith("Tableau 1.12"))
    end = next(i for i, x in enumerate(ls) if x.startswith("1.8"))
    ints = []
    for x in ls[start + 1:end]:
        t = x.lstrip("-.·• ").replace(" ", "")
        if re.fullmatch(r"\d+", t):
            ints.append(int(t))
    return ints


def check(docs):
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    nat, dio, man = docs["national"], docs["diourbel"], docs["manuel"]
    print("Senegal — RGPH 1988, Tableau 1.15 and the Diourbel report's Tableau 1.12\n")
    for key, doc in docs.items():
        say(doc.page_count == FILES[key][3], f"{key}: {doc.page_count} pages "
            f"(expected {FILES[key][3]})")

    # ---- 1. the form and the manual
    form = norm(" ".join(lines(nat, NAT_FORM)))
    found = [w for w in ("religion", "khadr", "layen", "mouride", "tidiane", "catholique",
                         "chretien") if w in form]
    say(len(found) >= 5, f"the household form on PDF p75 carries P11's codes: {found}")
    p11 = norm(" ".join(lines(man, MAN_P11)))
    say(all(w in p11 for w in ("confrerie", "tidane", "animistes", "juifs")),
        "the manual's P11 puts animists and Jews under code 8, `autres` (p26)")

    # ---- 2. Tableau 1.15 off the page
    parsed = read_t115(nat)
    rows_ok = all(parsed.get(k) == v + (100.0,) for k, v in T115.items())
    say(rows_ok and len(parsed) == 11,
        f"Tableau 1.15 parsed off PDF p30: {len(parsed)} rows, every région identical to the "
        "transcription, printed `-` cells included")
    if not rows_ok:
        for k, v in T115.items():
            print(f"        {k}: page {parsed.get(k)} transcribed {v}")
    say(parsed.get("Ensemble") == T115_ENSEMBLE + (100.0,), f"Ensemble {parsed.get('Ensemble')}")

    z = lambda v: 0.0 if v is None else v      # noqa: E731
    worst_o = max(abs(sum(z(x) for x in row[1:6]) - row[0]) for row in T115.values())
    worst_t = max(abs(row[0] + row[6] + row[7] - 100) for row in T115.values())
    say(worst_o <= 0.25 + 1e-9, f"every row's five orders sum to Musulmans within {worst_o:.2f}")
    say(worst_t <= 0.15 + 1e-9, f"and Musulmans + Chrétiens + Autres to 100 within {worst_t:.2f}")

    # ---- 3. populations
    t11 = " ".join(lines(nat, NAT_T11)).replace(" ", "")
    t12 = " ".join(lines(nat, NAT_T12)).replace(" ", "")
    say(str(RESIDENT) in t11 and str(RESIDENT) in t12,
        f"Tableaux 1.1 and 1.2 print the resident population {RESIDENT:,}")
    say(all(str(v) in t12 for v in T12.values()), "all ten région populations are on p9")
    say(sum(T12.values()) == RESIDENT, "and they sum to it")

    # ---- 4. the Diourbel misprint
    t112 = read_t112(dio)
    flat =[v for r in T112_ROWS for v in T112[r][:3]] + [v for r in T112_ROWS for v in T112[r][3:]]
    say(t112 == flat, f"Diourbel Tableau 1.12 parsed off PDF p32: {len(t112)} counts identical "
        "to the transcription")
    reg = {r: T112[r][0] for r in T112_ROWS}
    dshare = {T112_AS_T115[r]: 100.0 * reg[r] / reg["TOTAL"] for r in T112_AS_T115}
    print("        Diourbel as the regional report counts it: "
          + ", ".join(f"{c} {v:.2f}" for c, v in dshare.items()))
    say(abs(dshare["Khadriya"] - T115["Diourbel"][2]) < 0.06 and T115["Diourbel"][1] is None,
        f"the national row prints Diourbel's Khadriya ({dshare['Khadriya']:.2f}) under Layène "
        "and `-` under Khadriya")
    corr_ok = all((v is None and dshare[c] < 0.05) or (v is not None and abs(v - dshare[c]) < 0.06)
                  for c, v in zip(COLS, DIOURBEL_CORRECTED))
    say(corr_ok, "with the two cells swapped, every cell of the row equals the regional report's "
        "share to the rounding")

    def weighted(rows, col):
        i = COLS.index(col)
        return sum(z(rows[k][i]) * T12[k] for k in rows) / RESIDENT

    fixed = dict(T115, Diourbel=DIOURBEL_CORRECTED)
    lay_fixed, lay_printed = weighted(fixed, "Layène"), weighted(T115, "Layène")
    say(abs(lay_fixed - T115_ENSEMBLE[2]) <= 0.05 < abs(lay_printed - T115_ENSEMBLE[2]),
        f"the table's own Ensemble Layène {T115_ENSEMBLE[2]} comes back only swapped "
        f"({lay_fixed:.2f} swapped, {lay_printed:.2f} as printed)")

    # ---- 5. the Ensemble row
    print("        population-weighted région rows against the printed Ensemble:")
    agrees = set()
    for i, c in enumerate(COLS):
        w = weighted(fixed, c)
        if abs(w - T115_ENSEMBLE[i]) <= 0.15:
            agrees.add(c)
        print(f"          {c:<12} {w:6.2f}  printed {T115_ENSEMBLE[i]:5.1f}  {w - T115_ENSEMBLE[i]:+.2f}")
    say(agrees == ENSEMBLE_AGREES, f"only {sorted(ENSEMBLE_AGREES)} agree within 0.15, as pinned "
        f"(got {sorted(agrees)}); the Ensemble row sums to {sum(T115_ENSEMBLE[:1] + T115_ENSEMBLE[6:]):.1f}")

    # ---- 6. Diourbel's table closes, and meets its population tables
    for c, col in enumerate(["Région", "Rural", "Urbain"] + DEPTS):
        v = {r: T112[r][c] for r in T112_ROWS}
        good = (sum(v[r] for r in T112_ROWS[:5]) == v["TOTAL MUSULMAN"]
                and v["CATHOLIQUE"] + v["AUTRES CHRETIENS"] == v["TOTAL CHRETIEN"]
                and v["TOTAL MUSULMAN"] + v["TOTAL CHRETIEN"] + v["AUTRE RELIGION"] == v["TOTAL"])
        say(good, f"Tableau 1.12 {col}: Muslim, Christian and grand totals close")
    say(all(T112[r][1] + T112[r][2] == T112[r][0] for r in T112_ROWS),
        "rural + urban = région on all 11 rows")
    say(T112["TOTAL"][0] == T12["Diourbel"], f"région total {T112['TOTAL'][0]:,} = national "
        "Tableau 1.2's Diourbel")
    say(all(m + f == DEPT_POP[d] for d, (m, f) in DEPT_SEX.items())
        and sum(DEPT_POP.values()) == T12["Diourbel"],
        "Diourbel Tableaux 1.2/1.2A: men + women = each département, and they sum to 619,245")
    t12d = " ".join(lines(dio, DIO_T12)).replace(" ", "")
    say(all(str(x) in t12d for x in (95_485, 103_405, 107_842, 128_530, 290_666, 328_579)),
        "the Bambey and Mbacké sex counts and the région's are on PDF p11 as transcribed")
    short = {d: DEPT_POP[d] - T112["TOTAL"][3 + i] for i, d in enumerate(DEPTS)}
    say(short == DEPT_SHORTFALL, f"Tableau 1.12's départements fall short of Tableau 1.2 by "
        f"{short} ({sum(short.values()):,}), as pinned")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def round_within(vals, total):
    """Largest-remainder rounding of {key: float} to integers summing to `total`."""
    fl = {k: int(v) for k, v in vals.items()}
    rest = total - sum(fl.values())
    for k in sorted(vals, key=lambda k: -(vals[k] - int(vals[k])))[:rest]:
        fl[k] += 1
    return fl


def rake_diourbel():
    seed = {(d, r): float(T112[r][3 + i]) for i, d in enumerate(DEPTS) for r in T112_LEAVES}
    col = {r: float(T112[r][0]) for r in T112_LEAVES}
    for _ in range(200):
        for d in DEPTS:
            s = sum(seed[(d, r)] for r in T112_LEAVES)
            for r in T112_LEAVES:
                seed[(d, r)] *= DEPT_POP[d] / s
        for r in T112_LEAVES:
            s = sum(seed[(d, r)] for d in DEPTS)
            for d in DEPTS:
                seed[(d, r)] *= col[r] / s
    err = max(abs(sum(seed[(d, r)] for r in T112_LEAVES) - DEPT_POP[d]) for d in DEPTS)
    if err > 0.5:
        raise SystemExit(f"Diourbel rake did not converge ({err:.3f})")
    return seed


def emit():
    rows = []
    for reg, printed in T115.items():
        if reg == "Diourbel":
            continue
        leaf = {c: printed[COLS.index(c)] for c in LEAVES}
        s = sum(v for v in leaf.values() if v is not None)
        raw = {c: v / s * T12[reg] for c, v in leaf.items() if v is not None}
        counts = round_within(raw, T12[reg])
        for c in LEAVES:
            if leaf[c] is None or counts[c] <= 0:
                continue
            rows.append({"geo_id": reg, "geo_level": "region", "geo_name": reg,
                         "source_category": c, "count": counts[c], "basis": BASIS, "year": YEAR,
                         "source_id": SOURCE_ID,
                         "note": f"Tableau 1.15 pct={leaf[c]:.1f}; row's leaves sum {s:.1f}; "
                                 f"Tableau 1.2 pop={T12[reg]}"})
    raked = rake_diourbel()
    print("\n  Diourbel's départements raked to Tableau 1.2 and Tableau 1.12's région column:")
    for i, d in enumerate(DEPTS):
        counts = round_within({r: raked[(d, r)] for r in T112_LEAVES}, DEPT_POP[d])
        for r in T112_LEAVES:
            if counts[r] <= 0:
                continue
            rows.append({"geo_id": d, "geo_level": "department", "geo_name": d,
                         "source_category": r, "count": counts[r], "basis": BASIS, "year": YEAR,
                         "source_id": SOURCE_ID,
                         "note": f"Diourbel report Tableau 1.12 {d} count={T112[r][3 + i]}; raked to "
                                 f"Tableau 1.2 pop={DEPT_POP[d]} and the région column"})
        print(f"    {d:<9} " + "  ".join(f"{r[:5]} {counts[r]:>7,}" for r in T112_LEAVES))
    return rows


def main():
    import fitz

    if "--fetch" in sys.argv:
        fetch()
    docs = {}
    for key, (name, size, dig, _pages) in FILES.items():
        path = os.path.join(RAW, name)
        if not os.path.exists(path):
            raise SystemExit(f"{path} missing — run: python sources/sn.py --fetch")
        with open(path, "rb") as fh:
            check_body(fh.read(), "pdf", where=name, pin_size=size, pin_digest=dig)
        docs[key] = fitz.open(path)
    check(docs)
    rows = emit()

    total = sum(r["count"] for r in rows)
    print(f"\n  12 units, {total:,} people ({total - RESIDENT:+,} against the resident population)")
    if total != RESIDENT:
        raise SystemExit("the drawn units do not sum to the resident population")
    by_cat, by_unit = {}, {}
    for r in rows:
        by_cat[r["source_category"]] = by_cat.get(r["source_category"], 0) + r["count"]
        by_unit.setdefault(r["geo_id"], {})[r["source_category"]] = r["count"]
    for c, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"    {n:>11,}  {100.0 * n / total:6.2f}%  {c}")
    for u, d in by_unit.items():
        s = sum(d.values())
        top = sorted(d.items(), key=lambda kv: -kv[1])[:4]
        print(f"  {u:<12} {s:>10,}  " + "  ".join(f"{c} {100 * n / s:.1f}" for c, n in top))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".part", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(OUT + ".part", OUT)
    print("\nwrote", OUT, f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
