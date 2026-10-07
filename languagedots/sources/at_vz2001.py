"""Austria: Statistik Austria, Volkszaehlung 2001, Umgangssprache (everyday language).

    python sources/at_vz2001.py --fetch    the ten "Hauptergebnisse I" volumes, if not on disk
                                           (read from religiondots/data/raw/at/ when there)
    python sources/at_vz2001.py            parse -> data/normalized/at.csv, at_place.csv

THE LAST LANGUAGE COUNT. Austria's 2011 and 2021 censuses are register-based and carry no
language; the 2001 census is the last that asked, and the Mikrozensus has no language item.

THE QUESTION. "Umgangssprache": the language (also several) usually spoken in private life,
family, relatives, friends. Foreign-language knowledge was not to be given, though Statistik
Austria notes some did. In these volumes a double answer is folded onto its non-German half:
"Slowenisch" includes "Deutsch und Slowenisch" (Hauptergebnisse I, Erlaeuterungen, 8). So
"Deutsch" is German ALONE, every other column is "this language, alone or with German", and the
columns partition the population.

THE TABLES (one volume per Land, plus the Oesterreich volume):
  Tabelle 5  Insgesamt, Deutsch, Burgenland-Kroatisch, Kroatisch, Romanes, Slowakisch,
             Slowenisch, Tschechisch, Ungarisch, Windisch, Sonstige. Two blocks: the whole
             population, then Austrian citizens. BY GEMEINDE ONLY IN BURGENLAND AND KAERNTEN,
             the two Laender with autochthonous minorities; in Wien by Gemeindebezirk; in the
             other six by Politischer Bezirk. That is the finest grain this census's language
             is published at as far as was found (sources/at.md: not in the archived Gemeinde
             profiles, nor on data.gv.at; STATcube was not searched).
  Tabelle 14 by Land: 45 languages (Tuerkisch, Serbisch, Bosnisch, Englisch, Polnisch...)
             by Insgesamt / Oesterreicher / dar. in Oesterreich geboren / Auslaender. Its
             languages outside Tabelle 5's columns are what Tabelle 5 lumps as Sonstige.
  Tabelle 2  by Gemeinde (Wien: Zaehlbezirk and Gemeindebezirk): citizenship (Tuerkei,
             Jugoslawien, Bosnien, Polen, ...). Never a count here, only a placement weight:
             the IPF seed below and countries/at.py's weighter, via at_place.csv.

SONSTIGE IS SPLIT, INSIDE EACH LAND. Sonstige is 8% of Austria and holds its second and third
largest languages (Tuerkisch 183k, Serbisch 177k). For each Land and each citizenship block
(Austrians, foreigners) an IPF finds the unit x language table whose rows are each unit's
printed Sonstige and whose columns are Tabelle 14's printed Land totals, starting from a seed
that is the unit's count of citizens of the countries the language is spoken in (PROXY below),
90%, plus 10% spread by the unit's Sonstige. Every Land total and every unit's Sonstige is the
census's; only the split of a unit's Sonstige among languages is borrowed from citizenship.
Those rows are `derived`.

THE GEOMETRY. Every row is "code, name, 11 figures" (Tabelle 2's right-hand page: 15 figures and
the code at the right). Figures are taken as the LAST n tokens of the row, `-` is an in-band
zero, and religiondots' traps apply: the 14 Statutarstaedte are printed only at the Bezirk tier
(their Gemeinde row is minted), and tables are bounded by their printed title, which
continuation pages repeat.

CHECKS (all equalities; the script stops on any failure):
  * Tabelle 5: the ten columns sum to Insgesamt on every row of both blocks; Gemeinden sum to
    Bezirk (Burgenland, Kaernten), Bezirke to Land; Austrians <= everyone in every cell;
  * Tabelle 14: members sum to their printed group rows and to Insgesamt; Oesterreicher +
    Auslaender = Insgesamt; Tabelle 5's named columns equal Tabelle 14's at Land level; Tabelle
    14's Sonstige members sum to Tabelle 5's Sonstige, for both blocks;
  * the nine Laender sum to the Oesterreich volume's Tabelle 14, every label, every column, and
    to the census total 8,032,926;
  * Tabelle 2: Insgesamt equals Tabelle 5's Insgesamt and Oesterreicher Tabelle 5's Austrian
    block, unit by unit; its citizenship groups sum to Auslaender; Gemeinden sum to Bezirk;
  * the IPF reproduces both margins to 0.01 of a person.
"""
import csv
import os
import re
import ssl
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "at"
RD_RAW = ROOT.parent / "religiondots" / "data" / "raw" / "at"   # the same PDFs, read only
OUT = ROOT / "data" / "normalized" / "at.csv"
PLACE = ROOT / "data" / "normalized" / "at_place.csv"   # Tabelle 2 by Gemeinde, for placement
SOURCE_ID = "at_vz2001_hauptergebnisse"
CENSUS_TOTAL = 8_032_926

BASE = ("https://www.statistik.at/fileadmin/publications/"
        "Volkszaehlung_2001__Hauptergebnisse_I_-_%s.pdf")
VOLUMES = [  # site slug, file, Land code, name
    ("Burgenland", "bgld", "1", "Burgenland"),
    ("Kaernten", "ktn", "2", "Kärnten"),
    ("Niederoesterreich", "noe", "3", "Niederösterreich"),
    ("Oberoesterreich", "ooe", "4", "Oberösterreich"),
    ("Salzburg", "sbg", "5", "Salzburg"),
    ("Steiermark", "stmk", "6", "Steiermark"),
    ("Tirol", "tirol", "7", "Tirol"),
    ("Vorarlberg", "vbg", "8", "Vorarlberg"),
    ("Wien", "wien", "9", "Wien"),
]
NATIONAL = ("OEsterreich", "oesterreich")

T5_COLS = ["Insgesamt", "Deutsch", "Burgenland-Kroatisch", "Kroatisch", "Romanes", "Slowakisch",
           "Slowenisch", "Tschechisch", "Ungarisch", "Windisch", "Sonstige"]
NAMED = T5_COLS[1:10]

# Tabelle 2, by printed column number (1-based). 4 is a percentage.
T2_LEFT = 11
T2_RIGHT = 15
C_TOT, C_AUT, C_FOR = 1, 2, 3
C_EU, C_DE, C_IT = 5, 6, 7
C_YU, C_BA, C_RS, C_HR, C_MK, C_SI = 8, 9, 10, 11, 12, 13
C_PL, C_RO, C_CH, C_CS, C_SK, C_CZ, C_TR, C_HU, C_US, C_OTH = 14, 15, 16, 17, 18, 19, 20, 21, 22, 23

# Tabelle 14's members of Sonstige -> the Tabelle 2 citizenships whose Gemeinde counts seed its
# split. "eu" is EU-15 less Germany and Italy (UK, France, Netherlands, Spain, ...).
PROXY = {
    "Bosnisch": [C_BA], "Serbisch": [C_RS], "Mazedonisch": [C_MK],
    "Albanisch": [C_RS, C_MK],          # Kosovo Albanians held Yugoslav passports in 2001
    "Türkisch": [C_TR], "Kurdisch": [C_TR],
    "Polnisch": [C_PL], "Rumänisch": [C_RO], "Italienisch": [C_IT],
    "Englisch": ["eu", C_US],
    "Französisch": ["eu"], "Spanisch": ["eu"], "Portugiesisch": ["eu"],
    "Holländisch/Flämisch": ["eu"], "Dänisch": ["eu"], "Schwedisch": ["eu"], "Finnisch": ["eu"],
    "Griechisch": ["eu"],
}
DEFAULT_PROXY = [C_OTH]                 # "anderer Staat; unbekannt"
# Placement only (countries/at.py), for Tabelle 5's own columns where the unit is a Bezirk:
# which Gemeinden of the Bezirk their dots go to. None = by population. Austrians' own minority
# languages (Burgenland Croatian, Romani, Windisch) have no citizenship to follow.
NAMED_PROXY = {
    "Deutsch": [C_AUT], "Kroatisch": [C_HR], "Slowakisch": [C_SK], "Tschechisch": [C_CZ],
    "Ungarisch": [C_HU], "Slowenisch": [C_SI],
    "Burgenland-Kroatisch": None, "Romanes": None, "Windisch": None,
}
FLOOR = 0.10                            # share of the seed spread by the Gemeinde's Sonstige

STATUTARSTAEDTE = {
    "101": "Eisenstadt", "102": "Rust", "201": "Klagenfurt", "202": "Villach",
    "301": "Krems an der Donau", "302": "Sankt Pölten", "303": "Waidhofen an der Ybbs",
    "304": "Wiener Neustadt", "401": "Linz", "402": "Steyr", "403": "Wels",
    "501": "Salzburg", "601": "Graz", "701": "Innsbruck",
}


def key(label):
    """Labels compared without spaces and with every non-ASCII letter as '?': some volumes'
    text layer returns U+FFFD for umlauts (religiondots/sources/at.py)."""
    s = re.sub(r"\s+", "", label)
    s = "".join(c if ord(c) < 128 else "?" for c in s)
    return ALIAS.get(s, s)


# Vorarlberg's volume prints "Andere Sprachen" for the row every other volume calls
# "Andere Sprachen, unbekannt" (2 people there)
ALIAS = {"AndereSprachen": "AndereSprachen,unbekannt"}


# ---------------------------------------------------------------- fetch

def pdf_path(code):
    for d in (RAW, RD_RAW):
        p = d / f"{code}.pdf"
        if p.exists() and p.stat().st_size > 100_000:
            return p
    return None


def fetch():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False          # statistik.at omits a TLS intermediate
    ctx.verify_mode = ssl.CERT_NONE
    RAW.mkdir(parents=True, exist_ok=True)
    for slug, code in [(v[0], v[1]) for v in VOLUMES] + [NATIONAL]:
        if pdf_path(code):
            print(f"  have {code:12s} {pdf_path(code)}")
            continue
        req = urllib.request.Request(BASE % slug, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=300) as r:
            body = r.read()
        if not body.startswith(b"%PDF"):
            raise SystemExit(f"{BASE % slug} did not return a PDF")
        tmp = RAW / f"{code}.pdf.part"
        tmp.write_bytes(body)
        os.replace(tmp, RAW / f"{code}.pdf")
        print(f"  got  {code:12s} {len(body):,}")


# ---------------------------------------------------------------- parse helpers

NUM = re.compile(r"^(\d+|-)$")


def val(tok):
    return 0 if tok == "-" else int(tok)


def page_titles(doc):
    out = {}
    for i in range(doc.page_count):
        ws = doc[i].get_text("words")
        for w in ws:
            if w[4] == "Tabelle" and w[1] < 75 and w[0] < 45:
                nxt = sorted([v for v in ws if abs(v[1] - w[1]) < 2 and w[2] < v[0] < w[2] + 12],
                             key=lambda v: v[0])
                m = re.match(r"^(\d+):$", nxt[0][4]) if nxt else None
                if m:
                    out[i] = int(m.group(1))
                break
    return out


def table_pages(doc, n, titles):
    starts = [i for i, t in titles.items() if t == n]
    if not starts:
        raise SystemExit(f"no page titled 'Tabelle {n}:'")
    start = min(starts)
    later = [i for i in titles if i > start and titles[i] != n]
    return list(range(start, min(later) if later else doc.page_count))


def rows_of(page, y_min=105):
    rows = defaultdict(list)
    for w in page.get_text("words"):
        if w[1] < y_min:
            continue
        rows[round(w[1] / 2.0)].append(w)
    return [sorted(rows[k], key=lambda w: w[0]) for k in sorted(rows)]


def coded_rows(page, n_vals):
    """Left-page rows: code at the far left, a name, then n_vals figures."""
    out = []
    for ws in rows_of(page):
        if not (ws[0][2] < 62 and re.fullmatch(r"\d+", ws[0][4])):
            continue
        toks = [w[4] for w in ws]
        code, first = toks[0], 1
        if len(ws) > 1 and ws[1][2] < 62 and re.fullmatch(r"\d\d", toks[1]):
            code, first = toks[0] + toks[1], 2          # Wien's Zaehlbezirk: "901 01"
        if len(toks) < n_vals + first + 1:
            continue
        vals = toks[-n_vals:]
        if not all(NUM.match(t) or re.fullmatch(r"\d+,\d", t) for t in vals):
            continue
        out.append((code, " ".join(toks[first:-n_vals]), vals, ws[0][1]))
    return out


# ---------------------------------------------------------------- Tabelle 5

def parse_t5(doc, titles, vol):
    """{block: {code: (name, [11 values])}} for blocks 'all' and 'aut'."""
    out = {"all": {}, "aut": {}}
    block = None
    for i in table_pages(doc, 5, titles):
        page = doc[i]
        # the block labels sit centred above each block: "Bevölkerung" / "Österreicher"
        marks = sorted((w[1], "aut" if w[4].startswith("sterreicher", 1) else "all")
                       for w in page.get_text("words")
                       if 300 < w[0] < 420 and w[1] > 105 and len(w[4]) > 8
                       and (key(w[4]) in ("Bev?lkerung", "?sterreicher")))
        for code, name, vals, y in coded_rows(page, 11):
            for my, b in marks:
                if my < y:
                    block = b
            if block is None:
                raise SystemExit(f"{vol} Tabelle 5 p{i}: a row before any block label")
            if code in out[block]:
                raise SystemExit(f"{vol} Tabelle 5: {code} twice in block {block}")
            out[block][code] = (name, [val(v) for v in vals])
    for b in out:
        if not out[b]:
            raise SystemExit(f"{vol} Tabelle 5: block {b} empty")
    return out


# ---------------------------------------------------------------- Tabelle 2

def parse_t2(doc, titles, vol):
    """{code: [26 values, index 1-based in a dict]} joining the left and right pages."""
    left, right = {}, {}
    for i in table_pages(doc, 2, titles):
        page = doc[i]
        ws = page.get_text("words")
        is_right = any(w[4].startswith("Kenn") and w[0] > 500 for w in ws)
        if not is_right:
            for code, name, vals, _ in coded_rows(page, T2_LEFT):
                if code in left:
                    raise SystemExit(f"{vol} Tabelle 2: {code} twice")
                left[code] = (name, vals)
        else:
            for row in rows_of(page):
                toks = [w[4] for w in row]
                if row[-1][2] < 550 or not re.fullmatch(r"\d+", toks[-1]):
                    continue
                code, vals = toks[-1], toks[:-1]
                if len(row) > 1 and row[-2][0] > 530 and re.fullmatch(r"\d+", toks[-2]):
                    code, vals = toks[-2] + toks[-1], toks[:-2]     # Wien's "901 01"
                if len(vals) != T2_RIGHT or not all(NUM.match(t) for t in vals):
                    continue
                if code in right:
                    raise SystemExit(f"{vol} Tabelle 2 (right): {code} twice")
                right[code] = vals
    if set(left) != set(right):
        raise SystemExit(f"{vol} Tabelle 2: left and right pages disagree on codes: "
                         f"{sorted(set(left) ^ set(right))[:8]}")
    out = {}
    for code, (name, lv) in left.items():
        v = {}
        for j, t in enumerate(lv, start=1):
            v[j] = None if j == 4 else val(t)
        for j, t in enumerate(right[code], start=T2_LEFT + 1):
            v[j] = val(t)
        out[code] = v
    return out


def check_t2(t2, vol):
    for code, v in t2.items():
        if v[C_AUT] + v[C_FOR] != v[C_TOT]:
            raise SystemExit(f"{vol} T2 {code}: Österreicher + Ausländer != Insgesamt")
        groups = (v[C_EU] + v[C_YU] + v[C_PL] + v[C_RO] + v[C_CH] + v[C_CS] + v[C_TR] + v[C_HU]
                  + v[C_US] + v[C_OTH])
        if groups != v[C_FOR]:
            raise SystemExit(f"{vol} T2 {code}: citizenship groups sum to {groups}, "
                             f"Ausländer {v[C_FOR]}")
        if v[C_BA] + v[C_RS] + v[C_HR] + v[C_MK] + v[C_SI] != v[C_YU]:
            raise SystemExit(f"{vol} T2 {code}: ex-Yugoslav states != their group")
        if v[C_SK] + v[C_CZ] != v[C_CS] or v[C_DE] + v[C_IT] > v[C_EU]:
            raise SystemExit(f"{vol} T2 {code}: a group does not hold its members")
        if v[24] + v[25] + v[26] != v[C_FOR]:
            raise SystemExit(f"{vol} T2 {code}: Ausländer by Umgangssprache != Ausländer")


# ---------------------------------------------------------------- Tabelle 14

def parse_t14(doc, titles, vol):
    """[(label, indented, [tot, aut, aut_born_here, for])] in print order."""
    out = []
    for i in table_pages(doc, 14, titles):
        for ws in rows_of(doc[i], y_min=120):
            toks = [w[4] for w in ws]
            nums = [w for w in ws if w[0] > 240 and NUM.match(w[4])]
            lab = [w for w in ws if w[0] < 240]
            if len(nums) != 4 or not lab:
                continue
            label = " ".join(w[4] for w in lab)
            out.append((label, lab[0][0] > 42, [val(w[4]) for w in nums]))
    if not out or key(out[0][0]) != "Insgesamt":
        raise SystemExit(f"{vol} Tabelle 14: first row is not Insgesamt: {out[:1]}")
    return out


def t14_leaves(rows, vol):
    """Leaves of Tabelle 14 with the group subtotals checked and dropped."""
    tot = rows[0][2]
    leaves, i = [], 1
    while i < len(rows):
        label, ind, v = rows[i]
        members = []
        j = i + 1
        while j < len(rows) and rows[j][1]:
            members.append(rows[j])
            j += 1
        if members:                                     # a printed group and its members
            s = [sum(m[2][k] for m in members) for k in range(4)]
            if s != v:
                raise SystemExit(f"{vol} T14 group {label!r}: members {s}, printed {v}")
            leaves.extend((m[0], m[2]) for m in members)
        else:
            leaves.append((label, v))
        i = j
    s = [sum(v[k] for _, v in leaves) for k in range(4)]
    if s != tot:
        raise SystemExit(f"{vol} T14: leaves sum to {s}, Insgesamt {tot}")
    for label, v in leaves:
        if v[1] + v[3] != v[0] or v[2] > v[1]:
            raise SystemExit(f"{vol} T14 {label!r}: citizenship columns do not add up: {v}")
    return leaves


# ---------------------------------------------------------------- nesting

def add_statutarstaedte(rows, vol):
    have5 = {c for c in rows if len(c) == 5}
    minted = []
    for b in [c for c in rows if len(c) == 3]:
        if any(g.startswith(b) for g in have5):
            continue
        if b not in STATUTARSTAEDTE:
            raise SystemExit(f"{vol}: Bezirk {b} has no Gemeinden and is not a Statutarstadt")
        rows[b + "01"] = (STATUTARSTAEDTE[b], rows[b][1])
        minted.append(b + "01")
    return minted


def nest_check(rows, vol, wien):
    fine, mid = (3, 1) if wien else (5, 3)
    pairs = [(fine, mid)] + ([] if wien else [(3, 1)])
    for c_len, p_len in pairs:
        agg = defaultdict(lambda: None)
        for c, (_, v) in rows.items():
            if len(c) != c_len:
                continue
            p = c[:p_len]
            agg[p] = v if agg[p] is None else [a + b for a, b in zip(agg[p], v)]
        for p, (name, v) in rows.items():
            if len(p) == p_len and agg[p] != v:
                raise SystemExit(f"{vol}: {p} {name!r} children sum to {agg[p]}, printed {v}")


# ---------------------------------------------------------------- IPF

def ipf(seed, rows, cols, tol=0.01, it=5000):
    """seed: {(g, k): w}; rows: {g: r}; cols: {k: c}. Returns {(g, k): x}."""
    x = {gk: w for gk, w in seed.items() if w > 0 and rows[gk[0]] > 0 and cols[gk[1]] > 0}
    for n in range(it):
        rs = defaultdict(float)
        for (g, k), v in x.items():
            rs[g] += v
        x = {(g, k): v * rows[g] / rs[g] for (g, k), v in x.items()}
        cs = defaultdict(float)
        for (g, k), v in x.items():
            cs[k] += v
        x = {(g, k): v * cols[k] / cs[k] for (g, k), v in x.items()}
        rs = defaultdict(float)
        for (g, k), v in x.items():
            rs[g] += v
        err = max([abs(rs[g] - r) for g, r in rows.items()] + [0])
        if err < tol:
            return x, n + 1
    raise SystemExit(f"IPF did not converge: row error {err:.3f}")


# ---------------------------------------------------------------- main

GEM_LEVEL = {"1", "2"}     # the only Laender whose Tabelle 5 is by Gemeinde


def proxy_value(v, cols):
    s = 0
    for p in cols:
        s += (v[C_EU] - v[C_DE] - v[C_IT]) if p == "eu" else v[p]
    return s


def proxy_for(label):
    k = key(label)
    for p, cols in PROXY.items():
        if key(p) == k:
            return cols
    return DEFAULT_PROXY


def t2_nest(t2, vol):
    cols = [j for j in range(1, 27) if j != 4]
    for c_len, p_len in ((5, 3), (3, 1)):
        agg = defaultdict(lambda: [0] * len(cols))
        for c, v in t2.items():
            if len(c) == c_len:
                agg[c[:p_len]] = [a + v[j] for a, j in zip(agg[c[:p_len]], cols)]
        for p, v in t2.items():
            if len(p) == p_len and agg[p] != [v[j] for j in cols]:
                raise SystemExit(f"{vol} T2: {p}'s children do not sum to it")


def build():
    per_land = {}
    print("parsing:")
    for slug, vol, bl, land in VOLUMES:
        path = pdf_path(vol)
        if not path:
            raise SystemExit(f"missing {vol}.pdf: run with --fetch")
        doc = fitz.open(path)
        titles = page_titles(doc)
        gem = bl in GEM_LEVEL
        t5 = parse_t5(doc, titles, vol)
        t2 = parse_t2(doc, titles, vol)
        t14 = t14_leaves(parse_t14(doc, titles, vol), vol)
        doc.close()

        # Tabelle 5
        minted = []
        for b in ("all", "aut"):
            if gem:
                minted = add_statutarstaedte(t5[b], vol)
            for c, (name, v) in t5[b].items():
                if sum(v[1:]) != v[0]:
                    raise SystemExit(f"{vol} T5 {b} {c} {name!r}: columns sum {sum(v[1:])}, "
                                     f"Insgesamt {v[0]}")
            nest_check(t5[b], vol, not gem)
        if set(t5["all"]) != set(t5["aut"]):
            raise SystemExit(f"{vol} T5: the two blocks have different units")
        for c in t5["all"]:
            if any(a < b for a, b in zip(t5["all"][c][1], t5["aut"][c][1])):
                raise SystemExit(f"{vol} T5 {c}: Austrians exceed everyone in a cell")
        units = sorted(c for c in t5["all"] if len(c) == (5 if gem else 3))

        # Tabelle 2: Statutarstaedte get their Gemeinde row, then everything nests
        if bl != "9":
            for b in [c for c in t2 if len(c) == 3]:
                if not any(len(g) == 5 and g.startswith(b) for g in t2):
                    if b not in STATUTARSTAEDTE:
                        raise SystemExit(f"{vol} T2: Bezirk {b} has no Gemeinden")
                    t2[b + "01"] = t2[b]
        check_t2(t2, vol)
        t2_nest(t2, vol)
        for c in units + [bl]:
            if c not in t2:
                raise SystemExit(f"{vol}: {c} is in Tabelle 5 and not in Tabelle 2")
            if t2[c][C_TOT] != t5["all"][c][1][0] or t2[c][C_AUT] != t5["aut"][c][1][0]:
                raise SystemExit(f"{vol} {c}: Tabelle 2 ({t2[c][C_TOT]}, {t2[c][C_AUT]}) and "
                                 f"Tabelle 5 ({t5['all'][c][1][0]}, {t5['aut'][c][1][0]}) differ")

        # Tabelle 14 against Tabelle 5 at Land level
        lv_all, lv_aut = t5["all"][bl][1], t5["aut"][bl][1]
        t5keys = [key(c) for c in T5_COLS]
        k14 = {key(l) for l, _ in t14}
        son_all = son_aut = 0
        for l, v in t14:
            if key(l) in t5keys[1:10]:
                col = t5keys.index(key(l))
                if (v[0], v[1]) != (lv_all[col], lv_aut[col]):
                    raise SystemExit(f"{vol} {l}: T14 {v[:2]}, T5 {lv_all[col]}, {lv_aut[col]}")
            else:
                son_all += v[0]
                son_aut += v[1]
        for n in NAMED:
            col = T5_COLS.index(n)
            if key(n) not in k14 and (lv_all[col] or lv_aut[col]):
                raise SystemExit(f"{vol}: T5 prints {n} {lv_all[col]} and T14 has no row")
        if (son_all, son_aut) != (lv_all[10], lv_aut[10]):
            raise SystemExit(f"{vol}: T14's other languages sum to {son_all}/{son_aut}, T5's "
                             f"Sonstige {lv_all[10]}/{lv_aut[10]}")
        per_land[bl] = dict(vol=vol, land=land, t5=t5, t2=t2, t14=t14, units=units,
                            minted=minted, gem=gem)
        print(f"  {vol:6s} {len(units):4d} {'Gemeinden' if gem else 'Bezirke  '} "
              f"{lv_all[0]:>9,} people, Sonstige {lv_all[10]:>7,} ({lv_all[10] / lv_all[0]:.1%}), "
              f"{len(t14)} T14 leaves, T2 {sum(len(c) == 5 for c in t2)} Gemeinden"
              + (f", +{len(minted)} Statutarstadt" if minted else ""))

    # national volume: Tabelle 14 for the whole country
    doc = fitz.open(pdf_path(NATIONAL[1]))
    nat = t14_leaves(parse_t14(doc, page_titles(doc), "oesterreich"), "oesterreich")
    doc.close()
    agg = defaultdict(lambda: [0, 0, 0, 0])
    for bl, d in per_land.items():
        for l, v in d["t14"]:
            agg[key(l)] = [a + b for a, b in zip(agg[key(l)], v)]
    natk = {key(l): v for l, v in nat}
    natlab = {key(l): l for l, v in nat}      # labels as the Oesterreich volume prints them
    if set(natk) != set(agg):
        raise SystemExit(f"labels differ between the Laender and Österreich: "
                         f"{sorted(set(natk) ^ set(agg))}")
    for k, v in natk.items():
        if agg[k] != v:
            raise SystemExit(f"{k}: Laender sum {agg[k]}, Österreich volume {v}")
    total = sum(v[0] for v in natk.values())
    if total != CENSUS_TOTAL:
        raise SystemExit(f"Österreich T14 sums to {total:,}, census {CENSUS_TOTAL:,}")
    print(f"checks: Tabellen 2, 5 and 14 agree in every Land; the nine Laender's Tabelle 14 "
          f"equal the Österreich volume's for all {len(natk)} labels x 4 columns; total {total:,}")

    # ---- write
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nrow = 0
    worst = 0.0
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "geo_level", "geo_name", "source_category", "count", "tier",
                    "year", "source_id"])
        for bl, d in sorted(per_land.items()):
            t5, t2, units = d["t5"], d["t2"], d["units"]
            level = "gemeinde" if d["gem"] else ("gemeindebezirk" if bl == "9" else "bezirk")
            for l, v in d["t14"]:
                w.writerow([f"AT{bl}", "land", d["land"], natlab[key(l)], v[0], "measured", 2001,
                            SOURCE_ID])
                nrow += 1
            for c in units:
                name, v = t5["all"][c]
                for j, lab in enumerate(T5_COLS[1:10], start=1):
                    if v[j]:
                        w.writerow([f"AT{c}", level, name, lab, v[j], "measured", 2001,
                                    SOURCE_ID])
                        nrow += 1
            # Sonstige: Tabelle 14's members, split by IPF per citizenship block
            t5keys = {key(n) for n in T5_COLS[:10]}
            members = [(l, v) for l, v in d["t14"] if key(l) not in t5keys]
            cells = defaultdict(float)
            for b, colidx in (("aut", 1), ("for", 3)):
                if b == "aut":
                    rows = {c: t5["aut"][c][1][10] for c in units}
                else:
                    rows = {c: t5["all"][c][1][10] - t5["aut"][c][1][10] for c in units}
                cols = {l: v[colidx] for l, v in members}
                R = sum(rows.values())
                if R != sum(cols.values()):
                    raise SystemExit(f"{d['vol']} {b}: Sonstige {R} != members")
                if R == 0:
                    continue
                seed = {}
                for l, _ in members:
                    prox = proxy_for(l)
                    pv = {c: proxy_value(t2[c], prox) for c in units}
                    P = sum(pv.values())
                    for c in units:
                        seed[(c, l)] = (1 - FLOOR) * (pv[c] / P if P else 0.0) + FLOOR * rows[c] / R
                x, _ = ipf(seed, rows, cols)
                cs, rs = defaultdict(float), defaultdict(float)
                for (c, l), v in x.items():
                    cs[l] += v
                    rs[c] += v
                    cells[(c, l)] += v
                worst = max([worst] + [abs(cs[l] - v) for l, v in cols.items()]
                            + [abs(rs[c] - v) for c, v in rows.items()])
            for (c, l), v in sorted(cells.items()):
                if v > 0:
                    w.writerow([f"AT{c}", level, t5["all"][c][0], natlab[key(l)], round(v, 4), "derived",
                                2001, SOURCE_ID])
                    nrow += 1
    if worst > 0.02:
        raise SystemExit(f"IPF margins off by {worst:.3f}")
    print(f"wrote {OUT.relative_to(ROOT)}: {nrow:,} rows (IPF margins within {worst:.4f})")

    # ---- placement proxies: every Gemeinde's (Wien: Gemeindebezirk's) Tabelle 2 row
    with open(PLACE, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        cols = [j for j in range(1, 27) if j != 4]
        w.writerow(["geo_id", "unit"] + [f"c{j}" for j in cols] + ["eu"])
        n = 0
        for bl, d in sorted(per_land.items()):
            t2 = d["t2"]
            fine = [c for c in t2 if len(c) == (3 if bl == "9" else 5)]
            for c in sorted(fine):
                unit = c if (d["gem"] or bl == "9") else c[:3]
                v = t2[c]
                w.writerow([f"AT{c}", f"AT{unit}"] + [v[j] for j in cols]
                           + [v[C_EU] - v[C_DE] - v[C_IT]])
                n += 1
    print(f"wrote {PLACE.relative_to(ROOT)}: {n:,} Gemeinden and Wien districts")

    print("\nnational, 2001 (Tabelle 14, Österreich volume):")
    for l, v in sorted(nat, key=lambda t: -t[1][0])[:25]:
        print(f"  {l:40s} {v[0]:>10,}  {v[0] / CENSUS_TOTAL:6.2%}   foreigners {v[3]:>8,}")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    build()
