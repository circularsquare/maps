"""Mozambique — INE, IV RGPH 2017, religion by province, from Quadro 11 of each province's tables.

Reads (or fetches) data/raw/mz/ and writes data/normalized/mz.csv.

**Eight answers on eleven provinces for 26.9 million people**, about 2.45 million a unit. The
list is short and it carries the one category that makes Mozambique worth having:
**`Zione/Sião`, the Zionist churches, 4,199,108 people and 15.6% of the country**, counted as a
cell of its own by a census and published by province.

## WHERE THE TABLES ARE, AND WHY THEY COME FROM THE WAYBACK MACHINE

INE published the 2017 definitive results in 2019 as one set of xlsx tables per province on
its old Plone site, `www.ine.gov.mz/iv-rgph-2017/<province>/`, and Quadro 11 (religion by
residence, age and sex) is one of them in all eleven sets. That site was replaced by a
Liferay portal in 2022 and **every `/iv-rgph-2017/` path now returns 404**. The new portal
still has a `Censo 2017` library with one folder per province, but its folder pages render
no document list to a script, and both of Liferay's APIs (`/o/headless-delivery/` and
`/api/jsonws/`) answer 403 to a guest. The Wayback Machine captured all twelve files on
2019-11-14, as `200 application/vnd...spreadsheetml.sheet`, and the CDX query that finds
them is in `sources/mz.md`. They are fetched from there with the `id_` flag, which returns
the archived bytes untouched.

`www.ine.gov.mz` itself is not walled. Port 80 refuses connections and HTTPS fails
certificate verification, because the server does not send its intermediate certificate;
with verification off it answers normally. None of this module needs it.

## WHAT IS NOT PUBLISHED, SO NOBODY LOOKS AGAIN

No religion table below province exists in INE's release. `LISTA DE QUADROS.xlsx` on
`mozdata.ine.gov.mz` lists the census tables and only Quadros 3, 6, 7 and 8 are by district
(sources.md §11w). The PX-Web database the new portal links to, `41.94.86.11/Censo2017/`,
refuses connections from here. The microdata is `licensed` on mozdata.

## THE PROVINCIAL FILES HAVE TWO TOTAL COLUMNS AND THE NATIONAL FILE HAS ONE

A provincial Quadro 11 prints `TOTAL` twice. The first is everybody; the second is everybody
except `DESCONHEC.`, i.e. the sum of the seven answers that name something. Cabo Delgado:
2,267,715 and 2,254,229, a difference of 13,486, which is its `DESCONHEC.` cell. The
national file prints only the first. Both properties are asserted, and the columns are found
by their header text and never by position, because the extra column shifts every religion
one place to the right in the provincial files.

## THE UNSD FIGURES FOR THE SAME CENSUS ARE A DIFFERENT EDIT

UNSD's Demographic Yearbook table 28 (`tools/oracle.py Mozambique`) has the same national
total to the person, 26,899,105, and a different split: Zione/Sião **4,389,294** against
this table's 4,199,108, Evangélica/Pentecostal 4,481,176 against 4,124,710, and
`Unknown` **317,550** against 674,761. The UNSD version has roughly half the unknowns and
puts the difference mostly into the two largest Protestant cells, which is what an
imputation of the non-response would look like. Nobody publishes that version by province,
so it cannot be drawn. `check()` does not fail on the disagreement; it pins it
(`UNSD_DIFF_2017`), so a re-issued INE file or an edited UNSD row fails instead of moving it.

Usage:
    python sources/mz.py --fetch     12 xlsx from the Wayback Machine, ~290 KB
    python sources/mz.py             normalise from data/raw/mz/
"""

import csv
import os
import sys
import time
import unicodedata
import re

from fetch_checks import pinned_differences   # shared, sources/fetch_checks.py

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "mz")
OUT = os.path.join(ROOT, "data", "normalized", "mz.csv")

SOURCE_ID = "mz_rgph_2017"
YEAR = 2017
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

WAYBACK = "http://web.archive.org/web/{ts}id_/http://www.ine.gov.mz/iv-rgph-2017/{path}"

Q11 = "quadro-11-populacao-por-religiao-segundo-"
# (INE province code, INE's name, capture timestamp, path under /iv-rgph-2017/, what the
# sheet's own title must contain once folded). The codes are INE's standard order, north to
# south, and are what `geo_id` is built from. Gaza's file name is truncated on INE's own
# server; the capture is of that truncated name.
PROVINCES = [
    ("01", "Niassa", "20191114002733",
     "niassa/" + Q11 + "area-de-residencia-idade-e-sexo-provincia-do-niassa-2017.xlsx",
     ("niassa",)),
    ("02", "Cabo Delgado", "20191114004834",
     "cabo-delgado/" + Q11 + "area-de-residencia-idade-e-sexo-provincia-de-cabo-delgado-2017.xlsx",
     ("cabodelgado",)),
    ("03", "Nampula", "20191114011052",
     "nampula/" + Q11 + "area-de-residencia-idade-e-sexo-provincia-de-nampula-2017.xlsx",
     ("nampula",)),
    ("04", "Zambézia", "20191114013455",
     "zambezia/" + Q11 + "area-de-residencia-idade-e-sexo-provincia-de-zambezia-2017.xlsx",
     ("zambezia",)),
    ("05", "Tete", "20191114015723",
     "tete/" + Q11 + "area-de-residencia-idade-e-sexo-provincia-de-tete-2017.xlsx",
     ("tete",)),
    ("06", "Manica", "20191114023623",
     "manica/" + Q11 + "area-de-residencia-idade-e-sexo-provincia-de-manica-2017.xlsx",
     ("manica",)),
    ("07", "Sofala", "20191114025815",
     "sofala/" + Q11 + "area-de-residencia-idade-e-sexo-provincia-de-sofala-2017.xlsx",
     ("sofala",)),
    ("08", "Inhambane", "20191114031616",
     "inhambane/" + Q11 + "area-de-residencia-idade-e-sexo-provincia-de-inhambane-2017.xlsx",
     ("inhambane",)),
    ("09", "Gaza", "20191114035423",
     "gaza/" + Q11 + "area-de-residencia-idade-e-s.xlsx",
     ("gaza",)),
    ("10", "Maputo Província", "20191114043021",
     "maputo-provincia/" + Q11 + "area-de-residencia-idade-e-sexo-maputo-provincia-2017.xlsx",
     ("maputoprovincia", "provinciademaputo")),
    ("11", "Maputo Cidade", "20191114050516",
     "maputo-cidade/" + Q11 + "idade-e-sexo-maputo-cidade-2017.xlsx",
     ("maputocidade", "cidadedemaputo")),
]
NATIONAL = ("00", "Moçambique", "20191114164800",
            "mocambique/03-religiao/" + Q11 + "area-de-residencia-idade-e-sexo-mocambique-2017.xlsx",
            ("mocambique",))

# INE's eight columns, in the order it prints them, spelled as INE spells them. The printed
# headers wrap over two rows (`EVANGÉLICA/` over `PENTECOSTAL`, `SEM` over `RELIGIÃO`) and
# abbreviate the last, so the canonical names are INE's words unwrapped.
CATEGORIES = ["Católica", "Anglicana", "Islâmica", "Zione/Sião", "Evangélica/Pentecostal",
              "Sem religião", "Outra", "Desconhecida"]
STEMS = {"Católica": "catolica", "Anglicana": "anglicana", "Islâmica": "islamica",
         "Zione/Sião": "zione", "Evangélica/Pentecostal": "evangelica",
         "Sem religião": "sem", "Outra": "outra", "Desconhecida": "desconhec"}
assert set(STEMS) == set(CATEGORIES)
UNIVERSE = "Total"

CENSUS_TOTAL = 26_899_105

# THE PROVINCES ADD UP TO THE NATIONAL TOTAL TO THE PERSON AND NOT QUITE CATEGORY BY CATEGORY.
# Summed over the eleven provincial Quadro 11s, Católica is 29 short of the national Quadro
# 11, Zione/Sião 25 short, Desconhecida 81 short and Outra 150 over; the other four are
# within 7. The totals are identical, so people moved between answers between the two
# tabulations, which is INE's and not a parse error (a misread column would be off by
# hundreds of thousands). The provincial files are what is drawn.
SLACK = 200

# UNSD Demographic Yearbook table 28, Mozambique 2017, Total, via tools/oracle.py. Printed
# beside this table's national row; see the module docstring for why they differ.
UNSD_2017 = {"Católica": 7_344_788, "Anglicana": 461_714, "Islâmica": 5_131_177,
             "Zione/Sião": 4_389_294, "Evangélica/Pentecostal": 4_481_176,
             "Sem religião": 3_625_355, "Outra": 1_148_051, "Desconhecida": 317_550}
assert sum(UNSD_2017.values()) == CENSUS_TOTAL
# The disagreement itself, national Quadro 11 minus UNSD_2017, recorded 2026-09-14 and asserted
# by check(). It sums to zero, as two edits of one total must.
UNSD_DIFF_2017 = {"Católica": -31_212, "Anglicana": -3_998, "Islâmica": -37_153,
                  "Zione/Sião": -190_186, "Evangélica/Pentecostal": -356_466,
                  "Sem religião": 111_999, "Outra": 149_805, "Desconhecida": 357_211}
assert sum(UNSD_DIFF_2017.values()) == 0


def fold(s):
    s = unicodedata.normalize("NFKD", "" if s is None else str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def local_name(code):
    return f"mz_q11_{code}.xlsx"


# ---------------------------------------------------------------- fetch


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for code, name, ts, path, _ in [NATIONAL] + PROVINCES:
        dest = os.path.join(RAW, local_name(code))
        if os.path.exists(dest) and _is_xlsx(dest):
            print(f"have {name}")
            continue
        url = WAYBACK.format(ts=ts, path=path)
        # The Wayback Machine answers 503 and 504 in bursts (it did to the CDX query five
        # times out of seven while this was written), so retry with a pause rather than
        # treating one refusal as the file being gone.
        for attempt in range(8):
            try:
                r = requests.get(url, timeout=180,
                                 headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            except requests.RequestException as e:
                print(f"  {name}: {type(e).__name__}, retrying")
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code == 200:
                break
            print(f"  {name}: HTTP {r.status_code}, retrying")
            time.sleep(5 * (attempt + 1))
        else:
            raise SystemExit(f"{name}: the Wayback Machine did not return {url}")
        # §5a: a 200 is not a download.
        if r.content[:4] != b"PK\x03\x04":
            raise SystemExit(f"{name}: not an xlsx -- starts {r.content[:16]!r}")
        with open(dest + ".part", "wb") as fh:
            fh.write(r.content)
        os.replace(dest + ".part", dest)
        print(f"got  {name}: {len(r.content):,} bytes")


def _is_xlsx(path):
    with open(path, "rb") as fh:
        return fh.read(4) == b"PK\x03\x04"


# ---------------------------------------------------------------- read


def read_one(code, name, title_keys):
    """One Quadro 11 -> (universe, {category: count}) for the whole unit, both sexes, all ages."""
    import openpyxl

    path = os.path.join(RAW, local_name(code))
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run with --fetch first")
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = wb.worksheets[0]
    rows = [tuple(r) for r in ws.iter_rows(values_only=True)]
    wb.close()

    title = fold(" ".join(str(c) for r in rows[:3] for c in r if c is not None))
    if not title.startswith("quadro11populacaoporreligiao"):
        raise SystemExit(f"{name}: sheet title is not Quadro 11's: {title[:80]!r}")
    if not any(k in title for k in title_keys):
        raise SystemExit(f"{name}: sheet title names another place: {title!r}")

    hdr_i = next((i for i, r in enumerate(rows[:12])
                  if any(fold(c) == "catolica" for c in r)), None)
    if hdr_i is None:
        raise SystemExit(f"{name}: no header row with CATÓLICA in the first 12 rows")
    hdr = rows[hdr_i]
    cols = {}
    for j, c in enumerate(hdr):
        f = fold(c)
        hits = [cat for cat, s in STEMS.items() if f.startswith(s)]
        if len(hits) > 1:
            raise SystemExit(f"{name}: header {c!r} matches {hits}")
        if hits:
            if hits[0] in cols:
                raise SystemExit(f"{name}: two columns headed {hits[0]}")
            cols[hits[0]] = j
    if set(cols) != set(CATEGORIES):
        raise SystemExit(f"{name}: header has {sorted(cols)}, missing "
                         f"{sorted(set(CATEGORIES) - set(cols))}")
    totals = [j for j, c in enumerate(hdr) if fold(c) == "total"]
    if len(totals) not in (1, 2) or totals[0] > min(cols.values()):
        raise SystemExit(f"{name}: TOTAL columns at {totals}, religions start at "
                         f"{min(cols.values())}")

    # The first `T O T A L` row under the header is both sexes, all ages, all residence.
    data_i = next((i for i in range(hdr_i + 1, len(rows)) if fold(rows[i][0]) == "total"),
                  None)
    if data_i is None:
        raise SystemExit(f"{name}: no T O T A L row under the header")
    row = rows[data_i]
    universe = int(row[totals[0]])
    vals = {cat: int(row[j]) for cat, j in cols.items()}
    if sum(vals.values()) != universe:
        raise SystemExit(f"{name}: the eight answers sum to {sum(vals.values()):,}, not the "
                         f"printed total {universe:,}")
    if len(totals) == 2:
        known = int(row[totals[1]])
        if known != universe - vals["Desconhecida"]:
            raise SystemExit(f"{name}: second TOTAL {known:,} is not the total less "
                             f"DESCONHEC. ({universe - vals['Desconhecida']:,})")
    # The next two rows are Homens and Mulheres, and they must add up to the row above.
    h, m = rows[data_i + 1], rows[data_i + 2]
    if fold(h[0]) != "homens" or fold(m[0]) != "mulheres":
        raise SystemExit(f"{name}: the rows under T O T A L are {h[0]!r} and {m[0]!r}")
    for cat, j in cols.items():
        if int(h[j]) + int(m[j]) != vals[cat]:
            raise SystemExit(f"{name} {cat}: Homens + Mulheres != T O T A L")
    return universe, vals


def read():
    out = {}
    for code, name, _, _, keys in PROVINCES:
        out[code] = (name,) + read_one(code, name, keys)
    nat = read_one(NATIONAL[0], NATIONAL[1], NATIONAL[4])
    return out, nat


# ---------------------------------------------------------------- check


def check(data, nat):
    """The provinces against the national table, category by category, to the person."""
    nat_u, nat_v = nat
    print(f"{'province':18s} {'total':>11s} " +
          " ".join(f"{c[:10]:>10s}" for c in CATEGORIES))
    for code, (name, u, v) in data.items():
        print(f"{name:18s} {u:11,} " + " ".join(f"{v[c]:10,}" for c in CATEGORIES))
    su = sum(u for _, u, _ in data.values())
    print(f"{'sum of provinces':18s} {su:11,} " +
          " ".join(f"{sum(v[c] for _, _, v in data.values()):10,}" for c in CATEGORIES))
    print(f"{'national Quadro 11':18s} {nat_u:11,} " +
          " ".join(f"{nat_v[c]:10,}" for c in CATEGORIES))
    print(f"{'UNSD table 28':18s} {CENSUS_TOTAL:11,} " +
          " ".join(f"{UNSD_2017[c]:10,}" for c in CATEGORIES))

    bad = []
    if su != nat_u:
        bad.append(f"provinces sum to {su:,}, the national table prints {nat_u:,}")
    if nat_u != CENSUS_TOTAL:
        bad.append(f"national table total {nat_u:,} is not the census's {CENSUS_TOTAL:,}")
    diffs = {}
    for c in CATEGORIES:
        s = sum(v[c] for _, _, v in data.values())
        diffs[c] = s - nat_v[c]
        if abs(diffs[c]) > SLACK:
            bad.append(f"{c}: provinces {s:,}, national {nat_v[c]:,}")
    if bad:
        raise SystemExit("\n".join(["does not reconcile:"] + bad))
    print("\nthe eleven provinces reconcile with the national Quadro 11: the total to the "
          f"person, each category within {SLACK} ("
          + ", ".join(f"{c} {d:+,}" for c, d in diffs.items()) + ")")
    print(pinned_differences(nat_v, UNSD_2017, UNSD_DIFF_2017, what="against UNSD"))


# ---------------------------------------------------------------- write


def write(data):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for code, (name, u, v) in data.items():
        gid = f"MZ{code}"
        rows.append([gid, "province", name, UNIVERSE, u, BASIS, YEAR, SOURCE_ID,
                     "universe total, not a religion category"])
        for c in CATEGORIES:
            rows.append([gid, "province", name, c, v[c], BASIS, YEAR, SOURCE_ID, ""])
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        w.writerows(rows)
    print(f"wrote {OUT}  {len(rows)} rows")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    data, nat = read()
    check(data, nat)
    write(data)
