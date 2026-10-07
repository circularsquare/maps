"""Faroe Islands: Hagstova Føroya, Census 2011 (11 November 2011), first language.

    python sources/fo_census.py --fetch    three PxWeb queries into data/raw/fo/
    python sources/fo_census.py            normalise and check -> data/normalized/fo.csv

THE TABLES. Statbank `H2/MT` (Census 2011), open PxWeb v1 API, POST, no key, the same route
religiondots' sources/fo.py uses for MT325:
  - **MT16** *MT1.3.3 Population by primary language, country of birth of person, father and
    mother and place of usual residence*. Fetched twice: first language by the 7 districts
    (birth totals), which is drawn; and first language by country of birth (national), which is
    only read for the mapping calls (sources/fo.md).
  - **MT17** *MT1.3.4 ... age and sex*: first language by 5-year age band, national. A witness:
    its language totals must equal MT16's, and it shows who "No language" is (infants).

THE QUESTION. "First language" ("primary language" in the table title), one answer per person,
for the whole population of 48,346 including children: there is no not-stated row, the ten
answers sum to the total in every district. Ten answers: Faroese, Danish, Other Nordic
languages, Other European languages, Asian languages, Middle East/North African languages,
Other African languages, South American languages, Sign language, No language.

SUPPRESSION. Hagstova prints `...` for a cell under 3 (0, 1 or 2). In the district query that
hides 22 cells in five rows (ME/NA, Other African, South American, Sign language, No language).
Both margins of the hidden cells are known: each row's national total less its printed districts,
and each district's total less its printed rows. `fill()` fits the hidden cells to both margins
by averaging every integer table (each hidden cell 0, 1 or 2) that meets both margins: every
one is counted once. The filled cells are fractional and `tier` stays `measured` (the
rows and districts are the census's). Only two tables fit, differing in where 1 or 2 people sit.

CHECKS (all asserted): district rows sum to the national cell where nothing is hidden; each
district's answers sum to its total; MT17's language totals equal MT16's; the national total is
48,346, religiondots' MT1 resident count; the fitted cells meet both margins.
"""

import csv
import json
import os
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "fo")
OUT = os.path.join(ROOT, "data", "normalized", "fo.csv")

API = "https://statbank.hagstova.fo/api/v1/en/H2/MT/MT01/MT0103/"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

DISTRICTS = ["Norðoyar", "Eysturoy", "N-streymoy", "S-streymoy", "Vágar", "Sandoy", "Suðuroy"]
LANGS = ["Faroese", "Danish", "Other Nordic languages", "Other European languages",
         "Asian languages", "Middle East/North African languages", "Other African languages",
         "South American languages", "Sign language", "No language"]
TOTAL_LABEL = "SUM (first language)"
RESIDENTS = 48_346                       # religiondots sources/fo.py, MT1, 11 November 2011


def _all(code):
    return {"code": code, "selection": {"filter": "all", "values": ["*"]}}


def _one(code, v):
    return {"code": code, "selection": {"filter": "item", "values": [v]}}


_PARENTS = [_one("mother's country of birth", "MOTHER_5000"),
            _one("father's country of birth", "FATHER_5000")]
QUERIES = {
    "MT16_district": ("MT16.px", [_all("first language"), _one("country of birth", "5000"),
                                  *_PARENTS, _all("place of usual residence")]),
    "MT16_birth": ("MT16.px", [_all("first language"), _all("country of birth"), *_PARENTS,
                               _one("place of usual residence", "9999")]),
    "MT17_age": ("MT17.px", [_all("first language"), _one("country of birth", "5000"), *_PARENTS,
                             _one("sex", "TOT"), _all("age")]),
}


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for name, (tab, query) in QUERIES.items():
        body = json.dumps({"query": query, "response": {"format": "csv"}}).encode()
        req = urllib.request.Request(API + tab, data=body,
                                     headers={"User-Agent": UA, "Content-Type": "application/json"})
        print("POST", API + tab, name)
        data = urllib.request.urlopen(req, timeout=120).read()
        if not data.lstrip(b"\xef\xbb\xbf").startswith(b'"'):
            raise SystemExit(f"{name}: not a PxWeb CSV: {data[:160]!r}")
        path = os.path.join(RAW, name + ".csv")
        with open(path + ".part", "wb") as fh:
            fh.write(data)
        os.replace(path + ".part", path)
        with open(os.path.join(RAW, name + ".query.json"), "w", encoding="utf-8") as fh:
            json.dump({"url": API + tab, "query": query}, fh, ensure_ascii=False, indent=1)
        print(f"  {len(data):,} bytes")


def cell(tok, where):
    if tok == "...":
        return None
    if tok.isdigit():
        return int(tok)
    raise SystemExit(f"{where}: not a count: {tok!r}")


def read(name):
    path = os.path.join(RAW, name + ".csv")
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing; run with --fetch")
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def read_district():
    head, body = read("MT16_district")
    cols = [h.replace("SUM (father's country of birth) ", "") for h in head[3:]]
    if cols != ["SUM (place of usual residence)"] + DISTRICTS:
        raise SystemExit(f"MT16_district: unexpected columns {cols}")
    out = {}
    for r in body:
        out[r[0]] = [cell(v, f"MT16 {r[0]} / {c}") for c, v in zip(cols, r[3:])]
    if list(out) != [TOTAL_LABEL] + LANGS:
        raise SystemExit(f"MT16_district: unexpected rows {list(out)}")
    return out


def fill(t):
    """Fit the hidden district cells to both margins. -> {lang: [7 district counts]} (floats)."""
    tot = t[TOTAL_LABEL]
    if any(v is None for v in tot):
        raise SystemExit("MT16: a district total is hidden")
    nat = {}
    for lang in LANGS:
        n = t[lang][0]
        if n is None:                          # only South American: the total less the rest
            n = tot[0] - sum(t[x][0] for x in LANGS if x != lang)
            if not 0 <= n <= 2:
                raise SystemExit(f"{lang}: derived national {n} is not a hidden cell")
            print(f"  {lang}: national cell hidden; total less the other nine rows = {n}")
        nat[lang] = n
    hidden = [(lang, j) for lang in LANGS for j in range(7) if t[lang][j + 1] is None]
    row_res = {lang: nat[lang] - sum(v for v in t[lang][1:] if v is not None) for lang in LANGS}
    col_res = [tot[j + 1] - sum(t[lang][j + 1] for lang in LANGS if t[lang][j + 1] is not None)
               for j in range(7)]
    if sum(row_res.values()) != sum(col_res):
        raise SystemExit("hidden-cell margins disagree")
    for lang, r in row_res.items():
        if not any(h[0] == lang for h in hidden) and r != 0:
            raise SystemExit(f"{lang}: districts sum to {nat[lang] - r}, national {nat[lang]}")
    # Every integer table with each hidden cell in 0..2 that meets both margins, averaged
    # (each one counted once). Row by row, pruned on the districts' remaining residuals.
    rows = [lang for lang in LANGS if any(h[0] == lang for h in hidden)]
    cells = {lang: [j for (l, j) in hidden if l == lang] for lang in rows}

    def splits(n, k):
        if k == 0:
            if n == 0:
                yield ()
            return
        for v in range(min(2, n) + 1):
            for rest in splits(n - v, k - 1):
                yield (v,) + rest

    acc = {h: 0 for h in hidden}
    n_sol = 0

    def walk(i, left, chosen):
        nonlocal n_sol
        if i == len(rows):
            if all(v == 0 for v in left):
                n_sol += 1
                for h, v in chosen.items():
                    acc[h] += v
            return
        lang = rows[i]
        for sp in splits(row_res[lang], len(cells[lang])):
            new = list(left)
            ok = True
            for j, v in zip(cells[lang], sp):
                new[j] -= v
                ok &= new[j] >= 0
            if ok:
                ch = dict(chosen)
                ch.update({(lang, j): v for j, v in zip(cells[lang], sp)})
                walk(i + 1, new, ch)

    walk(0, list(col_res), {})
    if n_sol == 0:
        raise SystemExit("no table of hidden cells in 0..2 meets both margins")
    x = {h: acc[h] / n_sol for h in hidden}
    print(f"  {len(hidden)} hidden district cells; row residuals {row_res}; "
          f"district residuals {col_res}; {n_sol:,} integer tables meet both margins, averaged")
    out = {}
    for lang in LANGS:
        out[lang] = [float(t[lang][j + 1]) if t[lang][j + 1] is not None else x[(lang, j)]
                     for j in range(7)]
    return out, nat


def check_age(nat):
    head, body = read("MT17_age")
    rows = {r[0]: [cell(v, f"MT17 {r[0]}") for v in r[3:]] for r in body}
    for lang in [TOTAL_LABEL] + LANGS:
        n = rows[lang][0]
        want = nat.get(lang, RESIDENTS)
        if n is not None and n != want:
            raise SystemExit(f"MT17 {lang}: {n} against MT16 {want}")
    nl = rows["No language"]
    print(f"  MT17: language totals equal MT16's; No language by age: under 5 = {nl[1]}, "
          f"total {nl[0]}")


def main():
    t = read_district()
    if t[TOTAL_LABEL][0] != RESIDENTS or sum(t[TOTAL_LABEL][1:]) != RESIDENTS:
        raise SystemExit(f"MT16 total {t[TOTAL_LABEL][0]} / districts {sum(t[TOTAL_LABEL][1:])}, "
                         f"expected {RESIDENTS}")
    print(f"MT16: {RESIDENTS:,} residents, districts sum to the nation")
    filled, nat = fill(t)
    for j, d in enumerate(DISTRICTS):
        s = sum(filled[lang][j] for lang in LANGS)
        if abs(s - t[TOTAL_LABEL][j + 1]) > 1e-3:
            raise SystemExit(f"{d}: answers sum to {s}, total {t[TOTAL_LABEL][j + 1]}")
    print("  every district's ten answers sum to its total; no not-stated row")
    check_age(nat)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".part", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "geo_level", "geo_name", "source_category", "count", "tier", "year",
                    "source_id"])
        for j, d in enumerate(DISTRICTS):
            for lang in LANGS:
                w.writerow([d, "district", d, lang, round(filled[lang][j], 6), "measured", 2011,
                            "fo_census2011_MT16"])
    os.replace(OUT + ".part", OUT)
    print(f"wrote {OUT}")
    for lang in LANGS:
        print(f"  {lang:40s} {nat[lang]:>7,} {100 * nat[lang] / RESIDENTS:6.2f}%")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    main()
