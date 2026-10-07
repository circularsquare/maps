"""Bosnia and Herzegovina: BHAS, Popis 2013, mother tongue by municipality.

    python sources/ba_census.py --fetch    download the two workbooks (110 KB) if missing
    python sources/ba_census.py            normalise from data/raw/ba/

-> data/normalized/ba.csv (levels `country`, `entity`, `canton`, `municipality`; alternatives,
   never summed; `municipality` is the drawn cover, 142 units, Brcko District among them)

THE TABLE. BHAS's second results book, *Etnicka/nacionalna pripadnost, vjeroispovijest i maternji
jezik* (Knjiga 2), table 6.1, *Stanovnistvo prema maternjem jeziku i spolu, po opcinama/gradovima*:
https://popis.gov.ba/popis2013/doc/Knjiga2/BOS/K2_T6-1_B.xlsx (the book's index is
https://popis.gov.ba/popis2013/knjige.php). Sixteen named answers, "Ostali" (other) and
"Nepoznato" (unknown), by sex, at country, entity, canton (Federation only) and municipality.
The first results book's table 5 (and its PDF, RezultatiPopisa_BS.pdf table 5.3, which
religiondots' religion table sits beside) folds the same answers to Bosnian, Croatian, Serbian,
Other and Unknown. Knjiga 2 is the finer one and is what is drawn.

THE SECOND TABLE IS THE CHECK. The first book's table 5,
https://popis.gov.ba/popis2013/doc/RezultatiPopisa/BOS/FR_T5_B.xlsx, is the same census at the
same units: per unit its Bosnian, Croatian, Serbian and Unknown must equal Knjiga 2's exactly, and
its Other must equal the thirteen smaller named answers plus Knjiga 2's own Other.

Brcko District is an entity-level row in table 6.1 (level 1) and has no municipality row under it;
it is a single unit, so it is lifted into the municipality cover, as in religiondots.

geo_id is religiondots' fold of the municipality name (religiondots/sources/ba.py `fold`, copied
here): the census publishes no codes, and religiondots' hex layer data/geo/ba/ba_grid_400m.gpkg
carries the fold in its `unit` column. The PDF spells digraphs BANjA, the workbook BANJA, and
"F BiH" / "FBiH"; the fold drops case and spaces, so both land on one key. Asserted: every unit
here matches a unit in religiondots' ba.csv (the religion table of the same census) both ways,
with identical totals.

CHECKS: the 18 categories partition every row; the 141 municipalities plus Brcko sum to the
national row in every column; the entities sum to the national row; each Federation canton's
municipalities sum to the canton; the per-unit agreement with table 5 above; the join to
religiondots both ways with equal totals; the national total 3,531,159.
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
RAW = os.path.join(ROOT, "data", "raw", "ba")
OUT = os.path.join(ROOT, "data", "normalized", "ba.csv")
RD_NORM = os.path.join(os.path.dirname(ROOT), "religiondots", "data", "normalized", "ba.csv")

SOURCE_ID = "ba_popis_2013_k2_t6_1"
YEAR = 2013
BASE = "https://popis.gov.ba/popis2013/doc/"
FILES = {
    "K2_T6-1_B.xlsx": "Knjiga2/BOS/K2_T6-1_B.xlsx",
    "FR_T5_B.xlsx": "RezultatiPopisa/BOS/FR_T5_B.xlsx",
}
NATIONAL = 3_531_159
EXPECTED_UNITS = 142

# Table 6.1's English header row (row 4), left to right from column D. Asserted against the sheet.
CATEGORIES = [
    "Total", "Bosnian", "Serbian", "Croatian", "Serbo-Croatian", "Romani", "Albanian",
    "Bosnian-Croatian-Serbian", "Turkish", "Croato-Serbian", "Bosniak", "Ukrainian",
    "Bosnian-Serbian-Croatian", "Bosnian-Croatian", "Bosnian-Herzegovinian", "German", "Other",
    "Unknown",
]
# Table 5's columns after the total, and how table 6.1's columns fold into them.
T5_COLS = ["Bosnian", "Croatian", "Serbian", "Other", "Unknown"]
T5_FOLD = {c: (c if c in ("Bosnian", "Croatian", "Serbian", "Unknown") else "Other")
           for c in CATEGORIES[1:]}
LEVEL = {0: "country", 1: "entity", 2: "canton", 3: "municipality"}
BRCKO = "BRČKO DISTRIKT BOSNE I HERCEGOVINE"
TRUE_ZERO = ("-", "–", "—")


def fold(name):
    """religiondots/sources/ba.py's fold, copied: municipality name -> stable ASCII key."""
    s = name.replace("Đ", "DJ").replace("đ", "dj")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^A-Za-z]", "", s).upper()


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for name, path in FILES.items():
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
            print("already have", dest)
            continue
        url = BASE + path
        print("GET", url)
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                         timeout=300)
        r.raise_for_status()
        if not r.content.startswith(b"PK"):
            raise SystemExit(f"{url} is not a workbook: starts {r.content[:16]!r}")
        tmp = dest + ".part"
        with open(tmp, "wb") as fh:
            fh.write(r.content)
        os.replace(tmp, dest)
        print(f"  {len(r.content):,} bytes")


def _num(v, where):
    if isinstance(v, (int, float)):
        if v != int(v):
            raise SystemExit(f"{where}: {v!r} is not a whole number")
        return int(v)
    if v is None or str(v).strip() in TRUE_ZERO:
        return 0
    raise SystemExit(f"{where}: unexpected value {v!r}")


def _name(v):
    return str(v).split("\n")[0].strip()


def read_t61():
    import openpyxl

    path = os.path.join(RAW, "K2_T6-1_B.xlsx")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run with --fetch first")
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    if not str(rows[0][0]).startswith("6.1. Stanovništvo prema maternjem jeziku"):
        raise SystemExit(f"not table 6.1: {rows[0][0]!r}")
    header = [str(x).strip() for x in rows[3][3:3 + len(CATEGORIES)]]
    if header != CATEGORIES:
        raise SystemExit(f"table 6.1 header changed:\n  {header}\n  {CATEGORIES}")
    recs = []
    canton = None
    for i, r in enumerate(rows[4:], 5):
        if r[0] is None or "Ukupno" not in str(r[2]):
            continue       # blank rows, and the male and female rows (summing them doubles)
        lvl, name = int(r[0]), _name(r[1])
        vals = {c: _num(v, f"row {i} {c}") for c, v in zip(CATEGORIES, r[3:3 + len(CATEGORIES)])}
        if lvl == 2:
            canton = name
        elif lvl == 1:
            canton = None
        level = LEVEL[lvl]
        recs.append(dict(level=level, name=name, canton=canton if lvl == 3 else None, vals=vals))
        if lvl == 1 and name == BRCKO:
            # the district has no municipality row of its own; it is one unit
            recs.append(dict(level="municipality", name="BRČKO", canton=None, vals=vals))
    return recs


def read_t5():
    import openpyxl

    ws = openpyxl.load_workbook(os.path.join(RAW, "FR_T5_B.xlsx"), read_only=True,
                                data_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    if not str(rows[0][0]).startswith("5. Stanovništvo prema maternjem jeziku"):
        raise SystemExit(f"not table 5: {rows[0][0]!r}")
    out = {}
    for i, r in enumerate(rows[5:], 6):
        if r[0] is None or r[1] != "Ukupno":
            continue
        vals = [_num(v, f"T5 row {i}") for v in r[2:8]]
        out[fold(_name(r[0]))] = dict(zip(["Total"] + T5_COLS, vals))
    return out


def check(recs):
    ok = True
    by = {}
    for r in recs:
        by.setdefault(r["level"], []).append(r)
    nat = by["country"][0]["vals"]
    muni = by["municipality"]

    def say(good, msg):
        nonlocal ok
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    say(nat["Total"] == NATIONAL, f"national total {nat['Total']:,} (published {NATIONAL:,})")
    say(len(muni) == EXPECTED_UNITS, f"{len(muni)} municipality units (expected {EXPECTED_UNITS})")
    bad = [r["name"] for r in recs if sum(v for k, v in r["vals"].items() if k != "Total")
           != r["vals"]["Total"]]
    say(not bad, f"the {len(CATEGORIES) - 1} categories partition every one of {len(recs)} rows "
                 f"{bad[:5]}")
    for lev, group in (("entity", by["entity"]), ("municipality", muni)):
        bad = [k for k in CATEGORIES if sum(r["vals"][k] for r in group) != nat[k]]
        say(not bad, f"the {lev} rows sum to the national row in all {len(CATEGORIES)} columns "
                     f"{bad}")
    bad = []
    for c in by["canton"]:
        kids = [r for r in muni if r["canton"] == c["name"]]
        if not kids or any(sum(r["vals"][k] for r in kids) != c["vals"][k] for k in CATEGORIES):
            bad.append(c["name"])
    say(not bad, f"each of {len(by['canton'])} cantons equals the sum of its municipalities {bad}")
    keys = [fold(r["name"]) for r in muni]
    say(len(set(keys)) == len(keys), f"{len(set(keys))} distinct folded keys")

    # the first results book's table 5, same census, same units, coarser categories
    t5 = read_t5()
    bad, miss = [], []
    for r in muni:
        k = fold(r["name"])
        if k not in t5:
            miss.append(k)
            continue
        want = {c: 0 for c in T5_COLS}
        for c, v in r["vals"].items():
            if c != "Total":
                want[T5_FOLD[c]] += v
        got = {c: t5[k][c] for c in T5_COLS}
        if got != want or t5[k]["Total"] != r["vals"]["Total"]:
            bad.append((k, got, want))
    say(not miss and not bad, f"table 5 of the first results book agrees in all {len(muni)} units "
                              f"(Bosnian, Croatian, Serbian, Unknown equal; its Other = the 13 "
                              f"smaller named answers + Other)  missing {miss[:5]}")
    for b in bad[:5]:
        print("        ", b)

    if os.path.exists(RD_NORM):
        rd = {}
        with open(RD_NORM, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["geo_level"] == "municipality" and row["source_category"] == "Ukupno":
                    rd[row["geo_id"]] = int(row["count"])
        mine = {fold(r["name"]): r["vals"]["Total"] for r in muni}
        only = sorted(set(mine) ^ set(rd))
        diff = [(k, mine[k], rd[k]) for k in mine if k in rd and mine[k] != rd[k]]
        say(not only and not diff, f"all {len(mine)} keys match religiondots' ba.csv (religion "
                                   f"table) both ways, identical totals  {only[:5]} {diff[:5]}")
    else:
        say(False, f"{RD_NORM} missing, so the join is unchecked")

    print("\n  categories, national:")
    for k, v in sorted(nat.items(), key=lambda kv: -kv[1]):
        if k != "Total":
            print(f"    {v:>10,}  {100.0 * v / NATIONAL:6.2f}%  {k}")
    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    recs = read_t61()
    check(recs)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n = 0
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "geo_level", "geo_name", "source_category", "count", "tier",
                    "year", "source_id", "note"])
        for r in recs:
            gid = "BA" if r["level"] == "country" else fold(r["name"])
            for c, v in r["vals"].items():
                note = "BHAS Popis 2013, Knjiga 2 table 6.1"
                if r["canton"]:
                    note += f"; canton={r['canton']}"
                if c == "Total":
                    note += "; universe total, not a language"
                w.writerow([gid, r["level"], r["name"], c, v, "measured", YEAR, SOURCE_ID, note])
                n += 1
    print(f"\nwrote {OUT} ({n:,} rows)")


if __name__ == "__main__":
    main()
