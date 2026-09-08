"""Liechtenstein — Amt für Statistik, Volkszählung 2015, religion by municipality.

Reads (or fetches) data/raw/li/ and writes data/normalized/li.csv.

One PxWeb table, **`213.001e`** — *Permanent resident population by Reference date, Religion,
Citizenship, Sex and Municipality* — **11 religion categories plus a total, over 11
municipalities**, for 2010 and 2015. About 3 KB, no key, no wall, English labels.

**IT WAS FOUND IN SWITZERLAND'S CATALOGUE.** `ckan.opendata.swiss` carries the Liechtenstein
statistics office alongside BFS, so the one search that solved Switzerland (§9ad) returned
this too. Nobody had looked at Liechtenstein: §11c and §11k both swept Europe without it.
**A national open-data portal may index a neighbour**, and a microstate is exactly the kind of
country that gets swept past.

**THE SMALLEST COUNTRY ON THIS MAP AND ONE OF THE FINEST.** 37,622 people over 11 communes is
**3,420 per unit** — by §11k's rule, people per unit rather than unit count, that is finer
than everything here except Estonia, Ireland and Portugal, and about the same as Switzerland's
communes. Anita's own reading was that one region would have done for a country this size; the
municipalities cost nothing extra, because **all 11 are already in the GISCO LAU file** this
project has had on disk since Poland (§9e).

**ITS json-stat2 EMITTER IS BROKEN AND THE CSV IS NOT.** Asking for `json-stat2` returns HTTP
200, a well-formed document declaring `size: [1, 12, 1, 1, 12]` — 144 cells — and a `value`
array holding **one element**. No error anywhere. `json-stat` (v1) and `csv` both answer
correctly, so this is a bug in one serialiser rather than a wall, and CSV is what this module
reads. *A 200 with a valid-looking envelope is not a download (§5a); check the payload against
the shape the same response declares.*

**`filter: "all"` with `values: ["*"]` is also not honoured** — it has to be an explicit item
list. Two servers in two days (§9ad's BFS drops the geography dimension on an empty query);
**PxWeb selection semantics are not portable and the returned shape is the only thing worth
believing.**

**The `-` cells are true zeros and the partition proves it**, which is Kosovo's argument
(§9w) rather than Lithuania's (§9q): there is no disclosure threshold here at all — Planken
publishes a single person in `Other Christian communities` — so a blank cannot be a
suppression of something small. `check()` asserts the partition rather than assuming the
reading.

**Vintage: 2015 is the last one.** The table offers 2010 and 2015 and nothing since;
Liechtenstein's later population statistics are register-based and carry no religion.

Usage:
    python sources/li.py --fetch    one POST, ~3 KB
    python sources/li.py            normalise from data/raw/li/
"""

import csv
import io
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "li")
OUT = os.path.join(ROOT, "data", "normalized", "li.csv")

SOURCE_ID = "li_volkszaehlung_2015"
YEAR = 2015
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

API = ("https://etab.llv.li/PXWEb/api/v1/en/eTab/Bev%C3%B6lkerung/"
       "Bev%C3%B6lkerungsstruktur/213.001e.px")
BOOK = os.path.join(RAW, "213.001e_2015.csv")

# The dimension VALUES are positional indices on this server, not codes, so they are written
# out rather than derived. `1` is 31.12.2015, the later of the two reference dates.
DATE = "1"
N_RELIGION = 12               # 11 categories plus the total
N_MUNICIPALITY = 12           # 11 communes plus the country
CITIZENSHIP = ["0", "1", "2"]  # total, Liechtenstein nationals, foreign nationals
SEX_TOTAL = "0"

TOTAL_CAT = "Religion - total"
COUNTRY = "Liechtenstein"
CITIZENSHIP_TOTAL = "Citizenship - total"
NO_OCCURRENCE = "-"

NATIONAL = 37_622
EXPECTED_MUNICIPALITIES = 11
EXPECTED_CATEGORIES = 11


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(BOOK) and os.path.getsize(BOOK) > 1_000:
        print("already have", BOOK)
        return
    q = {"query": [
        {"code": "Reference date",
         "selection": {"filter": "item", "values": [DATE]}},
        # EXPLICIT ITEM LISTS. `{"filter": "all", "values": ["*"]}` is accepted with a 200
        # and silently returns a single cell.
        {"code": "Religion",
         "selection": {"filter": "item",
                       "values": [str(i) for i in range(N_RELIGION)]}},
        {"code": "Citizenship",
         "selection": {"filter": "item", "values": CITIZENSHIP}},
        {"code": "Sex", "selection": {"filter": "item", "values": [SEX_TOTAL]}},
        {"code": "Municipality",
         "selection": {"filter": "item",
                       "values": [str(i) for i in range(N_MUNICIPALITY)]}}],
        # NOT json-stat2: this server's json-stat2 declares 144 cells and returns one.
        "response": {"format": "csv"}}
    print("POST", API)
    r = requests.post(API, json=q, timeout=300,
                      headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    r.raise_for_status()
    text = r.content.decode("utf-8-sig")
    if not text.lstrip().startswith('"Reference date"'):
        raise SystemExit(f"not the expected CSV: first bytes {text[:120]!r}")
    with open(BOOK, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    print(f"  {os.path.getsize(BOOK):,} bytes")


def read():
    if not os.path.exists(BOOK):
        raise SystemExit(f"missing {BOOK} -- run with --fetch first")
    with open(BOOK, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))

    header, body = rows[0], rows[1:]
    municipalities = header[4:]
    if len(municipalities) != N_MUNICIPALITY or municipalities[0] != COUNTRY:
        raise SystemExit(f"unexpected municipality columns: {municipalities}")

    # The CSV is one row per (date, religion, citizenship, sex) with a column per
    # municipality — a wide shape the rest of this project does not use, so it is pivoted
    # here rather than anywhere downstream.
    out, split, blanks = [], {}, 0
    for r in body:
        _, religion, citizenship, _sex = r[:4]
        for name, cell in zip(municipalities, r[4:]):
            cell = cell.strip()
            if cell == NO_OCCURRENCE:
                blanks += 1
                n = 0
            else:
                n = int(cell)
            if citizenship != CITIZENSHIP_TOTAL:
                split.setdefault(citizenship, {}).setdefault(religion, {})[name] = n
                continue
            level = "country" if name == COUNTRY else "municipality"
            note = f"level={level}"
            if religion == TOTAL_CAT:
                note += "; universe total, not a religion category"
            out.append({"geo_id": name, "geo_level": level, "geo_name": name,
                        "source_category": religion, "count": n, "basis": BASIS,
                        "year": YEAR, "source_id": SOURCE_ID, "note": note})
    return out, {"blanks": blanks, "split": split, "municipalities": municipalities[1:]}


def check(rows, stats):
    ok = True
    nat = {r["source_category"]: r["count"] for r in rows if r["geo_level"] == "country"}
    cats = [c for c in nat if c != TOTAL_CAT]
    muni = {r["geo_id"] for r in rows if r["geo_level"] == "municipality"}

    print(f"  {len(cats)} religion categories over {len(muni)} municipalities")
    good = len(muni) == EXPECTED_MUNICIPALITIES and len(cats) == EXPECTED_CATEGORIES
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {len(muni)} municipalities, {len(cats)} categories "
          f"(expected {EXPECTED_MUNICIPALITIES} and {EXPECTED_CATEGORIES})")

    good = nat[TOTAL_CAT] == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national total {nat[TOTAL_CAT]:,} "
          f"(published {NATIONAL:,})")

    parts = sum(nat[c] for c in cats)
    good = parts == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the {len(cats)} categories partition the country "
          f"exactly ({parts:,})")

    # ---- the blank-is-zero proof (§3.8) ----------------------------------------------
    print(f"\n  {stats['blanks']} cells are published `-`. There is no disclosure threshold "
          "here — Planken\n  publishes a single person — so a blank cannot be a suppressed "
          "small number, and the\n  partition below is what proves it rather than assumes "
          "it (§9w's argument, not §9q's):")
    worst = 0
    for cat in sorted(nat, key=lambda k: -nat[k]):
        s = sum(r["count"] for r in rows
                if r["geo_level"] == "municipality" and r["source_category"] == cat)
        gap = nat[cat] - s
        worst = max(worst, abs(gap))
        if gap:
            ok = False
        print(f"    {'OK ' if gap == 0 else 'BAD'} {cat:<30} {s:>8,}  "
              f"{100.0 * nat[cat] / NATIONAL:6.2f}%  gap {gap}")
    if worst:
        print("    BAD a non-zero gap means at least one `-` hides a real number.")

    # ---- the citizenship split, carried for note_public rather than drawn -------------
    sp = stats["split"]
    if sp:
        print("\n  the citizenship split, which this build does NOT draw and which is the "
              "most\n  interesting thing in the table — Liechtenstein is 34% foreign:")
        for cit in sorted(sp):
            tot = sp[cit][TOTAL_CAT][COUNTRY]
            rc = sp[cit]["Roman Catholic"][COUNTRY]
            orth = sp[cit]["Christian Orthodox"][COUNTRY]
            isl = sp[cit]["Islamic religious communities"][COUNTRY]
            none = sp[cit]["No religious affiliation"][COUNTRY]
            print(f"    {cit:<26} {tot:>7,}   Catholic {100.0 * rc / tot:5.1f}%   "
                  f"Orthodox {100.0 * orth / tot:4.1f}%   Muslim {100.0 * isl / tot:4.1f}%   "
                  f"none {100.0 * none / tot:5.1f}%")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, stats = read()
    check(rows, stats)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
