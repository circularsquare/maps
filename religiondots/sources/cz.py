"""Czechia — CZSO Scitani 2021 (SLDB), religious belief, at municipality.

Reads (or fetches) data/raw/cz/ and writes data/normalized/cz.csv.

This is the best source in the project after ASARB, and on one axis it beats it: 78
named categories published at OBEC (municipality, 6,254 units) with NO suppression and
NO rounding.  A count of 1 is published as 1.  Spolecenstvi Josefa Zezulky has exactly
one adherent in Vysoke Myto and CZSO says so.

Every other country met so far does the opposite -- spec 3.9, "category detail and
spatial detail trade off inside one source", is the shape of Australia, Canada, New
Zealand, Mexico, Ireland and Brazil.  Czechia simply does not make the trade, which
means no allocate.py step is needed and, unlike Canada's, every Czech row may become a
ring (spec 3.10).

WHAT THE CATEGORY LIST CONTAINS, and why it is not a clean taxonomy:

  * Registered churches under their legal names -- Cirkev rimskokatolicka,
    Ceskobratrska cirkev evangelicka, Cirkev ceskoslovenska husitska, and the small
    ones down to Cirkev Oaza and Cirkev Novy Zivot.
  * Bare tradition names for the unregistered -- islam, buddhismus, hinduismus,
    judaismus, sikhismus, taoismus.  So "judaismus" (the tradition) and "Federace
    zidovskych obci v Ceske republice" (the institution) are SEPARATE ROWS covering
    overlapping people.  Same for islam vs Ustredi muslimskych obci, and buddhismus vs
    its two named schools.  Spec 2.3: these are not all the same kind of thing.
  * Positions rather than religions -- ateismus, agnosticismus, deismus, esoterismus.
  * Two residual categories that are enormous and mean different things:
    "verici - nehlasici se k zadne cirkvi ani nabozenske spolecnosti" (believer, no
    church) and "verici - hlasici se k cirkvi - nazev neuveden" (believer, church not
    named).  Neither is "no religion", which is its own category.
  * Joke and protest answers as first-class published categories -- Jedi (49), Sith
    (52), pastafarianstvi (64).  CZSO tabulated them because respondents wrote them in.
    They are left verbatim here per spec 2.4 and are a taxonomy decision, not a data
    error: the honest options are a "not a religion" node or exclusion, and that is not
    a decision this file gets to make.

The religion question was VOLUNTARY, and 30% of the country did not answer -- the
"Neuvedeno" category, larger than every church combined.  That is the dominant fact
about the Czech map and it belongs in note_public, not in a footnote.

Usage:
    python sources/cz.py --fetch    download the CSV (56MB) if missing
    python sources/cz.py            normalise from data/raw/cz/
"""

import csv
import os
import sys
import urllib.request

# Category names are Czech and the reconciliation prints them.  A Windows console is
# cp1252 by default, which cannot encode "u with ring above" and kills the run at the
# print rather than at anything to do with the data.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "cz")
OUT = os.path.join(ROOT, "data", "normalized", "cz.csv")

SOURCE_ID = "cz_sldb_2021"
YEAR = 2021
BASIS = "self_id"

CSV_NAME = "sldb2021_vira.csv"
CSV_URL = ("https://csu.gov.cz/docs/107508/4250766c-69e6-3845-0eb4-580f7a692558/"
           "sldb2021_vira.csv")
# The dataset is listed in the national open data catalogue as
# https://data.gov.cz/zdroj/datov%C3%A9-sady/00025593/d48571d456d56aa11a4f3488eeba47ec
# whose distribution URL is the one above.  Expected size ~55.5MB / ~390,000 rows; a
# short file is a truncated download, not an empty release (sources.md 5a).
MIN_BYTES = 50_000_000

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# CZSO territorial classification ids (ciselnik `uzemi_cis`), with the unit count each
# carries in this file.  The whole hierarchy is present, so summing the file as
# delivered counts everybody eight times over; `geo_level` keeps them apart and nothing
# downstream may mix two levels.
#
# `city_district` is the interesting one: 142 mestske casti / mestske obvody that
# SUBDIVIDE the statutory cities.  Prague as one polygon is 1.3M people in a single
# unit, which is the worst case for a dot map; at this level it is 57.  So Czechia is
# finer than obec exactly where it needs to be, and the two levels are alternatives --
# city_district REPLACES its parent obec, it does not nest under it for drawing.
# The third field is the population the level covers: the national total for the five
# levels that partition the whole country, and a smaller measured figure for the two
# that do not.  city_district and prague_sso exist ONLY for the statutory cities, so
# they cover 2.5M and 1.3M rather than 10.5M -- checking them against the national total
# is what first suggested a missing unit when nothing was missing.
NATIONAL = 10524167  # CZSO, usually resident population, census day 2021-03-26
UZEMI = {
    "43": ("municipality", 6254, NATIONAL),    # obec
    "44": ("city_district", 142, 2495217),     # mestska cast / mestsky obvod
    "65": ("orp", 206, NATIONAL),              # obec s rozsirenou pusobnosti
    "72": ("prague_sso", 22, 1301432),         # spravni obvody hl. m. Prahy
    "101": ("okres", 77, NATIONAL),            # district
    "100": ("kraj", 14, NATIONAL),             # region
    "99": ("nuts2", 8, NATIONAL),              # oblast
    "97": ("country", 1, NATIONAL),
}

TOTAL_NOTE = "universe total, not a religion category"


def fetch():
    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, CSV_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) >= MIN_BYTES:
        print("already have", dest)
        return
    print("downloading", CSV_URL)
    urllib.request.urlretrieve(CSV_URL, dest)
    size = os.path.getsize(dest)
    if size < MIN_BYTES:
        raise SystemExit(f"{dest} is {size:,} bytes, expected >= {MIN_BYTES:,} -- "
                         "truncated download (check free disk space)")
    print(f"  {size:,} bytes")


def read():
    path = os.path.join(RAW, CSV_NAME)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run with --fetch first")
    # A truncated file ends mid-row and csv still yields the partial line, so the size
    # is checked before parsing rather than trusting the reader to complain.
    size = os.path.getsize(path)
    if size < MIN_BYTES:
        raise SystemExit(f"{path} is {size:,} bytes, expected >= {MIN_BYTES:,} -- "
                         "truncated download, re-run with --fetch")

    rows, unknown_level = [], set()
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            known = UZEMI.get(r["uzemi_cis"])
            if known is None:
                unknown_level.add(r["uzemi_cis"])
                continue
            level = known[0]
            # A blank vira_kod is the municipality's own total, not a category.
            code = r["vira_kod"].strip()
            is_total = code == ""
            name = r["vira_txt"].strip() if not is_total else "Celkem"
            note = f"CZSO SLDB 2021 open data {CSV_NAME}"
            if is_total:
                note += "; " + TOTAL_NOTE
            else:
                note += f"; code={code}"
            rows.append({
                "geo_id": r["uzemi_kod"],
                "geo_level": level,
                "geo_name": r["uzemi_txt"],
                "source_category": name,
                "count": int(r["hodnota"]),
                "basis": BASIS,
                "year": YEAR,
                "source_id": SOURCE_ID,
                "note": note,
            })
    return rows, unknown_level


def check(rows, unknown_level):
    ok = True
    if unknown_level:
        print("  NOTE new territorial levels in the file, skipped:",
              sorted(unknown_level))

    expected = {name: (n, pop) for name, n, pop in UZEMI.values()}

    # Each level is checked INDEPENDENTLY against its own unit count and its own
    # population, which is what proves no level is short a unit and would also catch
    # two levels being summed together by mistake.
    tot_by_unit, cat_by_unit, units_by_level = {}, {}, {}
    for r in rows:
        key = (r["geo_level"], r["geo_id"])
        units_by_level.setdefault(r["geo_level"], set()).add(r["geo_id"])
        if r["source_category"] == "Celkem":
            tot_by_unit[key] = r["count"]
        else:
            cat_by_unit[key] = cat_by_unit.get(key, 0) + r["count"]

    print(f"  units and population per level — levels are alternatives, never summed "
          f"(national {NATIONAL:,}):")
    for level in sorted(units_by_level, key=lambda x: -len(units_by_level[x])):
        got = sum(n for (lv, _), n in tot_by_unit.items() if lv == level)
        n_units = len(units_by_level[level])
        want_units, want_pop = expected[level]
        good = got == want_pop and n_units == want_units
        ok &= good
        partial = "" if want_pop == NATIONAL else "  <- statutory cities only"
        print(f"    {'OK ' if good else 'BAD'} {level:<14} {n_units:>6,} units  "
              f"{got:>12,}  (expected {want_units:,} units, {want_pop:,}){partial}")

    # The strongest internal check available without a second source: within every unit
    # the categories must partition that unit's own total exactly.  If CZSO ever
    # suppresses or rounds a cell, this is what catches it.
    mismatch = {k: (t, cat_by_unit.get(k, 0))
                for k, t in tot_by_unit.items() if cat_by_unit.get(k, 0) != t}
    good = not mismatch
    ok &= good
    print(f"\n  {'OK ' if good else 'BAD'} categories partition the total in all "
          f"{len(tot_by_unit):,} units at every level ({len(mismatch)} mismatched)")
    for k, (t, c) in list(mismatch.items())[:5]:
        print(f"      {k}: total {t:,} vs categories {c:,}")

    cats = sorted({r["source_category"] for r in rows if r["source_category"] != "Celkem"})
    print(f"\n  {len(rows):,} rows, {len(cats)} categories")

    # Largest categories, as the eyeball check that the file is what it claims.  One
    # level only -- summing all eight would report the country eight times.
    by_cat = {}
    for r in rows:
        if r["source_category"] != "Celkem" and r["geo_level"] == "municipality":
            by_cat[r["source_category"]] = by_cat.get(r["source_category"], 0) + r["count"]
    print("\n  national totals, largest first:")
    for name, n in sorted(by_cat.items(), key=lambda x: -x[1])[:12]:
        print(f"    {n:>10,}  {name}")
    print("\n  smallest published categories:")
    for name, n in sorted(by_cat.items(), key=lambda x: x[1])[:8]:
        print(f"    {n:>10,}  {name}")
    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, unknown_level = read()
    check(rows, unknown_level)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
