"""Estonia — Statistics Estonia, Rahvaloendus 2021, religion, down to municipality.

Reads (or fetches) data/raw/ee/ and writes data/normalized/ee.csv.

Small — 1.33M people — and worth having anyway for three reasons:

  1. It is the LEAST religious country on the map by a distance. 57.9% of Estonians aged
     15+ say they feel no affiliation to any religion, against Czechia's 47% and Poland's
     6.9%. The map has been short of that end of the range.
  2. It names **Old Believers** (`Vanausuline`), the Russian communities on the west shore
     of Lake Peipus. That is the third source in a row to need
     `christianity.orthodox.oldbeliever`, after Poland and Romania.
  3. It names **Maausk and Taarausk** — `Earth Believer` and `Taara Believer` — the
     Estonian native-faith movement, which is a reconstruction with a real 1920s lineage
     rather than a residual, and which no other census in the world enumerates.

Estonia is also the ONLY country besides Czechia where the capital can be broken up: see
§3 below and `sources/ee_geo.md`.

THE UNIVERSE IS 15 AND OVER. The religion question is asked of persons aged at least 15,
so the table's own total is 1,116,000-odd rather than the 1.33M population. Everything
here is on that basis and the map draws no Estonian child (spec §3.7, the same shape as
the Philippines excluding institutional residents).

Usage:
    python sources/ee.py --fetch    pull both tables from the PxWeb API
    python sources/ee.py            normalise from data/raw/ee/
"""

import csv
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ee")
OUT = os.path.join(ROOT, "data", "normalized", "ee.csv")

SOURCE_ID = "ee_rahvaloendus_2021"
YEAR = 2021
BASIS = "self_id"

API = "https://andmed.stat.ee/api/v1/en/stat"
STEM = "rahvaloendus/rel2021/rahvastiku-demograafilised-ja-etno-kultuurilised-naitajad/usk"

# RL21452 is religion x administrative unit — 21 categories over 111 places, and the only
# 2021 table that goes below the county.
# RL21451 is religion x settlement TYPE — only 4 places, but 44 categories, which is where
# Anglicans, Quakers, Mormons, Hare Krishna, Wiccans and Satanists are named. Carried at
# its own geo_level so the two category sets can never be mixed (spec §3.9).
TABLES = {
    "RL21452.px": dict(file="RL21452.json", vars=("Elukoht", "Usk")),
    "RL21451.px": dict(file="RL21451.json", vars=("Elukoht", "Usk", "Sugu")),
}

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

TOTAL_LABEL = "Religion total"
TOTAL_NOTE = "universe total, not a religion category"

# Statistics Estonia carries the classification depth in the label text as leading dots:
# no dots is a top-level answer, ".." is a religion inside "Feels an affiliation".
# Place names use the same trick for geography.
def depth(text):
    n = 0
    while text[n:n + 2] == "..":
        n += 2
    return n // 2


def clean(text):
    return text.lstrip(".").strip()


def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    os.makedirs(RAW, exist_ok=True)
    for table, cfg in TABLES.items():
        dest = os.path.join(RAW, cfg["file"])
        if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
            print("already have", dest)
            continue
        url = f"{API}/{STEM}/{table}"
        query = {"query": [{"code": v, "selection": {"filter": "all", "values": ["*"]}}
                           for v in cfg["vars"]],
                 "response": {"format": "json-stat2"}}
        print("POST", url)
        r = requests.post(url, json=query, timeout=300, verify=False,
                          headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(r.text)
        print(f"  {os.path.getsize(dest):,} bytes")


def _load(name):
    p = os.path.join(RAW, name)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch first")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _cells(js):
    """Walk a json-stat2 cube and yield (dict of dimension->(code, label), value).

    json-stat stores one flat `value` array in row-major order over the dimensions listed
    in `id`, with sizes in `size`. Anything that reads it as a table of rows is guessing.
    """
    ids, sizes = js["id"], js["size"]
    dims = []
    for d in ids:
        cat = js["dimension"][d]["category"]
        order = sorted(cat["index"], key=lambda k: cat["index"][k])
        dims.append([(k, cat["label"][k]) for k in order])
    values = js["value"]

    strides, acc = [0] * len(sizes), 1
    for i in range(len(sizes) - 1, -1, -1):
        strides[i] = acc
        acc *= sizes[i]

    for flat, v in enumerate(values):
        if v is None:
            continue
        key = {}
        for i, d in enumerate(ids):
            key[d] = dims[i][(flat // strides[i]) % sizes[i]]
        yield key, v


# Place codes carry their own level, and the indent says which:
#   0 dots and all-caps      county          00370000000000
#   0 dots, code '1'         whole country
#   0 dots, code EEnnn/L1..  NUTS or settlement-type aggregate — NOT a place, skipped
#   2 dots                   municipality    00370141000001 / 003707840000L4
#   4 dots                   city district   003707840176L6
AGGREGATE_CODES = {"1", "L1", "V1", "M1"}


def _place_level(code, label):
    d = depth(label)
    if code in AGGREGATE_CODES:
        return "country" if code == "1" else None
    if code.startswith("EE"):
        return None                      # NUTS3 analytical regions, not administrative
    if d == 0:
        return "county"
    if d == 1:
        return "municipality"
    if d == 2:
        return "city_district"
    return None


def read():
    rows = []

    # ---------------------------------------------------- RL21452, by administrative unit
    js = _load(TABLES["RL21452.px"]["file"])
    for key, v in _cells(js):
        (pcode, plabel) = key["Elukoht"]
        (rcode, rlabel) = key["Usk"]
        level = _place_level(pcode, plabel)
        if level is None:
            continue
        cat = clean(rlabel)
        note = "Statistics Estonia RL21452; persons aged 15+"
        if cat == TOTAL_LABEL:
            note += "; " + TOTAL_NOTE
        elif cat == "Feels an affiliation to a religion":
            note += "; universe subtotal, not a religion category"
        rows.append({"geo_id": pcode, "geo_level": level, "geo_name": clean(plabel),
                     "source_category": cat, "count": int(v), "basis": BASIS,
                     "year": YEAR, "source_id": SOURCE_ID, "note": note})

    # ---------------------------------------------------- RL21451, the 44-category list
    js = _load(TABLES["RL21451.px"]["file"])
    for key, v in _cells(js):
        (pcode, plabel) = key["Elukoht"]
        (rcode, rlabel) = key["Usk"]
        (scode, slabel) = key["Sugu"]
        if scode != "1":                  # males and females
            continue
        cat = clean(rlabel)
        note = "Statistics Estonia RL21451; persons aged 15+; 44-category list"
        if cat == TOTAL_LABEL:
            note += "; " + TOTAL_NOTE
        rows.append({"geo_id": pcode, "geo_level": "settlement_type",
                     "geo_name": clean(plabel), "source_category": cat,
                     "count": int(v), "basis": BASIS, "year": YEAR,
                     "source_id": SOURCE_ID, "note": note})
    return rows


def check(rows):
    ok = True
    levels, totals, parts = {}, {}, {}
    UNIVERSE = (TOTAL_LABEL, "Feels an affiliation to a religion")
    ncat = 0
    for r in rows:
        lv, gid, cat = r["geo_level"], r["geo_id"], r["source_category"]
        levels.setdefault(lv, set()).add(gid)
        if cat == TOTAL_LABEL:
            totals[(lv, gid)] = r["count"]
        elif cat not in UNIVERSE and depth_of(rows, r) is not None:
            parts[(lv, gid)] = parts.get((lv, gid), 0) + r["count"]

    national = totals.get(("country", "1"))
    print(f"  national 15+ universe: {national:,}")

    # EVERY published figure is a multiple of 10. Statistics Estonia rounds to base 10 for
    # disclosure control, which is Canada's base-5 problem (spec §3.8) one size larger, and
    # it means NOTHING here reconciles exactly and nothing is supposed to.
    #
    # The bands below are the worst case that rounding can produce, not a fudge factor: a
    # unit's own total is within ±5 of the truth, so a sum of n units is within ±5n, and a
    # unit's k answers sum to within ±5k of its total. Exceeding those would be a real
    # error. The observed drift is printed so it can be watched.
    off = [n for n in totals.values() if n % 10] + [n for n in parts.values() if n % 10]
    print(f"  {'OK ' if not off else 'BAD'} every figure is a multiple of 10 "
          f"({len(off)} that are not) — base-10 rounding, spec §3.8")
    ok &= not off

    expected = {"country": 1, "county": 15, "municipality": 79, "city_district": 8,
                "settlement_type": 4}
    print("\n  units and population per level — levels are alternatives, never summed:")
    for lv in ("municipality", "county", "city_district", "settlement_type", "country"):
        got_units = len(levels.get(lv, ()))
        got_pop = sum(n for (l, _), n in totals.items() if l == lv)
        # city_district covers Tallinn only, settlement_type includes the national row
        full = lv in ("municipality", "county", "country")
        band = 5 * got_units
        drift = got_pop - national
        good = got_units == expected[lv] and (abs(drift) <= band if full else True)
        ok &= good
        tail = (f"   drift {drift:+,} of ±{band:,} allowed" if full
                else "   <- partial cover by design")
        print(f"    {'OK ' if good else 'BAD'} {lv:<16} {got_units:>4,} units  "
              f"{got_pop:>10,}  (expected {expected[lv]} units){tail}")

    ncat = len({r["source_category"] for r in rows
                if r["source_category"] not in UNIVERSE})
    band = 5 * ncat
    bad = [(k, parts[k], totals.get(k, 0)) for k in parts
           if parts[k] - totals.get(k, 0) > band]
    worst = max((parts[k] - totals.get(k, 0) for k in parts), default=0)
    ok &= not bad
    print(f"\n  {'OK ' if not bad else 'BAD'} answers never exceed the unit's 15+ total "
          f"by more than rounding allows ({len(bad)} over the ±{band} band; "
          f"worst observed {worst:+,})")
    for k, p, t in bad[:5]:
        print(f"      {k}: parts {p:,} vs total {t:,}")

    by_cat = {}
    for r in rows:
        if r["geo_level"] == "country":
            by_cat[r["source_category"]] = r["count"]
    print(f"\n  {len(rows):,} rows; the 21-category list at national level:")
    for name, n in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"    {n:>9,}  {name}")

    fine = {r["source_category"] for r in rows if r["geo_level"] == "settlement_type"}
    coarse = {r["source_category"] for r in rows if r["geo_level"] == "municipality"}
    print(f"\n  {len(fine)} categories in the settlement-type table, {len(coarse)} at "
          f"municipality — {len(fine - coarse)} named only nationally:")
    print("    " + ", ".join(sorted(fine - coarse)))

    if not ok:
        raise SystemExit("reconciliation FAILED")


def depth_of(rows, r):
    """Only the leaf answers are parts of the total; the two universe rows are not."""
    return None if r["source_category"] in (
        TOTAL_LABEL, "Feels an affiliation to a religion") else 1


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows = read()
    check(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
