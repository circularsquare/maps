"""Georgia — Geostat, 2014 General Population Census, religion by region.

Reads (or fetches) data/raw/ge/ and writes data/normalized/ge.csv.

One PxWeb table, **`22_Population by regions and religion.px`** — 12 religion answers plus a
total, over **11 regions**, 3,713,804 people. 4 KB. No key, no login, no wall.

**THE HOST IS `pc-axis.geostat.ge` AND NOTHING LINKS TO IT.** Geostat's census pages publish
this table as a single 33 KB `.xls` and that is what the earlier scouting pass found (§11k).
The same table is also in a live PxWeb that the site never mentions: `census.geostat.ge` is
404 and never archived, `api.geostat.ge` answers 200 with an empty body, and
`pc-axis.geostat.ge` serves the default IIS splash page at its root — **but
`pc-axis.geostat.ge/PXWeb/api/v1/en/` is a working PxWeb catalogue** with a whole
`Population Census 2014` database in it. A default IIS page is not a dead host; it is a host
with nothing mounted at `/`.

**AND ITS ROOT RETURNS `dbid`, NOT `id`** — the third instance of §11k's finding, after
Kosovo and Moldova. A catalogue walker keyed on `id` reads this server as empty.

**IT IS PxWeb v1 AND `json-stat2` 404s.** The metadata GET works, an empty `{"query": []}`
POST 404s, and only an explicit selection of every value in every dimension with
`"format": "json-stat"` (v1, not v2) returns data. Three separate departures from the modern
PxWeb contract on one host, each of which alone looks like "the table is not there".

**11 REGIONS, NOT 12 — ABKHAZIA IS ABSENT BECAUSE IT WAS NOT ENUMERATED.** The 2014 census
covers territory under Georgian government control, so Abkhazia and the Tskhinvali region
(South Ossetia) have no rows at all. That is a hole and not a wrong number, which is the
opposite of Kosovo's problem (§9w) and much easier to draw honestly.

**THE SUPPRESSION IS BOUNDED AND THE CODELIST SAYS SO.** A thirteenth "region" value is not a
region: it is the string `… is less or equal to 10`, the legend for the `...` and `..` cells,
leaking into the dimension. So a withheld cell is 0-10 rather than unknown, and `check()`
asserts the bound — every category's (national − sum of regions) must be at most ten times
the number of its suppressed cells. It holds: **7 suppressed cells in the drawn slice and 22
people unaccounted for in 3.7 million.**

Usage:
    python sources/ge.py --fetch    one GET + one POST, 4 KB
    python sources/ge.py            normalise from data/raw/ge/
"""

import csv
import json
import os
import sys
import urllib.parse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ge")
OUT = os.path.join(ROOT, "data", "normalized", "ge.csv")

SOURCE_ID = "ge_census_2014"
YEAR = 2014
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

BASE = "https://pc-axis.geostat.ge/PXWeb/api/v1/en/"
TABLE = ("Database/Population Census 2014/Demographic And Social Characteristics/"
         "22_Population by regions and religion.px")
API = BASE + urllib.parse.quote(TABLE)
CUBE = os.path.join(RAW, "22_regions_religion.json")

COUNTRY_LABEL = "GEORGIA"
TOTAL_CAT = "Total"
BOTH = "Total"                       # the Urban/Rural dimension
LEGEND_PREFIX = "…"             # the '…' that starts the fake region value
SUPPRESSED = ("...", "..")           # both markers mean the same thing: 0-10
SUPPRESSION_CAP = 10

NATIONAL = 3_713_804
EXPECTED_REGIONS = 11


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(CUBE) and os.path.getsize(CUBE) > 2_000:
        print("already have", CUBE)
        return
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    print("GET ", API)
    meta = requests.get(API, headers=ua, timeout=180)
    meta.raise_for_status()
    variables = meta.json()["variables"]
    codes = [v["code"] for v in variables]
    if codes != ["Regions", "Urban/Rural", "Religion"]:
        raise SystemExit(f"table shape has changed: dimensions are {codes}")

    # PxWeb v1 here refuses an empty query and refuses json-stat2. Ask for everything,
    # explicitly, in json-stat v1. Both departures are in the module docstring.
    query = {"query": [{"code": v["code"],
                        "selection": {"filter": "item", "values": v["values"]}}
                       for v in variables],
             "response": {"format": "json-stat"}}
    print("POST", API, "(explicit selection, json-stat v1)")
    r = requests.post(API, headers=ua, timeout=300, json=query)
    r.raise_for_status()
    doc = r.json()
    # §5a: HTTP 200 is not a download.
    if "dataset" not in doc or "value" not in doc.get("dataset", {}):
        raise SystemExit(f"not a json-stat v1 dataset, keys {list(doc)}")
    with open(CUBE, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False)
    print(f"  {os.path.getsize(CUBE):,} bytes")


def _load():
    if not os.path.exists(CUBE):
        raise SystemExit(f"missing {CUBE} -- run with --fetch first")
    with open(CUBE, encoding="utf-8") as fh:
        return json.load(fh)["dataset"]


def read():
    ds = _load()
    dim, size, val = ds["dimension"], ds["dimension"]["size"], ds["value"]
    ids = dim["id"]
    status = ds.get("status") or {}

    def axis(name):
        cat = dim[name]["category"]
        order = sorted(cat["index"].items(), key=lambda kv: kv[1])
        return [(c, cat.get("label", {}).get(c, c)) for c, _ in order]

    regions, urbrur, religions = axis("Regions"), axis("Urban/Rural"), axis("Religion")
    if [t for _, t in urbrur][0] != BOTH:
        raise SystemExit(f"Urban/Rural does not start with {BOTH!r}: {urbrur}")

    rows, stats = [], {"n": 0, "suppressed": 0, "legend_dropped": 0}
    for i, (rcode, rname) in enumerate(regions):
        rname = rname.strip()
        if rname.startswith(LEGEND_PREFIX):
            # Not a region: the suppression legend, sitting in the dimension. Dropping it
            # silently would be the easy mistake; it is counted so the check can see it.
            stats["legend_dropped"] += 1
            continue
        level = "country" if rname == COUNTRY_LABEL else "region"
        for k, (ccode, cname) in enumerate(religions):
            cname = cname.strip()
            n = i * size[1] * size[2] + 0 * size[2] + k        # Urban/Rural = Total
            v = val[n]
            note = f"level={level}; code={ccode}"
            if not isinstance(v, (int, float)):
                marker = status.get(str(n), str(v))
                if marker not in SUPPRESSED:
                    raise SystemExit(
                        f"{rname}/{cname}: non-numeric cell {v!r} with status {marker!r}, "
                        f"which is neither of the known markers {SUPPRESSED}")
                stats["suppressed"] += 1
                continue          # 0-10 people, withheld: not filled (spec §3.5, §3.8)
            stats["n"] += 1
            if cname == TOTAL_CAT:
                note += "; universe total, not a religion category"
            rows.append({"geo_id": str(rcode), "geo_level": level, "geo_name": rname,
                         "source_category": cname, "count": int(v), "basis": BASIS,
                         "year": YEAR, "source_id": SOURCE_ID, "note": note})
    return rows, stats, [t.strip() for _, t in religions]


def check(rows, stats, cats):
    ok = True
    print(f"  {stats['n']:,} cells kept; {stats['suppressed']} withheld as "
          f"'0-10'; {stats['legend_dropped']} legend row dropped from the Regions "
          "dimension")
    good = stats["legend_dropped"] == 1
    ok &= good
    if not good:
        print("    BAD expected exactly one '… is less or equal to 10' pseudo-region. If it "
              "is gone, the suppression convention may have changed with it.")

    regions = {r["geo_id"] for r in rows if r["geo_level"] == "region"}
    good = len(regions) == EXPECTED_REGIONS
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {len(regions)} regions "
          f"(expected {EXPECTED_REGIONS} — Abkhazia and the Tskhinvali region were not "
          "enumerated and have no rows)")

    nat = {r["source_category"]: r["count"] for r in rows if r["geo_level"] == "country"}
    good = nat.get(TOTAL_CAT) == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national total {nat.get(TOTAL_CAT):,} "
          f"(published {NATIONAL:,})")

    parts = sum(v for k, v in nat.items() if k != TOTAL_CAT)
    good = parts == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the 12 answers partition the country exactly "
          f"({parts:,}) — nothing is withheld at national level")

    # ---- the bound, which is what makes a withheld cell harmless here (§3.8) ----
    print("\n  every category: (national - sum of regions) must be <= 10 x suppressed cells,\n"
          "  because the codelist says a withheld cell is 'less or equal to 10':")
    n_sup = {}
    for c in cats:
        if c == TOTAL_CAT:
            continue
        drawn = [r for r in rows if r["geo_level"] == "region" and r["source_category"] == c]
        n_sup[c] = EXPECTED_REGIONS - len(drawn)
    total_gap = 0
    for c in sorted(nat, key=lambda k: -nat[k]):
        if c == TOTAL_CAT:
            continue
        s = sum(r["count"] for r in rows
                if r["geo_level"] == "region" and r["source_category"] == c)
        gap = nat[c] - s
        total_gap += gap
        cap = SUPPRESSION_CAP * n_sup[c]
        good = 0 <= gap <= cap
        ok &= good
        print(f"    {'OK ' if good else 'BAD'} {c[:26]:<28} {s:>9,}  gap {gap:>3,} "
              f"<= {cap:>3} ({n_sup[c]} withheld)")
    frac = total_gap / NATIONAL
    print(f"    total unaccounted {total_gap:,} = {frac:.5%} of the country")
    good = frac < 0.0001
    ok &= good

    print(f"\n  {len(rows):,} rows. Answers, national:")
    for c, n in sorted(nat.items(), key=lambda kv: -kv[1]):
        mark = "  <- universe" if c == TOTAL_CAT else ""
        print(f"    {n:>10,}  {100.0 * n / NATIONAL:6.2f}%  {c}{mark}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, stats, cats = read()
    check(rows, stats, cats)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
