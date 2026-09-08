"""Ghana — Ghana Statistical Service, 2021 PHC, religion by district.

Reads (or fetches) data/raw/gh/ and writes data/normalized/gh.csv.

One PxWeb table, **`religion_table.px`** in the `PHC 2021 StatsBank` database: 9 religion
categories x 295 geographic rows, 30.75M people. §12's "try a PxWeb API before anything
else", collecting again — but with a twist worth its own paragraph, below.

THE API IS NOT UNDER THE UI PREFIX. StatsBank's web interface lives at `/pxweb/en/...`, so
the obvious API root is `/pxweb/api/v1/en/`. That returns a **PX-Web ASP.NET error page with
HTTP 500** — not a 404, which makes it read like a broken or disabled API rather than a wrong
path. The API is at **`/api/v1/en/`**, with no `/pxweb` in front of it, and works perfectly.
One session nearly wrote Ghana off as "PxWeb with the API switched off" on the strength of
that 500. Try the API prefix with AND without the UI prefix before concluding anything.

**THE GEOGRAPHY DIMENSION HOLDS THREE NESTED LEVELS AND ONE ACRONYM COLLISION.** The 295
`Geographic_Area` values are Ghana, its 16 regions, 261 districts (MMDAs) — and, for six
metropolitan districts, a further breakdown into 17 sub-metros. So:

  * summing the file as delivered counts the country three times over;
  * the six metros appear as parent AND children, and the children sum to the parent
    exactly, so the sub-metros REPLACE their parent (§12's Czechia/Estonia rule) and the
    drawn tier is **272 units**, not 261;
  * **`TMA` is Tema Metropolitan Area in Greater Accra and Tamale Metropolitan Area in
    Northern.** Two different metros, 600 km apart, one acronym. Resolving the `TMA-` rows
    against a single global acronym gives Tamale all four and Tema none, while every
    national and regional total still reconciles. The prefix is only unique WITHIN a region
    block, so that is where it is resolved — see `_nest()`.

The source publishes NO CODES of any kind, only names, so the geo_id is the name (Romania's
shape). `sources/gh_geo.py` joins those names to GSS's own boundary files.

**THE UNIVERSE IS 30,753,327, NOT THE CENSUS'S 30,832,019**, and the 78,692-person gap is
item non-response on the religion question — GSS drops it rather than publishing a "not
stated" category. It is spread evenly across every age band and every education band (both
checked), so it is missingness and not a group. Per §3.1 and Chile's 15+ precedent it is
NOT scaled up: shares are taken over the people who answered, and the map draws 99.74% of
the country.

Usage:
    python sources/gh.py --fetch    one metadata GET and one PxWeb POST, seconds
    python sources/gh.py            normalise from data/raw/gh/
"""

import csv
import json
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "gh")
OUT = os.path.join(ROOT, "data", "normalized", "gh.csv")

SOURCE_ID = "gh_phc_2021"
YEAR = 2021
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# NOTE THE MISSING /pxweb -- see the module docstring. /pxweb/api/v1/en/ returns HTTP 500.
API = ("https://statsbank.statsghana.gov.gh/api/v1/en/"
       "PHC%202021%20StatsBank/Population/religion_table.px")
RAW_JSON = "religion_table.json"

# statsbank.statsghana.gov.gh omits its TLS intermediate certificate, so curl, urllib and
# requests all fail identically with "unable to get local issuer certificate". Three
# clients failing the same way is the SERVER's fault, not this machine's (§12), which is
# `stat.gov.pl`'s case and takes `stat.gov.pl`'s fix: verification off for this one named
# host, and the bytes validated structurally instead — see the asserts in fetch() and the
# whole of check().
VERIFY_TLS = False

TOTAL_CAT = "Total"
PARENT_CAT = "Christian"
CHRISTIAN_CHILDREN = [
    "Protestant (Anglican, Lutheran, Presbyterian,  Methodist, etc.)",
    "Catholic",
    "Pentecostal/ Charismatic",
    "Other Christian",
]
FAMILIES = ["Christian", "Islam", "Traditionalist", "No Religion", "Other Religion"]

# GSS's own 16 regions, in the order the cube lists them. They are the block headers of the
# positional parse, so an added or renamed region must fail the run rather than silently
# fold a region's districts into its predecessor (Romania's county-header bug, §12).
REGIONS = ["Western", "Central", "Greater Accra", "Volta", "Eastern", "Ashanti",
           "Western North", "Ahafo", "Bono", "Bono East", "Oti", "Northern",
           "Savannah", "North East", "Upper East", "Upper West"]

NATIONAL = 30_753_327          # religion table universe, = 99.74% of the census count
CENSUS_POPULATION = 30_832_019  # 2021 PHC total, from population_table.px

EXPECTED_ROWS = 295
EXPECTED_REGIONS = 16
EXPECTED_DISTRICTS_261 = 261   # the MMDA tier: 255 plain districts + 6 metro parents
EXPECTED_UNITS_272 = 272       # the drawn tier: 255 plain districts + 17 sub-metros
EXPECTED_CATEGORIES = 10       # 9 categories + the universe total


def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, RAW_JSON)
    if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
        print("already have", dest)
        return

    ua = {"User-Agent": "Mozilla/5.0"}
    meta = requests.get(API, timeout=120, verify=VERIFY_TLS, headers=ua).json()
    codes = [v["code"] for v in meta["variables"]]
    if "Religious_Affiliation" not in codes or "Geographic_Area" not in codes:
        raise SystemExit(f"unexpected variables {codes} -- GSS reshaped the table")

    # Education, Locality, Sex and Age all carry elimination=true, so leaving them out of
    # the query returns their totals. Asking for them as well is 663,750 cells and PxWeb
    # answers that with HTTP 403, which looks like an auth failure and is a size limit.
    query = {"query": [{"code": c, "selection": {"filter": "all", "values": ["*"]}}
                       for c in ("Religious_Affiliation", "Geographic_Area")],
             "response": {"format": "json-stat2"}}
    print("POST", API)
    r = requests.post(API, json=query, timeout=300, verify=VERIFY_TLS, headers=ua)
    r.raise_for_status()
    doc = r.json()

    # §5a: HTTP 200 is not a download, and this host has already served an HTML error page
    # under a 200-shaped request once. Assert it is really the cube that was asked for.
    if "value" not in doc or "dimension" not in doc:
        raise SystemExit(f"not a json-stat2 cube, keys {list(doc)}")
    if sorted(doc["id"]) != ["Geographic_Area", "Religious_Affiliation"]:
        raise SystemExit(f"unexpected cube axes {doc['id']}")
    if doc["size"] != [EXPECTED_CATEGORIES, EXPECTED_ROWS] and \
       doc["size"] != [EXPECTED_ROWS, EXPECTED_CATEGORIES]:
        raise SystemExit(f"unexpected cube size {doc['size']}")

    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False)
    print(f"  {os.path.getsize(dest):,} bytes")


def _axes(doc):
    """json-stat2 is a FLAT cube in row-major order over `id`, not a table (§12)."""
    ids, sizes = doc["id"], doc["size"]
    codes, labels = {}, {}
    for d in ids:
        cat = doc["dimension"][d]["category"]
        idx = cat["index"]
        order = sorted(idx, key=lambda k: idx[k]) if isinstance(idx, dict) else list(idx)
        codes[d] = order
        labels[d] = {k: cat.get("label", {}).get(k, k) for k in order}
    if len(sizes) != len(ids):
        raise SystemExit("json-stat2 size/id mismatch")
    for d, n in zip(ids, sizes):
        if len(codes[d]) != n:
            raise SystemExit(f"dimension {d}: {len(codes[d])} codes but size {n}")
    return ids, sizes, codes, labels


def _nest(geo, total):
    """Classify all 295 geography rows into country / region / district / metro / submetro.

    Positional: the cube lists Ghana, then each region immediately followed by its own
    districts. That is the only structure there is — the source publishes no codes and no
    level column — so it is asserted rather than assumed: every region's children must sum
    to it, and the regions to Ghana, or the run stops.

    A metro parent is a row ending in a parenthesised acronym that some LATER ROW IN THE
    SAME REGION uses as a `ACR-` prefix. Both halves matter:
      * the region scope, because `TMA` is Tema in Greater Accra and Tamale in Northern;
      * "some row uses it", because `Nkwanta North (Kpassa)` is a parenthesised alias and
        not a metro, and has no children to find.
    """
    region_of, order = {}, []
    cur = None
    for g in geo:
        if g == "Ghana":
            continue
        if g in REGIONS:
            cur = g
            order.append(g)
            continue
        if cur is None:
            raise SystemExit(f"row {g!r} appears before any region header")
        region_of[g] = cur
    if order != REGIONS:
        raise SystemExit(f"region headers are {order}, expected {REGIONS} -- GSS has "
                         "renamed or reordered a region and the positional parse is stale")

    parent_of, submetro = {}, []
    for g in geo:
        m = re.search(r"\(([A-Za-z]+)\)\s*$", g)
        if not m or g in REGIONS:
            continue
        acr, reg = m.group(1), region_of[g]
        kids = [k for k in geo if region_of.get(k) == reg and k.startswith(acr + "-")]
        if not kids:
            continue                     # a parenthesised alias, not a metro
        s = sum(total[k] for k in kids)
        if s != total[g]:
            raise SystemExit(f"{g}: {len(kids)} sub-metros sum to {s:,} but the parent is "
                             f"{total[g]:,}")
        for k in kids:
            if k in parent_of:
                raise SystemExit(f"{k} claimed by two metros: {parent_of[k]} and {g}")
            parent_of[k] = g
        submetro += kids

    metro = sorted(set(parent_of.values()))
    level = {"Ghana": "country"}
    for g in REGIONS:
        level[g] = "region"
    for g in region_of:
        level[g] = ("submetro" if g in submetro else
                    "metro" if g in metro else "district")

    # regions partition the country, and each region's children partition it
    for reg in REGIONS:
        kids = [g for g, r in region_of.items()
                if r == reg and level[g] in ("district", "metro")]
        s = sum(total[g] for g in kids)
        if s != total[reg]:
            raise SystemExit(f"{reg}: districts sum to {s:,}, region row is {total[reg]:,}")
    s = sum(total[r] for r in REGIONS)
    if s != total["Ghana"]:
        raise SystemExit(f"regions sum to {s:,}, Ghana row is {total['Ghana']:,}")

    return level, region_of, parent_of


def read():
    p = os.path.join(RAW, RAW_JSON)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch first")
    with open(p, encoding="utf-8") as fh:
        doc = json.load(fh)

    ids, sizes, codes, labels = _axes(doc)
    cat_dim = "Religious_Affiliation"
    geo_dim = "Geographic_Area"
    if set(ids) != {cat_dim, geo_dim}:
        raise SystemExit(f"unexpected axes {ids}")

    stride, acc = {}, 1
    for d, n in zip(reversed(ids), reversed(sizes)):
        stride[d] = acc
        acc *= n
    values = doc["value"]

    def at(cat, g):
        return values[stride[cat_dim] * codes[cat_dim].index(cat)
                      + stride[geo_dim] * codes[geo_dim].index(g)]

    geo = codes[geo_dim]
    cats = codes[cat_dim]
    if len(cats) != EXPECTED_CATEGORIES:
        raise SystemExit(f"{len(cats)} categories, expected {EXPECTED_CATEGORIES}")
    if TOTAL_CAT not in cats or PARENT_CAT not in cats:
        raise SystemExit(f"categories reshaped: {cats}")

    total = {g: at(TOTAL_CAT, g) for g in geo}
    level, region_of, parent_of = _nest(geo, total)

    rows = []
    for g in geo:
        lv = level[g]
        for c in cats:
            n = at(c, g)
            if n is None:
                continue
            note = f"level={lv}"
            if lv in region_of:
                pass
            if g in region_of:
                note += f"; region={region_of[g]}"
            if g in parent_of:
                note += f"; submetro of {parent_of[g]}"
            if c == TOTAL_CAT:
                note += "; universe total, not a religion category"
            elif c == PARENT_CAT:
                note += ("; parent of the four Christian categories, published beside them "
                         "and equal to their sum")
            rows.append({"geo_id": g, "geo_level": lv, "geo_name": g,
                         "source_category": c, "count": int(n), "basis": BASIS,
                         "year": YEAR, "source_id": SOURCE_ID, "note": note})
    return rows, cats


def check(rows, cats):
    ok = True

    levels = {}
    for r in rows:
        levels.setdefault(r["geo_level"], set()).add(r["geo_id"])
    counts = {lv: len(v) for lv, v in levels.items()}
    want = {"country": 1, "region": EXPECTED_REGIONS, "metro": 6, "submetro": 17,
            "district": EXPECTED_DISTRICTS_261 - 6}
    for lv, n in want.items():
        good = counts.get(lv) == n
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {lv:<10} {counts.get(lv, 0):>4} units "
              f"(expected {n})")
    drawn = counts.get("district", 0) + counts.get("submetro", 0)
    good = drawn == EXPECTED_UNITS_272
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} drawn tier {drawn} units "
          f"(district + submetro, expected {EXPECTED_UNITS_272})")

    nat = {r["source_category"]: r["count"] for r in rows if r["geo_level"] == "country"}
    good = nat.get(TOTAL_CAT) == NATIONAL
    ok &= good
    print(f"\n  {'OK ' if good else 'BAD'} national universe {nat.get(TOTAL_CAT):,} "
          f"(expected {NATIONAL:,})")
    gap = CENSUS_POPULATION - nat.get(TOTAL_CAT, 0)
    print(f"      the census counted {CENSUS_POPULATION:,}; {gap:,} people "
          f"({100.0 * gap / CENSUS_POPULATION:.2f}%) did not answer and are absent from "
          "this\n      table entirely. Not scaled up — see the module docstring.")

    # Exactness is available here: GSS neither rounds nor suppresses this table.
    for tier, name in [(["district", "metro"], "the 261 MMDAs"),
                       (["district", "submetro"], "the 272 drawn units"),
                       (["region"], "the 16 regions")]:
        bad = []
        for c in cats:
            s = sum(r["count"] for r in rows
                    if r["geo_level"] in tier and r["source_category"] == c)
            if s != nat[c]:
                bad.append((c, s, nat[c]))
        good = not bad
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} all {len(cats)} categories sum from "
              f"{name} to the national row exactly")
        for c, s, n in bad:
            print(f"        {c}: {s:,} vs {n:,}")

    kids = sum(nat[c] for c in CHRISTIAN_CHILDREN)
    good = kids == nat[PARENT_CAT]
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the four Christian categories sum to "
          f"'Christian' ({kids:,} vs {nat[PARENT_CAT]:,}) — so the parent is a duplicate "
          "and\n      needs no remainder emitted (contrast Hungary, §12)")
    fam = sum(nat[c] for c in FAMILIES)
    good = fam == nat[TOTAL_CAT]
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the five families partition the universe "
          f"({fam:,})")

    # ...and the same two identities on EVERY unit, not just nationally. A parent/child
    # relation that holds at the top and fails in one district is the shape of a misparse.
    bad_kids = bad_fam = 0
    by_unit = {}
    for r in rows:
        by_unit.setdefault(r["geo_id"], {})[r["source_category"]] = r["count"]
    for g, d in by_unit.items():
        if sum(d[c] for c in CHRISTIAN_CHILDREN) != d[PARENT_CAT]:
            bad_kids += 1
        if sum(d[c] for c in FAMILIES) != d[TOTAL_CAT]:
            bad_fam += 1
    ok &= not (bad_kids or bad_fam)
    print(f"  {'OK ' if not bad_kids else 'BAD'} the Christian identity holds on all "
          f"{len(by_unit)} rows ({bad_kids} failures)")
    print(f"  {'OK ' if not bad_fam else 'BAD'} the family identity holds on all "
          f"{len(by_unit)} rows ({bad_fam} failures)")

    print(f"\n  {len(rows):,} rows. Categories, national:")
    for c in cats:
        n = nat[c]
        mark = ("  <- universe" if c == TOTAL_CAT else
                "  <- parent, not drawn" if c == PARENT_CAT else "")
        print(f"    {n:>10,}  {100.0 * n / NATIONAL:5.2f}%  {c}{mark}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, cats = read()
    check(rows, cats)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
