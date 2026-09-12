"""Kosovo — ASK, Census 2024, religion by municipality.

Reads (or fetches) data/raw/xk/ and writes data/normalized/xk.csv.

One PxWeb table, **`census2024_10.px`** — *Population by religion and sex at country and
municipal level for the years 2011 and 2024* — 6 religions plus a total, over 38
municipalities, for **both censuses**. 10 KB, no key, no login, no wall.

`askdata.rks-gov.net` is an ordinary open PxWeb v1 endpoint. The catalogue walk that found
this nearly missed it for a dull reason worth recording: **its root returns `dbid` where
every other PxWeb here returns `id`**, so a walker keyed on `id` reads the root as a list of
nameless nodes and descends into none of them. Two offices in this repo do that (Kosovo and
Moldova) and both looked empty for the same wrong reason.

**THE BLANK CELLS ARE TRUE ZEROS, AND THE PARTITION IS THE PROOF.** 162 of the 1,638 cells
come back with `status: ':'` and no value. That is PxWeb's "not available", which elsewhere
on this map means disclosure control (Lithuania, §9q) — reading it as zero there would
delete people. Here it cannot be suppression: with every blank read as zero, the 38
municipalities sum to the published national row **exactly, category by category**, and the
six categories sum to each unit's own total exactly. A suppressed positive value would break
both. `check()` asserts it rather than assuming it, so a vintage that starts truly
suppressing will fail instead of quietly losing people.

**AND THE THING THAT MATTERS MORE THAN ANY OF THAT: THE NORTH BOYCOTTED THE CENSUS.**
Kosovo's Serb population largely refused enumeration, and in the four northern
Serb-majority municipalities the result is not an undercount but a different population.
Zubin Potok returns **763 people, of whom 681 are recorded Muslim and 82 Orthodox**; Zveçan
returns 434, and North Mitrovica 2,326. These are municipalities usually put at 6,000-30,000
people and overwhelmingly Serb. What was enumerated there is disproportionately the
non-Serb minority, so the published religious composition of the north is **inverted**, not
merely thin. In 2011 the same four were not enumerated at all and the table is null for
them, which is at least legible; 2024 replaces a hole with a wrong number.

That is a spec §14 question and not a technical one. `NORTH` below names the four, `check()`
reports them every run, and `countries.py` decides what to do with them.

Usage:
    python sources/xk.py --fetch    one GET + one POST, 10 KB
    python sources/xk.py            normalise from data/raw/xk/
"""

import csv
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "xk")
OUT = os.path.join(ROOT, "data", "normalized", "xk.csv")

SOURCE_ID = "xk_census_2024"
YEAR = 2024
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

BASE = ("https://askdata.rks-gov.net/api/v1/en/ASKdata/Census population/"
        "1_Demographic_Characteristics/")
TABLE = "census2024_10.px"
API = BASE + TABLE
CUBE = os.path.join(RAW, "census2024_10.json")

# The two ETHNICITY tables. Identical to each other everywhere except the four northern
# municipalities, where `_63` restores the population the census did not reach. ASK's own
# correction, published by ASK; see §5 of sources/xk.md.
ETH_RAW = "census2024_05.px"           # as enumerated
ETH_EST = "census2024_63.px"           # "(with estimation)"
ETH_RAW_CUBE = os.path.join(RAW, "census2024_05.json")
ETH_EST_CUBE = os.path.join(RAW, "census2024_63.json")
NORTH_OUT = os.path.join(ROOT, "data", "normalized", "xk_north.csv")
NORTH_COLUMNS = ["geo_id", "geo_name", "ethnicity", "enumerated", "estimated", "added",
                 "source_id", "note"]

DRAWN_YEAR = "2024"
BOTH_SEXES = "Total"
TOTAL_CAT = "Total"
COUNTRY_LABEL = "KOSOVA"
SERB = "Serb"

NATIONAL = 1_585_566          # enumerated population, 2024
NATIONAL_ESTIMATED = 1_602_515
EXPECTED_MUNICIPALITIES = 38

# The four northern municipalities. Serb-majority, and the 2024 enumeration reached almost
# none of that population; see the module docstring. Named here so every consumer can find
# them rather than re-deriving the list from a news story.
NORTH = ("Leposaviq", "Zubin Potok", "Zveqan", "Mitrovicë e Veriut")


def _pull(table, cube, want_dims):
    import requests

    if os.path.exists(cube) and os.path.getsize(cube) > 5_000:
        print("already have", cube)
        return
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    url = BASE + table
    # TLS verifies normally on askdata.rks-gov.net; nothing is disabled here (§9h).
    print("GET ", url)
    meta = requests.get(url, headers=ua, timeout=120)
    meta.raise_for_status()
    dims = sorted(v["code"] for v in meta.json()["variables"])
    # ASSERT THE ARITY, NOT THE NAMES. These three tables sit in one folder and do not agree
    # on their own dimension codes: the religion table calls its geography `KOMUNA` and the
    # two ethnicity tables call it `Komuna`, and where one says `Etniciteti` the other says
    # `Përkatësia etnike`. A name check here would fail on a table that is perfectly fine,
    # and the identification that matters is done by LABEL in `_ethnicity`, which is what
    # survives a publisher renaming a code.
    if len(dims) != want_dims:
        raise SystemExit(f"{table}: expected {want_dims} dimensions, got {len(dims)}: "
                         f"{dims}")

    print("POST", url, "(empty query = every cell)")
    r = requests.post(url, headers=ua, timeout=300,
                      json={"query": [], "response": {"format": "json-stat2"}})
    r.raise_for_status()
    doc = r.json()
    # §5a: HTTP 200 is not a download. PxWeb answers a bad query with a 200 and an error body.
    for key in ("id", "size", "dimension", "value"):
        if key not in doc:
            raise SystemExit(f"{table}: not a json-stat2 dataset, keys {list(doc)}")
    with open(cube, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False)
    print(f"  {os.path.getsize(cube):,} bytes")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for table, cube in ((TABLE, CUBE), (ETH_RAW, ETH_RAW_CUBE), (ETH_EST, ETH_EST_CUBE)):
        _pull(table, cube, 4)


def _load(cube=None):
    cube = cube or CUBE
    if not os.path.exists(cube):
        raise SystemExit(f"missing {cube} -- run with --fetch first")
    with open(cube, encoding="utf-8") as fh:
        return json.load(fh)


def _ethnicity(cube):
    """{municipality name: {ethnicity: count}} for 2024, both sexes, from an ethnicity cube.

    Written against the DIMENSION LABELS rather than their order, because the two ethnicity
    tables do not agree on it — `census2024_05` is Sex/Municipality/Year/Ethnicity and the
    religion table is Municipality/Year/Sex/Religion. Indexing one by the other's positions
    silently transposes the country.
    """
    doc = _load(cube)
    ids, size, dim, val = doc["id"], doc["size"], doc["dimension"], doc["value"]
    order = {k: list(dim[k]["category"]["index"].keys()) for k in ids}
    label = {k: dim[k]["category"]["label"] for k in ids}

    def axis(*wanted):
        for k in ids:
            if set(wanted) <= set(label[k].values()):
                return k
        raise SystemExit(f"no dimension in {ids} carrying all of {wanted}")

    k_geo = axis(COUNTRY_LABEL)
    k_year = axis(DRAWN_YEAR)
    k_eth = axis(SERB, "Albanian")
    k_sex = next(k for k in ids if k not in (k_geo, k_year, k_eth))

    pick = {k_year: DRAWN_YEAR, k_sex: BOTH_SEXES}
    fixed = {k: next(n for n, c in enumerate(order[k]) if label[k][c] == v)
             for k, v in pick.items()}

    out = {}
    for gi, gc in enumerate(order[k_geo]):
        row = {}
        for ei, ec in enumerate(order[k_eth]):
            pos = dict(fixed, **{k_geo: gi, k_eth: ei})
            n = 0
            for k, s in zip(ids, size):
                n = n * s + pos[k]
            v = val[n]
            row[label[k_eth][ec].strip()] = 0 if v is None else int(v)
        out[label[k_geo][gc].strip()] = row
    return out


def read_north():
    """ASK's own estimate of the population the northern boycott left out.

    Returns rows for `xk_north.csv` plus the checks' raw material. Nothing here decides what
    to DO with the estimate — that is countries.py's call and spec §14's.
    """
    raw, est = _ethnicity(ETH_RAW_CUBE), _ethnicity(ETH_EST_CUBE)

    # The estimate must be a correction to four municipalities and nothing else. If ASK ever
    # widens it, this build must stop rather than quietly adopt a different scope.
    differing = sorted(g for g in raw if raw[g] != est.get(g))
    expected = sorted(NORTH + (COUNTRY_LABEL,))
    if differing != expected:
        raise SystemExit(
            f"the estimated table differs from the enumerated one on {differing}, not on "
            f"{expected}. ASK has changed the scope of its estimate; re-read the metadata "
            "before drawing any of it (spec §14).")

    rows = []
    for name in NORTH:
        for eth in sorted(raw[name], key=lambda e: -est[name].get(e, 0)):
            a, b = raw[name][eth], est[name][eth]
            if a == b and eth != SERB:
                continue
            note = "ASK's own estimate for a municipality the 2024 census did not reach"
            if eth == SERB:
                note += "; this is the row countries.py derives Orthodox dots from (§14.5)"
            rows.append({"geo_id": name, "geo_name": name, "ethnicity": eth,
                         "enumerated": a, "estimated": b, "added": b - a,
                         "source_id": SOURCE_ID, "note": note})
    return rows, raw, est


def read():
    doc = _load()
    ids, size, dim, val = doc["id"], doc["size"], doc["dimension"], doc["value"]
    order = {k: list(dim[k]["category"]["index"].keys()) for k in ids}
    label = {k: dim[k]["category"]["label"] for k in ids}

    def at(i, j, k, l):
        return val[((i * size[1] + j) * size[2] + k) * size[3] + l]

    yi = next(n for n, c in enumerate(order["Viti"]) if label["Viti"][c] == DRAWN_YEAR)
    si = next(n for n, c in enumerate(order["Gjinia"]) if label["Gjinia"][c] == BOTH_SEXES)

    rows, stats = [], {"n": 0, "blank": 0}
    for i, km in enumerate(order["KOMUNA"]):
        name = label["KOMUNA"][km]
        level = "country" if name == COUNTRY_LABEL else "municipality"
        for l, rc in enumerate(order["RELIGJIONI"]):
            cat = label["RELIGJIONI"][rc]
            n = at(i, yi, si, l)
            note = f"level={level}; code={rc}"
            if n is None:
                # A blank, which check() proves is a true zero rather than a suppression.
                stats["blank"] += 1
                n = 0
                note += "; published blank, verified a true zero by the exact partition"
            stats["n"] += 1
            if cat == TOTAL_CAT:
                note += "; universe total, not a religion category"
            if name in NORTH:
                note += ("; NORTHERN MUNICIPALITY — the Serb population largely boycotted "
                         "the 2024 census and this composition is not representative")
            rows.append({"geo_id": str(km), "geo_level": level, "geo_name": name,
                         "source_category": cat, "count": int(n), "basis": BASIS,
                         "year": YEAR, "source_id": SOURCE_ID, "note": note})
    return rows, stats


def check(rows, stats):
    ok = True
    print(f"  {stats['n']:,} cells for {DRAWN_YEAR}, both sexes; "
          f"{stats['blank']:,} published blank and read as zero")

    muni = {r["geo_id"] for r in rows if r["geo_level"] == "municipality"}
    good = len(muni) == EXPECTED_MUNICIPALITIES
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {len(muni)} municipalities "
          f"(expected {EXPECTED_MUNICIPALITIES})")

    nat = {r["source_category"]: r["count"] for r in rows if r["geo_level"] == "country"}
    good = nat.get(TOTAL_CAT) == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national total {nat.get(TOTAL_CAT):,} "
          f"(published {NATIONAL:,})")

    parts = sum(v for k, v in nat.items() if k != TOTAL_CAT)
    good = parts == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the 6 categories partition the country exactly "
          f"({parts:,})")

    # ---- the blank-is-zero proof (§3.8): per category, the 38 must reproduce the country
    print("\n  every category's 38 municipalities must sum to its national figure —\n"
          "  this is what proves the blanks are zeros and not disclosure control:")
    worst = 0
    for cat in sorted(nat, key=lambda k: -nat[k]):
        s = sum(r["count"] for r in rows
                if r["geo_level"] == "municipality" and r["source_category"] == cat)
        gap = nat[cat] - s
        worst = max(worst, abs(gap))
        flag = "OK " if gap == 0 else "BAD"
        if gap:
            ok = False
        print(f"    {flag} {cat:<26} {s:>10,}  gap {gap:>7,}")
    if worst:
        print("    BAD a non-zero gap means at least one blank hides a real number. "
              "STOP: reading it as zero deletes people (spec §3.8).")

    # ---- the north, reported every run because it is the reason to read the note ----
    print("\n  THE FOUR NORTHERN MUNICIPALITIES — a boycott, not an undercount:")
    tot = {}
    for r in rows:
        if r["geo_level"] == "municipality" and r["geo_name"] in NORTH:
            tot.setdefault(r["geo_name"], {})[r["source_category"]] = r["count"]
    north_pop = 0
    for name in NORTH:
        d = tot.get(name, {})
        t = d.get(TOTAL_CAT, 0)
        north_pop += t
        isl, orth = d.get("Islam", 0), d.get("Orthodox", 0)
        print(f"    {name:<20} enumerated {t:>6,}   Islam {isl:>6,}  Orthodox {orth:>6,}"
              f"   ({100.0 * orth / t if t else 0:.0f}% Orthodox)")
    print(f"    {north_pop:,} people enumerated across all four — "
          f"{100.0 * north_pop / NATIONAL:.2f}% of the country. These four are usually put "
          "at 40,000-60,000 people and are overwhelmingly Serb.")
    print("    countries.py decides whether they are drawn; spec §14.")

    print(f"\n  {len(rows):,} rows. Categories, national:")
    for cat, n in sorted(nat.items(), key=lambda kv: -kv[1]):
        mark = "  <- universe" if cat == TOTAL_CAT else ""
        print(f"    {n:>10,}  {100.0 * n / NATIONAL:6.2f}%  {cat}{mark}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def check_north(rows, raw, est):
    """The estimate is ASK's; these checks are about whether it means what it looks like."""
    ok = True
    print("\n  ---- ASK's OWN ESTIMATE OF THE BOYCOTTED POPULATION (census2024_63) ----")

    nat_r, nat_e = raw[COUNTRY_LABEL], est[COUNTRY_LABEL]
    good = nat_r[TOTAL_CAT] == NATIONAL and nat_e[TOTAL_CAT] == NATIONAL_ESTIMATED
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national: enumerated {nat_r[TOTAL_CAT]:,}, "
          f"estimated {nat_e[TOTAL_CAT]:,} "
          f"(+{nat_e[TOTAL_CAT] - nat_r[TOTAL_CAT]:,})")
    print(f"      Serbs:  enumerated {nat_r[SERB]:,}, estimated {nat_e[SERB]:,} "
          f"(+{nat_e[SERB] - nat_r[SERB]:,})")

    print(f"\n  {'municipality':<22} {'enum':>7} {'est':>8}   "
          f"{'Serb enum':>9} {'Serb est':>9}  {'added':>7} {'of which Serb':>14}")
    add_tot = add_serb = 0
    for name in NORTH:
        a, b = raw[name], est[name]
        d_tot = b[TOTAL_CAT] - a[TOTAL_CAT]
        d_serb = b[SERB] - a[SERB]
        add_tot += d_tot
        add_serb += d_serb
        bad = d_tot < 0 or d_serb < 0 or d_serb > d_tot
        ok &= not bad
        print(f"  {'BAD' if bad else '   '} {name:<19} {a[TOTAL_CAT]:>7,} {b[TOTAL_CAT]:>8,}"
              f"   {a[SERB]:>9,} {b[SERB]:>9,}  {d_tot:>7,} {d_serb:>14,}")
    share = add_serb / add_tot if add_tot else 0.0
    print(f"      total added {add_tot:,}, of which {add_serb:,} Serb ({share:.1%}) and "
          f"{add_tot - add_serb:,} other")

    # The whole §14.5 case rests on the addition being overwhelmingly ONE group. If a future
    # revision spreads it, the derivation stops being defensible and this must fail.
    good = share > 0.90
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the addition is {share:.1%} Serb — §14.5 needs a "
          "religiously homogeneous group, and\n      an estimate spread across ethnicities "
          "would not be one")

    # The non-Serb remainder is NOT derived and NOT filled (§3.5). Say how big it is.
    print(f"      the {add_tot - add_serb:,} non-Serb people in the estimate are left "
          "undrawn — nothing says\n      what they are, and the enumerated composition of "
          "these four is not representative")

    # Ethnicity is not religion, and the size of the disagreement is worth stating once.
    print(f"\n      for scale: nationally the census counts {nat_r[SERB]:,} Serbs and "
          "36,683 Orthodox —\n      close, and not the same question")

    if not ok:
        raise SystemExit("the northern estimate FAILED its checks")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, stats = read()
    check(rows, stats)
    north, raw, est = read_north()
    check_north(north, raw, est)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")
    with open(NORTH_OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=NORTH_COLUMNS)
        w.writeheader()
        w.writerows(north)
    print("wrote", NORTH_OUT, f"({len(north):,} rows)")


if __name__ == "__main__":
    main()
