"""Cyprus — Census of Population and Housing 2021 (CYSTAT), religion allocated to communities.

Reads (or fetches) data/raw/cy/ and writes data/normalized/cy.csv.

**CYPRUS PUBLISHES RELIGION AND PUBLISHES NO GEOGRAPHY FOR IT.** CYSTAT-DB carries the 2021
census in six tables under *Population - Language, Religion, Ethnic / Religious Group*, and
the split is exact and deliberate: **language is tabulated by district three ways** (1891610E,
1891613E, 1891616E) and **religion is tabulated by citizenship group, by country-of-birth
group and by sex, and by nothing spatial at all** (1891632E, 1891635E, 1891642E). The whole
2011 branch of the database is three tables and has no religion in it either. So the twelve
religion counts are national, they match UNSD Demographic Yearbook table 28 to the person on
all twelve, and there is no district row anywhere to draw them on.

**WHAT IS DRAWN INSTEAD IS CYSTAT'S OWN ARITHMETIC, NOT AN OUTSIDE COEFFICIENT.** Table
1891632E gives religion crossed with **citizenship group** — Cypriots, other EU citizens,
non-EU citizens, not stated — and table 1891213E gives those same four groups by
**municipality/community**, 396 of them. So

    count(religion, community) = SUM over the four groups of
                                 P(religion | group) x N(group, community)

Every input is the same census, the same office, the same universe and the same reference day.
The national totals come back exactly, because the four group populations sum to the four
group populations. **Nothing here is fitted and nothing is imported**, which is the difference
between this and the nationality derivations in `sources/gr.py`, `es.py`, `fr.py` and `it.py`:
those countries have to guess what a Romanian resident believes, and Cyprus has counted.

**WHAT THE ALLOCATION CANNOT DO, STATED PLAINLY.** It moves religion around the island only in
so far as the citizenship mix moves. That buys a lot for the migrant religions — Islam runs
0.9% to 7.8% across communities, Buddhism 0.3% to 3.6%, Roman Catholicism 0.6% to 4.6%,
Orthodoxy 62% to 82% — and **nothing at all for the three Cypriot minorities**, because they
are inside the `Cypriots` group and that group has one profile. The Armenian church comes out
flat at 0.21-0.23% in every community and the Maronite church at 0.40-0.56%, when both are in
fact concentrated in Lefkosia and Lemesos. Together they are 6,511 people, about six dots at
1:1,000, and no published table splits them by district; the alternative was to invent a
coefficient for them off the language table, which would have been a worse trade. Said again
in `taxonomy/cy2021.py`'s REVIEW and in the country note.

**THE JOIN IS AN INTEGER AND IT IS EXACT.** PxWeb's value CODES for the community dimension
are Cyprus's own LAU codes — `1000` Lefkosia, `1010` Agios Dometios — and Eurostat's GISCO
LAU 2021 layer carries the same code with the same Latin name. All 396 census communities are
present in GISCO, and **all 396 Latin names agree character for character with CYSTAT's English
labels**, so the name agreement is asserted as a free check on a join made on the code.
[[reference_name_join_wrong_neighbour]] is the reason it is done that way round.

**ONE FIFTH OF THE ISLAND'S DENSITY IS MISSING AND IT IS NOT THIS FILE'S FAULT.** The census
covers the government-controlled area only. GISCO has 615 Cypriot LAUs; the census enumerates
396 of them; the 219 left over are the north (182 with population `n.a.` in Eurostat's own
workbook, including all 47 communities of **Keryneia district, which has no code in the census
at all** — the district list runs 1, 3, 4, 5, 6) plus 37 with population zero, which are the
Turkish Cypriot villages of Pafos and Larnaka abandoned in 1974 and the uninhabited Troodos
summit. 5,846 km² are drawn of the island's 9,249.

**THE ONE MEASURED RELIGION GEOGRAPHY CYPRUS HAS EVER PUBLISHED IS 2001, AND IT IS USED AS A
CHECK RATHER THAN AS THE SOURCE.** `sources/cy_2001.py` reads Table 29 of the 2001 census
Volume 1, *Population by sex, religion, district and urban/rural area*, which is religion by
the same five districts and is the only such table in any Cypriot census — 2011 publishes
religion by age, by country of birth and by citizenship and by nothing spatial, exactly as
2021 does. It is not drawn because it is twenty years stale in the one dimension that moved:
Cyprus was **94.8%** Orthodox in 2001 and is 74.5% now, and the whole difference is the
280,000 people who arrived in between. What it can still do is measure the assumption this
file rests on, which is that religion does not vary between districts within a citizenship
group. See `sources/cy.md` §5 for what it says. Below district there is only the **1960**
census, which counts religious groups village by village across the whole island including
the north; it is a 20-page scan with no text layer and is recorded in `sources/cy.md` §6 for
whoever wants it.

**AND THE QUESTION WAS OPTIONAL.** 159,835 people, 17.3%, are `Not recorded/Not stated`, which
is far more than any neighbour and is the largest single thing this map does not know about
Cyprus. It is excluded rather than scaled up, per `taxonomy/cy2021.py`'s EXCLUDED, so 763,546
of 923,381 are drawn.

Usage:
    python sources/cy.py --fetch    five small JSON cubes from cystatdb.cystat.gov.cy
    python sources/cy.py            normalise from data/raw/cy/
"""

import argparse
import csv
import itertools
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

RAW = os.path.join(ROOT, "data", "raw", "cy")
OUT = os.path.join(ROOT, "data", "normalized", "cy.csv")

SOURCE_ID = "cy_census_2021"
YEAR = 2021
BASIS = "nationality_derived"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# CYSTAT-DB is classic ASP.NET PxWeb and its JSON API is at /api/v1/, NOT at the /pxweb/api/v1/
# that every other PxWeb here uses -- /pxweb/api/v1/en/ returns a 500 whose body is the "your
# saved query can not be found" page, because the server reads the path as a saved query.
# [[reference_spa_hidden_apis]] in its mildest form: the portal is not blocked, the prefix is.
API = "https://cystatdb.cystat.gov.cy/api/v1/en/8.CYSTAT-DB"
CENSUS = "Population/Census of Population and Housing 2021/Population"
LRE = f"{CENSUS}/Population - Language, Religion, Ethnic Religious Group"
CCB = f"{CENSUS}/Population - Country of Citizenship, Country of Birth"

TABLES = {
    # the two that do the work
    "rel_citizen.json": f"{LRE}/1891632E.px",   # religion x citizenship group x sex, national
    "cit_comm.json":    f"{CCB}/1891213E.px",   # citizenship group x sex x district/community
    # kept because they are the checks, and each is a few kB
    "rel_birth.json":   f"{LRE}/1891635E.px",   # religion x country-of-birth group; second
                                                #   allocator, used only to bound the first
    "lang_district.json": f"{LRE}/1891616E.px",  # language x district; the geography religion
                                                #   does not have, parsed and NOT drawn
    "ethnic.json":      f"{LRE}/1891642E.px",   # ethnic/religious group; the constitutional
                                                #   category, national only
}

# The four citizenship groups. CYSTAT spells them differently in the two tables -- the national
# one footnotes Cypriots as `Cypriots (2)` and hyphenates `Non-European`, the community one
# stars it as `Cypriots *` and does not. Mapped explicitly rather than normalised, so a
# re-labelled edition fails here instead of silently dropping a group.
GROUPS = {
    "Cypriots *": "Cypriots (2)",
    "Other European Union citizens": "Other European Union citizens",
    "Non European Union citizens": "Non-European Union citizens",
    "Not stated": "Not stated",
}

# UNSD Demographic Yearbook table 28's Cyprus 2021 return, which is a return CYSTAT forwarded
# rather than anything read off this database. Its 13 rows split what CYSTAT-DB lumps as
# `Not recorded/Not stated` into 146,943 not specified and 12,892 not stated; the other
# eleven are asserted category by category below.
UNSD_2021 = {
    "Christian Orthodox": 688075,
    "Armenian church": 2025,
    "Maronite church": 4486,
    "Roman Catholic": 13860,
    "Muslim": 19534,
    "Anglican/Protestant": 9621,
    "Buddhist": 7868,
    "Sikh": 2260,
    "Hindu": 1681,
    "Other Religion": 4545,
    "Atheist/No Religion": 9591,
    "Not recorded/Not stated (3)": 146943 + 12892,
}
NATIONAL_TOTAL = 923381


# ---------------------------------------------------------------- fetch

def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    os.makedirs(RAW, exist_ok=True)
    ua = {"User-Agent": "Mozilla/5.0"}
    for name, path in TABLES.items():
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 2_000:
            print("already have", dest)
            continue
        url = f"{API}/{path}"
        meta = requests.get(url, timeout=120, verify=False, headers=ua)
        meta.raise_for_status()
        codes = [v["code"] for v in meta.json()["variables"]]
        query = {"query": [{"code": c, "selection": {"filter": "all", "values": ["*"]}}
                           for c in codes],
                 "response": {"format": "json-stat2"}}
        print("POST", name, codes)
        r = requests.post(url, json=query, timeout=300, verify=False, headers=ua)
        r.raise_for_status()
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(r.text)
        print(f"  {os.path.getsize(dest):,} bytes")


# ---------------------------------------------------------------- json-stat2

def _load(name):
    p = os.path.join(RAW, name)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch first")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _order(js, dim):
    """(code, label) pairs for one dimension, in the cube's own storage order."""
    cat = js["dimension"][dim]["category"]
    idx = cat["index"]
    keys = sorted(idx, key=lambda k: idx[k]) if isinstance(idx, dict) else list(idx)
    return [(k, cat["label"][k]) for k in keys]


def _cells(js):
    """Walk a json-stat2 cube: yield (dim -> (code, label), value).

    One flat row-major `value` array over the dimensions in `id`. Anything that reads it as
    rows is guessing.
    """
    ids = js["id"]
    dims = [_order(js, d) for d in ids]
    vals = js["value"]
    for i, combo in enumerate(itertools.product(*dims)):
        yield dict(zip(ids, combo)), vals[i]


# ---------------------------------------------------------------- normalise

def build():
    # ---- national: religion x citizenship group
    rc = _load("rel_citizen.json")
    nat = {}
    for key, v in _cells(rc):
        if key["SEX"][1] != "Total":
            continue
        nat[(key["RELIGION (1)"][1], key["CITIZENSHIP GROUP"][1])] = v or 0

    rels = [t for _, t in _order(rc, "RELIGION (1)") if t != "Total"]
    groups = [t for _, t in _order(rc, "CITIZENSHIP GROUP") if t != "Total"]
    if sorted(groups) != sorted(GROUPS.values()):
        sys.exit(f"!! citizenship groups changed: {groups}")

    if nat[("Total", "Total")] != NATIONAL_TOTAL:
        sys.exit(f"!! national total is {nat[('Total', 'Total')]:,}, "
                 f"expected {NATIONAL_TOTAL:,}")

    print("national religion counts, against UNSD table 28:")
    for r in rels:
        got = nat[(r, "Total")]
        want = UNSD_2021.get(r)
        flag = "ok" if got == want else f"!! UNSD has {want:,}"
        print(f"  {got:>9,}  {r:<30} {flag}")
        if got != want:
            sys.exit(f"!! {r}: CYSTAT-DB {got:,} vs UNSD {want:,}")
    if sum(nat[(r, "Total")] for r in rels) != NATIONAL_TOTAL:
        sys.exit("!! the religion categories do not partition the national total")

    # P(religion | citizenship group)
    share = {g: {r: nat[(r, g)] / nat[("Total", g)] for r in rels} for g in groups}

    # ---- community: citizenship group
    cc = _load("cit_comm.json")
    dim = "DISTRICT, MUNICIPALITY/COMMUNITY"
    comm, names = {}, {}
    dist_totals = {}
    for key, v in _cells(cc):
        if key["SEX"][1] != "Total":
            continue
        code, label = key[dim]
        g = key["CITIZENSHIP GROUP"][1]
        if code == "TOTAL":
            if g == "Total" and (v or 0) != NATIONAL_TOTAL:
                sys.exit(f"!! community table totals {v:,}, expected {NATIONAL_TOTAL:,}")
            continue
        if len(code) == 1:                      # a district header row, e.g. `1`
            if g == "Total":
                dist_totals[label] = v or 0
            continue
        if g == "Total":
            names[code] = label
            continue
        comm.setdefault(code, {})[GROUPS[g]] = v or 0

    print(f"\n{len(comm)} municipalities/communities, "
          f"{len(dist_totals)} districts: "
          + ", ".join(f"{k.split()[0].title()} {v:,}" for k, v in dist_totals.items()))

    # every community's four groups sum to its own published total
    tot_from_groups = 0
    for code, d in comm.items():
        if sorted(d) != sorted(GROUPS.values()):
            sys.exit(f"!! {code} {names[code]}: groups {sorted(d)}")
        tot_from_groups += sum(d.values())
    if tot_from_groups != NATIONAL_TOTAL:
        sys.exit(f"!! communities sum to {tot_from_groups:,}, expected {NATIONAL_TOTAL:,}")
    if sum(dist_totals.values()) != NATIONAL_TOTAL:
        sys.exit("!! districts do not sum to the national total")

    # ---- allocate
    rows = []
    for code in sorted(comm):
        d = comm[code]
        for r in rels:
            n = sum(share[g][r] * d[g] for g in groups)
            if n <= 0:
                continue
            rows.append({
                "geo_id": code,
                "geo_level": "community",
                "geo_name": names[code],
                "source_category": r,
                "count": round(n, 4),
                "basis": BASIS,
                "year": YEAR,
                "source_id": SOURCE_ID,
                "note": "national religion x citizenship group (CYSTAT 1891632E) applied to "
                        "this community's citizenship groups (1891213E)",
            })

    # the allocation is exact by construction; assert it rather than believe it
    print("\nreconciling the allocation against the national counts:")
    worst = 0.0
    for r in rels:
        got = sum(x["count"] for x in rows if x["source_category"] == r)
        want = nat[(r, "Total")]
        worst = max(worst, abs(got - want))
        print(f"  {want:>9,}  {r:<30} allocated {got:>12,.2f}")
    if worst > 0.5:
        sys.exit(f"!! allocation is off by {worst:.4f} people on some category")
    print(f"  largest discrepancy {worst:.6f} people")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}  ({len(rows):,} rows, {len(comm)} communities)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true", help="download the JSON cubes first")
    args = ap.parse_args()
    if args.fetch:
        fetch()
    build()


if __name__ == "__main__":
    main()
