"""England and Wales: split the 2021 census `Muslim` column into Sunni and Shi'a, for the
Muslims whose ethnic group names one origin country, at local authority.

Writes data/normalized/uk_islam_split.csv: every English and Welsh Output Area's `Muslim` count,
replaced by `Muslim: Sunni (Pew 2012)`, `Muslim: Shi'a (Pew 2012)` and the census's own `Muslim`
for the rest, summing to the census figure exactly. countries/uk.py swaps these in for the
allocated file's E&W `Muslim` rows, as it does uk_split.csv's for `Christian`.

WHY, AND WHAT ANITA APPROVED (2026-10-16)
-----------------------------------------
Until now 3.87M Muslims in England and Wales sat on bare `islam`, while France, Belgium and the
Nordic countries draw theirs on `islam.sunni`. The research is `unspecified_splits/islam.md` §4.
Anita approved the first step only: Pakistani and Bangladeshi Muslims get their origin country's
sect shares; every other ethnic group stays on `islam`. The mixed groups (Arab, Other, Other Asian,
Black African, Indian, Other White) are NOT split, because no published number says which origins
they hold among Muslims.

THE SOURCES
-----------
1. ONS Census 2021, religion (10) x ethnic group (20) by lower-tier local authority, from the
   Cantabular API (the same cross as table RM031):
       https://api.beta.ons.gov.uk/v1/population-types/UR/census-observations
           ?area-type=ltla&dimensions=religion_tb,ethnic_group_tb_20b
   Saved as data/raw/uk/ons_religion_x_ethnic_ltla.json (`--fetch` refreshes it). One of 331
   LTLAs is blocked by ONS's disclosure rules and is left unsplit.
2. Pew Research Center, *The World's Muslims: Unity and Diversity* (2012), Q31, table on printed
   p. 30, % of Muslims:  Pakistan Sunni 81, Shi'a 6;  Bangladesh Sunni 92, Shi'a 2.
   Pakistan's sample excludes FATA, Gilgit-Baltistan, AJK and unstable parts of KP and
   Balochistan; Britain's Pakistanis are largely from Mirpur in AJK, a Sunni area outside that
   frame, so the 6% Shi'a is if anything high for them (sources/branches.md).

This passes feedback "when a proxy is allowed": ethnic group is a proxy for origin, and the part
that does not match (each origin's non-Sunni share) is a published number, carried as printed.

THE METHOD
----------
Per LTLA L:  Sunni(L) = 0.81 x Pakistani Muslims + 0.92 x Bangladeshi Muslims,
             Shi'a(L) = 0.06 x Pakistani Muslims + 0.02 x Bangladeshi Muslims,
each as a share of all of L's Muslims. Every Output Area in L takes L's two shares, so the split
is uniform inside a local authority (the ONS cross is not published finer for every area). The
rest of each OA's census Muslims stays on `Muslim`.

LAD 2023 vs LTLA 2021: data/raw/uk/oa_lad.csv maps Output Areas to 2023 districts, and four
of those (Cumberland, Westmorland and Furness, North Yorkshire, Somerset) merged 2021 districts.
MERGED23 sums the ONS figures of their predecessors, which are named and asserted.

TIER: `derived`, rolling back to `Muslim` (spec §2.7a, in_split.py's reasoning): the census
counted these Muslims at the Output Area; only the branch is from the survey.

Usage:
    python sources/uk_islam.py             build data/normalized/uk_islam_split.csv
    python sources/uk_islam.py --dry-run   report, write nothing
    python sources/uk_islam.py --fetch     re-download the ONS cross first
"""

import collections
import json
import os
import sys
import urllib.request

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "data", "raw", "uk")
NORM = os.path.join(HERE, "data", "normalized")
ONS_JSON = os.path.join(RAW, "ons_religion_x_ethnic_ltla.json")
ONS_URL = ("https://api.beta.ons.gov.uk/v1/population-types/UR/census-observations"
           "?area-type=ltla&dimensions=religion_tb,ethnic_group_tb_20b")
OA_LAD = os.path.join(RAW, "oa_lad.csv")
ALLOC = os.path.join(NORM, "uk_ew_allocated.csv")
OUT = os.path.join(NORM, "uk_islam_split.csv")

COL = "Muslim"
SUNNI = "Muslim: Sunni (Pew 2012)"
SHIA = "Muslim: Shi'a (Pew 2012)"

# Pew 2012 Q31, printed p. 30: (Sunni %, Shi'a %) of Muslims, by the ONS ethnic group that
# names the origin country.
PEW_Q31 = {
    "Asian, Asian British or Asian Welsh: Pakistani": ("Pakistan", 81, 6),
    "Asian, Asian British or Asian Welsh: Bangladeshi": ("Bangladesh", 92, 2),
}

# 2023 unitary authorities that merged 2021 districts -> the 2021 names ONS reports.
MERGED23 = {
    "E06000063": ("Cumberland", ["Allerdale", "Carlisle", "Copeland"]),
    "E06000064": ("Westmorland and Furness", ["Barrow-in-Furness", "Eden", "South Lakeland"]),
    "E06000065": ("North Yorkshire", ["Craven", "Hambleton", "Harrogate", "Richmondshire",
                                      "Ryedale", "Scarborough", "Selby"]),
    "E06000066": ("Somerset", ["Mendip", "Sedgemoor", "Somerset West and Taunton",
                               "South Somerset"]),
}

NOTE_NAMED = ("level=leaf; derivation=survey_share; structure=ons_2021_religion_x_ethnic_ltla "
              "x pew_2012_q31; structure_geo=ltla:{lad}; share={share:.4f}; parent_column=Muslim")
NOTE_REST = ("level=leaf; cat=Muslim; derivation=exact_single_child; branch not named (Muslims "
             "of other ethnic groups, and Pew's unnamed share); parent_column=Muslim")
NOTE_UNSPLIT = ("level=leaf; cat=Muslim; derivation=exact_single_child; branch not split, "
                "{why}; parent_column=Muslim")
OUT_COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count", "basis",
               "year", "source_id", "tier", "note"]


def fetch():
    req = urllib.request.Request(ONS_URL, headers={"User-Agent": "religiondots/uk_islam"})
    with urllib.request.urlopen(req, timeout=300) as r:
        body = r.read()
    json.loads(body)                                  # refuse to save an error page
    tmp = ONS_JSON + ".part"
    with open(tmp, "wb") as f:
        f.write(body)
    os.replace(tmp, ONS_JSON)
    print(f"  fetched {ONS_JSON} ({len(body):,} bytes)")


def read_ons():
    """{ltla code: (name, Counter{ethnic group: Muslims})}, and the blocked count."""
    d = json.load(open(ONS_JSON, encoding="utf-8"))
    out = {}
    for o in d["observations"]:
        dims = {x["dimension_id"]: x for x in o["dimensions"]}
        if dims["religion_tb"]["option"] != COL:
            continue
        code = dims["ltla"]["option_id"]
        name = dims["ltla"]["option"]
        out.setdefault(code, (name, collections.Counter()))[1][
            dims["ethnic_group_tb_20b"]["option"]] += o["observation"]
    for eth in PEW_Q31:
        if not any(eth in c for _, c in out.values()):
            raise SystemExit(f"ethnic group {eth!r} not in the ONS file; labels changed?")
    return out, d.get("blocked_areas", 0), d.get("total_areas")


def shares_by_lad(ons):
    """{LAD23 code: (sunni share, shia share, Muslims, Pakistani+Bangladeshi Muslims)}."""
    by_name = {name: code for code, (name, _) in ons.items()}
    groups = {code: [code] for code in ons}
    for lad, (name, olds) in MERGED23.items():
        missing = [n for n in olds if n not in by_name]
        if missing:
            raise SystemExit(f"{name}: 2021 districts not in the ONS file: {missing}")
        groups[lad] = [by_name[n] for n in olds]
    out = {}
    for lad, codes in groups.items():
        c = collections.Counter()
        for code in codes:
            c.update(ons[code][1])
        m = sum(c.values())
        if m <= 0:
            continue
        su = sum(c[e] * p[1] for e, p in PEW_Q31.items()) / 100.0
        sh = sum(c[e] * p[2] for e, p in PEW_Q31.items()) / 100.0
        out[lad] = (su / m, sh / m, m, sum(c[e] for e in PEW_Q31))
    return out


def build():
    ons, blocked, total_areas = read_ons()
    print(f"  ONS religion x ethnic group: {len(ons)} LTLAs returned, {blocked} blocked "
          f"of {total_areas}")
    shares = shares_by_lad(ons)

    lad = pd.read_csv(OA_LAD, dtype=str)
    oa2lad = dict(zip(lad["OA21CD"], lad["LAD23CD"]))
    df = pd.read_csv(ALLOC, dtype={"geo_id": str}, low_memory=False)
    mus = df[(df["source_category"] == COL) & df["geo_id"].str[:1].isin(["E", "W"])].copy()
    if (mus["tier"] != "measured").any():
        raise SystemExit("some E&W `Muslim` rows are not measured; the allocation changed shape")
    if mus["geo_id"].duplicated().any():
        raise SystemExit("an Output Area carries two `Muslim` rows")
    mus = mus[mus["count"] > 0]

    rows, unsplit = [], collections.Counter()
    tally = collections.Counter()
    for r in mus.itertuples(index=False):
        base = dict(geo_id=r.geo_id, geo_level=r.geo_level, geo_name=r.geo_name,
                    basis=r.basis, year=r.year, source_id=r.source_id)
        code = oa2lad.get(r.geo_id)
        s = shares.get(code)
        if s is None:
            why = ("Output Area not in oa_lad.csv" if code is None
                   else f"no ONS religion x ethnic group figure for {code}")
            unsplit[why] += r.count
            rows.append(dict(base, source_category=COL, count=float(r.count), tier="measured",
                             note=NOTE_UNSPLIT.format(why=why)))
            continue
        su, sh = r.count * s[0], r.count * s[1]
        rest = r.count - su - sh
        for label, n, share in ((SUNNI, su, s[0]), (SHIA, sh, s[1])):
            if n > 0:
                rows.append(dict(base, source_category=label, count=n, tier="derived",
                                 note=NOTE_NAMED.format(lad=code, share=share)))
                tally[label] += n
        if rest > 0:
            rows.append(dict(base, source_category=COL, count=rest, tier="measured",
                             note=NOTE_REST))
        tally[COL] += rest

    out = pd.DataFrame(rows, columns=OUT_COLUMNS)
    _check(out, mus, tally, unsplit, shares)
    return out


def _check(out, mus, tally, unsplit, shares):
    per_oa = out.groupby("geo_id")["count"].sum()
    measured = mus.set_index("geo_id")["count"]
    diff = (per_oa.reindex(measured.index).fillna(0) - measured).abs()
    if diff.max() > 1e-6 or set(per_oa.index) != set(measured.index):
        raise SystemExit(f"uk_islam_split does not conserve OA Muslims (worst {diff.max():.6f})")
    total = float(measured.sum())
    print(f"  OK every one of {len(measured):,} Output Areas sums to its census `Muslim` "
          f"count, {total:,.0f} in all")
    for k in (SUNNI, SHIA, COL):
        print(f"      {k:<28} {tally[k]:>12,.0f}  {100 * tally[k] / total:5.1f}%")
    for why, n in unsplit.items():
        print(f"      left unsplit, {why}: {n:,.0f}")
    # The OA-allocated census and the ONS cross are both 2021 census; they should agree on
    # each authority's Muslims to within disclosure noise.
    lad = pd.read_csv(OA_LAD, dtype=str)
    oa_m = measured.groupby(measured.index.map(dict(zip(lad["OA21CD"], lad["LAD23CD"])))).sum()
    worst = max((abs(oa_m.get(k, 0) - v[2]) / v[2], k) for k, v in shares.items()
                if v[2] >= 1000)
    print(f"  OK allocated OA Muslims vs ONS cross per LAD (1,000+ Muslims): worst "
          f"{100 * worst[0]:.2f}% ({worst[1]})")
    if worst[0] > 0.02:
        raise SystemExit("an authority's Muslims disagree by more than 2% between the two "
                         "census products; check the LAD join")


def main():
    if "--fetch" in sys.argv:
        fetch()
    out = build()
    if "--dry-run" in sys.argv:
        print("  --dry-run: nothing written")
        return
    tmp = OUT + ".part"
    out.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, OUT)
    print(f"  wrote {OUT} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
