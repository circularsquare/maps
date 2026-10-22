"""Northern Mariana Islands, 2020 Island Areas Census, language spoken at home
-> data/normalized/mp.csv.

    python sources/mp_census.py [--fetch]

SOURCE. U.S. Census Bureau, 2020 Island Areas Censuses, CNMI Demographic and Housing
Characteristics Summary File (DHC), public domain (sources/ia2020.py has the URLs).

THE TABLE. PCT24 "Age by language spoken at home for the population 5 years and over", 27 cells:
two age bands (5-17, 18+), each split into speak only English and 11 language groups (the same
twelve rows as Guam's PCT25). Unlike Guam's, the CNMI universe is everyone aged 5+, not only
people in households. One answer per person (the questionnaire asks whether the person speaks a
language other than English at home and, if so, which), so "English" is English only. Nothing is
suppressed in the CNMI file.

THE GRAIN. Census tracts (summary level 140): the 17 tracts of Saipan, two of Tinian, one of Rota
and one for the Northern Islands (7 people). Municipalities (050), districts (060) and block
groups (150, table PBG5) are read as checks.

CHECKS (asserted)
  1. every record of PCT24 adds up: each band's languages sum to the band, the bands to the total
  2. the tracts sum to each municipality, and the municipalities to the commonwealth, in every row
     (and the six districts equal their municipalities)
  3. PBG5 (the same universe in five coarser groups, published by block group) summed per tract
     equals PCT24 collapsed to those groups, in every tract whose block groups are all in the header
  4. the total equals P1 minus the people under 5 (P1 is the whole population)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ia2020  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CC = "mp"
HERE = ia2020.HERE
NORM = HERE / "data" / "normalized"
# PCT24's cells in order (dec/dhcmp groups/PCT24.json). The same 12 rows repeat for 5-17 (cells
# 3-14) and 18+ (cells 16-27); cell 1 is the total, 2 and 15 the bands.
LANGS = ["Speak only English", "Speak Chamorro", "Speak Carolinian", "Speak Palauan", "Speak Chuukese",
         "Speak Philippine languages", "Speak other Pacific Island languages", "Speak Chinese",
         "Speak Japanese", "Speak Korean", "Speak other Asian languages", "Speak other languages"]
# PBG5's five groups (5-17 cells 3-7, 18+ cells 9-13), and which PCT24 rows make each
PBG5 = {"English only": ["Speak only English"],
        "Chamorro": ["Speak Chamorro"],
        "Chinese": ["Speak Chinese"],
        "Philippine languages": ["Speak Philippine languages"],
        "Other languages": [l for l in LANGS if l not in ("Speak only English", "Speak Chamorro", "Speak Chinese",
                                                         "Speak Philippine languages")]}
TOTAL_5PLUS = 44_111
TOTAL_POP = 47_329
N_TRACTS = 26


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        ia2020.fetch(CC)
    NORM.mkdir(parents=True, exist_ok=True)

    z = ia2020.open_zip(CC)
    g = ia2020.geo(CC, z)
    p = ia2020.read_table(CC, z, g, "PCT24")
    young = {lab: 3 + i for i, lab in enumerate(LANGS)}
    adult = {lab: 16 + i for i, lab in enumerate(LANGS)}
    bad = (p[list(young.values())].sum(axis=1) != p[2]) | (p[list(adult.values())].sum(axis=1) != p[15]) \
        | (p[2] + p[15] != p[1])
    if bad.any():                                                                      # check 1
        raise SystemExit(f"check 1: {int(bad.sum())} PCT24 records do not add up")
    for lab in LANGS:
        p[lab] = p[young[lab]] + p[adult[lab]]
    print(f"check 1 ok: PCT24 adds up in all {len(p)} records")

    cw = p[p["SUMLEV"] == "040"]
    assert len(cw) == 1
    cw = cw.iloc[0]
    assert cw[1] == TOTAL_5PLUS, cw[1]
    print(f"CNMI, people aged 5+: {cw[1]:,}")
    for lab in LANGS:
        print(f"    {lab:40s} {cw[lab]:>8,}  {cw[lab] / cw[1]:7.2%}")

    # check 2
    mu = p[p["SUMLEV"] == "050"].set_index("GEOID")
    tr = p[p["SUMLEV"] == "140"].copy()
    if len(tr) != N_TRACTS:
        raise SystemExit(f"expected {N_TRACTS} tracts, got {len(tr)}")
    tr["muni"] = "0500000US" + tr["GEOID"].str[9:14]
    ds = p[p["SUMLEV"] == "060"].copy()
    ds["muni"] = "0500000US" + ds["GEOID"].str[9:14]
    for lev, d in (("tracts", tr), ("districts", ds)):
        s = d.groupby("muni")[LANGS].sum()
        diff = (s - mu.loc[s.index, LANGS]).abs().to_numpy().sum()
        missing = set(mu.index) - set(s.index)
        if diff or missing:
            raise SystemExit(f"check 2: {lev} do not sum to the municipalities (abs diff {diff}, missing {missing})")
    ms = mu[LANGS].sum()
    if (ms - cw[LANGS]).abs().sum():
        raise SystemExit("check 2: municipalities do not sum to the commonwealth")
    print("check 2 ok: 26 tracts and the 8 populated districts each sum exactly to their 4 municipalities, "
          "and those to the commonwealth, in all 12 rows")

    # check 3: PBG5 by block group, summed per tract
    b = ia2020.read_table(CC, z, g, "PBG5")
    b = b[b["SUMLEV"] == "150"].copy()
    b["tract"] = b["GEOID"].str[9:9 + 11]
    groups = list(PBG5)
    for i, gname in enumerate(groups):
        b[gname] = b[3 + i] + b[9 + i]
    assert (b[groups].sum(axis=1) == b[1]).all(), "PBG5 does not add up"
    # The geographic header leaves some block groups out altogether (Tract 8 lacks Block Group 1,
    # 1,840 against 1,579 in P1), so completeness is judged by total population (P1), not by count.
    p1 = ia2020.read_table(CC, z, g, "P1")
    p1_all = p1.set_index("GEOID")[1]
    b["P1"] = b["GEOID"].map(p1_all)
    tr["tract"] = tr["GEOID"].str[9:]
    p1_tr = tr["GEOID"].map(p1_all)
    p1_tr.index = tr["tract"]
    have = b.groupby("tract")["P1"].sum()
    whole = [t for t in tr["tract"] if t in have.index and have[t] == p1_tr[t]]
    bs = b[b["tract"].isin(whole)].groupby("tract")[groups].sum().reindex(whole, fill_value=0)
    t5 = pd.DataFrame({gname: tr.set_index("tract").loc[whole, rows].sum(axis=1) for gname, rows in PBG5.items()})
    diff = (bs - t5).abs()
    if diff.to_numpy().sum():
        raise SystemExit(f"check 3: PBG5 by block group != PCT24 by tract:\n{diff[diff.sum(axis=1) > 0]}")
    print(f"check 3 ok: PBG5 by block group sums to PCT24 in all five groups in {len(whole)} of {len(tr)} tracts "
          f"(the rest have no block groups published or one missing from the header: "
          + ", ".join(sorted(set(tr['tract']) - set(whole))) + ")")

    # check 4
    p1c = p1[p1["SUMLEV"] == "040"].iloc[0][1]
    assert p1c == TOTAL_POP, p1c
    print(f"check 4 ok: P1 {p1c:,}; aged 5+ {cw[1]:,}; under 5 or not in the universe {p1c - cw[1]:,} "
          f"({(p1c - cw[1]) / p1c:.1%})")

    p1t = p1[p1["SUMLEV"] == "140"].assign(geo_id=lambda d: d["GEOID"].str[9:]).set_index("geo_id")[1]
    tr["geo_id"] = tr["GEOID"].str[9:]
    tr = tr[tr[1] > 0]
    long = tr.melt(id_vars=["geo_id"], value_vars=LANGS, var_name="source_category", value_name="count")
    long["geo_level"] = "tract"
    long = long[long["count"] > 0]
    assert long["count"].sum() == cw[1]
    long[["geo_id", "geo_level", "source_category", "count"]].to_csv(NORM / "mp.csv", index=False)
    pd.DataFrame({"geo_id": p1t.index, "p1": p1t.to_numpy()}).to_csv(NORM / "mp_p1.csv", index=False)
    print(f"wrote mp.csv: {len(long)} rows over {tr['geo_id'].nunique()} tracts with people; "
          f"{long['count'].sum():,} people; mp_p1.csv (total population per tract, for the geography check)")
    print("  tracts, for the record:")
    for _, r in tr.sort_values(1, ascending=False).iterrows():
        print(f"    {r['NAME']:24s} {r[1]:>7,}  (P1 {p1t[r['geo_id']]:,})")


if __name__ == "__main__":
    main()
