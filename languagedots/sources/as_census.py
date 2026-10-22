"""American Samoa, 2020 Island Areas Census, language spoken at home -> data/normalized/as.csv.

    python sources/as_census.py [--fetch]

SOURCE. U.S. Census Bureau, 2020 Island Areas Censuses, American Samoa Demographic and Housing
Characteristics Summary File (DHC), public domain (sources/ia2020.py has the URLs).

THE TABLE. PCT24 "Age by language spoken at home for the population 5 years and over in
households", 15 cells: two age bands (5-17, 18+), each split into speak only English and five
language groups (Samoan, Tongan, other Pacific Island languages, Asian languages, other). One
answer per person (the questionnaire asks whether the person speaks a language other than
English at home and, if so, which), so "English" is English only. Universe: people in households,
which leaves out group quarters and children under 5. Nothing is suppressed in the file.

THE GRAIN. Counties (summary level 060, TIGER's county subdivisions, the villages' groupings):
the 16 records are joined to religiondots' 10 placement units (religiondots/data/geo/as/
as_lookup.csv, read in place): nine counties of Tutuila, and Manu'a, which religiondots draws as
one unit and the Bureau publishes as five counties (Faleasao, Fitiuta, Ofu, Olosega, Ta'u), summed
here. Rose Island and Swains Island hold nobody. Districts (050), tracts (140) and block groups
(150, table PBG5) are read as checks.

CHECKS (asserted)
  1. every record of PCT24 adds up: each band's languages sum to the band, the bands to the total
  2. the counties sum to the districts and the territory, in every row
  3. the 18 tracts sum to the districts, in every row
  4. each unit's total population (P1) equals religiondots' census_2020 (the join, both ways)
  5. PBG5 (the same universe in three groups, published by block group) summed per tract equals
     PCT24 collapsed to those groups, in every tract whose block groups are all in the header
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ia2020  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CC = "as"
HERE = ia2020.HERE
NORM = HERE / "data" / "normalized"
RD_LOOKUP = HERE.parent / "religiondots" / "data" / "geo" / "as" / "as_lookup.csv"
# PCT24's cells in order. The same 6 rows repeat for 5-17 (cells 3-8) and 18+ (cells 10-15);
# cell 1 is the total, 2 and 9 the bands.
LANGS = ["Speak only English", "Speak Samoan", "Speak Tongan", "Speak other Pacific Island languages",
         "Speak Asian languages", "Speak other languages"]
PBG5 = {"English only": ["Speak only English"], "Samoan": ["Speak Samoan"],
        "Other languages": [l for l in LANGS if l not in ("Speak only English", "Speak Samoan")]}
TOTAL_HH_5PLUS = 44_174
TOTAL_POP = 49_710


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
    adult = {lab: 10 + i for i, lab in enumerate(LANGS)}
    bad = (p[list(young.values())].sum(axis=1) != p[2]) | (p[list(adult.values())].sum(axis=1) != p[9]) \
        | (p[2] + p[9] != p[1])
    if bad.any():                                                                      # check 1
        raise SystemExit(f"check 1: {int(bad.sum())} PCT24 records do not add up")
    for lab in LANGS:
        p[lab] = p[young[lab]] + p[adult[lab]]
    print(f"check 1 ok: PCT24 adds up in all {len(p)} records")

    terr = p[p["SUMLEV"] == "040"]
    assert len(terr) == 1
    terr = terr.iloc[0]
    assert terr[1] == TOTAL_HH_5PLUS, terr[1]
    print(f"American Samoa, people aged 5+ in households: {terr[1]:,}")
    for lab in LANGS:
        print(f"    {lab:40s} {terr[lab]:>8,}  {terr[lab] / terr[1]:7.2%}")

    # check 2 and 3
    di = p[p["SUMLEV"] == "050"].set_index("GEOID")
    co = p[p["SUMLEV"] == "060"].copy()
    tr = p[p["SUMLEV"] == "140"].copy()
    for lev, d in (("counties", co), ("tracts", tr)):
        d = d.assign(dist="0500000US" + d["GEOID"].str[9:14])
        s = d.groupby("dist")[LANGS].sum()
        ref = di.loc[s.index, LANGS]
        diff = (s - ref).abs().to_numpy().sum()
        rest = di.loc[di.index.difference(s.index), LANGS].sum().sum()
        if diff or rest:
            raise SystemExit(f"check {2 if lev == 'counties' else 3}: {lev} do not sum to the districts "
                             f"(abs diff {diff}, districts with no {lev} hold {rest})")
    if (di[LANGS].sum() - terr[LANGS]).abs().sum():
        raise SystemExit("check 2: districts do not sum to the territory")
    print("check 2 ok: the 16 counties sum exactly to the 5 districts, and those to the territory, in all 6 rows")
    print("check 3 ok: the 18 tracts sum exactly to the 5 districts in all 6 rows")

    # check 4: join to religiondots' units
    lut = pd.read_csv(RD_LOOKUP, keep_default_na=False)
    p1 = ia2020.read_table(CC, z, g, "P1")
    p1c = p1[p1["SUMLEV"] == "060"].set_index("BASENAME")[1]
    p1_all = p1.set_index("GEOID")[1]
    co["BASENAME"] = co["BASENAME"].str.replace("’", "'")
    by_base = co.set_index("BASENAME")
    rows, used = [], set()
    for _, r in lut.iterrows():
        parts = r["tiger"].split("+") if "+" in r["tiger"] else [r["tiger"].replace(" county", "")]
        for part in parts:
            assert part in by_base.index, f"no county {part!r}"
            used.add(part)
        s = by_base.loc[parts, LANGS].sum()
        tot = int(p1c.loc[parts].sum())
        if tot != int(r["census_2020"]):                                               # check 4
            raise SystemExit(f"check 4: {r['unit']} P1 {tot} != religiondots census_2020 {r['census_2020']}")
        for lab in LANGS:
            rows.append(dict(geo_id=r["unit"], geo_level="county", source_category=lab, count=int(s[lab])))
    left = set(by_base.index) - used
    nobody = sorted(l for l in left if p1c.get(l, 0) > 0)
    if nobody or left != {"Rose Island", "Swains Island"}:
        raise SystemExit(f"check 4: counties not joined: {sorted(left)}")
    print(f"check 4 ok: all {len(used)} counties with people join to religiondots' {len(lut)} units and "
          f"their P1 equals census_2020 in every unit; Rose Island and Swains Island hold nobody")

    # check 5: PBG5 by block group, summed per tract, where all of a tract's block groups are listed
    b = ia2020.read_table(CC, z, g, "PBG5")
    b = b[b["SUMLEV"] == "150"].copy()
    b["tract"] = b["GEOID"].str[9:9 + 11]
    b["P1"] = b["GEOID"].map(p1_all)
    groups = list(PBG5)
    for i, gname in enumerate(groups):
        b[gname] = b[3 + i] + b[7 + i]
    assert (b[groups].sum(axis=1) == b[1]).all(), "PBG5 does not add up"
    tr["tract"] = tr["GEOID"].str[9:]
    p1_tr = pd.Series(tr["GEOID"].map(p1_all).to_numpy(), index=tr["tract"])
    have = b.groupby("tract")["P1"].sum()
    whole = [t for t in tr["tract"] if t in have.index and have[t] == p1_tr[t]]
    bs = b[b["tract"].isin(whole)].groupby("tract")[groups].sum().reindex(whole, fill_value=0)
    t5 = pd.DataFrame({gname: tr.set_index("tract").loc[whole, rws].sum(axis=1) for gname, rws in PBG5.items()})
    diff = (bs - t5).abs()
    if diff.to_numpy().sum():
        raise SystemExit(f"check 5: PBG5 by block group != PCT24 by tract:\n{diff[diff.sum(axis=1) > 0]}")
    print(f"check 5 ok: PBG5 by block group sums to PCT24 in all three groups in {len(whole)} of {len(tr)} "
          f"tracts (the rest: " + ", ".join(sorted(set(tr['tract']) - set(whole))) + ")")

    p1t = p1[p1["SUMLEV"] == "040"].iloc[0][1]
    assert p1t == TOTAL_POP, p1t
    out = pd.DataFrame(rows)
    out = out[out["count"] > 0]
    assert out["count"].sum() == terr[1]
    out.to_csv(NORM / "as.csv", index=False)
    print(f"wrote as.csv: {len(out)} rows over {out['geo_id'].nunique()} units; {out['count'].sum():,} people "
          f"(total population {p1t:,}; {p1t - terr[1]:,} under 5 or in group quarters, {(p1t - terr[1]) / p1t:.1%})")
    for u, d in out.groupby("geo_id"):
        d = d.set_index("source_category")["count"]
        print(f"    {u:12s} {d.sum():>7,}  " + "  ".join(f"{l.replace('Speak ', '')[:12]} {v / d.sum():.0%}"
                                                        for l, v in d.items()))


if __name__ == "__main__":
    main()
