"""
Egypt's two frontier governorates the Arab Barometer never sampled, from the census instead.

    python sources/eg_census.py      -> data/normalized/eg_census.csv

New Valley and South Sinai have no respondents in any Arab Barometer wave (sources/eg.md), nor in
Afrobarometer rounds 5 and 6, nor in the Global Flourishing Study (codes 423-427 exist in its
codebook and hold nobody). Anita's line, 2026-09-08 (ask/RULINGS.md, `ec`): *"the line is whether
anything measured the place"*. Both of these were measured, by Egypt's own census, and the
figures reached print through researchers rather than through CAPMAS:

  SOUTH SINAI, 1996 census. Arab-West Report's interns copied the 1996 religion table by
    governorate, city and countryside in the CAPMAS library in 2007 and 2011 and published it
    beside their paper 52 (data/raw/eg/awr_paper52_tablecensus1996capmas.pdf, Wayback
    20231227122900 of dialogueacrossborders.com/.../AWRpapers/paper52_tablecensus1996capmas.pdf).
    South Sinai total: 2,739 Christians, 51,723 Muslims, 33 others, 54,495 people. Its Cairo row
    is the 8.57% that sources/eg.md already checks against. New Valley and North Sinai are blank
    in it ("no information available during research in CAPMAS library").

  NEW VALLEY, 1976 census. E. J. Chitam, *The Coptic Community in Egypt: Spatial and Social
    Change* (University of Durham, Centre for Middle Eastern and Islamic Studies, Occasional
    Paper 32, 1986), Table 4.1 "Christian and Muslim urban proportions by governorate", source
    "Census for all Egypt, 1976" (data/raw/eg/chitam1986_table4-1_census1976.png, from
    dro.dur.ac.uk/132/1/32CMEIS.pdf, Wayback 20230521143125). New Valley: Christians 1.8% of the
    total population. Only the share is printed, to one decimal. The same table gives Cairo
    10.1% and all Egypt 6.3%, the two 1976 figures sources/eg.md and Denis (2000) quote.

  NORTH SINAI stays empty. No census figure for it has been found: blank in the 1996 table, and
    the 1976 table's "Sinai" (1.0%, wholly urban) is the strip Egypt held in 1976, about ten
    thousand people, not today's governorate. The 1986 census volumes (which did publish
    religion) are the place a figure would be; not found online.

The census share is applied to CAPMAS's 2026-01-01 population of the governorate, the same base
as the rest of Egypt (data/geo/eg/eg_lookup.csv). Every row is `modelled`: a share measured
thirty and fifty years ago, on a person-by-person census count, laid on today's people.
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOOKUP = os.path.join(ROOT, "data", "geo", "eg", "eg_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "eg_census.csv")

# South Sinai, 1996: the table's own counts, columns Others / Christians / Muslims / Total.
SS_1996 = dict(others=33, christian=2_739, muslim=51_723, total=54_495,
               christian_pct_printed=5.03)
# New Valley, 1976: the printed share only.
NV_1976_CHRISTIAN_PCT = 1.8

ROWS = {
    # geo_id: (name, christian share, year, source_id, note)
    "EG35": ("South Sinai",
             SS_1996["christian"] / (SS_1996["christian"] + SS_1996["muslim"]),
             "1996", "eg_census_1996_awr",
             "1996 census, governorate total as copied in the CAPMAS library by Arab-West "
             "Report (paper 52 table): 2,739 Christians, 51,723 Muslims, 33 others of 54,495; "
             "the 33 others are left out and the Christian share of the rest applied to "
             "CAPMAS's 2026-01-01 governorate population estimate"),
    "EG32": ("New Valley", NV_1976_CHRISTIAN_PCT / 100, "1976", "eg_census_1976_chitam",
             "1976 census, Christians 1.8% of the governorate as printed in Chitam (1986) "
             "Table 4.1; applied to CAPMAS's 2026-01-01 governorate population estimate"),
}


def main():
    t = SS_1996
    if t["others"] + t["christian"] + t["muslim"] != t["total"]:
        raise SystemExit("South Sinai 1996: the three religion columns do not add to the total; "
                         "the transcription is wrong")
    pct = t["christian"] / t["total"] * 100
    if abs(pct - t["christian_pct_printed"]) > 0.01:
        raise SystemExit(f"South Sinai 1996: {pct:.3f}% recomputed against "
                         f"{t['christian_pct_printed']}% printed")

    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    name = dict(zip(lut["geo_id"], lut["name"]))
    pop = dict(zip(lut["geo_id"], lut["pop"].astype(int)))
    rows = []
    for gid, (nm, share, year, sid, note) in ROWS.items():
        if name.get(gid) != nm:
            raise SystemExit(f"{gid} is {name.get(gid)!r} in eg_lookup.csv, expected {nm!r}")
        p = pop[gid]
        ch = int(round(p * share))
        for cat, n in (("Christian", ch), ("Muslim", p - ch)):
            rows.append(dict(geo_id=gid, geo_level="governorate", geo_name=nm,
                             source_category=cat, count=n, basis="self_id", year=year,
                             source_id=sid, note=note))
        print(f"  {nm:<12} {year}  Christian {share * 100:5.2f}%  of {p:,}  -> {ch:,}")
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT} ({len(out)} rows, {int(out['count'].sum()):,} people)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
