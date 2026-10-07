"""North Korea: Pew Research Center's 2020 national religious composition (the World Religion
Database's figure), one mix in every province, on the 2008 census's people in each.

Reads data/raw/estimates/pew.zip and data/geo/kp/kp_lookup.csv (`sources/kp_geo.py`); writes
data/normalized/kp.csv. `sources/kp.md` is the record in prose; `ask/RULINGS.md` 2026-09-15 and
2026-09-16 (a §14 case files an ask and carries on; a country with no source that asks is drawn on a
compiler's figure) are what it builds on.

## NOTHING ASKS

The 1993 and 2008 census forms have no religion item, nor does MICS 2009 (`sources.md` §11, North
Korea). No survey of residents has ever been allowed to ask. So the level is a compiler's.

## WHOSE FIGURE

Pew's 2020 North Korea row (`Religious Composition 2010-2020`, unrounded counts) is not a Pew survey:
Appendix A names the World Religion Database for 2010 and 2020 (UN WPP 2024 population), and its
first page names North Korea as one of about two dozen places where the World Religion Database is
the only source. 26,136,312 people: unaffiliated 19,044,889, other religions 6,591,569, Buddhists
396,455, Christians 100,372, Muslims 2,614, Hindus 414.

  * **"Other religions" is the World Religion Database's new religionists and ethnic religionists.**
    ARDA's free view of the WRD (2025) gives new religionists 12.88% (Cheondogyo) and ethnic
    religionists 12.28% (Korean shamanism), with Chinese folk religionists 0.06%; together 25.22%,
    Pew's figure. Pew's count is split in those proportions (`WRD_OTHER`) onto Cheondogyo, Korean
    folk religion and Chinese folk religion, on Anita's ruling on ask 052 (2026-10-03). From the
    first build until that ruling, the same day, it was one residual node, `other.kp`.
  * **The 3,028 Muslims and Hindus are not drawn** (`TAIL`): nothing places them, and at 1 dot per
    1,000 they would be three dots dropped wherever the Hilbert carry lands.

## WHAT IS DRAWN

Each province's 2008 census people (Table 2, re-cut to COD-AB's 11 units by `kp_geo.py`), less the
tail's share, at Pew's four-way mix. Every row is `modelled`. The 702,372 people in Table 1 and not
in Table 2 (the national tables' note includes military camps) belong to no province and are in the
`gap`, as is the tail.

Usage:
    python sources/kp.py            rebuild data/normalized/kp.csv
"""

import io
import os
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "6")

import pandas as pd

from afrobarometer import round_within_rows

PEW = os.path.join(ROOT, "data", "raw", "estimates", "pew.zip")
LOOKUP = os.path.join(ROOT, "data", "geo", "kp", "kp_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "kp.csv")

DRAWN = {"Religiously_unaffiliated": "No religion", "Other_religions": "Other religions",
         "Buddhists": "Buddhist", "Christians": "Christian"}
TAIL = ["Muslims", "Hindus", "Jews"]
# The WRD's parts of Pew's `Other_religions`, percent of the population (ARDA's view of the WRD,
# 2025, read 2026-10-03). Pew's count is split in these proportions (Anita, ask 052).
WRD_OTHER = {"New religionists": 12.88, "Ethnic religionists": 12.28,
             "Chinese folk religionists": 0.06}
CATS = ["No religion"] + list(WRD_OTHER) + ["Buddhist", "Christian"]
YEAR = 2020
SOURCE_ID = "kp_pew2020_wrd_on_census2008"

# Pew 2020, read 2026-10-03 and asserted, so a new Pew release cannot slip in unseen.
PEW_2020 = {"Population": 26_136_312, "Christians": 100_372, "Muslims": 2_614,
            "Religiously_unaffiliated": 19_044_889, "Buddhists": 396_455, "Hindus": 414,
            "Jews": 0, "Other_religions": 6_591_569}
CENSUS_PROVINCES = 23_349_859
CENSUS_NATIONAL = 24_052_231

# note_public's figures, measured 2026-10-03 and asserted.
NOTE = dict(drawn=23_347_154, no_religion=17_014_470, cheondogyo=3_007_458, folk=2_867_358,
            chinesefolk=14_010, buddhist=354_188, christian=89_670, tail=2_705)


def pew_row(year):
    with zipfile.ZipFile(PEW) as z:
        name = [n for n in z.namelist() if n.endswith("(unrounded counts).csv")][0]
        t = pd.read_csv(io.BytesIO(z.read(name)), thousands=",")
    r = t[(t["Country"] == "North Korea") & (t["Year"] == year)]
    if len(r) != 1:
        raise SystemExit(f"Pew has {len(r)} North Korea rows for {year}")
    return {k: int(r[k].iloc[0]) for k in PEW_2020}


def main():
    pew = pew_row(YEAR)
    if pew != PEW_2020:
        raise SystemExit(f"Pew's 2020 North Korea row changed: {pew}")
    parts = sum(pew[k] for k in DRAWN) + sum(pew[k] for k in TAIL)
    if abs(parts - pew["Population"]) > 2:           # Pew's unrounded counts are off by one
        raise SystemExit(f"Pew's North Korea families sum to {parts:,}, not {pew['Population']:,}")
    print("Pew 2020, North Korea (World Religion Database via Pew): " + ", ".join(
        f"{k} {pew[k]:,} ({100 * pew[k] / pew['Population']:.3f}%)" for k in list(DRAWN) + TAIL))

    tail_share = sum(pew[k] for k in TAIL) / pew["Population"]
    mix = {DRAWN[k]: pew[k] / sum(pew[j] for j in DRAWN) for k in DRAWN}
    if abs(sum(WRD_OTHER.values()) - 100 * pew["Other_religions"] / pew["Population"]) > 0.01:
        raise SystemExit(f"the WRD's parts sum to {sum(WRD_OTHER.values())}, not Pew's other")
    other = mix.pop("Other religions")
    for k, v in WRD_OTHER.items():
        mix[k] = other * v / sum(WRD_OTHER.values())

    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    if len(lut) != 11 or int(lut["pop"].sum()) != CENSUS_PROVINCES:
        raise SystemExit(f"kp_lookup.csv is not 11 units summing to {CENSUS_PROVINCES:,}")
    pop = lut.set_index("geo_id")["pop"].astype(float)
    drawn = (pop * (1.0 - tail_share)).round().astype(int)
    m = pd.DataFrame({c: drawn * s for c, s in mix.items()})[CATS]
    counts = round_within_rows(m)
    if not (counts.sum(axis=1) == drawn.reindex(counts.index)).all():
        raise SystemExit("a province's rounded counts do not sum to its drawn people")

    out = counts.stack().rename("count").reset_index()
    out.columns = ["geo_id", "source_category", "count"]
    out["geo_level"] = "province"
    out["geo_name"] = out["geo_id"].map(dict(zip(lut["geo_id"], lut["name"])))
    out["basis"] = "estimate"
    out["year"] = YEAR
    out["source_id"] = SOURCE_ID
    out["note"] = ("Pew Research Center 2020 (World Religion Database) national mix, applied to the "
                   "2008 census population of the province; nothing places anyone")
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count", "basis", "year",
            "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")

    tot = int(out["count"].sum())
    by = out.groupby("source_category")["count"].sum()
    print(f"\nwrote {OUT} ({len(out)} rows, {tot:,} people, 11 provinces)")
    for c in CATS:
        print(f"    {by[c] / tot:9.4%}  {c}  ({int(by[c]):,})")
    got = dict(drawn=tot, no_religion=int(by["No religion"]),
               cheondogyo=int(by["New religionists"]), folk=int(by["Ethnic religionists"]),
               chinesefolk=int(by["Chinese folk religionists"]),
               buddhist=int(by["Buddhist"]), christian=int(by["Christian"]),
               tail=CENSUS_PROVINCES - tot)
    gap = CENSUS_NATIONAL - tot
    print(f"  not drawn: Pew's Muslims and Hindus at {tail_share:.4%} ({CENSUS_PROVINCES - tot:,}) "
          f"and the {CENSUS_NATIONAL - CENSUS_PROVINCES:,} in no province; gap {gap:,} = "
          f"{gap / CENSUS_NATIONAL:.6f} of {CENSUS_NATIONAL:,}")
    print(f"\n  note_public's figures: {got}")
    if NOTE and got != NOTE:
        raise SystemExit(f"note_public's figures are {NOTE}; the build gives {got}")


if __name__ == "__main__":
    main()
