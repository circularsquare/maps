"""Syria: Pew Research Center's 2020 national religious composition, one mix in every governorate, on
the Central Bureau of Statistics' end-2011 estimate of the people living in each.

Reads data/raw/estimates/pew.zip and data/geo/sy/sy_lookup.csv (`sources/sy_geo.py`); writes
data/normalized/sy.csv. `sources/sy.md` is the record in prose; `ask/RULINGS.md` 2026-09-15 and
2026-09-16 (a §14 case files an ask and carries on; a country with no source that asks is drawn on a
compiler's figure) and ask 050 are what it builds on.

## NOTHING ASKS

The 2004 census form has no religion item (`sources.md` §scout-2026-09-15-negatives), no census
since 1960 has asked, and the Arab Barometer's first Syrian wave (October-November 2025, 1,229
interviews) is unreleased on 2026-10-03. So the level is a compiler's.

## WHOSE FIGURE

Pew's 2020 Syria row (`Religious Composition 2010-2020`, unrounded counts) is not a Pew survey: its
Appendix A names the World Religion Database for both 2010 and 2020, with UN World Population
Prospects 2024 for the population. It is still the figure the national estimate layer draws (spec
§15), so drawing it here keeps the two from disagreeing. 21,049,429 people: Muslims 19,821,403,
Christians 808,455, unaffiliated 416,788, Hindus 2,105, other religions 559, Jews 118.

  * **The Druze are inside the Muslims.** Pew's `Other_religions` is 559 people, 0.003%, where its
    Lebanon row (from Pew's own surveys) carries 4.3% for the Druze. So the World Religion Database
    files Syria's Druze under Islam, and so does this map; it says so in the note. No Druze share is
    split out at the national rate: that would draw Druze in every governorate at the same 3%, and
    putting them where they live (Suwayda) is the placement held back for Anita (ask 050).
  * **Alawites, Ismailis and Shia are inside the Muslims too**, on bare `islam`, as Oman and Saudi
    Arabia were ruled (2026-09-15): no sect split without placement.
  * **The 2,782 Hindus, Jews and others are not drawn** (`TAIL`): nothing places them, at 1 dot per
    1,000 they would be three dots dropped wherever the Hilbert carry lands, and a Jewish dot outside
    Damascus would be plainly false. They are the `gap`.

## WHAT IS DRAWN

Each governorate's CBS end-2011 people, less the tail's share, at Pew's three-way mix. Every row is
`modelled`. Palestinians are not split out: Pew's row is for everyone living in Syria and its
Palestinian row is within 5 points on the Muslim share, so a separate layer would move a few
thousand dots.

Usage:
    python sources/sy.py            rebuild data/normalized/sy.csv
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
LOOKUP = os.path.join(ROOT, "data", "geo", "sy", "sy_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "sy.csv")

DRAWN = {"Muslims": "Muslim", "Christians": "Christian", "Religiously_unaffiliated": "No religion"}
TAIL = ["Buddhists", "Hindus", "Jews", "Other_religions"]
CATS = list(DRAWN.values())
YEAR = 2020
SOURCE_ID = "sy_pew2020_wrd_on_cbs2011"

# Pew 2020, read 2026-10-03 and asserted, so a new Pew release cannot slip in unseen.
PEW_2020 = {"Population": 21_049_429, "Muslims": 19_821_403, "Christians": 808_455,
            "Religiously_unaffiliated": 416_788, "Buddhists": 0, "Hindus": 2_105, "Jews": 118,
            "Other_religions": 559}
CBS_2011 = 21_377_000

# note_public's figures, measured 2026-10-03 and asserted.
NOTE = dict(drawn=21_374_175, muslim=20_129_864, christian=821_038, no_religion=423_273,
            tail=2_825)


def pew_row(year):
    with zipfile.ZipFile(PEW) as z:
        name = [n for n in z.namelist() if n.endswith("(unrounded counts).csv")][0]
        t = pd.read_csv(io.BytesIO(z.read(name)), thousands=",")
    r = t[(t["Country"] == "Syria") & (t["Year"] == year)]
    if len(r) != 1:
        raise SystemExit(f"Pew has {len(r)} Syria rows for {year}")
    return {k: int(r[k].iloc[0]) for k in PEW_2020}


def main():
    pew = pew_row(YEAR)
    if pew != PEW_2020:
        raise SystemExit(f"Pew's 2020 Syria row changed: {pew}")
    p10 = pew_row(2010)
    parts = sum(pew[k] for k in DRAWN) + sum(pew[k] for k in TAIL)
    if abs(parts - pew["Population"]) > 2:           # Pew's unrounded counts are off by one
        raise SystemExit(f"Pew's Syria families sum to {parts:,}, not {pew['Population']:,}")
    print("Pew 2020, Syria (World Religion Database via Pew): " + ", ".join(
        f"{k} {pew[k]:,} ({100 * pew[k] / pew['Population']:.3f}%)" for k in list(DRAWN) + TAIL))
    print("Pew 2010, for the note: " + ", ".join(
        f"{k} {100 * p10[k] / p10['Population']:.2f}%" for k in DRAWN))
    print(f"  Other_religions is {100 * pew['Other_religions'] / pew['Population']:.4f}%: "
          "the Druze are inside the Muslims (sources/sy.md §3)")

    tail_share = sum(pew[k] for k in TAIL) / pew["Population"]
    mix = {DRAWN[k]: pew[k] / sum(pew[j] for j in DRAWN) for k in DRAWN}

    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    if len(lut) != 14 or int(lut["pop"].sum()) != CBS_2011:
        raise SystemExit(f"sy_lookup.csv is not 14 governorates summing to {CBS_2011:,}")
    pop = lut.set_index("geo_id")["pop"].astype(float)
    drawn = (pop * (1.0 - tail_share)).round().astype(int)
    m = pd.DataFrame({c: drawn * s for c, s in mix.items()})[CATS]
    counts = round_within_rows(m)
    if not (counts.sum(axis=1) == drawn.reindex(counts.index)).all():
        raise SystemExit("a governorate's rounded counts do not sum to its drawn people")

    out = counts.stack().rename("count").reset_index()
    out.columns = ["geo_id", "source_category", "count"]
    out["geo_level"] = "governorate"
    out["geo_name"] = out["geo_id"].map(dict(zip(lut["geo_id"], lut["name"])))
    out["basis"] = "estimate"
    out["year"] = YEAR
    out["source_id"] = SOURCE_ID
    out["note"] = ("Pew Research Center 2020 (World Religion Database) national mix, applied to the "
                   "CBS end-2011 estimate of people living in the governorate; nothing places anyone")
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count", "basis", "year",
            "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")

    tot = int(out["count"].sum())
    by = out.groupby("source_category")["count"].sum()
    print(f"\nwrote {OUT} ({len(out)} rows, {tot:,} people, 14 governorates)")
    for c in CATS:
        print(f"    {by[c] / tot:9.4%}  {c}  ({int(by[c]):,})")
    got = dict(drawn=tot, muslim=int(by["Muslim"]), christian=int(by["Christian"]),
               no_religion=int(by["No religion"]), tail=CBS_2011 - tot)
    print(f"  not drawn (Pew's Hindus, Jews and others at {tail_share:.4%}): {CBS_2011 - tot:,}")
    print(f"\n  note_public's figures: {got}")
    if NOTE and got != NOTE:
        raise SystemExit(f"note_public's figures are {NOTE}; the build gives {got}")


if __name__ == "__main__":
    main()
