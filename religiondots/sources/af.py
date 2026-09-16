"""Afghanistan: nobody is asked their religion. Every settled person NSIA estimates is drawn on Islam.

Reads data/geo/af/af_lookup.csv (written by `sources/af_geo.py`) and data/raw/estimates/pew.zip;
writes data/normalized/af.csv. `sources/af.md` is the record in prose; `ask/RULINGS.md` 2026-09-15
(priority) and 2026-09-16 (Mauritania, drawn on a compiler's national figure) the rulings it rests on.

## NOBODY IS ASKED

No census since 1979 (NSIA's own introduction to its 1404 estimates); the Asia Foundation's Survey of
the Afghan People has no religion or sect item among its demographics (2019 report, `sources.md`
§11ao); no Afghanistan row in UNSD's Demographic Yearbook table 28. Pew Research Center's 2011
survey asked Muslims their sect (Sunni 90%, Shia 7%, just a Muslim 3%) and reports it nationally.

## EVERYONE ON ISLAM, NOT ON PEW'S RESIDUAL

Pew Research Center's 2020 estimate for Afghanistan is 99.862% Muslim, with 7,571 Christians, 7,814
Buddhists, 3,304 unaffiliated, 50 Hindus, 10 Jews and 35,179 "other religions" on a population of
39,068,979. Nothing says who or where those people are, and the US State Department's 2023 report
says six Sikhs and Hindus remain and no reliable estimate exists for Christians or Baha'is. So the
whole settled estimate is drawn on Islam (Mauritania's construction, `sources/mr.md` §4), and
`pew_row` prints Pew's figures beside it for the record. Nothing is subtracted for the six.

## NOT DRAWN

The 1,500,000 nomadic Kuchis, whom NSIA holds at a fixed national figure with no province; they are
the country's `gap`.

Usage:
    python sources/af.py            rebuild data/normalized/af.csv
"""

import io
import os
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOOKUP = os.path.join(ROOT, "data", "geo", "af", "af_lookup.csv")
PEW = os.path.join(ROOT, "data", "raw", "estimates", "pew.zip")
OUT = os.path.join(ROOT, "data", "normalized", "af.csv")

SETTLED = 34_935_197
KUCHI = 1_500_000
# Pew 2020, unrounded counts, Afghanistan; asserted so the figures quoted in sources/af.md stay true
PEW_2020 = dict(Population=39_068_979, Muslims=39_015_051, Christians=7_571, Buddhists=7_814,
                Religiously_unaffiliated=3_304, Hindus=50, Jews=10, Other_religions=35_179)


def pew_row():
    with zipfile.ZipFile(PEW) as z:
        name = [n for n in z.namelist() if n.endswith("(unrounded counts).csv")][0]
        t = pd.read_csv(io.BytesIO(z.read(name)), thousands=",")
    r = t[(t["Country"] == "Afghanistan") & (t["Year"] == 2020)].iloc[0]
    got = {k: int(r[k]) for k in PEW_2020}
    if got != PEW_2020:
        raise SystemExit(f"Pew 2020's Afghanistan row is {got}, pinned {PEW_2020}")
    print(f"  Pew 2020 (for the record, not drawn): {got['Muslims'] / got['Population']:.3%} Muslim "
          f"of {got['Population']:,}; " + ", ".join(f"{k} {v:,}" for k, v in got.items()
                                                   if k not in ("Population", "Muslims")))


def main():
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    if len(lut) != 34 or int(lut["pop"].sum()) != SETTLED:
        raise SystemExit(f"{LOOKUP}: {len(lut)} provinces, {int(lut['pop'].sum()):,} people; "
                         "re-run sources/af_geo.py")
    pew_row()
    out = pd.DataFrame({"geo_id": lut["geo_id"], "geo_level": "province", "geo_name": lut["name"],
                        "source_category": "Muslim", "count": lut["pop"].astype(int),
                        "basis": "estimate", "year": 2025, "source_id": "af_nsia1404_settled",
                        "note": "no source asks religion; every settled person in NSIA's 1404 "
                                "estimate is drawn on Islam (sources/af.py)"})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: 34 provinces, {int(out['count'].sum()):,} settled people on Muslim; "
          f"{KUCHI:,} Kuchi not drawn ({KUCHI / (SETTLED + KUCHI):.4%} of NSIA's total)")


if __name__ == "__main__":
    main()
