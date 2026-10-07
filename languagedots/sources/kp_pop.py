"""North Korea: everyone on Korean, per province, on the 2008 census province populations
-> data/normalized/kp.csv.

    python sources/kp_pop.py

NO LANGUAGE QUESTION. The 1993 and 2008 censuses (Central Bureau of Statistics with UNFPA) ask
no language item, and no survey of people living in the country asks one. Korean is the first
language of practically the whole population; the only minority with any size, the ethnic
Chinese (hwagyo, a few thousand), has no published count. So the only number needed is
population, as Haiti (sources/ht_pop.py).

POPULATION. The 2008 census national report, Table 2, as religiondots re-cut it onto COD-AB's 11
provinces (kp_lookup.csv, read-only): 23,349,859 people. The census's 702,372 people in no
province (the institutional population, mostly military barracks) are not drawn and go in `gap`.
The 2008 census is the latest count by province; COD-PS (HDX cod-ps-prk) carries the same 2008
figures.

CHECKS: 11 provinces, summing to 23,349,859.
"""
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402

OUT = HERE / "data" / "normalized" / "kp.csv"
CENSUS2008 = 23_349_859


def main():
    lut = pd.read_csv(RD_GEO / "kp" / "kp_lookup.csv", dtype={"geo_id": str, "unit": str})
    assert len(lut) == 11 and int(lut["pop"].sum()) == CENSUS2008, (len(lut), lut["pop"].sum())
    assert (lut["geo_id"] == lut["unit"]).all()
    df = pd.DataFrame(dict(geo_id=lut["geo_id"], geo_level="province", geo_name=lut["name"],
                           source_category="Korean", count=lut["pop"].astype(int),
                           tier="derived", source_id="dprk_census_2008", year=2008,
                           note="no language question; everyone drawn as Korean"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {len(df)} provinces, {df['count'].sum():,} people")


if __name__ == "__main__":
    main()
