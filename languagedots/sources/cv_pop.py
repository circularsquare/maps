"""Cape Verde: everyone on Kabuverdianu, per concelho, on the 2021 census (RGPH-2021, INE)
concelho populations -> data/normalized/cv.csv.

    python sources/cv_pop.py

NO LANGUAGE QUESTION in the 2021 census tables: INE's 22 concelho workbooks (religiondots'
raw copies 119-148.xlsx, read-only) carry no language or nationality table. Kabuverdianu is the
home language of practically everyone: the Afrobarometer's home-language question, rounds 4-9
(2008-2022, 7,271 respondents, sources/mono_afro.py "Cape Verde" "Cabo Verde"), gives Crioulo /
Creole 99.5-99.9% in every round and Portuguese 0-0.45%. Portuguese is the school and official
language, learned (AGENT_BRIEF §2), and is not drawn. So the only number needed is
population, as Haiti (sources/ht_pop.py).

POPULATION. religiondots' cv_lookup.csv (read-only): RGPH-2021 resident population per
concelho, 491,233 on 22 concelhos.

CHECKS: 22 concelhos, summing to 491,233; geo_id == unit.
"""
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402

OUT = HERE / "data" / "normalized" / "cv.csv"
RGPH2021 = 491_233


def main():
    lut = pd.read_csv(RD_GEO / "cv" / "cv_lookup.csv", dtype={"geo_id": str, "unit": str})
    assert len(lut) == 22 and int(lut["pop"].sum()) == RGPH2021, (len(lut), lut["pop"].sum())
    assert (lut["geo_id"] == lut["unit"]).all()
    df = pd.DataFrame(dict(geo_id=lut["unit"], geo_level="concelho", geo_name=lut["name"],
                           source_category="Kabuverdianu", count=lut["pop"].astype(int),
                           tier="derived", source_id="cv_rgph2021", year=2021,
                           note="no language question; everyone drawn as Kabuverdianu"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {len(df)} concelhos, {df['count'].sum():,} people")


if __name__ == "__main__":
    main()
