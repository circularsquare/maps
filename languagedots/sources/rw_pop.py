"""Rwanda: everyone on Kinyarwanda, per district, on the RPHC-5 2022 census district counts
-> data/normalized/rw.csv.

    python sources/rw_pop.py

NO LANGUAGE QUESTION. RPHC-5 (2022) and RPHC-4 (2012) ask literacy by language (Kinyarwanda,
English, French, Swahili), which is not a buildable question (AGENT_BRIEF §2); the Afrobarometer
has never surveyed Rwanda. Kinyarwanda is the first language of practically every Rwandan; the
other official languages are learned (AGENT_BRIEF §2, learned second languages are not home
languages). So the only number needed is population.

The district populations are religiondots' `rw_districts.gpkg` (read-only): NISR's
Population_2002_2022 sector layer dissolved to district, which religiondots proved equal to the
thirty RPHC-5 district profiles to the person (its sources/rw_geo.py).

CHECKS: 30 districts, 30 distinct unit ids, total 13,246,394 (RPHC-5 resident population).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402

OUT = HERE / "data" / "normalized" / "rw.csv"
RPHC5 = 13_246_394


def main():
    g = gpd.read_file(RD_GEO / "rw" / "rw_districts.gpkg", ignore_geometry=True)
    assert len(g) == 30 and g["unit"].nunique() == 30, len(g)
    assert int(g["pop"].sum()) == RPHC5, int(g["pop"].sum())
    df = pd.DataFrame(dict(geo_id=g["unit"], geo_level="district", geo_name=g["name"],
                           source_category="Ikinyarwanda", count=g["pop"].astype(int),
                           tier="derived", source_id="nisr_rphc5_2022_population", year=2022,
                           note="no language question; everyone drawn as Kinyarwanda"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {len(df)} districts, {df['count'].sum():,} people")


if __name__ == "__main__":
    main()
