"""Comoros: everyone on their island's Comorian language, on the 2017 census island populations
-> data/normalized/km.csv.

    python sources/km_pop.py

NO LANGUAGE QUESTION. The 2017 RGPH asks none (scout 2026-10-05). Afrobarometer R10 (2024) asks
"Quelle est la langue que vous parlez le plus chez vous actuellement ?" but only its national
summary is out (religiondots' data/raw/km/COM_R10-Resume...pdf): Shikomori 89.9, French 9.9,
Swahili 0.1, Malagasy 0.1. That checks Comorian's place; French is not drawn (sources/km.md).

ONE LANGUAGE PER ISLAND. Glottolog splits Comorian into Ngazidja (ngaz1238), Mwali (mwal1237),
Ndzwani (ndzw1235) and Maore (maor1244, Mayotte, which is France's). Each is the language of its
island (Glottolog's AES notes: "statutory language of provincial identity" on Grande Comore,
Mohéli and Anjouan, Constitution 2002 art. 1), so each island's people go on its own one.

POPULATION. The 2017 RGPH island counts, religiondots' km_units.gpkg (read-only): Ngazidja
379,367, Ndzuwani 327,382, Mwali 51,567, 758,316 in all.
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

OUT = HERE / "data" / "normalized" / "km.csv"
RGPH2017 = 758_316
LANG = {"Ngazidja": "Shingazidja", "Ndzuwani": "Shindzuani", "Mwali": "Shimwali"}


def main():
    g = gpd.read_file(RD_GEO / "km" / "km_units.gpkg", ignore_geometry=True)
    assert sorted(g["unit"]) == sorted(LANG), list(g["unit"])
    assert int(g["pop"].sum()) == RGPH2017, int(g["pop"].sum())
    df = pd.DataFrame(dict(geo_id=g["unit"], geo_level="island", geo_name=g["unit"],
                           source_category=g["unit"].map(LANG), count=g["pop"].astype(int),
                           tier="derived", source_id="inseed_rgph_2017_population", year=2017,
                           note="no language question; everyone drawn on the island's Comorian"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {len(df)} islands, {df['count'].sum():,} people")
    print(df[["geo_name", "source_category", "count"]].to_string(index=False))


if __name__ == "__main__":
    main()
