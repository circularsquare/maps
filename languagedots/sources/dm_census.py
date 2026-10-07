"""Dominica: first language from the 2011 census preliminary report's settlement counts
-> data/normalized/dm.csv and the placement layer data/geo/dm/dm_hexes.gpkg.

    python sources/dm_census.py

NO CENSUS LANGUAGE QUESTION (2011; none in 2001 either). Built as Haiti (sources/ht.md) with two
groups the census's own figures let us place:

- Wesley and Marigot/Concord (Table 8, non-institutional population by town/village) speak
  Kokoy, the English-lexicon creole brought by Antiguan and Montserratian labourers after
  emancipation; Glottolog files it under Antiguan and Barbudan Creole (anti1245), the node from
  tree.d/bb.txt.
- The Haitian-born (1,054, the Review's "Foreign-born population" paragraph) on Haitian Creole.
- Everyone else on Kweyol (Antillean Creole, the node fr built).

THE TABLE: Commonwealth of Dominica, 2011 Population and Housing Census, Preliminary Results
(Central Statistical Office, September 2011),
stats.gov.dm/wp-content/uploads/2019/06/Population_and_Housing_Census_2011.pdf
-> data/raw/dm/dm_census_2011.pdf. Total population 71,293 (Table 3, including institutions).

PLACEMENT: religiondots' dm_hexes.gpkg is one unit, DM (read-only). This script re-keys its hexes:
a hex whose centroid is within RADIUS_KM of Wesley or Marigot goes to that village's unit, the
rest to DM-REST. Only placement inside the country moves (AGENT_BRIEF §4.4); the counts stay the
census's. Check: Kontur's population in each village circle is within a factor 2 of the census
figure.
"""
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import pandas as pd  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402

RAW = HERE / "data" / "raw" / "dm" / "dm_census_2011.pdf"
OUT = HERE / "data" / "normalized" / "dm.csv"
GEO = HERE / "data" / "geo" / "dm" / "dm_hexes.gpkg"
TOTAL = 71_293
HAITI = 1_054
# Table 8 rows (non-institutional, 2011). Village points: Wikipedia's coordinates are rounded to
# the minute (Marigot's lands 1.8 km inland), so each is moved to the centre of Kontur's built-up
# cluster within 2 km of it.
VILLAGES = {"DM-WESLEY": ("Wesley", 1_362, (-61.311, 15.564)),
            "DM-MARIGOT": ("Marigot/Concord", 2_411, (-61.284, 15.536))}
RADIUS_KM = 1.5


def text():
    import fitz
    return "\n".join(p.get_text() for p in fitz.open(RAW))


def main():
    t = text()
    # the figures used, as printed
    assert "numbered 71,293" in t, "total"
    assert "Haitian-born population numbered 1,054" in re.sub(r"\s+", " ", t), "Haitian-born"
    for _, (name, n, _) in VILLAGES.items():
        assert re.search(re.escape(name) + r"\s*\n[\d,]+\s*\n[\d,]+\s*\n" + f"{n:,}", t), name

    import geopandas as gpd
    g = gpd.read_file(RD_GEO / "dm" / "dm_hexes.gpkg")
    assert set(g["unit"]) == {"DM"}, set(g["unit"])
    m = g.to_crs(32620)
    cen = m.geometry.centroid
    g["unit"] = "DM-REST"
    pts = gpd.GeoSeries.from_xy([v[2][0] for v in VILLAGES.values()],
                                [v[2][1] for v in VILLAGES.values()], crs=4326).to_crs(32620)
    for (u, (name, n, _)), p in zip(VILLAGES.items(), pts):
        near = (cen.distance(p) <= RADIUS_KM * 1000) & (g["unit"] == "DM-REST")
        g.loc[near, "unit"] = u
        k = g.loc[near, "pop"].sum()
        print(f"  {name}: {near.sum()} hexes, Kontur {k:,.0f} vs census {n:,}")
        assert 0.5 <= k / n <= 2.0, (name, k, n)
    GEO.parent.mkdir(parents=True, exist_ok=True)
    g.to_file(GEO, driver="GPKG")
    print(f"  wrote {GEO.relative_to(HERE)}: {len(g)} hexes, units {g['unit'].value_counts().to_dict()}")

    rows = []
    rest = TOTAL - sum(v[1] for v in VILLAGES.values())
    for u, (name, n, _) in VILLAGES.items():
        rows.append(dict(geo_id=u, geo_name=name, source_category="Kokoy", count=n))
    rows.append(dict(geo_id="DM-REST", geo_name="rest of Dominica",
                     source_category="Haitian-born", count=HAITI))
    rows.append(dict(geo_id="DM-REST", geo_name="rest of Dominica",
                     source_category="Everyone else", count=rest - HAITI))
    df = pd.DataFrame(rows)
    df["geo_level"] = "settlement"
    df["tier"] = "derived"
    df["year"] = 2011
    assert df["count"].sum() == TOTAL
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {len(df)} rows, {df['count'].sum():,} people")


if __name__ == "__main__":
    main()
