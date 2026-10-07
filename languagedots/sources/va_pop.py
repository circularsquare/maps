"""Vatican City: no language statistics. The 882 inhabitants of 31 December 2024
(vaticanstate.va, "Population"), national -> data/normalized/va.csv; placement layer
data/geo/va/va_hexes.gpkg (Kontur VA, one unit).

    python sources/va_pop.py

The page: 673 citizens, 458 of them living inside the walls (120 of these the Pontifical Swiss
Guard); 882 inhabitants in all, citizens and not. No nationality or language breakdown.
  * the 120 Swiss Guards (all Swiss citizens by the Guard's own rules) on Switzerland's home mix
    through origin_mix (dest "va");
  * the other 762 on Italian: the state's working language and the language of its lay staff;
    the clergy among them come from many countries and no count of their nationalities exists.
Every row `derived`.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sources"))
sys.path.insert(0, str(ROOT / "taxonomy"))
OUT = ROOT / "data" / "normalized" / "va.csv"
HEX = ROOT / "data" / "geo" / "va" / "va_hexes.gpkg"
TOTAL, GUARDS = 882, 120
ITALIAN = "indoeuropean.romance.italian"


def main():
    from origin_mix import mix
    acc = {ITALIAN: float(TOTAL - GUARDS)}
    for node, s in mix("CH", "va").items():
        acc[node] = acc.get(node, 0) + GUARDS * s
    s = pd.Series(acc)
    fl = s.apply(int)
    fl[(s - fl).sort_values(ascending=False).index[:TOTAL - int(fl.sum())]] += 1
    df = pd.DataFrame([dict(geo_id="VA", geo_level="country", geo_name="Vatican City",
                            source_category=k, count=int(v), tier="derived", year=2024,
                            source_id="vaticanstate_population_2024") for k, v in fl.items() if v > 0])
    assert df["count"].sum() == TOTAL
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(df[["source_category", "count"]].to_string(index=False))
    if not HEX.exists():
        import geopandas as gpd
        from shapely.geometry import box
        from _grid import hex_layer
        units = gpd.GeoDataFrame({"unit": ["VA"]}, geometry=[box(12.44, 41.89, 12.47, 41.91)],
                                 crs=4326)
        HEX.parent.mkdir(parents=True, exist_ok=True)
        hex_layer("va", units, census={"VA": TOTAL}, out=HEX)


if __name__ == "__main__":
    main()
