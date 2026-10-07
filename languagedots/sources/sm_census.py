"""San Marino: no language statistics. Residents by citizenship (register, 31 December 2024),
Italian and Romagnol for Sammarinese and Italian citizens, other citizenships on `other`
-> data/normalized/sm.csv; placement layer data/geo/sm/sm_hexes.gpkg (Kontur SM, one unit).

    python sources/sm_census.py

SOURCE: Ufficio Informatica, Tecnologia, Dati e Statistica, Bollettino di Statistica I trimestre
2025, Tavola 1.8 "Popolazione presente per cittadinanza", column Dicembre 2024, residents
(popolazione con residenza anagrafica): Sammarinesi 28,204, Italiani 5,030, Altri 811, total
34,045 (data/raw/sm/bollettino_202503.pdf). San Marino holds no census with a language question.

ROMAGNOL: the Sammarinese dialect is a Romagnol variety (Glottolog's Romagnol lists SM). No
Sammarinese count of home use exists; Foresti (1998 survey, summarised in Treccani's "La
Repubblica di San Marino", 2020) found about 70% grew up with dialect present, alone or beside
Italian, and the family-domain figures match, with use concentrated among the old. Drawn: the
share of Rimini province's non-immigrant population that Italy's build draws on Romagnol
(sources/it.md: ISTAT 2024 "L'uso della lingua italiana, dei dialetti...", Emilia-Romagna's
dialect-in-the-family rate, "both" counting half) = 17.6%, applied to Sammarinese and Italian
citizens. A borrowed rate, not a Sammarinese figure; Foresti's 70% is the upper bound (dialect
present at all, a generation ago). Everyone else of those two citizenships on Italian. "Altri"
(811, no breakdown published) on `other`. Every row `derived`.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sources"))
OUT = ROOT / "data" / "normalized" / "sm.csv"
HEX = ROOT / "data" / "geo" / "sm" / "sm_hexes.gpkg"
SAMMARINESI, ITALIANI, ALTRI = 28_204, 5_030, 811
TOTAL = 34_045
RIMINI = "ITH59"
ROMAGNOL = "indoeuropean.romance.romagnol"
ITALIAN = "indoeuropean.romance.italian"


def rimini_rate():
    d = pd.read_csv(ROOT / "data" / "normalized" / "it.csv", dtype={"geo_id": str})
    x = d[d["geo_id"] == RIMINI].groupby("source_category")["count"].sum()
    rate = x["Romagnol"] / (x["Romagnol"] + x["Italian"])
    print(f"  Rimini (it.csv {RIMINI}): Romagnol {x['Romagnol']:,.0f}, Italian {x['Italian']:,.0f}"
          f" -> {rate:.3f}")
    assert 0.10 < rate < 0.30, rate
    return round(rate, 3)


def main():
    assert SAMMARINESI + ITALIANI + ALTRI == TOTAL
    rate = rimini_rate()
    base = SAMMARINESI + ITALIANI
    rom = round(base * rate)
    df = pd.DataFrame([dict(geo_id="SM", geo_level="country", geo_name="San Marino",
                            source_category=c, count=n, tier="derived", year=2024,
                            source_id="sm_bollettino_2025q1_t1.8") for c, n in
                       ((ROMAGNOL, rom), (ITALIAN, base - rom), ("other", ALTRI))])
    assert df["count"].sum() == TOTAL
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(df[["source_category", "count"]].to_string(index=False))
    if not HEX.exists():
        import geopandas as gpd
        from shapely.geometry import box
        from _grid import hex_layer
        units = gpd.GeoDataFrame({"unit": ["SM"]}, geometry=[box(12.38, 43.88, 12.53, 44.00)],
                                 crs=4326)
        HEX.parent.mkdir(parents=True, exist_ok=True)
        hex_layer("sm", units, census={"SM": TOTAL}, out=HEX)


if __name__ == "__main__":
    main()
