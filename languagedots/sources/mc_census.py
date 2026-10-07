"""Monaco: no language statistics. Residents by nationality, IMSEE Recensement de la population
2025 (Tableau 4), read as language -> data/normalized/mc.csv; placement layer
data/geo/mc/mc_hexes.gpkg (Kontur MC, one unit).

    python sources/mc_census.py

SOURCE: IMSEE, "Recensement de la population 2025", May 2026
(imsee.mc/content/download/310026/file/Rapport Recensement 2025.pdf; imsee.mc answers curl with a
403, the copy in data/raw/mc/ came through WebFetch). 38,857 residents on 31 December 2025;
Tableau 4: the 30 commonest nationalities with each one's share of plurinationals.

THE RULE (Anita, 2026-10-05, rich countries with no language question): national language
plus immigrant languages by citizenship, every row `derived`.
  * Monegasques (9,333, counted once whatever their other nationalities) on French. The
    Monegasque language (a Ligurian variety) is taught in school but has almost no native
    speakers; no count exists, so it is not drawn.
  * Foreign nationalities: Tableau 4 counts a plurinational in every community, so each
    nationality is weighted count x (1 - pluri/2) (a dual national counted half in each; the
    report gives no combinations), then through origin_mix (dest "mc").
  * Everyone else (38,857 - 9,333 - the weighted top 29): the 114 smaller nationalities, on
    `other`.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sources"))
sys.path.insert(0, str(ROOT / "taxonomy"))
OUT = ROOT / "data" / "normalized" / "mc.csv"
HEX = ROOT / "data" / "geo" / "mc" / "mc_hexes.gpkg"
TOTAL = 38_857
MONEGASQUE = 9_333
FRENCH = "indoeuropean.romance.french"
# Tableau 4, 2025: nationality -> (iso, count, % plurinational)
T4 = {
    "Français(e)": ("FR", 8270, 6.9), "Italien(ne)": ("IT", 7559, 11.0),
    "Britannique": ("GB", 3081, 23.4), "Suisse": ("CH", 1237, 25.1), "Russe": ("RU", 1209, 41.9),
    "Belge": ("BE", 1038, 9.5), "Allemand(e)": ("DE", 986, 15.3),
    "Néerlandais(e)": ("NL", 503, 9.0), "Portugais(e)": ("PT", 492, 12.9),
    "Américain(e)": ("US", 485, 43.8), "Grec(que)": ("GR", 454, 15.8),
    "Canadien(ne)": ("CA", 416, 40.3), "Espagnol(e)": ("ES", 366, 31.3),
    "Ukrainien(ne)": ("UA", 361, 15.9), "Suédois(e)": ("SE", 350, 13.4),
    "Australien(ne)": ("AU", 276, 36.7), "Danois(e)": ("DK", 269, 8.2),
    "Roumain(e)": ("RO", 261, 15.8), "Autrichien(ne)": ("AT", 233, 14.9),
    "Chypriote": ("CY", 225, 71.4), "Irlandais(e)": ("IE", 223, 36.6),
    "Libanais(e)": ("LB", 223, 44.6), "Brésilien(ne)": ("BR", 214, 56.9),
    "Philippin(e)": ("PH", 205, 4.9), "Israélien(ne)": ("IL", 191, 68.1),
    "Polonais(e)": ("PL", 189, 23.7), "Turc(que)": ("TR", 182, 56.9),
    "Marocain(e)": ("MA", 179, 14.0), "Maltais(e)": ("MT", 129, 70.9),
}


def main():
    from origin_mix import mix
    acc = {FRENCH: float(MONEGASQUE)}
    weighted = 0.0
    for lab, (iso, n, pluri) in T4.items():
        w = n * (1 - pluri / 200)
        weighted += w
        for node, s in mix(iso, "mc").items():
            acc[node] = acc.get(node, 0) + w * s
    rest = TOTAL - MONEGASQUE - weighted
    assert rest > 0, rest
    acc["other"] = acc.get("other", 0) + rest
    print(f"Monegasques {MONEGASQUE:,}; top 29 foreign nationalities weighted {weighted:,.0f}; "
          f"other nationalities {rest:,.0f}")
    s = pd.Series(acc)
    fl = s.apply(int)
    fl[(s - fl).sort_values(ascending=False).index[:TOTAL - int(fl.sum())]] += 1
    df = pd.DataFrame([dict(geo_id="MC", geo_level="country", geo_name="Monaco",
                            source_category=k, count=int(v), tier="derived", year=2025,
                            source_id="imsee_recensement_2025_t4") for k, v in fl.items() if v > 0])
    assert df["count"].sum() == TOTAL
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(df.sort_values("count", ascending=False).head(12)[["source_category", "count"]]
          .to_string(index=False))
    if not HEX.exists():
        hexes()


def hexes():
    import geopandas as gpd
    from shapely.geometry import box
    from _grid import hex_layer
    units = gpd.GeoDataFrame({"unit": ["MC"]}, geometry=[box(7.38, 43.70, 7.46, 43.77)], crs=4326)
    HEX.parent.mkdir(parents=True, exist_ok=True)
    hex_layer("mc", units, census={"MC": TOTAL}, out=HEX)


if __name__ == "__main__":
    main()
