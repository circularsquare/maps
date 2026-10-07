"""Tuvalu: 2022 Population and Housing Census report (Tuvalu CSD / SPC, 2025) -> data/normalized/
tv.csv, and a two-unit placement layer data/geo/tv/tv_hexes.gpkg (Nui apart from the rest).

    python sources/tv_census.py

NO LANGUAGE QUESTION (literacy by language only, Table 10). Ethnicity, Figure 6 (resident
population 10,632, Table 9): Tuvaluan 94%, Tuvaluan/I-Kiribati 4%, Tuvaluan/Other 1%, Other 1%,
not stated 1% (the figure's rounding sums to 101). AGENT_BRIEF section 2, ethnicity read as
language:
  * Nui (514 enumerated there, Table 9): the island speaks Nuian, a dialect of Gilbertese
    (Glottolog lists Tuvalu among Gilbertese's countries; the report's own literacy table names
    "Nuian" as one of Tuvalu's three languages). Everyone enumerated on Nui is drawn on
    Gilbertese.
  * Elsewhere (10,118): Tuvaluan for the three Tuvaluan answers (the mixed Tuvaluan/I-Kiribati
    households live in Tuvaluan-speaking islands; no retention source says otherwise), `other`
    for "Other", "not stated" left out (gap). Shares from Figure 6, applied to all of the rest.
Report: https://www.spc.int/digitallibrary/get/zskjx (data/raw/tv/tuvalu_2022_census_report.pdf).

PLACEMENT: religiondots' tv_hexes.gpkg (one unit, read-only) re-keyed: hexes whose centroid is
inside Nui's box (lon 177.10-177.20, lat -7.30 to -7.18) become `TV-NUI`, the rest `TV-REST`.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rdlink import RD_GEO  # noqa: E402

OUT = ROOT / "data" / "normalized" / "tv.csv"
HEX = ROOT / "data" / "geo" / "tv" / "tv_hexes.gpkg"
TOTAL = 10_632
NUI = 514
SHARES = {"Tuvaluan": 0.94, "Tuvaluan/I-Kiribati": 0.04, "Tuvaluan/Other": 0.01, "Other": 0.01,
          "Not stated": 0.01}
NUI_BOX = (177.10, -7.30, 177.20, -7.18)


def hexes():
    import geopandas as gpd
    g = gpd.read_file(RD_GEO / "tv" / "tv_hexes.gpkg")
    c = g.to_crs(3857).geometry.centroid.to_crs(4326)
    x0, y0, x1, y1 = NUI_BOX
    nui = (c.x > x0) & (c.x < x1) & (c.y > y0) & (c.y < y1)
    g["unit"] = "TV-REST"
    g.loc[nui, "unit"] = "TV-NUI"
    s = g.groupby("unit")["pop"].sum()
    print(f"  hexes: {nui.sum()} on Nui ({s['TV-NUI']:.0f} Kontur people), "
          f"{(~nui).sum()} elsewhere ({s['TV-REST']:.0f})")
    assert 4 <= nui.sum() <= 60 and 300 < s["TV-NUI"] < 1500, "Nui box caught the wrong hexes"
    HEX.parent.mkdir(parents=True, exist_ok=True)
    g.to_file(HEX, driver="GPKG")


def main():
    rest = TOTAL - NUI
    tot_share = sum(SHARES.values())
    rows = [("TV-NUI", "Nui (Nuian)", NUI)]
    acc = 0
    for lab, s in SHARES.items():
        n = round(rest * s / tot_share)
        rows.append(("TV-REST", lab, n))
        acc += n
    rows[1] = (rows[1][0], rows[1][1], rows[1][2] + rest - acc)   # rounding onto Tuvaluan
    df = pd.DataFrame([dict(geo_id=u, geo_level="island", geo_name=u, source_category=lab,
                            count=n, tier="derived", year=2022, source_id="tv_census_2022_fig6")
                       for u, lab, n in rows])
    assert df["count"].sum() == TOTAL
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(df[["geo_id", "source_category", "count"]].to_string(index=False))
    hexes()


if __name__ == "__main__":
    main()
