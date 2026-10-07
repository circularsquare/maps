"""Finland placement layer: Statistics Finland's own 1 km population grid, keyed to municipality.

    python sources/fi_geo.py --fetch     download the grid (WFS, about 30 MB) if missing
    python sources/fi_geo.py

-> data/geo/fi/fi_grid1km.gpkg (unit = three-digit municipality code, pop = residents)

WHY THIS GRID AND NOT KONTUR. Statistics Finland publishes the population register on 1 km
squares (`vaestoruutu:vaki2025_1km` on geo.stat.fi, CC BY 4.0): every inhabited square, its
resident count (squares under 10 people included; only their age and sex are blanked) and the
municipality the office files it under. It is the same register as the language table, so the
squares are where the people counted in the table are registered to live. Kontur models people
from buildings, and Finland has about half a million summer cottages: religiondots' Iceland
(playbooks/geography.md, "Kontur can put a country's countryside on its summer houses") read
rural municipalities at 6-10x their register population for exactly that reason. Here the
municipality of each square is the office's own, so there is no centroid join to get wrong.

A 1 km square is coarser than a Kontur hex (0.74 km2), which matters nowhere at this map's scale:
the median municipality is several hundred squares.

Squares are not clipped to municipal lines: a border square's dots may fall up to a kilometre
over the line, on the side the office did not file them under. scatter.py clips the sea.

CHECKS: square ids unique; every square's municipality is one of the 308 in the language table
and every one of the 308 has squares; per municipality, grid residents against the language
table's total, printed as a ratio and held to a band (see BAND). The grid is 5,573,552
residents against the table's 5,652,881 (31 Dec 2025; 5,635,971 a year earlier). Squares hold
only residents the register can place on a coordinate, and the layer's name `vaki2025` does not
say which year end it is. It is only a weight inside each municipality; every dot count comes
from the language table.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "fi"
GRID = RAW / "vaki2025_1km.geojson"
MUNI_JSON = RAW / "11rm_2025.json"
OUT = HERE / "data" / "geo" / "fi" / "fi_grid1km.gpkg"
URL = ("https://geo.stat.fi/geoserver/vaestoruutu/wfs?service=WFS&version=2.0.0"
       "&request=GetFeature&typeNames=vaestoruutu:vaki2025_1km&outputFormat=json"
       "&srsName=EPSG:3067&propertyName=grd_id,kunta,vaesto,geom")
N_MUNI = 308
# Grid residents against the table's municipality total. Measured 2026-10-04: nationally 0.986
# (see the docstring), p10 0.970,
# p90 0.995. The two tails are border squares filed under the neighbour: Kauniainen (235), an
# enclave in Espoo, 0.749; Jokioinen (170) 1.094. The band is set to catch a wrong join or a
# wrong vintage of the municipality codes (which would put whole towns at 0 or 2+), not those.
BAND = 0.30


def fetch():
    import requests

    if GRID.exists() and GRID.stat().st_size > 1_000_000:
        print("already have", GRID)
        return
    RAW.mkdir(parents=True, exist_ok=True)
    print("GET", URL)
    r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                     timeout=1200)
    r.raise_for_status()
    if not r.content.lstrip().startswith(b"{"):
        raise SystemExit(f"not GeoJSON: {r.content[:120]!r}")
    tmp = GRID.with_suffix(".part")
    tmp.write_bytes(r.content)
    os.replace(tmp, GRID)
    print(f"  {GRID.stat().st_size:,} bytes")


def table_totals():
    """{three-digit code: total population} from 11rm (sources/fi_register.py's raw file)."""
    js = json.loads(MUNI_JSON.read_text(encoding="utf-8"))
    areas = js["dimension"]["alue_23_20260101"]["category"]["index"]
    langs = js["dimension"]["kieli_15_20180102"]["category"]["index"]
    n_lang = len(langs)
    out = {}
    for code, ai in areas.items():
        if code == "SSS":
            continue
        out[code[2:]] = js["value"][ai * n_lang + langs["SSS"]]
    assert len(out) == N_MUNI and all(isinstance(v, int) for v in out.values())
    return out


def main():
    if "--fetch" in sys.argv:
        fetch()
    g = gpd.read_file(GRID)
    if len(g) == 0:
        raise SystemExit("the grid has ZERO features")
    print(f"grid: {len(g):,} squares, {g['vaesto'].sum():,.0f} residents, CRS {g.crs}")
    assert g["grd_id"].is_unique, "a square id repeats: is a border square split by municipality?"
    assert (g["vaesto"] > 0).all(), "a square with no residents, or a suppressed total"
    g["unit"] = g["kunta"].astype(str).str.zfill(3)

    tab = table_totals()
    a, b = set(g["unit"]), set(tab)
    print(f"join: {len(a)} municipalities in the grid, {len(b)} in the language table; "
          f"only in the grid {sorted(a - b)}, only in the table {sorted(b - a)}")
    assert a == b, "the grid's municipality codes are not the table's 2026 division"

    per = g.groupby("unit")["vaesto"].sum()
    ratio = (per / pd.Series(tab)).sort_values()
    nat = per.sum() / sum(tab.values())
    print(f"grid / table nationally {nat:.4f} ({per.sum():,} against {sum(tab.values()):,}); "
          f"per municipality p10 {ratio.quantile(0.1):.4f}  median {ratio.median():.4f}  "
          f"p90 {ratio.quantile(0.9):.4f}")
    print("  lowest:  " + ", ".join(f"{u} {r:.3f}" for u, r in ratio.head(5).items()))
    print("  highest: " + ", ".join(f"{u} {r:.3f}" for u, r in ratio.tail(5).items()))
    off = ratio[(ratio < 1 - BAND) | (ratio > 1 + BAND)]
    assert off.empty, f"municipalities outside {BAND:.0%} of the table: {off.round(3).to_dict()}"

    layer = gpd.GeoDataFrame({"unit": g["unit"], "pop": g["vaesto"].astype(float)},
                             geometry=g.geometry, crs=g.crs).to_crs(4326)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    layer.to_file(OUT, layer="grid", driver="GPKG")
    print(f"wrote {OUT} ({len(layer):,} squares, {layer['unit'].nunique()} municipalities)")


if __name__ == "__main__":
    main()
