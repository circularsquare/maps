"""Argentina placement layer: Kontur 400 m hexes keyed to the census's 527 departamentos.

    python sources/ar_geo.py [--fetch]

Writes data/geo/ar/ar_departamentos.gpkg (unit, name, geometry) and data/geo/ar/ar_hexes.gpkg
(unit, pop) through sources/_grid.py's hex_layer.

UNITS. The Instituto Geografico Nacional's `departamento` layer (wms.ign.gob.ar WFS,
typeName ign:departamento, 529 polygons, EPSG:4326). Its `in1` is INDEC's five-digit
provincia + departamento code, the same code REDATAM's DPTO prints, so the join is on code and
asserted both ways: every census departamento has exactly one polygon, and the only polygons
with no census row are 94021 Islas del Atlantico Sur (the Malvinas and South Atlantic
islands, not enumerated) and 94028 Antartida Argentina (81 people at the bases, all in
collective dwellings; not drawn, see countries/ar.py). The IGN layer is current: it has
Tolhuin (94011, Tierra del Fuego Ley 1.186) and CABA's 15 comunas, which the census uses;
religiondots' 2017 COD-AB does not have Tolhuin. The 140 MB WFS answer is deleted once the
gpkgs are built; --fetch gets it again.

THE KONTUR CHECK, 2026-10-04: 146,909 of Kontur's 45.8M people fall outside every polygon
(river and coast edges); per departamento Kontur/census median 1.01, log r 0.971 against a
shuffled best of 0.151. 26 of 527 units are outside a factor of 3, and they are Kontur's: all
25 of Corrientes' departamentos sit at 0.23-0.29 (Kontur under-counts the whole province
evenly, so placement inside each is unaffected), and Tolhuin at 5.9, where Kontur spreads
some 27,000 phantom people over the Fuegian forest around a town of 6,039.

KONTUR. kontur_population_AR_20231101, the copy religiondots already downloaded
(religiondots/data/raw/ar/, read only) copied into languagedots/data/geo/kontur/ so _grid.py
finds it without downloading again.
"""
import os
import shutil
import sys
from pathlib import Path

# the WFS answer is one 140 MB GeoJSON object, over GDAL's default 200 MB-in-memory guard
os.environ.setdefault("OGR_GEOJSON_MAX_OBJ_SIZE", "0")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "sources"))
import _grid  # noqa: E402

RAW = HERE / "data" / "raw" / "ar"
GEO = HERE / "data" / "geo" / "ar"
IGN_JSON = RAW / "ign_departamento.json"
UNITS = GEO / "ar_departamentos.gpkg"
WFS = ("https://wms.ign.gob.ar/geoserver/ign/ows?service=WFS&version=1.0.0&request=GetFeature"
       "&typeName=ign:departamento&outputFormat=application/json")
RD_KONTUR = HERE.parent / "religiondots" / "data" / "raw" / "ar" / "kontur_population_AR_20231101.gpkg"
NOT_ENUMERATED = {"94021"}           # Islas del Atlantico Sur
COLLECTIVE_ONLY = {"94028"}          # Antartida Argentina


def fetch():
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    if not (IGN_JSON.exists() and IGN_JSON.stat().st_size > 10_000_000):
        print("GET", WFS)
        r = requests.get(WFS, timeout=1800, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        IGN_JSON.write_bytes(r.content)
    dest = _grid.OUR_KONTUR / RD_KONTUR.name
    if not dest.exists() and RD_KONTUR.exists():
        _grid.OUR_KONTUR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(RD_KONTUR, dest)
        print("copied", RD_KONTUR.name, "from religiondots")


def main():
    if "--fetch" in sys.argv:
        fetch()
    import geopandas as gpd

    g = gpd.read_file(IGN_JSON)
    g = g.rename(columns={"in1": "unit", "nam": "name"})[["unit", "name", "geometry"]]
    g["unit"] = g["unit"].astype(str).str.zfill(5)
    if g["unit"].duplicated().any():
        raise SystemExit(f"duplicate IGN codes: {sorted(g.loc[g['unit'].duplicated(), 'unit'])}")
    if g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"IGN layer CRS {g.crs}, expected EPSG:4326")

    df = pd.read_csv(HERE / "data" / "normalized" / "ar.csv", dtype={"geo_id": str})
    pop = df.groupby("geo_id")["count"].sum()
    census = set(pop.index)
    ign = set(g["unit"])
    no_poly = sorted(census - ign)
    no_census = sorted(ign - census)
    print(f"{len(g)} IGN departamentos, {len(census)} census departamentos")
    if no_poly:
        raise SystemExit(f"census departamentos with no IGN polygon: {no_poly}")
    if set(no_census) != NOT_ENUMERATED:
        raise SystemExit(f"IGN polygons with no census row: {no_census}, expected "
                         f"{sorted(NOT_ENUMERATED)}")
    print(f"ok    join both ways on code; unmatched polygon {no_census} (not enumerated)")

    g = g[~g["unit"].isin(NOT_ENUMERATED | COLLECTIVE_ONLY)].copy()
    GEO.mkdir(parents=True, exist_ok=True)
    g.to_file(UNITS, layer="departamentos", driver="GPKG")
    print(f"wrote {UNITS} ({len(g)} units)")

    _grid.hex_layer("ar", g, census={u: int(pop[u]) for u in g["unit"]})


if __name__ == "__main__":
    main()
