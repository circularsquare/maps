"""Rohingya camp outlines (Cox's Bazar) -> data/geo/bd/bd_camps.gpkg, for placement only.

    python sources/bd_camps.py [--fetch]

The 2011 census predates the 2017 arrivals, and the camps were not in it, but Kontur's 2023 grid
puts close to half of Ukhia upazila's people inside them. Placed on Kontur as is, Ukhia's and
Teknaf's census-counted (Bengali-speaking) residents would be drawn mostly in the camps.
countries/bd.py zeroes the weight of hexes whose centre lies in a camp, so those residents are
drawn where they live. Nobody in the camps is drawn: no census counts them (sources/bd.md).

Source: RRRC / ISCG Site Management Sector / UNHCR / IOM, "Outline of camps sites of Rohingya
refugees in Cox's Bazar, Bangladesh", A1 camp outlines of 2023-04-12, on HDX, CC0.
"""
import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "bd" / "20230412_a1_camp_outlines.zip"
URL = ("https://data.humdata.org/dataset/1a67eb3b-57d8-4062-b562-049ad62a85fd/resource/"
       "ace4b0a6-ef0f-46e4-a50a-8c552cfe7bf3/download/20230412_a1_camp_outlines.zip")
OUT = HERE / "data" / "geo" / "bd" / "bd_camps.gpkg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch or not RAW.exists():
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        RAW.parent.mkdir(parents=True, exist_ok=True)
        RAW.write_bytes(urllib.request.urlopen(req, timeout=300).read())
    import geopandas as gpd
    shp = [n for n in zipfile.ZipFile(RAW).namelist() if n.lower().endswith(".shp")]
    if len(shp) != 1:
        raise SystemExit(f"bd_camps: expected one shapefile in the zip, found {shp}")
    g = gpd.read_file(f"zip://{RAW}!{shp[0]}").to_crs(4326)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    w, s, e, n = g.total_bounds
    # all camps lie in Ukhia and Teknaf, Cox's Bazar
    if not (91.9 < w and e < 92.4 and 20.7 < s and n < 21.4):
        raise SystemExit(f"bd_camps: bounds {g.total_bounds} are outside Ukhia-Teknaf")
    if not 30 <= len(g) <= 40:
        raise SystemExit(f"bd_camps: {len(g)} camp polygons, expected 33-34")
    area = g.to_crs(32646).area.sum() / 1e6
    OUT.parent.mkdir(parents=True, exist_ok=True)
    g[["geometry"]].to_file(OUT, driver="GPKG")
    print(f"bd_camps.gpkg: {len(g)} camps, {area:.1f} km2, bounds {[round(x, 3) for x in g.total_bounds]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
