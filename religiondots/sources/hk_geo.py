"""Hong Kong — the 18 District Council districts, from the Home Affairs Department.

Writes data/geo/hk/hk_districts.gpkg. `sources/hk_grid.py` labels the Kontur hexes with
these; `countries.py` never reads this file directly, because Hong Kong has no separate unit
layer -- the hexes ARE the geography, Tonga's wiring exactly.

THE BOUNDARY SOURCE IS ONE FILE AND THERE IS NO JOIN TO GET WRONG, which is worth saying
because §12's playbook is mostly about joins. HAD publishes `hksar_18_district_boundary.json`
as a WGS84 GeoJSON FeatureCollection of exactly 18 features, each carrying the district's
one-letter code in `地區號碼` and its English and Chinese names. **That letter is the same key
the census uses** -- `DC_21C.CSV`'s `dc_class` column -- so boundaries and counts pair on a
published code rather than on a name, which is the thing §12 keeps warning about.

WHAT THE 18 ARE. Four on Hong Kong Island, five in Kowloon, nine in the New Territories.
Mean 412,000 people, from Wan Chai's 167,000 to Sha Tin's 693,000. This is the finest tier at
which the census publishes the ethnicity detail Hong Kong's derived layer needs (the South
Asian split exists only in the thematic report's Table 8.1, which is district-level), so it is
also spec §14.5's ceiling and not an arbitrary stopping point.

Usage:
    python sources/hk_geo.py --fetch    one ~360 KB JSON from had.gov.hk
    python sources/hk_geo.py            rebuild data/geo/hk/hk_districts.gpkg
"""

import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "hk")
GEO = os.path.join(ROOT, "data", "geo", "hk")
OUT = os.path.join(GEO, "hk_districts.gpkg")

URL = ("https://www.had.gov.hk/psi/hong-kong-administrative-boundaries/"
       "hksar_18_district_boundary.json")
RAW_NAME = "hksar_18_district_boundary.json"

EXPECTED = 18

# HAD's English spellings, normalised to the census's. The only two that differ are the
# ampersand and one hyphen, so this is a spelling table and not a join.
RENAME = {
    "Central & Western": "Central and Western",
    "Islands ": "Islands",
}

# The census's own letter codes, from DC_21C.CSV's `dc_class`. Asserted against HAD's
# `地區號碼` rather than trusted: if either side ever renumbers, this fails loudly.
CENSUS_CODES = {
    "Central and Western": "A", "Wan Chai": "B", "Eastern": "C", "Southern": "D",
    "Yau Tsim Mong": "E", "Sham Shui Po": "F", "Kowloon City": "G", "Wong Tai Sin": "H",
    "Kwun Tong": "J", "Kwai Tsing": "S", "Tsuen Wan": "K", "Tuen Mun": "L",
    "Yuen Long": "M", "North": "N", "Tai Po": "P", "Sha Tin": "R",
    "Sai Kung": "Q", "Islands": "T",
}


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, RAW_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
        print("already have", dest)
        return
    print("GET", URL)
    r = requests.get(URL, timeout=300, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(dest + ".part", "wb") as fh:
        fh.write(r.content)
    os.replace(dest + ".part", dest)
    print(f"  {os.path.getsize(dest):,} bytes")


def main():
    import geopandas as gpd

    if "--fetch" in sys.argv:
        fetch()

    src = os.path.join(RAW, RAW_NAME)
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} — run with --fetch first")

    with open(src, encoding="utf-8-sig") as fh:
        fc = json.load(fh)
    feats = fc.get("features", [])
    print(f"HAD boundary file: {len(feats)} features")
    if len(feats) != EXPECTED:
        raise SystemExit(f"!! expected {EXPECTED} districts, found {len(feats)}")

    rows = []
    for f in feats:
        p = f["properties"]
        name = RENAME.get(p["District"].strip(), p["District"].strip())
        letter = p.get("地區號碼", "").strip()
        want = CENSUS_CODES.get(name)
        if want is None:
            raise SystemExit(f"!! HAD district {name!r} is not one of the census's 18")
        if letter != want:
            raise SystemExit(
                f"!! {name}: HAD calls it {letter!r}, the census calls it {want!r}. "
                "One of them has renumbered; do NOT guess which.")
        rows.append({"unit": want, "district": name,
                     "district_zh": p.get("地區", "").strip(),
                     "geometry": f["geometry"]})

    gdf = gpd.GeoDataFrame.from_features(
        [{"type": "Feature",
          "properties": {k: v for k, v in r.items() if k != "geometry"},
          "geometry": r["geometry"]} for r in rows],
        crs="EPSG:4326")

    if set(gdf["unit"]) != set(CENSUS_CODES.values()):
        raise SystemExit("!! the 18 letters do not match the census's set")

    # Hong Kong is 1,110 km2 and the whole of it is inside one degree, so a torn polygon
    # would be obvious here rather than subtle [[reference_antimeridian]].
    w, s, e, n = gdf.total_bounds
    print(f"  bbox {w:.4f} {s:.4f} {e:.4f} {n:.4f}  "
          f"({e - w:.3f}° x {n - s:.3f}°)")
    if not (0.1 < e - w < 1.5 and 0.1 < n - s < 1.5):
        raise SystemExit("!! Hong Kong does not fit in a degree; geometry is torn")

    os.makedirs(GEO, exist_ok=True)
    gdf.to_file(OUT, driver="GPKG", layer="districts")
    print(f"\nwrote {OUT}")
    for _, r in gdf.sort_values("unit").iterrows():
        print(f"  {r['unit']}  {r['district']:<22} {r['district_zh']}")


if __name__ == "__main__":
    main()
