"""T-288: England's inputs for a tier 0 city (London first). Stage a uses only what tier 0 takes
from a country: the outline as whole ITL3 units and district population totals; stage b (OA
population, BRES jobs, the curve fitted on 2011 flows, GIAS, HESA) replaces single inputs here.

- **Outline**: ONS International Territorial Level 3 (January 2025) boundaries, generalised 20 m,
  clipped to the coast (BGC), 182 UK units; the city config lists the units it takes.
- **District totals**: Census 2021 usual residents (TS001, Nomis NM_2021_1) per local authority
  district (April 2023), on ONS's Local Authority Districts (December 2023) BGC polygons. ITL3 units
  are whole districts, so a district is in the pack when its representative point is.

Both from ONS Open Geography and Nomis, open (OGL; Contains OS data). Cached in data/raw/tier0/.
"""
import json
import os

import pandas as pd
import requests
from shapely import make_valid
from shapely.geometry import shape

import build_city as b

ARC = "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/{}/FeatureServer/{}/query?where=1%3D1&outFields={}&outSR=4326&f=geojson"
ITL3_URL = ARC.format("ITL3_JAN_2025_UK_BGC_V2", 1, "ITL325CD,ITL325NM")
LAD_URL = ARC.format("Local_Authority_Districts_December_2023_Boundaries_UK_BGC", 0, "LAD23CD,LAD23NM")
TS001_URL = ("https://www.nomisweb.co.uk/api/v01/dataset/NM_2021_1.data.csv?date=latest&geography=2092957699TYPE424"
             "&c2021_restype_3=0&measures=20100&select=geography_code,geography_name,obs_value")


def raw(name):
    return os.path.join(b.RAW, "tier0", name)


def _get(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        r = requests.get(url, headers=b.UA, timeout=600)
        r.raise_for_status()
        with open(path + ".part", "wb") as f:
            f.write(r.content)
        os.replace(path + ".part", path)
    return path


def itl3_boundary(city, units):
    """Write the city's ITL3 units (codes) as one GeoJSON for tier0's `boundary_file`; its path."""
    out = raw(f"{city}_itl3.geojson")
    if os.path.exists(out):
        return out
    with open(_get(ITL3_URL, raw("itl3_jan2025_uk_bgc.geojson")), encoding="utf-8") as f:
        fc = json.load(f)
    feats = [ft for ft in fc["features"] if ft["properties"]["ITL325CD"] in units]
    missing = set(units) - {ft["properties"]["ITL325CD"] for ft in feats}
    if missing:
        raise ValueError(f"ITL3 units not found: {sorted(missing)}")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    return out


def lad_totals(boundary):
    """[(district name, polygon, Census 2021 usual residents)] for the districts in the boundary."""
    with open(_get(LAD_URL, raw("lad_dec2023_uk_bgc.geojson")), encoding="utf-8") as f:
        fc = json.load(f)
    pop = pd.read_csv(_get(TS001_URL, raw("census2021_ts001_lad_england.csv")))
    pop = dict(zip(pop.GEOGRAPHY_CODE, pop.OBS_VALUE))
    out = []
    for ft in fc["features"]:
        g = make_valid(shape(ft["geometry"]))
        if not boundary.contains(g.representative_point()):
            continue
        code, name = ft["properties"]["LAD23CD"], ft["properties"]["LAD23NM"]
        if code not in pop:
            raise ValueError(f"no Census 2021 total for {code} {name}")
        out.append((name, g, float(pop[code])))
    print(f"  Census 2021 district totals: {len(out)} districts, {sum(r[2] for r in out):,.0f} people", flush=True)
    return out
