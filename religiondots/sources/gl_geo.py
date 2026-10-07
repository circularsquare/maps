"""
Greenland: units and placement.

    python sources/gl_geo.py
    -> data/geo/gl/gl_units.gpkg     the five municipalities + Northeast Greenland National Park
    -> data/geo/gl/gl_hexes.gpkg     one placement disc per locality, `unit` and `pop`

UNITS: geoBoundaries GRL ADM1 (gbOpen, release 9469f09, CC BY-SA 3.0, data/raw/gl/
geoBoundaries-GRL-ADM1.geojson), keyed on shapeISO (GL-KU, GL-SM, GL-QE, GL-QT, GL-AV, GL-UO).

PLACEMENT IS BY LOCALITY, NOT KONTUR. Kontur 2023 holds about 20,000 people in Greenland against
56,740 registered, so its hexes would put most dots nowhere in particular. Statistics Greenland
counts every resident by locality (BEXSTD, sources/gl.py), so each of the 80-odd towns and
settlements gets a disc of DISC_KM radius, clipped to the unit's land where that leaves anything,
weighted by its Greenland-born residents. Localities are geocoded from GeoNames GL (populated
places, feature class P), on the name before any bracket; the three names Greenland uses twice
(Aappilattoq, Tasiusaq, Ikerasaarsuk) are told apart by which municipality the point is in.
"Uoplyst i X distrikt" (no fixed locality, a handful of people) goes to the district's town.
Every locality with people must geocode and fall in or within EDGE_KM of its own unit.
"""
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "gl")
GEO = os.path.join(ROOT, "data", "geo", "gl")
GB = os.path.join(RAW, "geoBoundaries-GRL-ADM1.geojson")
LOC = os.path.join(RAW, "gl_localities_born_2026.csv")
GEONAMES = os.path.join(RAW, "geonames_GL.zip")
UNITS_OUT = os.path.join(GEO, "gl_units.gpkg")
PLACE_OUT = os.path.join(GEO, "gl_hexes.gpkg")

DISC_KM = 1.5
TOWN_KM = 300.0     # a settlement this far from its district's town is a wrong namesake
EDGE_KM = 50.0      # geoBoundaries ADM1 leaves out Disko Island (Qeqertarsuaq ~36 km, Kangerluk ~40 km off Qeqertalik) and generalises island coasts
EXPECTED_UNITS = {"GL-KU", "GL-SM", "GL-QE", "GL-QT", "GL-AV", "GL-UO"}
# Hand coordinates where GeoNames has no populated place of the name (checked 2026-10-03).
HAND = {
    "9612099ZZZ": ("Daneborg", 74.30, -20.22),      # outside the municipalities: the park
    "9601609INN": ("Innaarsuit", 73.20, -56.02),    # Upernavik district; not in GeoNames GL
    "9561803ISO": ("Isertoq", 65.54, -38.97),       # Tasiilaq district; not in GeoNames GL
}
# Current Statistics Greenland names that GeoNames holds under an older spelling.
ALIAS = {"Tiilerilaaq": "Tiniteqilaaq", "Kangerluk": "Diskofjord", "Naajaat": "Naajat"}
GEONAMES_COLS = ["geonameid", "name", "asciiname", "alternatenames", "lat", "lon", "fclass",
                 "fcode", "cc", "cc2", "admin1", "admin2", "admin3", "admin4", "population",
                 "elevation", "dem", "timezone", "modified"]


def fold(s):
    s = str(s).lower().replace("ø", "o").replace("å", "a").replace("æ", "ae")
    return re.sub(r"[^a-z]", "", s)


def km(lat0, lon0, lat, lon):
    p = np.radians
    a = (np.sin(p(lat - lat0) / 2) ** 2
         + np.cos(p(lat0)) * np.cos(p(lat)) * np.sin(p(lon - lon0) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def disc(lat, lon, r_km):
    from pyproj import Transformer
    from shapely.geometry import Point
    from shapely.ops import transform
    aeqd = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +units=m"
    back = Transformer.from_crs(aeqd, "EPSG:4326", always_xy=True).transform
    return transform(back, Point(0, 0).buffer(r_km * 1000, 32))


def main():
    import geopandas as gpd
    from shapely.geometry import Point

    os.makedirs(GEO, exist_ok=True)
    u = gpd.read_file(GB)[["shapeISO", "shapeName", "geometry"]].rename(
        columns={"shapeISO": "unit", "shapeName": "name"}).to_crs(4326)
    if set(u["unit"]) != EXPECTED_UNITS:
        raise SystemExit(f"geoBoundaries units {sorted(u['unit'])}")
    u.to_file(UNITS_OUT, driver="GPKG")

    loc = pd.read_csv(LOC, dtype={"code": str})
    loc = loc[loc["n"] > 0].copy()
    with zipfile.ZipFile(GEONAMES) as zf:
        g = pd.read_csv(zf.open("GL.txt"), sep="\t", header=None, names=GEONAMES_COLS,
                        quoting=3, dtype=str, keep_default_na=False)
    g = g[g["fclass"] == "P"].copy()
    g["lat"], g["lon"] = g["lat"].astype(float), g["lon"].astype(float)
    keys, alt = {}, {}
    for i, r in g.iterrows():
        for nm in (r["name"], r["asciiname"]):
            keys.setdefault(fold(nm), set()).add(i)
        for nm in r["alternatenames"].split(","):
            if nm:
                alt.setdefault(fold(nm), set()).add(i)
    pts = gpd.GeoDataFrame(g, geometry=gpd.points_from_xy(g["lon"], g["lat"]), crs=4326)
    pts = gpd.sjoin(pts, u[["unit", "geometry"]], how="left", predicate="within")
    in_unit = dict(zip(pts.index, pts["unit"]))

    town_of = {c[:5]: c for c in loc["code"] if c[5:7] == "00"}
    # towns first, so a settlement's namesakes can be told apart by distance to its district town
    loc["is_town"] = loc["code"].str[5:7] == "00"
    loc = loc.sort_values("is_town", ascending=False)
    town_xy = {}
    rows = []
    for _i, r in loc.iterrows():
        c = r["code"]
        if c in HAND:
            nm, lat, lon = HAND[c]
        else:
            if "ZZZ" in c:
                c2 = town_of.get(c[:5])
                if c2 is None:
                    raise SystemExit(f"{r['locality']}: no town in its district")
                name = loc.loc[loc["code"] == c2, "locality"].iloc[0]
            else:
                name = r["locality"]
            nm = re.sub(r"\s*\(.*$", "", name).strip()
            f = fold(nm)
            fa = fold(ALIAS.get(nm, ""))
            # primary names only: GeoNames' alternate-name field matched dozens of unrelated
            # places for some settlement names and snapped them onto their town
            cand = sorted(keys.get(f, set()) or keys.get(fa, set()))
            geom_u = u.loc[u["unit"] == r["unit"], "geometry"].iloc[0]
            if len(cand) > 1 and not r["is_town"] and c[:5] in town_xy:
                tlat, tlon = town_xy[c[:5]]
                dd = {i: float(km(tlat, tlon, g.loc[i, "lat"], g.loc[i, "lon"])) for i in cand}
                best = min(cand, key=dd.get)
                print(f"  {nm}: {len(cand)} GeoNames candidates, took the one {dd[best]:.0f} km "
                      f"from its district town ({g.loc[best, 'lat']:.3f}, {g.loc[best, 'lon']:.3f})")
                cand = [best]
            if len(cand) > 1:
                inside = [i for i in cand if in_unit.get(i) == r["unit"]]
                if inside:
                    cand = inside
                else:
                    # coastal points often fall just off a generalised coastline: nearest to
                    # the unit, and only those within EDGE_KM of it
                    dd = {i: geom_u.distance(Point(g.loc[i, "lon"], g.loc[i, "lat"])) * 111
                          * np.cos(np.radians(g.loc[i, "lat"])) for i in cand}
                    cand = [i for i in cand if dd[i] <= EDGE_KM] or cand
            if len(cand) > 1:
                d = g.loc[cand]
                # Several GeoNames rows of one name in one municipality: take the one with a
                # population, else the first; printed.
                d = d.sort_values("population", key=lambda s: -pd.to_numeric(s, errors="coerce").fillna(0))
                print(f"  {nm}: {len(cand)} GeoNames candidates in {r['unit']}, took "
                      f"{d.iloc[0]['geonameid']} ({d.iloc[0]['lat']:.3f}, {d.iloc[0]['lon']:.3f})")
                cand = [d.index[0]]
            if not cand:
                raise SystemExit(f"{r['locality']} ({c}): {nm!r} not in GeoNames GL; add to HAND")
            lat, lon = g.loc[cand[0], "lat"], g.loc[cand[0], "lon"]
        if r["is_town"]:
            town_xy[c[:5]] = (lat, lon)
        elif c[:5] in town_xy:
            dt = float(km(*town_xy[c[:5]], lat, lon))
            if dt > TOWN_KM:
                raise SystemExit(f"{r['locality']} is {dt:.0f} km from its district town")
        geom_u = u.loc[u["unit"] == r["unit"], "geometry"].iloc[0]
        p = Point(lon, lat)
        if not geom_u.contains(p):
            dist = geom_u.boundary.distance(p) * 111 * np.cos(np.radians(lat))
            if dist > EDGE_KM:
                raise SystemExit(f"{r['locality']} at {lat:.3f},{lon:.3f} is ~{dist:.0f} km "
                                 f"outside {r['unit']}")
        d = disc(lat, lon, DISC_KM)
        clipped = d.intersection(geom_u)
        if clipped.area < 0.2 * d.area:
            clipped = d
        rows.append(dict(unit=r["unit"], locality=r["locality"], code=c, lat=lat, lon=lon,
                         pop=float(r["n"]), geometry=clipped))
    out = gpd.GeoDataFrame(rows, crs=4326)
    if set(out["unit"]) != EXPECTED_UNITS:
        raise SystemExit(f"placement units {sorted(set(out['unit']))}")
    if int(out["pop"].sum()) != int(loc["n"].sum()):
        raise SystemExit("placement pop does not add to the localities")
    out.to_file(PLACE_OUT, driver="GPKG")
    print(f"wrote {UNITS_OUT} ({len(u)} units) and {PLACE_OUT} ({len(out)} discs, "
          f"{int(out['pop'].sum()):,} Greenland-born)")
    print(out.groupby("unit")["pop"].agg(["count", "sum"]).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
