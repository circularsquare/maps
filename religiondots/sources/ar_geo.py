"""Argentina — the six CEIL survey regions, built from OCHA COD-AB provinces and partidos.

Writes data/geo/ar/ar_regions.gpkg: 6 polygons, `unit` (AMBA, CENTRO, NEA, NOA, CUYO,
PATAGONIA) + `name`.

**FIVE REGIONS ARE UNIONS OF PROVINCES AND ONE CUTS A PROVINCE IN TWO.** AMBA is the Ciudad
Autónoma plus the 24 partidos of Gran Buenos Aires, and the rest of Buenos Aires province is
in Centro. So provinces come from COD-AB level 1 and Buenos Aires comes from level 2. The 24
partidos are READ from the census table sources/ar.py uses (`read_gba24`) rather than typed
here, and joined on INDEC's code, which COD's pcode carries after `AR0` (AR006028 is 06028,
Almirante Brown); the NAMES on those codes are then asserted to agree one at a time, because
a code scheme that matches is a hypothesis and not a convention (sources.md §9bl).

Which province is in which region is not written down anywhere CEIL publishes; sources/ar.py's
docstring and sources/ar.md say how it was read off the survey's own map. Entre Ríos in NEA is
the one surprise, and it is theirs.

**THE MALVINAS, SOUTH GEORGIA AND ANY ANTARCTIC PART ARE DROPPED FROM PATAGONIA.** The census
enumerated nobody there and the survey sampled nobody there, so a Patagonian share placed on
those islands would be dots for people no source counted. This is a statement about coverage
and not about sovereignty: the unit is what INDEC counted. Kontur's hexes on the islands then
fall outside every region and are dropped in sources/ar_grid.py, which asserts it.

Usage:
    python sources/ar_geo.py --fetch     one 41 MB zip from HDX
    python sources/ar_geo.py             rebuild from data/raw/ar/
"""

import os
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ar")
GEO = os.path.join(ROOT, "data", "geo", "ar")
OUT = os.path.join(GEO, "ar_regions.gpkg")

COD_URL = ("https://data.humdata.org/dataset/c661e398-66cf-4a9f-9607-4962c72d1ccf/resource/"
           "9f3b2c43-ad1c-406a-85dd-d387c5e3ffb3/download/arg_adm_unhcr2017_shp.zip")
COD_ZIP = os.path.join(RAW, "arg_adm_unhcr2017_shp.zip")

# COD-AB ADM1_PCODE -> (COD's own spelling, CEIL region). Buenos Aires is split below.
ADM1 = {
    "AR002": ("Ciudad de Buenos Aires", "AMBA"),
    "AR006": ("Buenos Aires", None),
    "AR010": ("Catamarca", "NOA"),
    "AR014": ("Córdoba", "CENTRO"),
    "AR018": ("Corrientes", "NEA"),
    "AR022": ("Chaco", "NEA"),
    "AR026": ("Chubut", "PATAGONIA"),
    "AR030": ("Entre Ríos", "NEA"),
    "AR034": ("Formosa", "NEA"),
    "AR038": ("Jujuy", "NOA"),
    "AR042": ("La Pampa", "CENTRO"),
    "AR046": ("La Rioja", "NOA"),
    "AR050": ("Mendoza", "CUYO"),
    "AR054": ("Misiones", "NEA"),
    "AR058": ("Neuquén", "PATAGONIA"),
    "AR062": ("Río negro", "PATAGONIA"),
    "AR066": ("Salta", "NOA"),
    "AR070": ("San Juan", "CUYO"),
    "AR074": ("San Luis", "CUYO"),
    "AR078": ("Santa Cruz", "PATAGONIA"),
    "AR082": ("Santa Fe", "CENTRO"),
    "AR086": ("Santiago del Estero", "NOA"),
    "AR090": ("Tucumán", "NOA"),
    "AR094": ("Tierra del Fuego", "PATAGONIA"),
}
BA_PARTIDOS = 135

# The census's long names for the two jurisdictions COD spells shorter.
CENSUS_ALIAS = {
    "ciudad autonoma de buenos aires": "ciudad de buenos aires",
    "tierra del fuego antartida e islas del atlantico sur": "tierra del fuego",
}

# A polygon part whose representative point is east of this and south of -50 is the Malvinas
# or South Georgia; anything south of -56 is Antarctic. Isla de los Estados (-64.5 to -63.8)
# stays.
OFFSHORE_EAST_OF = -62.5
OFFSHORE_SOUTH_OF = -50.0
ANTARCTIC_SOUTH_OF = -56.0

MAX_SPAN_LON = 21.5          # mainland Argentina is about 20 degrees wide
MAX_AMBA_SPAN = 1.2


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.replace("\xa0", " ").replace(",", " ").replace(".", " ").replace("-", " ")
    return " ".join(s.lower().split())


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(COD_ZIP) and os.path.getsize(COD_ZIP) > 10_000_000:
        print("already have", os.path.basename(COD_ZIP))
        return
    print("GET", COD_URL)
    r = requests.get(COD_URL, timeout=900, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(COD_ZIP, "wb") as fh:
        fh.write(r.content)
    print(f"  {len(r.content):,} bytes")


def _to_lonlat(gdf, label):
    """Put one COD-AB layer in EPSG:4326, checking the coordinates rather than the .prj.

    **THE TWO LAYERS DO NOT SHARE A CRS.** ADM1 reads as EPSG:3857 and ADM2's .prj is a
    different file of a different size, so a union across them without reprojecting each on
    its own would join Web Mercator metres to degrees. And a .prj can simply be wrong, so the
    bounds decide: they must land on Argentina in lon/lat afterwards or nothing is written.
    """
    minx, miny, maxx, maxy = gdf.total_bounds
    looks_degrees = max(abs(minx), abs(maxx)) <= 180 and max(abs(miny), abs(maxy)) <= 90
    epsg = gdf.crs.to_epsg() if gdf.crs is not None else None
    if epsg == 4326:
        pass
    elif looks_degrees:
        print(f"  {label}: coordinates are already degrees although the .prj says "
              f"{gdf.crs}; labelling as EPSG:4326 rather than reprojecting")
        gdf = gdf.set_crs(4326, allow_override=True)
    elif gdf.crs is None:
        raise SystemExit(f"{label} has no CRS and its coordinates are not degrees")
    else:
        gdf = gdf.to_crs(4326)
    minx, miny, maxx, maxy = gdf.total_bounds
    if not (-75 < minx < -70 and -56.5 < miny < -54 and -55 < maxx < -52 and -23 < maxy < -21):
        raise SystemExit(f"{label} is not Argentina in lon/lat: bounds "
                         f"[{minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f}]")
    print(f"  {label}: {len(gdf)} features, source crs {epsg}, bounds "
          f"[{minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f}]")
    return gdf


def _valid(g):
    try:
        from shapely import make_valid
        return make_valid(g)
    except ImportError:
        return g.buffer(0)


def drop_offshore(geom, label):
    """Remove the Malvinas, South Georgia and Antarctic parts; return the rest and a report."""
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.ops import unary_union

    parts = []
    for g in getattr(geom, "geoms", [geom]):
        if isinstance(g, Polygon):
            parts.append(g)
        elif isinstance(g, MultiPolygon):
            parts.extend(g.geoms)
    keep, dropped = [], []
    for p in parts:
        c = p.representative_point()
        if (c.x > OFFSHORE_EAST_OF and c.y < OFFSHORE_SOUTH_OF) or c.y < ANTARCTIC_SOUTH_OF:
            dropped.append(p)
        else:
            keep.append(p)
    for p in dropped:
        c = p.representative_point()
        print(f"    dropped from {label}: part at ({c.x:.2f}, {c.y:.2f}), "
              f"{p.area:.4f} sq deg")
    return unary_union(keep), len(dropped)


def main():
    import geopandas as gpd
    from shapely.ops import unary_union

    sys.path.insert(0, HERE)
    import ar

    if "--fetch" in sys.argv:
        fetch()
    if not os.path.exists(COD_ZIP):
        raise SystemExit(f"missing {COD_ZIP} -- run with --fetch first")

    a1 = _to_lonlat(gpd.read_file(f"zip://{COD_ZIP}!arg_admbnda_adm1_unhcr2017.shp"), "ADM1")
    a2 = _to_lonlat(gpd.read_file(f"zip://{COD_ZIP}!arg_admbnda_adm2_unhcr2017.shp"), "ADM2")
    print(f"COD-AB: {len(a1)} provinces, {len(a2)} departments, both now EPSG:4326")

    # ---- provinces: codes and names both, in both directions
    got = dict(zip(a1["ADM1_PCODE"], a1["ADM1_ES"]))
    if set(got) != set(ADM1):
        raise SystemExit(f"COD's ADM1 codes {sorted(got)} are not the crosswalk's")
    for code, (name, _reg) in ADM1.items():
        if _norm(got[code]) != _norm(name):
            raise SystemExit(f"{code} is {got[code]!r} in COD-AB, expected {name!r}")

    # ---- the two crosswalks (this file's, by COD code; ar.py's, by census name) must agree
    by_name = {_norm(n): reg for n, reg in ADM1.values()}
    for census_name, reg in ar.CENSUS_TO_REGION.items():
        key = CENSUS_ALIAS.get(_norm(census_name), _norm(census_name))
        if key not in by_name:
            raise SystemExit(f"census jurisdiction {census_name!r} has no COD province")
        if by_name[key] != reg:
            raise SystemExit(f"{census_name}: ar.py says {reg}, ar_geo.py says {by_name[key]}")

    # ---- Buenos Aires partidos: the 24 from the census, joined on code, names asserted
    ba = a2[a2["ADM1_PCODE"] == "AR006"].copy()
    if len(ba) != BA_PARTIDOS:
        raise SystemExit(f"COD-AB has {len(ba)} Buenos Aires partidos, expected {BA_PARTIDOS}")
    ba["indec"] = ba["ADM2_PCODE"].str[3:]
    partidos, gba_people, _ = ar.read_gba24()
    cod_names = dict(zip(ba["indec"], ba["ADM2_ES"]))
    for code, (name, _people) in partidos.items():
        if code not in cod_names:
            raise SystemExit(f"census partido {code} {name} has no COD-AB polygon")
        if _norm(cod_names[code]) != _norm(name):
            raise SystemExit(f"code {code} is {name!r} in the census and {cod_names[code]!r} "
                             "in COD-AB -- the code join is pairing the wrong partido")
    ba["region"] = ba["indec"].map(lambda c: "AMBA" if c in partidos else "CENTRO")
    print(f"  Buenos Aires: {(ba['region'] == 'AMBA').sum()} partidos to AMBA "
          f"({gba_people:,} people in 2022), {(ba['region'] == 'CENTRO').sum()} to Centro; "
          "codes and names agree on all 24")

    # ---- union
    pieces = {r: [] for r in ar.REGIONS}
    for _, row in a1.iterrows():
        reg = ADM1[row["ADM1_PCODE"]][1]
        if reg is not None:
            pieces[reg].append(_valid(row.geometry))
    for reg, sub in ba.groupby("region"):
        pieces[reg].extend(_valid(g) for g in sub.geometry)

    rows, dropped = [], 0
    for reg in ar.REGIONS:
        geom = unary_union(pieces[reg])
        if reg == "PATAGONIA":
            geom, dropped = drop_offshore(geom, reg)
        rows.append({"unit": reg, "name": ar.REGION_NAMES[reg], "geometry": geom})
    out = gpd.GeoDataFrame(rows, crs=a1.crs)
    if dropped == 0:
        print("  (no offshore parts found in Patagonia; COD's Tierra del Fuego is the "
              "Isla Grande only)")

    minx, miny, maxx, maxy = out.total_bounds
    print(f"  bbox: [{minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f}]")
    if maxx - minx > MAX_SPAN_LON:
        raise SystemExit(f"the layer is {maxx - minx:.1f} degrees wide, which is not "
                         "mainland Argentina -- an offshore part survived")
    amba = out.loc[out["unit"] == "AMBA"].total_bounds
    if amba[2] - amba[0] > MAX_AMBA_SPAN or amba[3] - amba[1] > MAX_AMBA_SPAN:
        raise SystemExit(f"AMBA spans {amba}, far too large for the capital's conurbation")

    eq = out.to_crs("EPSG:6933")
    for (_, r), area in zip(out.iterrows(), eq.geometry.area / 1e6):
        b = r.geometry.bounds
        print(f"    {r['unit']:10s} {area:>10,.0f} km2  [{b[0]:.2f}, {b[1]:.2f}, "
              f"{b[2]:.2f}, {b[3]:.2f}]")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, driver="GPKG", layer="ar_regions")
    print(f"  wrote {OUT}  {len(out)} regions")


if __name__ == "__main__":
    main()
