"""T-041: a water mask from OpenStreetMap, for any city in the world (the US packs keep TIGER).

    python pipeline/osm_water.py read <extract> [<extract> ...]
    python pipeline/osm_water.py mask <id> --extracts a,b --bbox w,s,e,n --origin lon,lat [--cell 25] [--min-width 2] [--png out.png]

Two sources, both global, both (c) OpenStreetMap contributors, ODbL:

- **Sea**: the coastline-derived water polygons of osmdata.openstreetmap.de
  (`water-polygons-split-4326.zip`, WGS84, split into tiles of at most 100 km). One 906 MB zip for
  the whole world, downloaded once into data/raw/osmdata/ and kept; each city reads only the
  polygons meeting its box (GDAL reads the shapefile inside the zip with a box filter). Inside the
  coastline is land; tidal rivers above the point where OSM closes the coastline are inland water.
- **Inland water**: areas from the Geofabrik extract(s) of the city's region (the same .osm.pbf
  files routes.py and canada.py read): `natural=water` (lakes, ponds, rivers drawn as areas,
  reservoirs, basins with water in them, canals as areas), `waterway=riverbank|dock|canal` areas
  and `landuse=reservoir`; not `intermittent`/`seasonal` water, covered water (`covered=yes`,
  `tunnel`, `location=underground`) or `landuse=basin` alone (mostly dry detention basins).
  Read once per extract with GDAL's OSM driver and cached as `data/raw/osm/<extract>.water.fgb`
  (a few MB), so the next city in the same region reads only that.

`source(extracts)` is what `water.build_mask` takes as `source` (tier0 config `water_source`), so
the mask is rasterised, encoded and checked exactly as the US and Canadian ones (T-030; format in
notes/T-004.md). `mask` writes a water-only pack (the header's `water` block and the `water_row`/
`water_x` arrays, nothing else) to data/work/osm_water/<id>.json/.bin, runs read_pack's water
checks on it and draws it with look_water.py; a city build makes its own from the same source.
The extracts must cover the whole box: inland water outside them is missing (the sea is global).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")  # before numpy; Anita is using the machine

import json
import sys
import time
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OSM = os.path.join(RAW, "osm")
SEA_ZIP = os.path.join(RAW, "osmdata", "water-polygons-split-4326.zip")
SEA_URL = "https://osmdata.openstreetmap.de/download/water-polygons-split-4326.zip"
OUT = os.path.join(ROOT, "data", "work", "osm_water")
UA = {"User-Agent": "trainworld-pipeline/0.1 (rail game data build; python-requests)"}

INLAND_WHERE = ("\"natural\" = 'water' OR landuse = 'reservoir' OR "
                "waterway IN ('riverbank', 'dock', 'canal')")
NOT_WATER = {"intermittent": {"yes"}, "seasonal": {"yes"}, "covered": {"yes"},
             "tunnel": None, "location": {"underground", "underwater"}}  # None: any value but "no"


# ---------------------------------------------------------------- downloads

def _download(url, path):
    import requests
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"  downloading {url}", flush=True)
    t0 = time.time()
    with requests.get(url, headers=UA, timeout=600, stream=True) as r:
        r.raise_for_status()
        with open(path + ".part", "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    os.replace(path + ".part", path)
    print(f"  {os.path.getsize(path) / 1e6:.0f} MB in {time.time() - t0:.0f} s", flush=True)
    return path


def pbf_path(extract):
    return os.path.join(OSM, extract.replace("/", "_") + ".osm.pbf")


def water_path(extract):
    return os.path.join(OSM, extract.replace("/", "_") + ".water.fgb")


# ---------------------------------------------------------------- sea

def sea(bbox_ll):
    """The coastline-derived water polygons meeting bbox_ll = (w, s, e, n), clipped to it."""
    import pyogrio
    from shapely import box, make_valid
    path = _download(SEA_URL, SEA_ZIP)
    with zipfile.ZipFile(path) as z:
        shp = next(n for n in z.namelist() if n.endswith(".shp"))
    df = pyogrio.read_dataframe(f"/vsizip/{path}/{shp}", bbox=tuple(bbox_ll), columns=[])
    ext = box(*bbox_ll)
    out = []
    for g in df.geometry.to_numpy():
        g = make_valid(g).intersection(ext)
        if not g.is_empty:
            out.append(g)
    return out


# ---------------------------------------------------------------- inland water

def _osmconf():
    """GDAL's osmconf.ini with closed `waterway` ways made polygons (old riverbanks and docks
    drawn as plain closed ways) and the water keys as columns."""
    import pyogrio
    src = os.path.join(os.path.dirname(pyogrio.__file__), "gdal_data", "osmconf.ini")
    dst = os.path.join(OSM, "osmconf_water.ini")
    with open(src) as f:
        text = f.read()
    text = text.replace("closed_ways_are_polygons=aeroway,", "closed_ways_are_polygons=waterway,aeroway,", 1)
    head, sep, tail = text.partition("[multipolygons]")
    tail = tail.replace("attributes=name,type,", "attributes=name,type,waterway,water,", 1)
    os.makedirs(OSM, exist_ok=True)
    with open(dst + ".part", "w") as f:
        f.write(head + sep + tail)
    os.replace(dst + ".part", dst)
    return dst


def _tag(other, key):
    """A value from GDAL's other_tags text ('"k"=>"v","k2"=>"v2"'), or ''."""
    if not other:
        return ""
    i = other.find('"' + key + '"=>"')
    if i < 0:
        return ""
    j = i + len(key) + 5
    return other[j:other.find('"', j)]


def read_inland(extract):
    """The extract's inland water areas as a GeoDataFrame (kind, name), cached as .water.fgb."""
    import pyogrio
    out = water_path(extract)
    if os.path.exists(out):
        return pyogrio.read_dataframe(out)
    path = _download(f"https://download.geofabrik.de/{extract}-latest.osm.pbf", pbf_path(extract))
    t0 = time.time()
    df = pyogrio.read_dataframe(path, layer="multipolygons", where=INLAND_WHERE,
                                columns=["name", "natural", "landuse", "waterway", "water", "other_tags"],
                                CONFIG_FILE=_osmconf())
    n_read = len(df)
    other = df["other_tags"].fillna("").to_numpy()
    keep = np.ones(len(df), bool)
    for i, o in enumerate(other):
        for k, bad in NOT_WATER.items():
            v = _tag(o, k)
            if v and (v != "no" if bad is None else v in bad):
                keep[i] = False
                break
    df = df[keep & df.geometry.notna().to_numpy()].copy()
    kind = np.where(df["natural"].fillna("").to_numpy() == "water", "water:" + df["water"].fillna("").to_numpy(),
                    np.where(df["landuse"].fillna("").to_numpy() == "reservoir", "landuse:reservoir",
                             "waterway:" + df["waterway"].fillna("").to_numpy()))
    df = df.assign(kind=kind)[["kind", "name", "geometry"]]
    pyogrio.write_dataframe(df, out + ".part.fgb", driver="FlatGeobuf", SPATIAL_INDEX="YES")
    os.replace(out + ".part.fgb", out)
    print(f"  {extract}: {n_read:,} water areas read, {len(df):,} kept, {os.path.getsize(out) / 1e6:.1f} MB "
          f"-> {out}; {time.time() - t0:.0f} s", flush=True)
    return df


def inland(extracts, bbox_ll):
    """Inland water polygons of every extract meeting bbox_ll, clipped to it."""
    import pyogrio
    from shapely import box, make_valid
    ext = box(*bbox_ll)
    out = []
    for e in extracts:
        read_inland(e)  # make sure the cache exists
        df = pyogrio.read_dataframe(water_path(e), bbox=tuple(bbox_ll))
        for g in df.geometry.to_numpy():
            g = make_valid(g)
            if g.intersects(ext):
                g = g.intersection(ext)
                if not g.is_empty:
                    out.append(g)
    return out


# ---------------------------------------------------------------- the source

def source(extracts):
    """A `source(bbox_ll) -> (lon/lat water polygons, info)` for water.build_mask (tier0's
    `water_source`): the OSM sea plus the extracts' inland water."""
    extracts = list(extracts)

    def src(bbox_ll):
        t0 = time.time()
        s = sea(bbox_ll)
        i = inland(extracts, bbox_ll)
        print(f"  OSM water over {', '.join(f'{v:.3f}' for v in bbox_ll)}: {len(s):,} sea polygons, "
              f"{len(i):,} inland; {time.time() - t0:.0f} s", flush=True)
        info = {"source": "OpenStreetMap: coastline water polygons (osmdata.openstreetmap.de) + inland water "
                          f"areas from Geofabrik {', '.join(extracts)} (T-041)",
                "sea_polygons": len(s), "inland_polygons": len(i), "extracts": extracts}
        return s + i, info
    return src


def mask_on_grid(geoms, lon0, lat0, x0, y0, cell, W, H):
    """Rasterise lon/lat water polygons onto a given grid (the pack frame of (lon0, lat0))."""
    import water
    local = [water.to_local(g, lon0, lat0) for g in geoms]
    x1, y1, x2, y2, gid = water.polygon_edges(local)
    _, row, c0, c1 = water.scanline_runs(x1, y1, x2, y2, gid, x0, y0, cell, W, H)
    return water.WaterMask.from_runs(float(x0), float(y0), float(cell), W, H, row, c0, c1)


# ---------------------------------------------------------------- a water-only pack

def write_pack(name, lon0, lat0, mask, info, out_dir=OUT):
    """The mask in the pack format (header `water` block, arrays `water_row` u32 and `water_x` u16,
    8-byte aligned), with nothing else: data/work/osm_water/<name>.json/.bin."""
    os.makedirs(out_dir, exist_ok=True)
    row = mask.row_off.astype("<u4").tobytes()
    xs = mask.xs.astype("<u2").tobytes()
    off2 = (len(row) + 7) // 8 * 8
    buf = row + b"\0" * (off2 - len(row)) + xs
    header = {"format": 1, "kind": "water only (T-041)", "origin": {"lon": lon0, "lat": lat0},
              "arrays": [{"name": "water_row", "dtype": "u32", "group": "water", "offset": 0, "count": len(mask.row_off)},
                         {"name": "water_x", "dtype": "u16", "group": "water", "offset": off2, "count": len(mask.xs)}],
              "water": mask.header(info)}
    for ext, data, mode in ((".bin", buf, "wb"), (".json", json.dumps(header, indent=1), "w")):
        p = os.path.join(out_dir, name + ext)
        with open(p + ".part", mode) as f:
            f.write(data)
        os.replace(p + ".part", p)
    return header


def build(name, extracts, bbox_ll, origin, cell=25.0, min_width_px=2, png=None):
    import water
    import read_pack
    t0 = time.time()
    lon0, lat0 = origin
    mask, info = water.build_mask(lon0, lat0, bbox_ll, cell, None, None, source=source(extracts))
    raw_runs = len(mask.xs) // 2
    if min_width_px > 1:
        mask = mask.without_narrow(min_width_px)
    info["min_width_m"] = min_width_px * cell
    write_pack(name, lon0, lat0, mask, info)
    print(f"water: {mask.W} x {mask.H} pixels of {cell:.0f} m; {raw_runs:,} runs, {len(mask.xs) // 2:,} after dropping "
          f"water under {info['min_width_m']:.0f} m wide; water {mask.water_pixels() / (mask.W * mask.H):.1%} of the "
          f"extent; {time.time() - t0:.0f} s -> {os.path.join(OUT, name)}.json/.bin", flush=True)
    if png:
        import look_water
        look_water.main(name, png, packs=OUT)
    header, a = read_pack.load_pack(name, OUT)
    read_pack.check_water(name, header, a)  # read_pack.WATER_TESTS[name], if it has any


def _arg(argv, flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def main(argv):
    if not argv or argv[0] not in ("read", "mask"):
        raise SystemExit(__doc__)
    if argv[0] == "read":
        _download(SEA_URL, SEA_ZIP)
        for e in argv[1:]:
            read_inland(e)
        return
    name = argv[1]
    extracts = _arg(argv, "--extracts").split(",")
    bbox = tuple(float(v) for v in _arg(argv, "--bbox").split(","))
    origin = tuple(float(v) for v in _arg(argv, "--origin").split(","))
    build(name, extracts, bbox, origin, float(_arg(argv, "--cell", 25.0)), int(_arg(argv, "--min-width", 2)),
          _arg(argv, "--png"))


if __name__ == "__main__":
    main(sys.argv[1:])
