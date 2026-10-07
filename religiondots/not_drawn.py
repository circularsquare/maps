"""
Where the map draws nothing because nobody counted, for the viewer's grey hatching.

    python not_drawn.py                 # every country; reuses cached ones whose inputs are unchanged
    python not_drawn.py --countries ge,az   # recompute just these, keep the rest from the cache
    python not_drawn.py --fresh         # ignore the cache

    -> data/processed/not_drawn.geojson   one feature per hatched area, {cc, kind, name}

Anita, 2026-10-03 (ask/RULINGS.md): *"since there are few empty regions left on the map now (like
abkhazia, karabakh, sinai) we should mark them in a way that kinda indicates that theyre not
drawn. like diagonal grey bands or something."* A blank on a dot map reads as "nobody lives here",
and these blanks are the opposite: people live there and no source this map uses counted them.

TWO KINDS, one file.

  `part`   land inside a BUILT country's outline that no counted unit reaches: Abkhazia and South
           Ossetia, Karabakh, South Sudan's four unsampled states, the Galapagos.
  `whole`  a country, or a territory Natural Earth draws as its own feature, with no entry here
           at all: Lebanon until it is built, Northern Cyprus, Western Sahara.

HOW A `part` IS FOUND. The tempting rule is "the country's outline minus its placement polygons",
and it does not work, because most placement layers are Kontur hexes and Kontur keeps only
populated hexes: the Sahara and the Siberian taiga are outside every hex and are not "not drawn",
they are empty. So the question is asked about PEOPLE first and spread to the land afterwards:

  1. Kontur's global r6 layer (~36 km2 hexes) says where people live, independent of any source.
  2. A populated cell is DRAWN if a placement polygon of a unit that carries counts lies within
     NEAR_KM of it, and NOT DRAWN otherwise. A unit with polygons and no counts (South Sudan's
     Jonglei, Ecuador's Galapagos) does not count as reaching it, and nor does no polygon at all
     (Georgia's grid stops at the Inguri).
  3. Every cell of the outline then takes the answer of its nearest populated cell, so an empty
     mountain range goes with the people on either side of it and a desert with its oases.
     Kontur cells within BORDER_KM of another built outline are not seeds at all, because the
     town across a border river lands inside a simplified outline with nobody of ours near it.
  4. Floors. A hole surrounded by the country's own drawn land must reach ENCLOSED_KM2 and
     ENCLOSED_PEOPLE, or it is a Kontur cell the placement grid happened to miss. Anything else
     (an island, a coast, a border region) needs MIN_KM2 and MIN_PEOPLE, or DENSE_KM2 and
     DENSE_PEOPLE for a small crowded island; areas within GROUP_KM are judged together, so an
     archipelago is one place.
  5. NAMED territory is added whole: ground a country's `gap` says its source left out, taken from
     Natural Earth's disputed-areas layer, for where Kontur itself is nearly empty (South Ossetia).
  6. The dots have the last word (`dot_check`): a part where this country's 1:1,000 dots, or a
     neighbour's 1:10,000 ones, stand for a quarter of Kontur's people is drawn and is dropped.
     That is how Crimea leaves Russia's hatching: Natural Earth gives it to Russia, and the
     survey's dots are there.

The boundary between drawn and not drawn is therefore a nearest-people line, not an
administrative one, except for NAMED territory. It is a hatch on a map read at country zoom, and
that is enough to say "this part"; the country's `gap` sentence, which the viewer shows on hover,
says exactly which.

The OUTLINE is Natural Earth 10m exactly as country_shapes.py builds it (same ISO table, ALSO and
FROM_UNITS) except that CLIP is not applied: the wash leaves Abkhazia out precisely because it is
not drawn, and this layer is where that gets said. Another built country's outline is subtracted
afterwards, so the Netherlands does not hatch Bonaire, which `bq` draws, and the people of an
entry with `territory=False` (the West Bank settlements, `xs`) count as drawn wherever they are.

A `whole` country is its Natural Earth feature, minus every built outline, and only if the built
countries around it do not draw it already: a census can count ground Natural Earth gives a
feature of its own, so a feature whose 1:10,000 dots add up to half its POP_EST is left alone.
NOBODY lists the ones with no resident population to be not drawn (Antarctica, and three sets of
islands with only research or weather stations).

CACHE. Reading every placement layer is ~7 GB, so each country's result is kept in
data/not_drawn_cache.json against the mtimes of its inputs (countries/<cc>.py, its placement
file, its 1:1,000 dots, Natural Earth, its NAMED entries) and recomputed only when one of them
moves. tools/build_tail.py runs it after buffers.py, so a newly built country is picked up by the
tail that publishes it.
"""
import argparse
import json
import os
import sys
import tempfile
import gzip
import shutil
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")      # [[feedback_cap_cpu]]: she is using the box

import numpy as np
import shapely
import shapely.geometry

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
GEO = HERE / "data" / "geo"
OUT = HERE / "data" / "processed" / "not_drawn.geojson"
CACHE = HERE / "data" / "not_drawn_cache.json"
KONTUR_R6 = GEO / "kontur" / "kontur_population_20231101_r6.gpkg.gz"
SHAPES = HERE / "data" / "processed" / "country_shapes.geojson"

RES = 0.02          # degrees per cell, ~2 km. Grown for a country whose box would be too big.
MAX_CELLS = 40e6
# A populated cell this close to a counted unit's polygon is drawn. An r6 cell is ~7 km across,
# and a coastal one can have its centre out on the water: at 3 km, Cleveland's lakefront cells
# hatched 350 km2 of Lake Erie. Narrow territory does not depend on this being small, because
# Transnistria and the other breakaways are NAMED.
NEAR_KM = 4.5
KONTUR_MIN = 25     # people in an r6 cell (~36 km2) before it counts as somewhere people live
MIN_KM2 = 60.0      # smaller hatched areas are border and coast slivers
MIN_PEOPLE = 1000   # by Kontur, in the area, so a sliver of a neighbour's town is not hatched
# ...or a small place holding a lot of people: San Andres is 26 km2 and ~70,000 people, and the
# area floor alone dropped it although Colombia's `gap` names it.
DENSE_KM2, DENSE_PEOPLE = 10.0, 5000
BORDER_KM = 6.0    # Kontur cells this close to another built outline are no evidence
GROUP_KM = 15.0     # areas this close are one place for the two floors above
ENCLOSED_KM2 = 300.0       # a hole in the drawn map smaller than this is a speck, not a place
ENCLOSED_PEOPLE = 5000
SLIVER_KM2 = 5.0    # a part this small, once cut to the coastline, is a cutting artefact
SIMPLIFY = 0.01     # ~1 km; the area edge is a 2 km raster line, so this costs nothing
ROUND = 3

# A Natural Earth feature that is no country's and is still not hatched: nobody lives there to be
# left out. By ADM0_A3.
NOBODY = {
    "ATA",      # Antarctica
    "ATF",      # French Southern and Antarctic Lands: research stations
    "SGS",      # South Georgia and the South Sandwich Islands: research stations
    "CSI",      # Coral Sea Islands: a weather station
}

# NAMED: territory a country's own `gap` says its source did not count, hatched whole whatever
# Kontur says, because Kontur can be nearly empty exactly there: it "barely covers South Ossetia"
# (countries/ge.py), so the people test alone leaves it unhatched. Same entry shape as
# country_shapes.CLIP, which is read as well, so Georgia's two are not listed twice here.
# Natural Earth's Artsakh is the line held from 1994 to 2020, which is the line of Azerbaijan's
# 2019 census; Transnistria includes Bender, which Moldova's census also left out.
def _named():
    import country_shapes as cs
    named = {cc: list(v) for cc, v in cs.CLIP.items()}
    for cc, v in {
        "md": [(cs.DISPUTED, "BRK_NAME", ("Transnistria",))],
        "az": [(cs.DISPUTED, "BRK_NAME", ("Artsakh",))],
    }.items():
        named.setdefault(cc, []).extend(v)
    return named


def named_geoms(cc, named):
    out = []
    for path, prop, names in named.get(cc, []):
        src = json.loads(Path(path).read_text(encoding="utf-8"))
        found = {str(f["properties"].get(prop) or "").strip(): f for f in src["features"]}
        missing = [n for n in names if n not in found]
        if missing:
            raise SystemExit(f"{cc}: {Path(path).name} has no {prop} in {missing}; see NAMED in "
                             "not_drawn.py")
        out.extend(shapely.geometry.shape(found[n]["geometry"]) for n in names)
    return out


# What this layer's result depends on besides the country's own files. Bump to recompute all.
VERSION = 2


def _round(obj):
    if isinstance(obj, list):
        return [_round(o) for o in obj]
    if isinstance(obj, float):
        return round(obj, ROUND)
    return obj


def _mtime(p):
    try:
        return int(os.path.getmtime(p))
    except (OSError, TypeError):
        return None


# ------------------------------------------------------------------------------- outlines
def outlines(countries):
    """{cc: shapely geometry} for every country with a shape, built as country_shapes.py builds
    them less CLIP, and [(a3, name, pop_est, geometry)] for the Natural Earth features no built
    entry takes."""
    import country_shapes as cs
    ne = json.loads(cs.SRC.read_text(encoding="utf-8"))
    shaped = {cc for cc, m in countries.items() if m.get("territory", True)}
    want = {cs.ISO.get(cc, cc.upper()): cc for cc in shaped}
    also_cc = {a3: cc for cc, a3s in cs.ALSO.items() if cc in shaped for a3 in a3s}

    out, extra, rest = {}, {}, []
    for f in ne["features"]:
        p = f["properties"]
        a3 = str(p.get("ADM0_A3") or "").strip()
        g = shapely.geometry.shape(f["geometry"])
        if a3 in also_cc:
            extra.setdefault(also_cc[a3], []).append(g)
            continue
        cc = None
        for key in ("ISO_A2", "ISO_A2_EH"):
            code = str(p.get(key) or "").strip()
            if code in want:
                cc = want[code]
                break
        if cc is None:
            rest.append((a3, p.get("NAME") or p.get("ADMIN"), p.get("POP_EST") or 0, g))
        elif cc not in out:
            out[cc] = g
        else:
            # a second feature for a code: country_shapes.py leaves it out of the outline, and so
            # does this, but it is somebody's land and nobody draws it
            rest.append((a3, p.get("NAME") or p.get("ADMIN"), p.get("POP_EST") or 0, g))
    for cc, gs in extra.items():
        if cc in out:
            out[cc] = shapely.union_all([out[cc]] + gs)
    for cc, gu in cs.FROM_UNITS.items():
        if cc in shaped and cc not in out:
            mu = json.loads(cs.UNITS.read_text(encoding="utf-8"))
            for f in mu["features"]:
                if str(f["properties"].get("GU_A3") or "").strip() == gu:
                    out[cc] = shapely.geometry.shape(f["geometry"])
    return out, rest


# ------------------------------------------------------------------------------- kontur
def kontur_points():
    """(lon, lat, people) for every populated r6 cell, from the global layer's centroids."""
    import pyogrio
    tmp = Path(tempfile.gettempdir()) / f"not_drawn_r6_{os.getpid()}.gpkg"
    with gzip.open(KONTUR_R6, "rb") as f, open(tmp, "wb") as g:
        shutil.copyfileobj(f, g, 1 << 24)
    try:
        df = pyogrio.read_dataframe(tmp, columns=["population"])
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    c = shapely.centroid(df.geometry.values)
    x, y = shapely.get_x(c), shapely.get_y(c)
    # EPSG:3857 -> degrees, by hand: one projection of two million points needs no pyproj
    lon = x / 6378137.0 * 180.0 / np.pi
    lat = np.degrees(2 * np.arctan(np.exp(y / 6378137.0)) - np.pi / 2)
    pop = df["population"].to_numpy(dtype=float)
    keep = pop >= KONTUR_MIN
    return lon[keep], lat[keep], pop[keep]


# ------------------------------------------------------------------------------- one country
def drawn_units(cfg):
    df = cfg["counts"]()
    s = df.groupby("unit")["count"].sum(min_count=1)
    return set(s.index[s.fillna(0) > 0].astype(str))


def drawn_polygons(cc, cfg):
    """The placement polygons of units that carry counts, in EPSG:4326."""
    import scatter
    place = scatter.read_place(cfg)
    units = drawn_units(cfg)
    have = place["unit"].astype(str)
    keep = have.isin(units)
    print(f"  {cc}: {int(keep.sum()):,} of {len(place):,} placement polygons belong to a "
          f"counted unit ({len(set(have[keep])):,} of {have.nunique():,} units)")
    return place.geometry.values[keep.to_numpy()]


def grid_for(bounds):
    w, s, e, n = bounds
    res = RES
    while ((e - w) / res) * ((n - s) / res) > MAX_CELLS:
        res *= 1.25
    pad = 3 * res
    w, s, e, n = w - pad, s - pad, e + pad, n + pad
    nx, ny = int(np.ceil((e - w) / res)), int(np.ceil((n - s) / res))
    from rasterio.transform import from_origin
    return from_origin(w, n, res, res), (ny, nx), res


def part_of(cc, outline, drawn, kpts, also_drawn=(), named=(), others=None):
    """The not-drawn part of one country's outline, as a shapely geometry (or None), with the
    people Kontur puts in it: what the people test finds, plus NAMED territory."""
    geom = _people_part(outline, drawn, kpts, also_drawn, others)
    if named:
        ng = shapely.intersection(shapely.union_all(list(named)), outline)
        geom = ng if geom is None else shapely.union_all([geom, ng])
        geom = _clean(geom)
    if geom is None or geom.is_empty:
        return None, 0
    lon, lat, pop = kpts
    w, s, e, n = geom.bounds
    box = (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
    inside = shapely.contains_xy(geom, lon[box], lat[box])
    return geom, float(pop[box][inside].sum())


def _people_part(outline, drawn, kpts, also_drawn=(), others=None):
    """Steps 1-4 of the module docstring: the outline's ground whose nearest people no counted
    unit reaches."""
    from rasterio import features
    from scipy import ndimage

    tr, shape, res = grid_for(outline.bounds)
    inside = features.rasterize([(outline, 1)], out_shape=shape, transform=tr,
                                all_touched=False, dtype="uint8").astype(bool)
    if not inside.any():
        return None
    geoms = [g for g in list(drawn) + list(also_drawn) if g is not None and not g.is_empty]
    near = np.zeros(shape, dtype=bool)
    if geoms:
        near = features.rasterize(((g, 1) for g in geoms), out_shape=shape, transform=tr,
                                  all_touched=True, dtype="uint8").astype(bool)
    lat0 = (outline.bounds[1] + outline.bounds[3]) / 2
    kx = max(np.cos(np.radians(lat0)), 0.05)
    # Grow by NEAR_KM. Distance in km on the cell grid, longitude scaled at the middle latitude.
    if near.any():
        d = ndimage.distance_transform_edt(~near, sampling=(res * 111.32, res * 111.32 * kx))
        near = d <= NEAR_KM

    # Populated cells inside the outline, from Kontur.
    lon, lat, pop = kpts
    w, n = tr.c, tr.f
    col = ((lon - w) / res).astype(np.int64)
    row = ((n - lat) / res).astype(np.int64)
    ok = (col >= 0) & (col < shape[1]) & (row >= 0) & (row < shape[0])
    col, row, pp = col[ok], row[ok], pop[ok]
    ok = inside[row, col]
    col, row, pp = col[ok], row[ok], pp[ok]
    people = np.zeros(shape, dtype=float)
    np.add.at(people, (row, col), pp)
    seed = people > 0
    # People within BORDER_KM of another built country's outline say nothing either way: two
    # simplified outlines disagree there by a few km, and a Kontur cell of the town across the
    # Kwango, the Congo or the Niger lands inside this outline with nobody of ours near it. Those
    # cells are dropped as seeds, so the ground goes with the people further in.
    if others is not None and not others.is_empty:
        theirs = features.rasterize([(others, 1)], out_shape=shape, transform=tr,
                                    all_touched=True, dtype="uint8").astype(bool)
        if theirs.any():
            d = ndimage.distance_transform_edt(~theirs, sampling=(res * 111.32, res * 111.32 * kx))
            seed &= ~((d <= BORDER_KM) & ~near)
    # a counted unit's own polygons are people too, even where Kontur has nobody
    seed |= near & inside
    if not seed.any():
        return None
    undrawn_seed = seed & ~near
    if not undrawn_seed.any():
        return None

    # Every outline cell takes the answer of its nearest seed.
    _, (ri, ci) = ndimage.distance_transform_edt(~seed, sampling=(1.0, kx), return_indices=True)
    label = undrawn_seed[ri, ci] & inside

    # Groups, with their area and their people, then the two floors. A group is the label grown
    # by GROUP_KM, so an archipelago is judged as one place: the Galapagos' empty Fernandina goes
    # with inhabited Isabela beside it rather than being dropped for holding nobody.
    if not label.any():
        return None
    cell_km2 = (res * 111.32) ** 2
    rows_lat = n - (np.arange(shape[0]) + 0.5) * res
    area_grid = np.broadcast_to((cell_km2 * np.cos(np.radians(rows_lat)))[:, None], shape)
    drawn_land = inside & ~label

    # ENCLOSED areas first: a component whose edge is mostly the country's own drawn land is a
    # hole inside the drawn map, and below ENCLOSED_KM2 / ENCLOSED_PEOPLE it is a Kontur cell the
    # placement grid happened to miss, not a place. Specks like that, ten square kilometres each,
    # were most of the first run's 3,000 parts. An island, a coast or a border area has the sea or
    # a neighbour along most of its edge and is judged by the looser floors below.
    raw, kr = ndimage.label(label)
    idx = np.arange(1, kr + 1)
    edge = ndimage.binary_dilation(label) & ~label
    edge_id = ndimage.grey_dilation(raw, size=3) * edge
    edge_all = ndimage.sum(edge, edge_id, index=idx)
    edge_drawn = ndimage.sum(edge & drawn_land, edge_id, index=idx)
    r_area = ndimage.sum(area_grid, raw, index=idx)
    r_ppl = ndimage.sum(people, raw, index=idx)
    enclosed = edge_drawn >= 0.5 * np.maximum(edge_all, 1)
    keep_raw = np.zeros(kr + 1, dtype=bool)
    keep_raw[1:] = enclosed & (r_area >= ENCLOSED_KM2) & (r_ppl >= ENCLOSED_PEOPLE)
    open_raw = np.zeros(kr + 1, dtype=bool)
    open_raw[1:] = ~enclosed
    open_lab = open_raw[raw]

    # The rest in groups, so an archipelago is judged as one place: the Galapagos' empty
    # Fernandina goes with inhabited Isabela beside it rather than being dropped for holding nobody.
    grown = ndimage.distance_transform_edt(~open_lab, sampling=(res * 111.32, res * 111.32 * kx))
    comp, k = ndimage.label(grown <= GROUP_KM)
    comp = np.where(open_lab, comp, 0)
    gidx = np.arange(1, k + 1)
    area = ndimage.sum(area_grid, comp, index=gidx)
    ppl = ndimage.sum(people, comp, index=gidx)
    good = np.zeros(k + 1, dtype=bool)
    good[1:] = (((area >= MIN_KM2) & (ppl >= MIN_PEOPLE))
                | ((area >= DENSE_KM2) & (ppl >= DENSE_PEOPLE)))
    mask = (good[comp] & open_lab) | (keep_raw[raw] & label)
    if not mask.any():
        return None
    polys = [shapely.geometry.shape(g) for g, v in
             features.shapes(mask.astype("uint8"), mask=mask, transform=tr) if v == 1]
    geom = shapely.union_all(polys)
    geom = shapely.intersection(geom, outline)
    geom = shapely.simplify(geom, SIMPLIFY)
    if not shapely.is_valid(geom):
        geom = shapely.make_valid(geom)
    return _clean(geom)


_DOTS = {}


def _dots(cc, edition=""):
    """A country's dot positions as an (n, 2) array, read once per run."""
    key = (cc, edition)
    if key not in _DOTS:
        p = HERE / "data" / "processed" / f"dots_{cc}{edition}.geojson"
        xy = np.zeros((0, 2))
        if p.exists():
            fs = json.loads(p.read_text(encoding="utf-8"))["features"]
            if fs:
                xy = np.array([f["geometry"]["coordinates"] for f in fs], dtype=float)
        _DOTS[key] = xy
    return _DOTS[key]


def _dots_in(part, xy):
    w, s, e, n = part.bounds
    if not len(xy):
        return 0
    m = (xy[:, 0] >= w) & (xy[:, 0] <= e) & (xy[:, 1] >= s) & (xy[:, 1] <= n)
    return int(shapely.contains_xy(part, xy[m, 0], xy[m, 1]).sum()) if m.any() else 0


def dot_check(cc, geom, kpts, outl, named=None):
    """The last word goes to the dots: a hatched part where the dots stand for a quarter or more of
    the people Kontur puts in it is drawn after all, and is dropped. The country's own 1:1,000
    dots, and any neighbour's at 1:10,000, because a neighbour's source can draw ground inside
    this outline: Natural Earth gives Russia Crimea, and Bangladesh's camps at Cox's Bazar spill
    over the simplified border. It catches what the people test cannot see, a placement weight or
    a coarse grid. NAMED territory is never dropped: the source says it did not count it, so dots
    there are a placement error to fix, not a reason to stop saying so. Returns (what is kept, a
    line per part dropped)."""
    lon, lat, pop = kpts
    keep, dropped = [], []
    for part in shapely.get_parts(geom):
        if named is not None and shapely.area(shapely.intersection(part, named)) > 0.5 * part.area:
            keep.append(part)
            continue
        w, s, e, n = part.bounds
        people_drawn = _dots_in(part, _dots(cc)) * 1000
        for c2, o in outl.items():
            if c2 == cc:
                continue
            ow, os_, oe, on = o.bounds
            if ow > e or oe < w or os_ > n or on < s:
                continue
            people_drawn += _dots_in(part, _dots(c2, "_10k")) * 10_000
        k = (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
        ppl = float(pop[k][shapely.contains_xy(part, lon[k], lat[k])].sum()) if k.any() else 0.0
        if people_drawn and people_drawn >= 0.25 * ppl:
            c = part.centroid
            dropped.append(f"at {c.x:.2f},{c.y:.2f}: dots for {people_drawn:,} against "
                           f"~{ppl:,.0f} by Kontur")
        else:
            keep.append(part)
    return (shapely.union_all(keep) if keep else None), dropped


# ------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--countries", help="comma list to recompute; others come from the cache")
    ap.add_argument("--fresh", action="store_true", help="ignore the cache")
    args = ap.parse_args()

    from countries import COUNTRIES
    import country_shapes as cs

    t0 = time.time()
    outl, rest = outlines(COUNTRIES)
    ne_m = _mtime(cs.SRC)

    cache = {}
    if CACHE.exists() and not args.fresh:
        try:
            cache = json.loads(CACHE.read_text(encoding="utf-8"))
        except ValueError:
            cache = {}
    if cache.get("version") != VERSION:
        cache = {}
    entries = cache.get("countries", {})
    force = set(args.countries.split(",")) if args.countries else set()

    named = _named()

    def key_of(cc):
        cfg = COUNTRIES[cc]
        return [VERSION, ne_m, _mtime(HERE / "countries" / f"{cc}.py"), _mtime(cfg.get("place")),
                _mtime(HERE / "data" / "processed" / f"dots_{cc}.geojson"),
                repr([(Path(p).name, k, list(v)) for p, k, v in named.get(cc, [])])]

    # entries with territory=False: their people are drawn wherever they are, so their counted
    # polygons count as drawn ground for whichever outline they sit in
    floating = [cc for cc, m in COUNTRIES.items() if not m.get("territory", True)]

    # --countries recomputes exactly those and takes everything else from the cache as it is
    todo = [cc for cc in COUNTRIES if cc in outl
            and (cc in force if force else entries.get(cc, {}).get("key") != key_of(cc))]
    kpts = kontur_points() if todo else None
    if todo:
        print(f"{len(todo)} to compute: {', '.join(todo)}  (Kontur read in {time.time() - t0:.0f}s)")
    def others_near(cc, o):
        """Every other built country's outline that comes within reach of this one."""
        b = shapely.buffer(shapely.envelope(o), 0.2)
        near = [g for c2, g in outl.items() if c2 != cc and g.intersects(b)]
        return shapely.union_all(near) if near else None

    float_geoms = []
    if todo and floating:
        for cc in floating:
            float_geoms.extend(drawn_polygons(cc, COUNTRIES[cc]))
    for i, cc in enumerate(todo, 1):
        t = time.time()
        try:
            drawn = drawn_polygons(cc, COUNTRIES[cc])
        except SystemExit as e:
            print(f"  !! {cc}: {e}; left as it was")
            continue
        o = outl[cc]
        near_float = [g for g in float_geoms if g.intersects(o)]
        geom, ppl = part_of(cc, o, drawn, kpts, near_float, named_geoms(cc, named),
                            others_near(cc, o))
        entries[cc] = {"key": key_of(cc), "people": round(ppl),
                       "geometry": None if geom is None else json.loads(shapely.to_geojson(geom))}
        print(f"  [{i}/{len(todo)}] {cc}: " + ("nothing" if geom is None else
              f"{shapely.area(geom):.3f} deg2, ~{ppl:,.0f} people by Kontur")
              + f"  ({time.time() - t:.0f}s)", flush=True)
        # saved as it goes, so an interrupted run keeps what it did
        tmp = CACHE.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": VERSION, "countries": entries}), encoding="utf-8")
        os.replace(tmp, CACHE)

    # Another entry's outline is that entry's ground (bq inside the Netherlands' feature).
    shapes = {}
    if SHAPES.exists():
        for f in json.loads(SHAPES.read_text(encoding="utf-8"))["features"]:
            shapes.setdefault(f["properties"]["cc"], []).append(shapely.geometry.shape(f["geometry"]))
    built_union = shapely.union_all([g for gs in shapes.values() for g in gs]) if shapes else None

    feats, dot_dropped = [], []
    for cc in COUNTRIES:
        e = entries.get(cc)
        if not e or not e.get("geometry") or cc not in outl:
            continue
        raw = shapely.geometry.shape(e["geometry"])
        w, s, ee, n = raw.bounds
        near = sorted(c2 for c2, o in outl.items()
                      if not (o.bounds[0] > ee or o.bounds[2] < w or o.bounds[1] > n or o.bounds[3] < s))
        kept_key = [e["key"]] + [(c2, _mtime(HERE / "data" / "processed" / f"dots_{c2}_10k.geojson"))
                                 for c2 in near]
        if e.get("kept_key") != json.loads(json.dumps(kept_key)):
            if kpts is None:
                kpts = kontur_points()
            ng = named_geoms(cc, named)
            kept, dropped = dot_check(cc, raw, kpts, outl,
                                      shapely.union_all(ng) if ng else None)
            e["kept"] = None if kept is None else json.loads(shapely.to_geojson(kept))
            e["dropped"] = dropped
            e["kept_key"] = kept_key
            tmp = CACHE.with_suffix(".tmp")
            tmp.write_text(json.dumps({"version": VERSION, "countries": entries}), encoding="utf-8")
            os.replace(tmp, CACHE)
        dot_dropped += [f"{cc} {d}" for d in e.get("dropped", [])]
        if not e.get("kept"):
            continue
        g = shapely.geometry.shape(e["kept"])
        others = [x for c2, gs in shapes.items() if c2 != cc for x in gs if x.intersects(g)]
        if others:
            g = shapely.difference(g, shapely.union_all(others))
        g = _clean(g)
        if g is None:
            continue
        feats.append({"type": "Feature",
                      "properties": {"cc": cc, "kind": "part", "name": COUNTRIES[cc]["name"]},
                      "geometry": _round(json.loads(shapely.to_geojson(g)))})

    # A feature of its own in Natural Earth can still be drawn by a neighbour's source: a census
    # that counts a territory Natural Earth separates (Baikonur, the British bases on Cyprus). So
    # a `whole` feature is hatched only if the dots of the built countries around it put fewer
    # than half its Natural Earth population inside it. 1:1,000 dots for a feature under
    # 200,000 people, where a 1:10,000 dot spilling over a simplified border is most of Monaco.
    def drawn_in(g, pop):
        w, s, e, n = g.bounds
        ed, per = ("", 1000) if pop < 200_000 else ("_10k", 10_000)
        hits = 0
        for cc, o in outl.items():
            ow, os_, oe, on = o.bounds
            if ow > e + 1 or oe < w - 1 or os_ > n + 1 or on < s - 1:
                continue
            hits += _dots_in(g, _dots(cc, ed))
        return hits * per

    whole, by_neighbour = [], []
    for a3, name, pop, g in rest:
        if a3 in NOBODY or not pop or pop <= 0:
            continue
        if built_union is not None:
            g = shapely.difference(g, built_union)
        g = _clean(shapely.simplify(g, SIMPLIFY / 4), min_km2=0)
        if g is None:
            continue
        drawn_people = drawn_in(g, pop)
        if drawn_people >= 0.5 * pop:
            by_neighbour.append(f"{name} ({drawn_people:,} drawn of {int(pop):,})")
            continue
        whole.append((name, pop))
        feats.append({"type": "Feature", "properties": {"cc": None, "kind": "whole", "name": name},
                      "geometry": _round(json.loads(shapely.to_geojson(g)))})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps({"type": "FeatureCollection", "features": feats},
                              separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    # On Windows the swap is refused while a reader (the dev server) holds the old file open
    for attempt in range(60):
        try:
            os.replace(tmp, OUT)
            break
        except PermissionError:
            if attempt == 59:
                raise SystemExit(f"could not replace {OUT.name}, held open for a minute; the new "
                                 f"output is in {tmp.name}")
            time.sleep(1)
    parts = [f["properties"]["cc"] for f in feats if f["properties"]["kind"] == "part"]
    print(f"\nparts of built countries ({len(parts)}): " + ", ".join(
        f"{cc} ~{entries[cc]['people']:,}" for cc in parts))
    print(f"countries with no entry ({len(whole)}): " + ", ".join(
        f"{n} ({int(p):,})" for n, p in sorted(whole, key=lambda x: -x[1])))
    if by_neighbour:
        print("not hatched, a built neighbour draws them: " + ", ".join(by_neighbour))
    if dot_dropped:
        print("dropped, the dots say they are drawn:\n  " + "\n  ".join(dot_dropped))
    print(f"wrote {OUT.name}  ({len(feats)} features, {OUT.stat().st_size / 1024:.0f} KB) "
          f"in {time.time() - t0:.0f}s")
    return 0


def _clean(g, min_km2=SLIVER_KM2):
    """Polygons only, valid, and no part under `min_km2` (a degree is ~111 km, scaled by latitude)."""
    if g is None or g.is_empty:
        return None
    if not shapely.is_valid(g):
        g = shapely.make_valid(g)
    keep = []
    for p in shapely.get_parts(g):
        if p.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        lat = p.centroid.y
        km2 = shapely.area(p) * 111.32 ** 2 * max(np.cos(np.radians(lat)), 0.05)
        if km2 >= min_km2 or min_km2 == 0:
            keep.append(p)
    if not keep:
        return None
    return shapely.union_all(keep)


if __name__ == "__main__":
    sys.exit(main())
