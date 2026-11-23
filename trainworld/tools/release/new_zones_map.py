"""New zones map: one image of every zone a release adds or grows, for the release post.

    venv\\Scripts\\python.exe trainworld/tools/release/new_zones_map.py --version v0.4.0

Run from the maps root. Writes artifacts/trainworld/review/<version>-new-zones.png (2400x1350).

  --before  the outlines.json of the last published game: a file or an https URL. Default: the live game's.
            The live data version is read from the tw-data meta of https://anita.garden/trainsworld/ (the
            website repo's trainsworld/index.html if offline) and its outlines.json is downloaded to
            trainworld/data/work/ (gitignored, one file per data version).
  --after   the outlines.json about to ship. Default trainworld/data/packs/outlines.json.
  --title   the heading. Default "new zones in <version>".
  --grid    four equal panels in a 2x2 grid (the biggest cluster and the next three), not one big map and insets.
  --out     the png. Default as above.

A zone is new when its id is not in --before. A kept zone counts as grown when it gained more than
0.002 square degrees. The new and grown places are grouped into clusters (single linkage, about 1000 km);
the biggest cluster gets the main map and up to three more clusters, by new area, get insets named after
the countries holding their new area (largest first, up to 4). Each map is a Lambert azimuthal equal-area projection centred on its cluster.
A cluster under 1.5% of the new area, or already inside the main map, gets no inset. With nothing new
the script says so and writes nothing. Reads data only; nothing in data/packs or data/raw changes.
"""

import argparse
import gzip
import io
import json
import math
import os
import re
import sys
import urllib.request
import warnings

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, box
from shapely.ops import transform, unary_union

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.dirname(os.path.dirname(HERE))
MAPS = os.path.dirname(TW)
WORK = os.path.join(TW, "data", "work")
RAW = os.path.join(TW, "data", "raw", "world_field")
PAGE = "https://anita.garden/trainsworld/"
DATA_ROOT = "https://data.anita.garden/trainworld/"
SITE_INDEX = os.path.join(os.path.dirname(MAPS), "website", "trainsworld", "index.html")

W, H = 2400, 1350
BG, LAND, MUTED, NEW, EDGE, INK = "#dfe9f1", "#f3f0e8", "#a9bccf", "#ee4b2b", "#ffffff", "#3a4350"
NEW_EDGE = "#c23a1e"
GROWN_MIN = 0.002  # square degrees
LINK_DEG = 9.0  # clusters join when their boxes are closer than this (about 1000 km)
INSET_MIN_SHARE = 0.015
FONTS = [os.path.expanduser("~/AppData/Local/Microsoft/Windows/Fonts/Nunito-Regular.ttf"),
         "C:/Windows/Fonts/Nunito-Regular.ttf"]


def read_bytes(src):
    if re.match(r"https?://", src):
        req = urllib.request.Request(src, headers={"User-Agent": "trainworld-release-tools"})
        with urllib.request.urlopen(req, timeout=120) as r:
            b = r.read()
    else:
        with open(src, "rb") as f:
            b = f.read()
    return gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b


def live_data_url():
    html = None
    try:
        html = read_bytes(PAGE).decode("utf-8", "replace")
    except Exception as e:
        print(f"live page unreachable ({e}); using {SITE_INDEX}")
        with open(SITE_INDEX, encoding="utf-8") as f:
            html = f.read()
    m = re.search(r'name="tw-data"\s+content="([^"]+)"', html)
    if not m:
        raise SystemExit("no tw-data meta found on the live page")
    return m.group(1).rstrip("/") + "/outlines.json"


def cached_before(src):
    """Local path of the before file; downloads a URL once into data/work/."""
    if not re.match(r"https?://", src):
        return src
    os.makedirs(WORK, exist_ok=True)
    name = "outlines-" + re.sub(r"[^0-9A-Za-z.-]+", "_", src.split("//", 1)[1].replace("/outlines.json", "")) + ".json"
    path = os.path.join(WORK, name)
    if not os.path.exists(path):
        print(f"downloading {src}")
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(read_bytes(src))
        os.replace(tmp, path)
    return path


def load_rings(path):
    d = json.loads(read_bytes(path))
    return d["q"], d["zones"]


def decode(q, rings_list):
    ps = []
    for r in rings_list:
        x, y = r[0], r[1]
        pts = [(x / q, y / q)]
        for i in range(2, len(r), 2):
            x += r[i]
            y += r[i + 1]
            pts.append((x / q, y / q))
        if len(pts) >= 3:
            p = Polygon(pts)
            if not p.is_valid:
                p = p.buffer(0)
            ps.append(p)
    return unary_union(ps)


def laea(lon, lat):
    return CRS.from_proj4(f"+proj=laea +lat_0={lat:.4f} +lon_0={lon:.4f} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def project(geom, crs):
    t = Transformer.from_crs(4326, crs, always_xy=True).transform
    return transform(t, geom)


def cos_area(g):
    return g.area * math.cos(math.radians(g.centroid.y))


def cluster(pieces):
    """Single linkage on bounding boxes in lon-scaled degrees."""
    n = len(pieces)
    bb = np.array([p.bounds for p in pieces])
    lat = np.clip((bb[:, 1] + bb[:, 3]) / 2, -80, 80)
    sc = np.cos(np.radians(lat))
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        s = (sc[i] + sc[i:]) / 2
        dx = np.maximum(0, np.maximum(bb[i, 0] - bb[i:, 2], bb[i:, 0] - bb[i, 2]))
        dx = np.minimum(dx, 360 - dx) * s
        dy = np.maximum(0, np.maximum(bb[i, 1] - bb[i:, 3], bb[i:, 1] - bb[i, 3]))
        for j in np.nonzero(np.hypot(dx, dy) < LINK_DEG)[0] + i:
            parent[find(j)] = find(i)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def circ_mean_lon(lons, w):
    a = np.radians(lons)
    return math.degrees(math.atan2(np.sum(w * np.sin(a)), np.sum(w * np.cos(a))))


def frame_for(geoms, aspect, pad=0.10, min_m=30000):
    """Equal-area crs centred on the geoms, and an (x0,y0,x1,y1) frame with the axes' aspect."""
    cs = [g.representative_point() for g in geoms]
    w = np.array([cos_area(g) for g in geoms])
    lon = circ_mean_lon(np.array([c.x for c in cs]), w)
    lat = float(np.average([c.y for c in cs], weights=w))
    crs = laea(lon, lat)
    pg = [project(g, crs) for g in geoms]
    b = np.array([g.bounds for g in pg])  # bounds only: a union fails on invalid outlines
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = max(x1 - x0, min_m), max(y1 - y0, min_m)
    bw, bh = bw * (1 + 2 * pad), bh * (1 + 2 * pad)
    if bw / bh < aspect:
        bw = bh * aspect
    else:
        bh = bw / aspect
    return crs, (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)


def frame_lonlat(crs, fr):
    """The frame as a lon/lat polygon (edges densified), for choosing what to draw."""
    x0, y0, x1, y1 = fr
    xs = np.linspace(x0, x1, 40)
    ys = np.linspace(y0, y1, 40)
    ring = [(x, y0) for x in xs] + [(x1, y) for y in ys] + [(x, y1) for x in xs[::-1]] + [(x0, y) for y in ys[::-1]]
    t = Transformer.from_crs(crs, 4326, always_xy=True).transform
    pts = [t(x, y) for x, y in ring]
    return Polygon(pts).buffer(0)


def main():
    ap = argparse.ArgumentParser(description="image of the zones a release adds")
    ap.add_argument("--version", required=True, help="e.g. v0.4.0 (title and file name)")
    ap.add_argument("--before", help="outlines.json path or URL (default: the live game's)")
    ap.add_argument("--after", default=os.path.join(TW, "data", "packs", "outlines.json"))
    ap.add_argument("--title", help='image title (default: "new zones in <version>")')
    ap.add_argument("--grid", action="store_true", help="2x2 equal panels")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = a.out or os.path.join(MAPS, "artifacts", "trainworld", "review", f"{a.version}-new-zones.png")

    before_src = a.before or live_data_url()
    print(f"before: {before_src}")
    bpath = cached_before(before_src)
    qb, zb = load_rings(bpath)
    qa, za = load_rings(a.after)

    new_ids = sorted(set(za) - set(zb))
    pieces, labels = [], []  # new areas: whole new zones, and what kept zones gained
    for z in new_ids:
        pieces.append(decode(qa, za[z]))
        labels.append(z)
    before_geoms = {}
    grown = []
    for z, r in za.items():
        if z in zb and (qb != qa or zb[z] != r):
            g0 = decode(qb, zb[z])
            d = decode(qa, r).difference(g0.buffer(1e-4))
            if d.area > GROWN_MIN:
                pieces.append(d)
                labels.append(z)
                grown.append(z)
    print(f"{len(za)} zones now, {len(zb)} before: {len(new_ids)} new, {len(grown)} grown")
    if not pieces:
        print("no new or grown zones; nothing written")
        return

    groups = cluster(pieces)
    total = sum(cos_area(p) for p in pieces)
    clus = sorted(({"idx": g, "geoms": [pieces[i] for i in g], "area": sum(cos_area(pieces[i]) for i in g)} for g in groups),
                  key=lambda c: -c["area"])

    # layout: main left, insets stacked right
    def axes_aspect(pos):
        return pos[2] * W / (pos[3] * H)

    cty = gpd.read_file("zip://" + os.path.join(RAW, "ne_10m_admin_0_countries.zip"))
    parts = country_parts(cty)
    land = gpd.read_file("zip://" + os.path.join(RAW, "ne_10m_land.zip"))

    def pos_main(n_ins):
        return [0.015, 0.02, 0.66, 0.86] if n_ins else [0.015, 0.02, 0.97, 0.86]

    main = clus[0]
    mpos_wide = pos_main(1)
    main["crs"], main["frame"] = frame_for(main["geoms"], axes_aspect(mpos_wide))
    main["poly"] = frame_lonlat(main["crs"], main["frame"])
    insets = []
    for c in clus[1:]:
        if len(insets) == 3 or c["area"] < INSET_MIN_SHARE * total:
            continue
        if all(main["poly"].intersects(g) for g in c["geoms"]):
            continue
        insets.append(c)
    n = len(insets)
    mpos = pos_main(n)
    if n == 0:
        main["crs"], main["frame"] = frame_for(main["geoms"], axes_aspect(mpos))
        main["poly"] = frame_lonlat(main["crs"], main["frame"])
    gap = 0.02
    ih = (0.96 - gap * (n - 1)) / max(n, 1)
    for k, c in enumerate(insets):
        c["pos"] = [0.69, 0.98 - (k + 1) * ih - k * gap, 0.295, ih]
        c["crs"], c["frame"] = frame_for(c["geoms"], axes_aspect(c["pos"]), pad=0.12)
        c["poly"] = frame_lonlat(c["crs"], c["frame"])
        c["name"] = name_for(c, parts)
    main["pos"] = mpos
    if a.grid:
        insets = [c for c in clus[1:4] if c["area"] >= INSET_MIN_SHARE * total]
        gw, gh, x0g, y0g = 0.0075, 0.02, 0.015, 0.02
        pw, ph = (0.97 - gw) / 2, (0.86 - gh) / 2
        panels = [main] + insets
        for k, c in enumerate(panels):
            col, row = k % 2, k // 2
            c["pos"] = [x0g + col * (pw + gw), y0g + (1 - row) * (ph + gh), pw, ph]
            c["crs"], c["frame"] = frame_for(c["geoms"], axes_aspect(c["pos"]), pad=0.12)
            c["poly"] = frame_lonlat(c["crs"], c["frame"])
            c["name"] = name_for(c, parts)

    shown = {id(main)} | {id(c) for c in insets}
    for c in clus:
        if id(c) not in shown:
            print(f"left out (small): {sum(1 for _ in c['idx'])} pieces, {100 * c['area'] / total:.1f}% of new area, "
                  f"near {name_for(c, parts)}")

    fontfile = next((f for f in FONTS if os.path.exists(f)), None)
    if fontfile:
        fm.fontManager.addfont(fontfile)
        plt.rcParams["font.family"] = "Nunito"
    else:
        print("Nunito not found; default font")

    new_set = set(new_ids)
    zones_after = {}

    def zone_geoms(poly):
        if not zones_after:
            for z, r in za.items():
                zones_after[z] = decode(qa, r)
            for z, g in zones_after.items():
                if g.is_empty or not g.is_valid:
                    print(f"cannot draw zone {z}: empty or invalid outline")
        return {z: g for z, g in zones_after.items() if g.intersects(poly)}

    fig = plt.figure(figsize=(W / 200, H / 200), dpi=200, facecolor="white")
    for spec, lw in [(main, 0.35)] + [(c, 0.45) for c in insets]:
        ax = fig.add_axes(spec["pos"])
        ax.set_facecolor(BG)
        crs, poly = spec["crs"], spec["poly"]
        gl = land[land.intersects(poly)].geometry.intersection(poly)
        gpd.GeoSeries(gl.values, crs=4326).to_crs(crs).plot(ax=ax, color=LAND, lw=0)
        cb = cty[cty.intersects(poly)].geometry.boundary.intersection(poly)
        gpd.GeoSeries(cb.values, crs=4326).to_crs(crs).plot(ax=ax, color="#d6d1c4", lw=0.4)
        zs = zone_geoms(poly)
        old = [g for k, g in zs.items() if k not in new_set]
        if old:
            gpd.GeoSeries(old, crs=4326).to_crs(crs).plot(ax=ax, color=MUTED, edgecolor=EDGE, lw=lw)
        # new zones and grown bits: fill wins; edges between two new ones are a thin darker red, the outline
        # against old zones stays white
        nw = [g for k, g in zs.items() if k in new_set]
        nw += [pieces[i] for i, lab in enumerate(labels) if lab in grown and pieces[i].intersects(poly)]
        if nw:
            pn = gpd.GeoSeries(nw, crs=4326).to_crs(crs)
            pn.plot(ax=ax, color=NEW, edgecolor=NEW_EDGE, lw=0.2)
            pu = unary_union([g.buffer(0) for g in pn.values])
            gpd.GeoSeries([pu]).boundary.plot(ax=ax, color=EDGE, lw=lw)
        x0, y0, x1, y1 = spec["frame"]
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#9fb2c3")
            s.set_linewidth(0.8)
        if "name" in spec:
            ax.text(0.03, 0.04, spec["name"], transform=ax.transAxes, fontsize=9, color=INK, ha="left", va="bottom",
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=2))
    fig.text(0.02, 0.925, a.title or f"new zones in {a.version}", fontsize=24, color=INK, ha="left", va="center")
    for x, c, l in [(0.37, NEW, "new"), (0.44, MUTED, "already in the game")]:
        fig.patches.append(Rectangle((x, 0.913), 0.012, 0.024, transform=fig.transFigure, fc=c, ec="none"))
        fig.text(x + 0.017, 0.925, l, fontsize=12, color=INK, ha="left", va="center")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"main map: {name_for(main, parts)} ({len(main['idx'])} pieces)")
    for c in insets:
        print(f"inset: {c['name']} ({len(c['idx'])} pieces)")
    print(out)


COMMON = {"Lao PDR": "Laos", "Russian Federation": "Russia", "Republic of Korea": "South Korea",
          "Dem. Rep. Korea": "North Korea", "Brunei Darussalam": "Brunei", "Syrian Arab Republic": "Syria",
          "Viet Nam": "Vietnam", "Kyrgyz Republic": "Kyrgyzstan", "Slovak Republic": "Slovakia",
          "Republic of the Congo": "Congo", "Democratic Republic of the Congo": "DR Congo", "Cabo Verde": "Cape Verde"}
OVERSEAS = {"GF": "French Guiana", "MQ": "Martinique", "GP": "Guadeloupe", "RE": "Reunion", "YT": "Mayotte"}


def country_parts(cty):
    """One row per country part, with the common name (NAME_LONG: Dominican Republic). A part cut off from its
    country inside a multipolygon (French Guiana in France) is named from the populated places inside it:
    their own country row by ISO code, else OVERSEAS, else their region name."""
    pp = gpd.read_file("zip://" + os.path.join(RAW, "ne_10m_populated_places_simple.zip"))
    by_iso = dict(zip(cty["ISO_A2"], cty["NAME_LONG"]))
    ex = cty[["NAME", "NAME_LONG", "geometry"]].explode(index_parts=False).reset_index(drop=True)
    ex["name"] = ex["NAME_LONG"].replace(COMMON)
    ex["a"] = ex.geometry.area
    big = ex.groupby("NAME")["a"].transform("max")
    far = ex[ex["a"] < big]
    hit = gpd.sjoin(pp[["iso_a2", "adm1name", "geometry"]], far[["geometry"]], predicate="within")
    for idx, grp in hit.groupby("index_right"):
        iso = grp["iso_a2"].value_counts().index[0]
        nm = by_iso.get(iso) or OVERSEAS.get(iso) or (grp["adm1name"].dropna().iloc[0] if grp["adm1name"].notna().any() else None)
        if nm and nm != ex.at[idx, "NAME_LONG"]:
            ex.at[idx, "name"] = COMMON.get(nm, nm)
    return ex[["name", "geometry"]]


def name_for(c, parts):
    """Countries holding the new area of a cluster, largest first, up to 4, lowercase."""
    share = {}
    sx = parts.sindex
    for g in c["geoms"]:
        for i in sx.query(g, predicate="intersects"):
            row = parts.iloc[i]
            ar = cos_area(g.intersection(row.geometry))
            if ar > 0:
                share[row["name"]] = share.get(row["name"], 0) + ar
    if not share:
        return "new zones"
    tot = sum(share.values())
    names = [k.lower() for k, v in sorted(share.items(), key=lambda kv: -kv[1]) if v >= 0.03 * tot][:4]
    return ", ".join(names)


if __name__ == "__main__":
    main()
