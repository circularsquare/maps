"""The pack gate (T-538, stage 6 of notes/T-535.md): run on a finished pack, it decides ship or park
by numeric thresholds, so a city is built with no agent judgement.

    python checks/pack_gate.py <id> [<id> ...] [--all] [--browser] [--no-rust] [--log <id>=<build log>]
                               [--since <snapshot.json>] [--out <dir>] [--work <dir of joint reports>]
    python checks/pack_gate.py snapshot <snapshot.json>   every pack's arrays hashed, before a build
    python checks/pack_gate.py fleet                      checks/pack_gate_fleet.json from the results in --out
    python checks/pack_gate.py stations <id> --extracts a,b   the pack's station file, from the extracts' caches

From the maps root, with ..\\venv's Python. Each check prints PASS, WARN, FAIL or SKIP and its figure; a
pack ships when nothing FAILs and parks otherwise. One summary line a pack, and the result as JSON
in <out>/<id>.json (default ../artifacts/trainworld/pack_gate/), which the runner (T-540) reads.
`--browser` runs checks/city-browser.mjs on a row made here (BASE: the Vite to use, default
http://localhost:6125); with several ids, the packs' rows split over BROWSER_SESSIONS (3) browser
sessions at once (T-928, T-965), after all the figures. Thresholds are THRESHOLDS below; how they were set, notes/T-538.md.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import contextlib
import datetime
import glob
import hashlib
import io
import json
import math
import re
import subprocess
import sys
import time
import urllib.parse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
PACKS = os.path.join(ROOT, "data", "packs")
WORK = os.path.join(ROOT, "data", "work")
REGISTRY = os.path.join(ROOT, "app", "src", "game", "registry.json")
PLAN = os.path.join(ROOT, "pipeline", "world_plan.json")
FUA = os.path.join(ROOT, "data", "raw", "worldplan", "GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg")
FLEET = os.path.join(HERE, "pack_gate_fleet.json")
OUT = os.path.join(os.path.dirname(ROOT), "artifacts", "trainworld", "pack_gate")
R_EARTH = 6371008.8

# Every threshold the gate uses. `fail`/`warn`: (low, high), None for an open side; a figure outside
# `fail` FAILs, outside `warn` WARNs. Set on the 67 packs of 2026-10-16 (notes/T-538.md). A check
# that compares a pack with another source than its own only WARNs (the `_other` variants): a
# census pack against GHSL, a TIGER mask against OSM water, jobs from a table against the 1% rule.
THRESHOLDS = {
    # structural
    "water_points_in_water": {"fail": (0.95, None)},       # OSM water by polylabel, in an OSM mask
    "water_points_in_water_other": {"warn": (0.95, None)},  # ... in a TIGER or StatCan mask
    "stations_on_land": {"fail": (0.95, None)},            # OSM railway=station|halt, outside the mask
    "school_rate": {"fail": (0.06, 0.35)},                 # every_pack_everyday, outings a resident
    "university_rate": {"fail": (0.001, 0.08)},
    "everyday_purposes": {"fail": (6, 6)},
    # plausibility
    "pop_vs_plan": {"fail": (0.7, 1.5), "warn": (0.85, 1.2)},    # pack people / the plan row's GHSL people
    "pop_vs_fua": {"fail": (0.7, 1.5), "warn": (0.8, 1.3)},      # pack people / its FUAs' GHSL 2015 people
    "pop_vs_ghsl": {"fail": (0.7, 1.4), "warn": (0.85, 1.2)},    # GHSL-built pack / GHSL 2025 in its outline
    "pop_vs_ghsl_other": {"warn": (0.7, 1.4)},                   # census-built pack / the same
    "near_main_piece_pop_share": {"fail": (0.95, None)},         # a city's people within NEAR_KM of its main piece
    "area_km2": {"warn": (None, 40000)},                         # big zones are fine (Anita, T-903); flags a runaway outline
    "jobs_per_resident": {"fail": (0.25, 0.7)},
    "top_cell_jobs": {"fail": (None, 50000)},                    # New York's densest cell 48,900, London's 44,200
    # tier 0 proxy jobs: the densest cell's jobs over the square root of the pack's jobs in thousands.
    # Real packs of 38 thousand to 18 million jobs run 100-575 (Birmingham 575, Liverpool 514, Belfast and
    # Cardiff 283 and 290); a proxy with thin OSM and GHSL signals reached 850-2,600 (T-583, T-611).
    "top_cell_jobs_vs_size": {"fail": (None, 600)},
    "top_cell_jobs_share_other": {"warn": (None, 0.03)},         # jobs from a table: one employer can be 5%
    "top_zone_jobs_share": {"fail": (None, 0.2), "warn": (None, 0.1)},
    "mean_commute_km": {"fail": (3, 25)},
    "top_school_cell_pupils": {"fail": (None, 20000), "warn": (None, 5000)},
    "largest_hospital_beds": {"fail": (None, 5000)},
    "drive_factor_median": {"fail": (1.1, 2.0)},
    "road_snap_median_m": {"warn": (None, 300)},
    "osm_land_share": {"warn": (None, 0.8)},
    "osm_top_place_jobs_share": {"fail": (None, 0.01)},          # tier 0 proxy jobs only
}
NEAR_KM = 20.0               # pieces of a city's outline this near its main one count with it
FLEET_WIDEN = 1.5             # a figure outside the fleet's [min / 1.5, max x 1.5] WARNs
WATER_MIN_RADIUS_M = 40.0     # a water test point is the polylabel of a polygon at least 80 m across there
WATER_MAX_POINTS = 40         # the largest water polygons in the outline, one point each
STATION_KM = 15.0             # the test line takes stations this near the centre
LINE_GAP_M = 600.0            # ... each the next station on, this far
LINE_STEP_M = 4000.0          # ... to this far
LINE_TURN_DEG = 40.0          # ... turning no more than this
LINE_MAX = 14
LINE_KM = 24.0                # ... and the line no longer than this (a new game's money)
LINE_MIN = 5                 # fewer: a made-up line of 9 stops 1.2 km apart on the most populous heading
REACH_M = 1500.0              # a purpose is required of the test line when the pack has sites this near a station (T-615)
REACH_ARRAYS = {"shopping": "errand_jobs", "school": "school_enrollment", "medical": "medical_visits"}
GRAVITY_ONLY = {"zone_row", "zone_col", "zone_home", "zone_work"} | {f"everyday_kept_{p}" for p in ("errands", "medical", "school", "social", "university")}  # what a group re-solve may change (T-1175's cross_everyday writes the kept shares)


# ---------------------------------------------------------------- reading a pack

def registry():
    with open(REGISTRY, encoding="utf-8") as f:
        return {c["id"]: c for c in json.load(f)["cities"]}


def load(city):
    import read_pack
    return read_pack.load_pack(city, PACKS)


def lonlat(header, x, y):
    lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
    k = math.cos(math.radians(lat0))
    return lon0 + np.degrees(np.asarray(x, float) / (R_EARTH * k)), lat0 + np.degrees(np.asarray(y, float) / R_EARTH)


def to_xy(header, lon, lat):
    lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
    k = math.cos(math.radians(lat0))
    return R_EARTH * np.radians(np.asarray(lon, float) - lon0) * k, R_EARTH * np.radians(np.asarray(lat, float) - lat0)


def boundary(city):
    import shapely
    from shapely.geometry import shape
    with open(os.path.join(PACKS, f"{city}.boundary.geojson"), encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    return shapely.make_valid(shapely.union_all([shapely.make_valid(shape(f["geometry"])) for f in feats]))


class Mask:
    """The pack's water mask (row runs, notes/T-004.md; row 0 south, outside the grid is land)."""
    def __init__(self, header, a):
        w = header["water"]
        self.W, self.H = w["size"]
        self.x0, self.y0 = w["origin_m"]
        self.cell = w["cell_m"]
        self.off, self.xs = a["water_row"].astype(np.int64), a["water_x"].astype(np.int64)

    def runs(self, r):
        """Row r's water runs as (start, end) pixel pairs, end exclusive."""
        seg = self.xs[self.off[r]:self.off[r + 1]]
        return list(zip(seg[0::2].tolist(), seg[1::2].tolist()))

    def at(self, x, y):
        c = np.floor((np.atleast_1d(np.asarray(x, float)) - self.x0) / self.cell).astype(np.int64)
        r = np.floor((np.atleast_1d(np.asarray(y, float)) - self.y0) / self.cell).astype(np.int64)
        out = np.zeros(c.shape, bool)
        for i in np.flatnonzero((c >= 0) & (c < self.W) & (r >= 0) & (r < self.H)):
            seg = self.xs[self.off[r[i]]:self.off[r[i] + 1]]
            out[i] = int(np.searchsorted(seg, c[i], side="right")) % 2 == 1
        return out


def to_moll(g):
    import shapely
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)
    return shapely.transform(g, lambda p: np.column_stack(t.transform(p[:, 0], p[:, 1])))


# ---------------------------------------------------------------- figures

def fig_read_pack(city):
    import read_pack
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            read_pack.check(city)
        return {"ok": True, "detail": buf.getvalue().strip().splitlines()[0]}
    except Exception as e:  # AssertionError, a missing array: the pack is broken either way
        return {"ok": False, "detail": f"{type(e).__name__}: {e}"}


def water_polygons(header, outline):
    """OSM inland water polygons inside the outline, from the extracts' water caches, or None
    when no cache is on disk (deleted after a batch, docs/adding-a-city.md 11)."""
    import pyogrio
    import shapely
    import osm_water
    exs = (header.get("water") or {}).get("extracts") or (header.get("routes") or {}).get("extracts") or []
    have = [e for e in exs if os.path.exists(osm_water.water_path(e))]
    if not have:
        return None, exs
    shapely.prepare(outline)
    geoms = []
    for e in have:
        df = pyogrio.read_dataframe(osm_water.water_path(e), bbox=tuple(outline.bounds), columns=[])
        for g in df.geometry.to_numpy():
            if g is None:
                continue
            g = shapely.make_valid(g)  # an invalid water polygon fails even `intersects` (T-897: Donghai)
            if not outline.intersects(g):
                continue
            g = g.intersection(outline)
            geoms.extend(p for p in shapely.get_parts(g) if p.geom_type == "Polygon")
    return geoms, have


def water_points(header, mask, outline):
    """T-424: one point a polygon, its polylabel in pack metres, where the polygon is at least
    2 x WATER_MIN_RADIUS_M across; the share of them the mask calls water."""
    import shapely
    from shapely.ops import polylabel
    polys, exs = water_polygons(header, outline)
    if polys is None:
        return {"skip": f"no OSM water cache for {', '.join(exs) or 'its extracts'}"}
    lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
    k = math.cos(math.radians(lat0))

    def local(p):
        return shapely.transform(p, lambda c: np.column_stack([R_EARTH * np.radians(c[:, 0] - lon0) * k,
                                                               R_EARTH * np.radians(c[:, 1] - lat0)]))
    loc = sorted((local(p) for p in polys), key=lambda p: -p.area)
    pts = []
    for p in loc:
        if len(pts) >= WATER_MAX_POINTS or p.area < (2 * WATER_MIN_RADIUS_M) ** 2:
            break
        q = polylabel(p, tolerance=5.0)
        r = p.exterior.distance(q) if not p.interiors else p.boundary.distance(q)
        if r >= WATER_MIN_RADIUS_M:
            pts.append((q.x, q.y, r))
    if len(pts) < 3:
        return {"skip": f"{len(pts)} water polygons {2 * WATER_MIN_RADIUS_M:.0f} m across in the outline"}
    xy = np.array(pts)
    wet = mask.at(xy[:, 0], xy[:, 1])
    lo, la = lonlat(header, xy[~wet, 0], xy[~wet, 1])
    return {"value": float(wet.mean()), "n": len(pts), "extracts": exs, "points": xy,
            "dry": [[round(float(a), 5), round(float(b), 5), round(float(r))] for a, b, r in zip(lo, la, xy[~wet, 2])]}


def stations_path(city):
    return os.path.join(WORK, f"{city}.stations.json")


def write_stations(city, extracts, outline=None):
    """The pack's OSM railway=station|halt nodes in its outline, (name, lon, lat), from its extracts'
    station caches (osm_features), to data/work/<city>.stations.json: written by the runner's build
    step while the caches are on disk, since they are deleted after the region and a joint re-gate
    comes later (T-770). Raises when an extract's cache is missing."""
    import pyogrio
    import shapely
    import osm_features
    import routes
    outline = outline if outline is not None else boundary(city)
    missing = [e for e in extracts if not os.path.exists(osm_features.stations_path(e))]
    if missing:
        raise SystemExit(f"{city}: no station cache for {', '.join(missing)} (osm_features.build)")
    seen, rows = set(), []
    for e in extracts:
        df = pyogrio.read_dataframe(osm_features.stations_path(e), bbox=tuple(outline.bounds), columns=["osm_id", "name", "other_tags"])
        for oid, name, other, g in zip(df.osm_id, df.name, df.other_tags, df.geometry.to_numpy()):
            if oid in seen:  # Geofabrik regions overlap along their borders
                continue
            seen.add(oid)
            rows.append([routes.tag(other, "name:en") or name or "", round(g.x, 7), round(g.y, 7)])
    if rows:
        shapely.prepare(outline)
        inside = shapely.contains_xy(outline, [r[1] for r in rows], [r[2] for r in rows])
        rows = [r for r, i in zip(rows, inside) if i]
    doc = {"city": city, "extracts": list(extracts), "source": "OpenStreetMap contributors, ODbL (Geofabrik " +
           osm_features.label(list(extracts)) + ")", "stations": rows}
    tmp = stations_path(city) + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
    os.replace(tmp, stations_path(city))
    print(f"{city}: {len(rows)} stations in the outline -> {stations_path(city)}", flush=True)
    return [tuple(r) for r in rows]


def stations(city, outline, header):
    """OSM railway=station|halt nodes in the outline: (name, lon, lat). From the pack's station file;
    else made from its extracts' caches if on disk; else the Overpass answer cached for packs gated
    before T-770 (data/raw/stations/). Never the network: SystemExit when none is on disk."""
    import shapely
    import osm_features
    p = stations_path(city)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return [tuple(r) for r in json.load(f)["stations"]]
    exs = (header.get("routes") or {}).get("extracts") or (header.get("water") or {}).get("extracts") or []
    if exs and all(os.path.exists(osm_features.stations_path(e)) for e in exs):
        return write_stations(city, exs, outline)
    old = os.path.join(ROOT, "data", "raw", "stations", f"osm_stations_{city}.json")
    if not os.path.exists(old):
        raise SystemExit(f"no station file ({os.path.basename(p)}) and no station cache for {', '.join(exs) or 'its extracts'}")
    with open(old, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for el in data.get("elements", []):
        t = el.get("tags", {})
        rows.append((t.get("name:en") or t.get("name") or "", el["lon"], el["lat"]))
    if not rows:
        return rows
    shapely.prepare(outline)
    inside = shapely.contains_xy(outline, [r[1] for r in rows], [r[2] for r in rows])
    return [r for r, i in zip(rows, inside) if i]


def outline_figures(header, a, outline):
    import shapely
    lon, lat = lonlat(header, a["x_m"], a["y_m"])
    pop = a["pop"].astype(np.float64)
    moll = to_moll(outline)
    parts = list(shapely.get_parts(moll))
    mx = to_moll(shapely.points(lon, lat))
    people = []
    for p in parts:
        shapely.prepare(p)
        people.append(float(pop[shapely.contains(p, mx)].sum()))
    main = parts[int(np.argmax(people))]
    near = sum(n for p, n in zip(parts, people) if p.distance(main) <= NEAR_KM * 1000)
    return {"near_main_piece_pop_share": near / max(pop.sum(), 1.0), "pieces": len(parts), "area_km2": moll.area / 1e6,
            "moll": moll}


def ghsl_inside(moll):
    import world_plan
    _, _, p = world_plan.Pop().inside(moll)
    return float(p.sum())


def fua_people(moll, plan_row):
    """GHS-FUA 2015 people of the plan row's FUAs, else of every FUA whose representative point is
    inside the outline."""
    import pyogrio
    import shapely
    if not os.path.exists(FUA):
        return None, []
    df = pyogrio.read_dataframe(FUA, bbox=tuple(moll.bounds), columns=["eFUA_ID", "eFUA_name", "FUA_p_2015"])
    if plan_row and plan_row.get("source", {}).get("fua"):
        want = set(plan_row["source"]["fua"])
        df = df[df.eFUA_ID.astype(int).isin(want)]
    else:
        shapely.prepare(moll)
        rp = shapely.point_on_surface(df.geometry.to_numpy())
        df = df[shapely.contains(moll, rp)]
    return float(df.FUA_p_2015.sum()), [str(x) for x in df.eFUA_name]


def plan_row(city, header):
    if not os.path.exists(PLAN):
        return None
    with open(PLAN, encoding="utf-8") as f:
        rows = json.load(f).get("metros", [])
    lo, la = header["origin"]["lon"], header["origin"]["lat"]
    for r in rows:
        w, s, e, n = r.get("bbox", (0, 0, 0, 0))
        if r.get("id") == city and w - 0.5 <= lo <= e + 0.5 and s - 0.5 <= la <= n + 0.5:
            return r
    return None


def hospital_beds(city, header):
    p = os.path.join(WORK, f"{city}.medical.audit.json")
    beds = (header.get("medical") or {}).get("beds")
    if os.path.exists(p) and beds:
        with open(p, encoding="utf-8") as f:
            fac = json.load(f)["facilities"]
        b = [r.get("beds_used", 0.0) for r in fac if r.get("kind") == "hospital"]
        top = max(b) if b else 0.0
        name = next((r.get("name", "") for r in fac if r.get("kind") == "hospital" and r.get("beds_used", 0.0) == top), "")
        return {"value": float(top), "detail": f"{name or 'unnamed'}; cap {beds.get('cap_beds', 0):.0f}"}
    if beds:
        return {"value": float(beds.get("cap_beds", 0.0)), "detail": "the site cap (no audit file)"}
    return {"skip": "medical sizes predate T-387's rule"}


def osm_signal(city, header, logs):
    """T-460's `osm_signal:` line: from the build log when given, else remade from the tier 0
    signals and the cached OSM rows (packs built before the line existed)."""
    line = None
    if city in logs:
        with open(logs[city], encoding="utf-8", errors="replace") as f:
            line = next((ln.strip() for ln in f if ln.startswith("osm_signal: ")), None)
    if line is None:
        npz = os.path.join(WORK, f"{city}.tier0_signals.npz")
        if not os.path.exists(npz):
            return {"skip": "not a tier 0 pack"}
        try:
            import tier0
        except Exception as e:  # another agent mid-edit
            return {"skip": f"tier0.py does not import ({type(e).__name__})"}
        src = next((s["name"] for s in header.get("sources", []) if s.get("what") == "jobs"), "")
        m = re.search(r"osm (\d+)%", src)
        blend = dict(tier0.JOB_BLEND, **({"osm": int(m.group(1)) / 100} if m else {}))
        sig = dict(np.load(npz))
        line = tier0.osm_signal_line(city, sig, np.ones(len(sig["h3"])), blend)
    f = dict(kv.split("=", 1) for kv in line.split(" ")[1:])
    return {"land_share": float(f["land_share"]), "top_place_jobs_share": float(f["top_place_jobs_share"]),
            "top_place": f["top_place"], "top_place_name": urllib.parse.unquote(f.get("top_place_name", "-"))}


def figures(city, reg, logs):
    """Every figure of one pack, or a reason it could not be read."""
    import shapely
    out = {"read_pack": fig_read_pack(city), "kind": reg.get("kind", "city")}
    header, a = load(city)
    out["header"] = header
    src = {s.get("what"): s.get("name", "") for s in header.get("sources", [])}
    out["proxy_jobs"] = src.get("jobs", "").startswith("tier 0 proxy")
    out["ghsl_pop"] = src.get("population", "").startswith("GHSL")
    out["osm_mask"] = "OpenStreetMap" in (header.get("water") or {}).get("source", "")
    pop = a["pop"].astype(np.float64)
    jobs = a["jobs"].astype(np.float64)
    zone = a["zone"].astype(np.int64)
    out["jobs_per_resident"] = jobs.sum() / max(pop.sum(), 1.0)
    out["top_cell_jobs"] = float(jobs.max())
    out["top_cell_jobs_vs_size"] = float(jobs.max() / math.sqrt(max(jobs.sum(), 1.0) / 1000.0))
    out["top_cell_jobs_share"] = jobs.max() / max(jobs.sum(), 1.0)
    ic = int(np.argmax(jobs))
    out["top_cell_at"] = [round(float(v), 4) for v in lonlat(header, a["x_m"][ic], a["y_m"][ic])]
    zj = np.bincount(zone, weights=jobs)
    out["top_zone_jobs_share"] = zj.max() / max(jobs.sum(), 1.0)
    iz = int(np.argmax(zj))
    gx, gy, zm = int(a["zone_gx"][iz]), int(a["zone_gy"][iz]), header["gravity"]["zone_m"]
    x0, y0 = header["gravity"]["grid_origin_m"]
    out["top_zone_at"] = [round(float(v), 4) for v in lonlat(header, x0 + (gx + 0.5) * zm, y0 + (gy + 0.5) * zm)]
    out["mean_commute_km"] = header["gravity"].get("mean_trip_km")
    if "school_enrollment" in a:
        out["top_school_cell_pupils"] = float(a["school_enrollment"].max())
    out["university_enrollment"] = float(a["university_enrollment"].astype(np.float64).sum()) if "university_enrollment" in a else 0.0
    if "drive_near" in a:
        q = a["drive_near"].astype(np.float64)
        q = q[q < 255]
        out["drive_factor_median"] = float(np.median(np.exp(q / 64.0))) if len(q) else None
        out["road_snap_median_m"] = header["routes"]["drive"].get("snap_median_m")
    out["hospital"] = hospital_beds(city, header)
    out["osm"] = osm_signal(city, header, logs)
    outline = boundary(city)
    of = outline_figures(header, a, outline)
    out.update({k: v for k, v in of.items() if k != "moll"})
    out["ghsl"] = ghsl_inside(of["moll"])
    out["pop"] = pop.sum()
    row = plan_row(city, header)
    out["plan_people"] = row.get("people") if row else None
    out["gate_relax"] = (row or {}).get("gate_relax") or {}
    if reg.get("kind") != "zone":
        out["fua"] = fua_people(of["moll"], row)
    mask = Mask(header, a) if header.get("water") else None
    out["mask"] = mask
    out["outline"] = outline
    if mask is not None:
        out["water"] = water_points(header, mask, outline)
        try:
            st = stations(city, outline, header)
        except SystemExit as e:  # no station file or cache on disk: the check skips
            st, out["stations_error"] = [], str(e)
        out["stations"] = st
        if st:
            sx, sy = to_xy(header, [s[1] for s in st], [s[2] for s in st])
            wet = mask.at(sx, sy)
            out["stations_on_land"] = {"value": float(1 - wet.mean()), "n": len(st),
                                       "wet": [list(s) for s, w in zip(st, wet) if w][:20]}
    return out, header, a


# ---------------------------------------------------------------- judging

def judge(name, value, detail=None):
    """PASS/WARN/FAIL of a figure against THRESHOLDS[name]."""
    t = THRESHOLDS[name]
    res = {"name": name, "value": value, "detail": detail}
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        res["status"] = "SKIP"
        res["detail"] = detail or "no figure"
        return res

    def outside(rng):
        lo, hi = rng
        return (lo is not None and value < lo) or (hi is not None and value > hi)
    if "fail" in t and outside(t["fail"]):
        res["status"], res["bound"] = "FAIL", t["fail"]
    elif "warn" in t and outside(t["warn"]):
        res["status"], res["bound"] = "WARN", t["warn"]
    else:
        res["status"] = "PASS"
    return res


def skip(name, why):
    return {"name": name, "status": "SKIP", "value": None, "detail": why}


def joint_borders(city, header, work=None):
    """T-591: after a joint solve, every border pair in the group's joint report is between the
    countries the registry gives its packs (`country`). A pack whose header lacked a
    country once read as US, a false national border to its neighbours (T-590)."""
    work = work or WORK
    group = (header.get("gravity", {}).get("joint") or {}).get("packs") or []
    if len(group) < 2:
        return skip("joint_borders", "not in a joint group")
    reports = []
    for p in sorted(glob.glob(os.path.join(work, "*joint_report.json"))):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        if set(r.get("packs", [])) == set(group):
            reports.append((os.path.basename(p), r))
    if not reports:
        return skip("joint_borders", "no joint report for " + "+".join(group))
    reg = registry()
    want = {i: reg.get(i, {}).get("country") for i in group}
    known = {v for v in want.values() if v}
    bad = []
    for name, r in reports:
        seen = set()
        for a, b, k in r.get("border_pairs", []):
            for lab in (a, b):
                cc, _, region = lab.partition("-")
                seen.add(cc)
                pack = region[5:] if region.startswith("pack ") else None
                if pack in want and want[pack] and want[pack] != cc:
                    bad.append(f"{lab} but {pack} is {want[pack]} in the registry")
                elif known and cc not in known:
                    bad.append(f"{lab}: {cc} is not a country of the group ({', '.join(sorted(known))})")
        # packs on islands share no border, so a report with no pairs has none to mislabel (T-1018, Hawaii)
        if known - seen and r.get("border_pairs"):
            bad.append(f"{name}: no border pair for {', '.join(sorted(known - seen))}")
    return {"name": "joint_borders", "status": "FAIL" if bad else "PASS", "value": len(group),
            "detail": "; ".join(dict.fromkeys(bad))[:300] or "+".join(group) + ": " + ", ".join(sorted(known))}


def checks_of(city, f, rates, browser, changed, joint=None):
    c = []
    if joint is not None:
        c.append(joint)
    rp = f["read_pack"]
    c.append({"name": "read_pack", "status": "PASS" if rp["ok"] else "FAIL", "value": None, "detail": rp["detail"]})
    w = f.get("water")
    if w is None:
        c.append(skip("water_points_in_water", "no water mask"))
    elif "skip" in w:
        c.append(skip("water_points_in_water", w["skip"]))
    else:
        c.append(judge("water_points_in_water" if f["osm_mask"] else "water_points_in_water_other", w["value"],
                       f"{w['n']} points; dry: {w['dry'][:5]}" if w["dry"] else f"{w['n']} points"))
    s = f.get("stations_on_land")
    c.append(judge("stations_on_land", s["value"], f"{s['n']} stations; in water: {s['wet'][:5]}") if s
             else skip("stations_on_land", f.get("stations_error") or "no OSM stations in the outline"))
    if rates is None:
        for n in ("school_rate", "university_rate", "everyday_purposes"):
            c.append(skip(n, "Rust check not run"))
    elif city not in rates:
        c.append({"name": "everyday_rates", "status": "FAIL", "value": None, "detail": "every_pack_everyday gave no line"})
    elif "error" in rates[city]:
        c.append({"name": "everyday_rates", "status": "FAIL", "value": None, "detail": "the everyday layer does not open: " + rates[city]["error"]})
    else:
        r = rates[city]
        c.append(judge("school_rate", r["school"]))
        if f.get("university_enrollment", 1.0) <= 0:  # T-614: no sites is a pack without the purpose (T-582), not a rate of 0
            c.append({"name": "university_rate", "status": "WARN", "value": r["university"],
                      "detail": "no university sites in the pack, the purpose is empty"})
        else:
            c.append(judge("university_rate", r["university"]))
        c.append(judge("everyday_purposes", r["purposes"]))
    if browser is None:
        c.append(skip("browser", "not asked (--browser)"))
    else:
        bad = [k["name"] for k in browser.get("checks", []) if not k["ok"]]
        ok = browser.get("pass", False)
        un = browser.get("unreached") or {}
        note = "; not required of the test line: " + ", ".join(f"{k} ({v})" for k, v in un.items()) if un else ""
        c.append({"name": "browser", "status": "PASS" if ok else "FAIL", "value": None,
                  "detail": ("8 checks" if ok else f"failed: {bad or browser.get('error', 'no result')}") + note})
    if changed is None:
        c.append(skip("other_packs_unchanged", "no snapshot (--since)"))
    else:
        c.append({"name": "other_packs_unchanged", "status": "FAIL" if changed else "PASS", "value": len(changed),
                  "detail": "; ".join(changed[:10]) or None})
    pop = f["pop"]
    c.append(judge("pop_vs_plan", pop / f["plan_people"]) if f.get("plan_people") else skip("pop_vs_plan", "no world plan row"))
    fua = f.get("fua")
    if fua is None:
        c.append(skip("pop_vs_fua", "a zone, not a metro"))
    elif not fua[0]:
        c.append(skip("pop_vs_fua", "no FUA in the outline"))
    else:
        c.append(judge("pop_vs_fua", pop / fua[0], f"{fua[0]:,.0f} in {', '.join(fua[1][:6])}"))
    c.append(judge("pop_vs_ghsl" if f["ghsl_pop"] else "pop_vs_ghsl_other", pop / f["ghsl"] if f["ghsl"] else None,
                   f"GHSL 2025 {f['ghsl']:,.0f}"))
    if f["kind"] == "zone":  # whole admin units: their islands are theirs
        c.append(skip("near_main_piece_pop_share", "a zone"))
    else:
        c.append(judge("near_main_piece_pop_share", f["near_main_piece_pop_share"], f"{f['pieces']} pieces"))
    c.append(judge("area_km2", f["area_km2"]))
    c.append(judge("jobs_per_resident", f["jobs_per_resident"]))
    c.append(judge("top_cell_jobs", f["top_cell_jobs"], f"at {f['top_cell_at']}"))
    if f["proxy_jobs"]:
        c.append(judge("top_cell_jobs_vs_size", f["top_cell_jobs_vs_size"],
                       f"{f['top_cell_jobs']:,.0f} jobs, {f['top_cell_jobs_share']:.2%}, at {f['top_cell_at']}"))
    else:
        c.append(judge("top_cell_jobs_share_other", f["top_cell_jobs_share"], f"at {f['top_cell_at']}"))
    c.append(judge("top_zone_jobs_share", f["top_zone_jobs_share"], f"at {f['top_zone_at']}"))
    c.append(judge("mean_commute_km", f["mean_commute_km"]))
    c.append(judge("top_school_cell_pupils", f.get("top_school_cell_pupils")))
    h = f["hospital"]
    c.append(skip("largest_hospital_beds", h["skip"]) if "skip" in h else judge("largest_hospital_beds", h["value"], h["detail"]))
    if "drive_factor_median" in f:
        c.append(judge("drive_factor_median", f["drive_factor_median"]))
        c.append(judge("road_snap_median_m", f["road_snap_median_m"]))
    else:
        c.append(skip("drive_factor_median", "no routes"))
        c.append(skip("road_snap_median_m", "no routes"))
    o = f["osm"]
    if "skip" in o:
        c.append(skip("osm_land_share", o["skip"]))
        c.append(skip("osm_top_place_jobs_share", o["skip"]))
    else:
        c.append(judge("osm_land_share", o["land_share"]))
        if f["proxy_jobs"]:
            c.append(judge("osm_top_place_jobs_share", o["top_place_jobs_share"], f"{o['top_place']} {o['top_place_name']}"))
        else:  # a table's jobs are spread within its units, so one place takes at most its unit's share
            c.append(skip("osm_top_place_jobs_share", "jobs from a table, spread within its units"))
    # a plan row's `gate_relax` {check: why}: that check's FAIL counts as a WARN for this row only
    for k in c:
        why = f.get("gate_relax", {}).get(k["name"])
        if why and k["status"] == "FAIL":
            k["status"], k["detail"] = "WARN", f"{k.get('detail') or ''}; relaxed for this row: {why}"
    fleet_check(c)
    return c


def fleet_check(c):
    """One WARN listing every figure outside the fleet's range widened by FLEET_WIDEN."""
    if not os.path.exists(FLEET):
        c.append(skip("fleet_range", "no checks/pack_gate_fleet.json"))
        return
    with open(FLEET) as f:
        fleet = json.load(f)["ranges"]
    out = []
    for k in c:
        r = fleet.get(k["name"])
        v = k.get("value")
        if r is None or not isinstance(v, (int, float)) or k["status"] == "SKIP":
            continue
        if v < r[0] / FLEET_WIDEN or v > r[1] * FLEET_WIDEN:
            out.append(f"{k['name']} {v:.4g} (fleet {r[0]:.4g}-{r[1]:.4g})")
    c.append({"name": "fleet_range", "status": "WARN" if out else "PASS", "value": len(out), "detail": "; ".join(out) or None})


# ---------------------------------------------------------------- the Rust and browser checks

def everyday_rates(ids):
    """`every_pack_everyday` on these packs: {id: {school, university, ..., purposes}}, or
    {id: {"error": the panic}} for a pack whose everyday layer does not open. The test stops at
    the first pack that panics, so the packs after it run again one at a time."""
    rates = _everyday_rates(ids)
    missing = [i for i in ids if i not in rates]
    if rates and missing or len(ids) > 1 and not rates:
        for i in missing:
            rates.update(_everyday_rates([i]))
    return rates


def _everyday_rates(ids):
    env = dict(os.environ, TW_PACKS=",".join(ids), CARGO_BUILD_JOBS=os.environ.get("CARGO_BUILD_JOBS", "1"))
    env["PATH"] = os.path.join(os.path.expanduser("~"), ".cargo", "bin") + os.pathsep + env["PATH"]
    exe = os.environ.get("TW_EVERYDAY_EXE")  # T-931: the runner's test binary, built once a run
    if exe and os.path.exists(exe):
        cmd = [exe, "every_pack_everyday", "--ignored", "--nocapture"]
    else:
        cmd = ["cargo", "test", "--release", "--features", "demand", "--lib", "--manifest-path",
               # TW_SIM_DIR: a copy of sim/ while the tree's own is mid-edit (as routes.py)
               os.path.join(os.environ.get("TW_SIM_DIR", os.path.join(ROOT, "sim")), "Cargo.toml"), "every_pack_everyday", "--", "--ignored", "--nocapture"]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    rates = {}
    for ln in p.stdout.splitlines():
        if ln.startswith("gate: "):
            parts = ln.split()
            kv = dict(x.split("=", 1) for x in parts[2:])
            rates[parts[1]] = {k: (int(v) if k == "purposes" else float(v)) for k, v in kv.items()}
    m = re.search(r"panicked at [^\n]*\n([^\n]+)", p.stdout + "\n" + p.stderr)
    if len(ids) == 1 and ids[0] not in rates and m:
        rates[ids[0]] = {"error": m.group(1).strip()[:300]}
    if not rates:
        print(p.stdout[-2000:], p.stderr[-3000:], flush=True)
    return rates


def chain(sx, sy, start, heading, used):
    """Stations from `start` onward: each step the nearest unused station LINE_GAP_M to LINE_STEP_M
    on, within LINE_TURN_DEG of the heading so far (so the chain follows a curving line)."""
    out, i, h, run = [], start, heading, 0.0
    while len(out) < LINE_MAX // 2:
        dx, dy = sx - sx[i], sy - sy[i]
        d = np.hypot(dx, dy)
        turn = np.abs((np.degrees(np.arctan2(dy, dx)) - h + 180) % 360 - 180)
        ok = (d >= LINE_GAP_M) & (d <= min(LINE_STEP_M, LINE_KM * 500 - run)) & (turn <= LINE_TURN_DEG)
        ok[list(used | set(out))] = False
        if not ok.any():
            break
        j = int(np.flatnonzero(ok)[np.argmin(d[ok])])
        h = float(np.degrees(np.arctan2(dy[j], dx[j])))
        run += float(d[j])
        out.append(j)
        i = j
    return out


def test_line(header, a, f, reg):
    """T-424: the browser check's test line from OSM stations: through the station nearest the
    centre, each way along whichever chain of stations is longest; else made up on the most
    populous heading."""
    cx, cy = to_xy(header, reg["center"][0], reg["center"][1])
    st = [s for s in f.get("stations", []) if s[0]]
    best = []
    if st:
        sx, sy = to_xy(header, [s[1] for s in st], [s[2] for s in st])
        near = (np.hypot(sx - cx, sy - cy) < STATION_KM * 1000) & ~f["mask"].at(sx, sy)
        st = [s for s, k in zip(st, near) if k]
        sx, sy = sx[near], sy[near]
    if st:
        s0 = int(np.argmin(np.hypot(sx - cx, sy - cy)))
        for deg in range(0, 180, 10):
            fwd = chain(sx, sy, s0, deg, {s0})
            back = chain(sx, sy, s0, deg + 180, {s0} | set(fwd))
            line = back[::-1] + [s0] + fwd
            if len(line) > len(best):
                best = line
    if len(best) >= LINE_MIN:
        seen, line = set(), []
        for i in best:
            nm = st[i][0]
            while nm in seen:
                nm += " 2"
            seen.add(nm)
            line.append([nm, round(st[i][1], 5), round(st[i][2], 5)])
        return line, "OSM stations"
    # made up: 9 stops 1.2 km apart on the heading with most people within 600 m, 5 km each way
    px, py, pop = a["x_m"].astype(np.float64), a["y_m"].astype(np.float64), a["pop"].astype(np.float64)
    top, heading = -1.0, 0.0
    for deg in range(0, 180, 5):
        t = math.radians(deg)
        ux, uy = math.cos(t), math.sin(t)
        along = (px - cx) * ux + (py - cy) * uy
        off = np.abs(-(px - cx) * uy + (py - cy) * ux)
        s = pop[(off < 600) & (np.abs(along) < 5000)].sum()
        if s > top:
            top, heading = s, t
    pts = [(cx + k * 1200 * math.cos(heading), cy + k * 1200 * math.sin(heading)) for k in range(-4, 5)]
    lo, la = lonlat(header, [p[0] for p in pts], [p[1] for p in pts])
    return [[f"{reg['name']} {i + 1}", round(float(x), 5), round(float(y), 5)] for i, (x, y) in enumerate(zip(lo, la))], "made up"


def water_pair(header, f, reg):
    """Two land points either side of the water run nearest the centre, from the pack's own mask:
    rows within 5 km, a run at least 200 m wide (else 100 m), land on both sides; failing that, rows
    within 30 km (a desert centre: Flagstaff, T-1018)."""
    m = f["mask"]
    cx, cy = to_xy(header, reg["center"][0], reg["center"][1])
    c0 = int((float(cx) - m.x0) / m.cell)
    r0 = int((float(cy) - m.y0) / m.cell)
    for reach_m, min_px in ((5000, 8), (5000, 4), (30000, 8), (30000, 4)):
        best = None
        for dr in range(0, int(reach_m / m.cell)):
            for r in {r0 - dr, r0 + dr}:
                if not 0 <= r < m.H:
                    continue
                for s, e in m.runs(r):  # runs never touch, so land lies either side
                    if e - s < min_px or s < 3 or e + 3 > m.W:
                        continue
                    d = math.hypot((s + e) / 2 - c0, r - r0)
                    if best is None or d < best[0]:
                        best = (d, r, s, e)
            if best is not None and dr * m.cell > max(1000.0, best[0] * m.cell):
                break
        if best is not None:
            _, r, s, e = best
            y = m.y0 + (r + 0.5) * m.cell
            xs = [m.x0 + (s - 2) * m.cell, m.x0 + (e + 2) * m.cell]
            lo, la = lonlat(header, xs, [y, y])
            return [[round(float(lo[0]), 5), round(float(la[0]), 5)], [round(float(lo[1]), 5), round(float(la[1]), 5)]]
    return None


def purposes_in_reach(header, a, line):
    """T-615: the browser check's purposes the test line can carry: work and social always, the
    others when the pack has sites of that purpose within REACH_M of a station of the line.
    Returns (required, {purpose: why it is not required})."""
    x, y = to_xy(header, [s[1] for s in line], [s[2] for s in line])
    required, unreached = ["work", "social"], {}
    for purpose, key in REACH_ARRAYS.items():
        w = a[key].astype(np.float64) if key in a else None
        if w is None or w.sum() <= 0:
            unreached[purpose] = "the pack has no sites"
            continue
        near = np.zeros(len(w), bool)
        for sx, sy in zip(x, y):
            near |= np.hypot(a["x_m"] - sx, a["y_m"] - sy) < REACH_M
        if (w[near] > 0).any():
            required.append(purpose)
        else:
            unreached[purpose] = f"no sites within {REACH_M / 1000:g} km of the line ({int((w > 0).sum())} elsewhere)"
    return required, unreached


def browser_row(city, header, a, f, reg, out_dir):
    """The browser check's row for one pack ({id, row, out, json} for city-browser.mjs's ROWS, and
    what the result adds), or (None, a failed result) when it has no water pair."""
    line, how = test_line(header, a, f, reg)
    required, unreached = purposes_in_reach(header, a, line)
    pair = water_pair(header, f, reg) if f.get("mask") is not None else None
    if pair is None:
        return None, {"pass": False, "error": "no water run near the centre for the water pair"}
    names = os.path.join(ROOT, "app", "src", "game", "zone-names.json")  # the name the game shows (T-825)
    # with no hand-set name, zoneNames.ts's cleanName: "Minneapolis [Saint Paul]" shows "Minneapolis" (T-1078)
    shown = re.sub(r"\s+", " ", re.sub(r" CC\b", "", re.sub(r"\s*\([^)]*\)\s*$", "",
                                                             re.sub(r"\s*\[[^\]]*\]", "", reg["name"])))).strip() or reg["name"]
    if os.path.exists(names):
        with open(names, encoding="utf-8") as fh:
            shown = (json.load(fh).get(city) or {}).get("name") or shown
    row = {"task": "T-538", "name": shown, "centre": reg["center"], "water": pair, "lineName": "Gate test",
           "colour": "#2a7ab0", "view": reg["center"], "line": line, "purposes": required}
    rp = os.path.join(out_dir, f"{city}.row.json")
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump(row, fh, ensure_ascii=False)
    return ({"id": city, "row": rp, "out": os.path.join(out_dir, f"{city}-"), "json": os.path.join(out_dir, f"{city}.browser.json")},
            {"line_from": how, "stations": len(line), "unreached": unreached})


BROWSER_SESSIONS = int(os.environ.get("BROWSER_SESSIONS", 3))  # T-965: a zone's check slows with each zone open in its page


def browser_run(rows, out_dir):
    """city-browser.mjs on the rows, split in plan order over BROWSER_SESSIONS sessions at once
    (T-928, T-965): {id: its result}. city-browser.mjs tries a zone whose wait timed out once more
    in a fresh page (the machine busy, T-788)."""
    maps = os.path.dirname(ROOT)
    for r in rows:
        if os.path.exists(r["json"]):
            os.remove(r["json"])
    k = max(1, min(BROWSER_SESSIONS, len(rows)))
    size = -(-len(rows) // k)
    runs = []
    for n in range(k):
        part = rows[n * size:(n + 1) * size]
        if not part:
            continue
        lst = os.path.join(out_dir, f"browser-rows-{os.getpid()}-{n}.json")
        with open(lst, "w", encoding="utf-8") as fh:
            json.dump(part, fh, ensure_ascii=False)
        env = dict(os.environ, ROWS=lst, WORKERS=os.environ.get("WORKERS", "1"))
        log = open(os.path.join(out_dir, f"browser-{os.getpid()}-{n}.log"), "w+", encoding="utf-8", errors="replace")
        p = subprocess.Popen(["node", os.path.join(HERE, "city-browser.mjs")], cwd=maps, env=env,
                             stdout=log, stderr=subprocess.STDOUT)
        runs.append((part, lst, log, p))
    # each row's result printed as it lands, so the gate's log keeps growing for the runner's quiet watch (T-1308)
    seen, beat = set(), time.time()
    while any(p.poll() is None for *_, p in runs):
        for part, *_ in runs:
            for r in part:
                if r["id"] not in seen and os.path.exists(r["json"]):
                    seen.add(r["id"])
                    print(f"  browser {r['id']}: result in ({len(seen)} of {len(rows)})", flush=True)
        if time.time() - beat > 300:
            beat = time.time()
            print(f"  browser: {len(seen)} of {len(rows)} rows in", flush=True)
        time.sleep(5)
    out = {}
    for part, lst, log, p in runs:
        p.wait()
        log.seek(0)
        said = log.read()
        log.close()
        os.remove(lst)
        os.remove(log.name)
        for r in part:
            got = {}
            if os.path.exists(r["json"]):
                with open(r["json"], encoding="utf-8") as fh:
                    got = json.load(fh)
            checks = got.get("checks", [])
            res = {"pass": len(checks) == 8 and all(c["ok"] for c in checks), "checks": checks}
            if got.get("tries", 1) > 1:
                res["tries"] = got["tries"]
            if not res["pass"] and all(c["ok"] for c in checks):  # died after passing checks (T-788)
                res["error"] = (got.get("error") or said or "no result")[-400:]
            out[r["id"]] = res
    return out


def browser_check(city, header, a, f, reg, out_dir):
    row, extra = browser_row(city, header, a, f, reg, out_dir)
    return dict(extra, **browser_run([row], out_dir)[city]) if row else extra


# ---------------------------------------------------------------- other packs unchanged

def _unmangle(s):
    """Text as first written: packs read in the locale code page re-encoded UTF-8 once per pass
    ("LÃ¤nder", "LÃƒÂ¤nder"), which is no change to the pack (T-768)."""
    for _ in range(4):
        try:
            t = s.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return s
        if t == s:
            return s
        s = t
    return s


def _round(v):
    if isinstance(v, str):
        return _unmangle(v)
    if isinstance(v, float):
        return float(f"{v:.9g}")
    if isinstance(v, dict):
        return {k: _round(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_round(x) for x in v]
    return v


def pack_hashes(city):
    """{item: sha} for one pack: each array of its .bin, its header without the arrays' offsets
    and the gravity block, the gravity block, and its outline."""
    import packio
    out = {}
    with open(os.path.join(PACKS, f"{city}.json"), encoding="utf-8") as f:
        h = json.load(f)
    with open(os.path.join(PACKS, f"{city}.bin"), "rb") as f:
        buf = f.read()
    for arr in h["arrays"]:
        size = packio.SIZES.get(arr["dtype"], 1)
        out[arr["name"]] = hashlib.sha1(buf[arr["offset"]:arr["offset"] + arr["count"] * size]).hexdigest()
    rest = {k: v for k, v in h.items() if k not in ("arrays", "gravity")}
    # the merge's stamp of the res-9 pack it came from changes with the res-9 pack's gravity (T-540)
    stamp = (rest.get("merged") or {}).get("source") if isinstance(rest.get("merged"), dict) else None
    if stamp is not None:
        # also the merge's own wording and its walk-part count (T-725): the arrays show any change to cells
        rest["merged"] = {k: v for k, v in rest["merged"].items() if k not in ("source", "rule", "parents_split_by_part")}
    # figures to 9 significant digits: a re-merge recomputes shares in another order (T-540:
    # Rotterdam's school location_match_share moved by one bit in its last place)
    out["header"] = hashlib.sha1(json.dumps(_round(rest), sort_keys=True).encode()).hexdigest()
    out["header:merged_source"] = hashlib.sha1(json.dumps(stamp, sort_keys=True).encode()).hexdigest()
    out["header:gravity"] = hashlib.sha1(json.dumps(h.get("gravity"), sort_keys=True).encode()).hexdigest()
    with open(os.path.join(PACKS, f"{city}.boundary.geojson"), "rb") as f:
        out["boundary"] = hashlib.sha1(f.read()).hexdigest()
    return out


_hashes = {}


def pack_hashes_now(city):
    """pack_hashes, kept while the pack's files are unchanged: a gate of many packs compares every
    other pack once, not once a pack (T-928)."""
    key = tuple((os.stat(p).st_mtime_ns, os.stat(p).st_size) if os.path.exists(p) else None
                for p in (os.path.join(PACKS, f"{city}{x}") for x in (".json", ".bin", ".boundary.geojson")))
    if _hashes.get(city, (None,))[0] != key:
        _hashes[city] = (key, pack_hashes(city))
    return _hashes[city][1]


def snapshot(path):
    snap = {c: pack_hashes(c) for c in registry()}
    with open(path, "w") as f:
        json.dump(snap, f)
    print(f"snapshot of {len(snap)} packs -> {path}")


def changed_since(city, header, path):
    with open(path) as f:
        snap = json.load(f)
    group = set((header.get("gravity", {}).get("joint") or {}).get("packs", []))
    bad = []
    now = registry()
    for other, before in snap.items():
        # a pack taken out of the registry since (another run's park or rebuild, T-786) is no concern of this gate
        if other == city or other not in now:
            continue
        if not os.path.exists(os.path.join(PACKS, f"{other}.json")):
            bad.append(f"{other} gone")
            continue
        h = pack_hashes_now(other)
        allowed = GRAVITY_ONLY | {"header:gravity", "header:merged_source"} if other in group else set()
        diff = sorted(k for k in set(before) | set(h) if before.get(k) != h.get(k) and k not in allowed)
        if diff:
            bad.append(f"{other}: {', '.join(diff[:6])}")
    return bad


# ---------------------------------------------------------------- main

def clean(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, tuple):
        return list(v)
    raise TypeError(type(v))


def prepare(city, reg, logs):
    """(figures, header, arrays) of one pack, or its result when its figures cannot be read."""
    try:
        return figures(city, reg, logs)
    except Exception as e:
        import traceback
        traceback.print_exc()
        checks = [{"name": "figures", "status": "FAIL", "value": None, "detail": f"{type(e).__name__}: {e}"}]
        return {"city": city, "verdict": "park", "checks": checks}


def verdict_of(city, reg, f, header, rates, br, opts):
    changed = changed_since(city, header, opts["since"]) if opts["since"] else None
    checks = checks_of(city, f, rates, br, changed, joint_borders(city, header, opts.get("work")))
    verdict = "park" if any(c["status"] == "FAIL" for c in checks) else "ship"
    return {"city": city, "kind": reg.get("kind", "city"), "gated": datetime.datetime.now().isoformat(timespec="seconds"),
            "verdict": verdict, "checks": checks, "browser": br}


def run(city, reg, logs, rates, opts):
    got = prepare(city, reg, logs)
    if isinstance(got, dict):
        return got
    f, header, a = got
    br = browser_check(city, header, a, f, reg, opts["out"]) if opts["browser"] else None
    return verdict_of(city, reg, f, header, rates, br, opts)


def run_many(ids, reg, logs, rates, opts):
    """T-928, --browser on several packs: every pack's figures and test row, then the browser
    check for them all (browser_run), then each verdict. Yields (id, result)."""
    ready, rows, extra = {}, [], {}
    for city in ids:
        got = prepare(city, reg[city], logs)
        if isinstance(got, dict):
            ready[city] = got
            continue
        f, header, a = got
        row, extra[city] = browser_row(city, header, a, f, reg[city], opts["out"])
        if row:
            rows.append(row)
        f.pop("mask", None)  # the mask and arrays only make the row; a region's worth would stay in memory
        ready[city] = (f, header)
    t0 = datetime.datetime.now()
    seen = browser_run(rows, opts["out"]) if rows else {}
    print(f"browser: {len(rows)} packs in {min(BROWSER_SESSIONS, len(rows))} sessions, {(datetime.datetime.now() - t0).total_seconds():.0f} s", flush=True)
    for city in ids:
        got = ready[city]
        if isinstance(got, dict):
            yield city, got
            continue
        br = dict(extra[city], **seen[city]) if city in seen else extra[city]
        yield city, verdict_of(city, reg[city], got[0], got[1], rates, br, opts)


def summary(r):
    fails = [c for c in r["checks"] if c["status"] == "FAIL"]
    warns = [c for c in r["checks"] if c["status"] == "WARN"]
    skips = sum(c["status"] == "SKIP" for c in r["checks"])

    def short(c):
        v = c.get("value")
        return c["name"] + (f" {v:.4g}" if isinstance(v, (int, float)) else "")
    s = f"{r['city']} {'SHIP' if r['verdict'] == 'ship' else 'PARK'}"
    if fails:
        s += "; FAIL " + ", ".join(short(c) for c in fails)
    if warns:
        s += "; WARN " + ", ".join(short(c) for c in warns)
    return s + f"; {skips} skipped"


def fleet(out_dir):
    reg = registry()
    vals = {}
    for city in reg:
        p = os.path.join(out_dir, f"{city}.json")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        for c in r["checks"]:
            v = c.get("value")
            if c["name"] in THRESHOLDS and isinstance(v, (int, float)) and c["status"] != "SKIP":
                vals.setdefault(c["name"], []).append(v)
    ranges = {k: [min(v), max(v)] for k, v in vals.items()}
    with open(FLEET, "w") as f:
        json.dump({"task": "T-538", "built": datetime.date.today().isoformat(), "packs": len(reg),
                   "what": "each figure's min and max over the registered packs; the gate WARNs outside [min / 1.5, max x 1.5]",
                   "ranges": ranges}, f, indent=1)
    print(f"{len(ranges)} ranges -> {FLEET}")


def main(argv):
    def opt(flag, default=None):
        return argv[argv.index(flag) + 1] if flag in argv else default
    out = opt("--out", OUT)
    os.makedirs(out, exist_ok=True)
    if argv and argv[0] == "snapshot":
        return snapshot(argv[1])
    if argv and argv[0] == "fleet":
        return fleet(out)
    if argv and argv[0] == "stations":
        return write_stations(argv[1], opt("--extracts").split(","))
    reg = registry()
    valued = {i for i, x in enumerate(argv) if i and argv[i - 1] in ("--log", "--since", "--out", "--work")}
    ids = list(reg) if "--all" in argv else [x for i, x in enumerate(argv) if not x.startswith("--") and i not in valued]
    if not ids:
        raise SystemExit(__doc__)
    logs = dict(x.split("=", 1) for i, x in enumerate(argv) if i and argv[i - 1] == "--log")
    rates = None if "--no-rust" in argv else everyday_rates(ids)
    opts = {"browser": "--browser" in argv, "since": opt("--since"), "out": out, "work": opt("--work")}
    lines = []
    for city in [i for i in ids if i not in reg]:
        print(f"{city}: not in the registry", flush=True)
    ids = [i for i in ids if i in reg]
    many = opts["browser"] and len(ids) > 1
    for city, r in (run_many(ids, reg, logs, rates, opts) if many else ((c, run(c, reg[c], logs, rates, opts)) for c in ids)):
        for c in r["checks"]:
            v = c.get("value")
            vs = f"{v:.4g}" if isinstance(v, (int, float)) else ""
            print(f"  {c['status']:4} {c['name']} {vs} {c.get('detail') or ''}".rstrip(), flush=True)
        with open(os.path.join(out, f"{city}.json"), "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1, default=clean, ensure_ascii=False)
        lines.append(summary(r))
        print(lines[-1], flush=True)
    if len(lines) > 1:
        print("\n".join(["", "summary:"] + lines))


if __name__ == "__main__":
    # T-980: a hand run on a cp1252 console (`-I` ignores PYTHONIOENCODING) met non-Latin details
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main(sys.argv[1:])
