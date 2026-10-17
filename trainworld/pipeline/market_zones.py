"""T-377 (T-362 stage 3): each registered zone's part of the world market, and the registry's market rows.

    python pipeline/market_zones.py [id ...]     every registered zone, or only those named
    python pipeline/market_zones.py --air        only every zone file's airports (T-378), then the rows

Writes `data/packs/<id>.market.json` per zone (seconds each), then runs the sim crate's `market_pack`
over every zone's file, which adds up each zone's registry row and the trips between zones whose
centres are under 250 km apart, and merges them into app/src/game/registry.json (`place`,
`marketRow`, and `market.rule` / `market.places`). A new zone needs only its own file and this merge.

A zone's file holds its endpoint sites (its pack's cells in 600 m squares: the Northeast's straight
from data/packs/intercity.json, the sites the joint commute gravity names, in the market frame; every
other zone's in its own pack frame), each site's parking for an arriving car and the people of its
H3 r5 cell (the long-haul tail's density, notes/T-376.md), its 6 km zones with their two normalisers
against the world field (T-375: ground and tail, constants in pipeline/longhaul.json), and the
commutes from it to other zones (the Northeast's, from the same intercity.json; a joint gravity
group's, from data/work/<group>.joint_commutes.json, T-463), and (T-378) the
airports its people fly from with the zones each flies to directly.
"""
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
import h3

sys.path.insert(0, str(Path(__file__).parent))
import world_field as W
from read_pack import load_pack
from build_city import R_EARTH
from intercity import park_by_cell

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "app/src/game/registry.json"
PACKS = ROOT / "data/packs"
RULE = json.loads((Path(__file__).parent / "longhaul.json").read_text(encoding="utf-8"))
SITE_M, ZONE_M = 600, 6000
# T-378: a zone's airports are OurAirports' large and medium ones with scheduled service within
# AIR_RADIUS_KM of its centre carrying AIR_MIN_PAX passengers a year (the biggest AIR_MAX); a route
# of DIRECT_MIN_PAX a year is a direct flight. Passengers and routes from the flights project's
# FR24 file, read here only: its terms forbid passing the data on, so only these choices leave it
# (without it: large airports, no direct flights).
OURAIRPORTS_CSV = ROOT / "data/raw/t142b/ourairports-airports.csv"
FLIGHTS = ROOT.parent / "flights/routes_fr24.json"
AIR_RADIUS_KM, AIR_MIN_PAX, DIRECT_MIN_PAX, AIR_MAX = 150.0, 1e6, 5e4, 8
AIR_SOURCE = "airports: OurAirports (public domain), chosen and flagged direct by FR24 passengers (maps/flights, offline, T-378)"


def airports_by_zone(reg):
    """Per registered zone id: its airports' [lon, lat], and for each the places it flies to directly."""
    ports = []
    with open(OURAIRPORTS_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["scheduled_service"] == "yes" and r["type"] in ("large_airport", "medium_airport") and r["iata_code"]:
                ports.append((r["iata_code"], float(r["longitude_deg"]), float(r["latitude_deg"]), r["type"]))
    pax, routes, fr = {}, {}, {}
    if FLIGHTS.exists():
        f = json.loads(FLIGHTS.read_text(encoding="utf-8"))
        fr = {k: (v[0], v[1]) for k, v in f["airports"].items()}
        for a, b, n, *_ in f["routes"]:
            pax[a] = pax.get(a, 0) + n
            pax[b] = pax.get(b, 0) + n
            if n >= DIRECT_MIN_PAX:
                routes.setdefault(a, set()).add(b)
                routes.setdefault(b, set()).add(a)
    fr_codes = list(fr)
    fr_lon, fr_lat = np.array([fr[k][0] for k in fr_codes]), np.array([fr[k][1] for k in fr_codes])

    def fr_of(p):
        """The FR24 codes of an OurAirports airport: its own, or any within 5 km (FR24 keeps some
        old codes: Berlin's BER is SXF)."""
        if p[0] in pax or not fr_codes:
            return {p[0]}
        return {fr_codes[k] for k in np.nonzero(W.km(p[1], p[2], fr_lon, fr_lat) <= 5.0)[0]}
    lon, lat = np.array([p[1] for p in ports]), np.array([p[2] for p in ports])
    chosen = {}
    for c in reg["cities"]:
        if "place" not in c:
            continue
        d = W.km(c["center"][0], c["center"][1], lon, lat)
        near = [(ports[k], fr_of(ports[k])) for k in np.nonzero(d <= AIR_RADIUS_KM)[0]]
        size = lambda pc: sum(pax.get(x, 0) for x in pc[1])
        near = [pc for pc in near if (size(pc) >= AIR_MIN_PAX if pax else pc[0][3] == "large_airport")]
        near.sort(key=lambda pc: -size(pc))
        chosen[c["id"]] = near[:AIR_MAX]
    codes = {cid: set().union(*(pc[1] for pc in v)) for cid, v in chosen.items()}
    place = {c["id"]: c["place"] for c in reg["cities"] if "place" in c}
    out = {}
    for cid, v in chosen.items():
        flies = lambda pc: set().union(*(routes.get(x, set()) for x in pc[1]))
        out[cid] = {"airports": [[round(p[1], 5), round(p[2], 5)] for p, _ in v],
                    "air_direct": [sorted(place[o] for o, oc in codes.items() if o != cid and flies(pc) & oc) for pc in v],
                    "air_codes": [p[0] for p, _ in v]}
    return out


class Field:
    """The world field with the ground attraction and the tail's destination weight per cell, the
    tail's summed from r5 cells upward (notes/T-376.md "For stage 3")."""
    def __init__(self):
        self.wf = wf = W.load()
        self.y_us = RULE["y_us"]
        r5, r4, r3 = wf.r5, wf.r4, wf.r3
        inc = np.minimum(r5.gdp / self.y_us, 1)
        up = lambda x5: [x5, np.add.reduceat(x5, r4.start), np.add.reduceat(np.add.reduceat(x5, r4.start), r3.start)]
        self.a = up(r5.pop * inc ** RULE["eps"])
        self.tb = up(r5.pop * inc ** RULE["eps"] * (np.maximum(r5.pop, 1) / RULE["u_ref_people"]) ** RULE["TH"])
        self.inc = [np.minimum(lv.gdp / self.y_us, 1) for lv in (r5, r4, r3)]
        self.index = {int(c): i for i, c in enumerate(r5.h)}

    def normalisers(self, lon, lat, country, inc_o):
        """A place's ground and tail normalisers over the field: the sums the two kinds of
        departures are shared by (notes/T-376.md)."""
        wf = self.wf
        d3 = W.km(lon, lat, wf.r3.lon, wf.r3.lat)
        far = np.nonzero(d3 > W.MID_KM)[0]
        i4 = W._children(wf.r3, np.nonzero(d3 <= W.MID_KM)[0])
        d4 = W.km(lon, lat, wf.r4.lon[i4], wf.r4.lat[i4])
        mid, i5 = i4[d4 > W.NEAR_KM], W._children(wf.r4, i4[d4 <= W.NEAR_KM])
        d5 = W.km(lon, lat, wf.r5.lon[i5], wf.r5.lat[i5])
        parts = [(0, i5, d5), (1, mid, d4[d4 > W.NEAR_KM]), (2, far, d3[far])]
        lv = [wf.r5, wf.r4, wf.r3]
        cat = lambda arrs: np.concatenate([arrs[l][ix] for l, ix, _ in parts])
        d = np.concatenate([p[2] for p in parts])
        c = np.concatenate([lv[l].country[ix] for l, ix, _ in parts])
        a, tb, inc = cat(self.a), cat(self.tb), cat(self.inc)
        b = np.where(c == country, 1.0, RULE["border"] * inc_o ** RULE["border_eps"])
        on0, on1 = RULE["onset_km"]
        on = np.clip((d - on0) / (on1 - on0), 0, 1)
        t0, t1 = RULE["tail_onset_km"]
        ont = np.clip((d - t0) / (t1 - t0), 0, 1)
        ng = float((b * a * on * np.maximum(d, RULE["d0_km"]) ** -RULE["A"]).sum())
        nt = float((b * tb * np.minimum(inc / inc_o, 1) ** RULE["eps_t"] * ont * np.maximum(d, RULE["D1_km"]) ** -RULE["G"]).sum())
        return ng, nt

    def cell_pop(self, lon, lat):
        """People in each place's H3 r5 cell (0 where the field has none)."""
        out = np.zeros(len(lon))
        for k, (lo, la) in enumerate(zip(lon, lat)):
            i = self.index.get(h3.str_to_int(h3.latlng_to_cell(float(la), float(lo), 5)))
            out[k] = 0.0 if i is None else self.wf.r5.pop[i]
        return out


def equirect(lon0, lat0):
    k = math.cos(math.radians(lat0))
    to_xy = lambda lon, lat: (R_EARTH * np.radians(np.asarray(lon) - lon0) * k, R_EARTH * np.radians(np.asarray(lat) - lat0))
    to_ll = lambda x, y: (lon0 + np.degrees(np.asarray(x) / (R_EARTH * k)), lat0 + np.degrees(np.asarray(y) / R_EARTH))
    return to_xy, to_ll


def own_sites(cid, frame):
    """A zone's sites from its own pack ([x, y, people, jobs] in `frame`) and their parking, grouped
    as pipeline/intercity.py groups the Northeast's."""
    h, a = load_pack(cid)
    o = h["origin"]
    _, pack_ll = equirect(o["lon"], o["lat"])
    lon, lat = pack_ll(a["x_m"].astype(np.float64), a["y_m"].astype(np.float64))
    park = park_by_cell(h, a, lon, lat)
    to_xy, _ = equirect(frame["lon"], frame["lat"])
    x, y = to_xy(lon, lat)
    groups = {}
    for xx, yy, pp, jj, pk in zip(x, y, a["pop"], a["jobs"], park):
        if pp + jj <= 0:
            continue
        r = groups.setdefault((int(xx // SITE_M), int(yy // SITE_M)), [0., 0., 0., 0., 0., 0.])
        r[0] += xx * (pp + jj); r[1] += yy * (pp + jj)
        r[2] += pp; r[3] += jj
        r[4] += pk * (jj + 1e-6); r[5] += jj + 1e-6
    sites, parks = [], []
    for r in groups.values():
        w = r[2] + r[3]
        sites.append([round(float(r[0] / w), 1), round(float(r[1] / w), 1), round(float(r[2]), 3), round(float(r[3]), 3)])
        parks.append(round(float(r[4] / r[5]), 2))
    return sites, parks


class Northeast:
    """The Northeast's sites and commutes between zones as data/packs/intercity.json has them (the
    joint commute gravity names those sites), split by zone."""
    def __init__(self):
        p = json.loads((PACKS / "intercity.json").read_text(encoding="utf-8"))
        self.frame = p["frame"]
        s = p["sites"]
        self.market = np.array([int(r[4]) for r in s])
        self.local = np.zeros(len(s), np.int64)
        self.sites, self.parks = {}, {}
        for i, r in enumerate(s):
            m = int(r[4])
            lst = self.sites.setdefault(m, [])
            self.local[i] = len(lst)
            lst.append(r[:4])
            self.parks.setdefault(m, []).append(p["site_park"][i])
        self.commutes = (np.array(p.get("commute_home", []), np.int64), np.array(p.get("commute_work", []), np.int64),
                         np.array(p.get("commute_trips", []), np.float64))


class Groups:
    """Commutes between the packs of a joint gravity group outside the Northeast (T-463: Japan's),
    data/work/<group>.joint_commutes.json from bin/joint_gravity.rs --packs: by 6 km square of each
    pack's own frame, matched here to the squares of the zone files' 6 km zones."""
    def __init__(self, reg):
        self.reg = {c["id"]: c for c in reg["cities"]}
        self.rows, self.source, self.keys = {}, {}, {}
        for path in sorted((ROOT / "data/work").glob("*.joint_commutes.json")):
            g = json.loads(path.read_text(encoding="utf-8"))
            ids = g["packs"]
            hp = np.array(g["home_pack"], np.int64)
            for i, cid in enumerate(ids):
                mine = np.nonzero(hp == i)[0]
                self.rows[cid] = [(tuple(g["home_zone"][k]), ids[g["work_pack"][k]], tuple(g["work_zone"][k]), g["trips"][k]) for k in mine]
                self.source[cid] = g["source"]

    def zone_keys(self, cid):
        """A pack's 6 km zones' squares, in its zone file's order."""
        if cid not in self.keys:
            o = self.reg[cid]["origin"]
            self.keys[cid] = zones_of(own_sites(cid, {"kind": "equirect", "lon": o["lon"], "lat": o["lat"]})[0])[0]
        return self.keys[cid]

    @staticmethod
    def index(keys, at, k):
        """The zone of square k, or of the nearest square (a merged cell can cross a 6 km line)."""
        if k in at:
            return at[k]
        return int(np.argmin([(a - k[0]) ** 2 + (b - k[1]) ** 2 for a, b in keys]))

    def commutes(self, cid, keys):
        at = {k: i for i, k in enumerate(keys)}
        ats = {}
        agg = {}
        for hk, wid, wk, t in self.rows.get(cid, []):
            wkeys = self.zone_keys(wid)
            wat = ats.setdefault(wid, {k: i for i, k in enumerate(wkeys)})
            k = (self.index(keys, at, hk), self.reg[wid]["place"], self.index(wkeys, wat, wk))
            agg[k] = agg.get(k, 0.0) + t
        return sorted(agg.items())


def zones_of(sites):
    """The 6 km zones of a zone's sites in first-seen order (the kernel's grouping), each site's zone,
    and each zone's centre weighted as the kernel weights it (people plus attraction)."""
    keys, zi = {}, np.zeros(len(sites), np.int64)
    for i, (x, y, *_r) in enumerate(sites):
        zi[i] = keys.setdefault((math.floor(x / ZONE_M), math.floor(y / ZONE_M)), len(keys))
    s = np.array(sites, np.float64)
    w = 2 * s[:, 2] + 2 * s[:, 3]
    n = len(keys)
    wz = np.bincount(zi, w, n)
    cx, cy = np.bincount(zi, w * s[:, 0], n) / np.maximum(wz, 1e-12), np.bincount(zi, w * s[:, 1], n) / np.maximum(wz, 1e-12)
    return list(keys), zi, cx, cy


def build(cid, reg, F, ne, air):
    city = next(c for c in reg["cities"] if c["id"] == cid)
    place = city["place"]
    lon0, lat0 = city["center"]
    country, road = F.wf.locate(lon0, lat0)
    cinfo = F.wf.countries[country]
    inc = min(cinfo["gdp"] / RULE["y_us"], 1.0)
    if city.get("market", -1) >= 0:
        frame = dict(ne.frame)
        sites, parks = ne.sites.get(city["market"], []), ne.parks.get(city["market"], [])
    else:
        frame = {"kind": "equirect", "lon": city["origin"]["lon"], "lat": city["origin"]["lat"]}
        sites, parks = own_sites(cid, frame)
    keys, zi, cx, cy = zones_of(sites)
    _, to_ll = equirect(frame["lon"], frame["lat"])
    zlon, zlat = to_ll(cx, cy)
    ng, nt = np.zeros(len(keys)), np.zeros(len(keys))
    for z in range(len(keys)):
        ng[z], nt[z] = F.normalisers(zlon[z], zlat[z], country, inc)
    s = np.array(sites, np.float64)
    slon, slat = to_ll(s[:, 0], s[:, 1])
    cell = F.cell_pop(slon, slat)
    out = {"format": 3, "zone": cid, "place": place, "frame": frame, "zone_m": ZONE_M,
           "country": cinfo["iso2"], "inc": round(inc, 6), "road": road,
           "zones": [list(k) for k in keys], "zone_ng": [float(f"{v:.7g}") for v in ng], "zone_nt": [float(f"{v:.7g}") for v in nt],
           "sites": [r + [int(z)] for r, z in zip(sites, zi)], "site_park": parks,
           "cell_pop": [round(float(v)) for v in cell],
           "sources": ["the zone pack's cells (600 m squares)", "the world field (T-375: Kontur 2023, Natural Earth, World Bank GDP per head)",
                       "pipeline/longhaul.json (T-376)"]}
    out.update(air.get(cid, {}))
    out["sources"].append(AIR_SOURCE)
    if city.get("market", -1) >= 0 and len(ne.commutes[0]):
        h, w, t = ne.commutes
        mine = ne.market[h] == city["market"]
        h, w, t = h[mine], w[mine], t[mine]
        wm = ne.market[w]
        # each Northeast market's place is its market id (pipeline/intercity.py)
        wz = np.array([zone_index[int(m)][int(ne.local[k])] for m, k in zip(wm, w)], np.int64) if len(w) else np.zeros(0, np.int64)
        out["commute_home"] = [int(zi[int(ne.local[k])]) for k in h]
        out["commute_place"] = [int(place_of_market[int(m)]) for m in wm]
        out["commute_work"] = [int(v) for v in wz]
        out["commute_trips"] = [float(v) for v in t]
        out["sources"].append("commutes between zones: bin/joint_gravity.rs (T-276), from data/packs/intercity.json")
    elif cid in groups.rows:
        rows = groups.commutes(cid, keys)
        out["commute_home"] = [k[0] for k, _ in rows]
        out["commute_place"] = [int(k[1]) for k, _ in rows]
        out["commute_work"] = [k[2] for k, _ in rows]
        out["commute_trips"] = [round(float(t), 3) for _, t in rows]
        out["sources"].append("commutes between zones: " + groups.source[cid])
    return out


zone_index = {}       # Northeast market id -> each of its sites' 6 km zone (local)
place_of_market = {}  # Northeast market id -> place
groups = None         # Groups, set in main


def cargo_bin(name, *args):
    env = dict(os.environ)
    env["PATH"] = os.path.join(os.path.expanduser("~"), ".cargo", "bin") + os.pathsep + env.get("PATH", "")
    env.setdefault("CARGO_BUILD_JOBS", "3")
    # build_city.py's target dir: the pipeline never waits on (or blocks) the app's wasm builds
    env["CARGO_TARGET_DIR"] = os.environ.get("TW_CARGO_TARGET_DIR", str(ROOT / "data/work/target"))
    cargo = shutil.which("cargo", path=env["PATH"])
    if not cargo:
        sys.exit("cargo not found: install Rust (rustup) or put ~/.cargo/bin on PATH")
    subprocess.run([cargo, "run", "--quiet", "--release", "--features", "demand", "--bin", name,
                    # TW_SIM_DIR: a copy of sim/ while the tree's own is mid-edit (as routes.py)
                    "--manifest-path", os.path.join(os.environ.get("TW_SIM_DIR", str(ROOT / "sim")), "Cargo.toml"), "--", *args],
                   env=env, check=True)


def assign_places(reg):
    """Every zone a market place: the Northeast's keep theirs (their market id), others append."""
    places = reg["market"]["places"]
    for c in reg["cities"]:
        if c.get("market", -1) >= 0:
            c["place"] = c["market"]
        elif "place" not in c:
            places.append({"name": c["name"], "regionName": c["name"] + " region"})
            c["place"] = len(places) - 1
    reg["market"]["rule"] = {k: v for k, v in RULE.items() if k not in ("task", "date", "form", "notes")}
    reg["market"]["rule"]["near_km"] = 250.0
    reg["market"]["rule"]["min_trips"] = 0.01


def write(path, out):
    part = path.with_suffix(".json.part")
    part.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    os.replace(part, path)


def main(ids=None):
    args = sys.argv[1:] if ids is None else ids
    air_only = "--air" in args
    reg = json.loads(REG.read_text(encoding="utf-8"))
    assign_places(reg)
    REG.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    air = airports_by_zone(reg)
    for cid, a in air.items():
        print(f"{cid}: airports {' '.join(a['air_codes']) or 'none'}", flush=True)
    if air_only:
        # every zone's airports into its file as it stands (no other field changes)
        for c in reg["cities"]:
            path = PACKS / f"{c['id']}.market.json"
            if c["id"] in air and path.exists():
                out = json.loads(path.read_text(encoding="utf-8"))
                out.update(air[c["id"]])
                if AIR_SOURCE not in out.setdefault("sources", []):
                    out["sources"].append(AIR_SOURCE)
                write(path, out)
        ids = []
    else:
        ids = [a for a in args if not a.startswith("--")] or [c["id"] for c in reg["cities"]]
    F = Field() if ids else None
    ne = Northeast()
    global groups
    groups = Groups(reg)
    for c in reg["cities"]:
        if c.get("market", -1) >= 0:
            place_of_market[c["market"]] = c["place"]
            zone_index[c["market"]] = zones_of(ne.sites[c["market"]])[1] if c["market"] in ne.sites else np.zeros(0, np.int64)
    for cid in ids:
        out = build(cid, reg, F, ne, air)
        path = PACKS / f"{cid}.market.json"
        write(path, out)
        print(f"{cid}: place {out['place']}, {out['country']}, income {out['inc']:.3f}, road {out['road']}, "
              f"{len(out['sites'])} sites, {len(out['zones'])} zones, {len(out.get('commute_trips', []))} commutes, "
              f"{path.stat().st_size / 1e6:.2f} MB", flush=True)
    rows = ROOT / "data/work/market_rows.json"
    cargo_bin("market_pack", "--root", str(ROOT), "--out", str(rows))
    got = json.loads(rows.read_text(encoding="utf-8"))
    reg = json.loads(REG.read_text(encoding="utf-8"))
    for c in reg["cities"]:
        if c["id"] in got["rows"]:
            c["marketRow"] = got["rows"][c["id"]]
    REG.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("registry rows:", ", ".join(sorted(got["rows"])))


if __name__ == "__main__":
    main()
