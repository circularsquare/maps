"""T-377 (T-362 stage 3): each registered zone's part of the world market, and the registry's market rows.

    python pipeline/market_zones.py [id ...]     every registered zone, or only those named
    python pipeline/market_zones.py --air        only every zone file's airports (T-378), then the rows
    python pipeline/market_zones.py --rows       only the rows (after a zone left the registry)
    ... --full    (with any of these) every row worked out again, not only those a change reaches

Writes `data/packs/<id>.market.json` per zone (seconds each), then runs the sim crate's `market_pack`,
which adds up each zone's registry row and the trips between zones whose centres are under 250 km
apart, and merges them into app/src/game/registry.json (`place`, `country`, `market.rule` /
`market.places`) and the rows into app/src/game/market-rows.json (T-776, market_rows.py). A new zone
needs only its own file and this merge. Only the rows a changed, new or gone zone file can reach are
worked out again (T-929, `plan_rows`; data/work/market_rows.inputs.json records what each row was made
from); a changed market_pack binary or market rule redoes them all.

A zone's file holds its endpoint sites (its pack's cells in 600 m squares of its own pack frame),
each site's parking for an arriving car and the people of its H3 r5 cell (the long-haul tail's
density, notes/T-376.md), its 6 km zones with their two normalisers against the world field (T-375:
ground and tail, constants in pipeline/longhaul.json), the commutes from it to other zones (its joint
gravity group's, from data/work/<group>.joint_commutes.json, T-463), the everyday outings from it
across zone lines (T-1175, from data/work/<unit>.cross_everyday.json), and (T-378) the airports its
people fly from with the zones each flies to directly.
"""
import csv
import hashlib
import json
import math
import os
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
import market_rows
import shared_lock
import simbin

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


# T-809: sim/src/demand/intercity.rs `OPEN_AREA` and `OPEN_AREA_BORDER`; keep the two the same
OPEN_AREA = {"AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE", "IT", "LV",
             "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE", "NO", "IS", "CH", "LI", "GB"}
OPEN_AREA_BORDER = 0.5


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
        self.open = np.array([c["iso2"] in OPEN_AREA for c in wf.countries])

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
        general = RULE["border"] * inc_o ** RULE["border_eps"]
        b = np.where(c == country, 1.0,
                     np.where(self.open[c] & self.open[country], max(OPEN_AREA_BORDER, general), general))
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
    """A zone's sites from its own pack ([x, y, people, jobs] in `frame`) and their parking: its
    cells grouped by 600 m square, each at its people-and-jobs centre."""
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


class Groups:
    """Commutes between the packs of a joint gravity group (T-463: Japan's first),
    data/work/<group>.joint_commutes.json from bin/joint_gravity.rs --packs: by 6 km square of each
    pack's own frame, matched here to the squares of the zone files' 6 km zones."""
    def __init__(self, reg):
        self.reg = {c["id"]: c for c in reg["cities"]}
        self.rows, self.source, self.keys = {}, {}, {}
        # each pair of packs from the newest file covering it (T-786: a country's file covers its own
        # packs' pairs and its fixed neighbours' pairs with them, not the neighbours' among themselves)
        import intercity
        pairs = {}
        for path, f in intercity.joint_files(ROOT / "data/work").items():  # oldest first
            g = json.loads(path.read_text(encoding="utf-8"))
            for k in [k for k in pairs if k[0] in f["packs"] and k[1] in f["packs"] and (k[0] in f["solved"] or k[1] in f["solved"])]:
                del pairs[k]
            ids = g["packs"]
            for k in range(len(g["trips"])):
                h, w = ids[g["home_pack"][k]], ids[g["work_pack"][k]]
                if h not in self.reg or w not in self.reg:
                    continue  # a pack that parked after this solve took its commutes out of the registry (T-997)
                pairs.setdefault((h, w), []).append((tuple(g["home_zone"][k]), w, tuple(g["work_zone"][k]), g["trips"][k]))
                self.source[h] = g["source"]
        for (h, _w), lst in pairs.items():
            self.rows.setdefault(h, []).extend(lst)
        self.everyday_rows, self.everyday_source = {}, {}
        self.add_everyday(ROOT / "data/work")

    def add_everyday(self, work):
        """T-1175: everyday outings across zone lines (pipeline/cross_everyday.py), each drawing pack's
        from the newest unit file that drew from it (a school or university draws from its site, so
        its pairs are the activity end's pack's), filed under the home end's pack."""
        import cross_everyday
        drawn = {}
        for path in cross_everyday.unit_files(work):  # oldest first
            g = json.loads(path.read_text(encoding="utf-8"))
            ids, by = g["packs"], {}
            for k in range(len(g["trips"])):
                h, a, kind = ids[g["home_pack"][k]], ids[g["act_pack"][k]], g["kind"][k]
                by.setdefault(a if kind in (2, 3) else h, []).append(
                    (h, tuple(g["home_zone"][k]), a, tuple(g["act_zone"][k]), kind, g["trips"][k], g["walk"][k]))
            for o in g["solved"]:
                drawn[o] = (by.get(o, []), g["source"])
        for lst, source in drawn.values():
            for h, hk, a, ak, kind, t, w in lst:
                if h in self.reg and a in self.reg:
                    self.everyday_rows.setdefault(h, []).append((hk, a, ak, kind, t, w))
                    self.everyday_source[h] = source

    def everyday(self, cid, keys):
        """A zone's everyday outings across zone lines by (home zone, place, activity zone, purpose):
        [outings a day, of them walking with no rail]."""
        at, ats, agg = {k: i for i, k in enumerate(keys)}, {}, {}
        for hk, aid, ak, kind, t, w in self.everyday_rows.get(cid, []):
            akeys = self.zone_keys(aid)
            aat = ats.setdefault(aid, {k: i for i, k in enumerate(akeys)})
            e = agg.setdefault((self.index(keys, at, hk), self.reg[aid]["place"], self.index(akeys, aat, ak), kind), [0.0, 0.0])
            e[0] += t
            e[1] += t * w
        return sorted(agg.items())

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


def build(cid, reg, F, air):
    city = next(c for c in reg["cities"] if c["id"] == cid)
    place = city["place"]
    lon0, lat0 = city["center"]
    country, road = F.wf.locate(lon0, lat0)
    # the pack's own country over the box centre's: Gorizia's box centre is in Slovenia (T-779)
    with open(PACKS / f"{cid}.json", encoding="utf-8") as f:
        own = (json.load(f).get("country") or "").upper()
    country = next((k for k, c in enumerate(F.wf.countries) if own and c["iso2"] == own), country)
    cinfo = F.wf.countries[country]
    inc = min(cinfo["gdp"] / RULE["y_us"], 1.0)
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
    if cid in groups.rows:
        rows = groups.commutes(cid, keys)
        out["commute_home"] = [k[0] for k, _ in rows]
        out["commute_place"] = [int(k[1]) for k, _ in rows]
        out["commute_work"] = [k[2] for k, _ in rows]
        out["commute_trips"] = [round(float(t), 3) for _, t in rows]
        out["sources"].append("commutes between zones: " + groups.source[cid])
    if cid in groups.everyday_rows:
        rows = groups.everyday(cid, keys)
        out["everyday_home"] = [k[0] for k, _ in rows]
        out["everyday_place"] = [int(k[1]) for k, _ in rows]
        out["everyday_zone"] = [k[2] for k, _ in rows]
        out["everyday_kind"] = [int(k[3]) for k, _ in rows]
        out["everyday_trips"] = [round(float(v[0]), 3) for _, v in rows]
        out["everyday_walk"] = [round(float(v[1] / max(v[0], 1e-12)), 4) for _, v in rows]
        out["sources"].append("everyday outings across zone lines: " + groups.everyday_source[cid])
    return out


groups = None  # Groups, set in main


# T-929: market_pack works out only the rows a change can reach. A row reads its own pack and the
# packs of places whose centres are under near_km from it, so the rows redone are those of the
# changed (or new, or gone) packs and of places near them, each from a composition of the places
# near it and those its commutes and everyday outings reach. A changed pack's centre is estimated
# here, so both reach MARGIN_KM further.
INPUTS = ROOT / "data/work/market_rows.inputs.json"
MARGIN_KM = 25.0


def _sha(b):
    return hashlib.sha1(b).hexdigest()


def _row_sha(row):
    return _sha(json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def pack_centre(z):
    """A market pack's place centre as the kernel works it out (sites weighted by people plus
    attraction, on the sphere), to well under MARGIN_KM; None for a frame other than equirect."""
    f = z["frame"]
    if f.get("kind") != "equirect" or not z["sites"]:
        return None
    s = np.array([r[:4] for r in z["sites"]], np.float64)
    _, to_ll = equirect(f["lon"], f["lat"])
    lon, lat = to_ll(s[:, 0], s[:, 1])
    u = ((2 * s[:, 2] + 2 * s[:, 3])[:, None] * W._unit(lon, lat)).sum(axis=0)
    return float(np.degrees(np.arctan2(u[1], u[0]))), float(np.degrees(np.arctan2(u[2], math.hypot(u[0], u[1]))))


def plan_rows(reg, have, code, full=False):
    """(rows, compose, record): the zone ids whose rows to work out and the ids their composition
    needs (both None: every row), and the inputs record to write once the rows are merged."""
    rule = _sha(json.dumps([reg["market"]["rule"], reg["market"].get("frame")], sort_keys=True).encode("utf-8"))
    old = json.loads(INPUTS.read_text(encoding="utf-8")) if INPUTS.exists() else None
    was = old["zones"] if old else {}
    cur = {}
    for c in reg["cities"]:
        p = PACKS / f"{c['id']}.market.json"
        if "place" in c and p.exists():
            s = p.stat()
            st = [s.st_size, s.st_mtime_ns]
            # a pack is only ever replaced (a new modified time), so the same size and time is the same file
            w = was.get(c["id"], {})
            cur[c["id"]] = (c["place"], w["pack"] if w.get("stat") == st else _sha(p.read_bytes()), st)
    record = {"code": code, "rule": rule, "packs": cur}
    if full or old is None or old.get("code") != code or old.get("rule") != rule:
        return None, None, record
    changed = [i for i, (pl, h, _) in cur.items()
               if i not in was or was[i]["place"] != pl or was[i]["pack"] != h or i not in have or _row_sha(have[i]) != was[i]["row"]]
    gone = [i for i in was if i not in cur]
    at = {i: (was[i]["lon"], was[i]["lat"]) for i in cur if i not in changed}
    to = {}

    def reached(z):  # the places a row's commutes and everyday outings across zone lines (T-1175) read
        return set(z.get("commute_place", [])) | set(z.get("everyday_place", []))
    for i in changed:
        z = json.loads((PACKS / f"{i}.market.json").read_text(encoding="utf-8"))
        c = pack_centre(z)
        if c is None:
            print(f"market rows: {i}'s centre is not estimated here (its frame), so every row", flush=True)
            return None, None, record
        at[i] = c
        to[i] = reached(z)
    near = float(reg["market"]["rule"]["near_km"]) + MARGIN_KM
    ids = list(cur)
    lon, lat = np.array([at[i][0] for i in ids]), np.array([at[i][1] for i in ids])

    def around(points):
        hit = np.zeros(len(ids), bool)
        for plon, plat in points:
            hit |= W.km(plon, plat, lon, lat) < near
        return {ids[k] for k in np.nonzero(hit)[0]}
    moved = [at[i] for i in changed] + [(was[i]["lon"], was[i]["lat"]) for i in changed + gone if i in was]
    rows = set(changed) | around(moved)
    of_place = {pl: i for i, (pl, _, _) in cur.items()}
    compose = around([at[i] for i in rows])
    for i in rows:
        # a row keeps no everyday places, so an unchanged row's come from its pack (T-1256)
        places = to[i] if i in to else reached(json.loads((PACKS / f"{i}.market.json").read_text(encoding="utf-8")))
        compose |= {of_place[p] for p in places if p in of_place}
    print(f"market rows: {len(rows)} of {len(cur)} places' rows from {len(compose)} packs "
          f"({len(changed)} changed, {len(gone)} gone)", flush=True)
    return sorted(rows), sorted(compose | rows), record


def unreached(want, compose, rows, record, reg):
    """The check on plan_rows' estimate, with the rows' own centres: a wanted row missing, or a
    registered place within near_km (and a km) of a new row left out of its composition."""
    if any(i not in rows for i in want):
        return [i for i in want if i not in rows]
    ids = [i for i in record["packs"] if i in rows]
    lon, lat = np.array([rows[i]["lon"] for i in ids]), np.array([rows[i]["lat"] for i in ids])
    inside, near = set(compose) | set(want), float(reg["market"]["rule"]["near_km"]) + 1.0
    out = set()
    for i in want:
        out |= {ids[k] for k in np.nonzero(W.km(rows[i]["lon"], rows[i]["lat"], lon, lat) < near)[0]} - inside
    return sorted(out)


def save_record(record, have):
    """What the merged rows were made from (plan_rows reads it next time)."""
    zones = {}
    for i, (pl, h, st) in record.pop("packs").items():
        r = have.get(i)
        if r is not None:
            zones[i] = {"place": pl, "pack": h, "stat": st, "row": _row_sha(r), "lon": r["lon"], "lat": r["lat"]}
    record["zones"] = zones
    part = INPUTS.with_suffix(".json.part")
    part.write_text(json.dumps(record, separators=(",", ":")), encoding="utf-8")
    os.replace(part, INPUTS)


def assign_places(reg):
    """Every zone a market place: a new one appends, and places are never renumbered (the
    Northeast's 13 keep their old market ids, T-1014)."""
    places = reg["market"]["places"]
    for c in reg["cities"]:
        if "place" not in c:
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
    # registry, market rows and data/work/market_rows.json: one writer at a time (T-786)
    with shared_lock.held("registry", "market_zones " + " ".join(args)[:200]):
        _main(args)


def _main(args):
    global groups
    air_only = "--air" in args
    reg = shared_lock.read_json(REG)
    assign_places(reg)
    shared_lock.write_atomic(REG, json.dumps(reg, indent=2, ensure_ascii=False) + "\n")
    if "--rows" in args:
        ids = []  # T-540: only the market rows, after a zone left the registry
    else:
        ids = [a for a in args if not a.startswith("--")] or [c["id"] for c in reg["cities"]]
    air = airports_by_zone(reg) if ids or air_only else {}
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
    if ids:
        F = Field()
        groups = Groups(reg)
    for cid in ids:
        out = build(cid, reg, F, air)
        path = PACKS / f"{cid}.market.json"
        write(path, out)
        print(f"{cid}: place {out['place']}, {out['country']}, income {out['inc']:.3f}, road {out['road']}, "
              f"{len(out['sites'])} sites, {len(out['zones'])} zones, {len(out.get('commute_trips', []))} commutes, "
              f"{path.stat().st_size / 1e6:.2f} MB", flush=True)
    rows = ROOT / "data/work/market_rows.json"
    exe = simbin.exe(ROOT, "market_pack")
    reg = shared_lock.read_json(REG)
    have = market_rows.load(ROOT)
    want, compose, record = plan_rows(reg, have, _sha(exe.read_bytes()), full="--full" in args)
    if want is None:
        subprocess.run([str(exe), "--root", str(ROOT), "--out", str(rows)], check=True)
        got = json.loads(rows.read_text(encoding="utf-8"))
    elif want:
        only = ROOT / "data/work/market_rows.only.json"
        only.write_text(json.dumps({"rows": want, "compose": compose}), encoding="utf-8")
        subprocess.run([str(exe), "--root", str(ROOT), "--out", str(rows), "--only", str(only)], check=True)
        got = json.loads(rows.read_text(encoding="utf-8"))
        miss = unreached(want, compose, {**have, **got["rows"]}, record, reg)
        if miss:
            print(f"market rows: {miss} not in the composition, so every row", flush=True)
            subprocess.run([str(exe), "--root", str(ROOT), "--out", str(rows)], check=True)
            got = json.loads(rows.read_text(encoding="utf-8"))
    else:
        got = {"rows": {}}
    have.update(got["rows"])
    market_rows.with_countries(reg, have)
    shared_lock.write_atomic(REG, json.dumps(reg, indent=2, ensure_ascii=False) + "\n")
    save_record(record, market_rows.save(reg, have, ROOT))
    print("registry rows:", ", ".join(sorted(got["rows"])))


if __name__ == "__main__":
    main()
