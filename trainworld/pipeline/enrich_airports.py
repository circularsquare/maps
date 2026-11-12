"""T-142: airport ground trips for a finished city pack, in place. `enrich_airports.py <city>`.

Each commercial airport in the pack's counties (OurAirports, global, `scheduled_service` yes)
gets its annual passenger boardings from a national source (US: FAA CY2024 enplanements; a
Wikidata P3872 patronage figure, halved, where the national source lacks the airport). Then:

    daily outings at the airport = boardings x O&D share x pack share / days in the year

- O&D share: the share of the airport's boardings that start the passenger's air journey there
  rather than connect (BTS DB1B Market, 10% ticket sample, 2024 Q2, domestic itineraries;
  1 where the sample has under 50 passengers, about 500 real ones). Connecting passengers never leave the airport.
- Pack share: the airport's ground trips that start or end in the pack's counties: county
  residents (2020 centres of population) weighted by exp(-distance / 20 km), the pack's counties
  against every US county within 300 km. The rest belong to neighbouring areas.
- One outing is a ground trip to the airport (a passenger leaving) paired with one from it (a
  passenger arriving): boardings and alightings are about equal, so legs = 2 x outings.

The outings sit at the airport's passenger terminals (OSM aeroway=terminal within 4 km of the
airport's point, split by footprint area; the airport's point if it has none). Where in the city
they go to and come from is the kernel's (`everyday.rs`, kind 4): half residents, half visitors
(jobs and hotels), by a 20 km distance decay, normalised per airport so each airport keeps its
total. Airport staff are already commuters (LODES jobs) and are not counted again.
"""
import hashlib
import json
import os
import re
import sys
import zipfile

import h3
import numpy as np
import pandas as pd
import requests
import shapely
from shapely.geometry import Point

import build_city as b
import packio
import overpass

YEAR = 2024
DAYS = 366
DECAY_M = 20000.0          # same as everyday.rs DECAY[4]
SHARE_RADIUS_M = 300000.0
MIN_DB1B_SAMPLE = 50       # sampled passengers (10% sample: about 500 a quarter)
TERMINAL_RADIUS_M = 4000.0
MIN_TERMINAL_M2 = 1500.0
# T-2896 (3): Shijiazhuang's terminal is 1,423 m from the Gaocheng zone's nearest cell
SNAP_M = 3000.0
FAA = "https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger/arp-cy2024-commercial-service-enplanements.xlsx"
OURAIRPORTS = "https://davidmegginson.github.io/ourairports-data/airports.csv"
DB1B = "https://transtats.bts.gov/PREZIP/Origin_and_Destination_Survey_DB1BMarket_2024_2.zip"
CENPOP = "https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO.txt"
WIKIDATA = "https://query.wikidata.org/sparql"
SUB = "t142b/"
NOT_PASSENGER = re.compile(r"cargo|freight|fbo|general aviation|private|executive|maintenance|hangar|signature|"
                           r"atlantic aviation|jet aviation|million air|sheltair|ross aviation|heliport|fire|"
                           r"crew|catering|post office|mail|squadron|military|air national guard|air force", re.I)


def metres(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * b.R_EARTH * np.arcsin(np.sqrt(a))


def enplanements():
    """FAA CY2024 boardings by FAA location id."""
    x = pd.read_excel(b.fetch(FAA, SUB + FAA.rsplit("/", 1)[1]))
    x.columns = [str(c).strip() for c in x.columns]
    col = next(c for c in x.columns if c.startswith("CY 24"))
    return {str(r["Locid"]).strip(): float(r[col]) for _, r in x.iterrows() if pd.notna(r["Locid"]) and pd.notna(r[col])}


def od_shares():
    """{airport code: [passengers starting there, passengers connecting there]} from DB1B Market
    (10% sample, one quarter), cached in data/work so the 110 MB zip can go."""
    cache = os.path.join(b.WORK, "db1b_2024q2_od.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    path = b.fetch(DB1B, SUB + DB1B.rsplit("/", 1)[1])
    start, connect = {}, {}
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(name) as f:
            for chunk in pd.read_csv(f, usecols=["Origin", "AirportGroup", "Passengers"], chunksize=2_000_000):
                for k, v in chunk.groupby("Origin").Passengers.sum().items():
                    start[k] = start.get(k, 0.0) + float(v)
                via = chunk[chunk.AirportGroup.str.count(":") >= 2]
                mids = via.AirportGroup.str.split(":").str[1:-1]
                ex = pd.DataFrame({"code": mids, "pax": via.Passengers}).explode("code")
                for k, v in ex.groupby("code").pax.sum().items():
                    connect[k] = connect.get(k, 0.0) + float(v)
                print(f"  DB1B: {len(start)} origins so far", flush=True)
    out = {k: [start.get(k, 0.0), connect.get(k, 0.0)] for k in set(start) | set(connect)}
    with open(cache + ".part", "w") as f:
        json.dump(dict(sorted(out.items())), f)
    os.replace(cache + ".part", cache)
    return out


# T-2896 (2): which Wikidata patronage statement an airport takes. Years of the pandemic's
# collapse in air travel lose to any other year from OLDEST_GOOD on; part-year figures never count.
PANDEMIC = (2020, 2021, 2022)
OLDEST_GOOD = 2016
WIKIDATA_BATCH = 200
# the rule's answers; a new name when the rule changes, so every code is asked again
BOARDINGS_CACHE = "wikidata_boardings.v2.json"


def wikidata_rows(codes):
    """Every non-deprecated Wikidata patronage statement (P3872, passengers both ways) of the
    airports with these IATA codes (P238), up to WIKIDATA_BATCH a query: [{iata, v, t, prec, start,
    end, part}], dates as ISO strings or None. `part` is set where a qualifier limits the figure to
    a part of the traffic (P518 applies to part, P642 of)."""
    codes = sorted({c for c in codes if c and re.fullmatch(r"[A-Z0-9]{3}", c)})
    if not codes:
        return []
    assert len(codes) <= WIKIDATA_BATCH, len(codes)
    q = f"""SELECT ?iata ?v ?t ?prec ?start ?end ?part WHERE {{
      VALUES ?iata {{ {" ".join(f'"{c}"' for c in codes)} }}
      ?a wdt:P238 ?iata; p:P3872 ?s. ?s ps:P3872 ?v; wikibase:rank ?rank.
      FILTER(?rank != wikibase:DeprecatedRank)
      OPTIONAL {{ ?s pqv:P585 ?tv. ?tv wikibase:timeValue ?t; wikibase:timePrecision ?prec }}
      OPTIONAL {{ ?s pq:P580 ?start }} OPTIONAL {{ ?s pq:P582 ?end }}
      OPTIONAL {{ ?s pq:P518|pq:P642 ?part }} }}"""
    for attempt in range(6):
        r = requests.post(WIKIDATA, data={"query": q, "format": "json"}, headers=b.UA, timeout=120)
        if r.status_code in (429, 502, 503, 504) and attempt < 5:
            import time
            time.sleep(float(r.headers.get("Retry-After") or 0) or 5 * 2 ** attempt)
            continue
        r.raise_for_status()
        break
    val = lambda row, k: row[k]["value"] if k in row else None  # noqa: E731
    return [dict(iata=val(x, "iata"), v=float(val(x, "v")), t=val(x, "t"), prec=int(val(x, "prec")) if "prec" in x else None,
                 start=val(x, "start"), end=val(x, "end"), part=val(x, "part")) for x in r.json()["results"]["bindings"]]


def statement_year(row):
    """The calendar year a patronage statement covers, or None when it is not a whole year's: a
    point in time of year precision, or of day precision on 1 January or 31 December (how annual
    figures get entered); a start and end a year apart. Month figures (Hong Kong's 2024-12) and any
    other day are part years."""
    if row.get("part"):
        return None
    if row.get("start") and row.get("end"):
        s, e = row["start"][:10].lstrip("+"), row["end"][:10].lstrip("+")
        try:
            import datetime as dt
            days = (dt.date.fromisoformat(e) - dt.date.fromisoformat(s)).days
        except ValueError:
            return None
        return int(s[:4]) if 360 <= days <= 370 else None
    t, prec = row.get("t"), row.get("prec")
    if not t:
        return None
    y, md = int(t.lstrip("+")[:4]), t.lstrip("+")[5:10]
    if prec == 9:
        return y
    if prec == 11 and md in ("01-01", "12-31"):
        return y
    return None


def pick(rows):
    """One airport's statements -> (boardings a year, "<year>") or (None, None): the newest whole
    year outside PANDEMIC from OLDEST_GOOD on, else the newest whole year at all; two figures for one
    year take the lower (Anita, 2026-11-12: where sources disagree, the lower). Boardings are half the
    passengers both ways."""
    by_year = {}
    for r in rows:
        y = statement_year(r)
        if y is not None and r["v"] > 0:
            by_year[y] = min(by_year.get(y, r["v"]), r["v"])
    if not by_year:
        return None, None
    good = [y for y in by_year if y not in PANDEMIC and y >= OLDEST_GOOD]
    y = max(good) if good else max(by_year)
    return by_year[y] / 2.0, str(y)


def _cache_path():
    return os.path.join(b.WORK, BOARDINGS_CACHE)


def _cached():
    try:
        with open(_cache_path()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def fill_wikidata_boardings(codes, batch=WIKIDATA_BATCH):
    """Ask Wikidata for every IATA code in `codes` not yet in the cache (data/work/BOARDINGS_CACHE),
    WIKIDATA_BATCH codes a query, and cache each code's `pick` (codes with no statement too). Returns
    {code: (boardings, year)} for all of `codes`."""
    want = sorted({c for c in codes if c})
    todo = [c for c in want if c not in _cached()]
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        rows = wikidata_rows(part)
        got = {c: [] for c in part}
        for r in rows:
            got.setdefault(r["iata"], []).append(r)
        new = {c: list(pick(rs)) for c, rs in got.items() if c in part}
        try:  # builds run at once: read theirs again and write whole files; a busy file is skipped
            have = _cached() | new
            os.makedirs(b.WORK, exist_ok=True)
            tmp = f"{_cache_path()}.{os.getpid()}.part"
            with open(tmp, "w") as f:
                json.dump(have, f, indent=0, sort_keys=True)
            os.replace(tmp, _cache_path())
        except OSError as e:
            print(f"  Wikidata answers for {len(new)} codes not cached: {e}", flush=True)
        if len(todo) > batch:
            print(f"  Wikidata boardings: {min(i + batch, len(todo))} of {len(todo)} codes asked", flush=True)
    have = _cached()
    return {c: tuple(have.get(c, (None, None))) for c in want}


def wikidata_boardings(iata):
    """Global fallback: Wikidata patronage (P3872, passengers both ways) / 2, the statement `pick`
    chooses: (boardings a year, "<year>") or (None, None). Answers are kept in
    data/work/BOARDINGS_CACHE (T-1083); delete it to ask again."""
    return fill_wikidata_boardings([iata])[iata]


# T-2896 (1): passenger counts a national authority publishes for each of its airports
# (pipeline/airport_passengers.csv: country, OurAirports ident, IATA, ICAO, the source's name,
# passengers both ways, year, source). China: CAAC's 2025 bulletin.
PUBLISHED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "airport_passengers.csv")
_published = None


def published():
    """{OurAirports ident or IATA code: (boardings a year, year, source)} from PUBLISHED."""
    global _published
    if _published is None:
        _published = {}
        t = pd.read_csv(PUBLISHED, dtype=str, keep_default_na=False)
        for r in t.itertuples():
            v = (float(r.passengers) / 2.0, str(r.year), r.source)
            for k in (r.ident, r.icao, r.iata):
                if k:
                    _published.setdefault(k, v)
    return _published


def published_idents():
    t = pd.read_csv(PUBLISHED, dtype=str, keep_default_na=False)
    return (set(t.ident) | set(t.icao)) - {""}


def best_boardings(ap):
    """An airport's boardings a year: (boardings, year, source) or (None, None, None). A national
    authority's count (`published`) and Wikidata's `pick`: the newer year wins; one year in both takes
    the lower (Anita, 2026-11-12: at or a bit under published figures, never over)."""
    got = []
    pub = published().get(ap["ident"]) or (published().get(ap["iata"]) if ap.get("iata") else None)
    if pub:
        got.append(pub)
    if ap.get("iata"):
        v, y = wikidata_boardings(ap["iata"])
        if v:
            got.append((v, y, "Wikidata P3872"))
    if not got:
        return None, None, None
    newest = max(int(g[1]) for g in got)
    return min((g for g in got if int(g[1]) == newest), key=lambda g: g[0])


# T-2896: the share of a hub's passengers who start or end their air journey there (1 - the
# operator's published transfer share, rounded down under Anita's undershoot rule, 2026-11-12;
# notes/T-2896.md). A city's own `od_share` config comes first; US airports take DB1B's.
HUB_OD_SHARE = {
    "LTFM": 0.5,   # Istanbul: close to half connecting (2025)
    "EDDF": 0.5,   # Frankfurt: 48.7% transfer (2025)
    "EDDM": 0.55,  # Munich: 42% transfer (2025)
    "EHAM": 0.6,   # Amsterdam: 36.6% transfer (2025, Royal Schiphol Group)
    "OTHH": 0.25,  # Doha: 74% transfer (OAG via Gulf Times, 2025)
    "OMDB": 0.5,   # Dubai: 47% connecting (OAG via Gulf Times, 2025)
    "WSSS": 0.7,   # Singapore: no published share found; judgement as Heathrow's
}


def od_share_of(ap, cfg):
    """(O&D share, where it came from) for a tier 0 airport: the city's config, HUB_OD_SHARE, DB1B
    for a US airport (its 10% sample's cache only; the hand-built packs' rule), else 1."""
    if ap["ident"] in cfg.get("od_share", {}):
        return cfg["od_share"][ap["ident"]], "config"
    if ap["ident"] in HUB_OD_SHARE:
        return HUB_OD_SHARE[ap["ident"]], "hub table"
    if ap.get("country") == "US" and ap.get("iata") and os.path.exists(os.path.join(b.WORK, "db1b_2024q2_od.json")):
        start, connect = od_shares().get(ap["iata"], [0.0, 0.0])
        if start + connect >= MIN_DB1B_SAMPLE:
            return start / (start + connect), "DB1B 2024 Q2"
    return 1.0, "none"


def place_sites(index, sites, limit=None):
    """An airport's sites (lat, lon, m2, ...) -> [(cell, share of the airport, snap m)] for the sites
    with a cell within `limit` (SNAP_M), shares by footprint over the placed sites only, so an airport
    keeps its total while any site places. Empty when none does (the caller warns)."""
    limit = SNAP_M if limit is None else limit
    placed = []
    for s in sites:
        c, snap = cell_of(index, s[0], s[1], limit)
        if c is not None:
            placed.append((c, s[2], snap))
    m2 = sum(p[1] for p in placed)
    return [(c, a / m2, snap) for c, a, snap in placed] if m2 > 0 else []


def pack_share(cfg, lat, lon):
    if cfg.get("country") == "ca":  # T-277: Ontario's census subdivisions and the US counties
        import canada
        pts = canada.population_points()
        d = metres(lat, lon, pts.lat.to_numpy(), pts.lon.to_numpy())
        w = pts["pop"].to_numpy() * np.exp(-d / DECAY_M) * (d <= SHARE_RADIUS_M)
        inside = pts.cd.isin(set(cfg["counties"])).to_numpy()
        return float(w[inside].sum() / w.sum()) if w.sum() > 0 else 0.0
    co = pd.read_csv(b.fetch(CENPOP, SUB + "CenPop2020_Mean_CO.txt"), dtype={"STATEFP": str, "COUNTYFP": str}, encoding="utf-8-sig")
    co["geoid"] = co.STATEFP + co.COUNTYFP
    d = metres(lat, lon, co.LATITUDE.to_numpy(), co.LONGITUDE.to_numpy())
    near = d <= SHARE_RADIUS_M
    w = co.POPULATION.to_numpy() * np.exp(-d / DECAY_M) * near
    inside = co.geoid.isin(set(cfg["counties"])).to_numpy()
    return float(w[inside].sum() / w.sum()) if w.sum() > 0 else 0.0


def terminals(city, boundary, airports):
    """Passenger terminal footprints near each airport: {ident: [(lat, lon, m2, name)]}."""
    s, w, n, e = overpass.bbox_of(boundary, pad=0.05)
    data = overpass.fetch(f"{SUB}osm_terminals_{city}",
                          f'[out:json][timeout:300];(way["aeroway"="terminal"]({s},{w},{n},{e});'
                          f'relation["aeroway"="terminal"]({s},{w},{n},{e}););out geom tags;')
    out = {a["ident"]: [] for a in airports}
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name", "")
        if NOT_PASSENGER.search(name) or tags.get("terminal") in ("cargo", "general_aviation", "fbo"):
            continue
        poly = overpass.polygon(el)
        if poly is None:
            continue
        m2 = overpass.area_m2(poly)
        if m2 < MIN_TERMINAL_M2:
            continue
        c = poly.representative_point()
        best = min(airports, key=lambda a: metres(a["lat"], a["lon"], c.y, c.x))
        if metres(best["lat"], best["lon"], c.y, c.x) <= TERMINAL_RADIUS_M:
            out[best["ident"]].append((c.y, c.x, m2, name, el["type"][0] + str(el["id"])))
    return out


def cell_of(index, lat, lon, limit):
    hid = h3.latlng_to_cell(lat, lon, 9)
    cell = index.get(h3.str_to_int(hid))
    if cell is not None:
        return cell, 0.0
    def dist(c):
        la, lo = h3.cell_to_latlng(c)
        return float(metres(lat, lon, la, lo))
    # res-9 centres are about 300 m apart: ring k reaches at least 0.26k km
    candidates = [c for c in h3.grid_disk(hid, max(6, int(limit / 260.0) + 1)) if h3.str_to_int(c) in index]
    if not candidates:
        return None, None
    nearest = min(candidates, key=dist)
    return (index[h3.str_to_int(nearest)], dist(nearest)) if dist(nearest) <= limit else (None, None)


def airports_in(boundary):
    """OurAirports' airports with scheduled service in the boundary, and any a national count in
    PUBLISHED names (CAAC lists ten that OurAirports does not mark scheduled)."""
    a = pd.read_csv(b.fetch(OURAIRPORTS, SUB + "ourairports-airports.csv"), low_memory=False, keep_default_na=False)
    listed = published_idents()
    a = a[((a.scheduled_service == "yes") & a.type.isin(["large_airport", "medium_airport", "small_airport"])) | a.ident.isin(listed)]
    rows = []
    shapely.prepare(boundary)  # T-425
    for r in a.itertuples():
        lat, lon = float(r.latitude_deg), float(r.longitude_deg)
        if boundary.covers(Point(lon, lat)):
            rows.append(dict(ident=r.ident, name=r.name, iata=r.iata_code, local=r.local_code, type=r.type,
                             country=r.iso_country, lat=lat, lon=lon, municipality=r.municipality))
    return rows


def main(city="nyc"):
    cfg = b.CITIES[city]
    boundary = b.build_boundary(city, cfg)
    airports = airports_in(boundary)
    canadian = cfg.get("country") == "ca"
    if canadian:  # T-277: StatCan's enplaned passengers by airport; no connecting share published
        import canada
        statcan = canada.airport_boardings(YEAR)
        faa, od = {}, {}
    else:
        faa = enplanements()
        od = od_shares()
    found = terminals(city, boundary, airports) if airports else {}
    stem = os.path.join(b.PACKS, city)
    with open(stem + ".json") as f:
        header = json.load(f)
    with open(stem + ".bin", "rb") as f:
        buf = f.read()
    a = next(a for a in header["arrays"] if a["name"] == "h3")
    cells = np.frombuffer(buf, dtype="<u8", count=a["count"], offset=a["offset"])
    index = {int(c): i for i, c in enumerate(cells)}
    trips = np.zeros(len(cells), dtype="<f4")
    audit = []
    for ap in airports:
        code = ap["local"] or ap["iata"]
        boardings, source = faa.get(code), f"FAA CY{YEAR} enplanements"
        if canadian:
            geo, score = canada.match_airport(ap, statcan)
            boardings, source = (statcan[geo][0], f"StatCan 23-10-0253-01 enplaned {statcan[geo][1]} ({geo})") if geo else (None, None)
        if boardings is None and ap["iata"]:
            boardings, when = wikidata_boardings(ap["iata"])
            source = f"Wikidata P3872 patronage / 2 ({when})"
        if not boardings:
            audit.append(ap | dict(skipped="no passenger count"))
            continue
        start, connect = od.get(ap["iata"] or code, [0.0, 0.0])
        od_share = start / (start + connect) if start + connect >= MIN_DB1B_SAMPLE else 1.0
        if canadian:  # no national connecting-share source: a judgement value for hubs (T-277)
            od_share = canada.OD_SHARE_JUDGEMENT.get(ap["ident"], 1.0)
        share = pack_share(cfg, ap["lat"], ap["lon"])
        daily = boardings * od_share * share / DAYS
        sites = found.get(ap["ident"]) or [(ap["lat"], ap["lon"], 1.0, "airport point", None)]
        total_m2 = sum(s[2] for s in sites)
        placed = []
        for lat, lon, m2, name, osm in sites:
            cell, snap = cell_of(index, lat, lon, SNAP_M)
            if cell is None:
                placed.append(dict(name=name, osm=osm, skipped=f"no cell within {SNAP_M / 1000:g} km"))
                continue
            trips[cell] += daily * m2 / total_m2
            placed.append(dict(name=name, osm=osm, m2=round(m2), share=m2 / total_m2, cell=int(cell), snap_m=snap))
        if not any("cell" in p for p in placed):
            print(f"  WARNING {code} {ap['name']}: {daily:,.0f} outings a day and no cell within {SNAP_M / 1000:g} km of any "
                  f"terminal: nothing placed", flush=True)
        audit.append(ap | dict(boardings=boardings, source=source, db1b_start=start, db1b_connect=connect,
                               od_share=od_share, pack_share=share, daily_outings=daily, terminals=placed))
        print(f"  {code} {ap['name']}: {boardings:,.0f} boardings, O&D {od_share:.2f}, pack {share:.2f}, "
              f"{daily:,.0f} outings a day at {sum('cell' in p for p in placed)} terminal(s)", flush=True)
    buf = packio.replace_cell_arrays(header, buf, {"airport_trips": trips})
    header["airport"] = dict(
        coverage="Commercial airports in the pack's counties (OurAirports scheduled service); air passengers' ground trips to and from them, the share starting or ending in the pack",
        year=YEAR, airports=sum(1 for r in audit if "daily_outings" in r),
        outings=float(trips.sum(dtype=np.float64)), decay_m=DECAY_M,
        origin_mix="half residents, half visitors (jobs and leisure/hotel jobs, a quarter each)",
        formula="daily outings = boardings x O&D share (DB1B 2024 Q2) x pack share (county population x exp(-d/20 km)) / 366; legs = 2 x outings",
        limitations="Domestic DB1B connecting shares applied to all boardings; no airport staff (already commuters); no trips from outside the pack; terminals by OSM footprint area",
        sources=[FAA, OURAIRPORTS, DB1B, CENPOP, "OpenStreetMap aeroway=terminal (ODbL)", WIKIDATA + " (fallback only)"],
        source_sha256={n: hashlib.sha256(open(os.path.join(b.RAW, SUB + n), "rb").read()).hexdigest()
                       for n in ["arp-cy2024-commercial-service-enplanements.xlsx", "ourairports-airports.csv", "CenPop2020_Mean_CO.txt"]})
    if canadian:
        header["airport"].update(
            formula="daily outings = enplaned passengers (StatCan 23-10-0253-01) x O&D share (judgement: Pearson 0.75, others 1) x pack share (Ontario census subdivisions' "
                    "and US counties' population x exp(-d/20 km)) / 366; legs = 2 x outings",
            limitations="No national connecting-share source for Canadian airports (no DB1B): Pearson's 0.75 is judgement; "
                        "no airport staff (already commuters); no trips from outside the pack; terminals by OSM footprint area",
            sources=[canada.STATCAN_CSV.format(t=canada.AIR_TABLE), OURAIRPORTS, canada.CSD_LAYER, CENPOP,
                     "OpenStreetMap aeroway=terminal (ODbL)", WIKIDATA + " (fallback only)"],
            source_sha256={n: hashlib.sha256(open(os.path.join(b.RAW, SUB + n), "rb").read()).hexdigest()
                           for n in ["ourairports-airports.csv", "CenPop2020_Mean_CO.txt"]})
    packio.write(stem, header, buf)
    with open(os.path.join(b.WORK, city + ".airports.audit.json"), "w") as f:
        json.dump(dict(metadata=header["airport"], airports=audit), f, indent=2)
    print(f"{city}: {header['airport']['airports']} airports, {header['airport']['outings']:,.0f} outings a day", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:2])
