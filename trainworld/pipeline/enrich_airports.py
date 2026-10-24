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
SNAP_M = 1000.0
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


def wikidata_boardings(iata):
    """Global fallback: the latest Wikidata patronage (P3872, passengers both ways) / 2. Answers are
    kept in data/work/wikidata_boardings.json (T-1083: a cold query waits up to 8 s an airport);
    delete it to ask again."""
    path = os.path.join(b.WORK, "wikidata_boardings.json")

    def cached():
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}
    have = cached()
    if iata in have:
        return tuple(have[iata])
    q = f"""SELECT ?v ?t WHERE {{ ?a wdt:P238 "{iata}"; p:P3872 ?s. ?s ps:P3872 ?v.
            OPTIONAL {{ ?s pq:P585 ?t }} }} ORDER BY DESC(?t) LIMIT 1"""
    r = requests.get(WIKIDATA, params={"query": q, "format": "json"}, headers=b.UA, timeout=60)
    r.raise_for_status()
    rows = r.json()["results"]["bindings"]
    out = (float(rows[0]["v"]["value"]) / 2.0, rows[0].get("t", {}).get("value")) if rows else (None, None)
    try:  # builds run at once: read theirs again and write whole files; a busy file is skipped
        have = cached() | {iata: out}
        os.makedirs(b.WORK, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.part"
        with open(tmp, "w") as f:
            json.dump(have, f, indent=0, sort_keys=True)
        os.replace(tmp, path)
    except OSError as e:
        print(f"  Wikidata answer for {iata} not cached: {e}", flush=True)
    return out


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
    candidates = [c for c in h3.grid_disk(hid, 6) if h3.str_to_int(c) in index]
    if not candidates:
        return None, None
    nearest = min(candidates, key=dist)
    return (index[h3.str_to_int(nearest)], dist(nearest)) if dist(nearest) <= limit else (None, None)


def airports_in(boundary):
    a = pd.read_csv(b.fetch(OURAIRPORTS, SUB + "ourairports-airports.csv"), low_memory=False, keep_default_na=False)
    a = a[(a.scheduled_service == "yes") & a.type.isin(["large_airport", "medium_airport", "small_airport"])]
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
                placed.append(dict(name=name, osm=osm, skipped="no cell within 1 km"))
                continue
            trips[cell] += daily * m2 / total_m2
            placed.append(dict(name=name, osm=osm, m2=round(m2), share=m2 / total_m2, cell=int(cell), snap_m=snap))
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
