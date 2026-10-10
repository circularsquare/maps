"""Sri Lanka: Sri Lanka Railways' lines as en.wikipedia's "List of railway stations in Sri Lanka by
line" lists them, station by station with km from Colombo Fort, written into rinf.py's input
format (nafrica_register.py's recipe), so rinf.py traces them over OSM track, places the stops
and names the lines. bd_register.py uses the same engine for Bangladesh.

    python lk_register.py --clip lk               # after every extract (CLIPPING below)
    python build_model.py --region lk --register lk_register:data/raw/rinf/lk
    python lk_register.py --dry lk                # the conversion alone, with its log
    python lk_register.py --trace lk "Colombo Fort" "Maradana"    # a trace, to write lists

`--register lk_register:data/raw/rinf/lk` converts and then runs rinf.build on the result. The
settings rinf.py reads are in rinf_countries/lk.py (`country_conf` below). Sources, what runs
and the numbers: lk_sources.md.

THE REGISTER is the by-line tables (data/raw/lk/survey/wp_stations_by_line.json, read
2026-10-08): one table per line, every station with SLR's distance from Colombo Fort. Those
distances are this register's chainage: a section's length is the difference of its two ends'
km, shipped as `chain` / `km_official`, so check_model compares every line with it. A section
with an end the table gives no km for is our own trace and is counted in the log.

POINTS are written as
    "Name=km"              an OSM rail station of that name (`skey`), the one nearest the
                           line's previous point, with its km from Colombo Fort
    "Name@lon,lat=km"      that station near the coordinate (within NEAR_M); with none of the
                           name there, the nearest OSM rail station within BLIND_M
    "~Name@lon,lat=km"     a junction, no stop, at the coordinate
    "#ID"                  a border point (the engine's BORDERS; Sri Lanka has none)
(the "=km" is optional). A line may have `more` pieces under the same id: one line drawn in
pieces where the stretch between is another, greyed, line (the cyclone breaks below).
`osm_stops: "all"`: every OSM rail station lying on a traced section is a stop, so a list need
not name every halt.

THE LINE UNIT is SLR's line (Main, Matale, Puttalam, Kelani Valley, Northern, Mannar,
Trincomalee, Batticaloa, Coastal), cut where Cyclone Ditwah's damage (November 2025) still
stops trains: rinf.py greys a whole line or none, so a closed stretch is a line of its own
("Main Line (Gampola - Nanu Oya)"), suspended. lk_sources.md "What runs" has the evidence.

CLIPPING. `--clip <cc>` drops what lies inside ANOTHER country's outline (nafrica_register's
clip, with this module's NOT_SERVICE): Geofabrik's Sri Lanka extract reaches into India's
Rameswaram; Bangladesh's into West Bengal, Assam, Tripura and Meghalaya.
"""
import argparse
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "raw" / "rinf"

NEAR_M = 3000      # "Name@lon,lat": a station of the name this close to the coordinate
BLIND_M = 400      # ... else any OSM rail station this close
MAX_JUMP_KM = 120  # "Name": a station of the name at most this far from the line's last point


# ============================================================== names and keys

def skey(s):
    """One key for a station name: case, accents, "railway station", "Junction", "Halt" and
    punctuation folded away ("Ragama Junction" and "Ragama" are "ragama")."""
    s = unicodedata.normalize("NFKD", (s or "").casefold())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[’'`.]", "", s)
    s = re.sub(r"\b(?:railway|train|rail|station|stn|halt|junction|jn|jct|road halt)\b", " ", s)
    return re.sub(r"[^0-9a-z]", "", s)


def dist_m(lon1, lat1, lon2, lat2):
    dx = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)) * 111320
    return math.hypot(dx, (lat2 - lat1) * 110570)


def L(lid, name, pts, im, more=(), suspended=False, name_en="", colour="", note="",
      listed_only=False):
    return {"id": lid, "name": name, "name_en": name_en, "pts": list(pts),
            "more": [list(m) for m in more], "im": im, "suspended": suspended,
            "colour": colour, "note": note, "listed_only": listed_only}


# ============================================================== Sri Lanka's lines
#
# Each list is the by-line table's rows that carry a km, in order (rows without one are left
# to `osm_stops`). Where a name in OSM differs, the list writes OSM's name and the table's in a
# comment.

SLR = "Sri Lanka Railways"

MAIN_A = [  # Colombo Fort - Gampola: running (Rambukkana - Gampola reopened 2 October 2026)
    "Colombo Fort=0", "Maradana=2.08", "Dematagoda=4.54", "Kelaniya=7.72", "Wanawasala=9.42",
    "Hunupitiya=10.84", "Ederamulla=12.58", "Horape=14.98", "Ragama=16.42", "Walpola=19",
    "Batuwatta=20.08", "Bulugahagoda=21.64", "Ganemulla=23.44", "Yagoda=25.28",
    "Gampaha=28.4", "Daraluwa=30.8", "Bemmulla=32.78", "Magalegoda=35.03",
    "Heendeniya-Pattigoda=36.5", "Veyangoda=38.3", "Wadurawa=40.14", "Keenawala=42.36",
    "Pallewala=44.7", "Ganegoda=46.98", "Wijaya Rajadahana=49.32", "Mirigama=51.04",
    "Wilwatta=52.86", "Botale=54.9", "Ambepussa=56.98", "Yaththalgoda=60.58",
    "Bujjomuwa=62.66", "Alawwa=66.48", "Walakumbura=70.52", "Polgahawela=73.92",
    "Panaliya=77.5", "Tismalpola=79.74", "Korossa=80.5", "Yatagama=82.02",
    "Rambukkana=85.14", "Kadigamuwa=90.14", "Gangoda=94.42", "Ihala Kotte=96.74",
    "Balana=101.66", "Kadugannawa=106.1", "Pilimatalawa=109.94", "Peradeniya=115.34",
    "Gelioya=119.92", "Gampola=127.38",
]
MAIN_B = [  # Gampola - Nanu Oya: closed since Cyclone Ditwah
    "Gampola=127.38", "Tembiligala=131.3", "Ulapane=134.68", "Nawalapitiya=141.64",
    "Inguruoya=147.4", "Galboda=153.12", "Watawala=162.52", "Rozelle=167.98",
    "Hatton=175.98", "Kotagala=180.1", "Talawakele=187.44", "Watagoda=194.2",
    "Great Western=199.3", "Radella=203.16", "Nanu Oya=206.9",
]
MAIN_C = [  # Nanu Oya - Badulla: running (reopened 20 June 2026)
    "Nanu Oya=206.9", "Perakumpura=211", "Ambewela=222.96", "Pattipola=226.78",
    "Ohiya", "Idalgashinna=240.28", "Haputale=247.66", "Diyatalawa=252.68",
    "Bandarawela=258.72", "Kinigama=261", "Heeloya=264.84", "Kital Ella=269.83",
    "Ella=271.03", "Demodera=278.14", "Uduwara=282.98", "Hali Ela=285.92", "Badulla=291.6",
]

LINES = {"lk": [
    L("main", "Main Line", MAIN_A, SLR, colour="#1f4e9c",
      note="Colombo Fort - Gampola; Gampola - Nanu Oya and Nanu Oya - Badulla are lines of "
           "their own until the line runs through again"),
    L("main-badulla", "Main Line (Nanu Oya - Badulla)", MAIN_C, SLR, colour="#1f4e9c"),
    L("main-closed", "Main Line (Gampola - Nanu Oya)", MAIN_B, SLR, suspended=True,
      colour="#1f4e9c", note="closed by Cyclone Ditwah's landslides"),
    L("matale-closed", "Matale Line (Peradeniya - Kandy)",
      ["Peradeniya", "Sarasavi Uyana", "Kandy"], SLR, suspended=True,
      colour="#8a4baf", note="the Kalu Palama bridge at Peradeniya is still being rebuilt"),
    L("matale", "Matale Line",
      ["Kandy=119.5", "Asgiriya=120.66", "Mahaiyawa=121.16", "Katugastota Road=121.74",
       "Mavilmada=123.52", "Katugastota=125.02", "Pallethalawinna=126.38",
       "Udatalawinna=127.52", "Meegammana=129.47", "Yatirawana=130.88", "Wattegama=131.6",
       "Yatawara=133.91", "Pathanpaha=136.08", "Marukona=139.36", "Ukuwela=141.6",
       "Tawalankoya=142.94", "Elwala=144.06", "Kohobiliwala=145.22", "Matale=147.14"],
      SLR, colour="#8a4baf"),
    L("puttalam", "Puttalam Line",
      ["Ragama=16.42", "Kapuwatta", "Ja-Ela=21.04",
       "Thudella=22.8", "Kudahakapola=23.92", "Alawathupitiya=25.38", "Seeduwa=26.92",
       "Liyanagemulla=29.02", "Investment Promotion Zone=30.4", "Katunayake=32.3",
       "Kurana=34.14", "Negombo=37.64", "Kattuwa=40.94", "Kochchikade=43.84",
       "Waikkala=45.42", "Bolawatta=47.64", "Boralessa=50.14", "Lunuwila=53.08",
       "Thummodara=58", "Nattandiya=60.08", "Walahapitiya=63.92", "Kudawewa=66.46",
       "Madampe=70.64", "Kakkapalliya=75.04", "Sawarana=78.24",
       "Chilaw=81.04", "Manuwangama=86.4", "Bangadeniya=89.6", "Arachchikattuwa=92.8",
       "Battuluoya=99.2", "Pulichchakulama=103.7", "Mundalama=108.24",
       "Mangala Eliya=115.14", "Madurankuliya=120", "Palavi=128.62", "Puttalam=133.24"],
      SLR, colour="#e07b00"),
    L("kelani", "Kelani Valley Line",
      ["Maradana=2.08", "Narahenpita",
       "Kirulapona=7.26", "Nugegoda=9.04", "Udahamulla=11.46", "Nawinna=13.14",
       "Maharagama=14.36", "Pannipitiya=16.98", "Kottawa=19.29", "Malapalla=20.46",
       "Homagama=24.46", "Panagoda=25.6", "Meegoda=29.56", "Watareka=31.26",
       "Padukka=35.08", "Arukwatta=36.8", "Angampitiya=37.92", "Pinnawala=39.92",
       "Gammana=41.24", "Morakele=42", "Waga=44.14", "Kadugoda=46.5", "Arapangama=47.5",
       "Kosgama=49.16", "Aluthambalama=51.2", "Miriswatta=51.94", "Hingurala=53.76",
       "Puwakpitiya=55.24", "Kiriwandala=57.24", "Avissawella=58.98"],
      SLR, colour="#2e9e4f"),
    L("northern", "Northern Line",
      ["Polgahawela=73.92", "Girambe=76.86", "Talawattegedara=78.52", "Potuhera=84.99",
       "Nailiya=89.86", "Kurunegala=93.8", "Muttetugala=96.24", "Wellawa=103.42",
       "Ganewatta=115.04", "Nagollagama=127.24",
       "Timbiriyagedara=131.11", "Maho=136.68", "Randenigama=140.02", "Ambanpola=148.3",
       "Galgamuwa=158.28", "Senarathgama=168.3", "Thambuttegama=176.84", "Talawa=187.2",
       "Srawasthipura=196.18", "Anuradhapura New Town=199.82", "Anuradhapura@80.4108,8.3445=202.66",
       "Parasangaswewa=215.4", "Medagama=219.29", "Medawachchiya=228.56", "Poonewa=235.3",
       "Eratperiyakulam=246.7", "Vavuniya=251.9", "Thandikulam=254.9", "Omanthai=264.9",
       "Puliyankulam=275.72", "Mankulam=297.04", "Murikandy=314",
       "Kilinochchi=328", "Paranthan=333.22", "Elephant Pass=340.16", "Pallai=355.06",
       "Eluthumadduval=362.4", "Mirusuvil=367.04", "Kodikamam=369.9", "Meesalai=373.48",
       "Sankathanai=375.42", "Chavakachcheri=377.14", "Thachchanthoppu=382.16",
       "Navatkuli=385.82", "Punkankulam=390.88", "Jaffna=392.9", "Kokuvil=395.16",
       "Kondavil=397.68", "Inuvil=400.01", "Chunnakam=402.2", "Mallakam=404.1",
       "Tellippalai=406.58", "Maviddapuram=408.2", "Kankesanturai=410.3"],
      SLR, colour="#c0392b"),
    L("mannar", "Mannar Line",
      ["Medawachchiya=228.56", "Neriyakulam=241.46", "Cheddikulam=250.25",
       "Madhu Road=271.24", "Murunkan=283.76", "Mathottam=291.23",
       "Thirukketheeswaram=297.23", "Mannar=306.87", "Thoddaweli=313.75", "Pesalai=321.56",
       "Talaimannar=332.36"],
      SLR, colour="#7f8c2b"),
    L("batticaloa", "Batticaloa Line",
      ["Maho=136.68", "Yapahuwa=140.98", "Konwewa=147.14", "Ranamukgama=153.6",
       "Moragollagama=157.98", "Siyambalangamuwa=163.6", "Nagama=169.32", "Avukana=174",
       "Kalawewa=177.2", "Ihalagama=183.2", "Kekirawa=186.22", "Horiwila=197.06",
       "Palugaswewa=201.22", "Habarana=207.94", "Hatares Kotuwa=221.39", "Gal Oya=224.24",
       "Minneriya=234.6", "Hingurakgoda=241.8", "Hathamuna=244.58", "Jayanthipura=247.2",
       "Parakum Uyana=252.5", "Polonnaruwa=257.74",
       "Gallella=262.16", "Manampitiya=267.6", "Sevanapitiya=276", "Welikanda=283.22",
       "Punani=299.2", "Kadadasi Nagar=313.4", "Valaichchenai=316.8", "Kalkudah=319.72",
       "Devapuram=327.76", "Vandaramoolai=331.2", "Eravur=334.74", "Batticaloa=347.26"],
      SLR, colour="#16a0a0"),
    L("trincomalee", "Trincomalee Line",
      ["Gal Oya=224.24", "Aluth Oya=233.86", "Agbopura=249.4", "Kantale=253.78",
       "Ganthalawa=261.2", "Mullipotana=265.84", "Thambalagamuwa=272.4",
       "China Bay=287.54", "Trincomalee=294.08"],
      SLR, colour="#2a7ab0"),
    L("coastal", "Coastal Line",
      ["Colombo Fort=0", "Kompannavidiya", "Kollupitiya=5.22",
       "Bambalapitiya=7.22", "Wellawatta=9.32", "Dehiwala=12.02", "Mount Lavinia=14.21",
       "Ratmalana=16.04", "Angulana=17.96", "Lunawa=19.38", "Moratuwa=20.92",
       "Koralawalla=22.72", "Egoda Uyana=24.58", "Panadura=28.2", "Pinwatta=31.38",
       "Wadduwa=34.36", "Kalutara North=41.72", "Kalutara South=43.76",
       "Katukurunda=46.54", "Payagala North=49.9", "Payagala South=51.14", "Maggona=53",
       "Beruwala=56.16", "Hettimulla=58.16", "Aluthgama=61.36", "Bentota=62.62",
       "Induruwa=66.68", "Maha Induruwa=69.36", "Kosgoda=72.7", "Piyagama=73.96",
       "Ahungalla=75.62", "Patagamgoda=77.5", "Balapitiya=79.82", "Kandegoda=83.06",
       "Ambalangoda=84.72", "Madampagama=87.36", "Kahawe=91.14", "Hikkaduwa=96.66",
       "Kumarakanda=101.06", "Dodanduwa=102.98", "Rathgama=104", "Boossa=107.3",
       "Gintota=109.72", "Piyadigama=111.3", "Richmond Hill=112.82", "Galle=115.42",
       "Katugoda=119.4", "Unawatuna=120.94", "Talpe=125.7", "Habaraduwa=128.7",
       "Kathaluwa=132.38", "Ahangama=135.42", "Midigama=138.84",
       "Kumbalgama=140.8", "Weligama=144.02", "Polwathumodara=147.2", "Mirissa=148.8",
       "Kamburugamuwa=152.52", "Walgama=153.6", "Matara=157.88", "Piladuwa=160.28",
       "Weherahena=162.65", "Kekanadura=165.01",
       "Beliatte=185.17"],
      SLR, colour="#d4a017"),
]}
BORDERS = {}
NOT_SERVICE = {"lk": {}}
LANGS = {"lk": ["en", "si", "ta"]}
ISO3 = {"lk": "LKA"}
# km posts are the register's chainage here; bd has none (its lengths are our traces)
CHAINAGE = {"lk": True}
REGISTRY = {"lk": (LINES, BORDERS, NOT_SERVICE)}


def register_country(cc, lines, borders, not_service, langs, iso3, chainage, path_checks=()):
    """For bd_register.py: add a country to this engine."""
    REGISTRY[cc] = (lines, borders, not_service)
    LANGS[cc], ISO3[cc], CHAINAGE[cc] = langs, iso3, chainage
    PATH_CHECKS[cc] = list(path_checks)


def lines_of(cc):
    return REGISTRY[cc][0][cc]


# ============================================================== conversion

def station_index(stops):
    """rinf.osm_stations' stations under every name tag's key."""
    import rinf
    ost = rinf.osm_stations(stops)
    by_key = {}
    for sid in ost:
        t = stops[sid][0]
        names = {t.get(k) for k in ("name", "name:en", "official_name", "alt_name", "old_name",
                                    "short_name", "int_name") if t.get(k)}
        for n in list(names):
            names |= set(re.split(r"\s*[;/|(),]\s*", n))
        for n in names:
            k = skey(n)
            if k:
                by_key.setdefault(k, set()).add(sid)
    return ost, {k: sorted(v) for k, v in by_key.items()}


def parse(cc, p):
    km = None
    m = re.match(r"^(.*)=([-\d.]+)$", p)
    if m:
        p, km = m.group(1), float(m.group(2))
    at = None
    m = re.match(r"^(.*)@([-\d.]+),([-\d.]+)$", p)
    if m:
        p, at = m.group(1), (float(m.group(2)), float(m.group(3)))
    if p.startswith("#"):
        return "border", p[1:], at or tuple(REGISTRY[cc][1][p[1:]][:2]), km
    if p.startswith("~"):
        return "junction", p[1:], at, km
    return "station", p, at, km


def place(cc, seq, ost, by_key, log, lid, placed_line):
    """Each point of a list as (kind, label, lon, lat, osm id or None, km)."""
    placed = []
    for i, raw in enumerate(seq):
        kind, label, at, km = parse(cc, raw)
        if kind in ("border", "junction"):
            placed.append((kind, label, at[0], at[1], None, km))
            continue
        cs = by_key.get(skey(label), [])
        hit = None
        if at:
            near = sorted((dist_m(ost[c]["lon"], ost[c]["lat"], *at), c) for c in cs)
            near = [x for x in near if x[0] <= NEAR_M]
            if near:
                hit = near[0][1]
            else:
                blind = sorted((dist_m(s["lon"], s["lat"], *at), c) for c, s in ost.items()
                               if abs(s["lon"] - at[0]) < 0.01 and abs(s["lat"] - at[1]) < 0.01)
                if blind and blind[0][0] <= BLIND_M:
                    hit = blind[0][1]
                    log(f"    {lid}: {label} taken as OSM's {ost[hit]['name']!r} "
                        f"({blind[0][0]:.0f} m)")
        elif cs:
            ref = placed[-1][2:4] if placed else None
            if ref is None:
                # a piece's first point: the candidate nearest the next named point's
                for nxt in seq[i + 1:i + 4]:
                    ncs = by_key.get(skey(parse(cc, nxt)[1]), [])
                    if ncs:
                        hit = min(cs, key=lambda c: min(
                            dist_m(ost[c]["lon"], ost[c]["lat"], ost[n]["lon"], ost[n]["lat"])
                            for n in ncs))
                        break
                else:
                    hit = cs[0]
            else:
                d = sorted((dist_m(ost[c]["lon"], ost[c]["lat"], *ref), c) for c in cs)
                if d[0][0] <= MAX_JUMP_KM * 1000:
                    hit = d[0][1]
        if hit is None:
            log(f"    {lid}: no OSM station for {raw!r}")
            continue
        s = ost[hit]
        placed.append(("station", label, s["lon"], s["lat"], hit, km))
        placed_line[raw] = (s["lon"], s["lat"])
    return placed


def convert(cc, log=print, write=True):
    t0 = time.time()
    import build_model as bm
    import rinf
    ways, rels, stops, cid, cx, cy = bm.load(cc, log)
    coords = bm.Coords(cid, cx, cy)
    track = rinf.Track(ways, coords, log)
    ost, by_key = station_index(stops)
    chainage = CHAINAGE.get(cc)

    rows, pts_out, names = [], {}, {}
    stat = Counter()
    for l in lines_of(cc):
        traced_total, k, chain_total = 0.0, 0, 0.0
        placed_line = {}
        for seq in [l["pts"]] + l["more"]:
            placed = place(cc, seq, ost, by_key, log, l["id"], placed_line)
            for a, b in zip(placed[:-1], placed[1:]):
                sa, sb = track.snap(a[2], a[3]), track.snap(b[2], b[3])
                crow = dist_m(a[2], a[3], b[2], b[3]) / 1000
                got = track.trace(sa, sb, crow * 2.5 + 5) if sa and sb else None
                if got:
                    tkm = got[1]
                    stat["sections traced"] += 1
                    if tkm > 2.0 * crow + 3:
                        log(f"    {l['id']}: {a[1]} - {b[1]} traced {tkm:.1f} km for "
                            f"{crow:.1f} km crow-fly: check the list")
                else:
                    tkm = crow * 1.2
                    stat["sections with no trace (crow-fly x 1.2)"] += 1
                    log(f"    {l['id']}: {a[1]} - {b[1]} NO TRACE ({crow:.1f} km crow-fly)")
                km = tkm
                if chainage and a[5] is not None and b[5] is not None:
                    km = abs(b[5] - a[5])
                    stat["sections measured by the register's km"] += 1
                    chain_total += km
                    if abs(tkm - km) > max(1.0, 0.15 * km):
                        log(f"    {l['id']}: {a[1]} - {b[1]}: register {km:.2f} km, traced "
                            f"{tkm:.2f}")
                elif chainage:
                    stat["sections with an end the register gives no km (traced)"] += 1
                traced_total += tkm
                k += 1
                ops = []
                for q in (a, b):
                    if q[0] == "border":
                        op = f"{cc}:b:{q[1]}"
                        pts_out[op] = {"op": op, "uopid": q[1], "type": "90", "name": q[1],
                                       "lon": q[2], "lat": q[3]}
                    elif q[0] == "station":
                        op = f"{cc}:s:{q[4]}"
                        pts_out[op] = {"op": op, "uopid": f"{cc.upper()}{q[4]}", "type": "10",
                                       "name": ost[q[4]]["name"], "lon": q[2], "lat": q[3]}
                    else:
                        jk = re.sub(r"[^0-9a-z]", "", skey(q[1]))[:24] or f"{q[2]:.4f}{q[3]:.4f}"
                        op = f"{cc}:j:{jk}"
                        pts_out[op] = {"op": op, "uopid": f"{cc.upper()}J{jk}", "type": "80",
                                       "name": q[1], "lon": q[2], "lat": q[3]}
                    ops.append(op)
                rows.append({"sol": f"{cc}:{l['id']}:{k}", "line": l["id"], "a": ops[0],
                             "b": ops[1], "len": f"{km:.3f}", "im": l["im"],
                             "label": f"{a[1]} - {b[1]}"})
        names[l["id"]] = {"name": l["name"], "name_en": l["name_en"], "im": l["im"],
                          "suspended": bool(l.get("suspended")),
                          "listed_only": bool(l.get("listed_only")),
                          "colour": l.get("colour", ""),
                          "traced_km": round(traced_total, 3),
                          "register_km": round(chain_total, 3)}
        log(f"  {l['id']:>18} {l['name'][:44]:<44} traced {traced_total:7.1f} km"
            + (f"  register {chain_total:7.1f}" if chainage else "")
            + ("  [suspended]" if l.get("suspended") else ""))
    log(f"{cc.upper()}: {len(lines_of(cc))} lines, {len(rows)} section rows, {len(pts_out)} "
        f"points; {dict(stat)}")
    if write:
        d = OUT / cc
        d.mkdir(parents=True, exist_ok=True)
        stamp = {"endpoint": f"{cc}_register.py (hand-written line list)",
                 "fetched": date.today().isoformat()}
        for fn, obj in (("sections.json", {**stamp, "rows": rows}),
                        ("points.json", {**stamp, "rows": list(pts_out.values())}),
                        ("names.json", names)):
            tmp = d / (fn + ".tmp")
            tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=0 if fn == "names.json"
                                      else None), "utf-8")
            os.replace(tmp, d / fn)
        log(f"wrote {d} in {time.time() - t0:.0f} s")
    return rows, pts_out, names


def build(path, log):
    """build_model's register hook: convert, then rinf.py traces it over OSM track; each line
    takes its colour from the list. A junction- or border-ended section of a running line is
    one its trains run over (`served_sections`), as in nafrica_register."""
    cc = Path(path).name
    convert(cc, log)
    import rinf
    lines, stations, geoms = rinf.build(path, log)
    names = _names(cc)
    n = 0
    for l in lines:
        info = [names.get(x.split("#")[0], {}) for x in l.get("rinf_ids", ())]
        col = next((i.get("colour") for i in info if i.get("colour")), "")
        if col:
            l["colour"] = col
        if l.get("suspended") or any(i.get("suspended") for i in info):
            continue
        keys = [f"{a}|{b}" for a, b, *_ in l["sections"]
                if stations.get(a, {}).get("junction") or stations.get(b, {}).get("junction")]
        if keys:
            l["served_sections"] = keys
            n += len(keys)
    log(f"{cc.upper()}: {n} junction- or border-ended sections of running lines marked served")
    path_checks(cc, lines, stations, log)
    return lines, stations, geoms


# Outside numbers for a path over the built register lines, where no line's own length is
# published: {cc: [(label, (lon, lat), (lon, lat), km, source)]}, each end the built station
# nearest the coordinate. Logged by build() ("path check"), as za_register does.
PATH_CHECKS = {}


def path_checks(cc, lines, stations, log):
    import heapq
    adj = {}
    for ln in lines:
        if ln.get("src", "osm") == "osm":
            continue
        for sec in ln["sections"]:
            adj.setdefault(sec[0], []).append((sec[1], sec[2]))
            adj.setdefault(sec[1], []).append((sec[0], sec[2]))

    def near(pt):
        cs = [(dist_m(s["lon"], s["lat"], *pt), sid) for sid, s in stations.items() if sid in adj]
        return min(cs)[1] if cs else None
    out = []
    for label, a, b, km, note in PATH_CHECKS.get(cc, []):
        s, t = near(a), near(b)
        dist, h, got = {s: 0.0}, [(0.0, s)], None
        while h:
            d, u = heapq.heappop(h)
            if u == t:
                got = d
                break
            if d > dist.get(u, math.inf):
                continue
            for v, w in adj.get(u, ()):
                if d + w < dist.get(v, math.inf):
                    dist[v] = d + w
                    heapq.heappush(h, (d + w, v))
        out.append((label, got, km))
        log(f"path check {label}: {'no path' if got is None else f'{got:.1f} km'} against "
            f"{km:.1f} ({note})" + ("" if got is None else f", ratio {got / km:.3f}"))
    return out


# ============================================================== rinf.py settings

def _names(cc):
    f = OUT / cc / "names.json"
    return json.loads(f.read_text("utf-8")) if f.exists() else {}


def country_conf(cc):
    def id_name(lid, _uop=None):
        e = _names(cc).get(lid.split("#")[0])
        return (e["name"], e.get("name_en") or "") if e else None

    def suspended(_ref, lids):
        ns = _names(cc)
        return any(ns.get(x.split("#")[0], {}).get("suspended") for x in lids)

    def listed_only(lids):
        ns = _names(cc)
        return any(ns.get(x.split("#")[0], {}).get("listed_only") for x in lids)

    ims = {l["im"] for l in REGISTRY[cc][0].get(cc, [])}
    conf = {
        "osm_stops_skip": listed_only,
        "iso3": ISO3[cc], "langs": LANGS[cc],
        "ref": lambda _lid: None,
        "id_name": id_name,
        "suspended": suspended,
        "im": {x: x for x in ims},
        "osm_stops": "all",
        # OSM's route=railway relations here carry no line numbers to lend.
        "osm_rel": lambda _t: None,
        # a station's OSM node is anywhere along its platforms and yard
        "tol_abs": 1.0,
    }
    if not CHAINAGE.get(cc):
        conf["no_chain"] = True    # the lengths are our own traces, not a register's
    return conf


# ============================================================== clipping, tracing

# "Abroad" is shrunk by this much (degrees, ~1.1 km) before clipping: the outlines are
# simplified, and Bangladesh's Akhaura - Shaistaganj line runs within a kilometre of Tripura,
# so a way of it fell "abroad" and the line was cut in two (2026-10-08). Track of the
# neighbour's within the band stays in the extract; ownership leaves it to the neighbour.
CLIP_INSET_DEG = 0.01


def clip(cc, log=print):
    """nafrica_register.clip with this engine's NOT_SERVICE and the neighbours' outlines shrunk
    by CLIP_INSET_DEG."""
    import shapely
    import nafrica_register as nr
    saved = nr.NOT_SERVICE, nr.abroad
    orig = nr.abroad

    def shrunk(c):
        g = orig(c).buffer(-CLIP_INSET_DEG)
        shapely.prepare(g)
        return g
    nr.NOT_SERVICE, nr.abroad = REGISTRY[cc][2], shrunk
    try:
        nr.clip(cc, log)
    finally:
        nr.NOT_SERVICE, nr.abroad = saved


def fill(cc, log=print):
    """nafrica_register.fill over this engine's lists: a "Name@lon,lat" point with no OSM
    station of its name near and no station within BLIND_M becomes a halt at the coordinate
    (Bangladesh's Jashore has no station node). Rewrites data/proc/<cc>/stops.pkl; run after
    --clip."""
    import nafrica_register as nr
    saved = nr.LINES
    nr.LINES = {cc: [dict(l, pts=[re.sub(r"=[-\d.]+$", "", p) for p in l["pts"]],
                          more=[[re.sub(r"=[-\d.]+$", "", p) for p in m] for m in l["more"]])
                     for l in lines_of(cc)]}
    try:
        nr.fill(cc, log)
    finally:
        nr.LINES = saved


def trace_cmd(cc, names, log=print):
    """Trace station to station and print each leg's km and the stations it passes."""
    import build_model as bm
    import rinf
    ways, rels, stops, cid, cx, cy = bm.load(cc, lambda *_: None)
    coords = bm.Coords(cid, cx, cy)
    track = rinf.Track(ways, coords, lambda *_: None)
    ost, by_key = station_index(stops)
    sidx = rinf.StationIndex(ost)
    prev, total = None, 0.0
    for raw in names:
        kind, label, at, _km = parse(cc, raw)
        if kind != "station":
            pt, nm = at, label
        else:
            cs = list(by_key.get(skey(label), []))
            ref = at or prev
            if ref:
                cs.sort(key=lambda c: dist_m(ost[c]["lon"], ost[c]["lat"], *ref))
            if not cs:
                print(f"  ?? no station {label!r}")
                continue
            s = ost[cs[0]]
            pt, nm = (s["lon"], s["lat"]), s["name"]
        if prev:
            got = track.trace(track.snap(*prev), track.snap(*pt),
                              dist_m(*prev, *pt) / 1000 * 2.5 + 5)
            if got is None:
                print(f"  -> {nm}: NO PATH")
            else:
                passed = []
                for x, y in got[0][::3]:
                    for d, sid in sidx.within(x, y, 120):
                        if ost[sid]["name"] not in passed:
                            passed.append(ost[sid]["name"])
                total += got[1]
                print(f"  -> {nm}: {got[1]:.1f} km (total {total:.1f}); via {' / '.join(passed)}")
        else:
            print(f"  {nm} {pt}")
        prev = pt


def stations_cmd(cc, bbox=None):
    """Every OSM rail station of the extract: name, name:en, lon, lat (to write lists)."""
    import build_model as bm
    import rinf
    _w, _r, stops, *_ = bm.load(cc, lambda *_: None)
    ost = rinf.osm_stations(stops)
    for sid, s in sorted(ost.items(), key=lambda kv: (kv[1]["lat"], kv[1]["lon"])):
        print(f"{sid}\t{s['name']}\t{s.get('name_en', '')}\t{s['lon']:.5f}\t{s['lat']:.5f}")


def split_pieces(lines, stations, geoms, reg_ways, state, log):
    """build_model's hook, after drop_unridden_sections: rinf.split_pieces."""
    import rinf
    rinf.split_pieces(lines, stations, geoms, reg_ways, state, log)


def main(default_module=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--convert", metavar="CC")
    ap.add_argument("--dry", metavar="CC", help="convert without writing")
    ap.add_argument("--clip", metavar="CC")
    ap.add_argument("--trace", nargs="+", metavar="ARG")
    ap.add_argument("--stations", metavar="CC")
    ap.add_argument("--fill", metavar="CC")
    a = ap.parse_args()
    t = time.time()
    lg = lambda m: print(f"[{time.time() - t:6.1f}s] {m}", flush=True)  # noqa: E731
    if a.clip:
        clip(a.clip, lg)
    if a.fill:
        fill(a.fill, lg)
    if a.convert:
        convert(a.convert, lg)
    if a.dry:
        convert(a.dry, lg, write=False)
    if a.trace:
        trace_cmd(a.trace[0], a.trace[1:])
    if a.stations:
        stations_cmd(a.stations)


if __name__ == "__main__":
    main()
