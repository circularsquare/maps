"""Georgia (ge), Armenia (am), Azerbaijan (az) and Abkhazia (xa, a user-assigned code like
Kosovo's xk): the CIS tariff guide's sheets for their three railways written into rinf.py's
input format (as ua_register.py does for Ukraine), plus the timetable feeds that decide which
sections trains run over. caucasus_sources.md has the research and the numbers; this
docstring is how it works.

    python caucasus_register.py --esr ge data/raw/georgia-latest.osm.pbf   # (and am, az) before
    #   the .pbf is deleted: ESR codes and every station's names -> data/raw/<cc>/osm_*.json
    #   (Georgia's holds Abkhazia's)
    python caucasus_register.py --outline            # data/raw/ge/ and data/raw/xa/outline.geojson
    python caucasus_register.py --clip ge            # after every extract (and am, az; see CLIP)
    python caucasus_register.py --clip xa            # after Georgia's: data/proc/xa from ge/full
    python caucasus_register.py --timetable          # data/raw/gtfs/<cc>/ (see THE TIMETABLES)
    python caucasus_register.py --convert ge         # -> data/raw/rinf/ge/{sections,points,names}.json
    python build_model.py --region ge --register caucasus_register:data/raw/rinf/ge
    #   (converts, then rinf.py builds; `rinf:data/raw/rinf/ge` builds from the last conversion)
    python caucasus_register.py --colours ge         # colours/ge.csv from names.json

THE REGISTER.  Book 1 of the tariff guide (data/raw/ru/tr4_kniga1_*.xls, the file Russia's and
Ukraine's builds read) has a sheet for each railway: Грз (Georgian Railway, road 57, 25
sections), Ю-Кав (South Caucasus Railway, Armenia, road 56, 14) and Азерб (Azerbaijan
Railways, road 55, 32). Every station, halt and post on a tariff section in order with its ESR
code and integer tariff km; one tariff section is one register line, as in Russia and
Ukraine. A point is placed at the OSM node carrying its ESR code, else at Wikidata's item with
that code (P2815), else at an OSM station whose name (any of name, name:ru, the national
name, old_name) matches near the point's placed neighbours. It is named as OSM names that
station (Georgian, Armenian, Azerbaijani), else by Wikidata's label in that language, else by
Book 1's own Russian spelling. Book 1's extra codes of one station ("Гори (эксп.)" at Gori's
km; the ports' ferry codes) are dropped; the export codes at the two borders trains cross are
the border points (BORDER). Name matching is by a consonant skeleton Russian and Azerbaijani
spellings share (`skeleton`: "Гянджа" and "Gəncə"), so Azerbaijan, where OSM has 2 ESR codes,
places by name; a match is the candidate whose distances from the placed neighbours best fit
the tariff km, and a two-consonant name only within 3 km of that. An ESR node far off its
line (OSM's 568205) is dropped. NAME_AT and POINT_AT pin what names cannot find (Baku's
Soviet names, Alyat, two junctions by Masis); other unnamed posts and loops are placed by
their km between placed neighbours. Unnamed halts ("Платформа 54 км") stay unplaced.

STOPS.  As Ukraine's, except that a call decides first: Book 2 gives most Georgian stations
no passenger operation (Mtskheta, Kobuleti: "1,3"). A point is a stop when a train calls there
(a timetable call or an OSM train route stop within SERVED_M), or it is a halt ("ОП" in Book
1, or an OSM railway=halt within HALT_M) or has Book 2's passenger operation, on a stretch
between called points that has halts. Ukraine's freight-stretch rule is off (STRETCH_KM).

CLIP (`--clip <cc>`, after every extract).  Out of the extract: route relations calling
mostly abroad (Russia's Derbent - Samur train in Azerbaijan's) and an operator's
all-routes route_master (the South Caucasus Railway's, which made one 321 km "line" of every
Armenian train); for Georgia also TERRITORY below. Abkhazia has no extract of its own:
`--clip xa` cuts data/proc/xa out of Georgia's (data/proc/ge/full), what lies within XA_REACH
of Abkhazia, less the New Athos cave railway (SKIP_RELS). Added: a route_master over the two
directions of a train OSM maps without one (Armenia's "Երևան - Արաքս" / "Արաքս - Երևան"), and
one over every numbered train between the same two ends (Georgian Railway's 801-808 are three
masters for one Tbilisi - Batumi service; Azerbaijan's 731-734).

THE TIMETABLES.  Georgia and Armenia: Transitous' generated feeds (jbb.ghsq.de), Georgian
Railway's (October 2026 - January 2027) and the South Caucasus Railway's (June 2026 - June
2027, with the summer trains to Lake Sevan and Batumi). Azerbaijan: ADY publishes no feed and
its sites answer 403 to scripts, so `--timetable` writes one from AZ_TRAINS, a hand list of the
trains running in 2026 by ADY's and the Azerbaijani press's announcements (caucasus_sources.md
lists each with its source), stations by OSM name. Abkhazia: its trains are all Russian
Railways' (FPC from Moscow and St Petersburg, the Dioskuria electric trains from Sochi's
Olympic Park), which publish no feed; `--timetable` writes one from XA_TRAINS, a hand list
from poezdato.net, stations by OSM node. Each country's folder in data/raw/gtfs
holds every feed that runs trains there (the Georgian feed's Baku and Yerevan trains, the
Armenian feed's Batumi train), and gtfs_served reads them as it reads every national feed.

TERRITORY (Anita: de facto, as trains run).  Georgia as drawn is OSM's Georgia (relation
28699) less Abkhazia (1152720) and South Ossetia (1152717). Abkhazia's railway (Psou -
Sukhum - Ochamchira; the track on to the Inguri is gone) is run by its own company, with
Russian trains from Moscow, St Petersburg and Sochi's Olympic Park to Sukhum: it is its own
region, xa (Anita, 2026-10-04: "for abkhazia maybe we should make it its own country?"),
outline OSM's relation 1152720, built from the Abkhazian part of Book 1's 57-001 (TRACK_END:
to Ochamchira, where OSM's track ends; 57-007 to Tkvarcheli has none), its rows under the
Abkhazian Railway (IM_OF), its border with Russia at the Psou bridge (BORDER "574704", the
point eXARUPSOU that Russia's 51-032 ends at too). South Ossetia's (Gori - Tskhinvali) has
had no train since 2008 and is given to no country (`--clip ge` takes its track, stations and
the trains' route relations out of Georgia's extract, and Abkhazia's; Book 1's pairs there are
left out). Karabakh and Nakhchivan are Azerbaijan's.
"""
import argparse
import json
import math
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = RAW / "rinf"
GTFS = RAW / "gtfs"
CCS = ("ge", "am", "az", "xa")
# Book 1's sheets: sheet -> road code; the road's operator, native and English.
SHEETS = {"Грз": "57", "Ю-Кав": "56", "Азерб": "55"}
ROAD = {"57": ("საქართველოს რკინიგზა", "Georgian Railway"),
        "56": ("Հարավկովկասյան երկաթուղի", "South Caucasus Railway"),
        "55": ("Azərbaycan Dəmir Yolları", "Azerbaijan Railways")}
LANG = {"ge": "ka", "am": "hy", "az": "az", "xa": "ab"}
# Abkhazia's track is Book 1's road 57 (the Georgian sheet), run by its own company: its section
# rows carry this manager code instead.
IM_OF = {"xa": "57A"}
ROAD["57A"] = ("Абхазская железная дорога", "Abkhazian Railway")
USER_AGENT = "noritetsu-build/1.0 (hobby rail map)"


def log(msg, t0=time.time()):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def dist_m(lon1, lat1, lon2, lat2):
    dx = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)) * 111320
    return math.hypot(dx, (lat2 - lat1) * 110570)


# ================================================================ the .pbf pass

NAME_TAGS = ("name", "name:ru", "name:en", "name:ka", "name:hy", "name:az", "old_name",
             "official_name", "alt_name", "name:ka-Latn", "name:hy-Latn")


def esr_pass(cc, pbf):
    """From the .pbf (extract.py keeps neither ESR codes nor name:ru): every node with an ESR
    code, and every rail station or halt node with all its names, into data/raw/<cc>/."""
    os.environ.setdefault("OSMIUM_POOL_THREADS", "2")
    import osmium
    esr, st = defaultdict(list), []
    fp = (osmium.FileProcessor(str(pbf), osmium.osm.NODE)
          .with_filter(osmium.filter.KeyFilter("esr:user", "esr", "railway:esr", "railway",
                                               "public_transport")))
    for obj in fp:
        t = obj.tags
        code = t.get("esr:user") or t.get("esr") or t.get("railway:esr")
        rw = t.get("railway")
        is_st = rw in ("station", "halt") or (t.get("public_transport") == "station"
                                              and t.get("train") == "yes")
        if not code and not is_st:
            continue
        lon, lat = round(obj.location.lon, 6), round(obj.location.lat, 6)
        row = {"id": obj.id, "lon": lon, "lat": lat, "rw": rw,
               "pt": t.get("public_transport"), "train": t.get("train"),
               "station": t.get("station") or "",
               "usage": t.get("usage") or t.get("disused:railway") or "",
               "names": {k: t.get(k) for k in NAME_TAGS if t.get(k)}}
        if code:
            row["esr"] = code
            for c in code.replace(",", ";").split(";"):
                c = c.strip()
                if c.isdigit() and len(c) == 6:
                    esr[c].append(row)
        if is_st:
            st.append(row)
    d = RAW / cc
    d.mkdir(parents=True, exist_ok=True)
    (d / "osm_esr.json").write_text(json.dumps(esr, ensure_ascii=False), "utf-8")
    (d / "osm_stations.json").write_text(json.dumps(st, ensure_ascii=False), "utf-8")
    log(f"--esr {cc}: {sum(len(v) for v in esr.values())} nodes with {len(esr)} ESR codes; "
        f"{len(st)} station and halt nodes -> {d}")


def osm_station_rows():
    """Every OSM station and halt node the three --esr passes found (a border station can be
    in the neighbour's extract), one row per node."""
    rows = {}
    for cc in CCS:
        p = RAW / cc / "osm_stations.json"
        if p.exists():
            for r in json.loads(p.read_text("utf-8")):
                rows[r["id"]] = r
    return list(rows.values())


def osm_esr():
    out = defaultdict(dict)
    for cc in CCS:
        p = RAW / cc / "osm_esr.json"
        if p.exists():
            for c, rs in json.loads(p.read_text("utf-8")).items():
                for r in rs:
                    out[c][r["id"]] = r
    return {c: list(v.values()) for c, v in out.items()}


# ================================================================ territory

def shapes():
    """{"ge", "am", "az", "xa": the country as drawn, "none": South Ossetia}, from OSM's
    boundaries (polygons.openstreetmap.fr, relations 28699, 364066, 364110, 1152720 for
    Abkhazia, 1152717 for South Ossetia; data/raw/<cc>/*_boundary.geojson)."""
    from shapely.geometry import shape

    def rd(p):
        return shape(json.loads((RAW / p).read_text("utf-8")))
    ab, so = rd("ge/abkhazia_boundary.geojson"), rd("ge/south_ossetia_boundary.geojson")
    return {"ge": rd("ge/ge_boundary.geojson").difference(ab.union(so)),
            "am": rd("am/am_boundary.geojson"), "az": rd("az/az_boundary.geojson"),
            "xa": ab, "none": so}


OUTLINE_WHAT = {
    "ge": "Georgia as noritetsu draws it: OSM relation 28699 less Abkhazia (1152720) and South "
          "Ossetia (1152717)",
    "xa": "Abkhazia as noritetsu draws it: OSM relation 1152720"}


def write_outline():
    """data/raw/ge/outline.geojson and data/raw/xa/outline.geojson: Georgia and Abkhazia as
    drawn, for the app's outline (tools/build_regions.py OUTLINE, as Ukraine's)."""
    from shapely.geometry import mapping
    shp_all = shapes()
    for cc, what in OUTLINE_WHAT.items():
        shp = shp_all[cc]
        (RAW / cc).mkdir(parents=True, exist_ok=True)
        (RAW / cc / "outline.geojson").write_text(json.dumps({
            "type": "FeatureCollection", "features": [{"type": "Feature", "properties": {
                "what": what}, "geometry": mapping(shp)}]}), "utf-8")
        log(f"--outline: {shp.area:.3f} sq deg -> data/raw/{cc}/outline.geojson")


MASTER_BASE = 9_000_000_000   # synthetic route_master ids: this + the lower route id
# Degrees: Abkhazia's clip keeps this much beyond its boundary (about 1 km), so the track over
# the Psou bridge reaches the border point.
XA_REACH = 0.01
# Route relations left out of a clipped extract.
SKIP_RELS = {"xa": {
    # The New Athos cave railway (route=subway "Афон Ҿыц аҳаԥытә метро"): a ride inside the
    # cave that is part of the cave tour, not transport between places; left out.
    16248679: "New Athos cave railway"}}


def clip(cc):
    """data/proc/<cc> (the module docstring's CLIP). For Georgia, Abkhazia's and South
    Ossetia's track, stations and the route relations of the trains there are taken out (de
    facto: no country's; see TERRITORY): a way goes if half its nodes are inside, a stop if it
    is, a route relation if half its members the extract holds went. The extract as extract.py
    wrote it is kept in data/proc/<cc>/full, so this can be rerun.

    Abkhazia (xa) has no extract of its own: Geofabrik's Georgia holds it, so data/proc/xa is
    cut from data/proc/ge/full, keeping what lies within XA_REACH of Abkhazia (the reach takes
    the Psou bridge to the border point; Geofabrik's buffer beyond Psou is dropped)."""
    import numpy as np
    import shapely
    proc = ROOT / "data" / "proc" / cc
    full = proc / "full" if cc != "xa" else ROOT / "data" / "proc" / "ge" / "full"
    proc.mkdir(parents=True, exist_ok=True)
    names = ("ways.pkl", "rels.pkl", "stops.pkl", "infra.pkl", "coords.npz")
    stamp = proc / "clip_stamp.json"
    fresh = cc != "xa" and (proc / "ways.pkl").exists() and (
        not stamp.exists() or (proc / "ways.pkl").stat().st_mtime > stamp.stat().st_mtime + 5)
    if fresh:
        full.mkdir(exist_ok=True)
        for fn in names:
            if (proc / fn).exists():
                os.replace(proc / fn, full / fn)
        log(f"clip: the extract in data/proc/{cc} moved to data/proc/{cc}/full")

    def rd(fn):
        with open(full / fn, "rb") as f:
            return pickle.load(f)
    ways, rels, stops = rd("ways.pkl"), rd("rels.pkl"), rd("stops.pkl")
    infra = rd("infra.pkl") if (full / "infra.pkl").exists() else {}
    with np.load(full / "coords.npz") as c:
        cid, cx, cy = c["id"], c["x"], c["y"]
    shp = shapes()
    # what goes: for Georgia, Abkhazia (its own region) and South Ossetia (no region's); for
    # Abkhazia, everything beyond XA_REACH of it
    if cc == "ge":
        none = shp["none"].union(shp["xa"])
    elif cc == "xa":
        none = shapely.box(-180, -90, 180, 90).difference(shp["xa"].buffer(XA_REACH))
    else:
        none = shapely.Polygon()
    shapely.prepare(none)
    home = shp[cc].buffer(0.02)
    land = shp[cc]
    shapely.prepare(home)
    shapely.prepare(land)
    inside = set(cid[shapely.contains_xy(none, cx / 1e7, cy / 1e7)].tolist())
    xy_of = dict(zip(cid.tolist(), zip((cx / 1e7).tolist(), (cy / 1e7).tolist())))
    keep_w, gone_w = {}, set()
    for wid, (tags, nodes) in ways.items():
        ns = [int(n) for n in nodes]
        if ns and 2 * sum(n in inside for n in ns) >= len(ns):
            gone_w.add(wid)
        else:
            keep_w[wid] = (tags, nodes)
    keep_s = {k: v for k, v in stops.items()
              if not shapely.contains_xy(none, v[1], v[2])}
    gone_s = set(stops) - set(keep_s)
    gone_r = {r for r in SKIP_RELS.get(cc, {}) if r in rels}
    for rid in gone_r:       # and their track (the cave railway's is in no other route)
        for t, r, _ in rels[rid][1]:
            if t == "w" and r in keep_w:
                del keep_w[r]
                gone_w.add(r)
    if cc == "xa":           # the cave railway's sidings and yard (usage=tourism), too
        for wid in [w for w, (t, _n) in keep_w.items() if t.get("usage") == "tourism"]:
            del keep_w[wid]
            gone_w.add(wid)
    for rid, (tags, members) in rels.items():
        if rid in gone_r:
            continue
        # members the extract holds: the Moscow - Sukhum train's 1,200 Russian ways are not in
        # it, only Geofabrik's buffer beyond Psou
        wm = [r for t, r, _ in members if t == "w" and r in ways]
        nm = [r for t, r, _ in members if t == "n" and r in stops]
        lost = sum(r in gone_w for r in wm) + sum(r in gone_s for r in nm)
        if (wm or nm) and 2 * lost >= len(wm) + len(nm):
            gone_r.add(rid)
        # a train calling only abroad, in Geofabrik's buffer (Russia's Derbent - Samur border
        # electric train in Azerbaijan's extract)
        elif nm and 2 * sum(1 for r in nm if shapely.contains_xy(land, stops[r][1], stops[r][2])) \
                < len(nm):
            gone_r.add(rid)
        elif not nm and wm and tags.get("type") == "route" and 2 * sum(
                1 for r in wm if any(int(n) in xy_of and shapely.contains_xy(home, *xy_of[int(n)])
                                     for n in ways[r][1][:1])) < len(wm):
            gone_r.add(rid)
        # a route_master that is an operator's list of all its routes, not a line (the South
        # Caucasus Railway's, named as the company: it made one 321 km "line" of the Araks,
        # Yeraskh, Gyumri and Shorzha trains)
        elif tags.get("type") == "route_master" and tags.get("name") and \
                tags.get("name") == tags.get("operator"):
            gone_r.add(rid)
    keep_r = {k: v for k, v in rels.items() if k not in gone_r
              and not (v[1] and all(t == "r" and r in gone_r for t, r, _ in v[1]))}
    if cc == "xa":
        # a route_master none of whose routes is kept (Georgia's Borjomi - Bakuriani, whose
        # routes the extract does not hold)
        keep_r = {k: v for k, v in keep_r.items() if v[0].get("type") != "route_master"
                  or any(t == "r" and r in keep_r for t, r, _ in v[1])}
    keep_i = {k: v for k, v in infra.items()
              if not any(t == "w" and r in gone_w for t, r, _ in v[1])
              or any(t == "w" and r in keep_w for t, r, _ in v[1])}
    for rid in sorted(gone_r):
        log(f"  route relation out: {rid} {rels[rid][0].get('name', '')}")
    # One line for a train's two directions: OSM Armenia maps "Երևան - Արաքս" and "Արաքս -
    # Երևան" as two routes with no route_master (tr_register.pair_directions' recipe; the
    # master's id is MASTER_BASE + the lower route id, so the line id is stable).
    in_master = {r for t, ms in keep_r.values() if t.get("type") == "route_master"
                 for ty, r, _ in ms if ty == "r"}

    def pkey(name):
        m = re.match(r"^\s*(\S.*?)\s+-\s+(\S+)(.*)$", name or "")
        return (m.group(1), m.group(2), m.group(3).strip()) if m else None
    by = {}
    for k, (t, ms) in keep_r.items():
        if t.get("type") == "route" and t.get("route") == "train" and k not in in_master:
            kk = pkey(t.get("name"))
            if kk:
                by[kk] = k
    made = 0
    for (a, b, suf), k in sorted(by.items(), key=lambda kv: kv[1]):
        o = by.get((b, a, suf))
        if o is None or o < k:
            continue
        t = keep_r[k][0]
        keep_r[MASTER_BASE + k] = ({"type": "route_master", "route_master": "train",
                                    "name": t.get("name"), "operator": t.get("operator", ""),
                                    "network": t.get("network", "")},
                                   [("r", k, ""), ("r", o, "")])
        made += 1
    log(f"  {made} two-direction trains given a route_master")
    # Georgian Railway's trains, one line per pair of ends: OSM maps each train pair with its
    # own route_master ("მატარებელი #802/801: ბათუმი → თბილისი → ბათუმი", the same for 804/803
    # and 808/807), which made three Tbilisi - Batumi lines over one corridor. Routes named
    # "#<number>: A → B" are grouped by their two ends (and operator) under one route_master
    # (MASTER_BASE + the lowest route id) whose ref lists the numbers; the trains' own masters
    # go.
    pat = re.compile(r"[#№]\s*(\d+)\s*:\s*(.+?)\s*→\s*(.+?)\s*$")
    groups = defaultdict(list)
    for k, (t, ms) in keep_r.items():
        if t.get("type") == "route" and t.get("route") == "train":
            m = pat.search(t.get("name") or "")
            if m and "/" not in m.group(3):
                groups[(frozenset((m.group(2), m.group(3))), t.get("operator", ""))].append(
                    (int(m.group(1)), k, m.group(2), m.group(3)))
    n_grp = 0
    for (_ends, op), rs in groups.items():
        nums = sorted({n for n, *_r in rs})
        if len(nums) < 2:
            continue
        rs.sort()
        ids = {k for _n, k, _a, _b in rs}
        for mk, (t, ms) in list(keep_r.items()):
            if t.get("type") == "route_master" and any(ty == "r" and r in ids for ty, r, _ in ms):
                del keep_r[mk]
        _n, k0, a, b = rs[0]
        tags = {"type": "route_master", "route_master": "train", "name": f"{a} — {b}",
                "ref": "/".join(str(n) for n in nums), "operator": op}
        m_en = pat.search(keep_r[k0][0].get("name:en") or "")
        if m_en and "/" not in m_en.group(3):
            tags["name:en"] = f"{m_en.group(2)} — {m_en.group(3)}"
        keep_r[MASTER_BASE + min(ids)] = (tags, [("r", k, "") for k in sorted(ids)])
        n_grp += 1
    log(f"  {n_grp} lines made of trains between the same two ends")
    log(f"clip {cc}: kept {len(keep_w)}/{len(ways)} ways, {len(keep_s)}/{len(stops)} stops, "
        f"{len(keep_r)}/{len(rels)} relations, {len(keep_i)}/{len(infra)} infra relations")
    for fn, obj in (("ways.pkl", keep_w), ("rels.pkl", keep_r), ("stops.pkl", keep_s),
                    ("infra.pkl", keep_i)):
        tmp = proc / (fn + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        os.replace(tmp, proc / fn)
    tmp = proc / "coords.tmp.npz"
    np.savez_compressed(tmp, id=cid, x=cx, y=cy)
    os.replace(tmp, proc / "coords.npz")
    stamp.write_text(json.dumps({"clipped_from": str(full), "ways": len(keep_w)}), "utf-8")


# ================================================================ names and keys

GENERIC = {"станция", "ст", "рзд", "разъезд", "оп", "пл", "платформа", "пост", "бп",
           "stansiyası", "stansiyasi", "stansiyas", "stansiya", "dəmiryol", "demiryol",
           "sərnişin", "st", "station", "railway", "halt", "կայարան", "կայարանը",
           "სადგური", "rzd"}
_CYR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "i", "k", "l", "m", "n", "o",
                 "p", "r", "s", "t", "u", "f", "h", "ts", "ch", "sh", "sh", "", "i", "", "e",
                 "u", "a"]))
_AZ = {"ə": "a", "ı": "i", "ş": "sh", "ç": "ch", "x": "h", "q": "g", "ğ": "g", "ö": "o",
       "ü": "u", "c": "j", "ǝ": "a", "y": "i", "w": "v"}


def _latin(s):
    """Russian Cyrillic or Azerbaijani/English Latin, to one rough Latin."""
    s = (s or "").casefold().replace("дж", "j").replace("dzh", "j").replace("dj", "j")
    s = s.replace("kh", "h").replace("zh", "j").replace("ch", "ch")
    out = []
    for ch in s:
        if ch in _CYR:
            out.append(_CYR[ch])
        elif ch in _AZ:
            out.append(_AZ[ch])
        else:
            out.append(ch)
    return "".join(out).replace("zh", "j")


def _words(name):
    s = re.sub(r"\(.*?\)", " ", name or "")
    s = re.sub(r"^(?:ОП|О\.П\.)\s+", "", s.strip())
    toks = [t for t in re.split(r"[\s\-–—.,/\\№;:]+", s) if t]
    return [t for t in toks if t.casefold() not in GENERIC]


SKEL = [("sh", "S"), ("ch", "S"), ("ts", "s"), ("j", "S"), ("z", "s"), ("k", "K"), ("g", "K"),
        ("h", "K"), ("q", "K"), ("x", "K"), ("d", "T"), ("t", "T"), ("b", "P"), ("p", "P"),
        ("v", "F"), ("f", "F"), ("s", "s")]


def skeleton(name, words=None):
    """A consonant skeleton one place's Russian and Azerbaijani (or English) spellings share:
    "Гянджа" and "Gəncə" are KnS, "Евлах" and "Yevlax Stansiyası" lK, "Гаджигабул" and
    "Hacıqabul" KSKPl. v and f are left out: Russian writes Azerbaijani "ov" and "əv" as a
    vowel ("Тауз" Tovuz, "Мингечаур" Mingəçevir)."""
    s = _latin(" ".join(_words(name) if words is None else words))
    s = re.sub(r"[^a-z0-9]", "", s)
    for a, b in SKEL:
        s = s.replace(a, b)
    s = re.sub(r"[aeiouyFvf]", "", s)
    s = re.sub(r"(.)\1+", r"\1", s)
    return s


def name_keys(name):
    """Strongest first: the whole name folded (rinf.norm), its skeleton, then each word's
    skeleton ("Şəmkir Təzəkənd Stansiyası" is Shamkir's station; "Baş Ələt" Alyat's)."""
    from rinf import norm
    out = []
    w = _words(name)
    full = norm(" ".join(w)) if w else ""
    sk = skeleton(name)
    for k in (full, "~" + sk if len(sk) >= 2 else ""):
        if k and k not in out:
            out.append(k)
    if len(w) > 1:
        for x in w:
            sx = skeleton(None, [x])
            if len(x) >= 4 and len(sx) >= 2 and "~" + sx not in out:
                out.append("~" + sx)
    return out


def clean_native(name):
    """OSM's station name less its station words: "Yevlax Stansiyası" -> "Yevlax", "Bakı
    Dəmiryol Stansiyası" -> "Bakı", "Անուշավան կայարան" -> "Անուշավան"."""
    s = re.sub(r"\s*\\.*$", "", name or "").strip()
    s = re.sub(r"\s+(?:Sərnişin\s+)?(?:Dəmiryol\s+)?(?:Stansiyas[ıi;]?|stansiyası|"
               r"Stansiyasi|Stansiyas;|Qəsəbə Stansiyası|Yük stansiyası)\s*$", "", s)
    s = re.sub(r"^Stansiya\s+", "", s)
    s = re.sub(r"\s+(?:Qəsəbə)$", "", s)
    s = re.sub(r"\s+(?:երկաթուղային\s+)?կայարան$", "", s)
    s = re.sub(r"\s+სადგური$", "", s)
    return s.strip() or (name or "")


class NameIndex:
    def __init__(self):
        self.by = defaultdict(list)

    def add(self, name, lon, lat, ref):
        for rank, k in enumerate(name_keys(name)):
            self.by[k].append((lon, lat, ref, rank))

    def find(self, name):
        out = []
        for qr, k in enumerate(name_keys(name)):
            out += [(x, y, ref, max(qr, r)) for x, y, ref, r in self.by.get(k, [])]
        if not out:
            return []
        best = min(r for *_x, r in out)
        seen, res = set(), []
        for x in out:
            if x[3] <= best and (x[0], x[1]) not in seen:
                seen.add((x[0], x[1]))
                res.append(x)
        return res


# ================================================================ the timetables

# Azerbaijan's trains in 2026 (caucasus_sources.md has each one's source), by OSM station
# name (a name OSM gives twice takes the place nearest the coordinate after "@"). Days per
# week. The Baku - Tbilisi train is in Georgian Railway's feed, which is copied in too.
AZ_TRAINS = [
    ("Bakı - Qazax", 7, 2, ["Bakı Dəmiryol Stansiyası", "Kürdǝmir Stansiyası", "Ucar Stansiyası",
                            "Ləki Stansiyası", "Yevlax Stansiyası", "Gəncə",
                            "Ağstafa Stansiyası", "Qazax Stansiyası"]),
    ("Bakı - Balakən", 7, 1, ["Bakı Dəmiryol Stansiyası", "Ucar Stansiyası", "Ləki Stansiyası",
                              "Yevlax Stansiyası", "Xanabad Stansiyası", "Şəki stansiyası",
                              "Qax Stansiyası", "Balakən stansiyası"]),
    ("Bakı - Qəbələ", 2, 2, ["Bakı Dəmiryol Stansiyası", "Ucar Stansiyası", "Ləki Stansiyası",
                             "Qəbələ Stansiyası"]),
    ("Gəncə - Qəbələ", 7, 1, ["Gəncə", "Yevlax Stansiyası", "Ləki Stansiyası",
                              "Qəbələ Stansiyası"]),
    ("Bakı - Ağdam", 1, 1, ["Bakı Dəmiryol Stansiyası", "Binəqədi", "Ucar Stansiyası",
                            "Ləki Stansiyası", "Yevlax Stansiyası", "Bərdə", "Köçərli",
                            "Təzəkənd", "Ağdam"]),
    ("Gəncə - Mingəçevir", 7, 2, ["Gəncə", "Mingəçevir Stansiyası@47.0028,40.6371"]),
    ("Gəncə - Ağstafa", 7, 2, ["Gəncə", "Ağstafa Stansiyası"]),
    ("Abşeron dairəvi dəmir yolu", 7, 40, [
        "Bakı Dəmiryol Stansiyası", "Keşlə Stansiyası", "Koroğlu@49.9196,40.4193",
        "Bakıxanov Stansiyası@49.9537,40.4320", "Sabunçu Stansiyası", "Zabrat 1 Stansiyası",
        "Zabrat 2 Stansiyası", "Məmmədli Stansiyası", "Pirşağı Stansiyası", "Goradil Stansiyası",
        "Novxanı Stansiyası", "Sumqayıt", "Xırdalan Stansiyası", "Biləcəri Qəsəbə Stansiyası",
        "Dərnəgül stansiyası", "Bakı Dəmiryol Stansiyası"]),
    ("Bakı - Xırdalan", 7, 20, ["Bakı Dəmiryol Stansiyası", "Dərnəgül stansiyası",
                                "Biləcəri Qəsəbə Stansiyası", "Xırdalan Stansiyası"]),
]
AZ_AGENCY = "Azərbaycan Dəmir Yolları"
# Abkhazia's trains (2026, poezdato.net's Sukhum station page and train pages, 2026-10-04): all
# Russian Railways', from Russia; their calls in Abkhazia only, by OSM node ("#<id>"; the
# names are Abkhaz). Days per week: the Dioskuria 929/930 runs daily; the others "по особому
# графику" (on set dates, not daily), taken as three days a week, which only says they run
# more often than weekly. None runs beyond Guma.
XA_PSOU_SIDE = ["#4857939998"]                                   # Tsandrypsh (border control)
XA_TRAINS = [
    ("304М/304С Москва — Сухум", 3, 1, XA_PSOU_SIDE + [
        "#941843160", "#506825034", "#506660194", "#504895671"]),  # Gagra, Gudauta, New Athos, Sukhum
    ("479А/480С Санкт-Петербург — Сухум", 3, 1, XA_PSOU_SIDE + [
        "#941843160", "#506825034", "#506660194", "#504895671"]),
    ("929С/930Ж «Диоскурия» Олимпийский парк — Сухум", 7, 1, XA_PSOU_SIDE + [
        "#941690746", "#506420760", "#941843160", "#506487924",    # Abaata, Gagripsh, Gagra, Bzypta
        "#506825034", "#506661455", "#506660194", "#504895671"]),  # Gudauta, Psyrtskha, New Athos, Sukhum
    ("925Э/926Й «Диоскурия» Олимпийский парк — Гума", 3, 1, XA_PSOU_SIDE + [
        "#941690746", "#506420760", "#941843160", "#506487924",
        "#506825034", "#506661455", "#506660194", "#504895671", "#790332148"]),   # ... Guma
]
HAND_FEEDS = {
    "az": (AZ_TRAINS, AZ_AGENCY, "https://ady.az/", "Asia/Baku", "az_ady_handlist.gtfs.zip",
           "noritetsu, a hand list of ADY's 2026 trains (caucasus_register.AZ_TRAINS)", "az"),
    "xa": (XA_TRAINS, "Российские железные дороги (ФПК, ДОСС)", "https://www.rzd.ru/",
           "Europe/Moscow", "xa_handlist.gtfs.zip",
           "noritetsu, a hand list of the 2026 trains in Abkhazia (caucasus_register.XA_TRAINS)",
           "ru"),
}
# Feeds copied into each country's timetable folder: every feed with trains there.
FEED_FILES = {"ge": [("ge", "ge-georgian-railway.gtfs.zip"), ("am", "am-railway.gtfs.zip")],
              "am": [("am", "am-railway.gtfs.zip"), ("ge", "ge-georgian-railway.gtfs.zip")],
              "az": [("ge", "ge-georgian-railway.gtfs.zip")]}


def az_feed(cc="az"):
    """data/raw/gtfs/az/az_ady_handlist.gtfs.zip from AZ_TRAINS (and Abkhazia's from
    XA_TRAINS; HAND_FEEDS): one trip each way per train a day, on the train's days of a 28-day
    window from today (times are not modelled). A call is an OSM station name, or "#<node>"."""
    trains, agency, url, tz, fname, publisher, lang = HAND_FEEDS[cc]
    import csv
    import io
    import zipfile
    from datetime import timedelta
    by_name = defaultdict(list)
    for r in osm_station_rows():
        if r["names"].get("name"):
            by_name[r["names"]["name"]].append(r)
        by_name[f"#{r['id']}"].append(r)
    stops, routes, trips, st, cal = {}, [], [], [], []
    d0 = date.today()
    for i, (name, per_week, per_day, calls) in enumerate(trains):
        ids = []
        for c in calls:
            nm, at = (c.split("@") + [None])[:2]
            cands = by_name.get(nm) or []
            if not cands:
                raise SystemExit(f"{cc} hand feed: no OSM station named {nm!r}")
            if at:
                x, y = map(float, at.split(","))
                r = min(cands, key=lambda r: dist_m(r["lon"], r["lat"], x, y))
            else:
                if len({(round(r["lon"], 2), round(r["lat"], 2)) for r in cands}) > 1:
                    raise SystemExit(f"{cc} hand feed: {nm!r} names several places; add @lon,lat")
                r = cands[0]
            sid = f"n{r['id']}"
            stops[sid] = (clean_native(r["names"].get("name") or nm), r["lat"], r["lon"])
            ids.append(sid)
        rid = f"{cc}{i}"
        routes.append((rid, "ady", name, "2"))
        days = [d0 + timedelta(days=k) for k in range(28)
                if (k % 7) < per_week]
        for direction, seq in ((0, ids), (1, ids[::-1])):
            for n in range(per_day):
                tid = f"{rid}_{direction}_{n}"
                trips.append((rid, tid, tid))
                for j, s in enumerate(seq, 1):
                    st.append((tid, "", "", s, j))
                for d in days:
                    cal.append((tid, d.strftime("%Y%m%d"), 1))

    def tbl(header, rows):
        b = io.StringIO()
        w = csv.writer(b, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
        return b.getvalue()
    d = GTFS / cc
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / (fname + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("agency.txt", tbl(("agency_id", "agency_name", "agency_url",
                                      "agency_timezone"),
                                     [("ady", agency, url, tz)]))
        z.writestr("stops.txt", tbl(("stop_id", "stop_name", "stop_lat", "stop_lon"),
                                    [(k, n, f"{y:.6f}", f"{x:.6f}")
                                     for k, (n, y, x) in sorted(stops.items())]))
        z.writestr("routes.txt", tbl(("route_id", "agency_id", "route_long_name", "route_type"),
                                     routes))
        z.writestr("trips.txt", tbl(("route_id", "service_id", "trip_id"), trips))
        z.writestr("stop_times.txt", tbl(("trip_id", "arrival_time", "departure_time",
                                          "stop_id", "stop_sequence"), st))
        z.writestr("calendar_dates.txt", tbl(("service_id", "date", "exception_type"), cal))
        z.writestr("feed_info.txt", tbl(
            ("feed_publisher_name", "feed_lang", "feed_version"),
            [(publisher, lang, d0.isoformat())]))
    os.replace(tmp, d / fname)
    log(f"--timetable {cc}: {len(trains)} trains, {len(stops)} stations -> {d}")


def timetable():
    """Each country's timetable folder (data/raw/gtfs/<cc>/): the feeds downloaded into
    data/raw/<cc>/ (Transitous' jbb.ghsq.de copies) and, for Azerbaijan, the hand list."""
    import shutil
    for cc, files in FEED_FILES.items():
        d = GTFS / cc
        d.mkdir(parents=True, exist_ok=True)
        for src_cc, fn in files:
            shutil.copyfile(RAW / src_cc / fn, d / fn)
        log(f"--timetable {cc}: {', '.join(fn for _c, fn in files)}")
    az_feed("az")
    az_feed("xa")


def feed_calls():
    """Every station a rail trip with running days calls at in the three countries' feeds
    [(lon, lat)], and consecutive calls of one trip as index pairs into that list."""
    import gtfs_served as gs
    gs.WRITE_CACHE = False
    xy, pairs = {}, set()
    for cc in CCS:
        for p in sorted((GTFS / cc).glob("*.zip")) if (GTFS / cc).is_dir() else []:
            st, pats, _info = gs.read_feed(p, "", (), lambda *_a: None)
            for seq, _n, days in pats:
                if not days:
                    continue
                pts = []
                for s in seq:
                    if s in st and st[s][1] and abs(st[s][2]) > 0.1:
                        k = (round(st[s][1], 6), round(st[s][2], 6))
                        xy.setdefault(k, len(xy))
                        pts.append(xy[k])
                for a, b in zip(pts, pts[1:]):
                    if a != b:
                        pairs.add((min(a, b), max(a, b)))
    return sorted(xy, key=xy.get), pairs


# ================================================================ the register

SERVED_M = 400            # a timetable call or an OSM train route stop this close serves a point
NAME_REACH_KM = 10        # a name match may lie this much further than the tariff km say
# Ukraine's freight-stretch rule (a called-to-called stretch this long with no halt answers to
# the timetable as junction-ended track) is off: with these small feeds it cut the main line
# at Mingachevir and Geran, and the Baku - Gazakh trains found no path from Yevlakh to Ganja.
# A stretch no train runs over is greyed instead.
STRETCH_KM = 10 ** 6
HALT_M = 400              # an OSM railway=halt this close to a point marks it a halt
# Book 1's codes of one station besides its own: export, transfer, junction, ferry and port,
# terminal codes. Left out, except BORDER's.
EXTRA = re.compile(r"\((?:\s*эксп|перев|стык|паром|\s*порт|терминал|на терминал)", re.I)
# The two crossings passenger trains take, by Book 1's export codes on each side: the point of
# each is where OSM's track crosses the boundary (OSM ways named in caucasus_sources.md).
# Sadakhlo - Ayrum: Yerevan - Tbilisi (daily) and Yerevan - Batumi (summer); Gardabani - Böyük
# Kəsik: Baku - Tbilisi (daily since 26 May 2026).
# Psou: Book 1's Abkhazian sheet starts at "Гантиади (эксп.)" (574704, km 0), the handover to
# Russia's 51-032, which ends at "Веселое (эксп.)" 2 km past Veseloe: the Psou bridge. OSM
# carries 574704 on Psou halt, 400 m east of the bridge; no train stops there (they stop for
# the border at Tsandrypsh and Veseloe), so the code is the border point itself.
BORDER = {"564204": "XAMGE1", "569706": "XAMGE1", "563606": "XAZGE1", "558701": "XAZGE1",
          "574704": "XARUPSOU"}
BORDER_XY = {
    # the Debed bridge: OSM node where ways 48754465 / 1103672429 cross the boundary
    "XAMGE1": (44.898762, 41.211857, ["am", "ge"], "Armenia – Georgia border"),
    # ways 453284165 and 1186547878 (two tracks 3 m apart) over the boundary
    "XAZGE1": (45.165160, 41.405885, ["az", "ge"], "Azerbaijan – Georgia border"),
    # the Psou bridge: where way 124797554 (the only track over the river) crosses OSM's
    # boundary of Abkhazia (relation 1152720). Moscow, St Petersburg and Dioskuria trains.
    "XARUPSOU": (40.008443, 43.393733, ["ru", "xa"], "Abkhazia – Russia border")}
# A border code named as the station at the crossing rather than the one Book 1's name gives:
# "Гантиади (эксп.)" is at Psou (OSM node 2115569917), not Gantiadi (Tsandrypsh), 9 km on.
BORDER_STATION_NODE = {"574704": 2115569917}
TYPE_RANK = {"Основной тарифный участок": 0, "Кольцевая линия": 1}
# Points whose OSM station no name key finds: ESR code -> OSM's name of the station.
NAME_AT = {"548502": "Baş Ələt Stansiyası",     # Алят, the old Alyat station on the coast
           "548703": "Yeni Ələt",               # Алят-Новая
           # Baku: Book 1's names are the Soviet ones, and the metro has stations of the same
           # names (Nəriman Nərimanov, Koroğlu) a few hundred metres off the railway
           "547016": "Montin \\ Nərimanov Stansiyası",   # ОП Нариманов
           "547001": "Keşlə stansiyası",                 # Кишлы (OSM name:ru Баку-Товарная)
           "547035": "Keşlə Stansiyası",                 # ОП Кишлы, the ring's halt
           # Sumgait: Book 1's "Сумгаит" is 11 km from Hacı Zeynalabdin and 7 from Сумгаит-
           # Новый, so it is OSM's Sumqayıtçay (name:ru Сумгаит-Главный), and the ring's
           # Sumqayıt station is Сумгаит-Новый
           "546403": "Sumqayıtçay Stansiyası",
           "546456": "Sumqayıt",
           "546901": "Çeşidləmə Stansiyası",             # Г.З. Тагиев-Сортировочная
           "547046": "Koroğlu@49.9196,40.4193"}          # Беюк-Шор: the ring's halt
# Junctions no OSM node names, at the OSM junction node their tariff km fit (Masis 2 km,
# Mkhchyan 7, Noragavit 6 for the post; the post 5 and Noragavit 1 for the loop).
POINT_AT = {"567429": (44.425637, 40.068279),   # Пост 2865 км, node 9570630674
            "567734": (44.448600, 40.107029)}   # Разъезд 9 км, node 292970842
# ESR nodes OSM puts away from their halt, which the off-line test above does not catch (the
# halt is near enough the line, but not where the km say). Abkhazia: Бармыш (574136), node
# 11642431951, is 4.4 km of track past Мюссера where Book 1 has 7 and ru.wikipedia's route
# diagram 6-8; Цицквара (574013), node 11148782080, is 3.2 km from New Athos where Book 1 has 7
# and the diagram 6 (it sits about where Гвандра is). Unplaced, the trace steps over them.
UNPLACE = {"574136", "574013"}
# Sections whose track ends short of Book 1's list, per region: section -> the last point with
# track, or None for none. Abkhazia: OSM has track from Psou to Ochamchira only. Beyond it, to Gali and
# the Inguri, the track was damaged or taken up in the 1990s (ru.wikipedia, "Абхазская железная
# дорога": from Achguara to Ingiri); OSM maps no railway there, nor on the Tkvarcheli branch
# (57-007, coal trains only, ru.wikipedia). Left out, the line ends where its track does.
TRACK_END = {"xa": {"57-001": "573307", "57-007": None}}
# Where a sheet's points lie: a name match with no placed neighbour must be there.
SHEET_LAND = {"57": {"ge", "xa", "none"}, "56": {"am"}, "55": {"az"}}


def book1():
    import ru_register as rr
    saved = rr.ROADS
    rr.ROADS = {sh: (code, ROAD[code][0]) for sh, code in SHEETS.items()}
    try:
        return rr.book1()
    finally:
        rr.ROADS = saved


def wd_stations():
    """ESR code -> {lon, lat, labels by language}, Wikidata's items with P2815 in the three
    countries, Abkhazia and South Ossetia (data/raw/ge/wd_stations_caucasus.json)."""
    p = RAW / "ge" / "wd_stations_caucasus.json"
    if not p.exists():
        return {}
    got = defaultdict(list)
    for r in json.loads(p.read_text("utf-8"))["rows"]:
        m = re.match(r"Point\(([-\d.]+) ([-\d.]+)\)", r.get("coord", ""))
        if not m:
            continue
        got[r["esr"]].append({"lon": float(m.group(1)), "lat": float(m.group(2)),
                              **{k: r.get(k, "") for k in ("ka", "hy", "az", "ru", "en")}})
    return {c: v[0] for c, v in got.items()
            if len({(round(x["lon"], 3), round(x["lat"], 3)) for x in v}) == 1}


def proc_routes():
    """(lon, lat) of every stop of an OSM route=train relation in the three extracts."""
    out = []
    for cc in CCS:
        d = ROOT / "data" / "proc" / cc
        if not (d / "rels.pkl").exists():
            continue
        with open(d / "stops.pkl", "rb") as f:
            stops = pickle.load(f)
        with open(d / "rels.pkl", "rb") as f:
            rels = pickle.load(f)
        for tags, members in rels.values():
            if tags.get("type") == "route" and tags.get("route") == "train":
                for ty, ref, role in members:
                    if ty == "n" and role.startswith(("stop", "platform")) and ref in stops:
                        out.append((stops[ref][1], stops[ref][2]))
    return out


def convert(cc, write=True):
    import numpy as np
    import shapely
    from scipy.spatial import cKDTree
    import ru_register as rr
    import ua_register as ua
    secs, b1name = book1()
    ops, b2name = rr.book2()
    log(f"Book 1 ({b1name}): {len(secs)} sections on the three sheets; Book 2 ({b2name})")
    esr = osm_esr()
    wd = wd_stations()
    # rail stations only: not the metros' (Tbilisi's Samgori metro station took the Kakheti
    # line's halt Самгори, 15 km away)
    ost = [r for r in osm_station_rows()
           if r.get("station") not in ("subway", "light_rail", "monorail", "funicular", "tram")
           and (r["usage"] not in ("disused", "abandoned") and r["rw"] in ("station", "halt")
                or (r["pt"] == "station" and r["train"] == "yes"))]
    nidx = NameIndex()
    for i, s in enumerate(ost):
        for nm in s["names"].values():
            for part in nm.split(";"):
                nidx.add(part, s["lon"], s["lat"], i)
    stat = Counter()

    # --- 1. each section's points: extra codes out (BORDER's kept)
    for s in secs:
        pts = []
        for p in s["points"]:
            if EXTRA.search(p["name"]) and p["esr"] not in BORDER:
                stat["extra codes left out"] += 1
                continue
            p = dict(p, name=rr.clean(p["name"]), raw=p["name"], km0=p["km"][0])
            if pts and pts[-1]["esr"] == p["esr"]:
                continue
            pts.append(p)
        s["pts"] = pts
        s["km"] = (pts[-1]["km0"] or 0) - (pts[0]["km0"] or 0) if len(pts) > 1 else 0
        s["road"] = SHEETS[s["sheet"]]
    raw_of = {}
    for s in secs:
        for p in s["pts"]:
            raw_of.setdefault(p["esr"], p)

    # --- 2. placement: the border points at the crossing; OSM's ESR node; Wikidata's item; an
    # OSM station of a matching name near the point's placed neighbours
    pos, how, node_of, taken = {}, {}, {}, {}
    shp_all = shapes()
    for g in shp_all.values():
        shapely.prepare(g)

    def country_of_xy(x, y):
        for k in ("none", "xa", "ge", "am", "az"):
            if shapely.contains_xy(shp_all[k], x, y):
                return k
        return "abroad"
    for c, b in BORDER.items():
        x, y = BORDER_XY[b][:2]
        if x is not None:
            pos[c], how[c] = (x, y), "border"
    for c in raw_of:
        if c in pos:
            continue
        rows = esr.get(c) or []
        best = ([r for r in rows if r["rw"] in ("station", "halt")]
                or [r for r in rows if r["pt"] == "station"] or rows)
        if best:
            pos[c], how[c], node_of[c] = (best[0]["lon"], best[0]["lat"]), "esr", best[0]
            taken[best[0]["id"]] = c
    # An ESR code OSM gives to the wrong node (568205, Arevik on the Gyumri - Maralik line, sits
    # on a halt by Lake Sevan): a point further from both placed neighbours on a section than
    # the tariff km allow (plus NAME_REACH_KM) is unplaced again.
    bad = set()
    for s in secs:
        placed = [p for p in s["pts"] if p["esr"] in pos and how[p["esr"]] == "esr"]
        for i, p in enumerate(placed):
            nb = [q for q in (placed[i - 1] if i else None,
                              placed[i + 1] if i + 1 < len(placed) else None) if q]
            if nb and all(dist_m(*pos[p["esr"]], *pos[q["esr"]]) / 1000 >
                          abs((q["km0"] or 0) - (p["km0"] or 0)) + NAME_REACH_KM for q in nb):
                bad.add(p["esr"])
    bad |= {c for c in UNPLACE if c in pos}
    for c in bad:
        log(f"  ESR node off its line, unplaced: {c} {raw_of[c]['name']}")
        taken.pop(node_of[c]["id"], None)
        del pos[c], how[c], node_of[c]
    by_osm_name = defaultdict(list)
    for r in ost:
        by_osm_name[r["names"].get("name")].append(r)
    for c, xy in POINT_AT.items():
        if c in raw_of and c not in pos:
            pos[c], how[c] = xy, "interpolated"
    for c, nm in NAME_AT.items():
        nm, at = (nm.split("@") + [None])[:2]
        if c in raw_of and by_osm_name.get(nm):
            if c in pos:
                taken.pop(node_of.get(c, {}).get("id"), None)
            r = by_osm_name[nm][0]
            if at:
                x, y = map(float, at.split(","))
                r = min(by_osm_name[nm], key=lambda r: dist_m(r["lon"], r["lat"], x, y))
            pos[c], how[c], node_of[c] = (r["lon"], r["lat"]), "name (NAME_AT)", r
            taken[r["id"]] = c
    for _round in range(5):
        for s in secs:
            pts = s["pts"]
            for i, p in enumerate(pts):
                c = p["esr"]
                if c in pos:
                    continue
                anchors = []
                for j in list(range(i - 1, -1, -1)) + list(range(i + 1, len(pts))):
                    q = pts[j]
                    if q["esr"] in pos:
                        anchors.append((pos[q["esr"]], abs((q["km0"] or 0) - (p["km0"] or 0))))
                        if len(anchors) >= 2:
                            break
                if not anchors and _round < 3:
                    continue

                def fits(lon, lat):
                    return all(dist_m(lon, lat, *a) / 1000 <= km + NAME_REACH_KM
                               for a, km in anchors)
                w = wd.get(c)
                if w and fits(w["lon"], w["lat"]):
                    pos[c], how[c] = (w["lon"], w["lat"]), "wikidata"
                    continue
                weak = len(skeleton(p["name"])) <= 2

                def fits_strict(x):
                    # a skeleton of two consonants ("Алят" lT, "Тапа" TP) is weak: only near
                    # where the tariff km put it
                    if x[3] == 0 or not weak:
                        return True
                    return all(dist_m(x[0], x[1], *a) / 1000 <= km + 3 for a, km in anchors)
                cands = [x for x in nidx.find(p["name"]) if fits(x[0], x[1])
                         and taken.get(ost[x[2]]["id"], c) == c and fits_strict(x)]
                if not cands:
                    continue
                if anchors:
                    # the place whose distances from the placed neighbours best fit the tariff
                    # km ("Пойлы", 14 km from Ağstafa: Poylu, not Yeni Poylu 2 km away)
                    lon, lat, ref, _r = min(cands, key=lambda x: sum(
                        abs(dist_m(x[0], x[1], *a) / 1000 - km) for a, km in anchors))
                elif (_round >= 3 and not weak
                      and all(dist_m(x[0], x[1], cands[0][0], cands[0][1]) < 3000 for x in cands)
                      and country_of_xy(cands[0][0], cands[0][1]) in SHEET_LAND[s["road"]]):
                    # no placed neighbour: a single place of that name, in the sheet's country
                    lon, lat, ref, _r = cands[0]
                else:
                    continue
                pos[c], how[c], node_of[c] = (lon, lat), "name", ost[ref]
                taken[ost[ref]["id"]] = c
    # Junction posts and passing loops no OSM node names ("Пост 2865 км", where the Yerevan
    # line leaves the Gyumri - Masis line; "Разъезд 9 км"): placed by their km between the
    # placed points either side on a section, so a line ending there can be traced (rinf.py
    # snaps them to the track and checks the lengths). Halts are not: an unnamed halt drawn at
    # a guessed place would be a stop where there is none.
    for _round in range(2):
        for s in secs:
            pts = s["pts"]
            for i, p in enumerate(pts):
                c = p["esr"]
                if c in pos or not re.search(r"^(?:Пост|Разъезд)\b|\((?:рзд|бп|п)\)", p["name"]):
                    continue
                a = next((q for q in pts[i - 1::-1] if q["esr"] in pos), None) if i else None
                b = next((q for q in pts[i + 1:] if q["esr"] in pos), None)
                if a is None or b is None:
                    continue
                ka, kb, kp = a["km0"] or 0, b["km0"] or 0, p["km0"] or 0
                if kb == ka or abs(kb - ka) > 30:
                    continue
                f = (kp - ka) / (kb - ka)
                (xa, ya), (xb, yb) = pos[a["esr"]], pos[b["esr"]]
                pos[c], how[c] = (xa + f * (xb - xa), ya + f * (yb - ya)), "interpolated"
    codes = set(raw_of)
    log(f"points: {len(codes)}; placed {dict(Counter(how.get(c, 'unplaced') for c in codes))}")
    if os.environ.get("CAUCASUS_DEBUG"):
        for s in secs:
            miss = [p["name"] for p in s["pts"] if p["esr"] not in pos]
            if miss:
                log(f"  unplaced on {s['id']} {s['name']}: {', '.join(miss)}")
                for p in s["pts"]:
                    if p["esr"] not in pos and os.environ["CAUCASUS_DEBUG"] == "2":
                        got = nidx.find(p["name"])[:3]
                        nb = [(q["name"], pos[q["esr"]], abs(q["km0"] - p["km0"]))
                              for q in s["pts"] if q["esr"] in pos][:3]
                        log(f"      {p['name']}: cands {[(round(x, 4), round(y, 4), r) for x, y, _i, r in got]}"
                            f" placed on the section {nb}")
            for p in s["pts"]:
                if how.get(p["esr"]) == "name":
                    log(f"    by name on {s['id']}: {p['name']} -> "
                        f"{node_of[p['esr']]['names'].get('name')} {pos[p['esr']]}")

    # --- names: OSM's (the country's language) where the point is an OSM station, else
    # Wikidata's label in that language, else Book 1's Russian; English from OSM's name:en or
    # Wikidata's, where it reads as a romanisation of Book 1's name
    lang = LANG[cc]

    def country_at(q):
        return country_of_xy(*q)
    near_tree = cKDTree(np.array([[s["lon"] * 0.75, s["lat"]] for s in ost]))
    native, en_of, name_how = {}, {}, Counter()
    for c, p in raw_of.items():
        n = node_of.get(c)
        nm, en = "", ""
        # (Abkhazia: Ochamchira's ESR node is its yard, which carries the station's names)
        if n and (n["rw"] in ("station", "halt", None) or (cc == "xa" and n["rw"] == "yard"))                 and n["names"].get("name"):
            nm, en = n["names"]["name"], n["names"].get("name:en", "")
            name_how["OSM"] += 1
        elif c in pos:
            ks = set(name_keys(p["name"]))
            for j in near_tree.query_ball_point([pos[c][0] * 0.75, pos[c][1]], 0.004):
                if any(set(name_keys(x)) & ks for x in ost[j]["names"].values()):
                    nm, en = ost[j]["names"].get("name", ""), ost[j]["names"].get("name:en", "")
                    name_how["OSM nearby"] += 1
                    break
        # a point in Georgia named in Abkhaz or the like: the country's own language first
        if n and n["names"].get(f"name:{lang}") and c in pos and country_at(pos[c]) == cc:
            nm = n["names"][f"name:{lang}"]
        if not nm and wd.get(c, {}).get(lang):
            nm = wd[c][lang]
            name_how["Wikidata"] += 1
        if cc == "xa" and re.search("[Ⴀ-ჿ]", nm or ""):
            # a Georgian name on an OSM node in Abkhazia ("ბესლახუბა"): its signs are Abkhaz
            # and Russian, so OSM's Russian name, else Book 1's
            nm = (n or {}).get("names", {}).get("name:ru", "")
            name_how["Georgian name replaced (Abkhazia)"] += 1
        if not nm:
            nm = p["name"]
            name_how["Book 1 (Russian)"] += 1
        native[c] = clean_native(nm)
        if not en and wd.get(c, {}).get("en"):
            en = rr.clean_en(wd[c]["en"])
        en = clean_native(en)
        # Abkhazia: many stations were renamed after 1992 (Gantiadi is Tsandrypsh), so an
        # English name may also romanise OSM's own Russian name rather than Book 1's
        ru_osm = (n or {}).get("names", {}).get("name:ru", "") if cc == "xa" else ""
        if en and (rr.en_score(rr.EXTRA_CODE.sub("", p["name"]), en) >= 0.7
                   or (ru_osm and rr.en_score(ru_osm, en) >= 0.7)):
            en_of[c] = en
    # a border export code is named as its station ("Айрум (эксп.)" as Ayrum), for line names
    by_nkey = {rr.nkey(p["name"]): c for c, p in raw_of.items() if c not in BORDER}
    for c in BORDER:
        if c in raw_of:
            base = by_nkey.get(rr.nkey(rr.EXTRA_CODE.sub("", raw_of[c]["name"])))
            if base:
                native[c] = native[base]
                if base in en_of:
                    en_of[c] = en_of[base]
    st_by_id = {r["id"]: r for r in ost}
    for c, nid in BORDER_STATION_NODE.items():
        r = st_by_id.get(nid)
        if c in raw_of and r:
            native[c] = clean_native(r["names"].get("name", native.get(c, "")))
            if r["names"].get("name:en"):
                en_of[c] = r["names"]["name:en"]
    log(f"names: {dict(name_how)}; English {len(en_of)}")

    # --- 3. which country each point is in
    where = {c: country_at(pos[c]) for c in codes if c in pos}
    for c, b in BORDER.items():
        where[c] = "border"
    for s in secs:
        pts = s["pts"]
        for i, p in enumerate(pts):
            c = p["esr"]
            if c in where:
                continue
            prev = next((where[pts[j]["esr"]] for j in range(i - 1, -1, -1)
                         if pts[j]["esr"] in where), None)
            nxt = next((where[pts[j]["esr"]] for j in range(i + 1, len(pts))
                        if pts[j]["esr"] in where), None)
            # between a border point and a station: the station's side (Ruisbolo, between
            # Gardabani and the Azerbaijani border)
            if prev == "border":
                prev = nxt
            if nxt == "border":
                nxt = prev
            where[c] = prev if prev == nxt and prev not in (None, "border") else "unknown"

    def inside(c):
        w = where.get(c)
        return w == cc or (w == "border" and cc in BORDER_XY[BORDER[c]][2])

    # --- 4. which points are stops
    kx = 111.32 * math.cos(math.radians(41.0))
    tt, tt_pairs = feed_calls()
    served_xy = proc_routes()
    n_osm = len(set(served_xy))
    served_xy += list(tt)
    log(f"served places: {n_osm} OSM train route stops, {len(tt)} timetable stations")
    stree = cKDTree(np.array([[x * kx, y * 110.57] for x, y in set(served_xy)]))
    halts = [(s["lon"], s["lat"]) for s in ost if s["rw"] == "halt"]
    htree = cKDTree(np.array([[x * kx, y * 110.57] for x, y in halts]))
    ttree = cKDTree(np.array([[x * kx, y * 110.57] for x, y in tt])) if tt else None

    def near(tree, c, m):
        q = pos.get(c)
        return bool(q and tree is not None and tree.query_ball_point(
            [q[0] * kx, q[1] * 110.57], m / 1000))

    def call_ids(c):
        q = pos.get(c)
        if not q or ttree is None:
            return set()
        return set(ttree.query_ball_point([q[0] * kx, q[1] * 110.57], SERVED_M / 1000))

    def consecutive(c1, c2):
        a, b = call_ids(c1), call_ids(c2)
        return any((min(x, y), max(x, y)) in tt_pairs for x in a for y in b)
    kind = {}
    for c, p in raw_of.items():
        # Unlike Russia's and Ukraine's, Book 2 gives most Georgian stations no passenger
        # operation (Mtskheta, Kobuleti, Ureki: "1,3"), so a call or a halt decides first.
        if c in BORDER:
            kind[c] = "border"
        elif how.get(c) == "interpolated":
            kind[c] = "none"
        elif near(stree, c, SERVED_M):
            kind[c] = "served"
        elif re.match(r"(?:ОП|О\.П\.)\s", p["raw"]) or near(htree, c, HALT_M):
            kind[c] = "halt"
        elif rr.passenger_op(ops.get(c)):
            kind[c] = "flag"
        else:
            kind[c] = "none"
    log(f"point kinds: {dict(Counter(kind.values()))}")

    # --- 5. sections: owners, this country only, unserved stretches
    owner = {}
    for s in sorted(secs, key=lambda s: (TYPE_RANK.get(s["type"], 9), s["id"])):
        for a, b in zip(s["pts"], s["pts"][1:]):
            owner.setdefault(frozenset((a["esr"], b["esr"])), s["id"])
    rows, names, used, stop_codes = [], {}, set(), set()
    km_kept = defaultdict(float)
    for s in secs:
        if s["km"] == 0:
            continue
        pairs = []
        past_end = False
        for a, b in zip(s["pts"], s["pts"][1:]):
            key = frozenset((a["esr"], b["esr"]))
            km = (b["km0"] or 0) - (a["km0"] or 0)
            ends = TRACK_END.get(cc, {})
            if s["id"] in ends and (ends[s["id"]] is None or past_end):
                km_kept["no track (TRACK_END)"] += km
                continue
            past_end = past_end or b["esr"] == ends.get(s["id"])
            if owner[key] != s["id"]:
                stat["pairs on another section"] += 1
                continue
            if not (inside(a["esr"]) and inside(b["esr"])):
                w = {where.get(a["esr"]), where.get(b["esr"])} - {cc, "border"}
                km_kept["outside: " + "/".join(sorted(str(x) for x in w))] += km
                continue
            if a["esr"] in BORDER and b["esr"] in BORDER:
                continue
            pairs.append((a, b, km))
        if not pairs:
            continue
        is_stop = {}
        runs, cur = [], []
        for i, (a, b, km) in enumerate(pairs):
            if cur and cur[-1][1]["esr"] != a["esr"]:
                runs.append(cur)
                cur = []
            cur.append((a, b, km))
        runs.append(cur)
        clone, stretch_idx = set(), set()
        idx0 = 0
        for run in runs:
            pts_run = [run[0][0]] + [b for _a, b, _k in run]
            stop_at = [i for i, p in enumerate(pts_run) if kind[p["esr"]] == "served"]
            cuts = sorted(set([0, len(pts_run) - 1] + stop_at))
            for u, v in zip(cuts, cuts[1:]):
                inner = pts_run[u:v + 1]
                km = sum(k for _a, _b, k in run[u:v])
                has_halt = any(kind[p["esr"]] == "halt" for p in inner)
                for p in inner:
                    if kind[p["esr"]] == "served" or (has_halt and kind[p["esr"]] in
                                                      ("halt", "flag")):
                        is_stop[p["esr"]] = True
                ends_stop = (kind[inner[0]["esr"]] == "served"
                             and kind[inner[-1]["esr"]] == "served")
                if (not has_halt and ends_stop and km >= STRETCH_KM
                        and not consecutive(inner[0]["esr"], inner[-1]["esr"])):
                    clone |= {inner[0]["esr"], inner[-1]["esr"]}
                    stretch_idx |= set(range(idx0 + u, idx0 + v))
                    stat["unserved stretches with no halt left to the timetable"] += 1
                    km_kept["stretches left to the timetable"] += km
            idx0 += len(run)
        flat = [x for run in runs for x in run]
        for k, (a, b, km) in enumerate(flat, 1):
            ca, cb = a["esr"], b["esr"]
            if k - 1 in stretch_idx:
                ca = f"{ca}@{s['id']}" if ca in clone else ca
                cb = f"{cb}@{s['id']}" if cb in clone else cb
            rows.append({"sol": f"{s['id']}:{k}", "line": s["id"], "a": f"esr:{ca}",
                         "b": f"esr:{cb}", "len": str(km), "im": IM_OF.get(cc, s["road"]),
                         "label": f"{a['name']} - {b['name']}"})
            used |= {ca, cb}
            km_kept["kept"] += km
        for c in sorted(clone):
            cl = f"{c}@{s['id']}"
            rows.append({"sol": f"{s['id']}:{c}@", "line": s["id"], "a": f"esr:{c}",
                         "b": f"esr:{cl}", "len": "0", "im": IM_OF.get(cc, s["road"]),
                         "label": "clone"})
            rows.append({"sol": f"{s['id']}:{c}@x", "line": s["id"], "a": f"esr:{cl}",
                         "b": f"esr:{cl}x", "len": "0", "im": IM_OF.get(cc, s["road"]),
                         "label": "stub"})
            used |= {c, cl, f"{cl}x"}
        stop_codes |= {c for c, v in is_stop.items() if v}
        by_key = {}
        for p in s["pts"]:
            by_key.setdefault(rr.nkey(p["name"]), p)
            by_key.setdefault(rr.nkey(rr.EXTRA_CODE.sub("", p["name"])), p)
        pa, pb, vias = ua.section_ends(s, by_key)
        # the line's own ends as built (a section cut at a border or the de facto line ends
        # where this country's part does)
        ends = [x for x in (flat[0][0], flat[-1][1])]
        if not inside(pa["esr"]) or pa["esr"] in BORDER:
            pa = ends[0] if ends[0]["esr"] not in BORDER else pa
        if not inside(pb["esr"]) or pb["esr"] in BORDER:
            pb = ends[1] if ends[1]["esr"] not in BORDER else pb
        name = f"{native[pa['esr']]} — {native[pb['esr']]}"
        ea, eb = en_of.get(pa["esr"], ""), en_of.get(pb["esr"], "")
        name_en = f"{ea} — {eb}" if ea and eb else ""
        names[s["id"]] = {"name": name, "name_en": name_en, "type": s["type"],
                          "sheet": s["sheet"], "road": s["road"], "tariff_km": s["km"],
                          "header": s["name"]}
    cnt = Counter(e["name"] for e in names.values())
    for sid, e in names.items():
        if cnt[e["name"]] > 1:
            e["name"] += f" ({sid})"
            if e["name_en"]:
                e["name_en"] += f" ({sid})"
    log(f"sections: {dict(stat)}")
    log(f"tariff km: { {k: round(v) for k, v in km_kept.items()} }")
    pts_out = []
    for c in sorted(used):
        base = c.split("@")[0]
        if base in BORDER:
            b = BORDER[base]
            x, y, _ccs, bname = BORDER_XY[b]
            pts_out.append({"op": f"esr:{c}", "uopid": b, "name": bname, "type": "90",
                            "lon": x, "lat": y})
            continue
        p = raw_of[base]
        q = pos.get(base)
        typ = (("70" if re.match(r"(?:ОП|О\.П\.)\s", p["raw"]) else "10")
               if base in stop_codes and base == c else "80")
        r = {"op": f"esr:{c}", "uopid": f"{cc.upper()}{c}", "name": native[base], "type": typ,
             "name_ru": p["name"]}
        if en_of.get(base) and not c.endswith("x"):
            r["name_en"] = en_of[base]
        if c.endswith("x"):
            r["name"] = ""
            q = None
        if q:
            r["lon"], r["lat"] = q
        pts_out.append(r)
    log(f"{cc}: {len(rows)} section rows on {len(names)} lines, {len(pts_out)} points "
        f"({sum(1 for r in pts_out if r['type'] in ('10', '70'))} stops, "
        f"{sum(1 for r in pts_out if 'lon' not in r)} unplaced)")
    if write:
        d = OUT / cc
        d.mkdir(parents=True, exist_ok=True)
        stamp = {"endpoint": f"Тарифное руководство № 4, {b1name}, {b2name} "
                             "(caucasus_register.py)", "fetched": date.today().isoformat()}
        (d / "sections.json").write_text(json.dumps({**stamp, "rows": rows}, ensure_ascii=False),
                                         "utf-8")
        (d / "points.json").write_text(json.dumps({**stamp, "rows": pts_out},
                                                  ensure_ascii=False), "utf-8")
        (d / "names.json").write_text(json.dumps(names, ensure_ascii=False, indent=0), "utf-8")
    return rows, pts_out, names


def build(path, log_):
    """build_model's register hook: convert, then rinf.py traces it over OSM track."""
    convert(Path(path).name)
    import rinf
    return rinf.build(path, log_)


# ================================================================ rinf.py settings

def _names(cc):
    f = OUT / cc / "names.json"
    return json.loads(f.read_text("utf-8")) if f.exists() else {}


def country_conf(cc, iso3):
    def id_name(lid, _uop=None):
        e = _names(cc).get(lid)
        return (e["name"], e.get("name_en") or "") if e else None

    en_cache = {}

    def station_en(point, _name):
        if not en_cache:
            pf = OUT / cc / "points.json"
            en_cache.update({r["uopid"]: r["name_en"] for r in
                             (json.loads(pf.read_text("utf-8"))["rows"] if pf.exists() else [])
                             if r.get("name_en")})
            en_cache.setdefault("", "")
        return en_cache.get(point.get("uopid"), "")

    def stop_name(point):
        if point.get("type") in ("10", "70"):
            return point.get("name") or None
        return None
    return {"stop_name": stop_name, "iso3": iso3, "langs": [LANG[cc], "en"],
            "osm_rel": lambda _tags: None, "id_name": id_name, "tol_abs": 1.5,
            "direct_near_m": 1500, "im": {k: v[0] for k, v in ROAD.items()},
            "station_en": station_en}


# ================================================================ colours

# Tariff sections have no colours of their own: one picked colour per railway, as Russia's
# and Ukraine's, the three neighbours unlike (`picked` in colours/<cc>.csv).
COLOUR = {"ge": "#C8102E", "am": "#E08E0B", "az": "#0B7FBF", "xa": "#1E9E5A"}


def colours(cc):
    import csv
    names = _names(cc)
    road = lambda e: ROAD[IM_OF.get(cc, e["road"])]   # noqa: E731
    rows = [{"line": e["name"], "operator": road(e)[0], "colour": COLOUR[cc],
             "source": "picked", "url": "", "note": f"{road(e)[1]}: one colour"}
            for _sid, e in sorted(names.items())]
    path = ROOT / "colours" / f"{cc}.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["line", "operator", "colour", "source", "url", "note"])
        w.writeheader()
        w.writerows(rows)
    log(f"--colours {cc}: {len(rows)} register lines -> {path}")


def split_pieces(lines, stations, geoms, reg_ways, state, log):
    """build_model's hook, after drop_unridden_sections: rinf.split_pieces (folds a stop's 0 km
    link junctions back into it, and bridges where the country's settings ask)."""
    import rinf
    rinf.split_pieces(lines, stations, geoms, reg_ways, state, log)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--esr", nargs=2, metavar=("CC", "PBF"))
    ap.add_argument("--outline", action="store_true")
    ap.add_argument("--clip", metavar="CC")
    ap.add_argument("--timetable", action="store_true")
    ap.add_argument("--convert", metavar="CC")
    ap.add_argument("--dry", metavar="CC")
    ap.add_argument("--colours", metavar="CC")
    a = ap.parse_args()
    if a.esr:
        p = Path(a.esr[1])
        esr_pass(a.esr[0], p if p.is_absolute() else ROOT / p)
    if a.outline:
        write_outline()
    if a.clip:
        clip(a.clip)
    if a.timetable:
        timetable()
    if a.convert:
        convert(a.convert)
    if a.dry:
        convert(a.dry, write=False)
    if a.colours:
        colours(a.colours)
