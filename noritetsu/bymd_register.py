"""Belarus (by) and Moldova (md): Tariff Guide No. 4 (Тарифное руководство № 4), the sheets of
Belarusian Railway (Бел, road 13) and of CFM (Млд, road 39), written into rinf.py's input
format, plus a timetable feed for each. Ukraine's recipe (ua_register.py), whose parser is
Russia's (ru_register.book1). by_sources.md and md_sources.md have the research and numbers;
this docstring is how it works.

    python bymd_register.py --cc by --esr data/raw/belarus-latest.osm.pbf   # ESR codes + station names
    python bymd_register.py --cc by --wikidata       # data/raw/by/wd_stations.json
    python bymd_register.py --cc by --outline --clip # after every extract (data/proc/by kept as .../full)
    python bymd_register.py --cc by --crawl          # poezdato.net, Belarusian stations and trains (~1 h)
    python bymd_register.py --cc md --merstren       # merstren.md's timetable data (one file)
    python bymd_register.py --cc by --timetable      # -> data/raw/gtfs/<cc>/..., timetable_calls.json
    python bymd_register.py --cc by --convert        # -> data/raw/rinf/<cc>/{sections,points,names}.json
    python bymd_register.py --cc by --colours        # -> colours/<cc>.csv
    python bymd_register.py --cc by --crossings      # rail ways over the country's boundary, with routes
    python build_model.py --region by --register rinf:data/raw/rinf/by

THE REGISTER.  Book 1 (data/raw/ru/tr4_kniga1_*.xls, the file Russia's build reads) has one sheet
per CIS railway; "Бел" lists Belarusian Railway's 87 tariff sections (5,330 tariff km), "Млд"
CFM's 21 (1,216 km). One tariff section is one register line, as in Russia and Ukraine, named
by its two ends as shown. Points are placed at OSM's node with their ESR code, else at
Wikidata's item with that code (P2815), else at an OSM station of the same name (any of its
name tags; Book 1 spells every name in Russian) near the point's placed neighbours.

NAMES SHOWN.  The OSM `name` of the point's station: Belarusian in Belarus (OSM Belarus names
in Belarusian; the Russian name is name:ru), Romanian in Moldova. Else Wikidata's label in that
language, else an OSM station of a matching name within 300 m, else Book 1's own Russian.

TERRITORY.  Belarus: OSM's boundary (relation 59065). Moldova: OSM's boundary (relation 58974),
Transnistria included (Anita, 2026-10-04: "we can gray transnistria if no trains"). Its railway
is run separately from CFM and no source shows a passenger train there today (CFM's Bender-3 -
Chișinău train was suspended in January 2025, Chișinău - Odesa in February 2022), so its
sections are in the register and the timetable, which has no train there, greys them. Both
outlines reach REACH_DEG into the neighbours so a crossing keeps its track up to the border
point.

STOPS.  As Ukraine's: a point is a stop when Book 2 gives it a passenger operation and either a
train calls there (the timetable feed, or an OSM train route, within SERVED_M) or it is a halt
on a stretch that has halts. A long halt-free stretch no train runs over is freight track: its
ends are cloned as junctions so the timetable decides (ua_register's docstring).

TIMETABLES.  Belarus: Belarusian Railway's own site (pass.rw.by) answers 403 from here.
poezdato.net, the Russian-language sister of the poizdato.net Ukraine's build crawled, lists
Belarusian trains with calls and a running calendar for the current and next month, and its
robots.txt allows the pages: `--crawl` walks its station pages from Belarusian stations (names
matched to Book 1's) and fetches every train page they list, 1.2 s apart. Moldova:
merstren.md's timetable file (orar.js, every CFM, CFR and UZ train in Moldova, "verificat
29.09.2026"; robots.txt allows everything), with stop coordinates from CFM's GTFS (Transitous'
rehost) and the neighbours' built stations. `--timetable` writes either into a GTFS feed in
data/raw/gtfs/<cc>/ that gtfs_served reads like any national feed.

BORDERS.  BORDER lists crossings with passenger trains; each gets a piece from the last point of
the line to the border point (ERA RINF's where it has one, else BORDER_POINTS, proposed for
borders.EXTRA), so both countries' builds end at one id.
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
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
NE = ROOT.parent / "religiondots" / "data" / "geo" / "ne_10m_admin_0_countries.geojson"
USER_AGENT = "noritetsu-build/1.0 (hobby rail map)"
WIKIDATA = "https://query.wikidata.org/sparql"

CC = {
    "by": {
        "sheet": "Бел", "road": "13", "operator": "Беларуская чыгунка", "lang": "be",
        "names": ("name", "name:be", "name:ru", "old_name", "name:be-tarask"),
        "boundary": "by_boundary.geojson", "minus": [], "wd_country": "Q184",
        "reach_into": ("pl", "lt", "lv", "ru", "ua"), "via": "праз",
        "agency": "Беларуская чыгунка", "tz": "Europe/Minsk", "url": "https://www.rw.by/",
        "clone": True,
    },
    "md": {
        "sheet": "Млд", "road": "39", "operator": "Calea Ferată din Moldova", "lang": "ro",
        "names": ("name", "name:ro", "name:ru", "old_name", "name:uk"),
        # Transnistria is drawn as Moldova's, greyed (Anita, 2026-10-04: "we can gray
        # transnistria if no trains"); the timetable has no train there, so gtfs_served marks
        # its sections not running. Before that it was "minus": ["transnistria_boundary.geojson"].
        "boundary": "md_boundary.geojson", "minus": [],
        "wd_country": "Q217", "reach_into": ("ro", "ua"), "via": "prin",
        "agency": "Calea Ferată din Moldova", "tz": "Europe/Chisinau", "url": "https://railway.md/",
        # every train in Moldova is in merstren's data, so a halt-free stretch between two
        # called stations needs no cloning: the timetable greys it if no train runs over it
        "clone": False,
    },
}
REACH_DEG = 0.03
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
# Transnistria (OSM relation 65335, with Bender): drawn as Moldova's, but its track is run by
# the Pridnestrovian Railway. A section row with both ends in it carries this manager code
# (rinf_countries/md.py IM: CFM and the Pridnestrovian Railway), so a line mostly there names
# both.
TRANSNISTRIA = "transnistria_boundary.geojson"
PMR_IM = "39P"
# Export codes placed where OSM's track crosses the boundary (--crossings), not at their
# station: Transnistria's two lines to Ukraine end at the border, as Ukraine's greyed lines
# from Rozdilna and Podilsk end at Kuchurhan and Slobidka beside it. No passenger train over
# either, so no border point; the piece to the border is greyed with its line.
EXPORT_AT = {"md": {
    "392306": (29.974995, 46.753474, "Moldova – Ukraine border"),   # Новосавицкая (эксп.): ways 1037371396/7, towards Kuchurhan
    "394803": (29.252470, 47.809810, "Moldova – Ukraine border"),   # Колбасна (эксп.): way 855238117, towards Slobidka
}}

SERVED_M = 400            # an OSM train route stop or a timetable call this close serves a point
NAME_REACH_KM = 10        # a name match may lie this much further than the tariff km say
STRETCH_KM = 8            # an unserved stop-to-stop stretch at least this long with no halt is freight
HALT_M = 400              # an OSM railway=halt this close to a point marks it a passenger halt

# Crossings with passenger trains: (border point uopid, the Book 1 point the line reaches it
# from). The point is ERA RINF's (border_points.json) or one of BORDER_POINTS.
BORDER = {
    "by": [
        ("EU00250", ["163000"]),          # Гудогай - Kena (Lithuania): trains to Kaliningrad
        ("BYRUOSINOVKA", ["169026"]),     # Осиновка - Krasnoye (Russia): Minsk - Moscow
        ("BYRUZAOLSHA", ["165608"]),      # Заольша - Rudnya (Russia): Vitebsk - Smolensk
        ("BYRUEZERISHCHE", ["160801"]),   # Езерище - Nevel (Russia): St Petersburg - Vitebsk
        ("BYRUALESHA", ["161607"]),       # Алеща - Novosokolniki (Russia): Pskov-region trains
        ("BYRUZAKOPYTYE", ["150513"]),    # Закопытье - Novozybkov (Russia): Minsk - Adler
    ],
    "md": [
        ("EU00244", ["392005"]),            # Унгень - Iași (Romania)
        ("MDUAVALCINET", ["394316", "394301"]),  # Отачь / Вэлчинец - Mohyliv-Podilskyi: Kyiv trains
    ],
}
# Border points ERA RINF has none for: where OSM's track crosses OSM's boundary (--crossings,
# 2026-10-03 extracts; a double track's two crossings averaged). Each is proposed for
# borders.EXTRA under the id "e" + its key, so this register's "e"+uopid is that id.
BORDER_POINTS = {
    "BYRUOSINOVKA": (30.987625, 54.682434, "Belarus – Russia border"),    # ways 1228916719/20
    "BYRUZAOLSHA": (30.955662, 54.979000, "Belarus – Russia border"),     # way 24946520
    "BYRUEZERISHCHE": (29.962024, 55.856790, "Belarus – Russia border"),  # way 1255571734
    "BYRUALESHA": (29.379967, 55.754028, "Belarus – Russia border"),      # way 398157453
    "BYRUZAKOPYTYE": (31.592222, 52.453851, "Belarus – Russia border"),   # way 197985220
    "MDUAVALCINET": (27.779499, 48.448785, "Moldova – Ukraine border"),   # way 41922077
}


# Stations Book 1 does not list on the section they lie on: (section, the point it follows,
# ESR code, Book 1-style name, tariff km). Book 1 runs Minsk's sections to Minsk-
# Sortirovochny and never names Minsk-Passazhirsky, the main station, which 13-090's track to
# Orsha passes 1.2 km past Institut Kultury (OSM's ESR node 140210; Book 2 gives it
# passenger operations). Without it the main station was on no register line, and 1,100-odd
# timetable calls there were stepped over.
INSERT = {
    "by": [("13-090", "140013", "140210", "Минск-Пассажирский", 5),
           # 13-087's tariff route Minsk-Severny - Minsk-Sortirovochny (5 km against 2.7 crow-fly)
           # runs through Minsk-Passazhirsky: the line from Molodechno ends there, and its last
           # 3 km on to Sortirovochny are 13-090's (left to it as "listed finer elsewhere").
           ("13-087", "140102", "140210", "Минск-Пассажирский", 19)],
}


def raw(cc):
    return ROOT / "data" / "raw" / cc


def proc(cc):
    return ROOT / "data" / "proc" / cc


def out_dir(cc):
    return ROOT / "data" / "raw" / "rinf" / cc


def dist_m(lon1, lat1, lon2, lat2):
    dx = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)) * 111320
    dy = (lat2 - lat1) * 110570
    return math.hypot(dx, dy)


# ================================================================ OSM side (needs the .pbf)

def esr_pass(cc, pbf):
    """From the .pbf (extract.py keeps neither esr:user nor the name:xx tags): every node with
    an ESR code, and every rail station or halt node with all its names."""
    os.environ.setdefault("OSMIUM_POOL_THREADS", "2")
    import osmium
    esr, st = defaultdict(list), []
    keys = CC[cc]["names"]
    fp = (osmium.FileProcessor(str(pbf), osmium.osm.NODE)
          .with_filter(osmium.filter.KeyFilter("esr:user", "esr", "railway:esr", "railway",
                                               "public_transport")))
    for obj in fp:
        t = obj.tags
        code = t.get("esr:user") or t.get("esr") or t.get("railway:esr")
        rw = t.get("railway")
        if not code and rw not in ("station", "halt") and not (
                t.get("public_transport") == "station" and t.get("train") == "yes"):
            continue
        row = {"id": obj.id, "lon": round(obj.location.lon, 6), "lat": round(obj.location.lat, 6),
               "names": {k: t.get(k) for k in keys if t.get(k)}, "en": t.get("name:en") or "",
               "rw": rw, "pt": t.get("public_transport"), "train": t.get("train"),
               "usage": t.get("usage") or t.get("disused:railway"), "esr": code or ""}
        if code:
            for c in code.replace(",", ";").split(";"):
                c = c.strip()
                if c.isdigit() and len(c) == 6:
                    esr[c].append(row)
        if rw in ("station", "halt") or t.get("public_transport") == "station":
            st.append(row)
    raw(cc).mkdir(parents=True, exist_ok=True)
    (raw(cc) / "osm_esr.json").write_text(json.dumps(esr, ensure_ascii=False), "utf-8")
    (raw(cc) / "osm_stations.json").write_text(json.dumps(st, ensure_ascii=False), "utf-8")
    print(f"--esr: {sum(len(v) for v in esr.values())} nodes, {len(esr)} codes; "
          f"{len(st)} station nodes -> {raw(cc)}")


def load_osm_stations(cc):
    """OSM rail station and halt nodes (from --esr), every name a node carries; `name` the one
    shown (OSM's `name`)."""
    out = []
    for r in json.loads((raw(cc) / "osm_stations.json").read_text("utf-8")):
        if r["usage"] in ("disused", "abandoned") or r["rw"] in ("disused", "abandoned"):
            continue
        if r["rw"] not in ("station", "halt") and not (r["pt"] == "station" and r["train"] == "yes"):
            continue
        nm = r["names"].get("name") or next(iter(r["names"].values()), "")
        out.append({"id": r["id"], "lon": r["lon"], "lat": r["lat"], "name": nm,
                    "names": list(dict.fromkeys(r["names"].values())), "en": r["en"],
                    "rw": r["rw"], "esr": r["esr"]})
    return out


# ================================================================ Wikidata

Q_WD = """SELECT ?s ?esr ?coord ?lab ?ru ?en WHERE {
  ?s wdt:P2815 ?esr ; wdt:P17 wd:%s .
  OPTIONAL { ?s wdt:P625 ?coord }
  OPTIONAL { ?s rdfs:label ?lab FILTER(LANG(?lab) = "%s") }
  OPTIONAL { ?s rdfs:label ?ru FILTER(LANG(?ru) = "ru") }
  OPTIONAL { ?s rdfs:label ?en FILTER(LANG(?en) = "en") }
}"""


def fetch_wikidata(cc):
    import urllib.parse
    import urllib.request
    q = Q_WD % (CC[cc]["wd_country"], CC[cc]["lang"])
    req = urllib.request.Request(WIKIDATA + "?" + urllib.parse.urlencode({"query": q}), headers={
        "Accept": "application/sparql-results+json", "User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    rows = [{v: b[v]["value"] for v in b} for b in d["results"]["bindings"]]
    (raw(cc) / "wd_stations.json").write_text(json.dumps(
        {"fetched": time.strftime("%Y-%m-%d"), "rows": rows}, ensure_ascii=False), "utf-8")
    print(f"--wikidata: {len(rows)} rows, {len({r['esr'] for r in rows})} codes")


def clean_label(label):
    s = re.sub(r"\s*\((?:станцыя|станция|stație|stația|gară|halt|station)[^)]*\)", "", label or "",
               flags=re.I).strip()
    s = re.sub(r"^(?:чыгуначная станцыя|станцыя|станция|gara|stația|halta)\s+", "", s, flags=re.I)
    s = re.sub(r"\s+(?:railway station|station)$", "", s, flags=re.I)
    return s


def wd_stations(cc):
    """ESR code -> {lon, lat, name (the country's language), ru, en}."""
    p = raw(cc) / "wd_stations.json"
    if not p.exists():
        return {}
    got = defaultdict(list)
    for r in json.loads(p.read_text("utf-8"))["rows"]:
        m = re.match(r"Point\(([-\d.]+) ([-\d.]+)\)", r.get("coord", ""))
        got[r["esr"]].append({"lon": float(m.group(1)) if m else None,
                              "lat": float(m.group(2)) if m else None,
                              "name": clean_label(r.get("lab")), "ru": clean_label(r.get("ru")),
                              "en": clean_label(r.get("en")), "q": r["s"]})
    out = {}
    for c, v in got.items():
        if len({x["q"] for x in v}) == 1:
            out[c] = v[0]
    return out


# ================================================================ outline and clip

def outline(cc, reach=True):
    """The country as this build draws it: OSM's boundary less CC[cc]["minus"] (Moldova:
    Transnistria). With `reach`, plus REACH_DEG into the neighbours (Natural Earth 10m)."""
    import shapely
    from shapely.geometry import shape
    from shapely.ops import unary_union
    base = shapely.make_valid(shape(json.loads((raw(cc) / CC[cc]["boundary"]).read_text("utf-8"))))
    minus = [shapely.make_valid(shape(json.loads((raw(cc) / fn).read_text("utf-8"))))
             for fn in CC[cc]["minus"]]
    out = base
    for m in minus:
        out = out.difference(m)
    if reach:
        nb = []
        for f in json.loads(NE.read_text("utf-8"))["features"]:
            if (f["properties"].get("ISO_A2_EH") or "").lower() in CC[cc]["reach_into"]:
                nb.append(shape(f["geometry"]))
        near = base.buffer(REACH_DEG).intersection(unary_union(nb).buffer(0.02)).difference(base)
        out = out.union(near)
    polys = [g for g in shapely.get_parts(out) if g.geom_type in ("Polygon", "MultiPolygon")]
    return shapely.multipolygons(shapely.get_parts(shapely.geometrycollections(polys)))


def write_outline(cc):
    from shapely.geometry import mapping
    shp = outline(cc, reach=False)
    what = {"by": "Belarus as noritetsu draws it: OSM relation 59065",
            "md": "Moldova as noritetsu draws it: OSM relation 58974, Transnistria included"}[cc]
    (raw(cc) / "outline.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"what": what}, "geometry": mapping(shp)}]}), "utf-8")
    print(f"--outline: {shp.area:.3f} sq deg -> {raw(cc) / 'outline.geojson'}")


# OSM route relations of trains that no longer run, left out of the clipped extract: as OSM
# routes they would draw a line and keep the timetable from greying the track ("unknown": a
# route runs there, so an operator may be missing from the feed). Each checked against the
# timetable (md_sources.md, by_sources.md).
STALE_RELS = {
    "by": {
        2637899: "Крычаў - Шасцёраўка: no train in poezdato.net's timetable calls at Шестеровка",
        1164815: "Praha — Moskva: no such train (none to or from Poland at all in poezdato.net)",
        1636400: "Rīga - Minsk: no such train in poezdato.net's timetable",
    },
    "md": {
        18163097: "804Ц Bender - Chișinău, suspended 2025-01-13 (merstren.md)",
        6819989: "6931 Bălți Slobozia - Ocnița, suspended 2024-01-19 (merstren.md)",
        16001954: "Чернівці - Ларга over CFM's Lipcani track: Ukraine's Larha trains come "
                  "from Kamianets-Podilskyi (poizdato.net), none from Chernivtsi",
    },
}


def clip(cc):
    """data/proc/<cc> from data/proc/<cc>/full (the extract as extract.py wrote it; moved there
    on the first run after an extract): what lies in `outline()` kept. A way goes if at least
    half its nodes are outside, a stop if it is, a relation if no member is left."""
    import numpy as np
    import shapely
    P = proc(cc)
    full = P / "full"
    names = ("ways.pkl", "rels.pkl", "stops.pkl", "infra.pkl", "coords.npz")
    stamp = P / "clip_stamp.json"
    fresh = (P / "ways.pkl").exists() and (
        not stamp.exists() or (P / "ways.pkl").stat().st_mtime > stamp.stat().st_mtime + 5)
    if fresh:
        full.mkdir(exist_ok=True)
        for fn in names:
            os.replace(P / fn, full / fn)
        print(f"clip: the extract in {P} moved to {full}")

    def rd(fn):
        with open(full / fn, "rb") as f:
            return pickle.load(f)
    ways, rels, stops, infra = (rd(f) for f in names[:4])
    with np.load(full / "coords.npz") as c:
        cid, cx, cy = c["id"], c["x"], c["y"]
    shp = outline(cc)
    shapely.prepare(shp)
    out = ~shapely.contains_xy(shp, cx / 1e7, cy / 1e7)
    outside = set(cid[out].tolist())
    known = set(cid.tolist())
    keep_w, cut = {}, Counter()
    n_trim = 0
    for wid, (tags, nodes) in ways.items():
        ns = [int(n) for n in nodes if int(n) in known]
        if ns and 2 * sum(n in outside for n in ns) >= len(ns):
            cut[tags.get("name") or "(unnamed)"] += 1
            continue
        # A long way over the border (one way of 37 nodes ran 20 km into Russia from
        # Osinovka) keeps its inside run and one node beyond each end, not its far reaches.
        ins = [int(n) not in outside for n in nodes]
        if not all(ins) and any(ins):
            i0 = ins.index(True)
            i1 = len(ins) - 1 - ins[::-1].index(True)
            if i0 > 1 or i1 < len(ins) - 2:
                nodes = nodes[max(0, i0 - 1):i1 + 2]
                n_trim += 1
        if len(nodes) >= 2:
            keep_w[wid] = (tags, nodes)
    print(f"clip: {n_trim} ways over the edge trimmed to their inside run")
    keep_s = {k: v for k, v in stops.items() if shapely.contains_xy(shp, v[1], v[2])}
    kept = {("w", k) for k in keep_w} | {("n", k) for k in keep_s}
    stale = STALE_RELS.get(cc, {})
    for k in stale:
        if k in rels:
            print(f"clip: stale route relation {k} left out: {stale[k]}")
    routes = {k for k, (tags, members) in rels.items()
              if tags.get("type") == "route" and k not in stale
              and any((t, r) in kept for t, r, _ in members)}
    keep_r = {k: v for k, v in rels.items()
              if k in routes or any(t == "r" and r in routes for t, r, _ in v[1])}
    gone = set(ways) - set(keep_w)
    keep_i = {k: v for k, v in infra.items()
              if not (any(t == "w" and r in gone for t, r, _ in v[1])
                      and not any(t == "w" and r in keep_w for t, r, _ in v[1]))}
    if cc == "by":
        keep_r.update(per_train_masters(keep_r))
    for name, v in cut.most_common(20):
        print(f"  cut {v:5d}  {name}")
    print(f"clip: kept {len(keep_w)}/{len(ways)} ways, {len(keep_s)}/{len(stops)} stops, "
          f"{len(keep_r)}/{len(rels)} relations, {len(keep_i)}/{len(infra)} infra relations")
    for fn, obj in (("ways.pkl", keep_w), ("rels.pkl", keep_r), ("stops.pkl", keep_s),
                    ("infra.pkl", keep_i)):
        tmp = P / (fn + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        os.replace(tmp, P / fn)
    tmp = P / "coords.tmp.npz"
    np.savez_compressed(tmp, id=cid, x=cx, y=cy)
    os.replace(tmp, P / "coords.npz")
    stamp.write_text(json.dumps({"clipped_from": str(full), "ways": len(keep_w)}), "utf-8")


PER_TRAIN = re.compile(r"^Цягнік\s*№\s*(\d{4})\s*:\s*(.+?)\s*=>\s*(.+?)\s*$")


def per_train_masters(rels):
    """OSM Belarus maps the Grodno and Lida regional trains one relation per train, with no
    route_master ("Цягнік №6252: Гродна => Ліда", "Цягнік №6251: Ліда => Гродна"), and
    build_model groups orphan routes by (operator, network, ref, name), so every train became
    its own line: 26 lines for four services. Here the four-digit (regional, a line by
    rules/by.py) per-train relations between the same two places, either way, are put under one
    route_master of our own, named "A — B" (the two places as OSM names them, in the order
    the lowest-numbered train runs), so they build as one line. Its id is fixed by the two
    names (9e15 + a hash), so it is stable across builds."""
    import hashlib
    groups = defaultdict(list)
    # A route_master of its own per train ("Цягнік №6252", one route) is no grouping: it goes,
    # and its route joins the others. A real route_master's routes stay with it.
    per_train_master = {k for k, (t, _mem) in rels.items() if t.get("type") == "route_master"
                        and re.match(r"^Цягнік\s*№\s*\d{4}\s*$", t.get("name") or "")}
    claimed = {r for k, (t, mem) in rels.items() if t.get("type") == "route_master"
               and k not in per_train_master for ty, r, _ in mem if ty == "r"}
    for k in per_train_master:
        rels.pop(k, None)
    for rid, (tags, _mem) in rels.items():
        if tags.get("type") != "route" or rid in claimed:
            continue
        m = PER_TRAIN.match(tags.get("name") or "")
        if m:
            groups[frozenset((m.group(2), m.group(3)))].append((int(m.group(1)), rid, m))
    out = {}
    for key, rows in groups.items():
        rows.sort()
        _n, _rid, m = rows[0]
        a, b = m.group(2), m.group(3)
        h = int(hashlib.blake2b("|".join(sorted(key)).encode(), digest_size=6).hexdigest(), 16)
        mid = 9_000_000_000_000_000 + h % 1_000_000_000_000
        out[mid] = ({"type": "route_master", "route_master": "train", "name": f"{a} — {b}",
                     "network": "Беларуская чыгунка", "operator": "Беларуская чыгунка",
                     "noritetsu:made_from": ";".join(str(r) for _n, r, _m in rows)},
                    [("r", r, "") for _n, r, _m in rows])
    if out:
        print(f"clip: {sum(len(v[1]) for v in out.values())} per-train relations grouped under "
              f"{len(out)} route_masters of our own ({len(per_train_master)} one-train "
              f"route_masters left out)")
    return out


def crossings(cc):
    """Where rail track crosses the country's boundary (OSM's, before any minus), from the
    unclipped extract: the crossing point, the way, the neighbour (Natural Earth), and the OSM
    route relations over that way. For choosing BORDER and BORDER_POINTS."""
    import numpy as np
    from shapely.geometry import LineString, Point, shape
    full = proc(cc) / "full"
    src = full if (full / "ways.pkl").exists() else proc(cc)
    with open(src / "ways.pkl", "rb") as f:
        ways = pickle.load(f)
    with open(src / "rels.pkl", "rb") as f:
        rels = pickle.load(f)
    with np.load(src / "coords.npz") as c:
        xy = dict(zip(c["id"].tolist(), zip((c["x"] / 1e7).tolist(), (c["y"] / 1e7).tolist())))
    base = shape(json.loads((raw(cc) / CC[cc]["boundary"]).read_text("utf-8")))
    edge = base.boundary
    minus = [shape(json.loads((raw(cc) / fn).read_text("utf-8"))) for fn in CC[cc]["minus"]]
    nbs = []
    for f in json.loads(NE.read_text("utf-8"))["features"]:
        c2 = (f["properties"].get("ISO_A2_EH") or "").lower()
        if c2 in CC[cc]["reach_into"]:
            nbs.append((c2, shape(f["geometry"])))
    on_way = defaultdict(set)
    for rid, (tags, members) in rels.items():
        if tags.get("type") == "route" and tags.get("route") in ("train", "light_rail"):
            for t, r, _role in members:
                if t == "w":
                    on_way[r].add(f"{rid} {tags.get('name') or tags.get('ref') or ''}")
    for wid, (tags, nodes) in ways.items():
        if tags.get("railway") not in ("rail", "narrow_gauge") or tags.get("service"):
            continue
        pts = [xy[int(n)] for n in nodes if int(n) in xy]
        if len(pts) < 2:
            continue
        ls = LineString(pts)
        if not ls.intersects(edge):
            continue
        hit = ls.intersection(edge)
        for p in getattr(hit, "geoms", [hit]):
            if p.geom_type != "Point":
                continue
            nb = min(nbs, key=lambda kv: kv[1].distance(p))[0] if nbs else "?"
            inner = any(m.buffer(0.001).contains(p) for m in minus)
            print(f"{p.x:.6f} {p.y:.6f}  way {wid}  -> {nb}{' (minus area)' if inner else ''}  "
                  f"{tags.get('name') or ''} {tags.get('usage') or ''}")
            for r in sorted(on_way.get(wid, ()))[:8]:
                print(f"      {r}")


# ================================================================ names

GENERIC = {"пасажырскі", "пасажырская", "пасажырскае", "пасс", "пас", "пассажирский",
           "пассажирская", "пассажирское", "галоўны", "главный", "главная", "оп", "о", "п",
           "платформа", "пл", "пункт", "остановочный", "прыпынак", "прыпыначны", "станцыя",
           "станция", "ст", "рзд", "разъезд", "раз'езд", "рз", "блокпост", "бп", "обп", "пп",
           "вокзал", "station", "halt", "railway", "gara", "halta", "stația", "statia", "оп.",
           "ост", "эксп", "перев", "стык"}
ROMAN = {"i": "1", "і": "1", "ii": "2", "іі": "2", "iii": "3", "ііі": "3", "iv": "4"}
VOWELS = set("аеєиіїйоуўюяыэёьъ'")


def _tokens(name):
    import unicodedata
    s = unicodedata.normalize("NFKC", name or "").lower()
    for a, b in (("ё", "е"), ("ґ", "г"), ("’", "'"), ("ʼ", "'"), ("`", "'"), ("«", ""),
                 ("»", ""), ("\"", "")):
        s = s.replace(a, b)
    s = re.sub(r"\(([^)]*)\)", " ", s)
    toks = [t for t in re.split(r"[\s\-–—.,/№]+", s) if t]
    out = []
    for t in toks:
        t = ROMAN.get(t, t)
        if t in GENERIC:
            continue
        out.append(t)
    return out


def name_keys(name):
    """Keys for one station name, strongest first: the whole name folded to Latin
    (rinf.norm), the name less generic words and brackets ("Минск-Пасс." -> "minsk"), and that
    base's consonant skeleton, which a Russian and a Belarusian spelling of one place often
    share ("Молодечно" / "Маладзечна" -> млдчн / млдзчн do not; "Лида" / "Ліда" do)."""
    from rinf import norm
    full = norm(name)
    toks = _tokens(name)
    base = norm("".join(toks))
    sk = "".join(c for c in "".join(toks) if c not in VOWELS and c.isalnum())
    out = []
    for k in (full, base, "~" + sk if len(sk) >= 3 else ""):
        if k and k not in out:
            out.append(k)
    return out


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


# Moldova: Book 1 writes Moldovan places in a Russian rendering of the Romanian ("Кишинэу",
# "Унгень", "Бэлць-Слобозия"), which neither OSM's name:ru ("Кишинёв") nor a plain Cyrillic fold
# meets. Both sides are folded to one rough Latin form and compared letter by letter.
MD_CYR = dict(zip("абвгдежзийклмнопрстуфхцчшщъыьэюяё",
                  ["a", "b", "v", "g", "d", "e", "j", "z", "i", "i", "c", "l", "m", "n", "o", "p",
                   "r", "s", "t", "u", "f", "h", "t", "c", "s", "s", "", "i", "", "a", "iu", "ia",
                   "io"]))
MD_GENERIC = {"оп", "ост", "пункт", "рзд", "эксп", "пп", "km", "statia", "halta", "gara",
              "ост.", "платформа"}


def tidy_name(nm):
    """A station name as shown: an editor's bracketed note left out ("Pelinia
    (localitate-stație de cale ferată)" -> "Pelinia")."""
    s = re.sub(r"\s*\((?:localitate|sta[țţt]i|halt|gar[ăa]|cale ferat|s\.t\.v\.c|direc)[^)]*\)",
               "", nm or "", flags=re.I)
    return s.strip() or (nm or "")


def _numbers(name):
    s = (name or "").lower()
    s = re.sub(r"\b(iv|iii|ii|i)\b", lambda m: str({"i": 1, "ii": 2, "iii": 3, "iv": 4}[m.group(1)]), s)
    s = re.sub(r"(?<![а-яa-z])(ііі|іі|і)(?![а-яa-z])",
               lambda m: str(len(m.group(1))), s)
    return sorted(re.findall(r"\d+", s))


def md_latin(name):
    import unicodedata
    s = (name or "").lower().replace("км", "km")
    s = re.sub(r"\([^)]*\)", " ", s)
    s = "".join(MD_CYR.get(c, c) for c in s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    for a, b in (("ch", "c"), ("gh", "g"), ("sh", "s"), ("ts", "t"), ("â", "a"), ("ea", "a"),
                 ("ia", "a"), ("ie", "e"), ("iu", "u"), ("io", "o"), ("y", "i"), ("w", "v"),
                 ("ei", "e"), ("ii", "i")):
        s = s.replace(a, b)
    toks = [t for t in re.split(r"[^a-z0-9]+", s) if t and t not in MD_GENERIC]
    return "".join(toks)


def md_fuzzy(name, cands_names):
    """The best of [(key, name, ...)] by md_latin likeness to `name`, if 0.75 or more."""
    from difflib import SequenceMatcher
    a = md_latin(name)
    nums = _numbers(name)
    best, br = None, 0.0
    for c in cands_names:
        rs = [SequenceMatcher(None, a, md_latin(n)).ratio() for n in c[1]
              if _numbers(n) == nums]
        r = max(rs, default=0.0)
        if r > br:
            best, br = c, r
    # a name that is only a number ("Ост. пункт 18 км" / "18 Km") must match it exactly
    ok = br >= 0.75 and (len(a) >= 3 or (a.isdigit() and br == 1.0))
    return (best, br) if ok else (None, br)


def en_check(cc, shown, ru, en):
    """Does an English name read as a romanisation of the shown or the Russian name? A check,
    never a source of names. Moldova's names are Latin already: no English name."""
    if cc == "md" or not en or re.search(r"\btrain\b|[′ʹ]", en, re.I):
        return False
    import ru_register as rr
    from difflib import SequenceMatcher
    from rinf import norm
    if rr.en_score(ru, en) >= rr.EN_AGREE:
        return True
    a, b = norm(" ".join(_tokens(shown))), norm(" ".join(_tokens(en)))
    for x, y in (("dz", "z"), ("ts", "c"), ("ch", "c"), ("sh", "s"), ("zh", "z"), ("kh", "h"),
                 ("y", "i"), ("j", "i"), ("w", "v")):
        a, b = a.replace(x, y), b.replace(x, y)
    return bool(a and b) and SequenceMatcher(None, a, b).ratio() >= 0.88


# ================================================================ Book 1

def book1(cc):
    import ru_register as rr
    saved = rr.ROADS
    rr.ROADS = {CC[cc]["sheet"]: (CC[cc]["road"], CC[cc]["operator"])}
    try:
        return rr.book1()
    finally:
        rr.ROADS = saved


def book1_points(cc):
    """{ESR code: Russian name}, every point on the country's sheet."""
    import ru_register as rr
    secs, _n = book1(cc)
    return {p["esr"]: rr.clean(p["name"]) for s in secs for p in s["points"]}


# ================================================================ timetable: poezdato.net (by)

POEZDATO = "https://poezdato.net"
CRAWL_DELAY = 1.2
SEED_STATIONS = ["minsk-pass", "gomel-pass", "brest-tsentralnyj", "grodno", "vitebsk",
                 "mogilev-1", "orsha-tsentralnaya", "baranovichi-polesskie", "molodechno",
                 "polotsk", "zhlobin", "kalinkovichi", "luninets", "osipovichi-1", "lida",
                 "volkovysk", "slutsk", "krichev-1", "pinsk", "bobrujsk", "krulevshchizna",
                 "zhabinka", "rechitsa", "borisov", "unecha"]


def _get(url, tries=3):
    import urllib.request
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as e:                                   # noqa: BLE001
            if getattr(e, "code", None) in (404, 410):
                return None
            print(f"  retry {i}: {url} {type(e).__name__} {str(e)[:80]}", flush=True)
            time.sleep(10 * (i + 1))
    return None


def _slug_file(kind, slug):
    import urllib.parse
    s = re.sub(r'[\\/:*?"<>|]', "_", urllib.parse.unquote(slug))
    return raw("by") / "poezdato" / kind / f"{s}.html"


def crawl_poezdato(limit=None):
    """Belarusian station pages, then every train page they list. A station page's own links
    name further stations ("/raspisanie-po-stancyi/<slug>/" with the station's name as text);
    one is followed when its name matches a point of Book 1's Belarusian sheet (Russian
    spellings both). Pages already on disk are read, not fetched again."""
    import html as H

    def base(n):
        k = name_keys(n)
        return k[1] if len(k) > 1 else k[0]
    import ru_register as rr
    keys = {base(n) for n in book1_points("by").values() if n}
    # The second round reads only stations at the ends of tariff sections (junctions and
    # branch ends): a branch shuttle that reaches no hub still reaches one of those. All 761
    # Belarusian stations the trains call at would take an hour more at the site's pace.
    secs, _n = book1("by")
    end_keys = {base(rr.clean(p["name"])) for s in secs for p in (s["points"][:1] + s["points"][-1:])}
    todo, seen, trains = list(SEED_STATIONS), set(SEED_STATIONS), {}
    n_fetch = 0

    def fetch(kind, slug):
        nonlocal n_fetch
        p = _slug_file(kind, slug)
        if p.exists():
            return p.read_text("utf-8")
        if limit and n_fetch >= limit:
            return None
        t = _get(f"{POEZDATO}/{kind}/{slug}/")
        n_fetch += 1
        time.sleep(CRAWL_DELAY)
        if t:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(t, "utf-8")
        if n_fetch % 50 == 0:
            print(f"  {n_fetch} fetched; stations seen {len(seen)}, trains {len(trains)}",
                  flush=True)
        return t
    while todo:
        slug = todo.pop(0)
        t = fetch("raspisanie-po-stancyi", slug)
        if not t:
            continue
        for kind, s in re.findall(r'href="/(raspisanie-elektrichki|raspisanie-poezda)/([^"/]+)/"', t):
            trains.setdefault(s, kind)
        for s, txt in re.findall(r'href=["\']/raspisanie-po-stancyi/([^"\'/]+)/["\']>\s*([^<]+?)\s*<', t):
            if s in seen:
                continue
            nm = H.unescape(txt).strip()
            if nm and base(nm) in keys:
                seen.add(s)
                todo.append(s)
        if not todo:
            # Then every train page those stations list, and the Belarusian stations those
            # trains call at whose own pages have not been read: a branch's halts list
            # trains no hub does (a shuttle that never reaches one).
            for s2, kind in sorted(trains.items()):
                tt = fetch(kind, s2)
                if not tt:
                    continue
                for s3, txt in re.findall(r'href="/raspisanie-po-stancyi/([^"/]+)/">\s*([^<]+?)\s*<', tt):
                    nm = H.unescape(txt).strip()
                    if s3 not in seen and nm and base(nm) in end_keys:
                        seen.add(s3)
                        todo.append(s3)
            print(f"  round done: stations {len(seen)}, trains {len(trains)}, {len(todo)} new "
                  f"stations to read", flush=True)
    print(f"stations: {len(seen)}; trains listed: {len(trains)}", flush=True)
    print(f"crawl: {n_fetch} pages fetched", flush=True)


def parse_poezdato(path):
    """One poezdato train page: number, kind, calls (name, station slug, times) in order, and
    the dates its running calendar marks."""
    import html as H
    from datetime import date, timedelta
    t = path.read_text("utf-8", errors="replace")
    title = re.search(r"<title>(.*?)</title>", t, re.S)
    title = H.unescape(title.group(1)).strip() if title else ""
    m = re.search(r"(?:Поезд|Электричка)\s+(\S+)", title)
    num = m.group(1) if m else ""
    i0 = t.find("<th>Станция</th>")
    if i0 < 0:
        i0 = t.find("Расписание</")
    i1 = t.find("График движения", i0)
    body = t[i0:i1 if i1 > 0 else len(t)]
    calls = []
    for m in re.finditer(r'<tr[^>]*>\s*<td>\s*(?:<a href="/raspisanie-po-stancyi/([^"]+)/">)?'
                         r'\s*([^<]+?)\s*(?:</a>)?\s*</td>(.*?)</tr>', body, re.S):
        slug, name, rest = m.group(1) or "", H.unescape(m.group(2)).strip(), m.group(3)
        times = re.findall(r'_time">\s*(\d{1,2})\.(\d{2})', rest)
        calls.append({"name": name, "slug": slug, "times": [f"{h}:{mm}" for h, mm in times]})
    days = []
    y0 = date(2026, 1, 1)
    for m in re.finditer(r"<td id='i_(\d+)' class=\"[^\"]*\"><span class=\"([^\"]*)\">\d+</span>", t):
        if "ui-state-active" in m.group(2):
            days.append((y0 + timedelta(days=int(m.group(1)))).isoformat())
    return {"num": num, "title": title, "kind": path.parent.name, "calls": calls,
            "days": sorted(set(days))}


# ================================================================ timetable: merstren.md (md)

MERSTREN = "https://www.merstren.md/orar.js"
# Places merstren leaves out that the Kyiv trains pass, added as calls so the timetable check
# finds their path (a non-stop run longer than 1.6x crow-fly is stepped over): CFM's own feed
# lists Ocnița and Pîrlița for 351Щ/351Ф, and the only track from Mohyliv-Podilskyi to Bălți
# runs through Vălcineț and Ocnița, from Bălți to Chișinău through Ungheni (Book 1: 39-006 ends
# at Ungheni, 39-007 starts there).
MD_EXTRA_CALLS = {
    "tr351": [("Vălcineț", ["Ocnița"]), ("Bălți Oraș", ["Ungheni", "Pîrlița"])],
    "tr352": [("Chișinău", ["Pîrlița", "Ungheni"]), ("Bălți Oraș", ["Ocnița"])],
    "uz100k": [("Bălți Oraș", ["Ocnița", "Vălcineț"])],
    "uz99k": [("Mohyliv-Podilskyi", ["Vălcineț", "Ocnița"])],
}
# The Kyiv - Chișinău train runs on to Revaca (Chișinău airport's shuttle, from
# 2026-09; merstren: "Revaca 10:20 · shuttle ≈ 5 min", only on that direction).
MD_TAIL = {"tr351": [("Revaca", "10:20")]}


def fetch_merstren():
    import urllib.request
    req = urllib.request.Request(MERSTREN, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        t = r.read().decode("utf-8")
    (raw("md") / "merstren_orar.js").write_text(t, "utf-8")
    print(f"--merstren: {len(t)} bytes -> {raw('md') / 'merstren_orar.js'}")


def parse_merstren():
    """merstren's DATA.trains as [{id, label, zile, stops: [{name, arr, dep}]}]. The file is
    JavaScript object literals; each train is read field by field."""
    t = (raw("md") / "merstren_orar.js").read_text("utf-8")
    t = t[t.find("const DATA"):t.find("end DATA")]
    trains = []
    for m in re.finditer(r"\{\s*id:'([^']+)'(.*?)stops:\s*\[(.*?)\]\s*\}", t, re.S):
        tid, head, body = m.group(1), m.group(2), m.group(3)
        zile = re.search(r"zile:'([^']*)'", head)
        label = re.search(r"label:'([^']*)'", head)
        stops = []
        for s in re.finditer(r"\{name:'([^']+)'([^}]*)\}", body):
            arr = re.search(r"arr:'([\d:]+)'", s.group(2))
            dep = re.search(r"dep:'([\d:]+)'", s.group(2))
            stops.append({"name": s.group(1).replace("*", "").strip(),
                          "arr": arr.group(1) if arr else "", "dep": dep.group(1) if dep else ""})
        trains.append({"id": tid, "label": label.group(1) if label else tid,
                       "zile": zile.group(1) if zile else "", "stops": stops})
    return trains


def merstren_days(zile, start, n=61):
    """Dates a merstren train runs on, over n days from start: zilnic (daily), luni-vineri
    (Monday-Friday), "vineri, sâmbătă și duminică" (Friday-Sunday)."""
    from datetime import timedelta
    z = zile.lower()
    if "zilnic" in z:
        wd = set(range(7))
    elif "luni" in z and "vineri" in z and "–" in z or "luni-vineri" in z:
        wd = set(range(5))
    else:
        wd = {i for i, w in enumerate(("luni", "marți", "miercuri", "joi", "vineri", "sâmbătă",
                                        "duminică")) if w in z}
    return [(start + timedelta(days=i)).isoformat() for i in range(n)
            if (start + timedelta(days=i)).weekday() in wd]


# ================================================================ the timetable feed

SKIP_KM = 150.0
MAX_HOP_KM = 900.0


def timetable(cc):
    """The crawled train pages (by) or merstren's data (md) as a GTFS feed in
    data/raw/gtfs/<cc>/, and data/raw/<cc>/timetable_calls.json + timetable_pairs.json (the
    places trains call at, and consecutive calls of running trains), which --convert reads.

    Each call's name is matched to OSM's rail stations (every name tag), Wikidata's stations
    and Book 1's points where OSM has their ESR node (Russian names: poezdato spells in
    Russian), by name_keys; of the places a name finds, a dynamic programme picks one per call
    (or none, at SKIP_KM) so the train's path is shortest (ua_register.timetable). For
    Moldova, stops abroad (Iași, Mohyliv-Podilskyi) come from CFM's GTFS and the neighbours'
    built stations."""
    import csv
    import io
    import zipfile
    from datetime import date
    ost = load_osm_stations(cc)
    places = [(s["lon"], s["lat"], s["name"]) for s in ost]
    nidx = NameIndex()
    for i, s in enumerate(ost):
        for nm in s["names"]:
            nidx.add(nm, s["lon"], s["lat"], i)
    for c, w in wd_stations(cc).items():
        if w["lon"] is not None:
            for nm in {w["name"], w["ru"]} - {""}:
                places.append((w["lon"], w["lat"], w["name"] or w["ru"]))
                nidx.add(nm, w["lon"], w["lat"], len(places) - 1)
    esr = json.loads((raw(cc) / "osm_esr.json").read_text("utf-8"))
    for c, ru in book1_points(cc).items():
        rows = esr.get(c) or []
        if rows:
            r = rows[0]
            places.append((r["lon"], r["lat"], r["names"].get("name") or ru))
            nidx.add(ru, r["lon"], r["lat"], len(places) - 1)
    abroad_names = set()
    if cc == "md":
        # stops abroad: CFM's feed (Iași-Socola, Ukraine's), then Romania's and Ukraine's builds
        for zp in sorted(raw("md").glob("*.gtfs.zip")):
            with zipfile.ZipFile(zp) as z:
                for r in csv.DictReader(io.TextIOWrapper(z.open("stops.txt"), "utf-8-sig")):
                    places.append((float(r["stop_lon"]), float(r["stop_lat"]), r["stop_name"]))
                    nidx.add(r["stop_name"], float(r["stop_lon"]), float(r["stop_lat"]),
                             len(places) - 1)
        for nb in ("ro", "ua"):
            p = ROOT / "dist" / "data" / nb / "stations.json"
            if p.exists():
                for sid, s in json.loads(p.read_text("utf-8"))["stations"].items():
                    if s.get("j"):
                        continue
                    places.append((s["x"], s["y"], s["n"]))
                    nidx.add(s["n"], s["x"], s["y"], len(places) - 1)
                    abroad_names.add(s["n"])
        start = date.today()
        trains = []
        for tr in parse_merstren():
            st = list(tr["stops"])
            for after, adds in MD_EXTRA_CALLS.get(tr["id"], ()):
                i = next((k for k, s in enumerate(st) if s["name"] == after), None)
                if i is not None:
                    st[i + 1:i + 1] = [{"name": a, "arr": "", "dep": ""} for a in adds]
            for add, tm in MD_TAIL.get(tr["id"], ()):
                st.append({"name": add, "arr": tm, "dep": ""})
            trains.append({"num": tr["label"], "title": tr["label"], "slug": tr["id"],
                           "kind": "rail",
                           "calls": [{"name": s["name"], "times": [x for x in (s["arr"], s["dep"])
                                                                   if x]} for s in st],
                           "days": merstren_days(tr["zile"], start)})
    else:
        files = sorted(f for f in (raw(cc) / "poezdato").glob("raspisanie-*/*.html")
                       if f.parent.name in ("raspisanie-elektrichki", "raspisanie-poezda"))
        trains = []
        for f in files:
            t = parse_poezdato(f)
            t["slug"] = f.stem
            trains.append(t)
    stat = Counter()
    out_trips = []
    for t in trains:
        calls = t["calls"]
        cands = []
        for c in calls:
            got = nidx.find(c["name"])
            seen, cc_ = [], []
            for lon, lat, ref, _r in got:
                if all(dist_m(lon, lat, x, y) > 300 for x, y in seen):
                    seen.append((lon, lat))
                    cc_.append(ref)
            cands.append(cc_[:80])
        n = len(calls)
        best = [dict() for _ in range(n)]
        for i in range(n):
            for r in cands[i]:
                bc, bp = SKIP_KM * i, None
                for j in range(max(0, i - 5), i):
                    for r2, (c2, _p) in best[j].items():
                        d = dist_m(*places[r][:2], *places[r2][:2]) / 1000
                        if d > MAX_HOP_KM:
                            continue
                        v = c2 + d + SKIP_KM * (i - j - 1)
                        if v < bc:
                            bc, bp = v, (j, r2)
                best[i][r] = (bc, bp)
        end, ec = None, SKIP_KM * n
        for i in range(n):
            for r, (c, _p) in best[i].items():
                v = c + SKIP_KM * (n - 1 - i)
                if v < ec:
                    end, ec = (i, r), v
        chosen = {}
        while end is not None:
            i, r = end
            chosen[i] = r
            end = best[i][r][1]
        stat["calls"] += n
        stat["matched"] += len(chosen)
        stat["no candidate"] += sum(1 for c in cands if not c)
        if len(chosen) < 2:
            stat["trains with fewer than 2 calls matched"] += 1
            continue
        out_trips.append((t, [(i, chosen[i]) for i in sorted(chosen)]))
    print(f"timetable: {len(trains)} trains; {dict(stat)}")
    unmatched = Counter()
    for t in trains:
        ch = next((c for tt, c in out_trips if tt is t), [])
        got = {i for i, _r in ch}
        for i, c in enumerate(t["calls"]):
            if i not in got:
                unmatched[c["name"]] += 1
    print("  most frequent unmatched calls: " + ", ".join(
        f"{k} ({v})" for k, v in unmatched.most_common(40)))
    gdir = ROOT / "data" / "raw" / "gtfs" / cc
    gdir.mkdir(parents=True, exist_ok=True)
    used = sorted({r for _t, ch in out_trips for _i, r in ch})
    calls_xy = sorted({(round(places[r][0], 6), round(places[r][1], 6)) for r in used})
    (raw(cc) / "timetable_calls.json").write_text(json.dumps(calls_xy), "utf-8")
    at = {xy: i for i, xy in enumerate(calls_xy)}
    pairs = set()
    for t, ch in out_trips:
        if not t["days"]:
            continue
        for (_i, r1), (_j, r2) in zip(ch, ch[1:]):
            a = at[(round(places[r1][0], 6), round(places[r1][1], 6))]
            b = at[(round(places[r2][0], 6), round(places[r2][1], 6))]
            if a != b:
                pairs.add((min(a, b), max(a, b)))
    (raw(cc) / "timetable_pairs.json").write_text(json.dumps(sorted(pairs)), "utf-8")

    def tbl(header, rows):
        b = io.StringIO()
        w = csv.writer(b, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
        return b.getvalue()

    def secs(x):
        h, m = x.split(":")
        return int(h) * 3600 + int(m) * 60
    stops_rows = [(f"p{r}", places[r][2], f"{places[r][1]:.6f}", f"{places[r][0]:.6f}")
                  for r in used]
    routes, trips, st, cal = [], [], [], []
    for t, ch in out_trips:
        rid = t["slug"]
        rtype = "109" if t["kind"] == "raspisanie-elektrichki" else "2"
        routes.append((rid, "a", t["num"], t["title"][:120], rtype))
        trips.append((rid, rid, rid))
        prev = -1
        for seq, (i, r) in enumerate(ch, 1):
            tm = t["calls"][i]["times"]
            arr, dep = (tm[0], tm[-1]) if tm else ("", "")
            if arr:
                a, d = secs(arr), secs(dep)
                while a < prev:
                    a += 86400
                while d < a:
                    d += 86400
                prev = d
                arr = f"{a // 3600:02d}:{a % 3600 // 60:02d}:00"
                dep = f"{d // 3600:02d}:{d % 3600 // 60:02d}:00"
            st.append((rid, arr, dep, f"p{r}", seq))
        for d in t["days"]:
            cal.append((rid, d.replace("-", ""), 1))
    name = {"by": "by_poezdato.gtfs.zip", "md": "md_merstren.gtfs.zip"}[cc]
    src = {"by": ("noritetsu, from poezdato.net's train pages", "https://poezdato.net/", "ru"),
           "md": ("noritetsu, from merstren.md's timetable data", "https://www.merstren.md/",
                  "ro")}[cc]
    zp = gdir / name
    tmp = gdir / (name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("agency.txt", tbl(("agency_id", "agency_name", "agency_url", "agency_timezone"),
                                     [("a", CC[cc]["agency"], CC[cc]["url"], CC[cc]["tz"])]))
        z.writestr("stops.txt", tbl(("stop_id", "stop_name", "stop_lat", "stop_lon"), stops_rows))
        z.writestr("routes.txt", tbl(("route_id", "agency_id", "route_short_name",
                                      "route_long_name", "route_type"), routes))
        z.writestr("trips.txt", tbl(("route_id", "service_id", "trip_id"), trips))
        z.writestr("stop_times.txt", tbl(("trip_id", "arrival_time", "departure_time", "stop_id",
                                          "stop_sequence"), st))
        z.writestr("calendar_dates.txt", tbl(("service_id", "date", "exception_type"), cal))
        z.writestr("feed_info.txt", tbl(
            ("feed_publisher_name", "feed_publisher_url", "feed_lang", "feed_version"),
            [(*src, date.today().isoformat())]))
    os.replace(tmp, zp)
    print(f"timetable: {len(out_trips)} trains, {len(used)} stations, {len(cal)} train-days -> {zp}")


# ================================================================ the register

def section_ends(s, by_key):
    """The two end points a section header names, else the first and last point; plus the
    header's via names (ua_register.section_ends)."""
    import ru_register as rr
    hdr = rr.VIA.sub("", s["name"])
    hdr = re.sub(r"\((?:[^()]*ж\.\s*д\.?|[^()]*(?:ПАССАЖИР|пассажир)[^()]*)\)", "", hdr)
    ends = [x.strip() for x in re.split(r"\s+-\s+", hdr)]
    pa, pb = s["pts"][0], s["pts"][-1]
    if len(ends) == 2 and all(ends):
        ga = by_key.get(rr.nkey(ends[0])) or by_key.get(rr.nkey(rr.EXTRA_CODE.sub("", ends[0])))
        gb = by_key.get(rr.nkey(ends[1])) or by_key.get(rr.nkey(rr.EXTRA_CODE.sub("", ends[1])))
        pa, pb = ga or pa, gb or pb
    vias = []
    m = rr.VIA.search(s["name"])
    if m:
        via = re.split(r"\s*\(", m.group(1))[0].strip()
        for v in re.split(r",|\s+и\s+", via):
            v = re.sub(r"^(?:ст\.?|оп|о\.п\.)\s+", "", v.strip(), flags=re.I)
            vias.append(by_key.get(rr.nkey(v)))
    return pa, pb, vias


def border_points():
    """uopid -> {lon, lat, name}: RINF's (border_points.json) and BORDER_POINTS."""
    out = {p["id"][1:]: p for p in json.loads(
        (ROOT / "border_points.json").read_text("utf-8"))["points"]}
    for k, (lon, lat, name) in BORDER_POINTS.items():
        out[k] = {"lon": lon, "lat": lat, "name": name}
    return out


def convert(cc):
    import numpy as np
    import shapely
    from datetime import date
    from scipy.spatial import cKDTree
    import ru_register as rr
    road, operator = CC[cc]["road"], CC[cc]["operator"]
    secs, b1name = book1(cc)
    ops, b2name = rr.book2()
    print(f"Book 1 ({b1name}): {len(secs)} sections; Book 2 ({b2name}): {len(ops)} points")
    esr = json.loads((raw(cc) / "osm_esr.json").read_text("utf-8"))
    wd = wd_stations(cc)
    ost = load_osm_stations(cc)
    # every OSM station node's name tags by node id (for name:ro in Transnistria)
    osm_names = {r["id"]: r["names"] for r in
                 json.loads((raw(cc) / "osm_stations.json").read_text("utf-8"))}
    nidx = NameIndex()
    for i, s in enumerate(ost):
        for nm in s["names"]:
            nidx.add(nm, s["lon"], s["lat"], i)
    stat = defaultdict(int)

    # --- 0. stations Book 1 leaves out of the section they lie on
    for s in secs:
        for sid, after, code, name, km in INSERT.get(cc, []):
            if s["id"] != sid:
                continue
            i = next((k for k, p in enumerate(s["points"]) if p["esr"] == after), None)
            if i is not None and all(p["esr"] != code for p in s["points"]):
                s["points"].insert(i + 1, {"esr": code, "name": name, "km": [km, None, None]})
                stat["points Book 1 leaves out, inserted"] += 1

    # --- 1. each section's points, extra codes at the same km dropped (as ru_register)
    for s in secs:
        pts = []
        for p in s["points"]:
            p = dict(p, name=rr.clean(p["name"]), raw=p["name"], km0=p["km"][0])
            if pts and p["km0"] == pts[-1]["km0"] and rr.EXTRA_CODE.search(p["name"]):
                stat["extra code dropped"] += 1
                continue
            if pts and p["km0"] == pts[-1]["km0"] and rr.EXTRA_CODE.search(pts[-1]["name"]):
                stat["extra code dropped"] += 1
                pts[-1] = p
                continue
            pts.append(p)
        s["pts"] = pts
        s["km"] = (pts[-1]["km0"] or 0) - (pts[0]["km0"] or 0) if pts else 0
    raw_of = {}
    for s in secs:
        for p in s["pts"]:
            raw_of.setdefault(p["esr"], p)

    # --- 2. place the points: OSM's ESR node, else Wikidata's item, else an OSM station of a
    # matching name near the point's placed neighbours.
    pos, how, node_of = {}, {}, {}
    for c in raw_of:
        rows = esr.get(c) or []
        best = ([r for r in rows if r["rw"] in ("station", "halt")]
                or [r for r in rows if r["pt"] == "station"] or rows)
        if best:
            pos[c], how[c], node_of[c] = (best[0]["lon"], best[0]["lat"]), "esr", best[0]
    rounds = 6
    for _round in range(rounds):
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
                if not anchors and _round < rounds - 1:
                    continue

                def fits(lon, lat):
                    return all(dist_m(lon, lat, *a) / 1000 <= km + NAME_REACH_KM
                               for a, km in anchors)
                w = wd.get(c)
                if w and w["lon"] is not None and fits(w["lon"], w["lat"]):
                    pos[c], how[c] = (w["lon"], w["lat"]), "wikidata"
                    continue
                cands = [x for x in nidx.find(p["name"]) if fits(x[0], x[1])]
                if cands and anchors:
                    lon, lat, ref, _r = min(cands, key=lambda x: min(
                        dist_m(x[0], x[1], *a) for a, _k in anchors))
                    pos[c], how[c], node_of[c] = (lon, lat), "name", ost[ref]
                elif cc == "md" and anchors:
                    near_ = [(k, o["names"]) for k, o in enumerate(ost) if fits(o["lon"], o["lat"])]
                    got, _r = md_fuzzy(p["name"], near_)
                    if got:
                        o = ost[got[0]]
                        pos[c], how[c], node_of[c] = (o["lon"], o["lat"]), "name (Romanian)", o
    for c, (x, y, _nm) in EXPORT_AT.get(cc, {}).items():
        if c in raw_of:
            pos[c], how[c] = (x, y), "export code at the crossing"
            node_of.pop(c, None)
    codes = set(raw_of)
    print(f"points: {len(codes)}; placed by ESR {sum(1 for c in codes if how.get(c) == 'esr')}, "
          f"Wikidata {sum(1 for c in codes if how.get(c) == 'wikidata')}, by name "
          f"{sum(1 for c in codes if how.get(c) == 'name')}, by a Romanian name "
          f"{sum(1 for c in codes if how.get(c) == 'name (Romanian)')}, unplaced "
          f"{sum(1 for c in codes if c not in pos)}")

    # --- names: OSM's where the point is an OSM station, else Wikidata's label in the
    # country's language, else an OSM station of a matching name within 300 m, else Book 1's
    # own (Russian), counted.
    shown, en_of, name_how = {}, {}, defaultdict(int)
    near_tree = cKDTree(np.array([[s["lon"] * 0.67, s["lat"]] for s in ost]))
    for c, p in raw_of.items():
        n = node_of.get(c)
        nm, en = "", ""
        if n is not None and "names" in n and isinstance(n["names"], dict):
            if n["rw"] in ("station", "halt") and n["names"].get("name"):
                nm, en = n["names"]["name"], n["en"]
                name_how["OSM by ESR"] += 1
        elif n is not None:
            nm, en = n["name"], n["en"]
            name_how["OSM by name"] += 1
        if not nm and c in pos:
            ks = set(name_keys(p["name"]))
            js = near_tree.query_ball_point([pos[c][0] * 0.67, pos[c][1]], 0.003)
            for j in js:
                if any(set(name_keys(x)) & ks for x in ost[j]["names"]):
                    nm, en = ost[j]["name"], ost[j]["en"]
                    n = n or {"id": ost[j]["id"]}
                    name_how["OSM nearby"] += 1
                    break
            if not nm and cc == "md":
                got, _r = md_fuzzy(p["name"], [(j, ost[j]["names"]) for j in js])
                if got:
                    nm = ost[got[0]]["name"]
                    name_how["OSM nearby (Romanian)"] += 1
        if not nm and wd.get(c, {}).get("name"):
            nm = wd[c]["name"]
            name_how["Wikidata"] += 1
        if c in EXPORT_AT.get(cc, {}):
            nm = EXPORT_AT[cc][c][2]
            name_how["export code at a crossing"] += 1
        if not nm:
            nm = p["name"]
            name_how["Book 1 (Russian)"] += 1
        shown[c] = tidy_name(nm)
        if not en and wd.get(c, {}).get("en"):
            en = rr.clean_en(wd[c]["en"])
        if en_check(cc, nm, p["name"], en):
            en_of[c] = en
        elif cc == "md" and CYRILLIC.search(shown[c]):
            # Transnistria: OSM names its stations in Russian, as their signs are ("Бендеры-1",
            # "Тирасполь"); the English name is OSM's name:en, else its Romanian one ("Bender
            # 1", "Tiraspol"), the Latin spelling the rest of Moldova uses.
            nd = osm_names.get((n or {}).get("id")) or {}
            lat = tidy_name(en or nd.get("name:ro") or "")
            if lat and not CYRILLIC.search(lat):
                en_of[c] = lat
                name_how["English from name:en/name:ro (Transnistria)"] += 1
    print(f"names: {dict(name_how)}; English {len(en_of)}")

    # --- 3. inside the country as drawn
    shp = outline(cc, reach=False)
    shapely.prepare(shp)
    placed = sorted(c for c in codes if c in pos)
    xy = np.array([pos[c] for c in placed])
    in_out = dict(zip(placed, shapely.contains_xy(shp, xy[:, 0], xy[:, 1]).tolist()))
    for s in secs:
        pts = s["pts"]
        for i, p in enumerate(pts):
            c = p["esr"]
            if c in pos:
                continue
            # A border-junction code with no node ("Осиновка (эксп.)", "Гудогай-стык") is the
            # border itself: the line ends at its last real point and BORDER carries it on.
            if re.search(r"\(эксп|стык", p["raw"], re.I):
                in_out[c] = False
                continue
            prev = next((pts[j]["esr"] for j in range(i - 1, -1, -1) if pts[j]["esr"] in pos), None)
            nxt = next((pts[j]["esr"] for j in range(i + 1, len(pts)) if pts[j]["esr"] in pos), None)
            nb = [in_out[q] for q in (prev, nxt) if q is not None]
            # An unplaced point with a placed point on one side only is a line end nothing can
            # trace to ("Яссы-Сокола" across the border): left out.
            in_out[c] = in_out.get(c, True) and len(nb) == 2 and all(nb)

    for c in EXPORT_AT.get(cc, {}):
        in_out[c] = c in pos      # on the boundary itself, which contains_xy may not count

    def inside(c):
        return in_out.get(c, False)
    from shapely.geometry import shape as _shape
    minus = [_shape(json.loads((raw(cc) / fn).read_text("utf-8"))) for fn in CC[cc]["minus"]]

    def where_is(q):
        if q is None:
            return "unplaced, between points outside"
        if any(shapely.contains_xy(m, *q) for m in minus):
            return "Transnistria"
        return "abroad"
    pmr = (_shape(json.loads((raw(cc) / TRANSNISTRIA).read_text("utf-8")))
           if cc == "md" else None)

    def in_pmr(c):
        return pmr is not None and c in pos and bool(shapely.contains_xy(pmr, *pos[c]))

    # --- 4. which points are stops
    kx = 111.32 * math.cos(math.radians(50 if cc == "by" else 47))
    with open(proc(cc) / "stops.pkl", "rb") as f:
        stops = pickle.load(f)
    with open(proc(cc) / "rels.pkl", "rb") as f:
        rels = pickle.load(f)
    served_xy = []
    for tags, members in rels.values():
        if tags.get("type") == "route" and tags.get("route") == "train":
            for ty, ref, role in members:
                if ty == "n" and role.startswith(("stop", "platform")) and ref in stops:
                    served_xy.append((stops[ref][1], stops[ref][2]))
    n_osm = len(set(served_xy))
    tt = json.loads((raw(cc) / "timetable_calls.json").read_text("utf-8")) \
        if (raw(cc) / "timetable_calls.json").exists() else []
    served_xy += [(x, y) for x, y in tt]
    print(f"served places: {n_osm} OSM train route stops, {len(tt)} timetable stations")
    stree = cKDTree(np.array([[x * kx, y * 110.57] for x, y in set(served_xy)]))
    halts = [(s["lon"], s["lat"]) for s in ost if s["rw"] == "halt"]
    htree = cKDTree(np.array([[x * kx, y * 110.57] for x, y in halts])) if halts else None
    ttree = cKDTree(np.array([[x * kx, y * 110.57] for x, y in tt])) if tt else None

    def near(tree, c, m):
        q = pos.get(c)
        return bool(q and tree is not None and tree.query_ball_point(
            [q[0] * kx, q[1] * 110.57], m / 1000))
    tt_pairs = set()
    if (raw(cc) / "timetable_pairs.json").exists() and ttree is not None:
        tt_pairs = {tuple(p) for p in json.loads((raw(cc) / "timetable_pairs.json").read_text("utf-8"))}

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
        flag = rr.passenger_op(ops.get(c))
        if not flag:
            kind[c] = "none"
        elif near(stree, c, SERVED_M):
            kind[c] = "served"
        elif re.match(r"(?:ОП|О\.П\.)\s", p["raw"]) or near(htree, c, HALT_M):
            kind[c] = "halt"
        else:
            kind[c] = "flag"
    print(f"point kinds: {dict(Counter(kind.values()))}; of the served, "
          f"{sum(1 for c in kind if kind[c] == 'served' and near(ttree, c, SERVED_M))} by the "
          f"timetable")

    # --- 5. sections: owners, cut at the outline, unserved stretches
    owner = {}
    order = sorted(secs, key=lambda s: (rr.TYPE_RANK.get(s["type"], 9), s["id"]))
    for s in order:
        for a, b in zip(s["pts"], s["pts"][1:]):
            owner.setdefault(frozenset((a["esr"], b["esr"])), s["id"])
    where = defaultdict(list)
    for s in secs:
        for i, p in enumerate(s["pts"]):
            where[p["esr"]].append((s["id"], i, p["km0"] or 0))
    coarse = set()
    for s in secs:
        for a, b in zip(s["pts"], s["pts"][1:]):
            km = (b["km0"] or 0) - (a["km0"] or 0)
            at_b = {sid: (i, k) for sid, i, k in where[b["esr"]]}
            for sid, i, k in where[a["esr"]]:
                if sid == s["id"] or sid not in at_b:
                    continue
                j, kb = at_b[sid]
                if abs(j - i) >= 2 and abs(abs(kb - k) - km) <= 0.15 * km + 2:
                    coarse.add((s["id"], frozenset((a["esr"], b["esr"]))))
                    break
    rows, names, used, stop_codes = [], {}, set(), set()
    border_rows = []
    km_kept = defaultdict(float)
    for s in secs:
        if s["km"] == 0:
            stat["node-list sections dropped (all 0 km)"] += 1
            continue
        pairs = []
        for a, b in zip(s["pts"], s["pts"][1:]):
            key = frozenset((a["esr"], b["esr"]))
            km = (b["km0"] or 0) - (a["km0"] or 0)
            if a["esr"] == b["esr"]:
                continue
            if owner[key] != s["id"]:
                stat["pairs on another section"] += 1
                km_kept["on another section"] += km
                continue
            if (s["id"], key) in coarse:
                stat["pairs another section lists finer"] += 1
                km_kept["listed finer elsewhere"] += km
                continue
            if not (inside(a["esr"]) and inside(b["esr"])):
                stat["pairs outside the country as drawn"] += 1
                c = a["esr"] if not inside(a["esr"]) else b["esr"]
                km_kept["outside: " + where_is(pos.get(c))] += km
                continue
            pairs.append((a, b, km))
        if not pairs:
            continue
        is_stop = {}
        brk = [0]
        for i in range(1, len(pairs)):
            if pairs[i][0]["esr"] != pairs[i - 1][1]["esr"]:
                brk.append(i)
        runs, cur = [], []
        for i, (a, b, km) in enumerate(pairs):
            if i in brk and cur:
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
                    if kind[p["esr"]] == "served" or (has_halt and kind[p["esr"]] in ("halt", "flag")):
                        is_stop[p["esr"]] = True
                ends_stop = (kind[inner[0]["esr"]] == "served" and kind[inner[-1]["esr"]] == "served")
                if (CC[cc]["clone"] and not has_halt and ends_stop and km >= STRETCH_KM and v - u >= 1
                        and not consecutive(inner[0]["esr"], inner[-1]["esr"])):
                    clone |= {inner[0]["esr"], inner[-1]["esr"]}
                    stretch_idx |= set(range(idx0 + u, idx0 + v))
                    stat["unserved stretches with no halt left to the timetable"] += 1
                    km_kept["stretches left to the timetable"] += km
            idx0 += len(run)
        flat = [x for run in runs for x in run]
        last_pmr = False
        for k, (a, b, km) in enumerate(flat, 1):
            ca, cb = a["esr"], b["esr"]
            if k - 1 in stretch_idx:
                ca = f"{ca}@{s['id']}" if ca in clone else ca
                cb = f"{cb}@{s['id']}" if cb in clone else cb
            # an unplaced end (a halt with no node) counts as in Transnistria when the
            # section's previous placed point was
            pa_, pb_ = in_pmr(a["esr"]), in_pmr(b["esr"])
            if a["esr"] not in pos:
                pa_ = last_pmr
            if b["esr"] not in pos:
                pb_ = pa_
            last_pmr = pb_
            im = PMR_IM if pa_ and pb_ else road
            rows.append({"sol": f"{s['id']}:{k}", "line": s["id"], "a": f"esr:{ca}",
                         "b": f"esr:{cb}", "len": str(km), "im": im,
                         "label": f"{a['name']} - {b['name']}"})
            used |= {ca, cb}
            km_kept["kept"] += km
        for c in sorted(clone):
            cl = f"{c}@{s['id']}"
            rows.append({"sol": f"{s['id']}:{c}@", "line": s["id"], "a": f"esr:{c}",
                         "b": f"esr:{cl}", "len": "0", "im": road, "label": "clone"})
            rows.append({"sol": f"{s['id']}:{c}@x", "line": s["id"], "a": f"esr:{cl}",
                         "b": f"esr:{cl}x", "len": "0", "im": road, "label": "stub"})
            used |= {c, cl, f"{cl}x"}
        stop_codes |= {c for c, v in is_stop.items() if v}
        by_key = {}
        for p in s["pts"]:
            by_key.setdefault(rr.nkey(p["name"]), p)
            by_key.setdefault(rr.nkey(rr.EXTRA_CODE.sub("", p["name"])), p)
        pa, pb, vias = section_ends(s, by_key)
        if not (inside(pa["esr"]) and inside(pb["esr"])):
            # a header end abroad or in Transnistria: named by the ends it is built between
            pa, pb, vias = flat[0][0], flat[-1][1], []
        name = f"{shown[pa['esr']]} — {shown[pb['esr']]}"
        ea, eb = en_of.get(pa["esr"], ""), en_of.get(pb["esr"], "")
        if cc == "md" and (ea or eb):
            # a line into Transnistria: its Latin end is its own English ("Chișinău — Bender 1")
            ea = ea or ("" if CYRILLIC.search(shown[pa["esr"]]) else shown[pa["esr"]])
            eb = eb or ("" if CYRILLIC.search(shown[pb["esr"]]) else shown[pb["esr"]])
        name_en = f"{ea} — {eb}" if ea and eb else ""
        if vias:
            if all(vias):
                name += f" ({CC[cc]['via']} {', '.join(shown[v['esr']] for v in vias)})"
                ven = [en_of.get(v["esr"], "") for v in vias]
                name_en = f"{name_en} (via {', '.join(ven)})" if name_en and all(ven) else ""
            else:
                name += " (2)"
                name_en = f"{name_en} (2)" if name_en else ""
        names[s["id"]] = {"name": name, "name_en": name_en, "type": s["type"], "road": road,
                          "tariff_km": s["km"], "header": s["name"]}
    # --- border pieces: from the line's point to the border point
    bpts = border_points()
    n_border = 0
    for uop, codes_ in BORDER.get(cc, []):
        b = bpts.get(uop)
        code = next((x for x in codes_ if x in used and x in pos), None)
        if b is None or code is None:
            print(f"  border {uop}: no piece ({'no point' if b is None else 'none of ' + str(codes_) + ' built'})")
            stat["border pieces with no point"] += 1
            continue
        own = sorted((s for s in secs if s["pts"] and code in (s["pts"][0]["esr"], s["pts"][-1]["esr"])
                      and s["id"] in names),
                     key=lambda s: (rr.TYPE_RANK.get(s["type"], 9), s["id"]))
        if not own:
            own = sorted((s for s in secs if s["id"] in names
                          and any(p["esr"] == code for p in s["pts"])), key=lambda s: s["id"])
        if not own:
            continue
        crow = dist_m(*pos[code], b["lon"], b["lat"]) / 1000
        rows.append({"sol": f"{own[0]['id']}:{uop}", "line": own[0]["id"], "a": f"esr:{code}",
                     "b": f"eu:{uop}", "len": f"{max(0.5, crow * 1.2):.1f}", "im": road,
                     "label": f"{shown[code]} - {b['name']}"})
        border_rows.append({"op": f"eu:{uop}", "uopid": uop, "name": b["name"], "type": "90",
                            "lon": b["lon"], "lat": b["lat"]})
        n_border += 1
    stat["border pieces"] = n_border
    cnt = Counter(e["name"] for e in names.values())
    for sid, e in names.items():
        if cnt[e["name"]] > 1:
            e["name"] += f" ({sid})"
            if e["name_en"]:
                e["name_en"] += f" ({sid})"
    print(f"sections: {dict(stat)}")
    print(f"tariff km: { {k: round(v) for k, v in km_kept.items()} }")
    pts_out = []
    for c in sorted(used):
        base = c.split("@")[0]
        p = raw_of[base]
        q = pos.get(base)
        typ = ("70" if re.match(r"(?:ОП|О\.П\.)\s", p["raw"]) else "10") \
            if base in stop_codes and base == c else "80"
        r = {"op": f"esr:{c}", "uopid": f"{cc.upper()}{c}", "name": shown[base], "type": typ}
        if en_of.get(base) and not c.endswith("x"):
            r["name_en"] = en_of[base]
        if c.endswith("x"):
            r["name"] = ""
            q = None
        if q:
            r["lon"], r["lat"] = q
        pts_out.append(r)
    pts_out += border_rows
    O = out_dir(cc)
    O.mkdir(parents=True, exist_ok=True)
    stampd = {"endpoint": f"Тарифное руководство № 4, {b1name}, {b2name}",
              "fetched": date.today().isoformat()}
    (O / "sections.json").write_text(json.dumps({**stampd, "rows": rows}, ensure_ascii=False), "utf-8")
    (O / "points.json").write_text(json.dumps({**stampd, "rows": pts_out}, ensure_ascii=False), "utf-8")
    (O / "names.json").write_text(json.dumps(names, ensure_ascii=False, indent=0), "utf-8")
    print(f"wrote {len(rows)} section rows on {len(names)} lines, {len(pts_out)} points "
          f"({sum(1 for r in pts_out if r['type'] != '80')} stops, "
          f"{sum(1 for r in pts_out if 'lon' not in r)} unplaced) -> {O}")


# ================================================================ colours

# Tariff sections have no colours of their own (as Russia's and Ukraine's): every register line
# takes its railway's colour, picked. Belarus: by Belarusian Railway's six branches
# (отделения), told by the first three digits of the section's first ESR code; Moldova: one.
BY_BRANCH = [
    # (ESR prefixes, branch, colour, hue)
    (("130", "131", "139"), "Брэсцкае аддзяленне", "#2F7FD8", "blue"),
    (("132", "133", "134", "135", "136", "137", "138"), "Баранавіцкае аддзяленне", "#D7263D", "red"),
    (("140", "141", "142", "143", "144", "145", "146", "147", "148", "149"), "Мінскае аддзяленне",
     "#1E9E8F", "teal"),
    (("150", "151", "152", "153", "154", "155"), "Гомельскае аддзяленне", "#E08E0B", "orange"),
    (("156", "157", "158", "159"), "Магілёўскае аддзяленне", "#8A5CC2", "purple"),
    (("160", "161", "162", "163", "164", "165", "166", "167", "168", "169"),
     "Віцебскае аддзяленне", "#3FA34D", "green"),
]
MD_COLOUR = ("#1F6FB2", "blue")


def colours(cc):
    import csv
    names = json.loads((out_dir(cc) / "names.json").read_text("utf-8"))
    secs = {r["line"]: r for r in reversed(json.loads(
        (out_dir(cc) / "sections.json").read_text("utf-8"))["rows"])}
    rows = []
    for sid, e in sorted(names.items()):
        if cc == "md":
            col, hue = MD_COLOUR
            note = "CFM: one colour"
        else:
            code = secs[sid]["a"].split(":")[1].split("@")[0] if sid in secs else ""
            col, hue, br = "#777777", "grey", "?"
            for pre, b, c, h in BY_BRANCH:
                if code[:3] in pre:
                    col, hue, br = c, h, b
                    break
            note = f"{br}: one {hue} per branch (by ESR code)"
        rows.append({"line": e["name"], "operator": CC[cc]["operator"], "colour": col,
                     "source": "picked", "url": "", "note": note})
    path = ROOT / "colours" / f"{cc}.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["line", "operator", "colour", "source", "url", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"--colours: {len(rows)} register lines -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cc", required=True, choices=sorted(CC))
    ap.add_argument("--esr", metavar="PBF")
    ap.add_argument("--wikidata", action="store_true")
    ap.add_argument("--outline", action="store_true")
    ap.add_argument("--clip", action="store_true")
    ap.add_argument("--crossings", action="store_true")
    ap.add_argument("--crawl", action="store_true")
    ap.add_argument("--merstren", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--timetable", action="store_true")
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--colours", action="store_true")
    args = ap.parse_args()
    cc = args.cc
    if args.esr:
        p = Path(args.esr)
        esr_pass(cc, p if p.is_absolute() else ROOT / p)
    if args.wikidata:
        fetch_wikidata(cc)
    if args.outline:
        write_outline(cc)
    if args.clip:
        clip(cc)
    if args.crossings:
        crossings(cc)
    if args.crawl:
        crawl_poezdato(args.limit)
    if args.merstren:
        fetch_merstren()
    if args.timetable:
        timetable(cc)
    if args.convert:
        convert(cc)
    if args.colours:
        colours(cc)
