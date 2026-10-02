"""Lines, stations and sections from the Swiss federal railway network register.

    python build_model.py --region ch --register schienennetz:data/raw/schienennetz_2056_de.gdb.zip

Two files, both open government data, both downloaded as they are:

  data/raw/schienennetz_2056_de.gdb.zip   BAV Geobasisdatensatz "Schienennetz" (ch.bav.schienennetz)
      https://data.geo.admin.ch/ch.bav.schienennetz/schienennetz/schienennetz_2056_de.gdb.zip
  data/raw/ch_servicepoints.csv           the national service-point list (DiDok), which is the
      only thing that says which network nodes are PASSENGER stops
      https://data.sbb.ch/api/explore/v2.1/catalog/datasets/dienststellen-gemass-opentransportdataswiss/exports/csv?delimiter=%3B

WHAT THE REGISTER IS.  460 kilometre lines (KmLinie), each an official chainage axis with a
number and a name: "600 Immensee - Bellinzona - Chiasso". 3,424 network segments (Netzsegment),
each belonging to one km-line and running between two network nodes (Netzknoten), with the
infrastructure manager, gauge and electrification on it. 5,829 km in total, standard gauge,
narrow gauge, rack railways and some city trams. The km-line is the unit counted here, as the
N02 line is for Japan.

HOW IT DIFFERS FROM JAPAN, and why this is not a copy of n02.py:

- THE TOPOLOGY IS GIVEN. Every segment names its two end nodes, so the graph is built over
  nodes, not over rounded vertex coordinates, and there is nothing to snap.
- A KM-LINE DOES NOT START AT A STATION. It starts at a junction as often as not: 1,513 km of
  the register lies beyond a line's last passenger stop, and the Gotthard and Lotschberg base
  tunnels and the Mattstetten-Rothrist high-speed line have no stop at all. Cutting sections
  only at stations, as n02.py does, deleted them. So every end of a line is a section end
  too, and is written as a station flagged `junction` -- not a place anyone boards, but a
  place the register's line ends.
- A SECTION ENDING AT A JUNCTION IS EITHER A BASE TUNNEL OR FREIGHT, and the register cannot
  say which. build_model.drop_unridden_sections keeps it only if OpenStreetMap passenger
  routes run over it, so a freight curve or a yard throat falls out.
- A PLATFORM GROUP IS ITS STATION. "Chur [Gleis 10-14]" is a node of its own and not a stop
  point; it is matched by name onto the stop it belongs to.
- TWIN TUBES ARE TWO KM-LINES. "GBT Ost" and "GBT West" are the two bores of the Gotthard
  Base Tunnel. They are merged into one line and, because the graph is a graph, each section
  follows one bore, as a double-track line follows one track in n02.py. A line end is a node
  with one NEIGHBOUR, not one segment: the merged Lötschberg base tunnel (330/331) has two
  segments, one per bore, into St. German, and counting segments built it no section at all
  (until 2026-10-02).
- BORDER NODES ARE RINF'S BORDER POINTS. A line end that is not a stop and lies within
  BORDER_SNAP_M of a border point (borders.py; the register's own "La Plaine-Frontière",
  "Le Locle-Frontière", "Delle-Frontière" are 0-17 m from them) takes the point's id, which
  France's register, every RINF register and OSM's border tails end at too.

NOT IN THE REGISTER (checked 2026-10-02): the CEVA tunnels in Geneva. Line 152 has Lancy-Bachet
and Champel - Eaux-Vives - Chêne-Bourg, but not Lancy-Bachet - Champel nor Chêne-Bourg - the
border, opened December 2019. The published file is the 2021-07-06 edition (Stand), still the
only one on data.geo.admin.ch (STAC item updated 2025-01-18, same checksum), so the Léman
Express owns that track as an OSM line.
"""
import csv
import hashlib
import heapq
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

# Means of transport in the service-point list that make a stop a RAIL stop.
RAIL_MEANS = {"TRAIN", "TRAM", "RACK_RAILWAY", "METRO", "CABLE_RAILWAY"}
KIND_OF_MEANS = {"TRAM": "tram", "METRO": "subway", "CABLE_RAILWAY": "funicular"}

# Twin-bore km-lines: the same name but for a trailing direction.
TWIN_SUFFIX = re.compile(r"\s*\(?\b(Ost|West|Nord|Süd)\)?\s*$")

SERVICEPOINTS = "ch_servicepoints.csv"

# How far a platform-group node ("Chur [Gleis 10-14]") may be from the stop it is part of.
PLATFORM_M = 800
# A line-end node this close to a RINF border point is that point.
BORDER_SNAP_M = 30

# Trailing words on a node name that make it a part of the railway, not a railway's platforms.
NOT_A_RAILWAY = {w.casefold() for w in (
    "RB", "GB", "Rbf", "Gbf", "PB", "Triage", "Abzw", "Abzweigung", "bif", "Vzw", "Verzw",
    "dira", "Nord", "Süd", "Ost", "West", "Est", "Ouest", "Sud", "Mitte", "Depot",
    "Gleisende", "Werkstätte", "Ateliers", "Officina", "Areal", "Hafen", "Industrie")}


def line_id(name, operator):
    h = hashlib.blake2b(f"{name}|{operator}".encode("utf-8"), digest_size=5)
    return "c" + h.hexdigest()


def length_km(coords):
    total = 0.0
    for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
        dx = (x2 - x1) * math.cos(math.radians((y1 + y2) / 2)) * 111.320
        dy = (y2 - y1) * 110.570
        total += math.hypot(dx, dy)
    return total


def read(gdb_zip, log):
    import fiona
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import linemerge

    to_wgs = Transformer.from_crs(2056, 4326, always_xy=True)
    uri = "zip://" + str(Path(gdb_zip).resolve()).replace("\\", "/")

    kml = {f["properties"]["xtf_id"]: dict(f["properties"])
           for f in fiona.open(uri, layer="KmLinie")}
    nodes = {}
    for f in fiona.open(uri, layer="Netzknoten"):
        p = dict(f["properties"])
        x, y = f["geometry"]["coordinates"][:2]
        p["lon"], p["lat"] = to_wgs.transform(x, y)
        nodes[p["xtf_id"]] = p
    segs = []
    for f in fiona.open(uri, layer="Netzsegment"):
        p = dict(f["properties"])
        g = shape(f["geometry"])
        if g.geom_type == "MultiLineString":
            g = linemerge(g)
        parts = list(g.geoms) if g.geom_type.startswith("Multi") else [g]
        coords = []
        for part in parts:                     # rarely more than one; concatenated in order
            coords.extend(part.coords)
        lon, lat = to_wgs.transform([c[0] for c in coords], [c[1] for c in coords])
        pts = list(zip(lon, lat))
        # Orient from the start node to the end node, which the section walk relies on.
        a = nodes.get(p["rAnfangsknoten"])
        if a and len(pts) > 1:
            d0 = (pts[0][0] - a["lon"]) ** 2 + (pts[0][1] - a["lat"]) ** 2
            d1 = (pts[-1][0] - a["lon"]) ** 2 + (pts[-1][1] - a["lat"]) ** 2
            if d1 < d0:
                pts.reverse()
        p["pts"] = pts
        p["km"] = length_km(pts)
        p["chain"] = abs((p["KmEnde"] or 0) - (p["KmAnfang"] or 0))
        segs.append(p)
    log(f"Schienennetz: {len(kml)} km-lines, {len(segs)} segments "
        f"({sum(s['km'] for s in segs):,.0f} km), {len(nodes)} nodes")
    return kml, nodes, segs


def read_servicepoints(path, log):
    sp, orgs = {}, {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            try:
                sp[int(r["number"])] = r
            except ValueError:
                continue
            ab = r.get("businessorganisationabbreviationde")
            if ab and r.get("businessorganisationdescriptionde"):
                orgs.setdefault(ab, r["businessorganisationdescriptionde"])
    log(f"service points: {len(sp)}")
    return sp, orgs


def build(gdb_zip, log):
    kml, nodes, segs = read(gdb_zip, log)
    sp_path = Path(gdb_zip).with_name(SERVICEPOINTS)
    if not sp_path.exists():
        raise SystemExit(f"{sp_path} is needed beside the register; see this file's docstring")
    sp, orgs = read_servicepoints(sp_path, log)

    # --- which node is which station.
    #
    # A PLATFORM GROUP IS THE STATION IT IS PART OF. "Interlaken Ost [Gleis 1-2]", "Tavannes
    # [voie étroite]" and "Gossau SG AB" are nodes of their own with their own numbers, and are
    # not stop points in the service-point list, so the line that starts there lost its first
    # station and the track to the next one. Only 17 nodes say so with rUebergeordnet; the
    # rest are found by name -- the bracket or the " AB" dropped -- among the rail stops close
    # by in the service-point list.
    def rail_means(rec):
        if not rec or rec["stoppoint"] != "true":
            return set()
        return set((rec["meansoftransport"] or "").split("|")) & RAIL_MEANS

    def geopos(rec):
        try:
            lat, lon = (float(v) for v in rec["geopos"].split(","))
            return lon, lat
        except (ValueError, AttributeError):
            return None

    stops_by_name = defaultdict(list)
    for rec in sp.values():
        if rail_means(rec) and geopos(rec):
            stops_by_name[rec["designationofficial"]].append(rec)

    def resolve(nid):
        """The service-point record of the stop this node belongs to, or None."""
        seen = set()
        n = nid
        while True:
            rec = sp.get(nodes[n]["Betriebspunkt_Nummer"])
            if rail_means(rec):
                return rec
            par = nodes[n].get("rUebergeordnet")
            if not par or par not in nodes or par in seen:
                break
            seen.add(par)
            n = par
        p = nodes[nid]
        name = p["Betriebspunkt_Name"] or ""
        base = re.sub(r"\s*\[.*?\]\s*$", "", name)
        if base == name:
            # "Visp MGB-bvz", "Gossau SG AB": the station's name and the code of the railway
            # whose platforms these are. Never a yard or a junction: "Basel SBB RB" is a
            # marshalling yard, and resolving it onto Basel SBB would make freight track look
            # like a section between two stops, which is never questioned.
            m = re.match(r"^(.+?)\s+([A-Za-z][A-Za-z-]{0,7})$", name)
            if not m or m.group(2).casefold() in NOT_A_RAILWAY:
                return None
            base = m.group(1)
        best, best_d = None, PLATFORM_M
        for rec in stops_by_name.get(base, ()):
            lon, lat = geopos(rec)
            d = math.hypot((lon - p["lon"]) * math.cos(math.radians(lat)) * 111320,
                           (lat - p["lat"]) * 110570)
            if d <= best_d:
                best, best_d = rec, d
        return best

    stop_of = {nid: resolve(nid) for nid in nodes}
    n_platform = sum(1 for nid, rec in stop_of.items()
                     if rec and int(rec["number"]) != nodes[nid]["Betriebspunkt_Nummer"])
    log(f"Schienennetz: {sum(1 for r in stop_of.values() if r)} nodes are passenger stops, "
        f"{n_platform} of them platform groups resolved onto their station")
    for nid, rec in stop_of.items():
        if (rec and int(rec["number"]) != nodes[nid]["Betriebspunkt_Nummer"]
                and "[" not in nodes[nid]["Betriebspunkt_Name"]):
            log(f"    {nodes[nid]['Betriebspunkt_Name']} -> {rec['designationofficial']}")

    def means(nid):
        return rail_means(stop_of.get(nid))

    def station_of(nid):
        rec = stop_of.get(nid)
        return f"c{rec['number']}" if rec else f"c{nodes[nid]['Betriebspunkt_Nummer']}"

    def name_of(nid):
        rec = stop_of.get(nid)
        return (rec and rec["designationofficial"]) or nodes[nid]["Betriebspunkt_Name"] or ""

    def pos_of(nid):
        rec = stop_of.get(nid)
        return (rec and geopos(rec)) or (nodes[nid]["lon"], nodes[nid]["lat"])

    # --- national borders. A node that is not a stop and lies on a RINF border point (borders.py)
    # is that point: the register's own "Le Locle-Frontière" or "La Plaine-Frontière" is 0-17 m
    # from it. Ending there under the point's id joins the line to the neighbour's half, which
    # ends at the same id (France's register, every RINF register, OSM's border tails).
    border_of, border_rec = {}, {}
    try:
        import borders
        bpts = borders.load()
    except (ImportError, OSError, ValueError):
        bpts = []
    for nid, p in nodes.items():
        if means(nid):
            continue
        # Nearest; between points a few metres apart, an "eEU" one, which is what the
        # neighbour's RINF register ends at (St. Margrethen is both eEU00118 and eCH15472).
        near = min(((round(math.hypot((b["lon"] - p["lon"]) * math.cos(math.radians(p["lat"]))
                                      * 111320, (b["lat"] - p["lat"]) * 110570) / 5),
                     not b["id"].startswith("eEU"), b["id"], b) for b in bpts), default=None)
        if near is not None and near[0] * 5 <= BORDER_SNAP_M:
            border_of[nid] = near[2]
            border_rec[near[2]] = near[3]

    # --- km-lines, with twin bores merged
    by_kml = defaultdict(list)
    for s in segs:
        by_kml[s["rKmLinie"]].append(s)
    keys = {}
    for kid in by_kml:
        k = kml.get(kid, {})
        name = (k.get("Name") or "").strip()
        base = TWIN_SUFFIX.sub("", name)
        keys[kid] = (base, k.get("Datenherr_TUAbkuerzung") or "")
    count = Counter(keys.values())
    groups = defaultdict(list)
    for kid, (base, op) in keys.items():
        name = (kml.get(kid, {}).get("Name") or "").strip()
        # Only a real pair is merged: "Lyss Nord" ends in a direction and has no twin.
        groups[(base, op) if count[(base, op)] > 1 and base != name else (name, op, kid)].append(kid)
    twins = sum(1 for g in groups.values() if len(g) > 1)

    stations, lines, geoms = {}, [], {}
    unused_km, unused_by = 0.0, []
    for gkey, kids in groups.items():
        ss = [s for kid in kids for s in by_kml[kid]]
        adj = defaultdict(list)
        for s in ss:
            a, b = s["rAnfangsknoten"], s["rEndknoten"]
            adj[a].append((b, s["km"], s, False))
            adj[b].append((a, s["km"], s, True))

        # Every end of the line is a section end, whether a junction, the national border
        # ("Les Verrières-Frontière") or a siding. Telling those apart from the register alone
        # does not work: a line can end at a node the next line does not share ("Ins
        # (Verzw)"), which reads exactly like a dead end. build_model decides instead, keeping
        # a section that ends at one only where OSM passenger routes run over it.
        stop_nodes = {n for n in adj if means(n)}
        # An end is a node with ONE NEIGHBOUR, not one segment: in a merged twin-bore line the
        # far portal has two, one per bore, to the same node. Counting segments left the
        # Lötschberg base tunnel (330/331, St. German) with a single end, and so with no
        # section at all.
        ends = {n for n, nb in adj.items()
                if len({v for v, _w, _s, _r in nb}) == 1 and n not in stop_nodes}
        at = {n: border_of.get(n) or station_of(n) for n in stop_nodes | ends}
        if len({*at.values()}) < 2:
            continue

        # Absorbing Dijkstra from each section end, as in n02.adjacent, over NODES.
        sections = {}
        for src, sid in at.items():
            dist, prev, seen = {src: 0.0}, {}, set()
            heap = [(0.0, src)]
            while heap:
                d, u = heapq.heappop(heap)
                if u in seen:
                    continue
                seen.add(u)
                other = at.get(u)
                if u != src and other is not None:
                    if other != sid:
                        key = (sid, other) if sid <= other else (other, sid)
                        if key not in sections or d < sections[key]["km"]:
                            path, cur = [], u
                            while cur != src:
                                p, seg, rev = prev[cur]
                                path.append((seg, rev))
                                cur = p
                            path.reverse()
                            pts = []
                            for seg, rev in path:
                                sp_ = seg["pts"][::-1] if rev else seg["pts"]
                                pts.extend(sp_ if not pts else sp_[1:])
                            if sid > other:
                                pts.reverse()
                            sections[key] = {"km": d, "pts": pts,
                                             "chain": sum(sg["chain"] for sg, _r in path),
                                             "segs": [id(sg) for sg, _r in path]}
                    continue                                  # absorbed
                for v, w, seg, rev in adj[u]:
                    nd = d + w
                    if nd < dist.get(v, math.inf):
                        dist[v] = nd
                        prev[v] = (u, seg, rev)
                        heapq.heappush(heap, (nd, v))
        if not sections:
            continue
        used = {x for v in sections.values() for x in v["segs"]}
        left = [s for s in ss if id(s) not in used]
        unused_km += sum(s["km"] for s in left)
        if left:
            unused_by.append((sum(s["km"] for s in left), gkey[0], len(left)))

        k0 = kml.get(kids[0], {})
        name = gkey[0]
        infra = Counter()
        for s in ss:
            infra[s["Infrastrukturbetr_TUAbkuerzung"] or ""] += s["km"]
        op_ab = infra.most_common(1)[0][0] or k0.get("Datenherr_TUAbkuerzung") or ""
        lid = line_id(name, op_ab)

        kinds = Counter()
        for n in stop_nodes:
            for m in means(n):
                kinds[KIND_OF_MEANS.get(m, "rail")] += 1
        kind = kinds.most_common(1)[0][0] if kinds else "rail"

        for n, sid in at.items():
            if sid not in stations:
                if sid in border_rec:
                    b = border_rec[sid]
                    lon, lat, st_name = b["lon"], b["lat"], b["name"]
                else:
                    (lon, lat), st_name = pos_of(n), name_of(n)
                stations[sid] = {"id": sid, "name": st_name, "name_en": "",
                                 "lon": lon, "lat": lat, "lines": set()}
            if n in stop_nodes:
                stations[sid]["stop"] = True
            stations[sid]["lines"].add(lid)

        order = walk_order(sections.keys())
        lines.append({
            "id": lid, "src": "schienennetz", "service": False,
            "name": name, "name_en": "", "ref": "/".join(sorted(
                kml.get(k, {}).get("Nummer") or "" for k in kids)),
            "colour": "",
            "operator": orgs.get(op_ab, op_ab), "operator_en": "", "network": "",
            "kind": kind,
            "km": round(sum(v["km"] for v in sections.values()), 3),
            "km_official": round(sum(v["chain"] for v in sections.values()), 3),
            # Per section, so build_model can recompute km_official after dropping some.
            "chain": {f"{a}|{b}": round(v["chain"], 3) for (a, b), v in sections.items()},
            "variants": len(kids),
            "straight_sections": 0,
            "display": order,
            "sections": [[a, b, round(v["km"], 3)] for (a, b), v in sections.items()],
        })
        geoms[lid] = {f"{a}|{b}": [[round(x, 5), round(y, 5)] for x, y in v["pts"]]
                      for (a, b), v in sections.items()}

    for s in stations.values():
        if not s.pop("stop", False):
            s["junction"] = True
    n_junc = sum(1 for s in stations.values() if s.get("junction"))
    total = sum(l["km"] for l in lines)
    log(f"Schienennetz: {len(lines)} lines ({twins} twin-bore pairs merged), {total:,.0f} km; "
        f"{len(stations)} stations of which {n_junc} are line ends that are not stops")
    log(f"Schienennetz: {unused_km:,.0f} km of kept lines' track is in no section "
        f"(dead ends past the last stop, and the longer of two routes between two stops)")
    for km, name, n in sorted(unused_by, reverse=True)[:12]:
        log(f"    {km:6.1f} km in {n} segments  {name}")
    return lines, stations, geoms


def walk_order(keys):
    """A reading order for the strip diagram: from a terminus if there is one, else round."""
    from n02 import walk_order as w
    return w(keys)
