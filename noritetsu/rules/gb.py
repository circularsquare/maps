"""United Kingdom rules for build_model.py (read by build_model.country_rules)."""
import re

# Single trains, not lines: the sleepers, Eurostar (a named train in fr, be and nl too), the
# mainline steam and cruise trains, and the Channel Tunnel's car shuttle.
NAMED_OPERATORS = ("Caledonian Sleeper", "Eurostar", "West Coast Railways", "Belmond",
                   "Locomotive Services", "Statesman Rail", "Vintage Trains")
NAMED_SERVICE = {"night", "car_shuttle", "car"}
# LNER's once-a-day named trains: "The Highland Chieftain (London King's Cross => Inverness)"
# (ref GR HC), "The Northern Lights (... => Aberdeen)" (GR NL), the Flying Scotsman.
LNER_NAMED = re.compile(r"^The [A-Z][\w' ]+ \(|^GR (?:HC|NL|FS)$|Flying Scotsman")
NAMED_NAME = re.compile(r"\b(?:Night Riviera|Sleeper|Eurostar|Le Shuttle|Jacobite|"
                        r"Royal Scotsman|Northern Belle|Pullman)\b")


# A train route's stop node that build_model resolved to a metro station by proximity alone
# (the metro record's name lacks some word of the stop's) goes to the rail station whose name
# holds every word of the stop's, this close. Eurostar's
# London stop nodes are named "London St Pancras", which matches no station record exactly
# ("London St. Pancras International"), so the nearest record took them: the Underground's
# "King's Cross St Pancras", 80 m off, and Eurostar began at a Tube station (2026-10-05).
RAIL_NAME_M = 1200


def _words(s):
    return set(re.findall(r"[a-z0-9]+", (s or "").casefold().replace(".", "")))


def extra_route_stops(ways, rels, stops, coords, stations, resolved, log):
    import math
    rail = [(sid, s, _words(s.get("name"))) for sid, s in stations.items()
            if not s.get("_metro") and not s.get("_tram") and "_rank" in s]
    moved = {}
    for t, members in rels.values():
        if t.get("type") != "route" or t.get("route") != "train":
            continue
        for ty, n, role in members:
            if ty != "n" or not role.startswith("stop") or n in moved:
                continue
            st = resolved.get(n)
            rec = stops.get(n)
            if st is None or rec is None or not stations.get(st, {}).get("_metro"):
                continue
            w = _words(rec[0].get("name"))
            # only where the name did not match: "Barking" on Barking's Underground record is
            # build_model's own name match, and the Elizabeth line's stops stay as they are
            if not w or w <= _words(stations[st].get("name")):
                continue
            lon, lat = rec[1], rec[2]
            best = None
            for sid, s, sw in rail:
                if not w <= sw:
                    continue
                d = math.hypot((s["lon"] - lon) * math.cos(math.radians(lat)) * 111320,
                               (s["lat"] - lat) * 110570)
                if d <= RAIL_NAME_M and (best is None or d < best[0]):
                    best = (d, sid)
            if best is not None:
                moved[n] = best[1]
                resolved[n] = best[1]
                log(f"  gb: stop node {n} ({rec[0].get('name')}) of a train route moved from "
                    f"{stations[st]['name']} (metro) to {stations[best[1]]['name']}, "
                    f"{best[0]:.0f} m")
    return {}


def looks_like_service(tags, name, name_en):
    """Is this OSM route=train relation a named train rather than a line?

    OSM UK maps the train operators route by route (2026-10-03: 1,061 route=train relations):
    "AWC: London Euston → Holyhead", "London King's Cross => Edinburgh Waverley (fast)", "GWR:
    London Paddington → Bristol Temple Meads", "CrossCountry: Plymouth → Edinburgh", Lumo,
    Grand Central, the Enterprise. Those are interval products a rider uses as a line (Avanti,
    LNER, GWR and CrossCountry run each route hourly or so; Lumo and Grand Central five or six
    a day), so they stay lines, as Germany's ICE lines do; they are operating patterns over the
    register's named track. OSM tags 94 of them service=long_distance, which therefore decides
    nothing here (it does in the US).

    Named trains: the Caledonian Sleeper (10 relations, service=night) and GWR's Night
    Riviera, Eurostar (12; as in fr, be, nl), Eurotunnel's Le Shuttle (service=car_shuttle),
    LNER's once-a-day Highland Chieftain and Northern Lights, and the mainline tourist trains
    (West Coast Railways' Jacobite, Belmond's Royal Scotsman and Northern Belle). Heritage
    railways' own relations (Ffestiniog, Welsh Highland, service=tourism) stay lines, as the
    USA's do; whether they count is in gb_sources.md as a question for Anita.
    """
    name = name or tags.get("ref") or ""
    if set((tags.get("service") or "").split(";")) & NAMED_SERVICE:
        return True
    op = " ".join(tags.get(k) or "" for k in ("operator", "network"))
    if any(o in op for o in NAMED_OPERATORS):
        return True
    if NAMED_NAME.search(name) or NAMED_NAME.search(name_en or ""):
        return True
    if "London North Eastern" in op and LNER_NAMED.search(name):
        return True
    return bool(LNER_NAMED.search(tags.get("ref") or ""))
