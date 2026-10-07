"""France's rules for build_model.py (build_model.country_rules lists what it reads)."""
import re
import unicodedata

# Case-sensitive, so "ICE" is the German train and not a word that starts "Ice".
FR_TRAIN_BRAND = re.compile(r"^(TGV|OUIGO|Ouigo|OUIGo|Eurostar|Lyria|ICE|Intercités|"
                            r"INTERCITÉS|ICN?\s|Renfe|RENFE|Frecciarossa|Nightjet|Thalys|"
                            r"Train de nuit)")
# Eurotunnel's Le Shuttle (Coquelles - Folkestone, cars and coaches with their passengers):
# a named train, as in gb (fr_sources.md "The Channel Tunnel").
NAMED_SERVICE = {"car_shuttle", "car"}


def looks_like_service(tags, name, name_en):
    # OSM France maps each long-distance train (TGV 723, Intercités 3731, Ouigo TC 4071,
    # Eurostar, Lyria) as its own relation. TER, Transilien and RER relations are lines.
    # An unnamed relation is called by its ref, and judged by it too: route 5945159
    # ("Ouigo", Marne-la-Vallée - Lyon) has only ref=Ouigo.
    if set((tags.get("service") or "").split(";")) & NAMED_SERVICE:
        return True
    return bool(FR_TRAIN_BRAND.match(name or tags.get("ref") or ""))


# A Eurostar relation's `via` station is one of its stops when the relation does not list it:
# OSM's London - Brussels relations (112662, 2905886) list only St Pancras and Brussels-Midi,
# with via=Lille Europe, where Eurostar's London - Brussels trains call. With no stop in France
# they built nothing here, and the London - Brussels named train had no French part between the
# tunnel and the Belgian border (fr_sources.md "The Channel Tunnel").
VIA_NETWORKS = {"Eurostar"}
VIA_M = 200          # the route's own track nodes this close to the station are its stop nodes


def _fold(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]", "", s)


def extra_route_stops(ways, rels, stops, coords, stations, resolved, log):
    import math
    import numpy as np
    out = {}
    for rid, (t, members) in rels.items():
        if t.get("type") != "route" or t.get("network") not in VIA_NETWORKS or not t.get("via"):
            continue
        rw = [r for ty, r, _role in members if ty == "w" and r in ways]
        if not rw:
            continue
        listed = {resolved.get(r) for ty, r, _role in members if ty == "n"}
        nodes = np.unique(np.concatenate([np.asarray(ways[w][1], dtype=np.int64) for w in rw]))
        pos, ok = coords.many(nodes)
        nodes, pos = nodes[ok], pos[ok]
        nx, ny = coords.x[pos] / 1e7, coords.y[pos] / 1e7
        for via in t["via"].split(";"):
            key = _fold(via)
            best = None                  # the one rail station of that name on the route
            for sid, s in stations.items():
                if (not key or _fold(s.get("name")) != key or sid in listed
                        or s.get("_metro") or s.get("_tram")):
                    continue
                d = np.hypot((nx - s["lon"]) * math.cos(math.radians(s["lat"])) * 111320,
                             (ny - s["lat"]) * 110570)
                if d.size and d.min() <= VIA_M and (best is None or d.min() < best[0]):
                    best = (float(d.min()), sid, set(nodes[d <= VIA_M].tolist()))
            if best is not None:
                out.setdefault(rid, {})[best[1]] = best[2]
                # Listed too, in this build's copy of the relation: build_model.border_tails
                # runs a route's ends on to the border only from stops the relation lists
                # (a stop added only through this hook has no place in its stop order, so
                # Lille-Europe got no tail to the tunnel or to Belgium). The node nearest the
                # station goes in as a stop after the origin, and resolves to the station.
                s = stations[best[1]]
                d = np.hypot((nx - s["lon"]) * math.cos(math.radians(s["lat"])) * 111320,
                             (ny - s["lat"]) * 110570)
                node = int(nodes[int(np.argmin(d))])
                first = next((i for i, (ty, _r, role) in enumerate(members)
                              if ty == "n" and role.startswith("stop")), -1)
                members.insert(first + 1, ("n", node, "stop"))
                resolved[node] = best[1]
                log(f"  fr: {s['name']} a stop of {t.get('name')} (r{rid}, its via), "
                    f"{len(best[2])} stop nodes, {best[0]:.0f} m")
    return out
