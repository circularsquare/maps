"""Thailand's rules for build_model.py (build_model.country_rules lists what it reads).

NAMED TRAINS. Every train of the State Railway of Thailand has a number, and OSM Thailand
maps the few SRT trains it has one relation per numbered train (2026-10-03, 13 relations):
"405 (ท้องถิ่น) ศิลาอาสน์ - สวรรคโลก", "147 (เร็ว) อุดรธานี - เวียงจันทน์ (คำสะหวาด)",
"ขบวน 4302: มหาชัย => วงเวียนใหญ่", "Train 1123: Thonburi => Nakhon Pathom", and the new
Krung Thep Aphiwat - Ayutthaya commuter trains under network "Bangkok Connex" as "9001
กรุงเทพอภิวัฒน์ - อยุธยา" (ref 9001), each pair under its own route_master. Each is one
train, so each is a named train (option B: no percentage of its own, its track counts through
the register line it runs on). That includes the Mae Klong Railway's 4302, though the line
has about 17 trains a day each way, and Bangkok Connex: the LINE a rider uses there is the
register line (SRT's Mae Klong Railway, the Northern Line), which every SRT train runs over,
and three copies of one 63.5 km line, one per train pair, would be no line anyway.
SRT's lines are register lines (th_register.py), so no SRT track is left without a line.

Lines: the Airport Rail Link, the SRT Red Lines (operator SRTET, every 10-20 minutes), the
BTS, the MRT and the monorails. None carries a train number. Malaysia's KTM routes that
reach Padang Besar are KTM's (rules/my.py decides them); their ways stop at the border.

STOPS. OSM Thailand often lists a route's stations as node members with no role: both SRT
Light Red Line relations (13178788, 14071495) have only those, so build_model found no stop
and the line vanished; the BTS Silom Line's lists only its ends and Siam with a role
(Siam - Krung Thon Buri came out as one 6.9 km section), the Yellow Line leaves out Hua
Mak, Suan Luang Rama IX and Si Udom. A role-less node member that is a station (or a stop of
one by name within 1 km) is a stop: `extra_route_stops` adds each, as the nearest node of
the route's own track.
"""
import math
import re

import numpy as np

# A train number: one to four digits standing alone, in the ref or the name ("405 / 406
# (Local)", "ขบวน 4302", "Train 1123", "9001/9002"), never inside a word ("AERA1 City", the
# Airport Rail Link's ref).
TRAIN_NO = re.compile(r"(?<![\w.])\d{1,4}(?![\w.])")
SRT_NAMES = ("การรถไฟแห่งประเทศไทย", "SRT", "State Railway of Thailand", "Bangkok Connex")
NOT_SRT = ("SRTET", "สีแดง", "Red Line", "ARL", "แอร์พอร์ต", "Airport Rail Link", "เอราวัน")


def looks_like_service(tags, name, name_en):
    who = " ".join(tags.get(k) or "" for k in ("operator", "network"))
    text = " ".join(t for t in (name, name_en, tags.get("ref"), who) if t)
    if any(x in text for x in NOT_SRT):
        return False
    if not any(x in who for x in SRT_NAMES):
        return False
    return bool(TRAIN_NO.search(tags.get("ref") or "") or TRAIN_NO.search(name or "")
                or TRAIN_NO.search(name_en or ""))


STOP_ON_TRACK_M = 400      # a role-less station member's stop node: the route's nearest node


def extra_route_stops(ways, rels, stops, coords, stations, resolved, log):
    import build_model as bm
    out = {}
    for rid, (tags, members) in rels.items():
        if tags.get("type") != "route" or tags.get("route") not in bm.ROUTE_KINDS:
            continue
        loose = {}
        for ty, r, role in members:
            if ty != "n" or role:
                continue
            if r in resolved:
                loose[r] = resolved[r]
            elif r in stations:
                loose[r] = r
            elif r in stops and stops[r][0].get("name"):
                # a platform or stop node of a station by name, within 1 km
                t, lon0, lat0 = stops[r]
                best = None
                for k, s in stations.items():
                    if s["name"] == t["name"]:
                        d = math.hypot((s["lon"] - lon0) * math.cos(math.radians(lat0)) * 111320,
                                       (s["lat"] - lat0) * 110570)
                        if d <= 1000 and (best is None or d < best[0]):
                            best = (d, k)
                if best:
                    loose[r] = best[1]
        if not loose:
            continue
        rw =[r for ty, r, _ in members if ty == "w" and r in ways]
        if not rw:
            continue
        nodes = np.unique(np.concatenate([np.asarray(ways[w][1], dtype=np.int64) for w in rw]))
        pos, ok = coords.many(nodes)
        nodes, pos = nodes[ok], pos[ok]
        x, y = coords.x[pos] / 1e7, coords.y[pos] / 1e7
        got = {}
        for n, st in loose.items():
            s = stations.get(st)
            if s is None:
                continue
            lon, lat = s["lon"], s["lat"]
            d = np.hypot((x - lon) * math.cos(math.radians(lat)) * 111320, (y - lat) * 110570)
            j = int(np.argmin(d))
            if d[j] <= STOP_ON_TRACK_M:
                got.setdefault(st, set()).add(int(nodes[j]))
        if got:
            out[rid] = got
    if out:
        log(f"TH rules: {len(out)} routes given their role-less station members as stops "
            f"({sum(len(v) for v in out.values())} stops)")
    return out
