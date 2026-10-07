"""United Kingdom: register lines are OpenStreetMap's named track, kr_register's recipe.

    python gb_register.py --fetch      # the small sources into data/raw/gb (Wikidata, NaPTAN, NR GIS)
    python gb_register.py --names      # the line names the extract gives, folded, with km (needs data/proc/gb)
    python gb_register.py --clip       # after every extract: France's half of the Channel Tunnel out
    python build_model.py --region gb --register gb_register:data/raw/gb

THE LINE UNIT (Anita's call; gb_sources.md has the numbers and the alternatives). The UK has no
legal register of passenger lines. What it has:

  - Network Rail's ELRs (Engineer's Line References): 1,590 in NR's own reference-line file
    (OGL), 3,346 in Wikidata with a description and a length in miles (CC0). Engineering units:
    the East Coast Main Line is ECM1-ECM9, Paddington - Penzance is MLN1 + MLN2 + ..., a
    junction curve is an ELR of its own, and an ELR can change at a county boundary.
  - The names riders use, which are Wikipedia's ("Cotswold Line", "Hope Valley Line",
    "Settle-Carlisle Railway"), and which OSM puts on the TRACK: 97.2% of the passenger track-km
    in Great Britain carries a `name`, and 98.1% an ELR in `ref` (Overpass, 2026-10-03).

So the line is the OSM track name, as Korea's 경부선 is: each line is the graph of the ways that
carry its name, and the train operators' routes (Avanti, LNER, GWR, CrossCountry...) are
operating patterns over it, as JR's are over N02. Every way has one name, so the lines never
overlap: no piece of track belongs to two lines. But OSM UK names its track in two styles at
once, Wikipedia's ("West Coast Main Line") and Network Rail's ELR descriptions ("London Euston
to Crewe Line", "Carlisle Grand Junction Line"), leaves stretches unnamed, and puts bridge and
tunnel names on ways; each change of name in mid-line cut a line in two and lost the sections
across it (a first build had 10,800 km where this one has 15,300). The repairs, in order:

  - `tidy`/`fold_key`: one line spelt two ways is one ("Rugby-Birmingham-Stafford Line" and
    "Rugby Birmingham and Stafford Line"; a Welsh name before " / "; a track's "(Up
    Southport)"); a structure's or siding's name on the track ("Ouse Bridge", "Platform 6") is
    no name.
  - the ELR's main name (`name_table`): where one name has ELR_FILL_SHARE of an ELR's named
    track, the ELR's other ways take it, unless their own name has ELR_RENAME_MAX_KM of track
    on the ELR (MLN1 is Paddington - Penzance, and its Bristol to Exeter Line stays itself).
  - `propagate`: an unnamed way takes the name its neighbours at both ends share, then its
    ELR's main name, then a dead end's single same-ELR neighbour's.
  - `absorb`: a stray short piece of a line, away from the line's main piece and with other
    lines at all its ends, becomes the line it touches most.
  - `junction_ends`: where a line's track ends on another line's (Settle Junction, Ely North
    Junction) is a `junction` station "gj<node>", so the stretch from the last station to it
    is a section; build_model keeps it only where passenger routes run over it.
  - `drop_shortcuts`: a section running past its line's own stations on a parallel track (an
    avoiding line, a loop) is dropped when the line's other sections join its ends within
    SHORTCUT_SLACK.
  - AREA_NAME: a name mappers put on another line's track in one area ("Great Western Main
    Line" in Devon) is that line's there.

LINES IN PIECES (`split_pieces`, a hook build_model calls once it has dropped the junction-ended
sections no route runs over; Anita, 2026-10-04: a trip is entered station to station, so a line
that cannot be ridden across a gap is broken as a line). Since every way has one name, a line
that runs over another's rails for a stretch stops and starts again: the Birmingham to
Peterborough Line is Nuneaton - Wigston and Syston - Oakham, the Midland Main Line's Leicester
between. `bridge_gaps` joins the pieces over the track between them where trains run (the
`borrowed` sections, which credit the line whose track it is); what is still apart becomes one
line per piece, as in us_register. Both are in pieces.py since 2026-10-04 (shared with kr, cn
and tr); this module gives it the UK's settings and track graph.

WHAT IS LEFT OUT of the register (it stays on the map as OSM lines where OSM has routes):
  - London Underground, the DLR, the Tyne and Wear Metro, Glasgow Subway, Metrolink and the
    trams: railway=subway/light_rail/tram, never register track. The Underground's own track
    mapped as railway=rail (Metropolitan Line north of Harrow, the District's Wimbledon branch)
    is left to the Underground's OSM lines too (METRO_ON_RAIL).
  - Heritage railways: usage=tourism, railway=preserved, or a name Wikidata calls a heritage
    railway (from --fetch) or HERITAGE_EXTRA lists. 159 names, 538 km of track; see
    gb_sources.md.
  - Freight-only track: a named line no passenger route stops on has no stations and drops
    out, as in kr_register.

WHICH STATIONS ARE ON A LINE comes from OSM's passenger routes, since no open per-line station
list exists (Wikidata's P81 covers 972 of 2,635 stations, the West Coast Main Line 33): a
station a route stops at goes on every named line that route's own track runs on within
ALONG_M of the station. A train calling at Crewe on its way from Chester runs on the North
Wales Coast Line's track into the station and the West Coast Main Line's through it, so Crewe
is on both; a line that only passes Crewe on another track gets nothing. These lists go to
kr_register as its published lists, so a listed station is then found on the line's track as
there (MATCH_M), and kr_register's other rules apply unchanged: a stop node on the track, and a
station no route stops at joins the nearest named track within PROX_M.

BORDERS. A border point of borders.load() (border_points.json and borders.EXTRA) naming "gb"
and lying on a line's track becomes a station of that line under its own id, so the Channel
Tunnel ends at eEU00228 where France's track would; build_model names it and keeps the section
there only if passenger routes run over it (`junction`). The Channel Tunnel's UK half is folded
into High Speed 1 (NAME_ALIAS): it has no station of its own, so alone it would drop out. The
French half is fr_register's "Tunnel sous la Manche" from the same point; the extract's copy of
it is clipped out (`--clip`, clip_channel, after every extract).

The `path` argument is data/raw/gb; the OSM half is read from data/proc/gb (extract.py).
"""
import csv
import hashlib
import heapq
import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import kr_register as kr
import pieces

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "gb"
REGION = "gb"
USER_AGENT = "noritetsu-build/1.0 (hobby rail map)"

# A station is on every named line the route stopping there runs on within this of it.
ALONG_M = 250
# An unnamed way takes its ELR's main name if that name has this share of the ELR's named km;
# so does a way named otherwise, unless its own name has this much track on the ELR.
ELR_FILL_SHARE = 0.6
ELR_RENAME_MAX_KM = 40
# Border points this close to a line's track become its stations. RINF's Channel Tunnel point
# (eEU00228) lay 450 m off OSM's tunnels, whose ways are cut at the border 1.4963 E, 51.0150 N;
# borders.MOVE has put it between the two bores' cut nodes since 2026-10-05 (25 m from each).
BORDER_M = 600
# A name with less than this share of its track carrying an ELR is no part of the main network.
NOREF_SHARE = 0.1
# Northern Ireland Railways' lines (no Network Rail ELRs; OSM's refs there are B, D, R, P) and
# the Tyne and Wear Metro's Pelaw - Sunderland stretch, which Northern's trains share.
KEEP_NOREF = {"Derry~Londonderry Line", "Belfast Central Railway", "Bangor Line", "Larne Line",
              "Portrush Branch", "Great Northern Railway Main Line", "Antrim Branch",
              "Metro Shared Network"}

TRACK_KIND = {"rail": "rail", "narrow_gauge": "narrow_gauge"}
NOT_PASSENGER = {"industrial", "military", "test", "tourism"}

# Names that are one line under another spelling, or that would drop out alone. Exact, before
# folding. The folded spellings (dashes, "and", Line/Railway) need no entry.
NAME_ALIAS = {
    "Tunnel sous la Manche / Channel Tunnel ‒ Tunnel Ferroviaire Nord / Running Tunnel North":
        "High Speed 1",
    "Tunnel sous la Manche / Channel Tunnel ‒ Tunnel Ferroviaire Sud / Running Tunnel South":
        "High Speed 1",
    "Channel Tunnel": "High Speed 1",
    "Tunnel sous la Manche / Channel Tunnel": "High Speed 1",
    "Channel Tunnel Rail Link": "High Speed 1",
    # the few hundred metres at Cheriton between High Speed 1 and the tunnel mouths
    "Eurotunnel terminal": "High Speed 1",
    "Eurotunnel Track 1": "High Speed 1",
    "Eurotunnel Track 2": "High Speed 1",
    "District Line and Overground (North London Line)": "North London Line",
    "Overground (North London Line)": "North London Line",
    # Northern Ireland Railways' own name for it; OSM has three spellings.
    "Londonderry Line": "Derry~Londonderry Line",
    "Derry Line": "Derry~Londonderry Line",
    "Derry line": "Derry~Londonderry Line",
    "Rugby–Birmingham–Stafford Line (West Coast Main Line)": "Rugby–Birmingham–Stafford Line",
}

# Station records that are one station (kr_register.name_key spelling, dots already gone):
# HS1's platforms are mapped as "London St. Pancras International", the Midland's as "London
# St Pancras", and a 0.2 km section joined them.
STATION_ALIAS = {"London St Pancras International": "London St Pancras"}

# The Underground's own lines where OSM maps their track as railway=rail: left to the
# Underground's OSM routes (a subway line owns the track it alone runs on).
METRO_ON_RAIL = re.compile(
    r"^(Metropolitan|District|Bakerloo|Circle|Hammersmith|Jubilee|Piccadilly|Central|Victoria|"
    r"Waterloo (?:&|and) City)\b.*\blines?\b", re.IGNORECASE)

# Per-track and non-names: read as no name, so the way takes its ELR's.
# "DM - Down Main", "Down Goole", "Up Main"; and a structure's or a siding's name put on the
# track ("Victoria Bridge" on the Severn Valley, "Sharpthorne Tunnel", "Bamber Bridge Sidings").
JUNK_NAME = re.compile(r"^(?:[UD][MFSR]|Up|Down|Platform|Track)\b"
                       r"|\b(?:Bridge|Viaduct|Tunnel|Arches|Sidings?|Passing Loop|Yard|Depot|Dock|"
                       r"Station)$", re.IGNORECASE)
UPDOWN = re.compile(r"\s*\((?:Up|Down|Up (?:&|and) Down|Reversible)\b[^)]*\)\s*$"
                    r"|\s+(?:Up|Down)(?: Line)?$", re.IGNORECASE)
TRACK_WORDS = re.compile(r"\b(?:line|lines|railway|branch line|trackage)\b", re.IGNORECASE)


def line_id(name):
    h = hashlib.blake2b(f"gb|{name}".encode("utf-8"), digest_size=5)
    return "g" + h.hexdigest()


def tidy(name):
    """A track name as OSM writes it, before folding: aliases, the English half of a Welsh
    "Rheilffordd ... / ... Line", a track's "(Up Southport)" dropped; "" for no name."""
    n = (name or "").strip()
    n = NAME_ALIAS.get(n, n)
    if " / " in n:
        parts = [p.strip() for p in n.split(" / ")]
        eng = [p for p in parts if re.search(r"\b(Line|Railway|Branch|Loop)\b", p)]
        n = eng[-1] if eng else parts[-1]
        n = NAME_ALIAS.get(n, n)
    n = UPDOWN.sub("", n).strip()
    if JUNK_NAME.search(n):
        return ""
    return n


def fold_key(name):
    """One key for the spellings of one line: Rugby-Birmingham-Stafford Line, Rugby–Birmingham–
    Stafford Line and Rugby Birmingham and Stafford Line; Tyne Valley Line and Tyne Valley
    Railway; Leeds-Lancaster-Morecambe and Leeds to Morecambe are NOT folded (different words)."""
    k = name.lower().replace("&", " and ")
    k = re.sub(r"[-‐-―−~/,.'’()]", " ", k)
    k = TRACK_WORDS.sub(" ", k)
    k = re.sub(r"\b(?:and|the)\b", " ", k)
    return " ".join(k.split())


# ---------------------------------------------------------------- state shared with kr_register

_S = {}          # ways, rels, stops, coords, canonical names, ELR fill, stations


# Wikidata heritage railways whose folded name is a National Rail line's: the Barry Tourist
# Railway's "Vale of Glamorgan Railway" against Network Rail's Vale of Glamorgan Line.
HERITAGE_NOT = {"Vale of Glamorgan Railway"}
# Heritage railways whose track names Wikidata's labels do not catch: the Gwili Railway's
# carries its Welsh name first, and the Ecclesbourne Valley Railway runs on the old Wirksworth
# branch under that name (no National Rail train runs north of Duffield). Each was a register
# line in two pieces (gb_sources.md "Lines in pieces").
HERITAGE_EXTRA = {"Rheilffordd Gwili (Gwili Railway)", "Wirksworth Branch Line"}

# A register line's name on another line's track in one area: (the names, (west, south, east,
# north), the line there), by the way's middle, applied to the name a way ends up with (its
# own, its ELR's or its neighbours'). Mappers wrote "Great Western Main Line" and "Great
# Western Railway" (Network Rail's MLN1, which the ELR rule names the Great Western Main Line,
# runs Paddington - Penzance) on stretches in Devon: Exeter - Dawlish, Totnes - Plymouth (one
# of the two tracks) and around Tiverton Parkway, four pieces 100 km and more from Paddington
# - Bristol, the line the name means. They are the South Devon Main Line south of Exeter St
# Davids and the Bristol to Exeter Line north of it.
GWML = frozenset({"Great Western Main Line", "Great Western Railway"})
AREA_NAME = [
    (GWML, (-4.4, 50.2, -3.0, 50.731), "South Devon Main Line"),
    (GWML, (-3.7, 50.731, -2.9, 51.2), "Bristol to Exeter Line"),
]


def load_heritage():
    extra = {heritage_key(n) for n in HERITAGE_EXTRA}
    p = RAW / "wd_heritage.csv"
    if not p.exists():
        return extra
    with open(p, encoding="utf-8") as f:
        return extra | {heritage_key(r["label"]) for r in csv.DictReader(f)
                        if r.get("label") and r["label"] not in HERITAGE_NOT}


def area_ways(ways, coords, log):
    """{id(tags): (names, line)} for the rail ways inside an AREA_NAME box."""
    out = {}
    for t, nodes in ways.values():
        if t.get("railway") not in TRACK_KIND or not len(nodes):
            continue
        p = None
        for names, (w, s, e, n), new in AREA_NAME:
            if p is None:
                p = coords.get(int(nodes[len(nodes) // 2])) or (999.0, 999.0)
            if w <= p[0] <= e and s <= p[1] <= n:
                out[id(t)] = (names, new)
                break
    _S["area"] = out
    got = Counter(new for t, _n in ways.values() if id(t) in out
                  for names, new in [out[id(t)]] if _own_name(t) in names)
    log(f"GB: by area (AREA_NAME): " + ", ".join(f"{k} ways -> {v}" for v, k in got.items()))


def area_name(tags, n):
    got = _S.get("area", {}).get(id(tags))
    return got[1] if got and n in got[0] else n


def heritage_key(name):
    """Stricter than fold_key: "Line" and "Railway" stay apart, since the East Lancashire Line
    (Preston - Colne, Northern) is not the East Lancashire Railway (Bury, heritage)."""
    k = name.lower().replace("&", " and ")
    k = re.sub(r"[-‐-―−~/,.'’()]", " ", k)
    k = re.sub(r"^the\s+", "", k.strip())
    return " ".join(k.split())


def way_km(ways, coords):
    out = {}
    for wid, (tags, nodes) in ways.items():
        pos, ok = coords.many(np.asarray(nodes, dtype=np.int64))
        pos = pos[ok]
        if pos.size < 2:
            out[wid] = 0.0
            continue
        x, y = coords.x[pos] / 1e7, coords.y[pos] / 1e7
        lat = np.radians((y[:-1] + y[1:]) / 2)
        out[wid] = float(np.hypot(np.diff(x) * np.cos(lat) * 111.32, np.diff(y) * 110.57).sum())
    return out


def name_table(ways, coords, log):
    """Canonical name per fold key (the spelling with most km), the ELR fill, and what was left
    out, from the extract's own ways."""
    heritage = load_heritage()
    km = way_km(ways, coords)
    spell = defaultdict(Counter)                   # key -> spelling -> km
    elr = defaultdict(Counter)                     # ELR -> key -> km
    out = Counter()
    for wid, (t, _n) in ways.items():
        if t.get("railway") not in TRACK_KIND or t.get("service"):
            continue
        n = tidy(t.get("name"))
        if not n:
            continue
        why = None
        if t.get("usage") in NOT_PASSENGER:
            why = f"usage={t.get('usage')}"
        elif METRO_ON_RAIL.match(n):
            why = "Underground"
        elif heritage_key(n) in heritage:
            why = "heritage"
        if why:
            out[(why, n)] += km[wid]
            # a left-out line still claims its ELR, so the ELR rule never hands a heritage
            # railway's or the Underground's unnamed track to a National Rail line
            if t.get("ref"):
                elr[t["ref"]][None] += km[wid]
            continue
        k = fold_key(n)
        spell[k][n] += km[wid]
        if t.get("ref"):
            elr[t["ref"]][k] += km[wid]
    canon = {k: c.most_common(1)[0][0] for k, c in spell.items()}
    fill = {}
    for e, c in elr.items():
        k, v = c.most_common(1)[0]
        if v >= ELR_FILL_SHARE * sum(c.values()):
            fill[e] = canon[k] if k is not None else ""
    # names that are no part of Network Rail's or NIR's network: no ELR on any of their track
    # (miniature and estate railways, "Dock", a heritage line Wikidata does not list)
    reffed = Counter()
    for c in elr.values():
        for k, v in c.items():
            reffed[k] += v
    noref = {k for k, c in spell.items() if reffed[k] < NOREF_SHARE * sum(c.values())
             and canon[k] not in KEEP_NOREF}
    for k in sorted(noref, key=lambda k: -sum(spell[k].values())):
        out[("no ELR", canon[k])] += sum(spell[k].values())
    for e, nm in list(fill.items()):
        if nm and fold_key(nm) in noref:
            fill[e] = ""
    # how many named ways the ELR rule renames, by km
    moved = Counter()
    for wid, (t, _n) in ways.items():
        if t.get("railway") in TRACK_KIND and not t.get("service") and t.get("ref") in fill:
            n = tidy(t.get("name"))
            k = fold_key(n) if n else ""
            if (n and fill[t["ref"]] and canon.get(k, n) != fill[t["ref"]]
                    and elr[t["ref"]][k] < ELR_RENAME_MAX_KM):
                moved[(canon.get(k, n), fill[t["ref"]])] += km[wid]
    folded = {k: c for k, c in spell.items() if len(c) > 1}
    log(f"GB: {len(canon)} line names on rail track after folding ({sum(len(c) for c in spell.values())}"
        f" spellings; {len(folded)} names had several); {len(fill)} ELRs have a main name "
        f"(>= {ELR_FILL_SHARE:.0%} of their km), which {sum(moved.values()):,.0f} km of "
        f"differently named track takes")
    for (a, b), v in moved.most_common(40):
        log(f"    ELR rename: {a} -> {b} {v:.1f} km")
    for k, c in sorted(folded.items(), key=lambda kv: -sum(kv[1].values()))[:40]:
        log("    folded: " + " | ".join(f"{n} {v:.0f}" for n, v in c.most_common()))
    by = defaultdict(float)
    for (why, _n), v in out.items():
        by[why] += v
    log("GB: left out of the register (km of track): "
        + ", ".join(f"{w} {v:,.0f}" for w, v in sorted(by.items())))
    for (why, n), v in out.most_common(60):
        log(f"    {why}: {n} {v:.1f} km")
    _S["elrkm"] = elr
    return canon, fill, heritage, noref


def register_name(tags):
    """The register line a way belongs to, "" for none: its own name or its ELR's
    (`base_name`), or for an unnamed way the name `propagate` gave it."""
    got = _S.get("byobj", {}).get(id(tags))
    return area_name(tags, got) if got is not None else base_name(tags)


def propagate(ways, log):
    """Names for unnamed track from the track it joins. OSM UK leaves stretches of a named line
    unnamed, or puts a bridge's name on them ("Ouse Bridge", "Setchey Bridge" on the Fen Line,
    whose ELR, BGK, has no one main name to give them): each such gap cut the line in two, and
    the stations either side never met. An unnamed way takes a name that the ways at BOTH its
    ends carry; or, a dead end, the one name of the ways at its single joined end on its own
    ELR. Repeated until nothing changes, so a run of unnamed ways fills from both sides."""
    wids = list(ways)
    base = {w: base_name(ways[w][0]) for w in wids}
    cand = set()
    for w in wids:
        t = ways[w][0]
        if t.get("railway") not in TRACK_KIND or t.get("usage") in NOT_PASSENGER:
            continue
        if t.get("service") not in (None, "crossover"):
            continue
        if tidy(t.get("name")):
            continue                      # named: its own name (or its ELR's) stands
        ref = t.get("ref") or ""
        if ref in _S["fill"] and not _S["fill"][ref]:
            continue                      # its ELR belongs to a left-out line
        cand.add(w)
        base[w] = ""                      # the neighbours first, the ELR's name after
    # node -> ways, over every way, by sorted arrays
    idx = np.repeat(np.arange(len(wids)), [len(ways[w][1]) for w in wids])
    nodes = np.concatenate([np.asarray(ways[w][1], dtype=np.int64) for w in wids])
    order = np.argsort(nodes, kind="stable")
    nodes, idx = nodes[order], idx[order]

    def at(n):
        i, j = np.searchsorted(nodes, n), np.searchsorted(nodes, n, side="right")
        return [wids[k] for k in idx[i:j].tolist()]

    ends = {w: (at(int(ways[w][1][0])), at(int(ways[w][1][-1]))) for w in cand}
    name = dict(base)

    def spread(one_sided):
        n_new = 0
        for _round in range(400):
            new = {}
            for w in cand:
                if name[w]:
                    continue
                ref = ways[w][0].get("ref")
                a = {name[o] for o in ends[w][0] if o != w and name[o]}
                b = {name[o] for o in ends[w][1] if o != w and name[o]}
                both = a & b
                if len(both) == 1:
                    new[w] = next(iter(both))
                    continue
                if one_sided and (not a or not b):
                    side = a or b
                    same = {name[o] for o in (ends[w][0] if a else ends[w][1])
                            if o != w and name[o] and ref and ways[o][0].get("ref") == ref}
                    if len(side) == 1 and len(same) == 1:
                        new[w] = next(iter(same))
            if not new:
                break
            name.update(new)
            n_new += len(new)
        return n_new

    # a bridge or tunnel inside the Dawlish sea wall is the South Devon Main Line's, whatever
    # the ELR's main name (MLN1, the Great Western Main Line) says: neighbours first
    total = spread(False)
    by_elr = 0
    for w in cand:
        ref = ways[w][0].get("ref") or ""
        if not name[w] and _S["fill"].get(ref):
            name[w] = _S["fill"][ref]
            by_elr += 1
    total += spread(True)
    km = way_km({w: ways[w] for w in cand}, _S["coords"])
    got = sum(km[w] for w in cand if name[w])
    log(f"GB: {total + by_elr} unnamed ways named ({total} from the track they join, {by_elr} "
        f"from their ELR; {got:,.0f} km of track); {sum(1 for w in cand if not name[w])} left "
        f"unnamed ({sum(km[w] for w in cand if not name[w]):,.0f} km)")
    absorb(ways, name, at, log)
    _S["byobj"] = {id(ways[w][0]): name[w] for w in wids if name[w] != base_name(ways[w][0])}


ABSORB_KM = 12.0      # track-km; a named piece this short, with other lines at every end


def absorb(ways, name, at, log):
    """A short piece named otherwise in the middle of a line is that line: OSM's patchwork of
    Wikipedia names and Network Rail's ELR descriptions leaves 1-5 km pieces (the Great Western
    Main Line's name on a few ways at Dawlish and Totnes, on MLN1 but 200 km from the rest of
    it), each cutting the line around it in two. A stray piece of a line (not its biggest
    connected piece) under ABSORB_KM of track whose every dead end lies on other lines becomes
    the line it touches at most ends (the bigger one on a tie). A line in one piece, however
    short, and a piece with a free end (a terminus, a branch) stay themselves. One pass, so two
    lines never trade pieces. (Tried 2026-10-03 on whole short lines too: it folded the
    Pontefract Line, Guildford - Wokingham and Leamington - Stratford into their neighbours.)"""
    km = way_km(ways, _S["coords"])
    merged = []
    total_km = Counter()
    for w, n in name.items():
        if n:
            total_km[n] += km[w]
    for _round in range(1):
        by = defaultdict(list)
        for w, n in name.items():
            if n:
                by[n].append(w)
        ren = {}                                  # way -> new name
        for ln, ws in by.items():
            # the line's connected pieces
            parent = {w: w for w in ws}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            first = {}
            for w in ws:
                for n in np.asarray(ways[w][1]).tolist():
                    if n in first:
                        parent[find(w)] = find(first[n])
                    else:
                        first[n] = w
            pieces = defaultdict(list)
            for w in ws:
                pieces[find(w)].append(w)
            if len(pieces) == 1:
                continue                          # a line in one piece is a line, however short
            pkm = {p: sum(km[w] for w in pws) for p, pws in pieces.items()}
            biggest = max(pkm, key=pkm.get)
            for p, pws in pieces.items():
                pk = pkm[p]
                if p == biggest or pk >= ABSORB_KM:
                    continue
                inc = Counter()
                for w in pws:
                    nl = np.asarray(ways[w][1]).tolist()
                    inc[nl[0]] += 1
                    inc[nl[-1]] += 1
                    for n in nl[1:-1]:
                        inc[n] += 2
                dead = [n for n, k in inc.items() if k == 1]
                if len(dead) < 2:
                    continue
                touch = Counter()
                free = False
                for n in dead:
                    o = {name.get(x) for x in at(n)} - {ln, None, ""}
                    if not o:
                        free = True
                    touch.update(o)
                if free or not touch:
                    continue
                best = max(touch, key=lambda o: (touch[o], total_km[o]))
                for w in pws:
                    ren[w] = best
                merged.append((ln, best, pk))
        if not ren:
            break
        name.update(ren)
    log(f"GB: {len(merged)} short named stretches folded into the line around them "
        f"({sum(m[2] for m in merged):,.0f} km of track)")
    for a, b, k in sorted(merged, key=lambda m: -m[2])[:30]:
        log(f"    {a} -> {b} {k:.1f} km")


def base_name(tags):
    """The register line a way belongs to by its own tags (_own_name), AREA_NAME applied."""
    return area_name(tags, _own_name(tags))


def _own_name(tags):
    """The register line a way belongs to by its own tags, "" for none. kr_register has already
    left out usage industrial/military/test/tourism and every railway kind not in TRACK_KIND."""
    if tags.get("railway") not in TRACK_KIND or tags.get("usage") in NOT_PASSENGER:
        return ""
    n = tidy(tags.get("name"))
    if n and (METRO_ON_RAIL.match(n) or heritage_key(n) in _S["heritage"]):
        return ""
    svc = tags.get("service")
    if svc and svc != "crossover":
        # a siding or yard joins no line, unless it carries a line's own name
        k = fold_key(n) if n else ""
        return _S["canon"][k] if k in _S["canon"] and k not in _S["noref"] else ""
    ref = tags.get("ref") or ""
    k = fold_key(n) if n else ""
    if ref in _S["fill"]:
        # the ELR's main name, unless this way's own name is a line of its own on the ELR
        # (MLN1 is Paddington - Penzance: its Bristol to Exeter Line stays itself)
        if not (k and k in _S["canon"] and k not in _S["noref"]
                and _S["elrkm"][ref][k] >= ELR_RENAME_MAX_KM):
            return _S["fill"][ref]        # "" where a left-out line has the ELR
    if not n:
        return ""
    if k in _S["noref"]:
        return ""
    return _S["canon"].get(k, n)


def load_osm(log):
    import build_model as bm
    ways, rels, stops, cid, cx, cy = bm.load(REGION, log)
    coords = bm.Coords(cid, cx, cy)
    _S.update(ways=ways, rels=rels, stops=stops, coords=coords)
    _S["canon"], _S["fill"], _S["heritage"], _S["noref"] = name_table(ways, coords, log)
    _S.pop("byobj", None)
    area_ways(ways, coords, log)
    propagate(ways, log)
    return ways, stops, coords


# Fake node ids for border points: never an OSM id, and negative so kr_register's int(sid[1:])
# still reads them. Renamed to the border point's own id at the end.
BORDER_BASE = -9_000_000_000


def border_points(log):
    try:
        import borders
        pts = [p for p in borders.load(canonical_only=True) if "gb" in p["countries"]]
    except Exception as e:                          # noqa: BLE001 - a build without borders still works
        log(f"GB: no border points ({e})")
        return []
    return pts


def build_stations(stops, log):
    st, node_st, by_key, by_base = _orig["build_stations"](stops, log)
    border = {}
    for i, p in enumerate(border_points(log)):
        fid = BORDER_BASE - i
        st[fid] = {"name": p["id"], "name_en": "", "lon": p["lon"], "lat": p["lat"], "rank": 0}
        by_key[kr.name_key(p["id"])].append(fid)
        by_base[kr.base_key(p["id"])].append(fid)
        border[fid] = p["id"]
    _S.update(st=st, node_st=node_st, border=border)
    log(f"GB: {len(border)} border points offered to the lines ({', '.join(border.values())})")
    junc = junction_ends(log)
    for n in junc:
        if n in st:
            continue
        p = _S["coords"].get(n)
        if p is None:
            continue
        st[n] = {"name": f"gj{n}", "name_en": "", "lon": p[0], "lat": p[1], "rank": 3}
        by_key[kr.name_key(f"gj{n}")].append(n)
        by_base[kr.base_key(f"gj{n}")].append(n)
    _S["junction"] = {n: v for n, v in junc.items() if st.get(n, {}).get("name") == f"gj{n}"}
    return st, node_st, by_key, by_base


def junction_ends(log):
    """{node: {line}}: where a line's own track ends on another register line's track. The UK's
    named lines end at junctions, not stations: the Settle-Carlisle Railway at Settle Junction
    and Petteril Bridge Junction, the Fen Line at Ely North Junction. kr_register makes sections
    only between a line's stations, so the stretch from a line's last station to its junction
    belonged to no line. Each such end is a `junction` station of that line ("gj<node>"),
    and build_model keeps the section to it only where passenger routes run over it."""
    ways = _S["ways"]
    inc = defaultdict(Counter)                  # line -> node -> 1 per way end, 2 per interior
    on = defaultdict(set)                       # node -> lines whose track has it
    for w, (t, nodes) in ways.items():
        ln = register_name(t)
        if not ln:
            continue
        nl = np.asarray(nodes).tolist()
        c = inc[ln]
        c[nl[0]] += 1
        c[nl[-1]] += 1
        for n in nl[1:-1]:
            c[n] += 2
        for n in nl:
            on[n].add(ln)
    out = defaultdict(set)
    for ln, c in inc.items():
        for n, k in c.items():
            if k == 1 and len(on[n]) > 1:
                out[n].add(ln)
    log(f"GB: {len(out)} junction ends, where a line's track ends on another line's "
        f"({sum(len(v) for v in out.values())} line ends)")
    return out


def route_lists(log):
    """{line: [[station name, ...]]}: which stations each named line has, from the routes that
    stop there and the names of their own ways near each stop (the docstring)."""
    import build_model as bm
    ways, rels, coords = _S["ways"], _S["rels"], _S["coords"]
    st, node_st = _S["st"], _S["node_st"]
    # every way's vertices once, as arrays, for the ways routes use
    wxy = {}

    def xy(w):
        if w not in wxy:
            nodes = np.asarray(ways[w][1], dtype=np.int64)
            pos, ok = coords.many(nodes)
            pos = pos[ok]
            wxy[w] = (coords.x[pos] / 1e7, coords.y[pos] / 1e7)
        return wxy[w]

    lists = defaultdict(set)
    n_routes = n_pairs = 0
    for rid, (tags, members) in rels.items():
        if tags.get("type") != "route" or tags.get("route") not in bm.ROUTE_KINDS:
            continue
        rw = [r for ty, r, _ in members if ty == "w" and r in ways]
        names = {w: register_name(ways[w][0]) for w in rw}
        rw = [w for w in rw if names[w]]
        if not rw:
            continue
        n_routes += 1
        for n in bm.stop_members(members):
            s = node_st.get(n)
            if s is None:
                continue
            lon, lat = st[s]["lon"], st[s]["lat"]
            kx = math.cos(math.radians(lat)) * 111320
            for w in rw:
                x, y = xy(w)
                if x.size and np.min(np.hypot((x - lon) * kx, (y - lat) * 110570)) <= ALONG_M:
                    if st[s]["name"] not in lists[names[w]]:
                        lists[names[w]].add(st[s]["name"])
                        n_pairs += 1
    # border points: on the line whose track passes within BORDER_M
    for fid, pid in _S["border"].items():
        p = st[fid]
        kx = math.cos(math.radians(p["lat"])) * 111320
        for w, (t, _n) in ways.items():
            nm = register_name(t)
            if not nm or t.get("usage") in NOT_PASSENGER:
                continue
            x, y = xy(w)
            if x.size < 2 or not (x.min() - 0.01 <= p["lon"] <= x.max() + 0.01
                                  and y.min() - 0.01 <= p["lat"] <= y.max() + 0.01):
                continue
            # to the segments, not the vertices: a tunnel way is straight for kilometres
            ax, ay = (x[:-1] - p["lon"]) * kx, (y[:-1] - p["lat"]) * 110570
            bx, by = (x[1:] - p["lon"]) * kx, (y[1:] - p["lat"]) * 110570
            dx, dy = bx - ax, by - ay
            L2 = np.maximum(dx * dx + dy * dy, 1e-9)
            t = np.clip(-(ax * dx + ay * dy) / L2, 0, 1)
            if np.min(np.hypot(ax + t * dx, ay + t * dy)) <= BORDER_M:
                lists[nm].add(pid)
                log(f"GB: border point {pid} on {nm}")
    for n, lns in _S.get("junction", {}).items():
        for ln in lns:
            lists[ln].add(f"gj{n}")
    log(f"GB: station lists from {n_routes} OSM routes: {n_pairs} station-line pairs on "
        f"{len(lists)} lines")
    return lists


def load_lists(path, log):
    lists = route_lists(log)
    return ({k: [sorted(v)] for k, v in lists.items()}, defaultdict(list), {})


_orig = {}


def adopt():
    """Point kr_register's country-specific globals at the UK's, for this process."""
    if _orig:
        return
    _orig["build_stations"] = kr.build_stations
    kr.load_osm = load_osm
    kr.register_name = register_name
    kr.load_lists = load_lists
    kr.build_stations = build_stations
    kr.line_id = line_id
    kr.TRACK_KIND = TRACK_KIND
    kr.NOT_PASSENGER = NOT_PASSENGER
    kr.NAME_ALIAS = {}
    kr.STATION_ALIAS = STATION_ALIAS


# A line's junction-ended sections to these stations are served whatever OSM's routes cover
# (build_model.drop_unridden_sections reads `served_sections`). High Speed 1's London end, the
# 0.3 km from the throat junction into St Pancras International, was dropped as unridden: the
# station's nine HS1 platform roads (5-13) are mapped and the routes name only a few, so under
# half of the track beside the section is on a route, and HS1 had no London station at all
# (gb_sources.md "The Channel Tunnel"). Every Eurostar and Southeastern high-speed train uses it.
SERVED_END = {"High Speed 1": {"London St. Pancras International"}}

SHORTCUT_SLACK = 1.15


def drop_shortcuts(lines, geoms, log):
    """A section that runs past its line's stations on a parallel track: Bruton - Pewsey (58 km)
    on the Reading to Taunton Line by the Westbury avoiding line, beside Bruton - Westbury -
    Pewsey. kr_register pairs two stations whenever track joins them without a third station
    between, and the UK's avoiding lines and loops carry the line's name. A section is dropped
    when the line's other sections join its two ends within SHORTCUT_SLACK of its length,
    longest first, so of two such alternatives one always stays."""
    from n02 import walk_order
    n_drop, km_drop = 0, 0.0
    for l in lines:
        secs = sorted(l["sections"], key=lambda s: -s[2])
        keep = list(secs)
        for s in secs:
            rest = [x for x in keep if x is not s]
            adj = defaultdict(list)
            for a, b, km, *_ in rest:
                adj[a].append((b, km))
                adj[b].append((a, km))
            limit = s[2] * SHORTCUT_SLACK
            dist, heap = {s[0]: 0.0}, [(0.0, s[0])]
            found = False
            while heap:
                d, u = heapq.heappop(heap)
                if d > limit:
                    break
                if u == s[1]:
                    found = True
                    break
                if d > dist.get(u, 1e18):
                    continue
                for v, w in adj[u]:
                    nd = d + w
                    if nd < dist.get(v, 1e18):
                        dist[v] = nd
                        heapq.heappush(heap, (nd, v))
            if found:
                keep = rest
                n_drop += 1
                km_drop += s[2]
                geoms[l["id"]].pop(f"{s[0]}|{s[1]}", None)
                if isinstance(l.get("highspeed_sections"), dict):
                    l["highspeed_sections"].pop(f"{s[0]}|{s[1]}", None)
        if len(keep) != len(l["sections"]):
            ks = {(a, b) for a, b, *_ in keep}
            l["sections"] = [s for s in l["sections"] if (s[0], s[1]) in ks]
            l["km"] = round(sum(s[2] for s in l["sections"]), 3)
            l["display"] = walk_order([(a, b) for a, b, *_ in l["sections"]])
    log(f"GB: {n_drop} sections dropped as shortcuts past their line's own stations "
        f"({km_drop:,.0f} km)")


# ---------------------------------------------------------------- lines in pieces

# The bridging and splitting are in pieces.py (moved there 2026-10-04 so kr, cn and tr share
# them); these are the UK's settings, read at call time (pieces.Rules has the meaning of each).
# A gap in a line is bridged over the track between its pieces when the track found joining
# them is at most BRIDGE_SLACK times the crow-fly between its ends plus BRIDGE_PLUS_KM, no
# longer than BRIDGE_MAX_KM, at least BRIDGE_ROUTE_SHARE of it under an OSM passenger route,
# and no longer than the smaller side it joins unless under BRIDGE_FREE_KM (a 2 km stray piece
# of a name 40 km from the rest of it is that name on another line's track, not the line
# running on). gb_sources.md "Lines in pieces" has the measurements.
BRIDGE_SLACK = 1.5
BRIDGE_PLUS_KM = 5.0
BRIDGE_MAX_KM = 100.0
BRIDGE_ROUTE_SHARE = 0.5
BRIDGE_FREE_KM = 15.0
# A bridge this short needs no route over it: platform roads and station throats are often
# left out of route relations (Chester's, between the North Wales Coast Line's two pieces).
BRIDGE_ROUTE_FREE_KM = 2.0
# The search prefers track under a passenger route: a way no route uses costs this many times
# its length (the relief pair of a four-track main line, a freight curve).
UNROUTED_COST = 4.0
ATTACH_M = 150           # a station of a piece joins the track at its nearest vertex this close
CUT_M = 150              # a station of a line the bridge runs over cuts it this close
# Beside a borrowed section, a way is recorded at least this much further off than the nearest
# other register line's section, for ownership (see pieces.bridge_gaps): more than ownership's
# EXACT_TIE_M, less than its TIE_M.
BORROW_PAD_M = 2.0
# A piece of a line left in pieces with under two stops and shorter than this is not made a
# line of its own (us_register's rule, at 0.1 km there): London - Aylesbury's 0.2 km between
# two junctions at Harlesden.
PIECE_MIN_KM = 0.5
# {line id: [the ids of the pieces split off it]}, filled by split_pieces; build_model writes it
# into aliases.json as `pieces`, so the app moves a saved ride onto the piece that has its stops.
LINE_PIECES = {}
section_pieces = pieces.section_pieces


def rules():
    return pieces.Rules(
        tag="GB", id_prefix="g", lat=54.5, slack=BRIDGE_SLACK, plus_km=BRIDGE_PLUS_KM,
        max_km=BRIDGE_MAX_KM, route_share=BRIDGE_ROUTE_SHARE, free_km=BRIDGE_FREE_KM,
        route_free_km=BRIDGE_ROUTE_FREE_KM, unrouted_cost=UNROUTED_COST, attach_m=ATTACH_M,
        cut_m=CUT_M, borrow_pad_m=BORROW_PAD_M, piece_min_km=PIECE_MIN_KM)


def classify(wid, t, routed):
    """For pieces.track_graph: every way a register line's name is on, and every other way
    under an OSM passenger route, but track left out of the register on purpose stays out:
    heritage railways, and the Underground's own rail-tagged lines (London - Aylesbury's
    Harrow - Amersham). The register name, "" for none, None for a way left out."""
    if t.get("railway") not in TRACK_KIND or t.get("usage") in NOT_PASSENGER:
        return None
    nm = register_name(t)
    if not nm:
        if not routed:
            return None
        n = tidy(t.get("name"))
        if n and (heritage_key(n) in _S["heritage"] or METRO_ON_RAIL.match(n)):
            return None
    return nm


def track_graph(log, r=None):
    """The passenger track as a graph for bridging (pieces.track_graph over `classify`), or
    None when the extract was not loaded."""
    if "ways" not in _S:
        return None
    return pieces.track_graph(_S["ways"], _S["coords"], pieces.routed_ways(_S["rels"]),
                              classify, r or rules(), log)


def split_pieces(lines, stations, geoms, reg_ways, state, log):
    """pieces.split_pieces with the UK's rules: lines in pieces bridged over the track between
    them where trains run across, the rest one line per piece (the build_model hook)."""
    r = rules()
    pieces.split_pieces(lines, stations, geoms, reg_ways, state, log, r, LINE_PIECES,
                        lambda: track_graph(log, r), _S.get("ends", ()))


def build(path, log):
    adopt()
    lines, stations, geoms = kr.build(path, log)
    drop_shortcuts(lines, geoms, log)
    # kr_register's ids are k<node>; ours are g<node>, and a border point its own id
    border = {f"k{fid}": pid for fid, pid in _S.get("border", {}).items()}
    junc = {f"k{n}": f"gj{n}" for n in _S.get("junction", {})}

    def r(sid):
        return (border.get(sid) or junc.get(sid)
                or ("g" + sid[1:] if sid.startswith("k") else sid))

    real = [s for sid, s in stations.items() if sid not in border and sid not in junc]
    rx = np.array([s["lon"] for s in real]) if real else np.zeros(0)
    ry = np.array([s["lat"] for s in real]) if real else np.zeros(0)
    out_st = {}
    for sid, s in stations.items():
        nid = r(sid)
        s["id"] = nid
        if sid in border:
            s["junction"] = True
            s["name"] = nid
        elif sid in junc:
            s["junction"] = True
            d = np.hypot((rx - s["lon"]) * math.cos(math.radians(s["lat"])), ry - s["lat"])
            s["name"] = f"Junction near {real[int(np.argmin(d))]['name']}" if d.size else "Junction"
        out_st[nid] = s
    out_geoms = {}
    for l in lines:
        l["src"] = "gb"
        l["sections"] = [[r(a), r(b), *rest] for a, b, *rest in l["sections"]]
        l["display"] = [r(x) for x in l["display"]]
        if isinstance(l.get("highspeed_sections"), dict):
            l["highspeed_sections"] = {"|".join(r(x) for x in k.split("|")): v
                                       for k, v in l["highspeed_sections"].items()}
        out_geoms[l["id"]] = {"|".join(r(x) for x in k.split("|")): v
                              for k, v in geoms[l["id"]].items()}
    # station membership from the sections as they now are (drop_shortcuts may have removed
    # a station's last section on a line)
    for s in out_st.values():
        s["lines"] = set()
    for l in lines:
        for a, b, *_ in l["sections"]:
            out_st[a]["lines"].add(l["id"])
            out_st[b]["lines"].add(l["id"])
    out_st = {k: s for k, s in out_st.items() if s["lines"]}
    for l in lines:
        names = SERVED_END.get(l["name"])
        if names:
            l["served_sections"] = [f"{a}|{b}" for a, b, *_ in l["sections"]
                                    if {out_st[a]["name"], out_st[b]["name"]} & names]
    # where each station is on each line's track, for bridge_gaps after build_model's drops
    _S["ends"] = [(sid, l["name"], *pts[0 if k == 0 else -1])
                  for l in lines for key, pts in out_geoms[l["id"]].items()
                  for k, sid in enumerate(key.split("|"))]
    log(f"GB: {len(lines)} register lines, {sum(l['km'] for l in lines):,.0f} km, "
        f"{len(out_st)} stations, of which {sum(1 for s in out_st.values() if s.get('junction'))}"
        f" junction ends and border points")
    return lines, out_st, out_geoms


# ---------------------------------------------------------------- the small sources

WD = "https://query.wikidata.org/sparql"
Q_ELR = """SELECT ?item ?elr ?label ?miles WHERE {
  ?item wdt:P10271 ?elr .
  OPTIONAL { ?item rdfs:label ?label FILTER(LANG(?label) = "en") }
  OPTIONAL { ?item wdt:P2043 ?miles }
}"""
Q_LINES = """SELECT ?item ?label ?m ?osm ?closed WHERE {
  ?item wdt:P17 wd:Q145 .
  ?item wdt:P31/wdt:P279* wd:Q728937 .
  FILTER NOT EXISTS { ?item wdt:P10271 ?e }
  OPTIONAL { ?item rdfs:label ?label FILTER(LANG(?label) = "en") }
  OPTIONAL { ?item p:P2043/psn:P2043/wikibase:quantityAmount ?m }
  OPTIONAL { ?item wdt:P402 ?osm }
  OPTIONAL { ?item wdt:P3999 ?closed }
}"""
Q_HERITAGE = """SELECT DISTINCT ?item ?label WHERE {
  ?item wdt:P17 wd:Q145 .
  ?item wdt:P31/wdt:P279* wd:Q420962 .
  ?item rdfs:label ?label FILTER(LANG(?label) = "en")
}"""
Q_STATIONS = """SELECT ?item ?crs ?label ?coord WHERE {
  ?item wdt:P4755 ?crs .
  OPTIONAL { ?item rdfs:label ?label FILTER(LANG(?label) = "en") }
  OPTIONAL { ?item wdt:P625 ?coord }
}"""
NAPTAN = "https://naptan.api.dft.gov.uk/v1/access-nodes?atcoAreaCodes=910&dataFormat=csv"
NR_GIS = ("https://raw.githubusercontent.com/openraildata/network-rail-gis/HEAD/network-model/"
          "VectorReferenceLines/NetworkReferenceLines.")


def get(url, data=None, accept=None):
    h = {"User-Agent": USER_AGENT}
    if accept:
        h["Accept"] = accept
    if data is not None:
        h["Content-Type"] = "application/x-www-form-urlencoded"
        data = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=data, headers=h)
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, q in (("wd_elrs.csv", Q_ELR), ("wd_lines.csv", Q_LINES),
                    ("wd_heritage.csv", Q_HERITAGE), ("wd_stations.csv", Q_STATIONS)):
        body = get(WD, {"query": q}, "text/csv")
        (RAW / name).write_bytes(body)
        rows = body.count(b"\n") - 1
        print(f"{name}: {rows} rows")
        time.sleep(1)
    body = get(NAPTAN)
    (RAW / "naptan_910.csv").write_bytes(body)
    print(f"naptan_910.csv: {len(body):,} bytes")
    (RAW / "nr_gis").mkdir(exist_ok=True)
    for ext in ("shp", "shx", "dbf", "prj", "CPG"):
        body = get(NR_GIS + ext)
        (RAW / "nr_gis" / f"NetworkReferenceLines.{ext}").write_bytes(body)
    print("nr_gis/NetworkReferenceLines.*: the ELR reference lines (OGL, archived 2024)")


# ---------------------------------------------------------------- the Channel Tunnel's far half

# Geofabrik keeps a way whole when any of its nodes is inside an extract, and OSM cuts both
# bores of the Channel Tunnel at the boundary, so gb's extract holds France's half of the
# tunnel (ways 143253048, 143253058, 22.7 km each, and two crossover ways in the French undersea
# crossover) and fr's holds the UK's half (143253041, 148429732). Each country's tiles drew the
# other's half as track no line runs on (faint grey) over or under the neighbour's line, and
# gb's High Speed 1 name covered the French half too. `clip_channel` drops, from one country's
# data/proc, the ways lying wholly on the far side of the tunnel's boundary nodes: in the
# Channel box, every node at or east of CHANNEL_WEST_CUT (gb) or at or west of CHANNEL_EAST_CUT
# (fr). Run after every gb or fr extract: `python gb_register.py --clip`, `python
# fr_register.py --clip`. Idempotent.
CHANNEL_BOX = (1.0, 50.85, 2.0, 51.15)      # lon/lat; inside it, east of the cut is France, west the UK
CHANNEL_WEST_CUT = 1.49595                  # the south bore's boundary node, 1.4959543
CHANNEL_EAST_CUT = 1.49627                  # the north bore's, 1.4962632
CHANNEL_FAR = {"gb": lambda x: x >= CHANNEL_WEST_CUT, "fr": lambda x: x <= CHANNEL_EAST_CUT}


def clip_channel(region, log=print):
    import os
    import pickle
    far = CHANNEL_FAR[region]
    d = ROOT / "data" / "proc" / region
    with open(d / "ways.pkl", "rb") as f:
        ways = pickle.load(f)
    with open(d / "stops.pkl", "rb") as f:
        stops = pickle.load(f)
    c = np.load(d / "coords.npz")
    import build_model as bm
    co = bm.Coords(c["id"], c["x"], c["y"])
    w0, s0, e0, n0 = CHANNEL_BOX

    def abroad(lon, lat):
        return w0 <= lon <= e0 and s0 <= lat <= n0 and far(lon)
    gone = []
    for wid, (t, nodes) in ways.items():
        pos, ok = co.many(np.asarray(nodes, dtype=np.int64))
        pos = pos[ok]
        if not pos.size:
            continue
        xs, ys = co.x[pos] / 1e7, co.y[pos] / 1e7
        if all(abroad(x, y) for x, y in zip(xs.tolist(), ys.tolist())):
            gone.append(wid)
            log(f"  {region} clip: way {wid} {t.get('railway')} {t.get('service') or ''} "
                f"{t.get('name') or '(unnamed)'}")
    gone_s = [k for k, (_t, lon, lat) in stops.items() if abroad(lon, lat)]
    for w in gone:
        del ways[w]
    for k in gone_s:
        del stops[k]
    log(f"{region} clip: the Channel Tunnel's far half out of data/proc/{region}: "
        f"{len(gone)} ways, {len(gone_s)} stops")
    if not gone and not gone_s:
        return
    for fn, obj in (("ways.pkl", ways), ("stops.pkl", stops)):
        tmp = d / (fn + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        os.replace(tmp, d / fn)


def names_report():
    import build_model as bm
    ways, _r, _s, cid, cx, cy = bm.load(REGION, print)
    canon, fill, _h, _nr = name_table(ways, bm.Coords(cid, cx, cy), print)
    print(f"{len(canon)} names, {len(fill)} ELR fills")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    elif "--names" in sys.argv:
        names_report()
    elif "--clip" in sys.argv:
        clip_channel("gb")
    else:
        print(__doc__)
