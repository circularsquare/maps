"""T-3225: real networks with no hand work, from noritetsu's local OSM extract (notes/T-3225.md).

    python pipeline/realauto.py extract [id ...]          # the extract cut to each city -> $REALNET_OUT/osm_<id>.json
    python pipeline/realauto.py build [id ...] [--jobs N] # extract, net, geom, corridors, realsave, gamesave; a report per city
    python pipeline/realauto.py report [id ...]           # the reports' table again
    python pipeline/realauto.py ship [id ...]             # realnet.py ship for the cities that passed, and the app's list
    python pipeline/realauto.py drop <id>                 # (a build step) the lines in auto_<id>_drop.json out of the network JSON

The cities are `realauto.json` (a city pack id each, how they were picked in its "what"); with no ids a
command takes them all. Any registry city with a pack can be built: `build <id>` takes ids not in the
list too, and `ship` ships only listed ones with "ship": true, so a city goes on the new game sheet by
an entry there. realnet.py makes each listed city a `CITIES` entry with one `osm` source, so its own
commands (net, geom, corridors, gamesave, ship) run on these as on the hand-made seven.

The extract (`noritetsu/data/proc/<cc>/`, read only): every route relation (train, subway, light
rail, tram, monorail) with two or more stops in the pack's boundary (+1 km), less:
  - `service` high_speed, long_distance, night or tourism, `highspeed=yes`, a `state` of proposed,
    construction or disused;
  - routes whose ways are mostly high-speed track, or mostly preserved or tourist railway;
  - routes mostly outside the pack: under half their stops inside, unless 8 or more are;
  - routes whose stops inside lie over 15 km apart on average (intercity trains).
A station is the stops of one name within 400 m (realnet's `osm` adapter: the extract keeps no
stop areas). Names in Latin letters: OSM's `name` where it is, else `name:en`, `int_name` or
`name:latin`, else Cyrillic transliterated and (China, Hong Kong, Taiwan) Han characters in pinyin
run together; else `name` (Japanese kana and kanji). A line is its route master's `ref` (if Latin
letters or digits), else that name. Each relation's ways, chained, are its shape: where the router
finds no real-network track (trams, light rail), the track follows them. Trains an hour from
`interval` tags, else realnet.OSM_TPH by kind, shared by a route master's patterns each way
(`split_tph`); a suburban train with no route master and no ref every SUBURBAN_INTERVAL minutes.
Run times from the distance at realnet.OSM_KMH; train lengths realnet.OSM_TRAIN_M. Levels, shared
track, terminals and pacing as for the hand-made ones (realcorridors.py, reallevels.py, the
`realsave` bin).

`build` runs each city's steps as subprocesses, `--jobs` cities at once (default 2), each city's log
in $REALNET_OUT/auto_<id>.log. Lines the track model cannot run (a terminal or junction with more
tracks than a game node's 8 ends) go into auto_<id>_drop.json, `drop` takes them out of the network
JSON after `net`, and the city is built once more. It writes $REALNET_OUT/auto_<id>.json: stations,
lines, lines running, names in Latin letters, track km off the real-network graph, high-demand holds
against the free run, the network's price, the save's size, and `pass`: the save was written, 2 or
more lines, at least 90% of them running.
"""
import json
import math
import os
import pickle
import re
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAPS = os.path.dirname(ROOT)
NORI_PROC = os.path.join(MAPS, "noritetsu", "data", "proc")
LIST = os.path.join(HERE, "realauto.json")
OUT = os.path.abspath(os.environ.get("REALNET_OUT", os.path.join(ROOT, "data", "work", "realnet")))
APP_LIST = os.path.join(ROOT, "app", "src", "game", "realNetworks.json")

KINDS = ("train", "subway", "light_rail", "tram", "monorail")
SKIP_SERVICE = {"high_speed", "long_distance", "night", "tourism"}
SKIP_STATE = {"proposed", "construction", "disused", "abandoned"}
INSIDE_SHARE, INSIDE_ENOUGH = 0.5, 8
MAX_SPACING_M = 15000.0
MOSTLY = 0.5
PASS_RUNNING = 0.9
SUBURBAN_INTERVAL = "30"
# regions whose names in Han characters read as Mandarin (not Japan: kanji read otherwise)
HANZI_REGIONS = {"cn", "hk", "tw"}


def listed():
    return json.load(open(LIST, encoding="utf-8"))["cities"]


def registry():
    return {c["id"]: c for c in json.load(open(os.path.join(ROOT, "app", "src", "game", "registry.json"), encoding="utf-8"))["cities"]}


def region_of(pack, reg=None):
    """noritetsu's region code for a pack: its country, lower case (HK and TW are regions of their own)."""
    c = (reg or registry())[pack]
    cc = c["country"].lower()
    if not os.path.isdir(os.path.join(NORI_PROC, cc)):
        raise SystemExit(f"{pack}: no noritetsu extract for {cc}")
    return cc


def latin(s):
    """Every letter of `s` is a Latin one (digits, spaces and marks pass)."""
    letters = [ch for ch in s if ch.isalpha()]
    return bool(letters) and all("LATIN" in unicodedata.name(ch, "") for ch in letters)


CYR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюяіїєґ",
               ["a", "b", "v", "g", "d", "e", "yo", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s", "t", "u", "f", "kh", "ts",
                "ch", "sh", "shch", "", "y", "", "e", "yu", "ya", "i", "yi", "ye", "g"]))


def translit(s):
    """Cyrillic in Latin letters (BGN/PCGN, simplified), other letters as they are."""
    out = []
    for ch in s:
        lo = ch.lower()
        if lo in CYR:
            v = CYR[lo]
            out.append(v[:1].upper() + v[1:] if ch != lo else v)
        else:
            out.append(ch)
    return "".join(out)


def pick_name(t, cc=""):
    n = t.get("name", "")
    if not n or latin(n):
        return n
    for k in ("name:en", "int_name", "name:latin"):
        v = t.get(k, "")
        if v and latin(v):
            return v
    v = translit(n)
    if latin(v):
        return v
    if cc in HANZI_REGIONS:
        # pinyin without tones, run together as OSM's own name:en mostly has it ("Xianxinlu")
        from unidecode import unidecode
        v = "".join(unidecode(n).split()).capitalize()
        if latin(v):
            return v
    return n


def line_name(t, cc):
    """A route's or route master's name for a line with no usable ref: as `pick_name`, but Chinese in
    spaced pinyin (a line's long name run together is unreadable), and OSM's "Label: A => B"
    form shortened to the label when it holds a number ("Tramvay t2"), else to "A - B"."""
    n = pick_name(t, cc)
    own = t.get("name", "")
    if cc in HANZI_REGIONS and own and not latin(own) and not any(latin(t.get(k, "")) for k in ("name:en", "int_name", "name:latin")):
        from unidecode import unidecode
        n = " ".join(unidecode(own).split()).title()
    if ": " in n:
        label, rest = n.split(": ", 1)
        n = label if any(ch.isdigit() for ch in label) else rest
    return re.sub(r"\s*<?=>\s*", " - ", n)


def pick_ref(t):
    r = t.get("ref", "")
    return r if r and (latin(r) or r.replace(" ", "").isdigit()) else ""


def load_region(cc):
    p = os.path.join(NORI_PROC, cc)
    t0 = time.time()
    rels = pickle.load(open(os.path.join(p, "rels.pkl"), "rb"))
    stops = pickle.load(open(os.path.join(p, "stops.pkl"), "rb"))
    ways = pickle.load(open(os.path.join(p, "ways.pkl"), "rb"))
    print(f"{cc}: {len(rels):,} relations, {len(stops):,} stops, {len(ways):,} ways read in {time.time() - t0:.0f} s", flush=True)
    return rels, stops, ways


def coords(cc):
    """noritetsu's node coordinates (for stops missing from stops.pkl): sorted ids, lon, lat."""
    import numpy as np
    z = np.load(os.path.join(NORI_PROC, cc, "coords.npz"))
    ids, x, y = z["id"], z["x"], z["y"]
    o = np.argsort(ids)
    return ids[o], x[o] / 1e7, y[o] / 1e7


def cut(pack, rels, stops, ways, xy_of):
    """The pack's route relations as Overpass JSON (what realnet's `osm` adapter reads), and why the
    rest were dropped."""
    sys.path.insert(0, HERE)
    import realnet
    from shapely import prepared
    from shapely.geometry import Point
    cc = region_of(pack)
    bnd = prepared.prep(realnet.boundary(pack))
    w, s, e, n = realnet.boundary(pack).bounds
    k = math.cos(math.radians((s + n) / 2))
    masters = defaultdict(list)
    for rid, (t, ms) in rels.items():
        if t.get("type") == "route_master":
            for ty, r, _ in ms:
                if ty == "r":
                    masters[r].append(rid)
    why = Counter()
    keep, nodes = [], {}
    for rid, (t, ms) in rels.items():
        if t.get("type") != "route" or t.get("route") not in KINDS:
            continue
        pts = []
        for ty, r, role in ms:
            if ty != "n" or not role.startswith(("stop", "platform")):
                continue
            ll = (stops[r][1], stops[r][2]) if r in stops else xy_of(r)
            if ll is not None:
                pts.append((r, role, ll))
        # stops, else platforms (realnet's adapter does the same)
        st = [q for q in pts if q[1].startswith("stop")] or pts
        inside = [q for q in st if w <= q[2][0] <= e and s <= q[2][1] <= n and bnd.contains(Point(q[2]))]
        if len(inside) < 2:
            continue
        if t.get("service") in SKIP_SERVICE or t.get("highspeed") == "yes":
            why["service tag"] += 1
            continue
        if t.get("state") in SKIP_STATE or any(t.get(f"{x}:route") for x in ("disused", "proposed")) or \
                re.search(r"out of service|disused|abandoned|closed", t.get("name:en", "") + " " + t.get("name", ""), re.I):
            why["not running"] += 1
            continue
        wt = [ways[r][0] for ty, r, role in ms if ty == "w" and r in ways and role in ("", "forward", "backward")]
        if wt and sum(x.get("highspeed") == "yes" for x in wt) > MOSTLY * len(wt):
            why["high-speed track"] += 1
            continue
        if wt and sum(x.get("railway") == "preserved" or x.get("usage") == "tourism" for x in wt) > MOSTLY * len(wt):
            why["heritage track"] += 1
            continue
        if len(inside) < INSIDE_SHARE * len(st) and len(inside) < INSIDE_ENOUGH:
            why["mostly outside"] += 1
            continue
        gap = [math.hypot((a[2][0] - b[2][0]) * k, a[2][1] - b[2][1]) * 111195 for a, b in zip(inside, inside[1:])]
        if sum(gap) / len(gap) > MAX_SPACING_M:
            why["stops far apart"] += 1
            continue
        keep.append(rid)
        for r, _, ll in st:
            nodes[r] = ll
    kept = set(keep)
    shapes = defaultdict(list)
    for rid in keep:
        ms = rels[rid][1]
        q = chain([ways[r][1] for ty, r, role in ms if ty == "w" and r in ways and role in ("", "forward", "backward")], xy_of, k)
        if len(q) >= 2:
            key = f"m{max(masters[rid])}" if masters.get(rid) else f"r{rid}"
            shapes[key].append([[round(a, 6), round(b, 6)] for a, b in q])
    el = []
    for rid in keep:
        t, ms = rels[rid]
        tags = {**t, "name": line_name(t, cc), "ref": pick_ref(t)}
        # a suburban train with no line of its own (no route master, no ref) is one of many
        # destinations out of a terminal (Russia's elektrichki): every half hour, not the
        # commuter default, which would fill the terminal's tracks many times over
        if t.get("route") == "train" and not masters.get(rid) and not t.get("ref") and not t.get("interval"):
            tags["interval"] = SUBURBAN_INTERVAL
        mem = [{"type": "node", "ref": r, "role": role} for ty, r, role in ms if ty == "n" and r in nodes and role.startswith(("stop", "platform"))]
        el.append({"type": "relation", "id": rid, "tags": tags, "members": mem})
    for mid in sorted({m for rid in keep for m in masters.get(rid, [])}):
        t, ms = rels[mid]
        el.append({"type": "relation", "id": mid, "tags": {**t, "name": line_name(t, cc), "ref": pick_ref(t)},
                   "members": [{"type": "relation", "ref": r, "role": ""} for ty, r, _ in ms if ty == "r" and r in kept]})
    for r, (lon, lat) in nodes.items():
        tg = stops[r][0] if r in stops else {}
        el.append({"type": "node", "id": r, "lon": lon, "lat": lat, "tags": {"name": pick_name(tg, cc)}})
    kinds = Counter(rels[r][0].get("route") for r in keep)
    return {"elements": el, "shapes": shapes, "realauto": {"kept": len(keep), "kinds": dict(kinds), "dropped": dict(why)}}


def chain(wlist, xy_of, k):
    """A relation's ways (node id lists, in member order) as one polyline of (lon, lat): each way
    turned to join the one before at a shared node, else at its nearer end across a gap."""
    out = []
    for n, nodes in enumerate(wlist):
        nodes = [int(x) for x in nodes]
        if not out:
            out = nodes
            continue
        if n == 1 and out[-1] not in (nodes[0], nodes[-1]) and out[0] in (nodes[0], nodes[-1]):
            out = out[::-1]  # the first way ran the other way
        if out[-1] == nodes[0]:
            out += nodes[1:]
        elif out[-1] == nodes[-1]:
            out += nodes[::-1][1:]
        else:
            e = xy_of(out[-1])
            a, b = xy_of(nodes[0]), xy_of(nodes[-1])
            if e and a and b:
                da = math.hypot((a[0] - e[0]) * k, a[1] - e[1])
                db = math.hypot((b[0] - e[0]) * k, b[1] - e[1])
                out += nodes if da <= db else nodes[::-1]
            else:
                out += nodes
    pts = [xy_of(x) for x in out]
    return [p for p in pts if p is not None]


def cmd_extract(ids):
    reg = registry()
    by_cc = defaultdict(list)
    for i in ids:
        by_cc[region_of(i, reg)].append(i)
    os.makedirs(OUT, exist_ok=True)
    for cc, ps in by_cc.items():
        rels, stops, ways = load_region(cc)
        lookup = {}

        def xy_of(nid):
            import numpy as np
            if "c" not in lookup:
                lookup["c"] = coords(cc)
            a, x, y = lookup["c"]
            j = int(np.searchsorted(a, nid))
            return (float(x[j]), float(y[j])) if j < len(a) and a[j] == nid else None
        for p in ps:
            d = cut(p, rels, stops, ways, xy_of)
            path = os.path.join(OUT, f"osm_{p}.json")
            json.dump(d, open(path + ".part", "w", encoding="utf-8"), ensure_ascii=False)
            os.replace(path + ".part", path)
            print(f"{p}: {d['realauto']['kept']} route relations {d['realauto']['kinds']}, dropped {d['realauto']['dropped']} -> {path}", flush=True)


def py(*args):
    return [sys.executable, "-u", *args]


def line_key(net, l):
    """A line's key that stays put between builds: its route and its stations' keys in order."""
    return l["route"] + ":" + ",".join(net["stations"][i]["key"] for i in l["stops"])


def dropped(city):
    """Lines `build` left out after they did not run (auto_<city>_drop.json: line keys and names)."""
    p = os.path.join(OUT, f"auto_{city}_drop.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def cmd_drop(city):
    """Take the dropped lines out of the city's network JSON (after `realnet.py net`), its demand
    arrays made again as `net` makes them, stations no line stops at left out."""
    drop = dropped(city)
    p = os.path.join(OUT, f"{city}_real.json")
    net = json.load(open(p, encoding="utf-8"))
    keep = [l for l in net["lines"] if line_key(net, l) not in drop]
    print(f"{city}: {len(net['lines']) - len(keep)} of {len(net['lines'])} lines left out ({len(drop)} listed)")
    used = sorted({i for l in keep for i in l["stops"]})
    new = {i: k for k, i in enumerate(used)}
    net["stations"] = [net["stations"][i] for i in used]
    for l in keep:
        l["stops"] = [new[i] for i in l["stops"]]
    net["lines"] = keep
    d = {"st_xy": [v for s in net["stations"] for v in (s["x"], s["y"])], "line_n": [], "stops": [], "times": [], "tph": [], "cars": []}
    for l in keep:
        d["line_n"].append(len(l["stops"]))
        d["stops"] += l["stops"]
        for run in (l["t0"], l["t1"]):
            for lv in range(3):
                d["times"] += run[lv]
        d["tph"] += l["tph"]
        d["cars"].append(l["cars"])
    net["demand"] = d
    json.dump(net, open(p + ".part", "w"))
    os.replace(p + ".part", p)


def not_running(city):
    """{line key: name} of the lines the realsave log has not running."""
    log = open(os.path.join(OUT, f"auto_{city}.log"), encoding="utf-8", errors="replace").read()
    log = log[log.rfind("== realsave"):]  # this build's, not the one before a drop
    net = json.load(open(os.path.join(OUT, f"{city}_real.json"), encoding="utf-8"))
    out = {}
    for i in re.findall(r"^line (\d+) \(.*\) not running", log, re.M):
        l = net["lines"][int(i)]
        out[line_key(net, l)] = l["name"]
    return out


def build_one(city, exe):
    """The city's steps; when some lines do not run (a terminal with more tracks than a game node
    takes), they are left out (`drop`) and the city is built once more."""
    log = os.path.join(OUT, f"auto_{city}.log")
    env = {**os.environ, "REALNET_OUT": OUT, "OMP_NUM_THREADS": "1", "PYTHONIOENCODING": "utf-8"}
    steps = [("net", py(os.path.join(HERE, "realnet.py"), "net", city)),
             ("drop", py(os.path.abspath(__file__), "drop", city)),
             ("geom", py(os.path.join(HERE, "realnet.py"), "geom", city)),
             ("corridors", py(os.path.join(HERE, "realcorridors.py"), city, "--out", OUT)),
             ("realsave", [str(exe), "--dir", OUT, "--city", city, "--out", OUT, "--shared", "--slow"]),
             ("gamesave", py(os.path.join(HERE, "realnet.py"), "gamesave", city))]
    times = {}
    for attempt in (0, 1):
        with open(log, "a" if attempt else "w", encoding="utf-8") as f:
            for name, cmd in steps[attempt:]:  # the second time from the network JSON already made
                t0 = time.time()
                f.write(f"== {name}: {' '.join(cmd)}\n")
                f.flush()
                r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=MAPS)
                times[name] = round(times.get(name, 0) + time.time() - t0, 1)
                if r.returncode:
                    f.write(f"== {name} failed ({r.returncode})\n")
                    return report(city, times, failed=name)
        bad = not_running(city)
        if not bad or attempt:
            break
        drop = {**dropped(city), **bad}
        json.dump(drop, open(os.path.join(OUT, f"auto_{city}_drop.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"== again without {len(bad)} lines that did not run: {sorted(bad.values())}\n")
    return report(city, times)


def report(city, times=None, failed=None):
    """The city's figures from its files and log -> auto_<city>.json."""
    log = open(os.path.join(OUT, f"auto_{city}.log"), encoding="utf-8", errors="replace").read()
    if times is None and os.path.exists(os.path.join(OUT, f"auto_{city}.json")):  # `report` again: the build's own
        old = json.load(open(os.path.join(OUT, f"auto_{city}.json"), encoding="utf-8"))
        times, failed = old.get("seconds"), old.get("failed")
    r = {"city": city, "failed": failed, "seconds": times or {}}
    p = os.path.join(OUT, f"osm_{city}.json")
    if os.path.exists(p):
        r["extract"] = json.load(open(p, encoding="utf-8"))["realauto"]
    p = os.path.join(OUT, f"{city}_real.json")
    if os.path.exists(p):
        net = json.load(open(p, encoding="utf-8"))
        r["stations"] = len(net["stations"])
        r["lines"] = len(net["lines"])
        r["kinds"] = dict(Counter(l["kind"] for l in net["lines"]))
        r["coloured"] = sum(1 for l in net["lines"] if l.get("colour"))
        r["latin_names"] = round(sum(1 for s in net["stations"] if latin(s["name"])) / max(len(net["stations"]), 1), 3)
    p = os.path.join(OUT, f"{city}_real_corridors.json")
    if os.path.exists(p):
        pcs = json.load(open(p, encoding="utf-8"))["pieces"]
        # pieces the router found no real-network track for (laid on the relation's own ways instead)
        r["off_graph_km"] = round(sum(P["km"] for P in pcs if P.get("off_graph")) / max(sum(P["km"] for P in pcs), 1e-9), 3)
    m = re.findall(r"(\d+) lines running \((\d+) not\)", log)
    if m:
        r["running"], r["not_running"] = int(m[-1][0]), int(m[-1][1])
    m = re.findall(r"hops traced on real track (\d+) .*?from a shape or straight (\d+)", log)
    if m:
        a, b = int(m[-1][0]), int(m[-1][1])
        r["hops_on_track"] = round(a / max(a + b, 1), 3)
    m = re.findall(r"free run (\d+), with the capacity pass's holds (\d+) / (\d+) / (\d+)", log)
    if m:
        free, high = int(m[-1][0]), int(m[-1][1])
        r["held_high"] = round(high / max(free, 1) - 1, 3)
    m = re.findall(r"load \+ derive (\d+) ms", log)
    if m:
        r["derive_s"] = round(int(m[-1]) / 1000, 1)
    m = re.findall(r"crossings (\d+) \((\d+) flat\), total cost US\$(\d+)M", log)
    if m:
        r["crossings"], r["flat"], r["cost_busd"] = int(m[-1][0]), int(m[-1][1]), round(int(m[-1][2]) / 1000, 1)
    p = os.path.join(OUT, f"{city}_real.save")
    r["save_bytes"] = os.path.getsize(p) if os.path.exists(p) and not failed else None
    r["pass"] = bool(r["save_bytes"]) and r.get("lines", 0) >= 2 and r.get("running", 0) >= PASS_RUNNING * r.get("lines", 1)
    json.dump(r, open(os.path.join(OUT, f"auto_{city}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return r


def table(rows):
    """One row per city: stations, lines, lines running, share of station names in Latin letters,
    share of track km off the real-network graph, high-demand holds against the free run, the
    network's price, the save's size, build seconds."""
    print(f"{'city':5} {'pass':5} {'stations':>8} {'lines':>5} {'run':>4} {'latin':>6} {'off':>6} {'held':>6} {'US$B':>6} {'KB':>5}  seconds / failed")
    for r in rows:
        print(f"{r['city']:5} {str(r['pass']):5} {r.get('stations', '-'):>8} {r.get('lines', '-'):>5} {r.get('running', '-'):>4} "
              f"{r.get('latin_names', '-'):>6} {r.get('off_graph_km', '-'):>6} {r.get('held_high', '-'):>6} "
              f"{r.get('cost_busd', '-'):>6} {round((r.get('save_bytes') or 0) / 1e3):>5}  {sum((r['seconds'] or {}).values()):.0f} {r['failed'] or ''}")


def cmd_build(ids, jobs):
    sys.path.insert(0, HERE)
    import simbin
    exe = os.environ.get("REALSAVE_EXE") or simbin.exe(ROOT, "realsave")
    need = [i for i in ids if not os.path.exists(os.path.join(OUT, f"osm_{i}.json"))]
    if need:
        cmd_extract(need)
    with ThreadPoolExecutor(jobs) as ex:
        rows = list(ex.map(lambda c: build_one(c, exe), ids))
    table(rows)


def cmd_ship(ids):
    sys.path.insert(0, HERE)
    ok = {c["id"] for c in listed() if c.get("ship")}
    go = []
    for i in ids:
        p = os.path.join(OUT, f"auto_{i}.json")
        r = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        if i not in ok or not r.get("pass"):
            print(f"{i}: not shipped ({'not listed to ship' if i not in ok else 'did not pass'})")
            continue
        go.append(i)
    if go:
        subprocess.run(py(os.path.join(HERE, "realnet.py"), "ship", *go), env={**os.environ, "REALNET_OUT": OUT}, check=True)
    write_app_list()


def write_app_list():
    """The app's list of cities with a real network: every realnet.SHIPPED city whose save is in
    data/packs/real."""
    import realnet
    have = [c for c in realnet.SHIPPED if os.path.exists(os.path.join(ROOT, "data", "packs", "real", f"{c}.save"))]
    body = json.dumps(have) + "\n"
    if not os.path.exists(APP_LIST) or open(APP_LIST, encoding="utf-8").read() != body:
        open(APP_LIST + ".part", "w", encoding="utf-8").write(body)
        os.replace(APP_LIST + ".part", APP_LIST)
    print(f"{len(have)} real networks -> {APP_LIST}")


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0] if a else "report"
    jobs = 2
    if "--jobs" in a:
        i = a.index("--jobs")
        jobs = int(a[i + 1])
        del a[i:i + 2]
    ids = a[1:] or [c["id"] for c in listed()]
    if cmd == "extract":
        cmd_extract(ids)
    elif cmd == "build":
        cmd_build(ids, jobs)
    elif cmd == "report":
        table([report(i) for i in ids if os.path.exists(os.path.join(OUT, f"auto_{i}.log"))])
    elif cmd == "drop":
        cmd_drop(ids[0])
    elif cmd == "ship":
        cmd_ship(ids)
    else:
        sys.exit(__doc__)
