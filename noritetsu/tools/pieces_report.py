"""Which lines are in disconnected pieces, and a first guess at why.

    python tools/pieces_report.py                  # summary per country and kind
    python tools/pieces_report.py --list [cc ...]   # plus every line in pieces, worst first

Reads dist/data only. The list Anita asked for on 2026-10-05 ("make a list and go one by one
and try to find why they are multiple segments"); handoff_notes/lines_in_pieces.md is the
work through it. A line's sections are joined across countries by line id, as the app's
mergeRegion does, so a line shipped in two countries that meets at a shared border point is one
piece. Kinds: "register" (a national register's line), "osm" (an OSM line: metros, trams,
operating patterns), "named" (a named train: service=True). Closed sections count: a line
greyed in its middle is still one line.

For each line in pieces: the km of every piece, the km outside the biggest, and how far each
smaller piece lies from the biggest (the nearest two stations of the two, crow-fly). A small
gap is usually a hole in a register or a dropped section; a gap of tens of km is more often two
lines under one name or a named train whose middle lies in a country not built. GUESSES
below names the four first guesses; each still needs looking at.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist" / "data"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def km_between(p, q):
    return math.hypot((p["x"] - q["x"]) * math.cos(math.radians((p["y"] + q["y"]) / 2)),
                      p["y"] - q["y"]) * 111.32


def load():
    lines, stations = defaultdict(list), {}
    for d in sorted(DIST.iterdir()):
        if not (d / "lines.json").exists():
            continue
        cc = d.name
        for l in json.loads((d / "lines.json").read_text(encoding="utf-8"))["lines"]:
            lines[l["id"]].append((cc, l))
        for sid, s in json.loads((d / "stations.json").read_text(encoding="utf-8"))["stations"].items():
            stations.setdefault(sid, s)
    return lines, stations


def kind(l):
    if l.get("service"):
        return "named"
    return "osm" if l.get("src", "osm") == "osm" else "register"


GUESSES = {
    "crossing": "pieces in different countries: a crossing with no shared border point, or a "
                "country between them not built",
    "near": "under 1 km apart: a junction or station that does not quite meet, or a short "
            "dropped section",
    "hole": "1-25 km apart: a stretch left out that trains probably run through (a register "
            "hole, a dropped section, track missing from OSM)",
    "far": "over 25 km apart: separate passenger pieces of one line (a closed middle), or two "
           "lines under one name",
}


def guess(pieces, ccs):
    """A first guess at why a line is in pieces, from its smaller pieces' gaps and countries."""
    if any(c != ccs[0] for c in ccs[1:]):
        return "crossing"
    worst = max(gap for _km, gap, _n in pieces[1:])
    return "near" if worst < 1 else "hole" if worst <= 25 else "far"


def analyse(parts, stations):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    secs = []
    for cc, l in parts:
        for a, b, km, *_ in l["sections"]:
            parent[find(a)] = find(b)
            secs.append((cc, a, b, km))
    comp_km, comp_nodes, comp_cc = defaultdict(float), defaultdict(set), defaultdict(set)
    for cc, a, b, km in secs:
        r = find(a)
        comp_km[r] += km
        comp_nodes[r] |= {a, b}
        comp_cc[r].add(cc)
    pieces = sorted(comp_km, key=lambda r: -comp_km[r])
    out = {"pieces": [], "guess": ""}
    if len(pieces) > 1:
        big = [stations[s] for s in comp_nodes[pieces[0]] if s in stations]
        for r in pieces:
            mine = [stations[s] for s in comp_nodes[r] if s in stations]
            gap = (0.0 if r == pieces[0] or not big or not mine else
                   min(km_between(p, q) for p in mine for q in big))
            names = sorted({s.get("n", "?") for s in mine if not s.get("j")})
            out["pieces"].append((comp_km[r], gap, names))
        out["guess"] = guess(out["pieces"], [comp_cc[r] for r in pieces])
    return out


def main():
    args = sys.argv[1:]
    show_list = "--list" in args
    only = {a for a in args if not a.startswith("-")}
    lines, stations = load()
    summary = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    by_guess = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    rows = []
    for lid, parts in lines.items():
        ccs = sorted({cc for cc, _ in parts})
        if only and not (set(ccs) & only):
            continue
        l0 = parts[0][1]
        k = kind(l0)
        res = analyse(parts, stations)
        home = "+".join(ccs)
        if res["pieces"]:
            outside = sum(p[0] for p in res["pieces"][1:])
            summary[home][k][0] += 1
            summary[home][k][1] += outside
            by_guess[k][res["guess"]][0] += 1
            by_guess[k][res["guess"]][1] += outside
            rows.append((outside, home, k, lid, l0["name"], res["pieces"], res["guess"]))
    print("country  kind      in pieces  km outside biggest")
    tot = defaultdict(lambda: [0, 0.0])
    for home in sorted(summary):
        for k in ("register", "osm", "named"):
            v = summary[home].get(k)
            if not v or not v[0]:
                continue
            print(f"{home:8} {k:9} {v[0]:9}  {v[1]:17,.1f}")
            for i in range(2):
                tot[k][i] += v[i]
    for k, v in tot.items():
        print(f"{'ALL':8} {k:9} {v[0]:9}  {v[1]:17,.1f}")
    print("\nby first guess (--list gives each line's)")
    for k in ("register", "osm", "named"):
        for g in ("crossing", "near", "hole", "far"):
            v = by_guess[k].get(g)
            if v:
                print(f"  {k:9} {g:9} {v[0]:5} lines  {v[1]:10,.1f} km   {GUESSES[g]}"[:150])
    if show_list:
        print("\nLINES IN PIECES (km outside the biggest piece, worst first)")
        for outside, home, k, lid, name, pieces, g in sorted(rows, key=lambda r: -r[0]):
            print(f"\n{home:6} {k:8} {g:8} {lid}  {name}  ({len(pieces)} pieces, "
                  f"{outside:.1f} km outside)")
            for km, gap, names in pieces:
                ends = ", ".join(names[:4]) + (f" +{len(names) - 4}" if len(names) > 4 else "")
                print(f"    {km:8.1f} km  {'' if not gap else f'{gap:6.1f} km off  '}{ends}")

if __name__ == "__main__":
    main()
