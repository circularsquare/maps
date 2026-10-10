"""The rest of Asia: Cambodia, Laos, the Philippines, Myanmar, Mongolia and Nepal. Each country's
passenger lines are a hand-written station list (asia_lines.py), traced over OSM track by
rinf.py through lk_register.py's engine (nafrica_register's recipe; nothing of lk_register's
changes, it is extended in this process only).

    python tools/slot.py 2 -- python extract.py --region kh --pbf data/raw/cambodia-latest.osm.pbf --station-areas
    python asia_register.py --clip kh             # after every extract (drops the neighbours' track)
    python build_model.py --region kh --register asia_register:data/raw/rinf/kh
    python asia_register.py --dry kh              # the conversion alone, with its log
    python asia_register.py --stations kh         # every OSM station, to write lists
    python asia_register.py --trace kh "Phnom Penh" "Takeo"

WHY A LIST. None of the six publishes a line register with geometry, and OSM names too little
of their track for the named-track recipe everywhere (kh_ ... np_sources.md). Each line is its
stations in order (ends, junctions and enough between to keep the trace on the right track);
`osm_stops: "all"` makes every OSM station on a traced section a stop unless the line is
`listed_only`. Where a published table gives km posts (Laos) they are the register's
chainage; elsewhere the lengths are our traces and check_model.REGISTER the outside check.

THE LINE UNIT is the operator's line, cut where part runs and part does not (rinf.py greys a
whole line or none).

ACROSS A BORDER. A line ending at a border point (asia_lines.BORDERS, the same ids as
borders.EXTRA) is drawn to the point. A line in asia_lines.JOIN continues a built neighbour's
register line and takes that line's id and name (read from dist/data/<nb>/lines.json), as
cn_register's border pieces do, so a ride over the border is one ride on one line.
"""
import json
from pathlib import Path

import lk_register as eng
from asia_lines import BORDERS, JOIN, LINES, NOT_SERVICE, PATH_CHECKS

ROOT = Path(__file__).resolve().parent

CONF = {   # cc: (langs, iso3, km posts are the register's chainage)
    "kh": (["km", "en"], "KHM", False),
    "la": (["lo", "en"], "LAO", True),
    "ph": (["en", "tl"], "PHL", False),
    "mm": (["my", "en"], "MMR", False),
    "mn": (["mn", "en"], "MNG", False),
    "np": (["ne", "en"], "NPL", False),
}
for _cc in LINES:
    _langs, _iso3, _chain = CONF[_cc]
    eng.register_country(_cc, LINES, BORDERS, NOT_SERVICE, _langs, _iso3, _chain,
                         PATH_CHECKS.get(_cc, ()))

split_pieces = eng.split_pieces


def country_conf(cc):
    return eng.country_conf(cc)


def join_neighbours(cc, lines, stations, geoms, log):
    """asia_lines.JOIN: a line continuing a neighbour's register line over the border takes the
    neighbour line's id and name; the operator stays this list's."""
    import rinf
    for lid, (nb, pid) in JOIN.get(cc, {}).items():
        mine = [l for l in lines if any(x.split("#")[0] == lid for x in l.get("rinf_ids", ()))]
        try:
            other = next((l for l in json.loads(
                (ROOT / "dist" / "data" / nb / "lines.json").read_text("utf-8"))["lines"]
                if l.get("src", "osm") != "osm" and any(pid in s[:2] for s in l["sections"])),
                None)
        except (OSError, ValueError, KeyError):
            other = None
        if not mine or other is None:
            log(f"  {cc.upper()}: join {lid} -> {nb}: "
                f"{'no line here' if not mine else 'no ' + nb + ' line at ' + pid}; kept apart")
            continue
        l = mine[0]
        old = l["id"]
        l["id"], l["name"], l["name_en"] = other["id"], other["name"], other.get("name_en", "")
        if old in geoms:
            geoms[l["id"]] = geoms.pop(old)
        if old in rinf.GROUPS:
            rinf.GROUPS[l["id"]] = rinf.GROUPS.pop(old)
        for s in stations.values():
            if old in s.get("lines", ()):
                s["lines"].discard(old)
                s["lines"].add(l["id"])
        log(f"  {cc.upper()}: {lid} takes {nb}'s line id {other['id']} ({other['name']})")


def build(path, log):
    cc = Path(path).name
    lines, stations, geoms = eng.build(path, log)
    join_neighbours(cc, lines, stations, geoms, log)
    return lines, stations, geoms


def drop_stops(cc, log=print):
    """asia_lines.DROP_STOPS: OSM stop nodes that are no station of a line here (the
    Ulaanbaatar railbus's halts on the main line, a second node of one station), dropped from
    data/proc/<cc>/stops.pkl so `osm_stops: "all"` does not make them stops. Run after --clip
    (--clip runs it)."""
    import os
    import pickle
    from asia_lines import DROP_STOPS
    drop = DROP_STOPS.get(cc, {})
    if not drop:
        return
    f = ROOT / "data" / "proc" / cc / "stops.pkl"
    stops = pickle.load(open(f, "rb"))
    gone = [k for k in drop if k in stops]
    for k in gone:
        log(f"  stop dropped: {k} {stops[k][0].get('name')} ({drop[k]})")
        del stops[k]
    tmp = f.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(stops, fh, protocol=4)
    os.replace(tmp, f)
    log(f"{cc.upper()}: {len(gone)} of {len(drop)} listed stops dropped")


if __name__ == "__main__":
    import sys
    if "--clip" in sys.argv:
        _cc = sys.argv[sys.argv.index("--clip") + 1]
        eng.clip(_cc)
        drop_stops(_cc)
        sys.argv[sys.argv.index("--clip"):sys.argv.index("--clip") + 2] = []
        if len(sys.argv) == 1:
            sys.exit(0)
    if "--join" in sys.argv:
        # mideast_register.join: loose track ends within 40 m of other track joined (Nepal's
        # line is mapped in pieces whose end nodes sit on one spot without being shared)
        from mideast_register import join
        _cc = sys.argv[sys.argv.index("--join") + 1]
        join(_cc)
        sys.argv[sys.argv.index("--join"):sys.argv.index("--join") + 2] = []
        if len(sys.argv) == 1:
            sys.exit(0)
    eng.main()
