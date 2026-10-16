"""T-509: calibration targets, runs and the objective, per city.

    python pipeline/calib.py targets nyc                 # -> data/work/calib/nyc/targets.json
    python pipeline/calib.py run nyc [--tag t] [--set k=v,..] [--params p.json] [--rounds 3] [--pack p.json]
    python pipeline/calib.py score <run.json> [--city nyc] [--top 15]

`run` runs the `calib` bin (sim/src/bin/calib.rs; built with `cargo build --release --features
demand --bin calib`, target dir data/work/target) on the city's real network (realnet.py net
<city>) with the parameter set, then scores it: data/work/calib/<city>/run_<tag>.json and
score_<tag>.json. One run is a full weekday, commutes and everyday purposes, 3 crowding rounds.

The objective (notes/T-509.md section 4), each part a mean so every city weighs the same:
- stations: mean squared log ratio of entries (first boardings, all purposes) over the demand
  stations with at least `station_min` measured entries a day; hubs are reported apart (a
  station of an operator the counts do not cover within `hub_m`, where riders change from trains
  the counts miss, or one listed in `hubs`: bus and ferry terminals);
- operators: mean squared log ratio of trips using each operator (boardings less changes within
  it, as operators count; T-171); per-line totals where a city has them go in the same list;
- areas: rail share of commuters by home area, squared difference of logits weighted by
  commuters.
The total is the sum of the three. Fixed-value penalties come with the fit (stage 6).

Targets file (any city): {"city", "net", "areas_file", "station_min", "stations": [{"target",
"name", "entries"}], "hubs": [target ids], "operators": {operator: trips}, "operators_info":
{...}, "areas": [{"name", "rail_share", "commuters"}], "sources": {...}}. Station `target` ids
are the network's station `target` (realnet.py's group tables); operator names its lines' `operator`.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import json
import subprocess
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "data", "work")
CALIB = os.path.join(WORK, "calib")
BIN = os.path.join(WORK, "target", "release", "calib.exe" if os.name == "nt" else "calib")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------- targets

def targets_nyc():
    """New York from realnet.py's targets.json (MTA subway OD by complex, Oct 2024 Wednesdays;
    operator weekday totals; ACS 2019-23 B08301 by county) and its cell_county.u8."""
    import realnet as rn
    tg = json.load(open(os.path.join(rn.WORK, "targets.json")))
    st = []
    for k, v in tg["subway_od"]["stations"].items():
        st.append({"target": v["complex"], "name": "", "entries": float(sum(v["entries"])), "borough": v["borough"]})
    net = json.load(open(rn.CITIES["nyc"]["out"]))
    names = {s["target"]: s["name"] for s in net["stations"] if s.get("target")}
    for s in st:
        s["name"] = names.get(s["target"], "?")
    # bus and ferry terminals: Port Authority is in the Times Sq complex, the George Washington
    # Bridge bus station at 175 St (A), the Staten Island Ferry at Whitehall St-South Ferry
    hubs = [s["target"] for s in st if s["name"].startswith(("Times Sq", "Whitehall St")) or s["name"] == "175 St"]
    ops = tg["operators"]
    areas = []
    for f in rn.COUNTY_ORDER:
        a = tg["acs"][f]
        comm = a["001"] - a["021"]
        areas.append({"name": rn.COUNTIES[f], "rail_share": (a["012"] + a["013"] + a["014"]) / comm, "commuters": comm})
    return {
        "city": "nyc", "net": rn.CITIES["nyc"]["out"], "areas_file": os.path.join(rn.WORK, "cell_county.u8"),
        "station_min": 2000.0, "hub_m": 300.0, "stations": st, "hubs": hubs,
        "operators": {"subway": ops["subway"] + ops["sir"], "path": ops["path"], "lirr": ops["lirr"], "mnr": ops["mnr"]},
        "operators_info": {"njt": ops["njt_rough"]},
        "areas": areas,
        "sources": {"stations": tg["sources"]["subway_od"], "operators": tg["sources"]["operators"] + " (NJ Transit rough, not scored)",
                    "areas": tg["sources"]["acs"] + "; rail = subway, commuter rail and light rail over workers less those working from home"},
    }


TARGETS = {"nyc": targets_nyc}
TARGETS["seo"] = lambda: __import__("calib_seo").targets()
TARGETS["sdj"] = lambda: __import__("calib_sdj").targets()
TARGETS["tor"] = lambda: __import__("calib_tor").targets()
TARGETS["tyo"] = lambda: __import__("calib_tyo").targets()
TARGETS["lon"] = lambda: __import__("calib_lon").targets()
TARGETS["par"] = lambda: __import__("calib_par").targets()


def cmd_targets():
    city = sys.argv[2]
    t = TARGETS[city]()
    p = os.path.join(CALIB, city, "targets.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(t, open(p, "w"), indent=0)
    print(f"{len(t['stations'])} stations ({len(t['hubs'])} hubs listed), {len(t['operators'])} operators, {len(t['areas'])} areas -> {p}")


# ---------------------------------------------------------------- score

def score(run, tg, top=15, say=print):
    """The objective and its parts; prints the biggest misses with `say`."""
    net = json.load(open(tg["net"]))
    smap = run["st_map"]
    ent = np.array(run["entries"]["commute"]) + (np.array(run["entries"]["everyday"]) if run["entries"]["everyday"] else 0)
    # demand station -> the targets in it, and whether it holds a station the counts miss
    tid_d = defaultdict(set)
    uncounted = set()
    tset = {s["target"] for s in tg["stations"]}
    xy = np.array([[s["x"], s["y"]] for s in net["stations"]])
    free = np.array([s.get("target") not in tset for s in net["stations"]])
    hub_m = tg.get("hub_m", 300.0)
    for i, s in enumerate(net["stations"]):
        d = smap[i]
        if s.get("target") in tset:
            tid_d[d].add(s["target"])
            # a station the counts miss within a transfer walk: riders change here from trains
            # whose boardings are no entries in the counts
            if (free & (np.hypot(*(xy - xy[i]).T) < hub_m)).any():
                uncounted.add(d)
        else:
            uncounted.add(d)
    meas = {s["target"]: s for s in tg["stations"]}
    hubs = set(tg.get("hubs", []))
    rows = []
    for d, ts in tid_d.items():
        t = sum(meas[x]["entries"] for x in ts)
        name = " / ".join(meas[x]["name"] for x in sorted(ts))[:40]
        hub = d in uncounted or bool(ts & hubs)
        rows.append((d, name, t, float(ent[d]), hub))
    rows = [r for r in rows if r[2] >= tg.get("station_min", 0)]
    lr = np.array([np.log(max(r[3], 1.0) / r[2]) for r in rows])
    hub = np.array([r[4] for r in rows])
    parts = {}
    parts["stations"] = float(np.mean(lr[~hub] ** 2))
    st_level = float(np.mean(lr[~hub]))
    # operators
    ops = run["operators"]
    olr = {k: float(np.log(max(ops.get(k, 0.0), 1.0) / v)) for k, v in tg["operators"].items()}
    parts["operators"] = float(np.mean([v ** 2 for v in olr.values()])) if olr else 0.0  # Paris has no line counts
    # areas (none where no census mode table is on disk: Seoul)
    ra = run.get("areas") or {"rail": [], "trips": []}
    ar = np.array(ra["rail"], float)
    at = np.array(ra["trips"], float)
    w = np.array([a["commuters"] for a in tg["areas"]], float)
    ws = max(w.sum(), 1e-9)
    meas_share = np.array([a["rail_share"] for a in tg["areas"]], float)
    mod_share = ar / np.maximum(at, 1e-9)
    dl = logit(mod_share) - logit(meas_share)
    parts["areas"] = float((w * dl ** 2).sum() / ws)
    total = sum(parts.values())
    out = {"total": total, "parts": parts, "station_level": st_level, "station_spread": float(np.std(lr[~hub])),
           "n_stations": int((~hub).sum()), "n_hubs": int(hub.sum()),
           "hubs_log_ratio": float(np.mean(lr[hub])) if hub.any() else None,
           "operators": {k: {"model": ops.get(k, 0.0), "measured": v, "log_ratio": olr[k]} for k, v in tg["operators"].items()},
           "operators_info": {k: {"model": ops.get(k, 0.0), "measured": v} for k, v in tg.get("operators_info", {}).items()},
           "rail_share_model": float((mod_share * w).sum() / ws), "rail_share_measured": float((meas_share * w).sum() / ws),
           "day": run["day"], "ms": run["ms"], "set": run.get("set", {})}
    say(f"score {total:.3f} = stations {parts['stations']:.3f} + operators {parts['operators']:.3f} + areas {parts['areas']:.3f}")
    say(f"stations: {out['n_stations']} scored (>= {tg.get('station_min', 0):,.0f} entries), mean log ratio {st_level:+.2f} "
        f"(x{np.exp(st_level):.2f}), spread {out['station_spread']:.2f}; {out['n_hubs']} hubs apart, mean log ratio "
        f"{out['hubs_log_ratio'] if out['hubs_log_ratio'] is None else round(out['hubs_log_ratio'], 2)}")
    order = sorted(range(len(rows)), key=lambda i: -abs(rows[i][3] - rows[i][2]))
    say(f"\nBiggest station misses (entries a day, all purposes; * = hub):")
    for i in order[:top]:
        d, name, t, m, h = rows[i]
        say(f"  {'*' if h else ' '} {name:40s} measured {t:9,.0f}  model {m:9,.0f}  x{m / t:.2f}")
    say("\nOperators (trips using each, all purposes):")
    for k, v in sorted(out["operators"].items(), key=lambda kv: -abs(kv[1]["log_ratio"])):
        say(f"  {k:10s} measured {v['measured']:10,.0f}  model {v['model']:10,.0f}  x{np.exp(v['log_ratio']):.2f}")
    for k, v in out["operators_info"].items():
        say(f"  {k:10s} rough     {v['measured']:10,.0f}  model {v['model']:10,.0f}  (not scored)")
    say(f"\nRail share of commuters by home area (measured {out['rail_share_measured']:.1%}, model {out['rail_share_model']:.1%}); biggest misses:")
    contrib = w * dl ** 2 / ws
    for i in np.argsort(-contrib)[:top]:
        a = tg["areas"][i]
        say(f"  {a['name']:14s} measured {meas_share[i]:6.1%}  model {mod_share[i]:6.1%}  share of the part {contrib[i] / max(parts['areas'], 1e-12):.0%}")
    out["station_rows"] = [{"name": r[1], "measured": r[2], "model": r[3], "hub": r[4]} for r in rows]
    out["area_rows"] = [{"name": a["name"], "measured": float(meas_share[i]), "model": float(mod_share[i]), "commuters": a["commuters"]} for i, a in enumerate(tg["areas"])]
    say(f"\nday: commutes {run['day']['commute'][0]:,.0f} trips, rail {run['day']['commute'][1]:,.0f}; everyday {run['day']['everyday'][0]:,.0f}, "
        f"rail {run['day']['everyday'][1]:,.0f}; run {run['ms']['all'] / 1e3:.1f} s (solve {run['ms']['solve'] / 1e3:.1f} s)")
    return out


def opt(args, k, default=None):
    return args[args.index(k) + 1] if k in args else default


def cmd_run():
    args = sys.argv[2:]
    city = args[0]
    tag = opt(args, "--tag", "base")
    tg = json.load(open(os.path.join(CALIB, city, "targets.json")))
    d = os.path.join(CALIB, city)
    out = os.path.join(d, f"run_{tag}.json")
    cmd = [BIN, "--net", tg["net"], "--out", out, "--rounds", opt(args, "--rounds", "3")]
    # a pack other than data/packs/<net's pack>.json: --pack, else the targets' "pack" (a copy
    # kept fixed while packs are being rebuilt)
    if opt(args, "--pack") or tg.get("pack"):
        cmd += ["--pack", opt(args, "--pack") or tg["pack"]]
    if tg.get("areas_file"):
        cmd += ["--areas", tg["areas_file"]]
    for k in ("--set", "--params", "--fares"):
        if k in args:
            cmd += [k, opt(args, k)]
    if "--no-fares" in args:
        cmd.append("--no-fares")
    with open(os.path.join(d, f"run_{tag}.log"), "w") as log:
        subprocess.run(cmd, check=True, stderr=log, stdout=log)
    s = score(json.load(open(out)), tg)
    json.dump(s, open(os.path.join(d, f"score_{tag}.json"), "w"))


def cmd_score():
    args = sys.argv[2:]
    run = json.load(open(args[0]))
    city = opt(args, "--city") or json.load(open(run["net"])).get("city", "nyc")
    tg = json.load(open(os.path.join(CALIB, city, "targets.json")))
    score(run, tg, top=int(opt(args, "--top", "15")))


if __name__ == "__main__":
    {"targets": cmd_targets, "run": cmd_run, "score": cmd_score}[sys.argv[1]]()
