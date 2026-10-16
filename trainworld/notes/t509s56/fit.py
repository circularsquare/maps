"""T-509 stage 6: coordinate search over native calib runs, every fitted city at once.

    python fit.py eval <tag> [k=v ...]        # one parameter set, all cities, prints the table
    python fit.py search                       # coordinate search from today's values
    python fit.py loo                          # leave-one-city-out on the cached evaluations

Cache: cache.jsonl, one line per (city, parameter set): the score's parts and a few totals.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

# outputs (cache, logs, best.json) live with the check outputs, not beside the scripts
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "artifacts", "trainworld", "t509s56")
TW = r"C:\Users\anita\projects\maps\trainworld"
PY = r"C:\Users\anita\projects\maps\venv\Scripts\python.exe"
CALIB = os.path.join(TW, "data", "work", "calib")
CACHE = os.path.join(HERE, "cache.jsonl")
FIT = ["nyc", "lon", "par", "seo", "tyo", "sdj"]
HOLD = ["tor"]
ALL = FIT + HOLD
WORKERS = int(os.environ.get("FIT_WORKERS", "7"))

# today's values, a plausible range (the penalty's unit) and the search steps
PARAMS = {
    "asc_rail": (0.0, 1.0),
    "everyday_asc_rail": (0.0, 1.0),
    "everyday_walk_d0_km": (1.7, 0.5),
    "w_walk": (1.5, 0.5),
    "other_asc": (10.0, 10.0),
}
PENALTY = 0.05  # per parameter, per range moved, squared, summed over the fitted cities' mean


def key(city, p):
    return city + "|" + ",".join(f"{k}={p[k]:g}" for k in sorted(p))


def lg(p):
    import math
    p = min(max(p, 0.01), 0.99)
    return math.log(p / (1 - p))


def par_upper(r):
    """Paris's area part scored one-sided (its census share includes the bus), for records
    scored before calib.py did that."""
    if r["city"] != "par" or r.get("v", 1) >= 2:
        return r
    sc = json.load(open(os.path.join(CALIB, "par", f"score_{r['tag']}.json")))
    rows = sc["area_rows"]
    w = sum(a["commuters"] for a in rows)
    new = sum(a["commuters"] * max(lg(a["model"]) - lg(a["measured"]), 0.0) ** 2 for a in rows) / w
    r["total"] += new - r["parts"]["areas"]
    r["parts"]["areas"] = new
    r["v"] = 2
    return r


def load_cache():
    c = {}
    if os.path.exists(CACHE):
        for line in open(CACHE):
            try:
                r = par_upper(json.loads(line))
            except (ValueError, KeyError, OSError):
                continue
            c[r["key"]] = r
    return c


def run_one(city, p, tag):
    s = ",".join(f"{k}={v:g}" for k, v in sorted(p.items()))
    cmd = [PY, os.path.join(TW, "pipeline", "calib.py"), "run", city, "--tag", tag]
    if s:
        cmd += ["--set", s]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if tag.startswith("f_"):
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    else:
        with open(os.path.join(CALIB, city, f"score_{tag}.txt"), "w", encoding="utf-8") as out:
            subprocess.run(cmd, check=True, stdout=out, stderr=subprocess.STDOUT, env=env)
    sc = json.load(open(os.path.join(CALIB, city, f"score_{tag}.json")))
    d = sc["day"]
    r = {"key": key(city, p), "city": city, "p": p, "total": sc["total"], "parts": sc["parts"],
         "station_level": sc["station_level"], "rail_share_model": sc["rail_share_model"],
         "rail_share_measured": sc["rail_share_measured"],
         "commute": d["commute"], "everyday": d["everyday"], "kinds": d.get("everyday_kinds"),
         "walks_all": d.get("everyday_walks_all"), "ev_rows": sc.get("everyday_rows"),
         "operators": {k: v["log_ratio"] for k, v in sc["operators"].items()}, "tag": tag, "v": 2}
    if tag.startswith("f_"):
        for f in (f"run_{tag}.json", f"run_{tag}.log"):
            try:
                os.remove(os.path.join(CALIB, city, f))
            except OSError:
                pass
    return r


def evaluate(points, cities=ALL, prefix="f"):
    """points: list of parameter dicts (full). Runs what the cache lacks, in parallel."""
    cache = load_cache()
    jobs = []
    for i, p in enumerate(points):
        for c in cities:
            if key(c, p) not in cache:
                jobs.append((c, p))
    # longest first (Tokyo) so the pool stays busy
    order = {"tyo": 0, "lon": 1, "par": 2, "nyc": 3, "seo": 4, "tor": 5, "sdj": 6}
    jobs.sort(key=lambda j: order[j[0]])
    n0 = len(load_cache())

    def go(ij):
        i, (c, p) = ij
        tag = prefix if prefix != "f" else f"f_{os.getpid()}_{n0 + i}"
        return run_one(c, p, tag)

    if jobs:
        with ThreadPoolExecutor(WORKERS) as ex:
            for r in ex.map(go, enumerate(jobs)):
                with open(CACHE, "a") as f:
                    f.write(json.dumps(r) + "\n")
    cache = load_cache()
    return [{c: cache[key(c, p)] for c in cities} for p in points]


def penalty(p):
    return PENALTY * sum(((p[k] - v0) / rng) ** 2 for k, (v0, rng) in PARAMS.items())


def objective(res, p, fit=FIT):
    return sum(res[c]["total"] for c in fit) + len(fit) * penalty(p)


def today():
    return {k: v[0] for k, v in PARAMS.items()}


def table(res, say=print):
    say(f"{'city':5s} {'score':>6s} {'stat':>6s} {'oper':>6s} {'area':>6s} {'every':>6s} {'level':>6s} {'c.rail':>7s} {'meas':>6s} {'ev.rail':>7s}")
    for c, r in res.items():
        pa = r["parts"]
        ev = r["everyday"]
        say(f"{c:5s} {r['total']:6.3f} {pa['stations']:6.3f} {pa['operators']:6.3f} {pa['areas']:6.3f} {pa.get('everyday', 0):6.3f} "
            f"x{2.718281828 ** r['station_level']:5.2f} {r['rail_share_model']:7.1%} {r['rail_share_measured']:6.1%} {ev[1] / max(ev[0], 1):7.1%}")


def parse(args):
    p = today()
    for a in args:
        k, v = a.split("=")
        p[k] = float(v)
    return p


def cmd_eval():
    tag = sys.argv[2]
    p = parse(sys.argv[3:])
    res = evaluate([p], prefix=tag)[0]
    table(res)
    print(f"objective (fitted six, with penalty) {objective(res, p):.3f}; penalty {len(FIT) * penalty(p):.3f}")


STEPS = {"everyday_asc_rail": [-1.5, -1.0, -0.5, -0.25, 0.25], "asc_rail": [-0.5, -0.25, 0.25, 0.5],
         "everyday_walk_d0_km": [-0.3, 0.3, 0.6, 0.9], "w_walk": [-0.25, 0.25, 0.5],
         "other_asc": [-5.0, 5.0, 10.0, 20.0]}


def cmd_search():
    fit = FIT if len(sys.argv) < 3 else sys.argv[2].split(",")
    order = list(STEPS)
    p = today()
    log = open(os.path.join(HERE, "search.log"), "a")
    say = lambda s: (print(s), log.write(s + "\n"), log.flush())
    best = objective(evaluate([p])[0], p, fit)
    say(f"start {p} -> {best:.3f}")
    for sweep in range(2):
        scale = 1.0 if sweep == 0 else 0.5
        for k in order:
            cands = []
            for s in STEPS[k]:
                q = dict(p)
                q[k] = round(p[k] + s * scale, 4)
                cands.append(q)
            res = evaluate(cands)
            for q, r in zip(cands, res):
                o = objective(r, q, fit)
                say(f"  sweep {sweep} {k}={q[k]:g}: {o:.3f}  " + " ".join(f"{c} {r[c]['total']:.2f}" for c in fit))
                if o < best - 1e-4:
                    best, bp = o, q
            if best < objective(evaluate([p])[0], p, fit) - 1e-4:
                p = bp
            say(f" after {k}: {p} -> {best:.3f}")
    json.dump(p, open(os.path.join(HERE, "best.json"), "w"))
    say(f"best {p} -> {best:.3f}")


def search(fit, p, scales, logname):
    log = open(os.path.join(HERE, logname), "a")
    say = lambda s: (print(s, flush=True), log.write(s + "\n"), log.flush())
    best = objective(evaluate([p])[0], p, fit)
    say(f"start {p} fit {fit} -> {best:.3f}")
    for sweep, scale in enumerate(scales):
        for k in STEPS:
            cands = []
            for s in STEPS[k]:
                q = dict(p)
                q[k] = round(p[k] + s * scale, 4)
                cands.append(q)
            res = evaluate(cands)
            bp = None
            for q, r in zip(cands, res):
                o = objective(r, q, fit)
                say(f"  sweep {sweep} {k}={q[k]:g}: {o:.3f}  " + " ".join(f"{c} {r[c]['total']:.2f}" for c in ALL))
                if o < best - 1e-4:
                    best, bp = o, q
            if bp:
                p = bp
            say(f" after {k}: {p} -> {best:.3f}")
    return p, best


def cmd_loo():
    start = json.load(open(os.path.join(HERE, "best.json")))
    out = {}
    drop = sys.argv[2].split(",") if len(sys.argv) > 2 else FIT
    for c in drop:
        fit = [x for x in FIT if x != c]
        p, b = search(fit, dict(start), [0.5], f"loo_{c}.log")
        r = evaluate([p])[0]
        out[c] = {"p": p, "objective": b, "dropped_score": r[c]["total"], "scores": {x: r[x]["total"] for x in ALL}}
        json.dump(out, open(os.path.join(HERE, f"loo_{'_'.join(drop)}.json"), "w"), indent=1)


def cmd_refine():
    """More sweeps from best.json at smaller steps, all six."""
    start = json.load(open(os.path.join(HERE, "best.json")))
    p, b = search(FIT, start, [0.25], "refine.log")
    json.dump(p, open(os.path.join(HERE, "best.json"), "w"))


def cmd_pts():
    """Evaluate points given as JSON lists [asc_rail, everyday_asc_rail, everyday_walk_d0_km] (others today's)."""
    pts = []
    for a in json.loads(sys.argv[2]):
        p = today()
        p["asc_rail"], p["everyday_asc_rail"], p["everyday_walk_d0_km"] = a
        pts.append(p)
    evaluate(pts)


if __name__ == "__main__":
    {"pts": cmd_pts, "eval": cmd_eval, "search": cmd_search, "loo": cmd_loo, "refine": cmd_refine}[sys.argv[1]]()
