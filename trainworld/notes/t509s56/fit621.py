"""T-621 stage 2: the deep calibration's fit, on fit.py's harness with per-country parameters.

    python fit621.py eval <tag> [k=v ...]       # one point, every city, the table
    python fit621.py search [sweeps]            # coordinate search from START (or best.json with --resume)
    python fit621.py pts '<json list>'          # evaluate points (dicts over START)
    python fit621.py design <n> [seed]          # n random points around best.json, for the surrogates

Environment: FIT_DIR (required: this fit's own cache.jsonl, logs, best.json), FIT_FIXED
(k=v,... held in every run, e.g. car_own_usd_trip=1.5,car_own_usd_km=0.1), FIT_WORKERS,
CALIB_BIN and TW_ROOT (passed on to calib.py: the binary built from T-621's copy of sim/).

A parameter `name@CC` applies to the cities of country CC only (`country_asc_rail@JP` sets
Japan's rail constant); a walking constant with no value for a city's country takes the mean
of the countries' values (the worldwide value). Each city's run is keyed by the values that
reach it, so moving one country's parameter reruns only its cities. New York weighs half
(Anita, 2026-10-16); the penalty for each parameter is weighted by the fitted cities it reaches.
"""
import json
import math
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.environ["FIT_DIR"]
os.makedirs(HERE, exist_ok=True)
TW = r"C:\Users\anita\projects\maps\trainworld"
PY = r"C:\Users\anita\projects\maps\venv\Scripts\python.exe"
CALIB = os.path.join(TW, "data", "work", "calib")
CACHE = os.path.join(HERE, "cache.jsonl")
BEST = os.environ.get("FIT_BEST", "best.json")  # a variant keeps its own best point
FIT = ["nyc", "lon", "par", "seo", "tyo", "sdj"]
HOLD = ["tor"]
ALL = FIT + HOLD
COUNTRY = {"nyc": "US", "lon": "GB", "par": "FR", "seo": "KR", "tyo": "JP", "sdj": "JP", "tor": "CA"}
WEIGHT = {"nyc": 0.5}
WORKERS = int(os.environ.get("FIT_WORKERS", "8"))
PENALTY = 0.05
WORLD_MEAN = ("asc_walk", "everyday_walk")  # countries without a value take the countries' mean

# start (the shipped values), the plausible range (the penalty's unit) and the first sweep's steps
PARAMS = {
    "asc_walk@JP": (2.80, 1.0, [-2.5, -2.0, -1.5, -1.0]),
    "asc_walk@GB": (2.80, 1.0, [-2.0, -1.5, -1.0, -0.5]),
    "asc_walk@FR": (2.80, 1.0, [-2.0, -1.5, -1.0, -0.5]),
    "asc_walk@US": (2.80, 1.0, [-1.0, -0.5, 0.5]),
    # the everyday layer's whole walking constant (asc_walk + everyday_asc_walk), searched apart
    # from the commutes' so the two do not drag each other (`resolve` sets the offset)
    "everyday_walk@US": (3.21, 1.0, [-1.0, -0.5, 0.5, 1.0]),
    "everyday_walk@JP": (3.21, 1.0, [-1.0, -0.5, 0.5, 1.0]),
    "everyday_walk@GB": (3.21, 1.0, [0.5, 1.0, 1.5, 2.0]),
    "everyday_walk@FR": (3.21, 1.0, [0.5, 1.0, 1.5, 2.0]),
    "asc_rail": (0.4, 1.0, [-0.5, -0.25, 0.25, 0.5]),
    "everyday_asc_rail": (-1.1, 1.0, [-0.5, -0.25, 0.25, 0.5]),
    "country_asc_rail@JP": (0.5, 1.0, [-0.5, -0.25, 0.25, 0.5]),
    "other_asc": (10.0, 10.0, [-5.0, 5.0, 10.0]),
}
for _k in filter(None, os.environ.get("FIT_DROP", "").split(",")):
    PARAMS.pop(_k)
for _kv in filter(None, os.environ.get("FIT_ADD", "").split(";")):
    # name=start,range,step|step|...
    _n, _v = _kv.split("=")
    _s, _r, _st = _v.split(",")
    PARAMS[_n] = (float(_s), float(_r), [float(x) for x in _st.split("|")])
FIXED = {k: float(v) for k, v in (kv.split("=") for kv in filter(None, os.environ.get("FIT_FIXED", "").split(",")))}
START = {k: v[0] for k, v in PARAMS.items()}


def resolve(city, p):
    """The `--set` values a city's run takes from point p."""
    cc = COUNTRY[city]
    out = dict(FIXED)
    per = {}
    for k, v in p.items():
        if "@" in k:
            name, c = k.split("@")
            per.setdefault(name, {})[c] = v
        else:
            out[k] = v
    for name, by in per.items():
        if cc in by:
            out[name] = by[cc]
        elif name in WORLD_MEAN:
            out[name] = sum(by.values()) / len(by)
    if "everyday_walk" in out:
        out["everyday_asc_walk"] = out.pop("everyday_walk") - out.get("asc_walk", 2.80)
    return out


def key(city, p):
    s = resolve(city, p)
    return city + "|" + ",".join(f"{k}={s[k]:.6g}" for k in sorted(s))


def reaches(k):
    """The fitted cities parameter k reaches (directly; a mean's reach is ignored)."""
    if "@" not in k:
        return FIT
    return [c for c in FIT if COUNTRY[c] == k.split("@")[1]]


def load_cache():
    c = {}
    if os.path.exists(CACHE):
        for line in open(CACHE):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            c[r["key"]] = r
    return c


def run_one(city, p, tag):
    s = resolve(city, p)
    cmd = [PY, os.path.join(TW, "pipeline", "calib.py"), "run", city, "--tag", tag]
    if s:
        cmd += ["--set", ",".join(f"{k}={v:.6g}" for k, v in sorted(s.items()))]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    keep = not tag.startswith("f_")
    out = open(os.path.join(CALIB, city, f"score_{tag}.txt"), "w", encoding="utf-8") if keep else subprocess.DEVNULL
    subprocess.run(cmd, check=True, stdout=out, stderr=subprocess.STDOUT, env=env)
    sc = json.load(open(os.path.join(CALIB, city, f"score_{tag}.json")))
    d = sc["day"]
    kt, wl = d.get("everyday_kinds") or [], d.get("everyday_walks_all") or []
    r = {"key": key(city, p), "city": city, "set": s, "total": sc["total"], "parts": sc["parts"],
         "station_level": sc["station_level"], "rail_share_model": sc["rail_share_model"],
         "rail_share_measured": sc["rail_share_measured"], "walk_areas": sc.get("walk_areas"),
         "commute": d["commute"], "everyday": d["everyday"], "kinds": kt, "walks_all": wl,
         "ev_rows": sc.get("everyday_rows"), "operators": {k: v["log_ratio"] for k, v in sc["operators"].items()}, "tag": tag}
    if not keep:
        for f in (f"run_{tag}.json", f"run_{tag}.log", f"score_{tag}.json"):
            try:
                os.remove(os.path.join(CALIB, city, f))
            except OSError:
                pass
    return r


ORDER = {"tyo": 0, "lon": 1, "nyc": 2, "par": 3, "seo": 4, "tor": 5, "sdj": 6}


def evaluate(points, cities=ALL, prefix="f"):
    """points: parameter dicts (full). Runs what the cache lacks, in parallel; one result dict per point."""
    cache = load_cache()
    jobs, seen = [], set()
    for p in points:
        for c in cities:
            k = key(c, p)
            if k not in cache and k not in seen:
                seen.add(k)
                jobs.append((c, p))
    jobs.sort(key=lambda j: ORDER[j[0]])

    def go(ij):
        i, (c, p) = ij
        tag = prefix if prefix != "f" else f"f_{os.getpid()}_{i}_{random.randrange(1 << 30)}"
        return run_one(c, p, tag)

    if jobs:
        with ThreadPoolExecutor(WORKERS) as ex:
            for r in ex.map(go, enumerate(jobs)):
                with open(CACHE, "a") as f:
                    f.write(json.dumps(r) + "\n")
        cache = load_cache()
    return [{c: cache[key(c, p)] for c in cities} for p in points]


def penalty(p):
    out = 0.0
    for k, (v0, rng, _) in PARAMS.items():
        w = sum(WEIGHT.get(c, 1.0) for c in reaches(k))
        out += PENALTY * w * ((p[k] - v0) / rng) ** 2
    return out


# everyday walking against a travel survey outside New York (whose NHTS rows are calib.py's
# everyday part): {city: [(kinds, survey walk share, weight)]}, kinds by name, scored as logits
# (notes/T-621.md "Stage 2" has the sources: LTDS 2019/20 by purpose, derived from Travel in
# London 2024's figures; EGT 2010 home-based trips; Tokyo PT 2018 table c-1, home to private and
# home to school). Sendai shares Tokyo's constant; Seoul and Toronto have no survey on hand.
EV_WALK = {} if os.environ.get("FIT_NO_EVWALK") else {
    "lon": [(["errands", "medical"], 0.423, 1.0), (["social"], 0.410, 1.0), (["school"], 0.438, 0.5)],
    "par": [(["errands", "medical"], 0.45, 1.0), (["social"], 0.41, 1.0), (["school"], 0.55, 0.5)],
    "tyo": [(["errands", "social", "medical"], 0.290, 2.0), (["school"], 0.494, 0.5)],
}
KINDS = ["errands", "social", "school", "university", "airport", "medical"]


def lg(x):
    x = min(max(x, 0.01), 0.99)
    return math.log(x / (1 - x))


def ev_walk_part(c, r):
    rows = EV_WALK.get(c)
    if not rows or not r["kinds"]:
        return 0.0
    num = den = 0.0
    for kinds, share, w in rows:
        ix = [KINDS.index(k) for k in kinds]
        trips = sum(r["kinds"][3 * i] for i in ix)
        walk = sum(r["walks_all"][i] for i in ix)
        num += w * (lg(walk / max(trips, 1e-9)) - lg(share)) ** 2
        den += w
    return num / den


def total(c, r):
    return r["total"] + ev_walk_part(c, r)


def objective(res, p, fit=FIT):
    return sum(WEIGHT.get(c, 1.0) * total(c, res[c]) for c in fit) + penalty(p)


def ev_walk(r):
    kt, wl = r["kinds"], r["walks_all"]
    return sum(wl) / max(sum(kt[0::3]), 1) if kt and wl else float("nan")


def table(res, say=print):
    say(f"{'city':5s} {'score':>6s} {'stat':>6s} {'oper':>6s} {'area':>6s} {'walk':>6s} {'every':>6s} {'level':>6s} "
        f"{'a.rail':>7s} {'meas':>6s} {'a.walk':>7s} {'meas':>6s} {'c.rail':>7s} {'ev.rail':>7s} {'ev.walk':>7s}")
    for c, r in res.items():
        pa = r["parts"]
        wa = r.get("walk_areas") or {}
        cm, ev = r["commute"], r["everyday"]
        say(f"{c:5s} {total(c, r):6.3f} {pa['stations']:6.3f} {pa['operators']:6.3f} {pa['areas']:6.3f} {pa.get('walk', 0):6.3f} {pa.get('everyday', 0) + ev_walk_part(c, r):6.3f} "
            f"x{math.exp(r['station_level']):5.2f} {r['rail_share_model']:7.1%} {r['rail_share_measured']:6.1%} "
            f"{wa.get('model', float('nan')):7.1%} {wa.get('measured', float('nan')):6.1%} {cm[1] / cm[0]:7.1%} {ev[1] / max(ev[0], 1):7.1%} {ev_walk(r):7.1%}")


def parse(args, base=None):
    p = dict(base or START)
    for a in args:
        k, v = a.split("=")
        p[k] = float(v)
    return p


def cmd_eval():
    tag = sys.argv[2]
    p = parse(sys.argv[3:])
    res = evaluate([p], prefix=tag)[0]
    table(res)
    print(f"objective (fitted six, NY half, with penalty) {objective(res, p):.3f}; penalty {penalty(p):.3f}")


def search(fit, p, scales, logname, params=None):
    log = open(os.path.join(HERE, logname), "a")
    say = lambda s: (print(s, flush=True), log.write(s + "\n"), log.flush())  # noqa: E731
    best = objective(evaluate([p])[0], p, fit)
    say(f"start {json.dumps(p)} fit {fit} -> {best:.4f}")
    for sweep, scale in enumerate(scales):
        for k in params or PARAMS:
            if not any(c in fit for c in reaches(k)):
                continue
            cands = []
            for s in PARAMS[k][2]:
                q = dict(p)
                q[k] = round(p[k] + s * scale, 4)
                cands.append(q)
            res = evaluate(cands)
            bp = None
            for q, r in zip(cands, res):
                o = objective(r, q, fit)
                say(f"  sweep {sweep} {k}={q[k]:g}: {o:.4f}  " + " ".join(f"{c} {total(c, r[c]):.3f}" for c in ALL))
                if o < best - 1e-4:
                    best, bp = o, q
            if bp:
                p = bp
            say(f" after {k}: {json.dumps(p)} -> {best:.4f}")
        json.dump(p, open(os.path.join(HERE, f"{logname}.best.json"), "w"))
    return p, best


def cmd_search():
    sweeps = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    p = dict(START)
    if "--resume" in sys.argv:
        p.update(json.load(open(os.path.join(HERE, BEST))))
    scales = [1.0, 0.5, 0.25, 0.125][:sweeps]
    p, b = search(FIT, p, scales, "search.log")
    json.dump(p, open(os.path.join(HERE, BEST), "w"))
    res = evaluate([p])[0]
    table(res)
    print(f"best {json.dumps(p)} -> {b:.4f}")


def cmd_pts():
    pts = []
    arg = sys.argv[2]
    for a in json.loads(open(arg[1:]).read() if arg.startswith("@") else arg):
        p = dict(START)
        p.update(a)
        pts.append(p)
    evaluate(pts)


def cmd_design():
    """n points around best.json, each parameter uniform within +-DESIGN_SPAN ranges."""
    n = int(sys.argv[2])
    rnd = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    span = float(os.environ.get("DESIGN_SPAN", "0.4"))
    b = json.load(open(os.path.join(HERE, BEST)))
    pts = []
    for _ in range(n):
        q = dict(START)
        q.update(b)
        for k, (v0, rng, _) in PARAMS.items():
            q[k] = round(q[k] + rnd.uniform(-span, span) * rng, 4)
        pts.append(q)
    json.dump(pts, open(os.path.join(HERE, f"design_{sys.argv[3] if len(sys.argv) > 3 else 1}.json"), "w"))
    evaluate(pts)


if __name__ == "__main__":
    {"eval": cmd_eval, "search": cmd_search, "pts": cmd_pts, "design": cmd_design}[sys.argv[1]]()
