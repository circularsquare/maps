"""T-1140 US dry runs: today's rule and the partner-or-compact rule over every US and Puerto Rico zone
county, metros and the old Northeast packs fixed. Read-only; writes us_runs.pkl here (merged).

    python run_us.py <variant> [<variant> ...]
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import math
import pickle
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
import common as C  # noqa: E402
import rule1140 as R  # noqa: E402
import world_plan as wp  # noqa: E402
from shapely import union_all  # noqa: E402

t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def hull_u(gz, go, cz, co):
    return C.hull_ratio(union_all([gz, go]))


def sep(gz, go, cz, co):
    """The union's equivalent radius over the distance between the two people centres."""
    r = math.sqrt((gz.area + go.area) / math.pi)
    d = math.dist(cz, co)
    return r / d if d else 9.0


FNS = {"hull": hull_u, "sep": sep}
# name: kwargs for rule1140.zone_groups (flows "acs" or "grav")
VARIANTS = {
    "today": {},
    "p50": {"flows": "acs", "alpha": 0.5, "small_by": "flow"},
    "p50h70": {"flows": "acs", "alpha": 0.5, "compact": ("hull", 0.7), "small_by": "flow"},
    "p50h75": {"flows": "acs", "alpha": 0.5, "compact": ("hull", 0.75), "small_by": "flow"},
    "p50h80": {"flows": "acs", "alpha": 0.5, "compact": ("hull", 0.8), "small_by": "flow"},
    "p33h75": {"flows": "acs", "alpha": 0.33, "compact": ("hull", 0.75), "small_by": "flow"},
    "p67h75": {"flows": "acs", "alpha": 0.67, "compact": ("hull", 0.75), "small_by": "flow"},
    "p50s": {"flows": "acs", "alpha": 0.5, "compact": ("sep", 0.8), "small_by": "flow"},
    "g50h75": {"flows": "grav", "alpha": 0.5, "compact": ("hull", 0.75), "small_by": "flow"},
    "h75": {"compact": ("hull", 0.75)},
    "r": {"flows": "acs", "alpha": 0.5, "compact": ("hull", 0.75), "relative": 0.05, "small_by": "flow"},
    "rx": {"flows": "acs", "alpha": 0.5, "compact": ("hull", 0.75), "relative": 0.05, "small_by": "flow", "cross": True},
    "rxf": {"flows": "acs", "alpha": 0.5, "compact": ("hull", 0.75), "relative": 0.05, "small_by": "flow", "cross": True,
            "order": "flow"},
    "rf": {"flows": "acs", "alpha": 0.5, "compact": ("hull", 0.75), "relative": 0.05, "small_by": "flow", "order": "flow"},
    "grx": {"flows": "grav", "alpha": 0.5, "compact": ("hull", 0.75), "relative": 0.05, "small_by": "flow", "cross": True},
    "rb": {"flows": "acs", "alpha": 0.5, "compact": ("hull", 0.75), "relative": 0.05, "small_by": "flow", "cross": "fallback"},
    "rbp": {"flows": "acs", "alpha": 0.5, "compact": ("hull", 0.75), "relative": 0.05, "small_by": "flow", "cross": "fallback"},
    "grbp": {"flows": "grav", "alpha": 0.5, "compact": ("hull", 0.75), "relative": 0.05, "small_by": "flow", "cross": "fallback"},
    "rbpa": {"flows": "acs", "alpha": 0.5, "compact": ("hull", 0.75), "relative": 0.05, "small_by": "flow", "cross": "fallback",
             "aref": 10000},
    "rbpa20": {"flows": "acs", "alpha": 0.5, "compact": ("hull", 0.75), "relative": 0.05, "small_by": "flow", "cross": "fallback",
               "aref": 20000},
    "grb": {"flows": "grav", "alpha": 0.5, "compact": ("hull", 0.75), "relative": 0.05, "small_by": "flow", "cross": "fallback"},
    "rb33": {"flows": "acs", "alpha": 0.33, "compact": ("hull", 0.75), "relative": 0.05, "small_by": "flow", "cross": "fallback"},
    "rb67": {"flows": "acs", "alpha": 0.67, "compact": ("hull", 0.75), "relative": 0.05, "small_by": "flow", "cross": "fallback"},
    "rb70": {"flows": "acs", "alpha": 0.5, "compact": ("hull", 0.70), "relative": 0.05, "small_by": "flow", "cross": "fallback"},
    "rb80": {"flows": "acs", "alpha": 0.5, "compact": ("hull", 0.80), "relative": 0.05, "small_by": "flow", "cross": "fallback"},
    "gr": {"flows": "grav", "alpha": 0.5, "compact": ("hull", 0.75), "relative": 0.05, "small_by": "flow"},
}


def prep():
    path = os.path.join(C.HERE, "us_prep.pkl")
    if os.path.exists(path):
        return C.load("us_prep.pkl")
    D = C.load("us_data.pkl")
    P = {}
    for iso, d in D.items():
        U = d["units"]
        simple = {i: wp.polygonal(u.geom.simplify(500)) for i, u in U.items()}
        adj = wp.unit_borders(simple)
        acs = C.acs_sym(d["flows"], list(U)) if d["flows"] else None
        grav = C.gravity_flows(d["cells"], list(U))
        P[iso] = {"simple": simple, "adj": adj, "acs": acs, "grav": grav}
        log(f"{iso}: shapes, borders and flows")
    pickle.dump(P, open(path, "wb"))
    return P


def run(tag, kw, D, P):
    out = {}
    for iso, d in D.items():
        U = d["units"]
        p = P[iso]
        rule = wp.ZONE_RULES[iso]
        units = list(U.values())
        free = list(d["free"])
        region = {i: u.parents[0] for i, u in U.items()}
        k = dict(kw)
        fl = k.pop("flows", None)
        if fl:
            k["flows"] = p["acs"] if (fl == "acs" and p["acs"]) else p["grav"]
        if "compact" in k:
            k["compact"] = (FNS[k["compact"][0]], k["compact"][1])
        tests = []
        groups, lines = R.zone_groups(units, free, region, rule["floor"], rule["cap"], True, d["fixed"],
                                      fixed_units={f["id"]: f["units"] for f in d["fixed"]},
                                      simple=p["simple"], adj=p["adj"], tests=tests, **k)
        for lead in [g for g, ids in groups.items() if sum(U[i].free for i in ids) < wp.ZONE_ALONE_MIN]:
            groups.pop(lead)
        out[iso] = {"groups": groups, "lines": lines, "tests": tests}
        log(f"{tag} {iso}: {len(groups)} zones")
    return out


def main():
    D = C.load("us_data.pkl")
    P = prep()
    path = os.path.join(C.HERE, "us_runs.pkl")
    allr = C.load("us_runs.pkl") if os.path.exists(path) else {}
    for tag in sys.argv[1:]:
        allr[tag] = run(tag, VARIANTS[tag], D, P)
        pickle.dump(allr, open(path, "wb"))
    log("written")


if __name__ == "__main__":
    main()
