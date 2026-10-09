"""T-034, T-078 helper: rail share by county group for one or more realnet results, against ACS,
with the trips starting at each operator's stations and the fullest morning segment.

    python pipeline/realnet_groups.py data/work/realnet/result.json [...]
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realnet import COUNTIES, COUNTY_ORDER, WORK  # noqa: E402

GROUPS = {
    "Manhattan": ["36061"],
    "Bk Qn Bx": ["36047", "36081", "36005"],
    "Staten I": ["36085"],
    "Long Island": ["36059", "36103"],
    "NJ inner": ["34017", "34013", "34003", "34039", "34031"],
    "Westch/CT": ["36119", "09001"],
    "outer": ["36087", "36079", "34019", "34023", "34025", "34027", "34029", "34035", "34037", "36027", "36071"],
}
FEEDS = ["subway", "lirr", "mnr", "path", "njt"]


def groups(path):
    tg = json.load(open(os.path.join(WORK, "targets.json")))
    res = json.load(open(path))
    P = res["periods"]
    out = {}
    for g, fs in GROUPS.items():
        r = t = ar = ac = 0.0
        for f in fs:
            i = COUNTY_ORDER.index(f)
            r += sum(p["county"]["rail"][i] for p in P)
            t += sum(p["county"]["trips"][i] for p in P)
            a = tg["acs"][f]
            comm = a["001"] - a["021"]
            ar += a["012"] + a["013"] + a["014"]
            ac += comm
        out[g] = (r / t, ar / ac)
    day = res["day"]
    # trips starting at each operator's stations (first boardings), when the network has feeds
    feeds = defaultdict(float)
    net = json.load(open(res["net"])) if os.path.exists(res["net"]) else {}
    st = net.get("stations", [])
    smap = res.get("st_map") or list(range(len(st)))
    feed_of = {}
    for i, s in enumerate(st):
        d = smap[i]
        if d not in feed_of or s.get("feed") == "subway":
            feed_of[d] = s.get("feed", "?")
    for p in P:
        for d, e in enumerate(p["entries"]):
            feeds[feed_of.get(d, "?")] += e
    worst = max(P[0]["load_of_crush"]) if P[0]["load_of_crush"] else 0
    return out, day["rail"] / day["trips"], feeds, worst, day.get("subzones", 0)


if __name__ == "__main__":
    first = True
    for p in sys.argv[1:]:
        g, share, feeds, worst, nsub = groups(p)
        if first:
            print(f"{'':26}{'all':>6}" + "".join(f"{k[:9]:>10}" for k in g) + "".join(f"{f:>8}" for f in FEEDS) + f"{'AM worst':>9}{'subz':>7}")
            print(f"{'ACS / operators (all trips)':26}{20.3:6.1f}" + "".join(f"{100 * v[1]:10.1f}" for v in g.values()) +
                  f"{'4296k':>8}{'268k':>8}{'236k':>8}{'212k':>8}{'~250k':>8}")
            first = False
        print(f"{os.path.basename(p)[:26]:26}{100 * share:6.1f}" + "".join(f"{100 * v[0]:10.1f}" for v in g.values()) +
              "".join(f"{feeds.get(f, 0) / 1000:7.0f}k" for f in FEEDS) + f"{100 * worst:8.0f}%{nsub:7d}")
