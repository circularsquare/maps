import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_real_vars import VARS  # noqa: E402
base = {}
print("variant | city | rail trips | vs base | commute rail | everyday rail | rail of rail+car | everyday rail of everyday | walk of all | stations x | NHTS errands rail / social rail")
for var in VARS:
    d = os.path.join(HERE, "real", var.replace(" ", "_").replace("+", "_"))
    for city in ("nyc", "lon", "tyo"):
        f = os.path.join(d, f"score_{city}.json")
        if not os.path.exists(f):
            continue
        s = json.load(open(f))
        c, e = s["day"]["commute"], s["day"]["everyday"]
        rail = c[1] + e[1]
        trips, walk = c[0] + e[0], c[2] + e[2]
        if var == "base":
            base[city] = rail
        ev = s.get("everyday_rows") or {}
        evs = " / ".join(f"{100 * ev[k]['rail']:.1f}" for k in ("errands", "social") if k in ev)
        print(f"{var} | {city} | {rail / 1e6:.2f}M | {100 * (rail / base[city] - 1):+.1f}% | {c[1] / 1e6:.2f}M | {e[1] / 1e6:.2f}M | {100 * rail / (trips - walk):.1f}% | "
              f"{100 * e[1] / e[0]:.1f}% | {100 * walk / trips:.1f}% | x{math.exp(s['station_level']):.2f} | {evs}")
