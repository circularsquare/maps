"""T-710: one row per city and setting from the calib runs (score json + run log)."""
import json
import math
import os
import re
import sys

OUT = r"C:\Users\anita\projects\maps\artifacts\trainworld\t710"


def names(net):
    d = net["demand"]
    lines = net["lines"]
    st = net["stations"]
    rn = []
    off = 0
    for li, n in enumerate(d["line_n"]):
        for dr in range(2):
            for i in range(n):
                k = off + i if dr == 0 else off + n - 1 - i
                nx = (k + 1 if dr == 0 else k - 1) if i + 1 < n else None
                rn.append(f"{lines[li].get('name', '?')} {st[d['stops'][k]]['name']}" + (f" > {st[d['stops'][nx]]['name']}" if nx is not None else ""))
        off += n
    return rn


def main():
    tags = sys.argv[1].split(",")
    cities = sys.argv[2].split(",") if len(sys.argv) > 2 else ["nyc", "lon"]
    nets = {}
    for c in cities:
        print(f"\n{c}: tag | rail a day (commute + everyday) | stations x counts | score | operators x | solve s | AM: max crush, avg moved r1/r2/r3, raw moved r2/r3 | busiest AM segment")
        for t in tags:
            p = os.path.join(OUT, f"score_{c}_{t}.json")
            if not os.path.exists(p):
                continue
            s = json.load(open(p))
            run = json.load(open(os.path.join(OUT, f"run_{c}_{t}.json")))
            if c not in nets:
                nets[c] = names(json.load(open(run["net"])))
            log = open(os.path.join(OUT, f"run_{c}_{t}.log")).read()
            am = re.findall(r"T710 q 0 before round (\d): avg moved ([\d.]+), raw moved ([-\d.]+)", log)
            mc = re.search(r"T710 q 0 round \d max_crush ([\d.]+) over_seats (\d+) busiest (\S+)", log)
            top_rn, top_load = mc.group(3).split(":")
            day = s["day"]
            tot = [a + b for a, b in zip(day["commute"], day["everyday"])]
            car = tot[0] - tot[1] - tot[2] - tot[3]
            rc = f"rail of rail+car {tot[1] / (tot[1] + car):.1%}, of all {tot[1] / tot[0]:.1%}, walk {tot[2] / tot[0]:.1%}"
            ops = " ".join(f"{k} {math.exp(v['log_ratio']):.2f}" for k, v in s["operators"].items())
            print(f"{t:8s} | {(day['commute'][1] + day['everyday'][1]) / 1e6:.2f}M ({day['commute'][1] / 1e6:.2f} + {day['everyday'][1] / 1e6:.2f}) | x{math.exp(s['station_level']):.2f} | {s['total']:.2f} | {ops} | {s['ms']['solve'] / 1e3:.1f} | {mc.group(1)}, "
                  + "/".join(a[1] for a in am) + ", " + "/".join(a[2] for a in am[1:]) + f" | {nets[c][int(top_rn)]} {float(top_load):,.0f} | {rc}")


if __name__ == "__main__":
    main()
