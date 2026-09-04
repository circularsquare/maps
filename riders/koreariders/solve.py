# -*- coding: utf-8 -*-
"""Solve the whole network at once, instead of each line on its own.

Cumulating one line in isolation assumes every passenger who boards on it also
alights on it. They do not. Someone riding 서울 to 목포 by 무궁화 boards at 서울,
alights at 목포, and leaves 경부선 at 대전조차장 *without appearing in any
station's 하차* -- they simply carry on down 호남선. A per-line cumulation has no
way to represent that, so it carries them all the way to 부산, and 경부선 ends up
with negative loads once the real alightings are subtracted.

So the load on line L is modelled as

    load_i = base_i + E_L + sum of tau over the junctions at or before stop i

where `base` is the plain cumulation of that line's allocated station flows, E_L
is the flow entering the line at its first stop, and each `tau` is the traffic
stepping on or off the line at a junction without using a platform. Conservation
makes the taus at one junction sum to zero: what leaves one line joins another.

Unknowns, all solved together:

    E_L        one per line per direction -- entry flow
    tau        one per (line, junction, direction) -- interchange step
    share      how a shared station's 승하차 splits between the lines calling
               there; 익산's 무궁화 passengers are 호남선's, 전라선's and 장항선's
               and the yearbook reports one number

Constraints, as weighted residuals for scipy's least_squares:

    anchor     at a true terminus everything alights, so the last segment's load
               is known outright -- this is what made the single-line version
               work and it still carries most of the information
    positive   no segment may carry fewer than nobody; the hard evidence that a
               reconstruction has gone wrong
    passing    each line's rebuilt 통과인원 against the published count
    junction   taus at a junction sum to zero, per direction
    mirror     하행 and 상행 profiles should agree, as they do to ~2% wherever
               the single-line method was trustworthy
    prior      taus small unless the data demands them, shares near their
               traffic-weighted split -- these only pick among the solutions the
               constraints above leave open

    python solve.py            # solve and report
    python solve.py --line 경부선
"""

import argparse
import collections
import io
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

import build as B
import lines as LN
import membership as M

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "segments.geojson")

# Residual weights. The first three are evidence, the last two are taste; the
# ordering matters more than the values.
W_ANCHOR = 3.0
W_POSITIVE = 6.0
W_PASSING = 2.0
W_JUNCTION = 6.0
W_MIRROR = 1.0
W_TAU = 0.15
W_SHARE = 0.30

SCALE = 1e6         # work in millions of passengers/year, so residuals compare


def build_chains(table, serves, g, named):
    """Each line's stops in running order, keeping every station that calls."""
    chains = {}
    for canon, spec in table.items():
        if "error" in spec or spec["passing"] <= 0:
            continue
        # Membership by track proximity plus a train type that stops there,
        # rather than the roster's one-home-line-per-station.
        spec = dict(spec)
        spec["roster"] = set()
        stops, err = B.order_stations(spec, g, named, spec["flows"])
        if err:
            print("%-11s %s" % (canon, err))
            continue
        keep = [(km, nm) for km, nm in stops
                if canon in serves.get(nm, [canon]) or nm in (spec["first"],
                                                              spec["last"])]
        if len(keep) >= 3:
            chains[canon] = keep
    return chains


class Network(object):
    def __init__(self, table, chains, serves, flows_by_line):
        self.table, self.chains = table, chains
        self.flows = flows_by_line
        self.lines = sorted(chains)

        # Which lines call at each station, restricted to those actually mapped.
        on = collections.defaultdict(set)
        for L, stops in chains.items():
            for _, nm in stops:
                on[nm].add(L)
        self.on = on
        self.shared = sorted(nm for nm, ls in on.items() if len(ls) > 1)

        # Parameter layout.
        self.idx, n = {}, 0
        for L in self.lines:
            for d in (0, 1):
                self.idx[("E", L, d)] = n
                n += 1
        self.taus = []
        for L in self.lines:
            for i, (_, nm) in enumerate(chains[L]):
                if i > 0 and nm in self.shared:
                    for d in (0, 1):
                        self.idx[("T", L, nm, d)] = n
                        self.taus.append((L, nm, d, i))
                        n += 1
        self.shares = []
        for nm in self.shared:
            for L in sorted(on[nm]):
                self.idx[("S", nm, L)] = n
                self.shares.append((nm, L))
                n += 1
        self.n = n

        # Traffic-weighted prior for the shares: a busier line takes more of a
        # shared station's passengers. Crude, but it only breaks ties.
        self.share_prior = {}
        for nm in self.shared:
            w = {L: max(table[L]["passing"], 1.0) for L in on[nm]}
            tot = sum(w.values())
            for L in on[nm]:
                self.share_prior[(nm, L)] = w[L] / tot

    def x0(self):
        x = np.zeros(self.n)
        for L in self.lines:
            f = self.flows[L]
            last = self.chains[L][-1][1]
            for d in (0, 1):
                # Start from the single-line answer: anchor on the far end.
                v = f.get(last, (0, 0, 0, 0))[1 if d == 0 else 2]
                x[self.idx[("E", L, d)]] = v / SCALE
        for nm, L in self.shares:
            x[self.idx[("S", nm, L)]] = self.share_prior[(nm, L)]
        return x

    def profile(self, x, L, d):
        """Segment loads for one line and direction, in millions/year."""
        stops = self.chains[L]
        f = self.flows[L]
        base, cur = [], 0.0
        for i, (_, nm) in enumerate(stops[:-1]):
            s = self._share(x, nm, L)
            v = f.get(nm, (0, 0, 0, 0))
            b, a = (v[0], v[1]) if d == 0 else (v[2], v[3])
            cur += s * (b - a) / SCALE
            base.append(cur)
        # Entry flow, then a step at each junction.
        out, step = [], x[self.idx[("E", L, d)]]
        for i, val in enumerate(base):
            k = self.idx.get(("T", L, stops[i][1], d))
            if k is not None and i > 0:
                step += x[k]
            out.append(val + step)
        return np.array(out)

    def _share(self, x, nm, L):
        k = self.idx.get(("S", nm, L))
        return 1.0 if k is None else x[k]

    def residuals(self, x):
        r = []
        prof = {}
        for L in self.lines:
            for d in (0, 1):
                prof[(L, d)] = self.profile(x, L, d)

        for L in self.lines:
            spec, stops = self.table[L], self.chains[L]
            f = self.flows[L]
            down, up = prof[(L, 0)], prof[(L, 1)]

            # anchor -- only where the far end is a genuine terminus
            if spec["clean_end"]:
                last = stops[-1][1]
                v = f.get(last, (0, 0, 0, 0))
                r.append(W_ANCHOR * (down[-1] - v[1] / SCALE))
                r.append(W_ANCHOR * (up[-1] - v[2] / SCALE))

            # positivity
            r.extend(W_POSITIVE * np.minimum(down, 0.0))
            r.extend(W_POSITIVE * np.minimum(up, 0.0))

            # mirror
            r.extend(W_MIRROR * (down - up))

            # 통과인원
            if spec["passing"] > 0:
                interior = sum(self._share(x, nm, L)
                               * (f.get(nm, (0, 0, 0, 0))[0]
                                  + f.get(nm, (0, 0, 0, 0))[2])
                               for _, nm in stops[1:-1]) / SCALE
                users = down[0] + up[-1] + interior
                r.append(W_PASSING * (users - spec["passing"] / SCALE))

        # junction conservation, per direction
        by_junction = collections.defaultdict(list)
        for (L, nm, d, i) in self.taus:
            by_junction[(nm, d)].append(self.idx[("T", L, nm, d)])
        for key, ks in by_junction.items():
            r.append(W_JUNCTION * sum(x[k] for k in ks))

        # priors
        for (L, nm, d, i) in self.taus:
            r.append(W_TAU * x[self.idx[("T", L, nm, d)]])
        for nm in self.shared:
            ks = [self.idx[("S", nm, L)] for L in sorted(self.on[nm])]
            r.append(W_JUNCTION * (sum(x[k] for k in ks) - 1.0))
        for nm, L in self.shares:
            r.append(W_SHARE * (x[self.idx[("S", nm, L)]]
                                - self.share_prior[(nm, L)]))
        return np.array(r)

    def bounds(self):
        lo = np.full(self.n, -np.inf)
        hi = np.full(self.n, np.inf)
        for nm, L in self.shares:
            lo[self.idx[("S", nm, L)]] = 0.0
            hi[self.idx[("S", nm, L)]] = 1.0
        for L in self.lines:
            for d in (0, 1):
                lo[self.idx[("E", L, d)]] = 0.0
        return lo, hi


def report(net, x, only=None):
    rows = []
    for L in net.lines:
        down, up = net.profile(x, L, 0) * SCALE, net.profile(x, L, 1) * SCALE
        stops = net.chains[L]
        f = net.flows[L]
        interior = sum(net._share(x, nm, L)
                       * (f.get(nm, (0, 0, 0, 0))[0] + f.get(nm, (0, 0, 0, 0))[2])
                       for _, nm in stops[1:-1])
        users = down[0] + up[-1] + interior
        length = stops[-1][0]
        pkm = sum((down[i] + up[i]) * (stops[i + 1][0] - stops[i][0])
                  for i in range(len(down)))
        mirror = max(abs(down[i] - up[i]) / max(abs(down[i]), abs(up[i]), 1)
                     for i in range(len(down)))
        rows.append({
            "line": L, "stops": stops, "down": down, "up": up,
            "users": users, "passing": net.table[L]["passing"],
            "mirror": mirror, "length": length,
            "density": pkm / length / 365.0 if length else 0.0,
            "worst": min(down.min(), up.min()),
        })

    print("\n%-11s %5s %7s %11s %11s %7s %9s %10s"
          % ("line", "stops", "km", "통과인원", "yearbook", "mirror",
             "수송밀도", "min load"))
    print("-" * 82)
    for r in sorted(rows, key=lambda z: -z["density"]):
        print("%-11s %5d %7.1f %11.0f %11.0f %6.1f%% %9.0f %10.0f"
              % (r["line"], len(r["stops"]), r["length"], r["users"],
                 r["passing"], 100 * r["mirror"], r["density"], r["worst"]))

    bad = [r for r in rows if r["worst"] < -1000]
    print("\n%d of %d lines still carry a negative segment"
          % (len(bad), len(rows)))
    ratios = [r["users"] / r["passing"] for r in rows if r["passing"] > 0]
    print("통과인원 ratio: median %.3f, worst %.3f / %.3f"
          % (float(np.median(ratios)), min(ratios), max(ratios)))
    print("mirror: median %.1f%%, worst %.1f%%"
          % (100 * float(np.median([r["mirror"] for r in rows])),
             100 * max(r["mirror"] for r in rows)))

    if only:
        r = [z for z in rows if z["line"] == only][0]
        print("\n%-15s %8s %10s %10s %10s"
              % ("segment", "km", "하행", "상행", "명/일"))
        print("-" * 56)
        for i in range(len(r["down"])):
            a, b = r["stops"][i], r["stops"][i + 1]
            print("%-15s %8.1f %10.0f %10.0f %10.0f"
                  % ((a[1] + "-" + b[1])[:15], b[0] - a[0], r["down"][i],
                     r["up"][i], (r["down"][i] + r["up"][i]) / 365.0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line")
    args = ap.parse_args()

    table, _ = LN.resolve()
    serves = M.serves(LN)
    with io.open(B.STATIONS, encoding="utf-8") as f:
        named = collections.defaultdict(list)
        for n in json.load(f)["elements"]:
            nm = (n.get("tags", {}) or {}).get("name")
            if nm:
                named[nm].append((n["lat"], n["lon"]))
    print("loading the national track graph ...")
    g = B.load_network()
    chains = build_chains(table, serves, g, named)
    flows = {L: table[L]["flows"] for L in chains}
    net = Network(table, chains, serves, flows)
    print("   %d lines, %d junctions, %d unknowns"
          % (len(net.lines), len(net.shared), net.n))

    x0 = net.x0()
    print("   initial cost %.1f" % (0.5 * float((net.residuals(x0) ** 2).sum())))
    lo, hi = net.bounds()
    res = least_squares(net.residuals, np.clip(x0, lo, hi), bounds=(lo, hi),
                        method="trf", xtol=1e-10, ftol=1e-10, max_nfev=400)
    print("   solved: cost %.1f, %d evaluations, %s"
          % (res.cost, res.nfev, res.message.split(".")[0].lower()))

    rows = report(net, res.x, args.line)
    write_geojson(rows)


def write_geojson(rows):
    feats = []
    for r in rows:
        for i in range(len(r["down"])):
            a, b = r["stops"][i], r["stops"][i + 1]
            feats.append({
                "type": "Feature",
                "properties": {
                    "line": r["line"], "from": a[1], "to": b[1],
                    "km": round(b[0] - a[0], 3),
                    "down": round(float(r["down"][i])),
                    "up": round(float(r["up"][i])),
                    "daily": round(float(r["down"][i] + r["up"][i]) / 365.0),
                },
                "geometry": None,
            })
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f,
                  ensure_ascii=False)
    print("\nwrote %s (%d segments)" % (os.path.relpath(OUT, HERE), len(feats)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
