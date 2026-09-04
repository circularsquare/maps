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
import frequency as FQ
import lines as LN
import membership as M

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "segments.geojson")

# Residual weights. The first three are evidence, the last two are taste; the
# ordering matters more than the values.
W_ANCHOR = 3.0
W_POSITIVE = 6.0
W_CEILING = 20.0
W_PASSING = 2.0
W_JUNCTION = 2.0
W_SHARESUM = 6.0
W_MIRROR = 1.0
W_TAUSYM = 2.0
W_FREQ = 12.0
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
        self.rev = {L: bool(table[L].get("reversed")) for L in self.lines}

        # Published trains per section. Where the count does not change across a
        # station, no service joins or leaves there and a junction step is not
        # merely unlikely but impossible -- 경부선 runs 66 conventional trains a
        # day from 서울 unchanged as far as 천안, so the 3.7M step the fit kept
        # putting at 용산 cannot exist however well it suited the 승하차.
        # Stored in chain order, which is backwards from the sheet's for a
        # reversed line.
        self.freq = {}
        for L in self.lines:
            ch = FQ.changes(L, table[L]["types"])
            if ch is not None:
                self.freq[L] = {nm: ((b, a) if self.rev[L] else (a, b))
                                for nm, (a, b) in ch.items()}

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
        # A tau only exists where it can move a segment. Stop 0's entry is E,
        # and traffic joining at the very last stop rides no segment of this
        # line at all -- allocating one there gives the junction constraint a
        # free variable that satisfies it without touching any profile, which
        # is how 오송 came to inject 307k passengers into 충북선.
        self.taus = []
        for L in self.lines:
            for i, (_, nm) in enumerate(chains[L][:-1]):
                if i > 0 and nm in self.shared:
                    for d in (0, 1):
                        self.idx[("T", L, nm, d)] = n
                        self.taus.append((L, nm, d, i))
                        n += 1
        # Shares are per train type, not per line. A station's row for one type
        # is divided between the lines there that actually run it: the KTX at
        # 광주송정 are 호남고속선's and 광주선's to divide and have nothing to do
        # with 호남선's 무궁화 at the same platforms, and 부산's 무궁화 are
        # 경부선's alone however many KTX also terminate there. Dividing per line
        # instead let 광주선 -- which carries every type -- bridge the two, and
        # scaled the high-speed lines down against traffic that was never theirs.
        self.kinds = {L: list(table[L]["types"]) for L in self.lines}
        self.fk = {L: table[L].get("flows_by_kind", {}) for L in self.lines}
        self.groups = []
        for nm in self.shared:
            for k in LN.ALL_TYPES:
                ls = [L for L in sorted(on[nm])
                      if k in self.kinds[L] and any(self.fk[L].get(k, {})
                                                    .get(nm, (0,) * 4))]
                if len(ls) > 1:
                    self.groups.append((nm, k, ls))

        self.shares = []
        for nm, k, grp in self.groups:
            for L in grp:
                self.idx[("S", nm, k, L)] = n
                self.shares.append((nm, k, L))
                n += 1
        self.n = n

        # Traffic-weighted prior for the shares: a busier line takes more of a
        # shared station's passengers. Crude, but it only breaks ties.
        self.share_prior = {}
        for nm, k, grp in self.groups:
            w = {L: max(table[L]["passing"], 1.0) for L in grp}
            tot = sum(w.values())
            for L in grp:
                self.share_prior[(nm, k, L)] = w[L] / tot

        # Precomputed allocation terms, so a residual pass is a short walk over
        # a list rather than a dictionary lookup per station per train type.
        self.alloc = collections.defaultdict(list)
        for L in self.lines:
            for _, nm in chains[L]:
                for k in self.kinds[L]:
                    v = self.fk[L].get(k, {}).get(nm)
                    if v and any(v):
                        self.alloc[(L, nm)].append(
                            (self.idx.get(("S", nm, k, L), -1), v))

    def x0(self):
        """Start from the single-line answer, with every tau at zero."""
        x = np.zeros(self.n)
        for nm, k, L in self.shares:
            x[self.idx[("S", nm, k, L)]] = self.share_prior[(nm, k, L)]
        for L in self.lines:
            stops, f = self.chains[L], self.flows[L]
            prof = {d: B.reconstruct(stops, f, d == 0, self.rev[L])
                    for d in (0, 1)}
            # build.py's levelling: no clean terminus means the anchor read the
            # junction's whole traffic and lifted both profiles by a constant,
            # and 통과인원 is what pins it back down.
            spec = self.table[L]
            if not spec["clean_end"] and spec["passing"] > 0:
                users = (prof[0][0] + prof[1][-1]
                         + sum(f.get(nm, (0, 0, 0, 0))[0]
                               + f.get(nm, (0, 0, 0, 0))[2]
                               for _, nm in stops[1:-1]))
                shift = (spec["passing"] - users) / 2.0
                prof = {d: [v + shift for v in prof[d]] for d in (0, 1)}
            # E is the first segment's load, which is what build.py just gave.
            for d in (0, 1):
                x[self.idx[("E", L, d)]] = prof[d][0] / SCALE
        return x

    def profile(self, x, L, d):
        """Segment loads for one line and direction, in millions/year."""
        stops = self.chains[L]
        f = self.flows[L]
        # 상행 trains run the other way along the same stop order, so a boarding
        # at stop i adds to the load *behind* it, not ahead. build.reconstruct
        # gets this free by walking backwards; cumulating forwards must carry
        # the sign, and dropping it makes the up profile the near-negative of
        # the down one -- which the mirror residual then flattens both to kill.
        sgn = 1.0 if d == 0 else -1.0
        # Stop 0 stays out of the cumulation in both directions, so E is the
        # load on the first segment outright. It has to be exactly that for the
        # junction constraint below to read a through flow off it.
        base, cur = [0.0], 0.0
        for _, nm in stops[1:-1]:
            b, a = self._cols(L, d, self._flow(x, L, nm))
            cur += sgn * (b - a) / SCALE
            base.append(cur)
        # Entry flow, then a step at each junction.
        out, step = [], x[self.idx[("E", L, d)]]
        for i, val in enumerate(base):
            k = self.idx.get(("T", L, stops[i][1], d))
            if k is not None and i > 0:
                step += x[k]
            out.append(val + step)
        return np.array(out)

    def _flow(self, x, L, nm):
        """This line's allocated 승하차, as (하행승, 하행하, 상행승, 상행하)."""
        b0 = b1 = b2 = b3 = 0.0
        for j, v in self.alloc.get((L, nm), ()):
            s = 1.0 if j < 0 else x[j]
            b0 += s * v[0]
            b1 += s * v[1]
            b2 += s * v[2]
            b3 += s * v[3]
        return b0, b1, b2, b3

    def _cols(self, L, d, v):
        """(boardings, alightings) for trains running the way this d does.

        d == 0 is stop 0 -> stop n along the chain, which is 하행 only when the
        chain runs 기점 -> 종점. lines.resolve() flips it to put the clean anchor
        last, and the 승하차 columns then have to be swapped to match -- 경부선,
        중앙선 and 수서고속선 are the three lines that need it, and all three are
        ones the single-line build gave up on.
        """
        i = 0 if (d == 0) != self.rev[L] else 2
        return v[i], v[i + 1]

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

            # Anchor -- only where the far end is a genuine terminus. It takes
            # the line's share of the platform, not all of it: 익산's conventional
            # alightings are 호남선's, 전라선's and 장항선's together, and handing
            # every line the whole figure anchors each of them to all three.
            if spec["clean_end"]:
                v = self._flow(x, L, stops[-1][1])
                r.append(W_ANCHOR * (down[-1] - self._cols(L, 0, v)[1] / SCALE))
                r.append(W_ANCHOR * (up[-1] - self._cols(L, 1, v)[0] / SCALE))

            # positivity
            r.extend(W_POSITIVE * np.minimum(down, 0.0))
            r.extend(W_POSITIVE * np.minimum(up, 0.0))

            # mirror
            r.extend(W_MIRROR * (down - up))

            # 통과인원. It counts everyone who touched the line's metals, while
            # the reconstruction sums only the train types that line runs, so a
            # rebuild may legitimately come in under the published figure -- but
            # never over it. That makes it a ceiling on every line without
            # exception, and leaking traffic in from a junction is precisely
            # what breaks it, so it is the sharpest test available and worth far
            # more than the one-residual-per-line it used to get against four
            # hundred-odd mirror terms.
            #
            # Equality is only expected where the line carries every train type,
            # or where it has no clean terminus and takes its level from here.
            if spec["passing"] > 0:
                interior = sum(sum(self._flow(x, L, nm)[0::2])
                               for _, nm in stops[1:-1]) / SCALE
                users = down[0] + up[-1] + interior
                over = users - spec["passing"] / SCALE
                r.append(W_CEILING * max(over, 0.0))
                if (not spec["clean_end"]
                        or set(spec["types"]) == set(LN.ALL_TYPES)):
                    r.append(W_PASSING * over)

        # Junction conservation, per direction: every line meeting there
        # contributes the through flow it gains, and the total must be nil --
        # what steps off one line steps onto another.
        #
        # A line whose *end* is the junction used to contribute nothing, which
        # left the constraint useless exactly where it was needed. 호남선 starts
        # at 대전조차장, so 경부선's tau there had nothing to balance against and
        # was driven to zero -- the very correction 경부선 needs. A terminal
        # line's through flow is the end segment's load less the passengers who
        # actually use the platform, and at a genuine terminus that is nil of
        # its own accord, which is the consistency check on the whole idea.
        # Lines have to be grouped by the direction a passenger is actually
        # travelling, not by each chain's own order -- a reversed chain's d == 0
        # is 상행, and pairing that against another line's 하행 conserves nothing.
        # 하행 continues as 하행 through a junction because Korean lines are
        # numbered outward from Seoul, and the few that are not (경전선 at
        # 삼랑진, 영동선 at 영주, 경북선 at 김천) still join the radial ones the
        # right way round.
        gain = collections.defaultdict(list)

        def add(nm, L, d, val):
            # Reversing the chain also swaps the sense of a step: forwards along
            # the chain a positive one means joining, backwards it means leaving.
            gain[(nm, d ^ self.rev[L])].append(-val if self.rev[L] else val)

        for L in self.lines:
            stops = self.chains[L]
            for nm, sign, i in ((stops[0][1], 1.0, 0), (stops[-1][1], -1.0, -1)):
                if nm not in self.shared:
                    continue
                v = self._flow(x, L, nm)
                for d in (0, 1):
                    # The platform movement that belongs to the end segment:
                    # boardings where those trains set out, alightings where the
                    # arriving ones empty.
                    plat = self._cols(L, d, v)[d if i == 0 else 1 - d]
                    through = prof[(L, d)][i] - plat / SCALE
                    # A line's through flow at its own end cannot be negative:
                    # at stop 0 it is passengers joining from another line, at
                    # stop n passengers carrying on to one, and neither can be
                    # fewer than nobody. Leaving this out let a junction balance
                    # its books against a line that has no traffic to give --
                    # 경부선 gained 3.6M at 용산 against 경원선 losing the same
                    # there, at the very station 경원선 starts from, and half of
                    # everything it carries.
                    r.append(W_POSITIVE * min(through, 0.0))
                    # And the same smallness prior the taus get. Without it a
                    # line that *ends* at a junction can absorb any amount for
                    # free, which is what licensed 경부선's step at 용산: the
                    # traffic had somewhere to go, so taking it cost nothing.
                    r.append(W_TAU * through)
                    add(nm, L, d, sign * through)
        for (L, nm, d, i) in self.taus:
            add(nm, L, d, x[self.idx[("T", L, nm, d)]])
        for key, gs in sorted(gain.items()):
            r.append(W_JUNCTION * sum(gs))

        # The two directions' steps at a junction should match: as many people
        # join the outbound trains there as leave the inbound ones. This is the
        # mirror argument applied to the interchange rather than the platform,
        # and without it the taus are the one free way left to drive 하행 and
        # 상행 apart -- which is what put 경전선 and 동해선 behind where the
        # single-line build already had them.
        for (L, nm, d, i) in self.taus:
            if d == 0:
                r.append(W_TAUSYM * (x[self.idx[("T", L, nm, 0)]]
                                     - x[self.idx[("T", L, nm, 1)]]))

        # Junction steps, against the published section counts where there are
        # any. A station the train count runs straight through is not a junction
        # at all and its step is held to zero; a station where trains genuinely
        # join or leave keeps the weak prior and lets the 승하차 set the size.
        for (L, nm, d, i) in self.taus:
            k = self.idx[("T", L, nm, d)]
            ch = self.freq.get(L)
            step = FQ.match(nm, ch) if ch is not None else None
            flat = ch is not None and (step is None or step[0] == step[1])
            r.append((W_FREQ if flat else W_TAU) * x[k])
            if not flat and step is not None:
                # The count also gives the direction. Trains leaving cannot put
                # passengers on, and 삼랑진 is 47 trains down to 43 -- 경전선
                # taking services off 경부선 -- so the +2.35M the fit wanted
                # there had the sign of the thing backwards. Both directions
                # take the same rule: forwards a positive step means joining,
                # backwards it means leaving, and either way it is the extra
                # services that carry it.
                grew = step[1] > step[0]
                r.append(W_FREQ * (min(x[k], 0.0) if grew else max(x[k], 0.0)))
        for nm, k, grp in self.groups:
            ks = [self.idx[("S", nm, k, L)] for L in grp]
            r.append(W_SHARESUM * (sum(x[j] for j in ks) - 1.0))
        for nm, k, L in self.shares:
            r.append(W_SHARE * (x[self.idx[("S", nm, k, L)]]
                                - self.share_prior[(nm, k, L)]))
        return np.array(r)

    def bounds(self):
        lo = np.full(self.n, -np.inf)
        hi = np.full(self.n, np.inf)
        for nm, k, L in self.shares:
            lo[self.idx[("S", nm, k, L)]] = 0.0
            hi[self.idx[("S", nm, k, L)]] = 1.0
        for L in self.lines:
            for d in (0, 1):
                lo[self.idx[("E", L, d)]] = 0.0
        return lo, hi


def report(net, x, only=None):
    rows = []
    for L in net.lines:
        down, up = net.profile(x, L, 0) * SCALE, net.profile(x, L, 1) * SCALE
        stops = net.chains[L]
        interior = sum(sum(net._flow(x, L, nm)[0::2]) for _, nm in stops[1:-1])
        users = down[0] + up[-1] + interior
        length = stops[-1][0]
        pkm = sum((down[i] + up[i]) * (stops[i + 1][0] - stops[i][0])
                  for i in range(len(down)))
        mirror = max(abs(down[i] - up[i]) / max(abs(down[i]), abs(up[i]), 1)
                     for i in range(len(down)))
        # The worst single segment is a poor summary: it is dominated by
        # whichever segment carries almost nobody, where a handful of
        # passengers is 100 %. 충북선's 조치원-오송 stub reads 0 against 846 --
        # two people a day -- while its other fifteen segments agree to 3 %.
        # So weight the disagreement by the traffic it applies to.
        gap = np.abs(down - up).sum()
        held = np.maximum(np.abs(down), np.abs(up)).sum()
        weighted = gap / held if held else 0.0
        # d == 0 is the chain's own order; on a reversed chain that is 상행, so
        # swap the pair back before anything is labelled 하행.
        if net.rev[L]:
            down, up = up, down
        rows.append({
            "line": L, "stops": stops, "down": down, "up": up,
            "users": users, "passing": net.table[L]["passing"],
            "mirror": mirror, "weighted": weighted, "length": length,
            "density": pkm / length / 365.0 if length else 0.0,
            "worst": min(down.min(), up.min()),
        })

    print("\n%-11s %5s %7s %11s %11s %7s %7s %9s %10s"
          % ("line", "stops", "km", "통과인원", "yearbook", "mirror", "worst",
             "수송밀도", "min load"))
    print("-" * 90)
    for r in sorted(rows, key=lambda z: -z["density"]):
        print("%-11s %5d %7.1f %11.0f %11.0f %6.1f%% %6.1f%% %9.0f %10.0f"
              % (r["line"], len(r["stops"]), r["length"], r["users"],
                 r["passing"], 100 * r["weighted"], 100 * r["mirror"],
                 r["density"], r["worst"]))

    # The junction steps, largest first. These are the part of the answer with
    # the least evidence behind them, so they are where a wrong profile shows
    # up first -- a step of a size the junction cannot plausibly pass is the
    # signature of the fit using one to absorb something else.
    steps = sorted(((abs(x[net.idx[("T", L, nm, d)]]) * SCALE, L, nm, d)
                    for (L, nm, d, i) in net.taus), reverse=True)[:12]
    if steps:
        print("\nbiggest junction steps")
        for v, L, nm, d in steps:
            signed = x[net.idx[("T", L, nm, d)]] * SCALE
            print("   %-11s %-9s %-4s %12.0f   (%.0f%% of the line)"
                  % (L, nm, "하행" if (d ^ net.rev[L]) == 0 else "상행", signed,
                     100 * v / max(net.table[L]["passing"], 1)))

    bad = [r for r in rows if r["worst"] < -1000]
    print("\n%d of %d lines still carry a negative segment"
          % (len(bad), len(rows)))
    ratios = [r["users"] / r["passing"] for r in rows if r["passing"] > 0]
    print("통과인원 ratio: median %.3f, worst %.3f / %.3f"
          % (float(np.median(ratios)), min(ratios), max(ratios)))
    print("mirror, weighted by traffic: median %.1f%%, worst %.1f%% (%s)"
          % (100 * float(np.median([r["weighted"] for r in rows])),
             100 * max(r["weighted"] for r in rows),
             max(rows, key=lambda z: z["weighted"])["line"]))

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
