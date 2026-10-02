"""Today's crediting (a port of dist/index.html's) against the ownership prototype, per test ride.

    python compare.py <tag> [--all]

--all also prints every line either model credits, not only the differences.
"""
import os
import sys

os.environ["OMP_NUM_THREADS"] = "2"
sys.dont_write_bytecode = True
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = Path(r"C:\Users\anita\projects\maps\noritetsu")
from own import app_merge, span_len, intersect, merge  # noqa: E402
from rides import RIDES  # noqa: E402

OWN_LINE_SHARE = 0.4


class Data:
    def __init__(self, tag):
        self.m = pickle.load(open(HERE / "out" / tag / "model.pkl", "rb"))
        cc = self.cc = self.m["meta"]["cc"]
        d = ROOT / "dist" / "data" / cc
        self.L = json.load(open(d / "lines.json", encoding="utf-8"))["lines"]
        self.S = json.load(open(d / "stations.json", encoding="utf-8"))["stations"]
        C = json.load(open(d / "credits.json", encoding="utf-8"))
        self.covers = {int(k): v for k, v in C["covers"].items()}
        self.covered_by = defaultdict(list)
        for g, v in self.covers.items():
            for b, lo, hi in v:
                self.covered_by[b].append([g, lo, hi])
        self.byid = {l["id"]: l for l in self.L}
        self.closed = set()
        self.sec = {}
        for l in self.L:
            shut = set(l.get("closed") or [])
            off = 0.0
            for a, b, km, gid in l["sections"]:
                self.sec[gid] = (l, a, b, km)
                if f"{a}|{b}" in shut:
                    self.closed.add(gid)
                    off += km
            l["_km"] = max(0.0, l["km"] - off)       # the app's line.km
        self.reg = lambda l: l.get("src", "osm") != "osm"
        # scope (slices): sections with geometry in the model
        self.inscope = set(self.m["secs"])
        self.inside = {g for g, s in self.m["secs"].items() if s["inside"]}
        self.only = self.inside if self.m["meta"]["bbox"] else None
        self._g = {}

    def ok(self, g):
        return g not in self.closed and (self.only is None or g in self.only)

    # ------------------------------------------------------------ rides -> sections
    def graph(self, l):
        if l["id"] not in self._g:
            g = defaultdict(list)
            for a, b, km, gid in l["sections"]:
                g[a].append((b, km, gid))
                g[b].append((a, km, gid))
            self._g[l["id"]] = g
        return self._g[l["id"]]

    def path(self, l, a, b):
        import heapq
        g = self.graph(l)
        dist, prev, heap = {a: 0.0}, {}, [(0.0, a)]
        seen = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in seen:
                continue
            if u == b:
                break
            seen.add(u)
            for v, km, gid in g.get(u, ()):
                nd = d + km
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    prev[v] = (u, gid)
                    heapq.heappush(heap, (nd, v))
        if b not in prev and a != b:
            return []
        out, cur = [], b
        while cur != a:
            u, gid = prev[cur]
            out.append(gid)
            cur = u
        return out[::-1]

    def ride_gids(self, lid, a, b):
        l = self.byid[lid]
        if a is None:
            gids = [s[3] for s in l["sections"]]
        else:
            gids = self.path(l, a, b)
        return [g for g in gids if self.only is None or g in self.only]


# ---------------------------------------------------------------- today's model (the app)
class Old:
    def __init__(self, D):
        self.D = D
        self.own = {}
        self.own_track()
        self.countable_ids = {l["id"] for l in D.L if self.countable(l)}
        self.uniq = self.unique_shares([l for l in D.L if not D.reg(l) and l["id"] in self.countable_ids])

    def on_register(self, a, b, km):
        D = self.D
        tol = 0.15 * km + 0.5
        on_b = set(D.S.get(b, {}).get("l", []))
        for lid in D.S.get(a, {}).get("l", []):
            l = D.byid.get(lid)
            if lid not in on_b or not l or not D.reg(l):
                continue
            g = D.graph(l)
            if a not in g or b not in g:
                continue
            dist, seen = {a: 0.0}, set()
            while True:
                u, best = None, math.inf
                for k, d in dist.items():
                    if k not in seen and d < best:
                        best, u = d, k
                if u is None or best > km + tol:
                    break
                if u == b:
                    if abs(best - km) <= tol:
                        return True
                    break
                seen.add(u)
                for v, w, _g in g.get(u, ()):
                    nd = best + w
                    if nd < dist.get(v, math.inf):
                        dist[v] = nd
        return False

    def own_track(self):
        D = self.D
        reg = {s[3] for l in D.L if D.reg(l) for s in l["sections"]}
        for l in D.L:
            if D.reg(l) or l["service"]:
                continue
            for a, b, km, gid in l["sections"]:
                if gid in D.closed or not km > 0:
                    self.own[gid] = 0.0
                    continue
                fwd = span_len(app_merge([[lo, hi] for x, lo, hi in D.covered_by.get(gid, []) if x in reg], km))
                rev = sum((hi - lo) * D.sec[x][3] for x, lo, hi in D.covers.get(gid, []) if x in reg)
                own = max(0.0, min(1 - fwd, 1 - rev / km))
                if own > 0 and reg and self.on_register(a, b, km):
                    own = 0.0
                self.own[gid] = own

    def own_km(self, l):
        return sum(km * self.own.get(gid, 0.0) for a, b, km, gid in l["sections"] if gid not in self.D.closed)

    def countable(self, l):
        if self.D.reg(l):
            return True
        if l["service"]:
            return False
        o = self.own_km(l)
        return o > 0.05 and o >= OWN_LINE_SHARE * l["_km"]

    def unique_shares(self, parts):
        D = self.D
        secs = [s for p in parts for s in p["sections"]
                if s[3] not in D.closed and self.own.get(s[3], 0) > 0 and s[2] > 0]
        secs.sort(key=lambda s: (s[2], s[3]))
        out = {}
        for a, b, km, gid in secs:
            if gid in out:
                continue
            fwd = span_len(app_merge([[lo, hi] for x, lo, hi in D.covered_by.get(gid, [])
                                      if x in out and self.own.get(x, 0) > 0.999], km))
            rev = 0.0
            for x, lo, hi in D.covers.get(gid, []):
                u = out.get(x)
                if u:
                    rev += (hi - lo) * D.sec[x][3] * u
            out[gid] = max(0.0, self.own[gid] - max(fwd, rev / km))
        return out

    def credit(self, gids):
        spans = defaultdict(list)
        for g in gids:
            spans[g].append([0.0, 1.0])
            for b, lo, hi in self.D.covers.get(g, []):
                if hi > lo:
                    spans[b].append([lo, hi])
        frac = {}
        for g, iv in spans.items():
            frac[g] = span_len(app_merge(iv, self.D.sec[g][3]))
        return frac

    def line_km(self, l, frac):
        km = sum(k * frac.get(g, 0.0) for a, b, k, g in l["sections"] if self.D.ok(g))
        return min(km, l["_km"])

    def totals(self, frac, only=None):
        D = self.D
        tot = done = 0.0
        for l in D.L:
            if l["id"] not in self.countable_ids:
                continue
            for a, b, km, g in l["sections"]:
                if g in D.closed or (only is not None and g not in only):
                    continue
                u = 1.0 if D.reg(l) else self.uniq.get(g, 0.0)
                tot += km * u
                done += km * u * min(1.0, frac.get(g, 0.0))
        return tot, done


# ---------------------------------------------------------------- the ownership prototype
class New:
    def __init__(self, D):
        self.D = D
        m = D.m
        self.foot, self.own, self.secs, self.comp = m["foot"], m["own"], m["secs"], m["comp"]

    def credit(self, gids):
        sp = defaultdict(list)
        for g in gids:
            for t, lo, hi in self.foot.get(g, []):
                sp[t].append([lo, hi])
        return {t: app_merge(iv, self.secs[t]["km"]) for t, iv in sp.items()}, set(gids)

    def line_km(self, l, ridden):
        rid, direct = ridden
        D = self.D
        if D.reg(l):
            km = sum(k * span_len(rid.get(g, [])) for a, b, k, g in l["sections"] if D.ok(g))
            return min(km, l["_km"])
        km = 0.0
        for a, b, k, g in l["sections"]:
            if not D.ok(g):
                continue
            if g in direct:
                km += k
                continue
            f = self.foot.get(g)
            if not f:
                continue
            c = self.comp.get(g, {})
            glen = sum(c.values()) or 1.0
            unowned = sum(v for s, v in c.items() if s not in ("reg", "osm", "reg-throat", "reg-twin")) / glen
            tot = cov = 0.0
            for t, lo, hi in f:
                kt = self.secs[t]["km"]
                tot += kt * (hi - lo)
                cov += kt * intersect(rid.get(t, []), [[lo, hi]])
            if tot > 0:
                km += k * (1 - unowned) * min(1.0, cov / tot)
        return min(km, l["_km"])

    def totals(self, ridden, only=None):
        rid, _d = ridden
        tot = done = 0.0
        for t, sp in self.own.items():
            if only is not None and t not in only:
                continue
            kt = self.secs[t]["km"]
            tot += kt * span_len(sp)
            done += kt * intersect(rid.get(t, []), sp)
        return tot, done


def main():
    tag = sys.argv[1]
    show_all = "--all" in sys.argv
    D = Data(tag)
    old, new = Old(D), New(D)
    only = D.inside if D.m["meta"]["bbox"] else None
    ot, _ = old.totals({}, only)
    nt, _ = new.totals(({}, set()), only)
    print(f"== {tag}: country total, today {ot:,.1f} km, ownership {nt:,.1f} km")
    reg_km = sum(km for l in D.L if D.reg(l) for a, b, km, g in l["sections"] if D.ok(g))
    print(f"   register lines {reg_km:,.1f} km; today's OSM own track counted {ot - reg_km:,.1f} km; "
          f"ownership's OSM-owned {nt - reg_km:,.1f} km")
    # what each OSM line adds to the country total, today (countable + unique share) and owned
    per_old, per_new = defaultdict(float), defaultdict(float)
    for l in D.L:
        if D.reg(l) or l["id"] not in old.countable_ids:
            continue
        for a, b, km, g in l["sections"]:
            if g not in D.closed and (only is None or g in only):
                per_old[l["id"]] += km * old.uniq.get(g, 0.0)
    for t, sp in new.own.items():
        s = new.secs[t]
        if s["reg"] or (only is not None and t not in only):
            continue
        per_new[s["line"]] += s["km"] * span_len(sp)
    diffs = sorted(set(per_old) | set(per_new), key=lambda k: -abs(per_new[k] - per_old[k]))
    print("   OSM lines whose counted km differ by over 1 km (today -> ownership):")
    for k in diffs:
        if abs(per_new[k] - per_old[k]) <= 1.0:
            break
        l = D.byid[k]
        print(f"     {per_old[k]:7.2f} -> {per_new[k]:7.2f} of {l['_km']:6.1f}  {k} {l.get('ref') or ''} "
              f"{l['name'][:50]} [{l['kind']}]{'' if k in old.countable_ids else ' (not countable today)'}")
    if "--lines-only" in sys.argv:
        return
    allg = []
    for label, lid, a, b in RIDES.get(tag, RIDES.get(D.cc, [])):
        gids = D.ride_gids(lid, a, b)
        allg += gids
        fo = old.credit(gids)
        fn = new.credit(gids)
        tl = D.byid[lid]
        rk = sum(D.sec[g][3] for g in gids)
        o_t = old.totals(fo, only)[1]
        n_t = new.totals(fn, only)[1]
        print(f"\n-- {label}: {len(gids)} sections, {rk:.1f} km ridden on {tl.get('ref') or tl['name']}")
        print(f"   adds to the country total: today {o_t:7.1f} km   ownership {n_t:7.1f} km   diff {n_t - o_t:+.1f}")
        rows = []
        for l in D.L:
            ko, kn = old.line_km(l, fo), new.line_km(l, fn)
            if max(ko, kn) < 0.3:
                continue
            rows.append((l, ko, kn))
        rows.sort(key=lambda r: (not D.reg(r[0]), -max(r[1], r[2])))
        for l, ko, kn in rows:
            flag = "  <==" if abs(ko - kn) > 1.0 else ""
            if not (show_all or flag or l["id"] == lid):
                continue
            kind = "REG" if D.reg(l) else ("svc" if l["service"] else "osm")
            cnt = "" if l["id"] in old.countable_ids else " (not counted today)"
            print(f"     {kind} {ko:7.2f} | {kn:7.2f} of {l['_km']:7.1f}  {l.get('ref') or ''} {l['name'][:55]}{cnt}{flag}")
    fo, fn = old.credit(allg), new.credit(allg)
    print(f"\n== all rides together: today {old.totals(fo, only)[1]:,.1f} km of {ot:,.1f}; "
          f"ownership {new.totals(fn, only)[1]:,.1f} km of {nt:,.1f}")


if __name__ == "__main__":
    main()
