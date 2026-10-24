"""T-1163: trips a destination near a zone line gets from its own pack and from the neighbour's,
by purpose and distance band, now (everyday drawn over the home's own pack only, commutes across
the line at the shipped factor) against a seamless rule (everyday drawn over every listed pack's
zones; commutes across the line at factor 1, re-balanced over the listed packs).

    python border.py A B [more packs...] [--near KM]

Zone level (the sim's everyday draw is zone to zone, cells within a zone by attraction), read-only.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import json, math, sys
import numpy as np

PACKS = "C:/Users/anita/projects/maps/trainworld/data/packs"
DT = {"u64": "<u8", "f32": "<f4", "u32": "<u4", "i32": "<i4", "f64": "<f8", "u16": "<u2", "u8": "u1"}
R = 6371008.8
DECAY = {"errands": 1.0, "social": 1.0, "school": 1.4, "university": 3.5, "medical": 2.0}
RATE_DEFAULT = {"errands": 0.3, "social": 0.2, "school": 0.9, "university": 0.6, "medical": 0.04}
CENTRAL_SHARE, CENTRAL_KM, CELL_KM2 = 0.2, 6.0, 0.1053
BANDS = [0, 2, 5, 10, 25, 1e9]
BAND_NAMES = ["0-2", "2-5", "5-10", "10-25", "25+"]


def load(p):
    h = json.load(open(f"{PACKS}/{p}.json", encoding="utf-8"))
    buf = open(f"{PACKS}/{p}.bin", "rb").read()
    a = {}
    for x in h["arrays"]:
        if x["dtype"] in DT:
            a[x["name"]] = np.frombuffer(buf, dtype=np.dtype(DT[x["dtype"]]), count=x["count"], offset=x["offset"]).astype(np.float64)
    return h, a


class Pack:
    def __init__(self, pid):
        self.id = pid
        h, a = load(pid)
        self.h, self.a = h, a
        g = h["gravity"]
        self.g = g
        lon0, lat0 = h["origin"]["lon"], h["origin"]["lat"]
        self.lon0, self.lat0 = lon0, lat0
        nz = h["zones"]
        self.nz = nz
        zone = a["zone"].astype(np.int64)
        self.zone = zone
        side = a["zone_size"] if "zone_size" in a else np.ones(nz)
        zm = g["zone_m"]
        x0, y0 = g["grid_origin_m"]
        zx = x0 + (a["zone_gx"] + side / 2) * zm
        zy = y0 + (a["zone_gy"] + side / 2) * zm
        self.side = side
        # zone centres to lon/lat (the pack's equirectangular frame)
        self.zlon = lon0 + np.degrees(zx / (R * math.cos(math.radians(lat0))))
        self.zlat = lat0 + np.degrees(zy / R)
        self.ncell = np.bincount(zone, minlength=nz)
        sq = (side * zm / 1000) ** 2
        self.land = np.minimum(np.maximum(self.ncell, 1) * CELL_KM2, sq)
        zs = lambda v: np.bincount(zone, weights=v, minlength=nz)
        self.pop = zs(a["pop"])
        ev = h.get("everyday", {}).get("rate", {})
        self.rate = {k: ev.get(k, RATE_DEFAULT[k]) for k in DECAY}
        tp = a["pop"].sum()
        tl = a["leisure_jobs"].sum()
        park = a.get("park_draw")
        tk = park.sum() if park is not None else 0
        ps = 0.08 if tk > 0 else 0.0
        # attraction per zone (raw units; social split into its parts so a union can renormalise)
        self.att = {"errands": zs(a["errand_jobs"]), "medical": zs(a["medical_visits"]) if "medical_visits" in a else np.zeros(nz)}
        self.social_parts = (zs(a["pop"]), zs(a["leisure_jobs"]), zs(park) if tk > 0 else np.zeros(nz), ps)
        self.att["social"] = (0.5 * (1 - ps) * self.social_parts[0] / tp + 0.5 * (1 - ps) * self.social_parts[1] / tl
                              + (ps * self.social_parts[2] / tk if tk > 0 else 0))
        # origins (budgets): outings per resident; school/university are drawn from the site
        self.orig = {"errands": self.pop, "social": self.pop, "medical": self.pop}
        self.school = (zs(a["school_enrollment"]), zs(a["school_children"])) if "school_enrollment" in a else None
        self.univ = (zs(a["university_enrollment"]), zs(a["university_adults"])) if "university_enrollment" in a else None
        self.row, self.col = a["zone_row"], a["zone_col"]
        self.zhome, self.zwork = a.get("zone_home"), a.get("zone_work")

    def subset(self, keep):
        """Only these zones (every per-zone array cut to them)."""
        for k in ("zlon", "zlat", "X", "Y", "ncell", "land", "pop", "side", "row", "col"):
            setattr(self, k, getattr(self, k)[keep])
        self.att = {k: v[keep] for k, v in self.att.items()}
        self.social_parts = tuple(v[keep] if isinstance(v, np.ndarray) else v for v in self.social_parts)
        self.orig = {k: v[keep] for k, v in self.orig.items()}
        self.school = tuple(v[keep] for v in self.school) if self.school else None
        self.univ = tuple(v[keep] for v in self.univ) if self.univ else None
        gx, gy = self.a["zone_gx"][keep], self.a["zone_gy"][keep]
        self.a = dict(self.a, zone_gx=gx, zone_gy=gy)
        self.nz = len(keep)


def frame(packs):
    lat0 = np.mean([p.lat0 for p in packs])
    lon0 = np.mean([p.lon0 for p in packs])
    for p in packs:
        p.X = R * np.radians(p.zlon - lon0) * math.cos(math.radians(lat0)) / 1000
        p.Y = R * np.radians(p.zlat - lat0) / 1000


def dist(p, q):
    d = np.hypot(p.X[:, None] - q.X[None, :], p.Y[:, None] - q.Y[None, :])
    if p is q:
        np.fill_diagonal(d, 0.52 * 2.0 * p.side)
    return d


def commute_f(g, d):
    flat = g.get("decay_flat_km", 0.0)
    dd = np.maximum(d, flat)
    dd = np.maximum(dd, 1e-3)
    return dd ** -g.get("decay_pow", 0.0) * np.exp(-dd / g["decay_km"])


def main():
    args = sys.argv[1:]
    near = 3.0
    factor = 0.4
    if "--factor" in args:
        i = args.index("--factor")
        factor = float(args[i + 1])
        del args[i:i + 2]
    if "--near" in args:
        i = args.index("--near")
        near = float(args[i + 1])
        del args[i:i + 2]
    clip = None
    if "--clip" in args:
        i = args.index("--clip")
        clip = float(args[i + 1])
        del args[i:i + 2]
    packs = [Pack(p) for p in args]
    frame(packs)
    if clip:
        # keep only zones within `clip` km of another listed pack (a big pack's far side draws
        # next to nothing across the line at these decay lengths)
        for p in packs:
            others = [q for q in packs if q is not p]
            dmin = np.min([np.hypot(p.X[:, None] - q.X[None, :], p.Y[:, None] - q.Y[None, :]).min(axis=1) for q in others], axis=0)
            keep = np.nonzero(dmin <= clip)[0]
            p.subset(keep)
    A, B = packs[0], packs[1]
    D = {}
    for p, q in ((B, A), (A, B)):
        dmin = dist(p, q).min(axis=1)
        D[p.id] = np.nonzero((dmin <= near) & (p.pop + p.att["errands"] > 0))[0]
    print(f"destination zones within {near} km of the other pack: {A.id} {len(D[A.id])}, {B.id} {len(D[B.id])}")
    dists = {(p.id, q.id): dist(p, q) for p in packs for q in packs}
    bandof = lambda d: np.digitize(d, BANDS) - 1

    # union normalisation of social: the parts' shares over every listed pack together
    U = [sum(p.social_parts[k].sum() for p in packs) for k in range(3)]
    ps = packs[0].social_parts[3]
    soc_union = {p.id: 0.5 * (1 - ps) * p.social_parts[0] / U[0] + 0.5 * (1 - ps) * p.social_parts[1] / U[1]
                 + (ps * p.social_parts[2] / U[2] if U[2] > 0 else 0) for p in packs}

    def everyday(kind, union):
        """arrivals[(dest pack)][origin pack] = matrix origin zone x dest zone of outings a day."""
        out = {}
        s = DECAY[kind]
        att = {p.id: (soc_union[p.id] if (kind == "social" and union) else p.att[kind]) for p in packs}
        for o in packs:
            qs = packs if union else [o]
            w = {q.id: att[q.id] * np.exp(-dists[(o.id, q.id)] / s) for q in qs}
            tot = sum(v.sum(axis=1) for v in w.values())
            if kind in ("errands", "social"):
                cw = {q.id: (att[q.id] ** 2 / q.land) * np.exp(-dists[(o.id, q.id)] / CENTRAL_KM) for q in qs}
                ctot = sum(v.sum(axis=1) for v in cw.values())
            budget = o.orig[kind] * o.rate[kind]
            for q in qs:
                m = w[q.id] / np.where(tot > 0, tot, 1)[:, None]
                if kind in ("errands", "social"):
                    m = (1 - CENTRAL_SHARE) * m + CENTRAL_SHARE * cw[q.id] / np.where(ctot > 0, ctot, 1)[:, None]
                out.setdefault(q.id, {})[o.id] = m * budget[:, None]
        return out

    def site(kind, union):
        """school/university: each site zone's enrolment drawn over home zones (site-anchored)."""
        out = {}
        s = DECAY[kind]
        for q in packs:  # q holds the site
            t = q.school if kind == "school" else q.univ
            if t is None:
                continue
            enr = t[0] * q.rate[kind]
            os_ = packs if union else [q]
            w = {}
            for o in os_:
                to = o.school if kind == "school" else o.univ
                ch = to[1] if to is not None else np.zeros(o.nz)
                w[o.id] = ch[:, None] * np.exp(-dists[(o.id, q.id)] / s)
            tot = sum(v.sum(axis=0) for v in w.values())
            for o in os_:
                out.setdefault(q.id, {})[o.id] = w[o.id] / np.where(tot > 0, tot, 1)[None, :] * enr[None, :]
        return out

    def commutes(factor):
        """commuters home zone -> work zone among the listed packs: within a pack row x col x f,
        across packs row x col x f x factor (shipped factors); for a factor other than the shipped
        one, re-balanced over the listed packs with their flows to other packs held fixed."""
        T = {}
        for o in packs:
            for q in packs:
                f = commute_f(o.g, dists[(o.id, q.id)])
                if o is not q:
                    f = f * factor
                    f = np.where(dists[(o.id, q.id)] <= 150, f, 0)
                T[(o.id, q.id)] = o.row[:, None] * q.col[None, :] * f
        return T

    def rebalance(T0, factor0, factor1, steps=60):
        # targets: each zone's sums among the listed packs as shipped
        rt = {o.id: sum(T0[(o.id, q.id)].sum(axis=1) for q in packs) for o in packs}
        ct = {q.id: sum(T0[(o.id, q.id)].sum(axis=0) for o in packs) for q in packs}
        T = {k: (v * (factor1 / factor0) if k[0] != k[1] else v.copy()) for k, v in T0.items()}
        for _ in range(steps):
            for o in packs:
                s = sum(T[(o.id, q.id)].sum(axis=1) for q in packs)
                r = np.where(s > 0, rt[o.id] / np.where(s > 0, s, 1), 0)
                for q in packs:
                    T[(o.id, q.id)] *= r[:, None]
            for q in packs:
                s = sum(T[(o.id, q.id)].sum(axis=0) for o in packs)
                c = np.where(s > 0, ct[q.id] / np.where(s > 0, s, 1), 0)
                for o in packs:
                    T[(o.id, q.id)] *= c[None, :]
        return T

    T_now = commutes(factor)
    T_one = rebalance(T_now, factor, 1.0) if factor != 1.0 else T_now
    # check the cross-pack part against the shipped joint file
    tot_ab = T_now[(A.id, B.id)].sum() + T_now[(B.id, A.id)].sum()
    print(f"commuters {A.id} <-> {B.id} by the formula at {factor}: {tot_ab:,.0f} a day (both ways)")

    def report(dest, label, mats):
        """mats[origin pack] = origin x dest zone matrix; arrivals at the dest pack's D zones."""
        Dz = D[dest.id]
        res = {}
        for o in packs:
            m = mats.get(o.id)
            if m is None:
                continue
            sub = m[:, Dz]
            d = dists[(o.id, dest.id)][:, Dz]
            b = bandof(d)
            res[o.id] = np.array([sub[b == k].sum() for k in range(len(BAND_NAMES))])
        return res

    def per_resident(dest, o):
        """people of pack o by distance band from the nearest D zone of dest (for trips per resident)."""
        d = dists[(o.id, dest.id)][:, D[dest.id]].min(axis=1)
        b = bandof(d)
        return np.array([o.pop[b == k].sum() for k in range(len(BAND_NAMES))])

    summary = {}
    for dest, other in ((B, A), (A, B)):
        print(f"\n=== destinations: {dest.id} zones within {near} km of {other.id} ({len(D[dest.id])} zones, "
              f"{dest.pop[D[dest.id]].sum():,.0f} people, {dest.att['errands'][D[dest.id]].sum():,.0f} errand jobs)")
        print(f"   origins' people by distance band to the nearest of them: "
              + ", ".join(f"{o.id} " + "/".join(f"{v/1e3:,.0f}k" for v in per_resident(dest, o)) for o in (dest, other)))
        print(f"   {'purpose':<11}{'from':<6}{'rule':<6}" + "".join(f"{n:>10}" for n in BAND_NAMES) + f"{'all':>11}  share from {other.id}")
        tot = {"now": {dest.id: np.zeros(5), other.id: np.zeros(5)}, "fix": {dest.id: np.zeros(5), other.id: np.zeros(5)}}
        rows = []
        for kind in ("errands", "social", "medical", "school", "university"):
            for rule, union in (("now", False), ("fix", True)):
                arr = (site(kind, union) if kind in ("school", "university") else everyday(kind, union)).get(dest.id, {})
                r = report(dest, kind, arr)
                for o in (dest, other):
                    v = r.get(o.id, np.zeros(5))
                    tot[rule][o.id] += v
                    rows.append((kind, o.id, rule, v))
                sh = r.get(other.id, np.zeros(5)).sum() / max(1e-9, sum(x.sum() for x in r.values()))
                for o in (dest, other):
                    v = r.get(o.id, np.zeros(5))
                    print(f"   {kind:<11}{o.id:<6}{rule:<6}" + "".join(f"{x:>10,.0f}" for x in v) + f"{v.sum():>11,.0f}" + (f"  {sh:.1%}" if o is other else ""))
        for rule, T in (("now", T_now), ("fix", T_one)):
            r = report(dest, "commute", {o.id: T[(o.id, dest.id)] for o in packs})
            sh = r[other.id].sum() / max(1e-9, r[other.id].sum() + r[dest.id].sum())
            for o in (dest, other):
                v = r[o.id]
                tot[rule][o.id] += v
                print(f"   {'commute':<11}{o.id:<6}{rule:<6}" + "".join(f"{x:>10,.0f}" for x in v) + f"{v.sum():>11,.0f}" + (f"  {sh:.1%}" if o is other else ""))
        for rule in ("now", "fix"):
            a, b = tot[rule][dest.id], tot[rule][other.id]
            print(f"   {'ALL':<11}{dest.id:<6}{rule:<6}" + "".join(f"{x:>10,.0f}" for x in a) + f"{a.sum():>11,.0f}")
            print(f"   {'ALL':<11}{other.id:<6}{rule:<6}" + "".join(f"{x:>10,.0f}" for x in b) + f"{b.sum():>11,.0f}  {b.sum() / (a.sum() + b.sum()):.1%}")
            pr_d, pr_o = per_resident(dest, dest), per_resident(dest, other)
            print(f"   per 1,000 residents of the band, {rule}: " + "  ".join(
                f"{n}: {dest.id} {1e3 * a[k] / max(pr_d[k], 1):.1f} {other.id} {1e3 * b[k] / max(pr_o[k], 1):.1f}" for k, n in enumerate(BAND_NAMES[:4])))
        summary[dest.id] = {r: {k: v.tolist() for k, v in tot[r].items()} for r in tot}

    # everyday trips across the whole line, now none; with the seamless rule
    print("\n=== everyday outings a day across the line (both directions), seamless rule")
    grand = 0
    for kind in ("errands", "social", "medical"):
        arr = everyday(kind, True)
        x = arr[B.id][A.id].sum() + arr[A.id][B.id].sum()
        within = arr[A.id][A.id].sum() + arr[B.id][B.id].sum()
        grand += x
        print(f"   {kind:<10} across {x:>12,.0f}   within both {within:>14,.0f}   ({x / (x + within):.2%})")
    for kind in ("school", "university"):
        arr = site(kind, True)
        x = arr.get(B.id, {}).get(A.id, np.zeros(1)).sum() + arr.get(A.id, {}).get(B.id, np.zeros(1)).sum()
        grand += x
        print(f"   {kind:<10} across {x:>12,.0f}")
    print(f"   total {grand:,.0f} outings ({2 * grand:,.0f} legs); commutes across now {tot_ab:,.0f}, at 1.0 "
          f"{T_one[(A.id, B.id)].sum() + T_one[(B.id, A.id)].sum():,.0f}")
    # the cross-line outings as the regional layer would carry them: pairs of 6 km zones (each
    # pack's own frame, as the joint commutes file), kept at 1 outing a day or more
    zm = 6000.0
    def six(p):
        x0, y0 = p.g["grid_origin_m"]
        zx = x0 + (p.a["zone_gx"] + p.side / 2) * p.g["zone_m"]
        zy = y0 + (p.a["zone_gy"] + p.side / 2) * p.g["zone_m"]
        return np.floor(zx / zm).astype(np.int64) * 100000 + np.floor(zy / zm).astype(np.int64)
    sx = {p.id: six(p) for p in packs}
    allk = {}
    for kind in ("errands", "social", "medical"):
        arr = everyday(kind, True)
        for o, q in ((A, B), (B, A)):
            m = arr[q.id][o.id]
            ii, jj = np.nonzero(m > 1e-6)
            key = sx[o.id][ii] * 10**10 + sx[q.id][jj]
            k2 = (o.id, q.id)
            s = allk.setdefault(k2, {})
            u, inv = np.unique(key, return_inverse=True)
            v = np.bincount(inv, weights=m[ii, jj])
            for kk, vv in zip(u.tolist(), v.tolist()):
                s[kk] = s.get(kk, 0) + vv
    for k2, s in allk.items():
        v = np.array(list(s.values()))
        print(f"   6 km pairs {k2[0]}->{k2[1]}: {len(v):,} with any, {int((v >= 1).sum()):,} at 1 a day or more holding {v[v >= 1].sum() / v.sum():.1%} of {v.sum():,.0f} outings")

    # how far those cross-line outings go, and how deep into the other pack they end
    depth_edges = [0, 3, 5, 10, 1e9]
    depth_names = ["0-3", "3-5", "5-10", "10+"]
    for kind in ("errands", "social", "medical"):
        arr = everyday(kind, True)
        for o, q in ((A, B), (B, A)):
            m = arr[q.id][o.id]
            d = dists[(o.id, q.id)]
            b = bandof(d)
            depth = dist(q, o).min(axis=1)  # each dest zone's distance to the nearest zone of the home pack
            db = np.digitize(depth, depth_edges) - 1
            col = m.sum(axis=0)
            print(f"   {kind} {o.id}->{q.id} by trip length: " + ", ".join(f"{n} {m[b == k].sum():,.0f}" for k, n in enumerate(BAND_NAMES))
                  + " | by depth of the destination: " + ", ".join(f"{n} {col[db == k].sum() / max(col.sum(), 1e-9):.0%}" for k, n in enumerate(depth_names)))

main()
