"""T-171: the work gravity's border factor against LODES, and alternatives to it. Read-only on the
packs (solves copies in numpy, as t162_fit.py; LODES is the yardstick only).

    ..\\venv\\Scripts\\python.exe trainworld/notes/t171_gravity.py [nyc phl wdc bos pvd] [--variants a,b,..]

Per pack: the shipped curve d^-0.8 exp(-d/30 km) with
  none      no factor
  k         one factor between states (fitted per pack to LODES's between-state share; 0.23 shipped)
  pair      one factor per state pair (each fitted to LODES's share for that pair)
  kd        k(d) = k_far + (1 - k_far) exp(-d / d0) between states: weaker near the line
  metro     factor only between zones not in the same urban centre (GHSL-style: 1 km squares of
            1,500+ residents per km2, joined edge to edge, 50k+ people), else as k
  water     no state factor; a factor exp(-g * km of open water on the straight line) instead
Reports: between-state share, county pairs misplaced (t162's score), trip length misplaced, own
county, the groups below (model share of the home group's commuters over LODES's), and crossing
trips by distance band (model / LODES).
"""
import os
os.environ["OMP_NUM_THREADS"] = "3"

import sys
import time

import numpy as np
import pandas as pd
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import t162_fit as tf  # noqa: E402
from read_pack import load_pack, water_mask  # noqa: E402
import build_city as bc  # noqa: E402

PW, KM = 0.8, 30.0
BANDS = [0, 3, 6, 10, 15, 25, 40, 1000]


def groups(city, b):
    """A group label per block (None = not tabulated separately; the county name is used)."""
    g = b["geoid"]
    cty = g.str[:5]
    lat, lon = b["lat"].to_numpy(), b["lon"].to_numpy()
    lab = np.array(["other"] * len(b), dtype=object)
    if city == "nyc":
        xw = pd.read_csv(os.path.join(bc.RAW, "nj_xwalk.csv.gz"), dtype=str, usecols=["tabblk2020", "stplcname"])
        place = g.map(dict(zip(xw["tabblk2020"], xw["stplcname"].fillna("")))).fillna("").to_numpy()
        hud = (cty == "34017").to_numpy()
        hob = hud & np.char.startswith(place.astype(str), "Hoboken")
        jc = hud & np.char.startswith(place.astype(str), "Jersey City")
        lab[hud] = "Hudson rest"
        lab[hud & (lat >= 40.765)] = "North Hudson"
        lab[hob | (jc & (lon > -74.055))] = "Hudson waterfront"
        lab[(cty == "34013").to_numpy()] = "Essex"
        for c, n in (("34003", "Bergen"), ("34039", "Union"), ("34031", "Passaic")):
            lab[(cty == c).to_numpy()] = n
        far = cty.isin(["34019", "34023", "34025", "34027", "34029", "34035", "34037"]).to_numpy()
        lab[far] = "NJ outer"
        man = (cty == "36061").to_numpy()
        lab[man] = "Manhattan rest"
        lab[man & (lat < 40.722)] = "Lower Manhattan"
        lab[man & (lat >= 40.745) & (lat < 40.768)] = "Midtown"
        for c, n in (("36047", "Kings"), ("36081", "Queens"), ("36005", "Bronx"), ("36085", "Richmond"),
                     ("36119", "Westchester"), ("36087", "Rockland"), ("09001", "Fairfield CT"), ("36059", "Nassau")):
            lab[(cty == c).to_numpy()] = n
    elif city == "phl":
        xw = pd.read_csv(os.path.join(bc.RAW, "nj_xwalk.csv.gz"), dtype=str, usecols=["tabblk2020", "stplcname"])
        place = g.map(dict(zip(xw["tabblk2020"], xw["stplcname"].fillna("")))).fillna("").astype(str).to_numpy()
        lab[(cty == "42101").to_numpy()] = "Philadelphia rest"
        lab[(lat >= 39.9415) & (lat <= 39.9585) & (lon >= -75.1795) & (lon <= -75.1390) & (cty == "42101").to_numpy()] = "Center City"
        lab[(cty == "34007").to_numpy()] = "Camden Co rest"
        lab[pd.Series(place).str.startswith("Camden city").to_numpy()] = "Camden city"
        for c, n in (("34005", "Burlington"), ("34015", "Gloucester"), ("42045", "Delaware Co"), ("42091", "Montgomery"),
                     ("42017", "Bucks"), ("10003", "New Castle DE"), ("42029", "Chester")):
            lab[(cty == c).to_numpy()] = n
    elif city == "wdc":
        lab[(cty == "11001").to_numpy()] = "DC"
        lab[cty.isin(["24031", "24033"]).to_numpy()] = "MD inner (Montgomery, PG)"
        lab[cty.isin(["51013", "51510", "51059", "51600", "51610"]).to_numpy()] = "VA inner (Arl, Alex, Fairfax)"
        lab[(lab == "other") & (g.str[:2] == "24").to_numpy()] = "MD outer"
        lab[(lab == "other") & (g.str[:2] == "51").to_numpy()] = "VA outer"
    else:
        lab[:] = g.str[:2].map({"25": "MA", "44": "RI", "33": "NH"}).fillna("other").to_numpy()
    return lab


ROWS = {
    "nyc": [("Hudson waterfront", "Lower Manhattan"), ("Hudson waterfront", "Midtown"), ("Hudson waterfront", "MAN"),
            ("North Hudson", "MAN"), ("Hudson rest", "MAN"), ("Essex", "Lower Manhattan"), ("Essex", "Midtown"), ("Essex", "MAN"),
            ("Bergen", "MAN"), ("Union", "MAN"), ("Passaic", "MAN"), ("NJ outer", "MAN"), ("Fairfield CT", "MAN"),
            ("Fairfield CT", "Westchester"), ("Rockland", "Bergen"), ("Richmond", "Union"), ("Lower Manhattan", "Hudson waterfront"),
            ("Midtown", "Hudson waterfront"), ("Manhattan rest", "Hudson waterfront"), ("Kings", "MAN"), ("Queens", "MAN"),
            ("Westchester", "MAN"), ("Hudson waterfront", "Hudson waterfront")],
    "phl": [("Center City", "Camden city"), ("Center City", "Camden Co rest"), ("Philadelphia rest", "Camden Co rest"),
            ("Camden city", "Center City"), ("Camden Co rest", "Center City"), ("Camden Co rest", "PHL"),
            ("Burlington", "PHL"), ("Gloucester", "PHL"), ("New Castle DE", "Delaware Co"), ("Philadelphia rest", "Montgomery"),
            ("Center City", "Center City")],
    "wdc": [("DC", "MD inner (Montgomery, PG)"), ("DC", "VA inner (Arl, Alex, Fairfax)"), ("MD inner (Montgomery, PG)", "DC"),
            ("VA inner (Arl, Alex, Fairfax)", "DC"), ("MD outer", "DC"), ("VA outer", "DC"), ("MD inner (Montgomery, PG)", "VA inner (Arl, Alex, Fairfax)"), ("DC", "DC")],
    "bos": [("NH", "MA"), ("MA", "NH")],
    "pvd": [("MA", "RI"), ("RI", "MA")],
}
UNION = {"MAN": ["Lower Manhattan", "Midtown", "Manhattan rest"], "PHL": ["Center City", "Philadelphia rest"]}


class C(tf.City):
    def __init__(self, city):
        super().__init__(city)
        lab = groups(city, self.blocks)
        self.glabels = sorted(set(lab))
        gi = {g: i for i, g in enumerate(self.glabels)}
        bg = np.array([gi[x] for x in lab])
        self.bg = bg
        ng = len(self.glabels)
        ok = self.bzone >= 0
        b = self.blocks
        nz = len(self.pop)
        P = np.zeros((nz, ng)); Q = np.zeros((nz, ng))
        np.add.at(P, (self.bzone[ok], bg[ok]), b["pop"].to_numpy()[ok])
        np.add.at(Q, (self.bzone[ok], bg[ok]), b["jobs"].to_numpy()[ok])
        self.GP = (P / np.maximum(P.sum(1, keepdims=True), 1e-9)).astype(np.float32)
        self.GQ = (Q / np.maximum(Q.sum(1, keepdims=True), 1e-9)).astype(np.float32)
        self.dband = np.digitize(self.d, BANDS[1:-1]).astype(np.int8)
        self.metro = self.urban_centres()
        self.observe_more()

    def observe_more(self):
        od = tf.fd.lodes_od(self.name)
        bi = {g: i for i, g in enumerate(self.blocks["geoid"])}
        hi = od["h_geocode"].map(bi); wi = od["w_geocode"].map(bi)
        m = ~(hi.isna() | wi.isna())
        hi, wi = hi[m].to_numpy(np.int64), wi[m].to_numpy(np.int64)
        n = od["S000"].to_numpy(np.float64)[m.to_numpy()]
        ng = len(self.glabels)
        G = np.zeros((ng, ng)); np.add.at(G, (self.bg[hi], self.bg[wi]), n)
        self.obs_G = G
        zh, zw = self.bzone[hi], self.bzone[wi]
        ok = (zh >= 0) & (zw >= 0)
        cross = (self.bstate[hi] != self.bstate[wi]) & ok
        self.obs_band_cross = np.bincount(self.dband[zh[cross], zw[cross]], weights=n[cross], minlength=len(BANDS) - 1) / n.sum()
        same = (~cross) & ok
        self.obs_band_same = np.bincount(self.dband[zh[same], zw[same]], weights=n[same], minlength=len(BANDS) - 1) / n.sum()
        # by state pair (unordered), share of all
        ids = sorted(set(self.bstate))
        self.state_ids = ids
        si = {s: i for i, s in enumerate(ids)}
        hs = np.array([si[s] for s in self.bstate[hi]]); ws = np.array([si[s] for s in self.bstate[wi]])
        S = np.zeros((len(ids), len(ids))); np.add.at(S, (hs, ws), n)
        self.obs_S = S / S.sum()
        # within / between urban centres, crossing
        mh, mw = self.metro[zh[ok]], self.metro[zw[ok]]
        cr = cross[ok]
        samem = (mh == mw) & (mh >= 0)
        nn = n[ok]
        self.obs_cross_metro = (nn[cr & samem].sum() / n.sum(), nn[cr & ~samem].sum() / n.sum())

    def urban_centres(self, dens=1500.0, minpop=50_000):
        """GHSL-style urban centres from the pack's own cells: 1 km squares with 1,500+ residents
        a km2 (of the whole square, water included), joined edge to edge, 50k+ people. Each zone
        gets the centre holding most of its residents, or -1."""
        header, a = load_pack(self.name)
        x, y, pop = a["x_m"].astype(np.float64), a["y_m"].astype(np.float64), a["pop"].astype(np.float64)
        gx = np.floor((x - x.min()) / 1000).astype(int); gy = np.floor((y - y.min()) / 1000).astype(int)
        R = np.zeros((gx.max() + 1, gy.max() + 1)); np.add.at(R, (gx, gy), pop)
        lab, n = ndimage.label(R >= dens)
        sizes = ndimage.sum(R, lab, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, int) - 1
        big = np.where(sizes >= minpop)[0]
        for k, i in enumerate(big[np.argsort(-sizes[big])]):
            keep[i + 1] = k
        cell_c = keep[lab[gx, gy]]
        zone = a["zone"].astype(np.int64)
        nz = len(self.pop)
        out = np.full(nz, -1)
        cnt = {}
        for z, c, p in zip(zone, cell_c, pop):
            cnt.setdefault(z, {}).setdefault(c, 0.0)
            cnt[z][c] += p
        for z, d in cnt.items():
            out[z] = max(d, key=d.get)
        top = sorted(((sizes[i], i) for i in big), reverse=True)[:4]
        print(f"  urban centres: {len(big)}, the largest {', '.join(f'{s / 1e6:.2f}M' for s, _ in top)}; "
              f"zones in one {np.mean(out >= 0):.0%}, residents in one {np.sum(self.pop[out >= 0]) / self.pop.sum():.0%}")
        return out

    def water_km(self):
        """Km of water on the straight line between zone centres (pack water mask, 24 samples).
        Cached in the temp folder (minutes for New York)."""
        import tempfile
        cache = os.path.join(tempfile.gettempdir(), f"t171_water_{self.name}.npy")
        if os.path.exists(cache):
            return np.load(cache)
        out = self._water_km()
        np.save(cache, out)
        return out

    def _water_km(self):
        header, a = load_pack(self.name)
        m = water_mask(header, a)
        if m is None:
            return np.zeros_like(self.d)
        W, H = m["size"]
        grid = np.zeros((H, W), bool)
        off, xs = m["row"], m["x"]
        for r in range(H):
            seg = xs[off[r]:off[r + 1]]
            for s0, s1 in zip(seg[0::2], seg[1::2]):
                grid[r, s0:s1] = True
        g = header["gravity"]; zm = g["zone_m"]; x0, y0 = g["grid_origin_m"]
        za = a["zone_gx"].astype(np.float64) * zm + x0 + zm / 2
        zb = a["zone_gy"].astype(np.float64) * zm + y0 + zm / 2
        ox, oy, cm = m["origin_m"][0], m["origin_m"][1], m["cell_m"]
        nz = len(za)
        S = 24
        t = (np.arange(S) + 0.5) / S
        out = np.zeros((nz, nz), np.float32)
        for i in range(nz):
            px = za[i] + (za[None, :] - za[i]) * t[:, None]
            py = zb[i] + (zb[None, :] - zb[i]) * t[:, None]
            c = np.floor((px - ox) / cm).astype(np.int64); r = np.floor((py - oy) / cm).astype(np.int64)
            inside = (c >= 0) & (c < W) & (r >= 0) & (r < H)
            w = np.zeros(c.shape, bool)
            w[inside] = grid[r[inside], c[inside]]
            out[i] = w.mean(0) * self.d[i]
        return out

    def solve_m(self, M, iters=300, tol=1e-4):
        F = np.exp(-self.d / np.float32(KM)) * self.d ** np.float32(-PW)
        if M is not None:
            F *= M
        o = self.pop * self.jobs.sum() / self.pop.sum()
        dj = self.jobs
        col = dj.astype(np.float32)
        has = o > 0
        row = None
        for it in range(iters):
            s = (F @ col).astype(np.float64)
            if it > 0 and np.max(np.abs(row[has] * s[has] / o[has] - 1)) < tol:
                break
            row = np.where(has, o / np.maximum(s, 1e-30), 0.0).astype(np.float32)
            t = (row @ F).astype(np.float64)
            col = np.where(dj > 0, dj / np.maximum(t, 1e-30), 0.0).astype(np.float32)
        F *= row[:, None]; F *= col[None, :]
        return F

    def report(self, T):
        r = self.score(T)
        tot = float(T.sum(dtype=np.float64))
        G = (self.GP.T @ T @ self.GQ).astype(np.float64)
        r["G"] = G
        r["band_cross"] = np.bincount(self.dband[self.other], weights=T[self.other], minlength=len(BANDS) - 1) / tot
        same = ~self.other
        r["band_same"] = np.bincount(self.dband[same], weights=T[same], minlength=len(BANDS) - 1) / tot
        samem = (self.metro[:, None] == self.metro[None, :]) & (self.metro[:, None] >= 0)
        r["cross_metro"] = (float(T[self.other & samem].sum(dtype=np.float64)) / tot, float(T[self.other & ~samem].sum(dtype=np.float64)) / tot)
        return r

    def gshare(self, G, h, w):
        gi = {g: i for i, g in enumerate(self.glabels)}
        hs = [gi[x] for x in UNION.get(h, [h]) if x in gi]
        ws = [gi[x] for x in UNION.get(w, [w]) if x in gi]
        return G[np.ix_(hs, ws)].sum() / G[hs].sum()

    def fit_scalar(self, make, lo=0.03, hi=1.0, steps=10):
        """Bisection on a scalar so the between-state share matches LODES."""
        lo, hi = np.log(lo), np.log(hi)
        for _ in range(steps):
            mid = 0.5 * (lo + hi)
            T = self.solve_m(make(np.exp(mid)))
            c = float(T[self.other].sum(dtype=np.float64) / T.sum(dtype=np.float64))
            lo, hi = (lo, mid) if c > self.obs_cross else (mid, hi)
        return float(np.exp(0.5 * (lo + hi)))


class Sink(dict):
    """Solves and reports each variant as it is added, keeping only the report (memory)."""
    def __init__(self, c):
        super().__init__()
        self.c, self.t0 = c, time.time()

    def __setitem__(self, name, v):
        M, desc = v
        super().__setitem__(name, (self.c.report(self.c.solve_m(M)), desc))
        print(f"  [{name}] {desc} ({time.time() - self.t0:.0f} s)", flush=True)


def variants(c, which):
    oth = c.other
    out = Sink(c)
    if "none" in which:
        out["none"] = (None, "no factor")
    if "k" in which:
        k = c.fit_scalar(lambda k: np.where(oth, np.float32(k), np.float32(1)))
        out["k"] = (np.where(oth, np.float32(k), np.float32(1)), f"one k = {k:.3f}")
        out["k0.23"] = (np.where(oth, np.float32(0.23), np.float32(1)), "shipped k = 0.23")
    if "pair" in which:
        zr = c.zr
        ids = c.state_ids
        # region index -> state fips: the pack header's region ids are fips
        header, _ = load_pack(c.name)
        rid = header.get("regions", {}).get("ids", [])
        rs = np.array([ids.index(r) if r in ids else -1 for r in rid])
        zs = rs[zr]
        ns = len(ids)
        K = np.ones((ns, ns), np.float32) * 0.25
        np.fill_diagonal(K, 1)
        for it in range(6):
            M = K[zs[:, None], zs[None, :]]
            T = c.solve_m(M)
            Sm = np.zeros((ns, ns))
            for a in range(ns):
                for b in range(ns):
                    Sm[a, b] = T[np.ix_(zs == a, zs == b)].sum(dtype=np.float64)
            Sm /= Sm.sum()
            for a in range(ns):
                for b in range(a + 1, ns):
                    mo, ob = Sm[a, b] + Sm[b, a], c.obs_S[a, b] + c.obs_S[b, a]
                    if mo > 0 and ob > 0:
                        K[a, b] = K[b, a] = np.clip(K[a, b] * (ob / mo) ** 0.9, 0.01, 3)
        M = K[zs[:, None], zs[None, :]]
        desc = ", ".join(f"{ids[a]}-{ids[b]} {K[a, b]:.2f}" for a in range(ns) for b in range(a + 1, ns) if c.obs_S[a, b] + c.obs_S[b, a] > 0.001)
        out["pair"] = (M, "per state pair: " + desc)
    if "kd" in which:
        for d0 in (3.0, 8.0):
            f = lambda k, d0=d0: np.where(oth, np.float32(k) + (1 - np.float32(k)) * np.exp(-c.d / np.float32(d0)), np.float32(1)).astype(np.float32)
            k = c.fit_scalar(f, lo=0.01)
            out[f"kd{d0:g}"] = (f(k), f"k(d) = {k:.3f} + (1 - {k:.3f}) exp(-d / {d0:g} km)")
        for D in (6.0,):
            f = lambda k, D=D: np.where(oth & (c.d > D), np.float32(k), np.float32(1)).astype(np.float32)
            k = c.fit_scalar(f, lo=0.01)
            out[f"kbeyond{D:g}"] = (f(k), f"k = {k:.3f} only beyond {D:g} km, 1 nearer")
    if "metro" in which:
        samem = (c.metro[:, None] == c.metro[None, :]) & (c.metro[:, None] >= 0)
        f = lambda k: np.where(oth & ~samem, np.float32(k), np.float32(1)).astype(np.float32)
        k = c.fit_scalar(f, lo=0.001)
        out["metro"] = (f(k), f"k = {k:.3f} between states unless both zones in one urban centre")
    if "waterfit" in which:
        # the water term alone, strong enough to bring the between-state share to LODES's: does a
        # river-crossing cost do the factor's job? (a tolled-crossing stand-in, T-268's question)
        wk = c.water_km()
        lo, hi = np.log(0.05), np.log(20.0)
        for _ in range(9):
            mid = 0.5 * (lo + hi)
            T = c.solve_m(np.exp(-np.float32(np.exp(mid)) * wk).astype(np.float32))
            cr = float(T[oth].sum(dtype=np.float64) / T.sum(dtype=np.float64))
            lo, hi = (mid, hi) if cr > c.obs_cross else (lo, mid)
            del T
        g = float(np.exp(0.5 * (lo + hi)))
        out["waterfit"] = (np.exp(-np.float32(g) * wk).astype(np.float32), f"no state factor; exp(-{g:.2f} x water km), fitted to the between-state share")
        # a crossing toll: a fixed factor on any pair whose line crosses 300+ m of water
        f = lambda k: np.where(wk > 0.3, np.float32(k), np.float32(1)).astype(np.float32)
        k = c.fit_scalar(f, lo=0.01)
        out["crossing"] = (f(k), f"no state factor; k = {k:.3f} on any pair crossing 300+ m of water")
    if "water" in which:
        wk = c.water_km()
        c._wk = wk
        best = None
        for g in (0.05, 0.1, 0.2, 0.4):
            T = c.solve_m(np.exp(-np.float32(g) * wk).astype(np.float32))
            r = c.score(T)
            s = r["county_off"] + r["tld_off"]
            print(f"    water g {g}: between states {r['cross']:.2%}, county {r['county_off']:.1%}, length {r['tld_off']:.1%}", flush=True)
            if best is None or s < best[0]:
                best = (s, g)
        g = best[1]
        out["water"] = (np.exp(-np.float32(g) * wk).astype(np.float32), f"no state factor; exp(-{g} x water km)")
        k = c.fit_scalar(lambda k: (np.where(oth, np.float32(k), np.float32(1)) * np.exp(-np.float32(g) * wk)).astype(np.float32))
        out["water+k"] = ((np.where(oth, np.float32(k), np.float32(1)) * np.exp(-np.float32(g) * wk)).astype(np.float32), f"exp(-{g} x water km) and k = {k:.3f}")
    return out


def main(args):
    which = {"none", "k", "pair", "kd", "metro", "water"}
    if "--variants" in args:
        i = args.index("--variants")
        which = set(args[i + 1].split(","))
        args = args[:i] + args[i + 2:]
    cities = args or ["nyc", "phl", "wdc", "bos", "pvd"]
    for city in cities:
        c = C(city)
        res = variants(c, which)
        names = list(res)
        print(f"\n### {city}\n")
        for n in names:
            print(f"- {n}: {res[n][1]}")
        print("\n| measure | LODES | " + " | ".join(names) + " |")
        print("|---|---|" + "---|" * len(names))
        print("| between states | " + f"{c.obs_cross:.2%} | " + " | ".join(f"{res[n][0]['cross']:.2%}" for n in names) + " |")
        print("| county pairs misplaced | | " + " | ".join(f"{res[n][0]['county_off']:.1%}" for n in names) + " |")
        print("| length misplaced | | " + " | ".join(f"{res[n][0]['tld_off']:.1%}" for n in names) + " |")
        print("| own county | " + f"{np.trace(c.obs_C):.1%} | " + " | ".join(f"{res[n][0]['own']:.1%}" for n in names) + " |")
        print("| crossing within one urban centre / between | " + f"{c.obs_cross_metro[0]:.2%} / {c.obs_cross_metro[1]:.2%} | " + " | ".join(f"{res[n][0]['cross_metro'][0]:.2%} / {res[n][0]['cross_metro'][1]:.2%}" for n in names) + " |")
        for h, w in ROWS.get(city, []):
            lv = c.gshare(c.obs_G, h, w)
            print(f"| {h} -> {w} (x LODES) | {lv:.1%} | " + " | ".join(f"{c.gshare(res[n][0]['G'], h, w) / lv:.2f}" for n in names) + " |")
        print("\ncrossing trips by distance band, model / LODES (share of all in-pack commutes):\n")
        print("| band km | LODES crossing | LODES same-state | " + " | ".join(names) + " |")
        print("|---|---|---|" + "---|" * len(names))
        for b in range(len(BANDS) - 1):
            print(f"| {BANDS[b]}-{BANDS[b + 1]} | {c.obs_band_cross[b]:.2%} | {c.obs_band_same[b]:.2%} | " + " | ".join(
                f"{res[n][0]['band_cross'][b] / max(c.obs_band_cross[b], 1e-9):.2f}" for n in names) + " |")
        print("\nsame-state trips by band, model / LODES: | " + " | ".join(
            f"{n}: " + " ".join(f"{res[n][0]['band_same'][b] / max(c.obs_band_same[b], 1e-9):.2f}" for b in range(len(BANDS) - 1)) for n in names), flush=True)
        del c


if __name__ == "__main__":
    main(sys.argv[1:])
