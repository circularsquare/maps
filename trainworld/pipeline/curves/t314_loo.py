"""T-314: which fitted city's commute curve a city with no flows should borrow, by a leave-one-out
test over every pack with its own fitted curve (notes/T-314.md).

    ..\\venv\\Scripts\\python.exe -u trainworld/pipeline/curves/t314_loo.py setup [ids]     per city: margins, units, flows -> CACHE
    ..\\venv\\Scripts\\python.exe -u trainworld/pipeline/curves/t314_loo.py score1 [ids]    each city on every fitted curve
    ..\\venv\\Scripts\\python.exe -u trainworld/pipeline/curves/t314_loo.py score2          each city on its rule-made curves
    ..\\venv\\Scripts\\python.exe -u trainworld/pipeline/curves/t314_loo.py analyse         the table
    ..\\venv\\Scripts\\python.exe -u trainworld/pipeline/curves/t314_loo.py road            Mumbai's road factor

Every city is scored the same way: the shipped pack's 2 km zones and commute margins (data/packs),
the game's doubly constrained gravity (US packs with their border factor), and the share of
commuters misplaced (half the L1 distance between shares) against the city's own observed table:
municipality pairs (Japan's census, INSEE communes, London's districts from WU01EW, Toronto's CSDs),
county pairs (US, LODES), the published Gemeinde pairs (Berlin, as pipeline/curves/t290_curve.py), or the
distance bands the curve was fitted to (Mumbai's B-28, Sao Paulo's OD survey).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import math
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
sys.path.insert(0, HERE)
import build_city as b  # noqa: E402
from read_pack import load_pack  # noqa: E402

ART = os.path.join(ROOT, "..", "artifacts", "trainworld")
CACHE = os.path.join(ART, "T-314-cache")
RAW = os.path.join(ROOT, "data", "raw", "t314")
US = ["nyc", "phl", "bos", "wdc", "bal", "pvd", "nhv", "hfd", "spf"]
JP_METRO = ["tyo", "osa", "ngo", "fuk", "spk", "sdj", "hij"]
OTHER = {"tor": "CA", "lon": "GB", "par": "FR", "ber": "DE", "mum": "IN", "sao": "BR"}


def jp_zones():
    with open(os.path.join(ROOT, "pipeline", "japan_zones.json"), encoding="utf-8") as f:
        return sorted(z for z, v in json.load(f)["zones"].items() if "curve_fit" in v)


def cities():
    return US + list(OTHER) + JP_METRO + jp_zones()


def iso2(city):
    return "US" if city in US else OTHER.get(city, "JP")


# ---------------------------------------------------------------- setup

def pack_base(city):
    h, a = load_pack(city)
    g = h["gravity"]
    nz = h["zones"]
    zone = a["zone"].astype(np.int64)
    home, work = a["commute_home"].astype(np.float64), a["commute_work"].astype(np.float64)
    lon0, lat0 = h["origin"]["lon"], h["origin"]["lat"]
    lat = lat0 + np.degrees(a["y_m"].astype(np.float64) / b.R_EARTH)
    lon = lon0 + np.degrees(a["x_m"].astype(np.float64) / (b.R_EARTH * math.cos(math.radians(lat0))))
    out = dict(H=np.bincount(zone, weights=home, minlength=nz), J=np.bincount(zone, weights=work, minlength=nz),
               gx=a["zone_gx"].astype(np.int32), gy=a["zone_gy"].astype(np.int32),
               zm=np.float64(g["zone_m"]), intra=np.float64(g.get("intrazonal_km", 1.04)),
               pop=np.float64(h["totals"]["pop"]), own=np.array([g["decay_pow"], g["decay_km"]], np.float64))
    return out, dict(zone=zone, home=home, work=work, lat=lat, lon=lon, jobs=a["jobs"].astype(np.float64))


def unit_of_cells(lat, lon, units):
    import england
    from scipy.spatial import cKDTree
    li, near = england.cell_units(lat, lon, units)
    bad = li < 0
    if bad.any():
        c = math.cos(math.radians(float(np.mean(lat))))
        t = cKDTree(np.c_[lon[~bad] * c, lat[~bad]])
        li[bad] = li[~bad][t.query(np.c_[lon[bad] * c, lat[bad]])[1]]
    for u, p in near.items():
        li[p] = u
    return li


def shares(zone, li, w, nz, nl):
    M = np.zeros((nz, nl))
    np.add.at(M, (zone, li), w)
    return (M / np.maximum(M.sum(1, keepdims=True), 1e-12)).astype(np.float32)


def units_setup(city, units, flows_h, flows_w, flows_n):
    """Municipality-pair cities: units [(code, polygon)], flows as codes and counts."""
    base, c = pack_base(city)
    nl = len(units)
    ci = {u[0]: i for i, u in enumerate(units)}
    li = unit_of_cells(c["lat"], c["lon"], units)
    nz = len(base["H"])
    keep = np.array([h in ci and w in ci for h, w in zip(flows_h, flows_w)])
    hi = np.array([ci[h] for h in np.asarray(flows_h)[keep]])
    wi = np.array([ci[w] for w in np.asarray(flows_w)[keep]])
    Cl = np.zeros((nl, nl))
    np.add.at(Cl, (hi, wi), np.asarray(flows_n, np.float64)[keep])
    base.update(kind="pairs", P=shares(c["zone"], li, c["home"], nz, nl), Q=shares(c["zone"], li, c["work"], nz, nl), C=Cl)
    print(f"{city}: {nz:,} zones, {nl} units, {Cl.sum():,.0f} observed commuters with both ends inside, own unit {np.trace(Cl) / Cl.sum():.1%}", flush=True)
    return base


def setup_jp(city):
    import japan
    cfg = b.CITIES[city]
    ms = japan.municipalities(cfg["prefectures"])
    fl = japan.flows(cfg["prefectures"])
    return units_setup(city, [(m[0], m[3]) for m in ms], fl.h.to_numpy(), fl.w.to_numpy(), fl.n.to_numpy(float))


def setup_par():
    import france
    cs = france.communes(b.CITIES["par"]["departements"])
    fl = france.flux()
    return units_setup("par", [(c[0], c[3]) for c in cs], fl.h.to_numpy(), fl.w.to_numpy(), fl.n.to_numpy(float))


def setup_lon():
    import england
    from shapely import make_valid, union_all
    from shapely.geometry import shape
    with open(england.raw("lon_itl3.geojson"), encoding="utf-8") as f:
        boundary = union_all([make_valid(shape(ft["geometry"])) for ft in json.load(f)["features"]])
    with open(england.raw("lad_dec2023_uk_bgc.geojson"), encoding="utf-8") as f:
        lads = [(ft["properties"]["LAD23NM"], make_valid(shape(ft["geometry"]))) for ft in json.load(f)["features"]]
    lads = [x for x in lads if boundary.contains(x[1].representative_point())]
    pts = england.msoa11_points("lon", boundary)
    fl = england.wu01ew()
    fl = fl[fl.h.isin(pts) & fl.w.isin(pts)]
    codes = sorted(pts)
    mll = np.array([pts[c] for c in codes])
    ml, _ = england.cell_units(mll[:, 0], mll[:, 1], lads)
    lad_of = {c: (lads[m][0] if m >= 0 else None) for c, m in zip(codes, ml)}
    return units_setup("lon", lads, fl.h.map(lad_of).to_numpy(), fl.w.map(lad_of).to_numpy(), fl.n.to_numpy(float))


def setup_ber():
    import germany
    from t290_curve import published_pairs
    cfg = b.CITIES["ber"]
    kreise = cfg["kreise"]
    gs = germany.gemeinden("ber", kreise)
    units = [(g[0], g[2]) for g in gs]
    ci = {g[0]: i for i, g in enumerate(gs)}
    base, c = pack_base("ber")
    li = unit_of_cells(c["lat"], c["lon"], units)
    nz, nl = len(base["H"]), len(units)
    pairs = published_pairs(ci, sorted({k[:2] for k in kreise}))
    keys = np.array(list(pairs.keys()), np.int64)
    obs = np.array(list(pairs.values()), np.float64)
    base.update(kind="published", P=shares(c["zone"], li, c["home"], nz, nl), Q=shares(c["zone"], li, c["work"], nz, nl),
                keys=keys, obs=obs, wfh=np.float64(1 - c["work"].sum() / c["jobs"].sum()))
    print(f"ber: {nz:,} zones, {nl} Gemeinden, {len(obs):,} published pairs, {obs.sum():,.0f} workers", flush=True)
    return base


def setup_tor():
    import t277_fit
    f = t277_fit.Fit()
    base, _ = pack_base("tor")
    base.update(kind="pairs", P=f.P.astype(np.float32), Q=f.Q.astype(np.float32), C=f.O)
    return base


def setup_us(city):
    import t162_fit
    h, _ = load_pack(city)
    c = t162_fit.City(city)
    base, _ = pack_base(city)
    base.update(kind="pairs", P=c.P, Q=c.Q, C=c.obs_C, zr=c.zr.astype(np.int32), k=np.float64(h["gravity"].get("border_factor", 1.0)))
    return base


def setup_bands(city):
    cfg = b.CITIES[city]
    edges, counts = cfg["curve"]["bands"]()[:2]
    rf = cfg["curve"].get("road_factor", 1.0)
    base, _ = pack_base(city)
    base.update(kind="bands", edges=np.asarray(edges, np.float64), road=np.float64(rf),
                obs=np.asarray(counts, np.float64) / sum(counts))
    print(f"{city}: bands {list(edges)} / {rf}, shares {np.round(base['obs'], 3).tolist()}", flush=True)
    return base


def setup(city):
    os.makedirs(CACHE, exist_ok=True)
    t0 = time.time()
    if city in US:
        s = setup_us(city)
    elif city == "tor":
        s = setup_tor()
    elif city == "lon":
        s = setup_lon()
    elif city == "par":
        s = setup_par()
    elif city == "ber":
        s = setup_ber()
    elif city in ("mum", "sao"):
        s = setup_bands(city)
    else:
        s = setup_jp(city)
    np.savez(os.path.join(CACHE, f"{city}.npz"), **s)
    print(f"  {city} set up in {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------- scoring

class Scorer:
    def __init__(self, city):
        s = dict(np.load(os.path.join(CACHE, f"{city}.npz"), allow_pickle=False))
        self.city, self.s = city, s
        gx, gy = s["gx"].astype(np.float32), s["gy"].astype(np.float32)
        self.d = np.float32(s["zm"] / 1000.0) * np.sqrt((gx[:, None] - gx[None, :]) ** 2 + (gy[:, None] - gy[None, :]) ** 2)
        self.d[self.d == 0] = np.float32(s["intra"])
        self.other = (s["zr"][:, None] != s["zr"][None, :]) if "zr" in s and s["k"] != 1 else None
        self.H, self.J = s["H"], s["J"]
        self.kind = str(s["kind"])
        if self.kind == "bands":
            e = s["edges"] / s["road"]
            self.band = np.clip(np.searchsorted(e, self.d.ravel(), side="right") - 1, 0, len(s["obs"]) - 1)
        self.D0 = float(self.H @ self.d.astype(np.float64) @ self.J / (self.H.sum() * self.J.sum()))

    def gravity(self, pw, km, iters=300, tol=1e-4):
        F = (np.exp(-self.d / np.float32(km)) * self.d ** np.float32(-pw)).astype(np.float32)
        if self.other is not None:
            F[self.other] *= np.float32(self.s["k"])
        H, J = self.H, self.J
        o = H * J.sum() / H.sum()
        col = J.astype(np.float32)
        has = o > 0
        row = np.ones(len(o), np.float32)
        for it in range(iters):
            sm = (F @ col).astype(np.float64)
            if it > 0 and np.max(np.abs(row[has] * sm[has] / o[has] - 1)) < tol:
                break
            row = np.where(has, o / np.maximum(sm, 1e-30), 0.0).astype(np.float32)
            t = (row @ F).astype(np.float64)
            col = np.where(J > 0, J / np.maximum(t, 1e-30), 0.0).astype(np.float32)
        F *= row[:, None]
        F *= col[None, :]
        return F

    def score(self, pw, km):
        T = self.gravity(pw, km)
        tot = float(T.sum(dtype=np.float64))
        mean = float((T * self.d).sum(dtype=np.float64)) / tot
        s = self.s
        if self.kind == "bands":
            m = np.bincount(self.band, weights=T.ravel().astype(np.float64), minlength=len(s["obs"])) / tot
            miss = 0.5 * float(np.abs(m - s["obs"]).sum())
        else:
            C = (s["P"].T @ T @ s["Q"]).astype(np.float64)
            if self.kind == "pairs":
                miss = 0.5 * float(np.abs(C / C.sum() - s["C"] / s["C"].sum()).sum())
            else:
                m = C[s["keys"][:, 0], s["keys"][:, 1]] / (1 - s["wfh"])
                miss = 0.5 * float(np.abs(m - s["obs"]).sum() / s["obs"].sum())
        return dict(pow=round(float(pw), 4), km=round(float(km), 3), miss=miss, mean_km=mean)

    def km_for_mean(self, pw, want):
        lo, hi = math.log(2.0), math.log(400.0)
        for _ in range(12):
            km = math.exp((lo + hi) / 2)
            T = self.gravity(pw, km)
            mean = float((T * self.d).sum(dtype=np.float64) / T.sum(dtype=np.float64))
            lo, hi = (math.log(km), hi) if mean < want else (lo, math.log(km))
        return round(math.exp((lo + hi) / 2), 2)


def fitted_curves():
    out = {}
    for c in cities():
        s = np.load(os.path.join(CACHE, f"{c}.npz"))
        out[c] = (round(float(s["own"][0]), 4), round(float(s["own"][1]), 3))
    return out


def append(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def done_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r["city"], r["pow"], r["km"]) for r in map(json.loads, f)}


def score1(ids):
    out = os.path.join(ART, "T-314-scores.jsonl")
    curves = sorted(set(fitted_curves().values()))
    print(f"{len(curves)} distinct fitted curves", flush=True)
    have = done_keys(out)
    for c in ids:
        t0 = time.time()
        sc = Scorer(c)
        append(os.path.join(ART, "T-314-d0.jsonl"), {"city": c, "D0": sc.D0})
        for pw, km in curves:
            if (c, pw, km) not in have:
                append(out, {"city": c, **sc.score(pw, km)})
        print(f"{c}: {len(curves)} curves in {time.time() - t0:.0f} s", flush=True)
        del sc


# ---------------------------------------------------------------- global inputs and rules

def global_inputs():
    """{iso2: (income group, vehicles per 1,000)}: World Bank income groups, WHO's registered
    vehicles (GHO RS_194, 2016) over the World Bank's 2016 population."""
    with open(os.path.join(RAW, "wb_countries.json")) as f:
        cs = json.load(f)[1]
    iso3 = {c["iso2Code"]: c["id"] for c in cs}
    inc = {c["iso2Code"]: c["incomeLevel"]["id"] for c in cs}
    with open(os.path.join(RAW, "wb_pop2016.json")) as f:
        pop = {r["countryiso3code"]: r["value"] for r in json.load(f)[1] if r["value"]}
    with open(os.path.join(RAW, "gho_rs194.json")) as f:
        veh = {}
        for r in json.load(f)["value"]:
            if r["SpatialDimType"] == "COUNTRY" and r["NumericValue"]:
                veh[r["SpatialDim"]] = r["NumericValue"]
    return {k: (inc[k], 1000.0 * veh[iso3[k]] / pop[iso3[k]] if iso3[k] in veh and iso3[k] in pop else None) for k in iso3}


def table():
    """One row per fitted city: curve, country, income group, vehicles, population, D0, own mean."""
    gi = global_inputs()
    d0 = {}
    with open(os.path.join(ART, "T-314-d0.jsonl")) as f:
        for r in map(json.loads, f):
            d0[r["city"]] = r["D0"]
    sc = load_scores()
    rows = []
    for c, (pw, km) in fitted_curves().items():
        s = np.load(os.path.join(CACHE, f"{c}.npz"))
        k = iso2(c)
        own = sc[c][(pw, km)]
        rows.append(dict(city=c, country=k, income=gi[k][0], veh=gi[k][1], pop=float(s["pop"]), pow=pw, km=km,
                         D0=d0[c], own_mean=own["mean_km"], r=own["mean_km"] / d0[c]))
    return rows


def load_scores():
    sc = {}
    for name in ("T-314-scores.jsonl", "T-314-scores2.jsonl"):
        p = os.path.join(ART, name)
        if os.path.exists(p):
            with open(p) as f:
                for r in map(json.loads, f):
                    sc.setdefault(r["city"], {})[(r["pow"], r["km"])] = r
    return sc


def cmed(rows, key):
    """Median over countries of each country's median."""
    by = {}
    for r in rows:
        by.setdefault(r["country"], []).append(r[key])
    return float(np.median([np.median(v) for v in by.values()]))


def median_curve(pool):
    return (round(cmed(pool, "pow"), 4), round(cmed(pool, "km"), 3))


def pool_for(rows, c, variant):
    me = next(r for r in rows if r["city"] == c)
    if variant == "country" or me["country"] == "US":  # the US curve was fitted on all nine packs at once
        return [r for r in rows if r["country"] != me["country"]], me
    return [r for r in rows if r["city"] != c], me


def nearest(pool, me, keys):
    X = np.array([[math.log10(r[k]) for k in keys] for r in pool])
    sd = X.std(0)
    sd[sd == 0] = 1
    x = np.array([math.log10(me[k]) for k in keys])
    i = int(np.argmin((((X - x) / sd) ** 2).sum(1)))
    return pool[i]["pow"], pool[i]["km"], pool[i]["city"]


RULES = ["nearest pop and vehicles", "nearest pop", "income group median", "world median", "world best", "scaled to size"]


def rule_curves(rows, sc, c, variant):
    """{rule: (pow, km) or (pow, ('mean', target km))}"""
    pool, me = pool_for(rows, c, variant)
    out = {}
    pw, km, src = nearest(pool, me, ("pop", "veh"))
    out["nearest pop and vehicles"] = (pw, km)
    pw, km, src = nearest(pool, me, ("pop",))
    out["nearest pop"] = (pw, km)
    grp = [r for r in pool if r["income"] == me["income"]]
    out["income group median"] = median_curve(grp if grp else pool)
    out["world median"] = median_curve(pool)
    # the one fitted curve that loses least over the pool, by country
    cands = sorted({(r["pow"], r["km"]) for r in pool})
    best, bv = None, 1e9
    for cv in cands:
        by = {}
        for r in pool:
            own = sc[r["city"]][(r["pow"], r["km"])]["miss"]
            by.setdefault(r["country"], []).append(sc[r["city"]][cv]["miss"] - own)
        v = float(np.mean([np.mean(x) for x in by.values()]))
        if v < bv:
            best, bv = cv, v
    out["world best"] = best
    out["scaled to size"] = (median_curve(pool)[0], ("mean", round(cmed(pool, "r") * me["D0"], 3)))
    return out


def score2():
    rows = table()
    sc = load_scores()
    out = os.path.join(ART, "T-314-scores2.jsonl")
    have = done_keys(out)
    todo = {}
    for r in rows:
        for variant in ("city", "country"):
            for rule, cv in rule_curves(rows, sc, r["city"], variant).items():
                todo.setdefault(r["city"], set()).add((cv[0], cv[1]))
    # Lagos too (only the scaled rule needs its pack; the rest need its inputs)
    for c in cities():
        need = [cv for cv in todo.get(c, ()) if not isinstance(cv[1], tuple) and cv not in sc.get(c, {}) and (c, cv[0], cv[1]) not in have]
        mean_t = [cv for cv in todo.get(c, ()) if isinstance(cv[1], tuple)]
        if not need and not mean_t:
            continue
        t0 = time.time()
        s = Scorer(c)
        for pw, km in need:
            append(out, {"city": c, **s.score(pw, km)})
        for pw, (_, want) in mean_t:
            km = s.km_for_mean(pw, want)
            append(out, {"city": c, "target_mean": want, **s.score(pw, km)})
        print(f"{c}: {len(need)} curves, {len(mean_t)} scaled in {time.time() - t0:.0f} s", flush=True)
        del s


def lookup(sc, c, cv):
    if isinstance(cv[1], tuple):
        rs = [r for r in sc[c].values() if r.get("target_mean") == cv[1][1] and abs(r["pow"] - cv[0]) < 1e-9]
        return rs[-1]
    return sc[c][(round(cv[0], 4), round(cv[1], 3))]


def analyse():
    rows = table()
    sc = load_scores()
    res = {}
    for variant in ("city", "country"):
        lines = []
        for r in rows:
            c = r["city"]
            own = sc[c][(r["pow"], r["km"])]["miss"]
            line = dict(city=c, country=r["country"], own=own, own_curve=(r["pow"], r["km"]))
            for rule, cv in rule_curves(rows, sc, c, variant).items():
                s = lookup(sc, c, cv)
                line[rule] = s["miss"]
                line[rule + " curve"] = (s["pow"], s["km"])
            lines.append(line)
        summ = {}
        for rule in RULES:
            ex = np.array([ln[rule] - ln["own"] for ln in lines])
            by = {}
            for ln in lines:
                by.setdefault(ln["country"], []).append(ln[rule] - ln["own"])
            worst = max(lines, key=lambda ln: ln[rule] - ln["own"])
            summ[rule] = dict(mean_excess=float(ex.mean()), country_mean_excess=float(np.mean([np.mean(v) for v in by.values()])),
                              by_country={k: float(np.mean(v)) for k, v in by.items()},
                              worst=(worst["city"], worst[rule] - worst["own"], worst[rule + " curve"]))
        res[variant] = dict(lines=lines, summary=summ)
        print(f"=== leave one {variant} out: excess misplaced over the own curve, points")
        for rule in RULES:
            s = summ[rule]
            print(f"  {rule:26s} mean {100 * s['mean_excess']:5.2f}  by country {100 * s['country_mean_excess']:5.2f}  worst {s['worst'][0]} "
                  f"{100 * s['worst'][1]:.1f} {s['worst'][2]}  | " + " ".join(f"{k} {100 * v:.1f}" for k, v in s["by_country"].items()))
    with open(os.path.join(ART, "T-314-loo.json"), "w") as f:
        json.dump(dict(table=rows, **res), f, indent=1, default=float)
    print("=== the table")
    for r in rows:
        print(f"  {r['city']} {r['country']} {r['income']} veh {r['veh']:.0f} pop {r['pop'] / 1e6:.2f}M d^-{r['pow']} exp(-d/{r['km']}) "
              f"D0 {r['D0']:.1f} own mean {r['own_mean']:.1f} r {r['r']:.3f}")


# ---------------------------------------------------------------- Mumbai's road factor

def road():
    """Mumbai's B-28 bands are reported (road) distances. Fit both parameters with the edges divided
    by 1.3, by the pack's median near-pair drive factor, and by the trip-weighted factor measured at
    each edge (from routes.py's tables); how far apart the curves put the commuters."""
    import tier0
    h, a = load_pack("mum")
    nf = np.load(os.path.join(b.WORK, "mum.drive_near_f.npy"))
    s = Scorer("mum")
    r = h["routes"]["drive"]
    S, q = r["supercells"], r["q"]
    sup = a["drive_super"].astype(np.int64)
    F = np.exp(a["drive_far"].astype(np.float64).reshape(S, S)[sup][:, sup] / q)
    i, j = nf[:, 0].astype(np.int64), nf[:, 1].astype(np.int64)
    F[i, j] = nf[:, 2]
    T = s.gravity(*s.s["own"])
    off = ~np.eye(len(s.H), dtype=bool)
    w, dd, ff = T[off].astype(np.float64), s.d[off], F[off]

    def wmed(m):
        if not m.any():  # an edge under the nearest pair's road: the neighbouring zones' factor
            m = dd < 3.0
        o = np.argsort(ff[m]); c = np.cumsum(w[m][o])
        return float(ff[m][o][np.searchsorted(c, c[-1] / 2)])
    edges, counts = b.CITIES["mum"]["curve"]["bands"]()[:2]
    near_med = float(np.median(nf[:, 2]))
    # the factor at a reported edge e: trips whose straight distance x satisfies x * f(x) near e
    road_km = dd * ff
    per_edge = [1.0] + [wmed((road_km > e * 0.8) & (road_km < e * 1.25)) for e in edges[1:-1]] + [1.0]
    print(f"mum: near-pair median factor {near_med:.3f}; trip-weighted factor at each reported edge: "
          + ", ".join(f"{e} km {f:.3f}" for e, f in zip(edges, per_edge)), flush=True)
    # 0-1 and 2-5 km merged: at a factor over 1.44 the 1.5 km edge falls under the 1.04 km
    # intrazonal distance and the first band empties (and 2 km zones cannot split them anyway)
    edges, counts, per_edge = [edges[0]] + list(edges[2:]), [counts[0] + counts[1]] + list(counts[2:]), [per_edge[0]] + per_edge[2:]
    res = {}
    _, cells = pack_base("mum")
    for label, ed in (("1.3", np.asarray(edges) / 1.3), (f"near median {near_med:.2f}", np.asarray(edges) / near_med),
                      ("measured at each edge", np.asarray(edges) / np.asarray(per_edge))):
        lon0, lat0 = h["origin"]["lon"], h["origin"]["lat"]
        xm = b.R_EARTH * np.radians(cells["lon"] - lon0) * np.cos(np.radians(lat0))
        ym = b.R_EARTH * np.radians(cells["lat"] - lat0)
        pw, km = tier0.fit_curve_bands(xm, ym, cells["home"], cells["work"], {"bands": lambda ed=ed: (list(ed), counts), "road_factor": 1.0})
        res[label] = (pw, km)
        print(f"  {label}: d^-{pw:g} exp(-d/{km} km)", flush=True)
    Ts = {k: s.gravity(*v) for k, v in res.items()}
    ks = list(res)
    for k in ks[1:]:
        moved = 0.5 * float(np.abs(Ts[k] / Ts[k].sum() - Ts[ks[0]] / Ts[ks[0]].sum()).sum())
        mk = float((Ts[k] * s.d).sum() / Ts[k].sum())
        print(f"  {k} against 1.3: {moved:.1%} of commuters on other zone pairs; mean {mk:.1f} km against "
              f"{float((Ts[ks[0]] * s.d).sum() / Ts[ks[0]].sum()):.1f}", flush=True)
    with open(os.path.join(ART, "T-314-road.json"), "w") as f:
        json.dump(dict(near_median=near_med, per_edge=per_edge, edges=edges, fits=res), f, indent=1)


def write_table():
    """pipeline/fitted_curves.json, which tier0.borrowed_curve reads: every fitted city's own curve
    and its misplaced share under each candidate (the distinct fitted curves). A city whose table
    has one unit (every curve scores the same) is left out."""
    sc = load_scores()
    fc = fitted_curves()
    cands = sorted(set(fc.values()))
    cities_out = {}
    for c, own in fc.items():
        m = [round(sc[c][cv]["miss"], 5) for cv in cands]
        if max(m) - min(m) < 1e-6:
            print(f"{c}: every curve scores the same, left out", flush=True)
            continue
        cities_out[c] = {"country": iso2(c), "pow": own[0], "km": own[1], "misplaced": m}
    doc = {"task": "T-314", "notes": "notes/T-314.md; written by pipeline/curves/t314_loo.py table",
           "measure": "share of the city's observed commuters misplaced (unit pairs, or distance bands for mum and sao)",
           "candidates": [list(cv) for cv in cands], "cities": cities_out}
    with open(os.path.join(ROOT, "pipeline", "fitted_curves.json"), "w") as f:
        json.dump(doc, f, indent=1)
    print(f"{len(cities_out)} cities, {len(cands)} candidate curves", flush=True)


if __name__ == "__main__":
    cmd, ids = sys.argv[1], sys.argv[2:]
    if cmd == "table":
        write_table()
    if cmd == "setup":
        for c in ids or cities():
            if ids or not os.path.exists(os.path.join(CACHE, f"{c}.npz")):
                setup(c)
    elif cmd == "score1":
        score1(ids or cities())
    elif cmd == "score2":
        score2()
    elif cmd == "analyse":
        analyse()
    elif cmd == "road":
        road()
