"""T-376 (T-362 stage 2): fit the long-haul tail and the border factor of the world market.

    python pipeline/longhaul_fit.py [--write]      (about 3 minutes, 3 cores)

Model (notes/T-376.md): a resident of site i makes two kinds of departures a weekday, each
production constrained over the world field around i (never over the zone list):

- ground: R s_i, weight a_j b_ij on(d) max(d, 25)^-A, the NHTS ground fit unchanged (R 0.107, A 2.37);
- tail: Rt s_i u_i^THo / U_US, weight a_j u_j^TH min(y_j/y_i, 1)^eps_t b_ij ont(d) max(d, D1)^-G,
  its own normaliser;

s_i = min(y_i/y_US, 1)^eps (eps 0.7), a_j = pop_j min(y_j/y_US, 1)^eps, u = pop_r5 / 1e5 (tail
weights summed over r5 cells), b_ij = 1 in one country, else b min(y_i/y_US, 1)^eps_b; on() 0 at 25 km
and 1 at 50, ont() 0 at 300 and 1 at 700. Fitted: Rt, G, D1, b, THo, eps_b; TH and eps_t on a grid.
--form one is T-362's single-normaliser design, kept for comparison.

Targets: NHTS 2017 all-mode non-commute trips by band (Northeast and US), residents' trips abroad
(US by region from NTTO, Japan, the UK, India), the flights project's segment passengers between 48
metros over 1,000 km (x 0.6 for connections), NEC FUTURE 2012 pairs. FR24's raw rows are read offline and never written
out; only model-against-target pair totals are. --write writes pipeline/longhaul.json; the full
table goes to artifacts/trainworld/t376/fit_split.json."""
import argparse, json, math, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "3")
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))
import world_field as W

TW = Path(__file__).resolve().parents[1]
OUT = TW.parent / "artifacts/trainworld/t376"
PARAMS = Path(__file__).parent / "longhaul.json"
NHTS = TW / "research/intercity/nhts_long.json"
ROUTES = TW.parent / "flights/routes_fr24.json"
R_GROUND, A_GROUND, ON0, ON1, D0, UREF, EPS = 0.107, 2.37, 25.0, 50.0, 25.0, 1e5, 0.7
TH_GRID = [0.25, 0.5, 0.75]
EPS_T_GRID = [0.7, 1.0, 1.5, 2.0, 2.5]
# the tail rises from 0 at 300 km to 1 at 700: NHTS's Northeast air trips are near 0 under 500 km
# and the ground curve already matches ground trips there (T-276)
TAIL_ON = (300.0, 700.0)
NHTS_WEIGHT = 3.0  # the NHTS bands' weight against the flights pairs (a national survey against a judged 0.6)
CONNECT = 0.6  # share of segment passengers flying only between the two metros (judgement)
AIR_CLASS_WEIGHT = 3.0  # each flights distance class (1,000-2,500, 2,500-6,000, over 6,000 km) weighs ~3 bands
# US residents' legs abroad a day a person, by region: NTTO 2019, ~99.7M outbound trips by 328M
# residents (Mexico 39M, Canada 15M, Europe 18M, Asia and Oceania 8M, the rest 18M), two legs each
REGIONS = ["home", "Canada, Mexico", "Europe", "Asia, Oceania", "elsewhere"]
US_ABROAD = {1: 54.5e6, 2: 18.4e6, 3: 8.2e6, 4: 18.6e6}
US_ABROAD = {c: v * 2 / 365 / 328e6 for c, v in US_ABROAD.items()}
# residents' trips abroad a head a year, two legs each, a day: JNTO 2019 20.1M departures / 126M;
# ONS Travel Trends 2019 93.1M visits abroad / 66.8M. An island with no near neighbours against one
# beside Europe, so together they pin the border factor and the slope far out
ABROAD = {"jp": 0.16 * 2 / 365, "gb": 1.39 * 2 / 365, "in": 0.02 * 2 / 365}  # India: 26.9M departures 2019
US_OVER_5000 = 0.0006  # legs a day over 5,000 km: roughly 0.1 overseas trips a head a year (judgement)
TRIAL = {"R": R_GROUND, "k": 1e-4, "G": 1.0, "D1": 300.0, "TH": 0.5, "eps_t": 0.7, "b": 0.4}

METROS = [  # name, lon, lat, radius km, airports
    ("New York", -73.98, 40.75, 50, "JFK EWR LGA"), ("Philadelphia", -75.16, 39.95, 35, "PHL"),
    ("Boston", -71.06, 42.36, 40, "BOS"), ("Washington", -77.04, 38.90, 30, "DCA IAD"),
    ("Baltimore", -76.61, 39.29, 20, "BWI"), ("Chicago", -87.65, 41.88, 45, "ORD MDW"),
    ("Los Angeles", -118.25, 34.05, 55, "LAX BUR LGB SNA ONT"), ("San Francisco", -122.27, 37.6, 50, "SFO OAK SJC"),
    ("Atlanta", -84.39, 33.75, 40, "ATL"), ("Miami", -80.2, 25.9, 40, "MIA FLL"),
    ("Toronto", -79.38, 43.70, 45, "YYZ YTZ"), ("Montreal", -73.57, 45.50, 35, "YUL"),
    ("Mexico City", -99.13, 19.43, 40, "MEX"), ("Sao Paulo", -46.63, -23.55, 45, "GRU CGH"),
    ("London", -0.12, 51.51, 50, "LHR LGW LCY STN LTN"), ("Manchester", -2.24, 53.48, 35, "MAN"),
    ("Edinburgh", -3.19, 55.95, 25, "EDI"), ("Glasgow", -4.25, 55.86, 25, "GLA"),
    ("Paris", 2.35, 48.86, 45, "CDG ORY"), ("Amsterdam", 4.90, 52.37, 30, "AMS"),
    ("Brussels", 4.35, 50.85, 30, "BRU"), ("Frankfurt", 8.68, 50.11, 35, "FRA"),
    ("Berlin", 13.40, 52.52, 35, "BER SXF"), ("Munich", 11.58, 48.14, 35, "MUC"),
    ("Madrid", -3.70, 40.42, 35, "MAD"), ("Barcelona", 2.17, 41.39, 30, "BCN"),
    ("Rome", 12.50, 41.90, 30, "FCO CIA"), ("Milan", 9.19, 45.46, 35, "MXP LIN BGY"),
    ("Dublin", -6.26, 53.35, 25, "DUB"), ("Istanbul", 28.98, 41.01, 45, "IST SAW"),
    ("Dubai", 55.27, 25.20, 35, "DXB DWC"), ("Mumbai", 72.88, 19.08, 40, "BOM"),
    ("Delhi", 77.21, 28.61, 45, "DEL"), ("Bangalore", 77.59, 12.97, 35, "BLR"),
    ("Singapore", 103.82, 1.35, 25, "SIN"), ("Hong Kong", 114.17, 22.32, 25, "HKG"),
    ("Shanghai", 121.47, 31.23, 50, "PVG SHA"), ("Beijing", 116.40, 39.90, 50, "PEK PKX"),
    ("Guangzhou", 113.26, 23.13, 40, "CAN"), ("Chengdu", 104.07, 30.67, 40, "CTU TFU"),
    ("Seoul", 126.98, 37.57, 45, "ICN GMP"), ("Busan", 129.08, 35.18, 30, "PUS"),
    ("Tokyo", 139.69, 35.69, 50, "HND NRT"), ("Osaka", 135.50, 34.69, 40, "ITM KIX UKB"),
    ("Fukuoka", 130.40, 33.59, 30, "FUK"), ("Sapporo", 141.35, 43.06, 30, "CTS"),
    ("Sydney", 151.21, -33.87, 40, "SYD"), ("Melbourne", 144.96, -37.81, 40, "MEL"),
]
NEC = [("New York", "Philadelphia", 57000, 0.0), ("New York", "Boston", 50000, 1.0),
       ("New York", "Washington", 20000, 1.0), ("Boston", "Washington", 4900, 1.0)]
# New York - Philadelphia is reported, not fitted: the gap there is 25-80 km trips across a seam (T-299)

# distance bands: NHTS's from 25 km, then one beyond 5,000; each split into fine log bins for the tail
BANDS = np.array([25, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 700, 1000, 1500, 2500, 5000, 20040.0])
FINE = np.unique(np.concatenate([np.exp(np.linspace(np.log(BANDS[b]), np.log(BANDS[b + 1]), 5 if b < 18 else 13))
                                 for b in range(len(BANDS) - 1)]))
FINE_C = np.sqrt(FINE[:-1] * FINE[1:])
FINE_BAND = np.searchsorted(BANDS, FINE_C, side="right") - 1
# destination classes: 0 home, 1-4 abroad on the same landmass by region, 5-8 abroad across the sea
NB, NF, NR = len(BANDS) - 1, len(FINE) - 1, 2 * len(REGIONS) - 1
M_FB = np.zeros((NF, NB)); M_FB[np.arange(NF), FINE_BAND] = 1
COMBOS = [(th, et) for th in TH_GRID for et in EPS_T_GRID]


def region_of(countries):
    """Each country's region class for the US outbound target (home is decided per site)."""
    out = []
    for c in countries + [{"iso2": "", "subregion": ""}]:
        s = c["subregion"]
        if c["iso2"] in ("CA", "MX"):
            out.append(1)
        elif "Europe" in s:
            out.append(2)
        elif s in ("Eastern Asia", "South-Eastern Asia", "Southern Asia", "Central Asia", "Australia and New Zealand",
                   "Melanesia", "Micronesia", "Polynesia"):
            out.append(3)
        else:
            out.append(4)
    return np.array(out)


class Field:
    """The world field with ground attraction and tail weights summed exactly from r5 to r4 and r3."""
    def __init__(self):
        self.wf = wf = W.load()
        self.us = next(i for i, c in enumerate(wf.countries) if c["iso2"] == "US")
        self.y_us = wf.countries[self.us]["gdp"]
        self.region = region_of(wf.countries)
        r5, r4, r3 = wf.r5, wf.r4, wf.r3
        inc = np.minimum(r5.gdp / self.y_us, 1)
        up = lambda x5: [x5, np.add.reduceat(x5, r4.start), np.add.reduceat(np.add.reduceat(x5, r4.start), r3.start)]
        self.a = up(r5.pop * inc ** EPS)
        # tail weights: the ground's attraction a_j times density, before the factor that depends on
        # the origin, min(y_j / y_i, 1)^eps_t
        self.tb = {th: up(r5.pop * inc ** EPS * (np.maximum(r5.pop, 1) / UREF) ** th) for th in TH_GRID}
        self.inc = [np.minimum(lv.gdp / self.y_us, 1) for lv in (r5, r4, r3)]

    def around(self, lon, lat):
        wf = self.wf
        d3 = W.km(lon, lat, wf.r3.lon, wf.r3.lat)
        far = np.nonzero(d3 > W.MID_KM)[0]
        i4 = W._children(wf.r3, np.nonzero(d3 <= W.MID_KM)[0])
        d4 = W.km(lon, lat, wf.r4.lon[i4], wf.r4.lat[i4])
        mid, i5 = i4[d4 > W.NEAR_KM], W._children(wf.r4, i4[d4 <= W.NEAR_KM])
        d5 = W.km(lon, lat, wf.r5.lon[i5], wf.r5.lat[i5])
        parts = [(0, i5, d5), (1, mid, d4[d4 > W.NEAR_KM]), (2, far, d3[far])]
        lv = [wf.r5, wf.r4, wf.r3]
        cat = lambda arrs: np.concatenate([arrs[l][ix] for l, ix, _ in parts])
        d = np.concatenate([p[2] for p in parts])
        country = np.concatenate([lv[l].country[ix] for l, ix, _ in parts])
        land = np.concatenate([lv[l].landmass[ix] for l, ix, _ in parts])
        return d, country, land, cat(self.a), {th: cat(x) for th, x in self.tb.items()}, cat(self.inc)

    def tail_weights(self, tb, inc_j, inc_i):
        """Per grid combo: tail weight with destinations poorer than the origin damped."""
        rel = np.minimum(inc_j / inc_i, 1)
        return [tb[th] * rel ** et for th, et in COMBOS]


def site_terms(F, lon, lat, country, landmass, inc_i):
    """Per site: ground sums by band and tail sums by fine bin per grid combo, by destination class."""
    d, c, lm, a, tb, inc = F.around(lon, lat)
    on = np.clip((d - ON0) / (ON1 - ON0), 0, 1)
    keep = on > 0
    d, c, lm, a, on, inc = d[keep], c[keep], lm[keep], a[keep], on[keep], inc[keep]
    t = F.tail_weights({th: x[keep] for th, x in tb.items()}, inc, inc_i)
    reg = np.where(c == country, 0, F.region[c] + (len(REGIONS) - 1) * (lm != landmass))
    band = np.clip(np.searchsorted(BANDS, d, side="right") - 1, 0, NB - 1)
    fine = np.clip(np.searchsorted(FINE, d, side="right") - 1, 0, NF - 1)
    g = np.bincount(band * NR + reg, weights=on * np.maximum(d, D0) ** -A_GROUND * a, minlength=NB * NR).reshape(NB, NR)
    ont = np.clip((d - TAIL_ON[0]) / (TAIL_ON[1] - TAIL_ON[0]), 0, 1)
    h = np.stack([np.bincount(fine * NR + reg, weights=ont * x, minlength=NF * NR).reshape(NF, NR) for x in t])
    return g, h.astype(np.float32)


def sample_cells(wf, mask, n, rng):
    """n r5 cells drawn by population (with replacement), so each draw stands for an equal share."""
    idx = np.nonzero(mask)[0]
    p = wf.r5.pop[idx] / wf.r5.pop[idx].sum()
    return idx[rng.choice(len(idx), size=n, p=p)]


def precompute(F):
    wf, r5 = F.wf, F.wf.r5
    rng = np.random.default_rng(376)
    us = r5.country == F.us
    lon, lat = r5.lon, r5.lat
    # NHTS's Northeast: New England and the Middle Atlantic (New York, New Jersey, Pennsylvania)
    ne = us & (((lat > 39.72) & (lon > -80.52)) | ((lat > 38.9) & (lat <= 39.72) & (lon > -75.55) & (lon < -73.8)))
    ofc = lambda iso: r5.country == next(i for i, c in enumerate(wf.countries) if c["iso2"] == iso)
    groups = {"us": sample_cells(wf, us, 800, rng), "ne": sample_cells(wf, ne, 400, rng),
              "world": sample_cells(wf, r5.pop > 0, 1200, rng), "jp": sample_cells(wf, ofc("JP"), 300, rng),
              "gb": sample_cells(wf, ofc("GB"), 200, rng), "in": sample_cells(wf, ofc("IN"), 200, rng)}
    members = [np.nonzero(W.km(m[1], m[2], lon, lat) <= m[3])[0] for m in METROS]
    groups["metro"] = np.concatenate(members)
    S = {}
    t0 = time.time()
    for name, idx in groups.items():
        gs, hs = zip(*(site_terms(F, lon[i], lat[i], r5.country[i], r5.landmass[i], F.inc[0][i]) for i in idx))
        S[name] = {"idx": idx, "g": np.stack(gs), "h": np.stack(hs), "landmass": r5.landmass[idx], "inc": F.inc[0][idx],
                   "rate": F.inc[0][idx] ** EPS, "pop": r5.pop[idx], "country": r5.country[idx]}
        print(f"  {name}: {len(idx)} sites, {time.time() - t0:.0f} s", flush=True)
    # pair terms: from each metro origin cell to each other metro's r5 cells
    oi, nm = groups["metro"], len(METROS)
    PG, PD = np.zeros((len(oi), nm)), np.zeros((len(oi), nm))
    PT = np.zeros((len(COMBOS), len(oi), nm))
    for J, m in enumerate(members):
        d = 2 * W.R_KM * np.arcsin(np.clip(np.linalg.norm(r5.xyz[oi][:, None, :] - r5.xyz[m][None, :, :], axis=2) / 2, 0, 1))
        on = np.clip((d - ON0) / (ON1 - ON0), 0, 1)
        PG[:, J] = (on * np.maximum(d, D0) ** -A_GROUND * F.a[0][m][None, :]).sum(1)
        ont = np.clip((d - TAIL_ON[0]) / (TAIL_ON[1] - TAIL_ON[0]), 0, 1)
        rel = np.minimum(F.inc[0][m][None, :] / F.inc[0][oi][:, None], 1)
        for q, (th, et) in enumerate(COMBOS):
            PT[q, :, J] = (ont * F.tb[th][0][m][None, :] * rel ** et).sum(1)
        PD[:, J] = W.km(METROS[J][1], METROS[J][2], lon[oi], lat[oi])
    S["pairs"] = {"G": PG, "T": PT, "D": PD,
                  "metro_of": np.concatenate([np.full(len(m), I) for I, m in enumerate(members)]),
                  "metro_country": np.array([np.bincount(r5.country[m], weights=r5.pop[m]).argmax() for m in members]),
                  "metro_land": np.array([np.bincount(r5.landmass[m], weights=r5.pop[m]).argmax() for m in members])}
    return S


def targets():
    nh = json.load(open(NHTS))
    i50 = nh["edges_km"].index(50)
    tg = {}
    for reg in ("us", "ne"):
        blk = nh["us" if reg == "us" else "northeast"]["noncommute"]
        tg[reg] = (np.array(blk["rate"][i50:]), np.array(blk["n"][i50:]))  # bands 50-60 ... 2500-5000
    fr = json.load(open(ROUTES, encoding="utf-8"))
    ap = {c: I for I, m in enumerate(METROS) for c in m[4].split()}
    nm = len(METROS)
    F = np.zeros((nm, nm))
    for a, b, pax, *_ in fr["routes"]:
        if a in ap and b in ap and ap[a] != ap[b]:
            F[ap[a], ap[b]] += pax; F[ap[b], ap[a]] += pax
    tg["air"] = F / 365  # segment passengers a day, both directions
    return tg


class Model:
    """form "split": ground and tail are separate segments, each with its own rate and normaliser,
    so a site's long-trip rate does not depend on how much lies nearby. form "one": T-362's design."""
    def __init__(self, S, tg, form):
        self.S, self.tg, self.form = S, tg, form
        self.i50 = list(BANDS).index(50)
        nm = len(METROS)
        self.dist = np.array([[float(W.km(METROS[I][1], METROS[I][2], np.array(METROS[J][1]), np.array(METROS[J][2])))
                               for J in range(nm)] for I in range(nm)])
        self.air_pairs = [(I, J) for I in range(nm) for J in range(I + 1, nm) if self.dist[I, J] >= 1000 and tg["air"][I, J] > 0]
        names = [m[0] for m in METROS]
        self.nec = [(names.index(a), names.index(b), t, w) for a, b, t, w in NEC]
        # flights weights: three distance classes weigh the same, and inside a class a pair weighs
        # as the square root of its size (big pairs are less noisy and matter more in the game)
        tgt = np.array([tg["air"][I, J] for I, J in self.air_pairs])
        cls = np.searchsorted([2500, 6000], [self.dist[I, J] for I, J in self.air_pairs], side="right")
        w2 = np.sqrt(tgt)
        for c in range(3):
            w2[cls == c] *= AIR_CLASS_WEIGHT / w2[cls == c].sum()
        self.air_w = np.sqrt(w2)

    def origin_u(self, grp, tho):
        """Tail departures lean to dense places too: (pop_r5 / 1e5)^THo over its US people-weighted
        mean, so the US average rate stays Rt x income."""
        if tho == 0:
            return 1.0
        us = self.S["_us_pop"]
        mean = (us * (np.maximum(us, 1) / UREF) ** tho).sum() / us.sum()
        return (np.maximum(grp["pop"], 1) / UREF) ** tho / mean

    def dep(self, grp, p, q):
        """Per site and person: departures a weekday by band and destination region; and the two
        multipliers a pair needs per site (flow = mg x ground sum + mt x K x tail sum)."""
        Kf = np.maximum(FINE_C, p["D1"]) ** -p["G"]
        bw = bweights(p, grp["inc"])[:, None, :]
        tail = np.einsum("sfr,f,fb->sbr", grp["h"][:, q], Kf, M_FB) * bw
        ground = grp["g"] * bw
        rate = grp["rate"]
        if self.form == "one":
            N = (ground + p["k"] * tail).sum(axis=(1, 2))
            m = p["R"] * rate / N
            return (ground + p["k"] * tail) * m[:, None, None], m, m * p["k"]
        mg = p["R"] * rate / ground.sum(axis=(1, 2))
        mt = p["Rt"] * rate * self.origin_u(grp, p.get("THo", 0.0)) / tail.sum(axis=(1, 2))
        return ground * mg[:, None, None] + tail * mt[:, None, None], mg, mt

    def pairs(self, p, q):
        S, P = self.S["metro"], self.S["pairs"]
        _, mg, mt = self.dep(S, p, q)
        K = np.maximum(P["D"], p["D1"]) ** -p["G"]
        bij = np.where(S["country"][:, None] == P["metro_country"][None, :], 1.0,
                       (p["b"] * S["inc"] ** p.get("eps_b", 0.0))[:, None])
        flow = S["pop"][:, None] * bij * (mg[:, None] * P["G"] + mt[:, None] * K * P["T"][q])
        T = np.zeros((len(METROS), len(METROS)))
        np.add.at(T, P["metro_of"], flow)
        np.fill_diagonal(T, 0)
        return T + T.T  # both directions, trips a weekday

    def residuals(self, p, q, detail=False):
        r, rows = [], []
        for reg, wblk in (("ne", NHTS_WEIGHT), ("us", 0.5 * NHTS_WEIGHT)):
            tgt, n = self.tg[reg]
            dep = self.dep(self.S[reg], p, q)[0]
            mod = dep.sum(2).mean(0)[self.i50:NB - 1]
            for b in range(len(tgt)):
                lo = BANDS[self.i50 + b]
                if lo >= 300:  # under 300 km is the ground curve's, fitted in T-276 and kept
                    r.append(wblk * min(1.0, math.sqrt(n[b] / 100)) * math.log(mod[b] / tgt[b]))
                rows.append(("nhts_" + reg, f"{lo:.0f}-{BANDS[self.i50 + b + 1]:.0f} km", mod[b], tgt[b]))
            if reg == "us":
                for c, t in US_ABROAD.items():
                    m = (dep[:, :, c] + dep[:, :, c + len(REGIONS) - 1]).sum(1).mean()
                    r.append(1.0 * math.log(m / t))
                    rows.append(("us_abroad", REGIONS[c] + ", legs a day", m, t))
                far = dep[:, NB - 1, :].sum(1).mean()
                r.append(0.7 * math.log(far / US_OVER_5000))
                rows.append(("us_abroad", "over 5,000 km, legs a day", far, US_OVER_5000))
        for iso, t in ABROAD.items():  # other countries' residents abroad, all regions
            m = self.dep(self.S[iso], p, q)[0][:, :, 1:].sum((1, 2)).mean()
            r.append(1.0 * math.log(m / t))
            rows.append(("abroad", f"{iso.upper()} residents abroad, legs a day", m, t))
        T = self.pairs(p, q)
        for (I, J), w in zip(self.air_pairs, self.air_w):
            tgt = CONNECT * self.tg["air"][I, J]
            r.append(w * math.log(T[I, J] / tgt))
            rows.append(("air", f"{METROS[I][0]} - {METROS[J][0]}", T[I, J], tgt, round(self.dist[I, J])))
        for I, J, tgt, w in self.nec:
            if w:
                r.append(w * math.log(T[I, J] / tgt))
            rows.append(("nec", f"{METROS[I][0]} - {METROS[J][0]}", T[I, J], tgt, round(self.dist[I, J])))
        return (np.array(r), rows) if detail else np.array(r)


F_SCALE = 0.5


def soft_l1_cost(r):
    """least_squares' cost for loss soft_l1."""
    return float(0.5 * F_SCALE ** 2 * np.sum(2 * (np.sqrt(1 + (r / F_SCALE) ** 2) - 1)))


def bweights(p, inc):
    """Per site, destination class weights: home 1, abroad b x (origin income)^eps_b."""
    b = p["b"] * inc ** p.get("eps_b", 0.0)
    w = np.repeat(b[:, None], NR, axis=1)
    w[:, 0] = 1.0
    return w


def unpack(x, combo, form):
    th, et = combo
    p = {"R": R_GROUND, "G": x[1], "D1": math.exp(x[2]), "b": 1 / (1 + math.exp(-x[3])), "TH": th, "eps_t": et,
         "THo": x[4] if form == "split" else 0.0, "eps_b": x[5] if form == "split" else 0.0}
    p["k" if form == "one" else "Rt"] = math.exp(x[0])
    return p


def fit(model):
    """Least squares on log ratios (soft L1, so a few hub pairs cannot pull it), TH and eps_t on a grid."""
    best, form, key = None, model.form, "k" if model.form == "one" else "Rt"
    for q, combo in enumerate(COMBOS):
        x0 = np.array([math.log(1e-4 if form == "one" else 0.005), 1.0, math.log(300), math.log(0.4 / 0.6), 0.2, 0.5])
        lb = [math.log(1e-8 if form == "one" else 1e-4), 0.2, math.log(50), -5, 0.0, 0.0]
        ub = [math.log(1e-1), 4.0, math.log(3000), 4, 1.5, 4.0]
        if form == "one":
            x0, lb, ub = x0[:4], lb[:4], ub[:4]
        res = least_squares(lambda x: model.residuals(unpack(x, combo, form), q), x0, bounds=(lb, ub),
                            loss="soft_l1", f_scale=F_SCALE)
        p = unpack(res.x, combo, form)
        print(f"  {form} TH {combo[0]} eps_t {combo[1]}: cost {res.cost:.3f}  {key} {p[key]:.4g}  G {p['G']:.3f}"
              f"  D1 {p['D1']:.0f}  b {p['b']:.3f}  THo {p['THo']:.3f}  eps_b {p['eps_b']:.3g}", flush=True)
        if best is None or res.cost < best[0]:
            best = (res.cost, p, q)
    return best


def checks(model, p, q):
    S, out, b = model.S, {}, list(BANDS)
    wf = S["_wf"]
    for name in ("us", "ne", "world", "jp", "gb", "in"):
        dep = model.dep(S[name], p, q)[0]
        tot = dep.sum((1, 2))
        o = {"departures_per_day": float(tot.mean()),
             "abroad_share": float(dep[:, :, 1:].sum() / tot.sum()),
             "over_1000_per_day": float(dep[:, b.index(1000):, :].sum((1, 2)).mean()),
             "over_5000_share": float(dep[:, NB - 1, :].sum() / tot.sum()),
             "abroad_trips_per_head_year": float(365 * dep[:, :, 1:].sum((1, 2)).mean() / 2)}  # two legs a trip
        if name == "jp":  # domestic trips of 100 km and more by band, as MLIT's survey would see them
            home = dep[:, :, 0]
            t = home[:, b.index(100):].sum()
            for lo, hi in ((100, 300), (300, 700), (700, 1000), (1000, None)):
                o[f"dom_{lo}_{hi or 'up'}"] = float(home[:, b.index(lo):(b.index(hi) if hi else None)].sum() / t)
            o["dom_over_300_per_day"] = float(home[:, b.index(300):].sum(1).mean())
        out[name] = o
    grp = S["world"]
    dep = model.dep(grp, p, q)[0]
    byc = {}
    for c in np.unique(grp["country"]):
        m = grp["country"] == c
        if m.sum() >= 25:
            tot = dep[m].sum()
            byc[wf.countries[c]["iso2"]] = {"n": int(m.sum()), "dep": float(dep[m].sum((1, 2)).mean()),
                                            "abroad": float(dep[m][:, :, 1:].sum() / tot),
                                            "over_5000": float(dep[m][:, NB - 1, :].sum() / tot)}
    out["world_by_country"] = byc
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", default="split", choices=["one", "split", "both"])
    ap.add_argument("--nhts-weight", type=float, default=3.0)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    global NHTS_WEIGHT
    NHTS_WEIGHT = args.nhts_weight
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    F = Field()
    print(f"field {time.time() - t0:.1f} s", flush=True)
    S = precompute(F)
    S["_wf"] = F.wf
    S["_us_pop"] = F.wf.r5.pop[F.wf.r5.country == F.us]
    tg = targets()
    qt = COMBOS.index((TRIAL["TH"], TRIAL["eps_t"]))
    trial = Model(S, tg, "one")
    r0, rows0 = trial.residuals(TRIAL, qt, detail=True)
    for form in (["one", "split"] if args.form == "both" else [args.form]):
        model = Model(S, tg, form)
        print(f"{form}: {len(model.air_pairs)} flights pairs over 1,000 km", flush=True)
        cost, p, q = fit(model)
        r1, rows1 = model.residuals(p, q, detail=True)
        table = [list(a) + [c[2]] for a, c in zip(rows1, rows0)]  # kind, label, after, target, [km], before
        air = [t for t in table if t[0] == "air"]
        dom = [t[2] / t[3] for t, (I, J) in zip(air, model.air_pairs)
               if S["pairs"]["metro_country"][I] == S["pairs"]["metro_country"][J]]
        summary = {"form": form, "params": p, "cost": cost, "trial_cost": soft_l1_cost(r0),
                   "air_ratio_median": float(np.median([t[2] / t[3] for t in air])),
                   "air_ratio_dom_median": float(np.median(dom)),
                   "air_within_2x": float(np.mean([(1 / 2 <= t[2] / t[3] <= 2) for t in air])),
                   "checks_after": checks(model, p, q), "checks_before": checks(trial, TRIAL, qt), "table": table}
        tag = form + ("" if NHTS_WEIGHT == 3 else f"_w{NHTS_WEIGHT:g}")
        json.dump(summary, open(OUT / f"fit_{tag}.json", "w"), indent=1, default=float)
        print(json.dumps({k: v for k, v in summary.items() if k not in ("table", "checks_before")}, indent=1, default=float))
        if args.write and args.form != "both":
            out = {"task": "T-376", "date": time.strftime("%Y-%m-%d"), "form": form,
                   "R": R_GROUND, "A": A_GROUND, "onset_km": [ON0, ON1], "d0_km": D0, "eps": EPS,
                   "G": round(p["G"], 3), "D1_km": round(p["D1"], 1), "TH": p["TH"], "eps_t": p["eps_t"], "THo": round(p["THo"], 3),
                   "tail_onset_km": list(TAIL_ON), "border_eps": round(p["eps_b"], 3),
                   "u_mean_us": round(float(model.origin_u({"pop": np.array([UREF])}, p["THo"]) ** -1), 4),
                   "u_ref_people": UREF, "border": round(p["b"], 3), "y_us": round(F.y_us, 1),
                   "notes": "notes/T-376.md"}
            key = "k" if form == "one" else "Rt"
            out[key] = float(f"{p[key]:.4g}")
            json.dump(out, open(PARAMS, "w"), indent=1)
            print("wrote", PARAMS)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
