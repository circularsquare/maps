"""T-379: New York - Philadelphia in the world market, decomposed by distance band, purpose and county.

Recomputes the near-pair formula of sim/src/demand/intercity.rs (Market::eachway, ground curve only:
the tail is 0 under 300 km) from the two market packs, checks it against market_pack's totals, then
splits the trips.

    python decomp.py <trainworld root> <out.json> [onset0 onset1]
"""
import json, math, sys
from collections import defaultdict
import numpy as np
from shapely.geometry import shape, Point
from shapely.strtree import STRtree

root = sys.argv[1]
out = sys.argv[2]
reg = json.load(open(f"{root}/app/src/game/registry.json", encoding="utf-8"))
rule = reg["market"]["rule"]
if len(sys.argv) > 4:
    rule = dict(rule, onset_km=[float(sys.argv[3]), float(sys.argv[4])])
R_E = 6371.0088

counties = json.load(open(f"{root}/data/work/tiger_counties_simple.json"))
keep = {k: shape(v) for k, v in counties.items() if k[:2] in ("09", "10", "24", "34", "36", "42", "25", "44", "11", "51")}
cids = list(keep)
tree = STRtree([keep[c] for c in cids])

# NEC FUTURE's metros (OMB 2009 MSAs, DEIS 5.2.2)
NY_MSA = {"36005", "36047", "36061", "36081", "36085", "36059", "36103", "36119", "36087", "36079",
          "34003", "34013", "34017", "34019", "34023", "34025", "34027", "34029", "34031", "34035", "34037", "34039", "42103"}
PHL_MSA = {"42017", "42029", "42045", "42091", "42101", "34005", "34007", "34015", "34033", "10003", "24015"}
NAMES = {"34021": "Mercer", "42017": "Bucks", "34005": "Burlington", "42101": "Philadelphia", "42091": "Montgomery", "34007": "Camden",
         "36061": "Manhattan", "36047": "Brooklyn", "36081": "Queens", "36005": "Bronx", "36085": "Staten Island", "36059": "Nassau", "36103": "Suffolk",
         "34023": "Middlesex", "34025": "Monmouth", "34029": "Ocean", "34035": "Somerset", "34019": "Hunterdon", "34039": "Union", "34013": "Essex",
         "34017": "Hudson", "34003": "Bergen", "34027": "Morris", "34031": "Passaic", "36119": "Westchester", "42045": "Delaware PA", "42029": "Chester",
         "34015": "Gloucester", "10003": "New Castle", "42077": "Lehigh", "42095": "Northampton", "34041": "Warren", "34037": "Sussex", "36087": "Rockland",
         "36071": "Orange NY", "34033": "Salem", "24015": "Cecil", "42089": "Monroe PA", "42103": "Pike"}


def county_of(lon, lat):
    p = Point(lon, lat)
    for i in tree.query(p):
        if keep[cids[i]].contains(p):
            return cids[i]
    i = tree.nearest(p)
    return cids[i]


def load(pid):
    pk = json.load(open(f"{root}/data/packs/{pid}.market.json", encoding="utf-8"))
    f = pk["frame"]
    lon0, lat0 = f["lon"], f["lat"]
    assert f["kind"] == "equirect"
    nz = len(pk["zones"])
    S = np.array(pk["sites"])
    pop, jobs, z = S[:, 2], S[:, 3], S[:, 4].astype(int)
    attr = pop + 2 * jobs
    k_attr = pop.sum() / attr.sum() * min(pk["inc"], 1) ** rule["eps"]
    s_inc = min(pk["inc"], 1) ** rule["eps"]
    w = pop + attr
    zx = np.bincount(z, S[:, 0] * w, nz) / np.maximum(np.bincount(z, w, nz), 1e-12)
    zy = np.bincount(z, S[:, 1] * w, nz) / np.maximum(np.bincount(z, w, nz), 1e-12)
    lon = lon0 + np.degrees(zx / (R_E * 1000 * math.cos(math.radians(lat0))))
    lat = lat0 + np.degrees(zy / (R_E * 1000))
    ng = np.array(pk["zone_ng"])
    g = np.bincount(z, pop * rule["R"] * s_inc / ng[z], nz)
    a = np.bincount(z, attr * k_attr, nz)
    zpop = np.bincount(z, pop, nz)
    zjobs = np.bincount(z, jobs, nz)
    cty = [county_of(lon[k], lat[k]) for k in range(nz)]
    u = np.stack([np.cos(np.radians(lat)) * np.cos(np.radians(lon)), np.cos(np.radians(lat)) * np.sin(np.radians(lon)), np.sin(np.radians(lat))], 1)
    return dict(pk=pk, nz=nz, lon=lon, lat=lat, g=g, a=a, pop=zpop, jobs=zjobs, cty=cty, u=u, place=pk["place"])


def ground(d):
    o0, o1 = rule["onset_km"]
    on = np.clip((d - o0) / (o1 - o0), 0, 1)
    return np.where(on > 0, on / np.maximum(d, rule["d0_km"]) ** rule["A"], 0.0)


def pair(A, B, same=False):
    c = np.clip(np.linalg.norm(A["u"][:, None, :] - B["u"][None, :, :], axis=2) / 2, 0, 1)
    d = 2 * R_E * np.arcsin(c)
    f = ground(d)
    t = 0.5 * (A["g"][:, None] * B["a"][None, :] + B["g"][None, :] * A["a"][:, None]) * f
    if same:
        t = np.triu(t, 1)
    t = np.where(t >= rule["min_trips"], t, 0.0)
    return d, t


def nm(c):
    return NAMES.get(c, c)


res = {}
nyc, phl = load("nyc"), load("phl")
d, t = pair(nyc, phl)
both = 2 * t
res["nyc_phl_both"] = both.sum()
res["nyc_phl_over80"] = both[d >= 80].sum()
print(f"New York - Philadelphia: {both.sum():,.0f} both ways, {both[d >= 80].sum():,.0f} over 80 km (market_pack: 398,447 and 301,050)")
print(f"people: nyc {nyc['pop'].sum():,.0f}, phl {phl['pop'].sum():,.0f}")

# distance bands
edges = [25, 50, 80, 100, 120, 150, 200, 250]
bands = []
for lo, hi in zip(edges, edges[1:]):
    m = (d >= lo) & (d < hi)
    bands.append((f"{lo}-{hi}", both[m].sum()))
    print(f"  {lo}-{hi} km: {both[m].sum():,.0f}")
res["bands"] = bands

# by county pair, over 80 km and all
cp = defaultdict(lambda: [0.0, 0.0])
for i in range(nyc["nz"]):
    for j in range(phl["nz"]):
        if both[i, j] > 0:
            k = (nyc["cty"][i], phl["cty"][j])
            cp[k][0] += both[i, j]
            if d[i, j] >= 80:
                cp[k][1] += both[i, j]
top = sorted(cp.items(), key=lambda kv: -kv[1][1])[:25]
print("county pairs (New York side - Philadelphia side): all, over 80 km")
for (a, b), (al, ov) in top:
    print(f"  {nm(a)} - {nm(b)}: {al:,.0f}, {ov:,.0f}")
res["county_pairs"] = [[nm(a), nm(b), al, ov] for (a, b), (al, ov) in sorted(cp.items(), key=lambda kv: -kv[1][0])[:60]]

# by side county alone
for side, P, axis in (("New York side", nyc, 1), ("Philadelphia side", phl, 0)):
    tot = defaultdict(lambda: [0.0, 0.0])
    s_all = both.sum(axis=axis)
    s_ov = np.where(d >= 80, both, 0).sum(axis=axis)
    for k in range(P["nz"]):
        tot[P["cty"][k]][0] += s_all[k]
        tot[P["cty"][k]][1] += s_ov[k]
    print(side + ":")
    for c, (al, ov) in sorted(tot.items(), key=lambda kv: -kv[1][0])[:14]:
        print(f"  {nm(c)}: {al:,.0f}, {ov:,.0f}")
    res[side] = [[nm(c), al, ov] for c, (al, ov) in sorted(tot.items(), key=lambda kv: -kv[1][0])]

# NEC FUTURE's definition: the two 2009 MSAs, any distance
mn = np.array([c in NY_MSA for c in nyc["cty"]])
mp = np.array([c in PHL_MSA for c in phl["cty"]])
nec = both[np.ix_(mn, mp)].sum()
res["nec_def"] = nec
print(f"NEC FUTURE's metros (NY-NNJ-LI x Philadelphia-Camden-Wilmington, any distance): {nec:,.0f}")
merc = np.array([c == "34021" for c in nyc["cty"]])
res["mercer_phl"] = both[merc].sum()
print(f"Mercer - Philadelphia pack: {both[merc].sum():,.0f}; Mercer - Philadelphia MSA {both[np.ix_(merc, mp)].sum():,.0f}")
# the Philadelphia pack beyond its MSA
print(f"phl pack people outside the Philadelphia MSA: {phl['pop'][~mp].sum():,.0f}; nyc pack people outside the NY MSA: {nyc['pop'][~mn].sum():,.0f}")
# Manhattan-centre to Centre City (both 20 km discs)
def disc(P, lon, lat, r):
    u = np.array([math.cos(math.radians(lat)) * math.cos(math.radians(lon)), math.cos(math.radians(lat)) * math.sin(math.radians(lon)), math.sin(math.radians(lat))])
    return 2 * R_E * np.arcsin(np.clip(np.linalg.norm(P["u"] - u, axis=1) / 2, 0, 1)) < r
for r in (20, 40):
    dn, dp = disc(nyc, -73.985, 40.758, r), disc(phl, -75.1636, 39.9526, r)
    v = both[np.ix_(dn, dp)].sum()
    res[f"disc{r}"] = v
    print(f"centres within {r} km of Midtown and of City Hall: {v:,.0f}")

# commutes between zones (the joint gravity's, cross-pack)
def commutes(P, to_place, Q):
    pk = P["pk"]
    tot, by = 0.0, defaultdict(float)
    for h, wp, w, n in zip(pk["commute_home"], pk["commute_place"], pk["commute_work"], pk["commute_trips"]):
        if wp == to_place:
            tot += n
            by[(P["cty"][h], Q["cty"][w])] += n
    return tot, by
c1, b1 = commutes(nyc, phl["place"], phl)
c2, b2 = commutes(phl, nyc["place"], nyc)
res["commuters"] = [c1, c2]
print(f"commuters a day: New York pack to Philadelphia {c1:,.0f}, Philadelphia to New York {c2:,.0f}; legs {2 * (c1 + c2):,.0f}")
for (h, w), n in sorted(list(b1.items()) + list(b2.items()), key=lambda kv: -kv[1])[:10]:
    print(f"  {nm(h)} -> {nm(w)}: {n:,.0f}")
res["commute_top"] = [[nm(h), nm(w), n] for (h, w), n in sorted(list(b1.items()) + list(b2.items()), key=lambda kv: -kv[1])[:20]]

# where New York's and Philadelphia's departures go, by band: share to each other
for name, P, Q in (("New York", nyc, phl), ("Philadelphia", phl, nyc)):
    dep = P["g"].copy()  # g x N = departures, by zone; ground departures = pop x R
    print(f"{name}: ground departures {P['pop'].sum() * rule['R']:,.0f} a day")
c = np.clip(np.linalg.norm(nyc["u"][:, None, :] - phl["u"][None, :, :], axis=2) / 2, 0, 1)
f = ground(2 * R_E * np.arcsin(c))
ny_to = (nyc["g"][:, None] * phl["a"][None, :] * f).sum()
phl_to = (phl["g"][None, :] * nyc["a"][:, None] * f).sum()
res["directed"] = [ny_to, phl_to]
print(f"directed: New York residents to Philadelphia {ny_to:,.0f}, Philadelphia residents to New York {phl_to:,.0f}")
json.dump(res, open(out, "w"), indent=1)
