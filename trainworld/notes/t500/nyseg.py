"""T-500 US check on nyc: gravity split by LODES OD's sector groups SI01-03 (goods; trade,
transport, utilities; all other services), margins and flows from LODES OD itself, by county."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "6")
import sys
import numpy as np
import pandas as pd
TW = r"C:\Users\anita\projects\maps\trainworld"
sys.path.insert(0, os.path.join(TW, "pipeline"))
import build_city as bc
from read_pack import load_pack
from fit_decay import Zones, local_xy, LODES_OD

city = "nyc"
header, a = load_pack(city)
z = Zones(header, a)
g = header["gravity"]
print({k: v for k, v in g.items() if not isinstance(v, (list, dict))}, flush=True)
d = z.d.astype(np.float64)
d[d == z.intra] = g.get("intrazonal_km", z.intra)
reg = a["zone_region"].astype(np.int64) if "zone_region" in a else np.zeros(z.n, np.int64)
bf = np.where(reg[:, None] != reg[None, :], g.get("border_factor", 1.0), 1.0)
lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
blocks = pd.read_csv(os.path.join(bc.WORK, f"{city}.blocks.csv.gz"), dtype={"geoid": str}, usecols=["geoid", "lat", "lon"])
bx, by = local_xy(blocks["lat"].to_numpy(), blocks["lon"].to_numpy(), lon0, lat0)
bzone = z.of_xy(bx, by)
counties = sorted(bc.CITIES[city]["counties"]); cidx = {c: i for i, c in enumerate(counties)}; nc = len(counties)
bcounty = blocks["geoid"].str[:5].map(cidx).to_numpy()
bi = pd.Series(np.arange(len(blocks)), index=blocks["geoid"])

cs = set(counties)
parts = []
for fips in sorted({c[:2] for c in counties}):
    st = bc.STATES[fips]
    for part in ("main", "aux"):
        p = bc.fetch(LODES_OD.format(st=st, part=part, year=2023))
        for ch in pd.read_csv(p, usecols=["w_geocode", "h_geocode", "S000", "SI01", "SI02", "SI03"],
                              dtype={"w_geocode": str, "h_geocode": str}, chunksize=2_000_000):
            parts.append(ch[ch.w_geocode.str[:5].isin(cs) & ch.h_geocode.str[:5].isin(cs)])
od = pd.concat(parts, ignore_index=True)
hi = bi.reindex(od.h_geocode).to_numpy(); wi = bi.reindex(od.w_geocode).to_numpy()
ok = ~(np.isnan(hi) | np.isnan(wi)); od = od[ok]; hi = hi[ok].astype(np.int64); wi = wi[ok].astype(np.int64)
zh, zw = bzone[hi], bzone[wi]; ok = (zh >= 0) & (zw >= 0)
od, hi, wi, zh, zw = od[ok], hi[ok], wi[ok], zh[ok], zw[ok]
SEGS = ["SI01", "SI02", "SI03"]
H = np.zeros((z.n, 3)); J = np.zeros((z.n, 3)); C = np.zeros((3, nc, nc))
for k, s in enumerate(SEGS):
    v = od[s].to_numpy(np.float64)
    np.add.at(H[:, k], zh, v); np.add.at(J[:, k], zw, v)
    np.add.at(C[k], (bcounty[hi], bcounty[wi]), v)
print("LODES jobs by group:", {s: int(J[:, k].sum()) for k, s in enumerate(SEGS)}, flush=True)
# zone -> county shares by the LODES ends themselves
Pz = np.zeros((z.n, nc)); Qz = np.zeros((z.n, nc))
np.add.at(Pz, (zh, bcounty[hi]), od.S000.to_numpy(np.float64)); np.add.at(Qz, (zw, bcounty[wi]), od.S000.to_numpy(np.float64))
Pz /= np.maximum(Pz.sum(1, keepdims=True), 1e-9); Qz /= np.maximum(Qz.sum(1, keepdims=True), 1e-9)
Call = C.sum(0)
MAN = cidx["36061"]
OUTER = [cidx[c] for c in ("36005", "36047", "36081", "36085") if c in cidx]
SUB = [i for i, c in enumerate(counties) if c not in ("36061", "36005", "36047", "36081", "36085")]


def solve(F, o, dj, iters=400, tol=1e-4):
    col = np.ones(len(dj))
    for it in range(iters):
        row = np.where(o > 0, o / np.maximum(F @ col, 1e-300), 0)
        col = np.where(dj > 0, dj / np.maximum(row @ F, 1e-300), 0)
        if it % 10 == 9:
            e = np.max(np.abs((F @ col)[o > 0] * row[o > 0] / o[o > 0] - 1))
            if e < tol:
                break
    return F * row[:, None] * col[None, :]


def miss(x, y):
    return 0.5 * float(np.abs(x / x.sum() - y / y.sum()).sum())


def sh(M, rows, cols):
    return M[rows][:, cols].sum() / M[rows].sum()


def flows(T):
    return Pz.T @ T @ Qz


def F_of(a, L):
    return np.exp(-d / L - a * np.log(d)) * bf


def report(tag, M, Ms):
    print(f"{tag}: county misplaced {miss(M, Call):.1%} own county {np.trace(M) / M.sum():.1%} outer boroughs->Manhattan "
          f"{sh(M, OUTER, [MAN]):.1%} suburbs->Manhattan {sh(M, SUB, [MAN]):.1%} Manhattan->Manhattan {sh(M, [MAN], [MAN]):.1%} | by group "
          + " ".join(f"{s} {miss(Ms[k], C[k]):.1%}" for k, s in enumerate(SEGS)), flush=True)


a0, L0 = g["decay_pow"], g["decay_km"]
F0 = F_of(a0, L0)
T = solve(F0, H.sum(1) * J.sum() / H.sum(), J.sum(1))
M1 = flows(T)
report(f"single gravity d^-{a0:.2f} exp(-d/{L0:g})", M1, [M1] * 3)
Ms = [flows(solve(F0, H[:, k], J[:, k])) for k in range(3)]
report("by group, pack curve", sum(Ms), Ms)
best = []
for k, s in enumerate(SEGS):
    res = []
    for a_ in (0.4, 0.8, 1.2, 1.6):
        for L in (15, 30, 60):
            Mk = flows(solve(F_of(a_, L), H[:, k], J[:, k]))
            res.append((miss(Mk, C[k]), a_, L))
    res.sort()
    print(f"  {s}: best " + "; ".join(f"d^-{a_} exp(-d/{L}) {m:.1%}" for m, a_, L in res[:3]), flush=True)
    best.append(res[0][1:])
Ms = [flows(solve(F_of(*best[k]), H[:, k], J[:, k])) for k in range(3)]
report("by group, fitted " + " ".join(f"{s} {a_}/{L}" for s, (a_, L) in zip(SEGS, best)), sum(Ms), Ms)
print(f"LODES: own county {np.trace(Call) / Call.sum():.1%} outer->Manhattan {sh(Call, OUTER, [MAN]):.1%} "
      f"suburbs->Manhattan {sh(Call, SUB, [MAN]):.1%} Manhattan->Manhattan {sh(Call, [MAN], [MAN]):.1%}")
for k, s in enumerate(SEGS):
    print(f"  {s}: outer->Manhattan {sh(C[k], OUTER, [MAN]):.1%} suburbs->Manhattan {sh(C[k], SUB, [MAN]):.1%}; jobs in Manhattan share {C[k][:, MAN].sum() / C[k].sum():.1%}")
