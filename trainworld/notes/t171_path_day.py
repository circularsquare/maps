"""PATH on the real network day (t171net output): trips using PATH by purpose, by station,
transfers in, river crossings by operator, trip explorer answers."""
import json, sys, math, zipfile, io
import numpy as np
W = r"C:\Users\anita\projects\maps\trainworld\data\work\realnet"
S = r"C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\87037fc9-7e8f-4e0f-abce-7678aadf6b68\scratchpad\agent-demand\t171"
res_path = sys.argv[1] if len(sys.argv) > 1 else S + r"\day_all.json"
net = json.load(open(W + r"\nyc_real.json"))
res = json.load(open(res_path))
st = net["stations"]
lines = net["lines"]
smap = res["st_map"]
rn_line = np.array(res["rn_line"]); rn_st = np.array(res["rn_station"])
feed = np.array([lines[l]["feed"] for l in rn_line])
nd = max(smap) + 1
dname = {}
for i, s in enumerate(st):
    d = smap[i]
    dname.setdefault(d, []).append(s["name"][:22])
P = res["periods"]
KINDS = ["shopping", "social", "school", "university", "airport", "medical"]
nrn = len(rn_line)
board_c = sum(np.array(p["board"]) for p in P)
seg_c = sum(np.array(p["seg"]) for p in P)
ev = res["everyday"]
if ev:
    nk = len(P[0]["ev_kinds"]) // 3
    board_k = np.zeros((nk, nrn)); seg_k = np.zeros((nk, nrn))
    for p in P:
        parts = np.array(p["ev_parts"]).reshape(nk, 3, nrn)
        seg_k += parts[:, 0]; board_k += parts[:, 1]
    kt = np.zeros((nk, 3))
    for p in P:
        kt += np.array(p["ev_kinds"]).reshape(nk, 3)
ispath = feed == "path"

def xfers(arr):
    a = np.array(arr).reshape(-1, 3)
    pp = a[ispath[a[:, 0].astype(int)] & ispath[a[:, 1].astype(int)], 2].sum()
    into = a[~ispath[a[:, 0].astype(int)] & ispath[a[:, 1].astype(int)]]
    outof = a[ispath[a[:, 0].astype(int)] & ~ispath[a[:, 1].astype(int)]]
    return pp, into, outof

pp_c, into_c, out_c = xfers(res["xfer"])
b_c = board_c[ispath].sum()
print(f"COMMUTE: PATH boardings {b_c:,.0f}; PATH->PATH transfers {pp_c:,.0f}; trips using PATH {b_c - pp_c:,.0f}; "
      f"of which boarded from another operator {into_c[:, 2].sum():,.0f}; left PATH for another operator {out_c[:, 2].sum():,.0f}")
tot_trips = b_c - pp_c
if ev:
    pp_e, into_e, out_e = xfers(res["ev_xfer"])
    b_e = board_k[:, ispath].sum(1)
    print("EVERYDAY: PATH boardings by kind", {KINDS[k]: round(b_e[k]) for k in range(nk)})
    frac = 1 - pp_e / max(b_e.sum(), 1)
    print(f"  PATH->PATH transfers {pp_e:,.0f}; trips using PATH {b_e.sum() - pp_e:,.0f}")
    tot_trips += b_e.sum() - pp_e
    print(f"ALL: trips using PATH {tot_trips:,.0f}; commute share {(b_c - pp_c) / tot_trips:.0%}")
    print("everyday kind totals over the day (trips, rail, walk):")
    for k in range(nk):
        print(f"  {KINDS[k]}: {kt[k,0]:,.0f} trips, rail {kt[k,1]:,.0f} ({kt[k,1]/kt[k,0]:.1%}), walk {kt[k,2]/kt[k,0]:.0%}")
    print(f"  commute: rail {sum(p['summary'][2] for p in P):,.0f} of {sum(p['summary'][1] for p in P):,.0f}")
# by operator: trips using each operator (boardings less same-operator transfers)
print("\ntrips using each operator (boardings less transfers within the operator), commute / everyday:")
allx_c = np.array(res["xfer"]).reshape(-1, 3)
for f in ["subway", "path", "njt", "lirr", "mnr"]:
    m = feed == f
    bc = board_c[m].sum()
    be = board_k[:, m].sum() if ev else 0
    print(f"  {f}: boardings commute {bc:,.0f}, everyday {be:,.0f}")
# PATH boardings by station
print("\nPATH boardings by station (commute + everyday), and commute share; fresh starts vs from other operator:")
rows = []
for d in sorted(set(rn_st[ispath])):
    m = ispath & (rn_st == d)
    bc = board_c[m].sum(); be = board_k[:, m].sum() if ev else 0
    fromother = into_c[np.isin(into_c[:, 1].astype(int), np.where(m)[0]), 2].sum()
    rows.append((d, bc, be, fromother))
for d, bc, be, fo in sorted(rows, key=lambda r: -(r[1] + r[2])):
    print(f"  {'/'.join(dname[d])[:44]:44s} commute {bc:8,.0f}  everyday {be:8,.0f}  commute from other operator {fo:7,.0f}")
# river crossings: segments between stations in NJ and NY
import shapefile
from shapely.geometry import shape, Point
r = shapefile.Reader(r"C:\Users\anita\projects\maps\trainworld\data\raw\tl_2020_us_state.zip")
fi = [f[0] for f in r.fields[1:]].index("STUSPS")
polys = {sr.record[fi]: shape(sr.shape.__geo_interface__) for sr in r.iterShapeRecords() if sr.record[fi] in ("NJ", "NY", "CT")}
def state_of(s):
    p = Point(s["lon"], s["lat"])
    for k, g in polys.items():
        if g.contains(p):
            return k
    return min(polys, key=lambda k: polys[k].distance(p))
dstate = {}
for i, s in enumerate(st):
    dstate.setdefault(smap[i], state_of(s))
cross = {}
for rr in range(nrn - 1):
    if rn_line[rr] != rn_line[rr + 1]:
        continue
    a, b = dstate[rn_st[rr]], dstate[rn_st[rr + 1]]
    if {a, b} == {"NJ", "NY"}:
        f = feed[rr]
        c = cross.setdefault(f, [0.0, 0.0])
        c[0] += seg_c[rr]; c[1] += seg_k[:, rr].sum() if ev else 0
print("\nriders crossing the Hudson (NJ<->NY rail segments), both directions, a day: commute / everyday")
for f, (c, e) in cross.items():
    print(f"  {f}: {c:,.0f} / {e:,.0f}")
# trips
print("\ntrip explorer (morning peak):")
HEAD, PAIR_LEN, LEG_LEN = 32, 12, 12
names_d = {d: "/".join(v)[:30] for d, v in dname.items()}
for t in res["trips"]:
    v = t["am"]
    if v[0] != 1:
        print(" ", t["name"], "status", v[0]); continue
    line = (f"  {t['name']}: {v[6]/1000:.1f} km; walk {v[17]:.0%} rail {v[18]:.0%} drive {v[19]:.0%}; drive {v[9]:.1f} min x w = {v[10]:.1f} + park {v[11]:.1f} = {v[12]:.1f}; "
            f"rail {v[14]:.1f} perceived (walks {v[20]:.1f} + {v[21]:.1f}, fare {v[22]:.1f}); util {v[16]:.2f}")
    print(line)
    np_ = int(v[24]); o = HEAD + np_ * PAIR_LEN
    for k in range(min(np_, 3)):
        p = v[HEAD + k * PAIR_LEN: HEAD + (k + 1) * PAIR_LEN]
        print(f"     pair {p[9]:.0%}: {names_d.get(int(p[0]),'?')} ({p[4]:.0f} m, {p[5]:.1f}) -> {names_d.get(int(p[1]),'?')} ({p[6]:.0f} m, {p[7]:.1f}); strategy {p[8]:.1f}")
    for _ in range(int(v[25])):
        sh, n = v[o + 1], int(v[o + 2])
        legs = []
        for k in range(n):
            l = v[o + 3 + k * LEG_LEN: o + 3 + (k + 1) * LEG_LEN]
            if l[0] == 1:
                ln = lines[rn_line[int(l[1])]]
                legs.append(f"{ln['feed']} {ln['name']} {names_d.get(int(l[3]),'?')}->{names_d.get(int(l[4]),'?')} (wait {l[8]:.1f}, ride {l[5]:.1f})")
            else:
                legs.append(f"walk {l[5]:.1f} min")
        print(f"     route {sh:.0%}: " + "; ".join(legs))
        o += 3 + n * LEG_LEN
