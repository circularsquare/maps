"""Trips using each operator (boardings less transfers within the operator), commute and
everyday, against the operators' weekday counts; also realnet's old 'entries' measure."""
import json, sys
import numpy as np
W = r"C:\Users\anita\projects\maps\trainworld\data\work\realnet"
S = r"C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\87037fc9-7e8f-4e0f-abce-7678aadf6b68\scratchpad\agent-demand\t171"
res = json.load(open(sys.argv[1] if len(sys.argv) > 1 else S + r"\day_all2.json"))
net = json.load(open(W + r"\nyc_real.json"))
tg = json.load(open(W + r"\targets.json"))
lines = net["lines"]
rn_line = np.array(res["rn_line"])
LIGHT = {"4", "13", "17"}
def op(l):
    f = lines[l]["feed"]
    if f == "njt" and lines[l]["route"] in LIGHT:
        return "njt_light"
    return f
rop = np.array([op(l) for l in rn_line])
P = res["periods"]
nrn = len(rn_line)
board_c = sum(np.array(p["board"]) for p in P)
nk = len(P[0]["ev_kinds"]) // 3
board_e = np.zeros(nrn)
for p in P:
    board_e += np.array(p["ev_parts"]).reshape(nk, 3, nrn)[:, 1].sum(0)
def same_op_xfer(arr):
    a = np.array(arr).reshape(-1, 3)
    i, o = a[:, 0].astype(int), a[:, 1].astype(int)
    out = {}
    for f in set(rop):
        out[f] = a[(rop[i] == f) & (rop[o] == f), 2].sum()
    return out
xc, xe = same_op_xfer(res["xfer"]), same_op_xfer(res["ev_xfer"])
ops = tg["operators"]
real = {"subway": ops["subway"] + ops["sir"], "lirr": ops["lirr"], "mnr": ops["mnr"], "path": ops["path"], "njt": ops["njt_rough"], "njt_light": None}
print("| operator | real weekday (all trips) | model commute trips | x real | model everyday trips | model all | x real |")
print("|---|---|---|---|---|---|---|")
for f in ["subway", "path", "lirr", "mnr", "njt", "njt_light"]:
    m = rop == f
    c = board_c[m].sum() - xc.get(f, 0)
    e = board_e[m].sum() - xe.get(f, 0)
    r = real[f]
    rs = f"{r:,.0f}" if r else "?"
    print(f"| {f} | {rs} | {c:,.0f} | {c / r:.2f} | {e:,.0f} | {c + e:,.0f} | {(c + e) / r:.2f} |" if r else f"| {f} | ? | {c:,.0f} | | {e:,.0f} | {c + e:,.0f} | |")
# county rail shares (commute)
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
import realnet as rn
acs = tg["acs"]
print("\ncommute rail share by county of residence, ACS / model:")
for i, f in enumerate(rn.COUNTY_ORDER):
    a = acs[f]
    comm = a["001"] - a["021"]
    ra = (a["012"] + a["013"] + a["014"]) / comm
    r = sum(p["county"]["rail"][i] for p in P); t = sum(p["county"]["trips"][i] for p in P)
    if rn.COUNTIES[f] in ("Hudson", "Essex", "Bergen", "Union", "New York", "Kings", "Queens", "Fairfield CT", "Passaic", "Middlesex"):
        print(f"  {rn.COUNTIES[f]}: {ra:.1%} / {r / t:.1%}  (ACS subway {a['012'] / comm:.1%}, commuter rail {a['013'] / comm:.1%}, bus {a['011'] / comm:.1%}, ferry {a['015'] / comm:.1%})")
