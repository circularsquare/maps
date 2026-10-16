"""T-456: what a plausibility check sees in a sector source that mixes wholesale into retail (Japan's I)."""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, "japan_muni.csv"), dtype={"code": str}).set_index("code")
d = d[d.e_lvl.astype(str).isin(["0", "2", "3"]) & (d["pop"] > 0)]
d["o_err"] = d.o_shop + d.o_eam
d["ret"] = d.p_ret + d.p_pers
d["mix"] = d.p_ret + d.p_whl + d.p_pers
nat = {k: d[k].sum() for k in ("pop", "jobs", "ret", "mix", "o_err")}
print("national: retail+personal persons per 1k residents %.0f, with wholesale %.0f; per 1k jobs %.0f / %.0f" % (
    1000 * nat["ret"] / nat["pop"], 1000 * nat["mix"] / nat["pop"], 1000 * nat["ret"] / nat["jobs"], 1000 * nat["mix"] / nat["jobs"]))
d["group"] = d.index.str[:2]
rows = []
for g, y in d.groupby("group"):
    for k in ("ret", "mix"):
        s = y[k] / y[k].sum()
        # per unit: sector persons per OSM errand place, against the group's
        per = (y[k] / (y.o_err + 1)) / (y[k].sum() / (y.o_err.sum() + len(y)))
        # per unit: sector share of the unit's jobs against the national sector share of jobs
        sj = (y[k] / y.jobs) / (nat[k] / nat["jobs"])
        top = s.nlargest(3).sum()
        rows.append({"group": g, "k": k, "per1k": 1000 * y[k].sum() / y["pop"].sum(),
                     "top3": top, "max_per_osm": per[y[k] > 0.01 * y[k].sum()].max(),
                     "max_jobshare_ratio": sj[y[k] > 0.01 * y[k].sum()].max()})
r = pd.DataFrame(rows).pivot(index="group", columns="k")
print(r.round(2).loc[["13", "27", "23", "14", "01", "40", "04", "05", "39", "43"]].to_string())
print("\nacross prefectures, median (max) of max_jobshare_ratio: ret %.2f (%.2f), mix %.2f (%.2f)" % (
    r[("max_jobshare_ratio", "ret")].median(), r[("max_jobshare_ratio", "ret")].max(),
    r[("max_jobshare_ratio", "mix")].median(), r[("max_jobshare_ratio", "mix")].max()))
print("units with sector share of jobs > 2x national: ret %d, mix %d (of %d)" % (
    ((d.ret / d.jobs) / (nat["ret"] / nat["jobs"]) > 2).sum(), ((d.mix / d.jobs) / (nat["mix"] / nat["jobs"]) > 2).sum(), len(d)))
print("\nshare of the sector in units where its share is > 3x the unit's share of OSM errand places (and of residents):")
out = []
for g, y in d.groupby("group"):
    so = (y.o_err + 1) / (y.o_err + 1).sum()
    sp = y["pop"] / y["pop"].sum()
    rr = {}
    for k in ("ret", "mix"):
        s = y[k] / y[k].sum()
        rr[k] = s[(s > 3 * so) & (s > 3 * sp)].sum()
    out.append((g, rr["ret"], rr["mix"]))
o = pd.DataFrame(out, columns=["group", "ret", "mix"]).set_index("group")
print(o.round(3).loc[["13", "27", "23", "14", "01", "40", "04"]].T.to_string())
print("median over prefectures: ret %.3f, mix %.3f; max ret %.3f" % (o.ret.median(), o.mix.median(), o.ret.max()))
w = d.p_whl /(d.p_whl + d.p_ret)
print("wholesale share of I: national %.2f; Chiyoda %.2f, Chuo %.2f, Minato %.2f, Osaka Kita %.2f, Osaka Chuo %.2f" % (
    d.p_whl.sum() / (d.p_whl.sum() + d.p_ret.sum()), w.get("13101", np.nan), w.get("13102", np.nan), w.get("13103", np.nan), w.get("27127", np.nan), w.get("27128", np.nan)))
