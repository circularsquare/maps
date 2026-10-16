"""T-456: completeness tables and the allocation test, for any country table with columns
pop, o_shop, o_eam, o_food, off_err (official errand establishments), off_food, group (region for
the allocation test), and optionally jobs, income. Usage: analyse.py <country>"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
K = sys.argv[1]


def load():
    if K == "jp":
        d = pd.read_csv(os.path.join(HERE, "japan_muni.csv"), dtype={"code": str}).set_index("code")
        d = d[d.e_lvl.astype(str).isin(["0", "2", "3"]) & (d["pop"] > 0)]
        d["off_err"] = d.e_ret + d.e_pers
        d["off_food"] = d.e_food
        d["off_jobs_err"] = d.p_ret + d.p_pers
        d["group"] = d.index.str[:2]
        d["unit"] = d.index
        return d
    if K == "uk":
        d = pd.read_csv(os.path.join(HERE, "uk_lad.csv")).set_index("LAD23CD")
        d = d[d["pop"] > 0].fillna(0)
        d["off_err"] = d.e_ret + d.e_pers + d.e_repair
        d["off_food"] = d.e_food
        d["jobs"] = np.nan
        reg = pd.read_csv(os.path.join(HERE, "lad_region.csv")).set_index("LAD23CD").RGN
        d["group"] = reg.reindex(d.index).fillna("?")
        return d
    if K == "us":
        d = pd.read_csv(os.path.join(HERE, "us_county.csv"), dtype={"fips": str}).set_index("fips")
        d = d[d["pop"] > 0]
        d["group"] = d.index.str[:2]
        return d


d = load()
for c in ("o_shop", "o_eam", "o_food", "jobs", "income"):
    if c not in d:
        d[c] = np.nan
d["o_err"] = d.o_shop + d.o_eam
d["dens"] = d["pop"] / d.km2


def ratio_row(x, label):
    return (f"| {label} | {len(x)} | {x['pop'].sum() / 1e6:.1f}M | {1000 * x.o_shop.sum() / x['pop'].sum():.2f} | {1000 * x.off_err.sum() / x['pop'].sum():.2f} | "
            f"{x.o_shop.sum() / x.off_err.sum():.2f} | {x.o_err.sum() / x.off_err.sum():.2f} | {1000 * x.o_food.sum() / x['pop'].sum():.2f} | "
            f"{1000 * x.off_food.sum() / x['pop'].sum():.2f} | {x.o_food.sum() / x.off_food.sum():.2f} |")


print(f"\n### {K}\n")
print("| units | n | residents | OSM shops /1k | official errand est. /1k | shops / official | shops+errand amenities / official | OSM food /1k | official food /1k | food / official |")
print("|---|---|---|---|---|---|---|---|---|---|")
print(ratio_row(d, "all"))
bands = [(0, 200), (200, 1000), (1000, 4000), (4000, 10000), (10000, 1e9)]
for lo, hi in bands:
    x = d[(d.dens >= lo) & (d.dens < hi)]
    if len(x):
        print(ratio_row(x, f"density {lo}-{hi if hi < 1e9 else ''} /km2"))
if d.income.notna().any():
    q = pd.qcut(d.income, 4, labels=["poorest", "2nd", "3rd", "richest"])
    for k in ["poorest", "2nd", "3rd", "richest"]:
        print(ratio_row(d[q == k], f"income quartile {k}"))

# spread of completeness across units (population weighted percentiles)
comp = (d.o_shop / d.off_err).replace([np.inf], np.nan)
w = d["pop"]
o = np.argsort(comp.fillna(0).to_numpy())
cw = np.cumsum(w.to_numpy()[o]) / w.sum()
pct = {p: comp.fillna(0).to_numpy()[o][np.searchsorted(cw, p / 100)] for p in (10, 25, 50, 75, 90)}
print("\nshops/official, population-weighted percentiles across units: " + ", ".join(f"p{k} {v:.2f}" for k, v in pct.items()))
by = d.groupby("group").agg(o=("o_shop", "sum"), e=("off_err", "sum"), p=("pop", "sum"))
r = (by.o / by.e).sort_values()
print(f"by group (prefecture, region or state): lowest {r.index[0]} {r.iloc[0]:.2f}, median {r.median():.2f}, highest {r.index[-1]} {r.iloc[-1]:.2f}")
print("by group: " + ", ".join(f"{k} {v:.2f}" for k, v in r.items()))

# what predicts completeness: correlations across units (log), units with >= 5,000 people and >= 20 official
m = (d["pop"] >= 5000) & (d.off_err >= 20)
x = d[m]
lc = np.log((x.o_shop + 0.5) / x.off_err)
cands = {"log OSM shops per resident": np.log((x.o_shop + 0.5) / x["pop"]),
         "log OSM shops+food per resident": np.log((x.o_shop + x.o_food + 0.5) / x["pop"]),
         "log density": np.log(x.dens),
         "food share of OSM shops+food": x.o_food / (x.o_shop + x.o_food + 0.5),
         "log official errand est. per resident": np.log(x.off_err / x["pop"])}
if x.income.notna().any():
    cands["log income"] = np.log(x.income)
print(f"\ncorrelation with log(shops/official), {m.sum()} units:")
for k, v in cands.items():
    print(f"  {k}: r = {np.corrcoef(v, lc)[0, 1]:.2f}")
print(f"  sd of log OSM shops per resident {np.std(np.log((x.o_shop + 0.5) / x['pop'])):.2f}, sd of log official per resident {np.std(np.log(x.off_err / x['pop'])):.2f}")
if x.jobs.notna().any():
    print(f"  sd of log official per job {np.std(np.log(x.off_err / x.jobs)):.2f}")

# allocation test: within each group, share of official errand establishments by unit against candidate weights
def misplaced(wt):
    tot, err = 0.0, 0.0
    for g, y in d.groupby("group"):
        if len(y) < 3 or y.off_err.sum() == 0:
            continue
        a = y.off_err / y.off_err.sum()
        b = wt.loc[y.index]
        b = b / b.sum() if b.sum() > 0 else a * 0
        err += y["pop"].sum() * 50 * np.abs(a - b).sum()
        tot += y["pop"].sum()
    return err / tot


def share(v):
    return v / d.groupby("group")[v.name].transform("sum")


d["s_pop"], d["s_osm"] = share(d["pop"]), share(d.o_err)
cand = {"population": d.s_pop, "OSM errand places": d.s_osm, "geomean(pop, OSM)": np.sqrt(d.s_pop * d.s_osm)}
if d.jobs.notna().any():
    d["s_jobs"] = share(d.jobs)
    cand["jobs"] = d.s_jobs
    cand["geomean(jobs, OSM)  [fallback D]"] = np.sqrt(d.s_jobs * d.s_osm)
    gc = d.groupby("group").apply(lambda y: y.o_err.sum() / max(y["pop"].sum(), 1) * 1000)
    for ref in (10.0, 20.0):
        c = (d.group.map(gc) / ref).clip(0, 1)
        cand[f"jobs^(1-c/2) OSM^(c/2), ref {ref:.0f}/1k"] = d.s_jobs ** (1 - c / 2) * d.s_osm ** (c / 2)
print("\nmisplaced share of official errand establishments between units, within groups (%), population weighted:")
for k, v in cand.items():
    print(f"  {k}: {misplaced(v.fillna(0)):.1f}")
