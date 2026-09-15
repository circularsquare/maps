"""Bolivia — does the 1992 census geography still describe the LAPOP 2010-2023 respondents?

Printed, decides nothing. `sources/bo.md` §6 has the reading; this is the evidence behind two
calls there: that the 1992 census is a witness for LAPOP's DEPARTMENT ordering, and that it is
NOT used to split departments into provinces.

Two questions:
  1. across the nine departments: LAPOP's pooled share against 1992's, Spearman, exact p.
  2. inside departments: each sampled province's departure from its department's share, LAPOP
     against 1992, n-weighted correlation, null = provinces shuffled within each department.
     Printed beside it: the between-province spread of 1992's departures against LAPOP's
     sampling variance, because a check needs power first. LAPOP's province samples are one or
     two sampling points each, so the p-value is optimistic.

The four groups are the 1992 card's: Catholic; evangelical, which in 1992 is every non-Catholic
Christian and is compared with LAPOP's Protestant, evangelical, Mormon and Witness answers; none,
with LAPOP's believers without a church and agnostics; and other. Shares are of the four named
groups, leaving out 1992's unknown.

Usage:
    python sources/bo_checks.py     after sources/bo_geo.py and sources/bo_census.py
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import bo
import spearman_null

GROUPS = {"Catholic": ([1], ["catolica"]),
          "non-Catholic Christian": ([2, 5, 6, 12], ["evangelica"]),
          "none": ([4, 11], ["ninguna"]),
          "other": ([3, 7, 10, 77], ["otras"])}
NAMED = ["catolica", "evangelica", "otras", "ninguna"]


def shares92(path, key):
    t = pd.read_csv(path, dtype={"code": str})
    t[key] = "BO" + t["code"].str.zfill(4 if key == "prov" else 2)
    base = t[NAMED].sum(axis=1)
    for g, (_, cols) in GROUPS.items():
        t[g] = t[cols].sum(axis=1) / base
    t["named"] = base
    return t.set_index(key)


def respondents(names, munis, pop):
    """The pooled LAPOP respondents with a COD province each, via `bo.decode_wave`."""
    adm3, adm2 = {}, {}
    for n3, n2, p2, p1 in zip(munis["adm3_name"], munis["adm2_name"], munis["adm2_pcode"],
                              munis["adm1_pcode"]):
        adm3.setdefault((p1, bo.fold(n3)), set()).add(p2)
        adm2.setdefault((p1, bo.fold(n2)), set()).add(p2)

    def province(dept, name):
        f = bo.fold(name)
        f = bo.MUNI_ALIAS.get(f, f)
        s = adm3.get((dept, f)) or adm2.get((dept, f))
        return next(iter(s)) if s and len(s) == 1 else None

    frames = []
    for y in bo.WAVES:
        d, lab, _ = bo.read_wave(y)
        d = bo.decode_wave(y, d, lab, munis, names)
        if y == 2023:
            d["provpc"] = d["prov"].astype(int).map(
                lambda p: f"BO{p // 100 % 100:02d}{p % 100:02d}")
        else:
            ml = lab["muni"]
            d["provpc"] = [province(g, ml.get(m)) if pd.notna(m) and ml.get(m) is not None
                           else None for g, m in zip(d["geo_id"], d["muni"])]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["rel"].notna()].copy()
    df["code"] = df["rel"].astype(int)
    return bo.poststratify(df, pop)


def main():
    lut = pd.read_csv(bo.LOOKUP, dtype={"geo_id": str})
    names = dict(zip(lut["geo_id"], lut["name"]))
    pop = pd.read_csv(bo.POP, dtype={"geo_id": str}).set_index("geo_id")["pop"]
    munis = pd.read_csv(bo.MUNIS, dtype=str)
    pn = pd.read_csv(os.path.join(bo.GEO, "bo_prov_pop_2024.csv"), dtype={"geo_id": str})
    provname = dict(zip(pn["geo_id"], pn["name"]))
    p92 = shares92(os.path.join(bo.RAW, "cpv1992_religion_provin.csv"), "prov")
    d92 = shares92(os.path.join(bo.RAW, "cpv1992_religion_depto.csv"), "dept")

    df = respondents(names, munis, pop)
    print(f"\n{len(df):,} respondents; province placed for {df['provpc'].notna().mean():.1%}, "
          f"{df['provpc'].nunique()} provinces sampled")
    bad = sorted(set(df["provpc"].dropna()) - set(p92.index))
    if bad:
        raise SystemExit(f"provinces not in 1992: {bad}")

    print("\n1. departments, LAPOP 2010-2023 pooled against 1992 (shares of the four named groups):")
    for g, (codes, _) in GROUPS.items():
        s = (df.assign(h=df["code"].isin(codes) * df["w"]).groupby("geo_id")["h"].sum()
             / df.groupby("geo_id")["w"].sum())
        c = d92.loc[s.index, g]
        rho = spearmanr(s, c).statistic
        n92 = float((d92[g] * d92["named"]).sum() / d92["named"].sum())
        nl = float((df["code"].isin(codes) * df["w"]).sum() / df["w"].sum())
        print(f"   {g:<24} Spearman {rho:+.3f} (exact p {spearman_null.exact_p(rho, 9):.4f}); "
              f"national 1992 {n92 * 100:.1f}% -> LAPOP {nl * 100:.1f}%")
        print("      " + "  ".join(f"{names[u][:5]} {s[u] * 100:.0f}/{c[u] * 100:.0f}"
                                   for u in s.index))

    print("\n2. provinces inside departments (n-weighted departures; null shuffles provinces "
          "within each department, 20,000 draws):")
    sub = df.dropna(subset=["provpc"])
    rng = np.random.default_rng(0)
    for g, (codes, _) in GROUPS.items():
        cell = sub.assign(h=sub["code"].isin(codes) * sub["w"]).groupby(
            ["geo_id", "provpc"]).agg(h=("h", "sum"), w=("w", "sum"), n=("w", "size")).reset_index()
        cell["s"] = cell["h"] / cell["w"]
        cell["c"] = p92.loc[cell["provpc"], g].to_numpy()
        cell = cell[cell["n"] >= 10]
        k = cell.groupby("geo_id")["provpc"].transform("size")
        cell = cell[k >= 2].reset_index(drop=True)
        grp = cell["geo_id"].to_numpy()
        n = cell["n"].to_numpy(float)
        idx = {u: np.nonzero(grp == u)[0] for u in np.unique(grp)}

        def dev(v):
            out = np.empty_like(v)
            for ii in idx.values():
                out[ii] = v[ii] - np.average(v[ii], weights=n[ii])
            return out

        def wcorr(a, b):
            return np.sum(n * a * b) / np.sqrt(np.sum(n * a * a) * np.sum(n * b * b))

        ds, cvals = dev(cell["s"].to_numpy()), cell["c"].to_numpy()
        obs = wcorr(ds, dev(cvals))
        null = np.empty(20000)
        for i in range(len(null)):
            perm = cvals.copy()
            for ii in idx.values():
                perm[ii] = cvals[rng.permutation(ii)]
            null[i] = wcorr(ds, dev(perm))
        p = (1 + int((null >= obs).sum())) / (1 + len(null))
        spread = np.average(dev(cvals) ** 2, weights=n)
        samp = np.average(cell["s"] * (1 - cell["s"]) / n, weights=n)
        print(f"   {g:<24} r = {obs:+.3f}, p = {p:.4f}, null95 {np.quantile(null, 0.95):+.3f}; "
              f"{len(cell)} provinces in {len(idx)} departments; 1992 spread / LAPOP sampling "
              f"variance = {spread / samp:.2f}")
        if g in ("Catholic", "non-Catholic Christian"):
            show = cell.assign(ds=ds, dc=dev(cvals)).sort_values("dc")
            for _, r in pd.concat([show.head(4), show.tail(4)]).iterrows():
                print(f"      {names[r['geo_id']][:10]:<11}"
                      f"{provname.get(r['provpc'], r['provpc'])[:22]:<23}n={int(r['n']):<5} "
                      f"LAPOP {r['s'] * 100:5.1f}%  1992 {r['c'] * 100:5.1f}%  departures "
                      f"{r['ds'] * 100:+5.1f} / {r['dc'] * 100:+5.1f}")


if __name__ == "__main__":
    main()
