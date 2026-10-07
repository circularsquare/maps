"""Israeli settlers beyond the Green Line (`xs`): CBS Social Survey 2021 native language applied to
the population of religiondots' `xs` units -> data/normalized/xs.csv.

    python sources/xs_social.py             # from the survey tables il_social.py cached

The same survey, the same model and the same code paths as Israel (sources/il_social.py, whose
docstring has the routes and the method); only the units and the survey cells differ.

UNITS AND POPULATION: religiondots' data/normalized/xs.csv, read-only: the 267 CBS 2022 units in
data/geo/il/dropped_units.json (West Bank settlements and East Jerusalem), Jews and the register's
Others only, 723,899 people. Asserted: every unit is a dropped unit (so none is on Israel's entry)
and the total is religiondots' figure. The Muslims and Christians in the same units stay on
Palestine's entry, as in religiondots.

SURVEY CELLS: everyone here is in the survey's "Jews and others" group. Units outside Jerusalem
take sub-district 71, Judea and Samaria (215,496 adults; it is its own district, so the shrink
towards the district changes nothing). East Jerusalem (locality 3000) is not in that cell: CBS
files it in the Jerusalem sub-district (11), so its Israelis take Jerusalem's Jews-and-others
shares, shrunk towards the Jerusalem district as on Israel's entry.

CHILDREN: each area's under-20s (sa2022 age0_19_pcnt; settlements are about half children) take
the national 20-24 shares of Jews and others, as on Israel's entry.

PLACEMENT ACROSS UNITS: raked (IPF) inside each of the two cells from the area's origin profile,
exactly as il_social.py: Yiddish by the Haredi share (religiondots' "Jews [Ultra-religious]"
rows), Russian by European origin, and so on. Inside a unit, religiondots' Kontur pieces.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "sources"))
from rdlink import RD  # noqa: E402
from il_social import (ETHIOPIA, FSU, K_SHRINK, LANGS, RD_DROPPED, RD_SA,  # noqa: E402
                       read_table, shares)

RD_XS = RD / "data" / "normalized" / "xs.csv"
OUT = HERE / "data" / "normalized" / "xs.csv"
EXPECTED = 723_899
JERUSALEM = 3000
CELL = {"jerusalem": "11", "judea_samaria": "71"}


def build():
    J = "Jews and others"
    sub = read_table("subdistrict")
    dist = read_table("district")
    age = read_table("age")
    young = shares(age[(age.group == J) & (age.code == "1")].iloc[0])
    dshare = {str(r.code): shares(r) for _, r in dist[(dist.group == J) & (dist.code != "-1")].iterrows()}
    adult = {}
    for _, r in sub[(sub.group == J) & (sub.code != "-1")].iterrows():
        code = str(r.code)
        lam = r.total / (r.total + K_SHRINK)
        adult[code] = lam * shares(r) + (1 - lam) * dshare[code[0]]
    for k, c in CELL.items():
        print(f"  cell {k} ({c}): " + ", ".join(f"{l} {s:.1%}" for l, s in zip(LANGS, adult[c]) if s > 0.005))

    x = pd.read_csv(RD_XS, dtype={"geo_id": str}, low_memory=False,
                    keep_default_na=False, na_values=[""])
    x = x[x.geo_level.isin(("statarea", "locality"))].copy()
    dropped = set(json.loads(RD_DROPPED.read_text()))
    outside = sorted(set(x.geo_id) - dropped)
    if outside:
        raise SystemExit(f"religiondots xs.csv has {len(outside)} units Israel's entry draws: {outside[:5]}")
    x["rel"] = x.source_category.str.replace(r" \[.*\]$", "", regex=True)
    if set(x.rel) != {"Jews", "Others"}:
        raise SystemExit(f"unexpected groups in religiondots xs.csv: {set(x.rel)}")
    x["haredi"] = np.where(x.source_category == "Jews [Ultra-religious]", x["count"], 0.0)
    u = x.groupby("geo_id").agg(J=("count", "sum"), haredi=("haredi", "sum"),
                                name=("geo_name", "first"), level=("geo_level", "first"))
    tot = u.J.sum()
    if abs(tot - EXPECTED) > 1:
        raise SystemExit(f"xs population {tot:,.0f}, expected {EXPECTED:,}")
    u["loc"] = u.index.str.split("_").str[0].astype(int)
    u["cell"] = np.where(u["loc"] == JERUSALEM, CELL["jerusalem"], CELL["judea_samaria"])
    print(f"  {len(u)} units, {tot:,.0f} people; East Jerusalem {u.J[u.cell == '11'].sum():,.0f} "
          f"on {int((u.cell == '11').sum())} units")

    sa = pd.read_csv(RD_SA)
    sa = sa.dropna(subset=["LocalityCode"]).copy()
    sa["loc"] = sa.LocalityCode.astype(int)
    sa["geo_id"] = np.where(sa.StatAreaCmb.isna(), sa["loc"].astype(str),
                            sa["loc"].astype(str) + "_" + sa.StatAreaCmb.astype(str))
    sa = sa.drop_duplicates("geo_id").set_index("geo_id")
    prof = ["age0_19_pcnt", "j_isr_pcnt", "j_abr_pcnt", "europe_pcnt", "africa_pcnt",
            "america_pcnt", "shem_eretz1"]
    nump = prof[:-1]
    p = u[[]].join(sa[prof])
    direct = u.index.isin(sa.index).mean()
    locp = sa[sa.StatAreaCmb.isna()].set_index("loc")[prof]
    p[nump] = p[nump].fillna(u[["loc"]].join(locp, on="loc")[nump])
    p[nump] = p[nump].fillna(p[nump].join(u.cell).groupby("cell")[nump].transform("mean"))
    p[nump] = p[nump].fillna(p[nump].mean())
    print(f"  census profile: {direct:.1%} of units matched directly; mean under-20 share "
          f"{np.average(p.age0_19_pcnt, weights=u.J):.1f}%")
    c = (p.age0_19_pcnt / 100).clip(0, 0.8)

    fl = 0.5
    abr = p.j_abr_pcnt + fl
    w = pd.DataFrame({
        "Hebrew": p.j_isr_pcnt + fl,
        "Arabic": abr,
        "Russian": (p.europe_pcnt + fl) * np.where(p.shem_eretz1.isin(FSU), 2, 1),
        "English": p.america_pcnt + fl,
        "French": p.europe_pcnt + p.africa_pcnt + fl,
        "Spanish": p.america_pcnt + fl,
        "Yiddish": (u.haredi / u.J.replace(0, np.nan)).fillna(0) * 100 + fl,
        "Amharic": (p.africa_pcnt + fl) * np.where(p.shem_eretz1 == ETHIOPIA, 4, 1),
        "AnotherLanguage": abr,
    }, index=u.index)[LANGS]
    Jc = pd.DataFrame(0.0, index=u.index, columns=LANGS)
    targets = {}
    for n, idx in u.groupby("cell").groups.items():
        Ju = u.loc[idx, "J"].to_numpy()
        cu = c.loc[idx].to_numpy()
        target = (Ju * (1 - cu)).sum() * adult[n] + (Ju * cu).sum() * young
        targets[n] = target
        ww = w.loc[idx].to_numpy()
        ww = ww / np.average(ww, axis=0, weights=Ju + 1e-9)
        M = Ju[:, None] * target[None, :] / max(target.sum(), 1e-9) * ww
        for _ in range(200):
            rs = M.sum(1)
            M *= np.where(rs > 0, Ju / np.where(rs > 0, rs, 1), 0)[:, None]
            cs = M.sum(0)
            M *= np.where(cs > 0, target / np.where(cs > 0, cs, 1), 0)[None, :]
        err = np.abs(M.sum(1) - Ju).max()
        if err > 1:
            raise SystemExit(f"cell {n}: raking left a unit {err:.1f} off its population")
        Jc.loc[idx] = M
    Jc = Jc.mul(u.J / Jc.sum(1).replace(0, 1), axis=0)
    for n, idx in u.groupby("cell").groups.items():
        dev = np.abs(Jc.loc[idx].sum(0).to_numpy() - targets[n]).max()
        if dev > 5:
            raise SystemExit(f"cell {n}: a language total is {dev:.1f} off the survey's")

    rows = []
    for gid in u.index:
        for l in LANGS:
            v = Jc.at[gid, l]
            if v > 0:
                rows.append((gid, u.at[gid, "level"], u.at[gid, "name"], f"{l} (Jews and others)", v))
    out = pd.DataFrame(rows, columns=["geo_id", "geo_level", "geo_name", "source_category", "count"])
    out["count"] = out["count"].round(3)
    out["tier"] = "modelled"
    out["source_id"] = "cbs_social_survey_2021"
    out["year"] = 2021
    out["note"] = ("native language 20+, Jews and others, Judea and Samaria or Jerusalem "
                   "sub-district; CBS 2022 population of religiondots' xs units")
    if abs(out["count"].sum() - tot) > 5:
        raise SystemExit(f"xs.csv sums to {out['count'].sum():,.0f}, expected {tot:,.0f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"  wrote {OUT.name}: {len(out):,} rows, {out['count'].sum():,.0f} people")
    lang = out.assign(l=out.source_category.str.replace(r" \(.*\)$", "", regex=True)) \
              .groupby("l")["count"].sum().sort_values(ascending=False)
    for l, v in lang.items():
        print(f"    {l:16s} {v:10,.0f} {v / tot:6.1%}")
    # spot checks: the biggest settlements
    big = out.assign(loc=out.geo_id.str.split("_").str[0],
                     l=out.source_category.str.replace(r" \(.*\)$", "", regex=True))
    piv = big.pivot_table(index="loc", columns="l", values="count", aggfunc="sum", fill_value=0)
    piv = piv.div(piv.sum(1), axis=0)
    piv["pop"] = big.groupby("loc")["count"].sum()
    names = x.assign(loc=x.geo_id.str.split("_").str[0]).groupby("loc")["geo_name"].first()
    for loc, r in piv.sort_values("pop", ascending=False).head(8).iterrows():
        print(f"    {loc:>5} {names[loc]:<16} {r['pop']:9,.0f}  Hebrew {r.Hebrew:.0%} "
              f"Russian {r.Russian:.0%} English {r.English:.0%} Yiddish {r.Yiddish:.0%}")


if __name__ == "__main__":
    build()
