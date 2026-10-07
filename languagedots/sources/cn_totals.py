"""2020 census county totals for China's 15 estimated provinces (used by sources/cn_ethnic.py).

chinaethnicity's 15 estimated provinces carry the 2000 census county pattern scaled to 2020
prefecture or province totals, so cities that grew fast are far short (Shenzhen 10.5M against a
census 17.6M). This puts each of their counties on its 2020 census total and keeps its
nationality mix as it is.

Totals: Dong and Wang's county census panel (github.com/leiii/census), helper1m's copy
(../helper1m/data/china/census_county_2010-2020_v1.csv), with helper1m's own fixes: 衡东县's
transcription slip (PANEL_FIXES there) and Xinjiang, which the panel leaves empty, from
helper1m's scripts/china/xinjiang_counties.csv (hongheiku.com's county pages, summing to
Xinjiang's census bulletin in every prefecture). Read-only.

The join, by code. A panel row's county_code may list several codes (a city's districts pooled
with its development zones, "130108;130111;130171"); each code that is one of cn's polygons
joins, development-zone codes that are not polygons are dropped (their people are inside the
row's figure). ALIAS maps the three panel codes that differ from chinaethnicity's (DataV's)
codes for the same county. Panel rows and polygons that share a code form a group; a group's
panel total is split over its polygons by their current (2000-pattern) totals. Asserted: every
panel row of an estimated province reaches a polygon, no polygon sits in two groups, every
polygon of an estimated province is in a group (else listed and kept on its old share).

Last, each area is held to its census total as cn.csv already has it: the prefecture in Hebei,
Liaoning, Hunan and Sichuan (chinaethnicity anchored those to 2020 prefecture rows), else the
province (cn.csv's province sums are the 31 provinces' 2020 census figures). Where the panel's
rows fall short (people the census books to no county row: Xixian New Area's in Shaanxi, Dalian's
and Wuhu's development zones) the shortfall goes to the area's counties where the placement grid
holds more people than the panel, in proportion to that excess and never above it: helper1m's
rule (scripts/china/fetch.py). Where the panel is over, the area is scaled down. Printed per area.

Then rake(): with each county's mix fixed, the rescale moves the census nationality totals per
area (Kazakh -15%), so the estimated rows are raked to both the county totals and the area's
nationality totals. The record is sources/cn.md §9.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
H1M = HERE.parent / "helper1m"
PANEL = H1M / "data" / "china" / "census_county_2010-2020_v1.csv"
XINJIANG = H1M / "scripts" / "china" / "xinjiang_counties.csv"
PANEL_FIXES = {"430424": 565423}        # helper1m fetch.py: 衡东县 562,423 is a slip for 565,423
ALIAS = {"340212": "340211",            # 繁昌区: official code; DataV keeps the county's 340211
         "520204": "520221",            # 水城区 (2020, was 水城县 520221)
         "620201": "620200"}            # 嘉峪关市: DataV files the city itself, 620200
# chinaethnicity scaled these four provinces' 2000 pattern to their 2020 census PREFECTURE rows, so
# cn.csv's prefecture sums are census figures there; the other 11 only to the province total
PREF_ANCHORED = {13, 21, 43, 51}


def panel_rows():
    p = pd.read_csv(PANEL, dtype={"county_code": str}, usecols=["county", "county_code", "city", "popu_2020"])
    p = p.dropna(subset=["popu_2020"])
    for c, v in PANEL_FIXES.items():
        p.loc[p["county_code"] == c, "popu_2020"] = v
    x = pd.read_csv(XINJIANG, comment="#", dtype={"county_code": str})
    if (p["county_code"].str[:2] == "65").any():
        raise SystemExit("the panel has Xinjiang rows now; drop the supplement")
    x = pd.DataFrame({"county": x["name_cn"], "county_code": x["county_code"], "city": "",
                      "popu_2020": x["pop_2020"]})
    return pd.concat([p, x], ignore_index=True)


def rake(est, area_of, col_target, row_target, log=print, iters=500):
    """Iterative proportional fitting of the estimated provinces' county x nationality counts.
    est: rows (geo_id, source_category, count) already on the new county totals. Columns are held
    to col_target ((area, source_category) -> count: the census's own nationality totals per
    province, or per prefecture in PREF_ANCHORED), rows to row_target (geo_id -> census county
    total). Starting from the rescaled counts, so each county's mix moves as little as both sets
    of census totals allow. Returns est with count replaced."""
    import numpy as np
    est = est.copy()
    est["area"] = est["geo_id"].map(area_of)
    out = []
    worst = 0.0
    for a, g in est.groupby("area"):
        m = g.pivot_table(index="geo_id", columns="source_category", values="count",
                          aggfunc="sum", fill_value=0.0)
        r = row_target.reindex(m.index).to_numpy(dtype=float)
        c = np.array([col_target.get((a, k), 0.0) for k in m.columns])
        x = m.to_numpy(dtype=float)
        for _ in range(iters):
            rs = x.sum(axis=1)
            x *= np.divide(r, rs, out=np.ones_like(r), where=rs > 0)[:, None]
            cs = x.sum(axis=0)
            x *= np.divide(c, cs, out=np.ones_like(c), where=cs > 0)[None, :]
            if np.abs(x.sum(axis=1) - r).max() < 0.5:
                break
        worst = max(worst, np.abs(x.sum(axis=1) - r).max(), np.abs(x.sum(axis=0) - c).max())
        out.append(pd.DataFrame(x, index=m.index, columns=m.columns).stack().rename("count"))
    res = pd.concat(out).reset_index()
    res = res[res["count"] > 0]
    log(f"  raked to the census nationality totals of {est['area'].nunique()} areas; "
        f"largest county or nationality residual {worst:,.1f} people")
    if worst > 50:
        raise SystemExit("raking did not converge")
    keep = est.drop(columns=["count", "area", "source_category"]).drop_duplicates("geo_id")
    return res.merge(keep, on="geo_id")


def area_key(u, prov, pref):
    return f"{int(prov)} {pref}" if int(prov) in PREF_ANCHORED else str(int(prov))


def targets(tot, prov, est, pref, grid, name, log=print):
    """tot, prov, est: per polygon (index geo_id) current total, province, estimated flag;
    pref, name: polygon -> prefecture code, name; grid: polygon -> placement-grid population.
    Returns the new total per polygon of the estimated provinces (index geo_id)."""
    units = set(tot.index[est == 1])
    eprov = set(prov[est == 1].astype(int))
    rows = panel_rows()
    rows = rows[rows["county_code"].str[:2].astype(int).isin(eprov)].reset_index(drop=True)
    # group panel rows and polygons that share a code (union-find)
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    dropped = []
    for i, r in rows.iterrows():
        codes = [ALIAS.get(c.strip(), c.strip()) for c in r["county_code"].split(";")]
        hit = [c for c in codes if c in units]
        dropped += [c for c in codes if c not in units]
        if not hit:
            raise SystemExit(f"panel row {r['county_code']} {r['county']} reaches no polygon")
        for c in hit:
            parent[find(("p", i))] = find(("u", c))
    gu, gp = {}, {}
    for k in list(parent):
        g = find(k)
        (gu if k[0] == "u" else gp).setdefault(g, []).append(k[1])
    for g, us in gu.items():
        if g not in gp:
            raise SystemExit(f"group with polygons and no panel row: {us}")
    new = {}
    multi = []
    for g, us in gu.items():
        t = rows.loc[gp[g], "popu_2020"].sum()
        s = tot[us].sum()
        for u in us:
            new[u] = t * tot[u] / s
        if len(us) > 1 or len(gp[g]) > 1:
            multi.append((sorted(us), len(gp[g]), s, t))
    left = sorted(units - set(new))
    log(f"  panel join: {len(rows):,} panel rows (Xinjiang from helper1m's supplement) -> "
        f"{len(new):,} of {len(units):,} polygons in {len(gu):,} groups; "
        f"{len(multi)} groups pool several polygons or rows; "
        f"{len(set(dropped))} codes not polygons, dropped (development zones): "
        f"{', '.join(sorted(set(dropped)))}")
    for us, n, s, t in multi:
        log(f"    group {'+'.join(us)} ({n} panel rows): {s:,.0f} -> {t:,.0f}")
    if left:
        log(f"  polygons with no panel row, kept on their old share of the province: {left}")
    if left:
        raise SystemExit("every estimated polygon should have a panel row; see the list above")
    new = pd.Series(new, dtype=float)
    # Hold each area to its census total: the prefecture where cn.csv's prefecture sums are the
    # 2020 census's own rows (PREF_ANCHORED), else the province.
    area = pd.Series({u: area_key(u, prov[u], pref[u]) for u in units})
    out = {}
    for a, us in area.groupby(area).groups.items():
        us = list(us)
        want = tot[us].sum()
        t = new[us].copy()
        short = want - t.sum()
        how = "as is"
        if short > 0.5:
            # people the census books to no county row (Xixian New Area's, a city's development
            # zones): they live where the grid holds more people than the census rows, so hand
            # the shortfall to those counties by their excess, never above the grid
            ex = (grid.reindex(us).fillna(0) - t).clip(lower=0)
            back = min(short, ex.sum()) * ex / ex.sum() if ex.sum() > 0 else ex * 0
            t += back
            if want - t.sum() > 0.5:
                t *= want / t.sum()
            top = back.sort_values(ascending=False).head(4)
            how = (f"{short:,.0f} short, to the counties the grid holds over: " +
                   ", ".join(f"{name.get(u, u)} {v:,.0f}" for u, v in top.items() if v >= 1))
        elif short < -0.5:
            t *= want / t.sum()
            how = f"panel {-short:,.0f} over, scaled by {want / (want - short):.4f}"
        if abs(short) > 0.005 * want or a.isdigit():
            log(f"  {a}: census {want:,.0f}, panel {want - short:,.0f}; {how}")
        if abs(short) > 0.1 * want:
            raise SystemExit(f"{a}: panel and census totals differ by {short / want:.1%}")
        out.update(t.to_dict())
    out = pd.Series(out)
    if set(out.index) != units:
        raise SystemExit("not every estimated polygon got a total")
    return out
