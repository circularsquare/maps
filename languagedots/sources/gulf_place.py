"""Where inside a unit the Gulf states' citizens and foreign residents are placed.
Shared by countries/{sa,ae,qa,kw,om,bh}.py; the record is sources/gulf_place.md.

Every Gulf unit's counts say how many citizens (C) and foreign residents (F) it holds, but not
where inside it each lives. Until 2026-10-06 both were spread on the same population weight, so
every unit read as an even mix. Here each placement hex i of a unit u gets a foreign share

    logit s_i = a_u + BETA * ln(density_i) + GAMMA * industrial_i

with a_u solved so that sum_i pop_i * s_i = F_u exactly: foreign dots go on pop_i * s_i,
citizen dots on pop_i * (1 - s_i), so the two together still follow population and no count
moves. `density_i` is the hex's people per km2 (floored at 1); `industrial_i` the share of the
hex under OSM landuse=industrial or a named labour camp (sources/gulf_osm.py).

BETA and GAMMA are fitted (`python sources/gulf_place.py --fit`) on the only tables that give
citizens and foreigners below the drawn units' parents: Kuwait's 2021 census areas within each
governorate, and Oman's 2024 register wilayat within each governorate; checked against two tables
not used in the fit, the 2022 census's Riyadh and Ad Diriyah governorates within Riyadh region
(RCRC) and SCAD's 2016 Abu Dhabi regions within the emirate.

Those two tables are also finer counts than the units drawn, so they split units (`SPLITS`,
`split_units`): Saudi Arabia's Riyadh region into Riyadh governorate, Ad Diriyah and the rest;
Abu Dhabi emirate into its three regions. Placement layers with the split units are written by
`--layers` to data/geo/sa/sa_hexes.gpkg and data/geo/ae/ae_hexes.gpkg (religiondots' hexes,
re-keyed; read-only on religiondots).

    python sources/gulf_osm.py --fetch       OSM industrial land, labour camps, admin regions
    python sources/gulf_place.py --layers    the two re-keyed hex layers
    python sources/gulf_place.py --fit       the BETA / GAMMA search and the witnesses
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RD = ROOT.parent / "religiondots"
RD_GEO = RD / "data" / "geo"
GULF = ROOT / "data" / "geo" / "gulf"
NORM = ROOT / "data" / "normalized"

# fitted 2026-10-06 (`--fit`; sources/gulf_place.md): the search's best was BETA 0.375 and
# GAMMA rising without limit (14 the largest tried); rounded to 0.4 and 10, where an industrial
# hex is all but entirely foreign. Kuwait's areas do not improve at any BETA (Kontur's density
# does not tell its villa suburbs from its apartment districts), Oman's do, and the Riyadh
# witness is matched to the point.
BETA = 0.4
GAMMA = 10.0

# Finer citizen / foreign counts than the units drawn: {cc: {parent unit: {sub-unit: (C, F)}}}.
# SA: RCRC open data, census 2022 by citizenship and governorate (religiondots'
#   data/raw/sa/rcrc_population_citizenship_gender_2022.json), both sexes.
# AE: Abu Dhabi emirate, 2024 as drawn (665,515 Emiratis, 4,135,985 people; sources/ae.md):
#   people by region at the 2023 census's regional totals (SCAD release, 2,495,925 / 1,009,735 /
#   284,205 of 3,789,860), Emiratis at SCAD's mid-2016 regional shares (Statistical Yearbook of
#   Abu Dhabi 2017, Table 3.1.6: 293,860 / 226,285 / 31,390 of 551,535); the rest non-Emirati.
_AD_TOT, _AD_EMI = 4_135_985, 665_515
_AD_2023 = {"Abu Dhabi Region": 2_495_925, "Al Ain Region": 1_009_735, "Al Dhafra Region": 284_205}
_AD_2016_EMI = {"Abu Dhabi Region": 293_860, "Al Ain Region": 226_285, "Al Dhafra Region": 31_390}


def _ad_split():
    out = {}
    for k in _AD_2023:
        n = _AD_TOT * _AD_2023[k] / sum(_AD_2023.values())
        c = _AD_EMI * _AD_2016_EMI[k] / sum(_AD_2016_EMI.values())
        out[f"Abu Dhabi|{k}"] = (c, n - c)
    return out


SPLITS = {
    # Ad Diriyah joins Riyadh: OSM's Ad Diriyah polygon holds 508,635 of Kontur's people for
    # the census's 95,834 (it takes in Riyadh's north-western suburbs, which GASTAT counts in
    # Riyadh), while the two together are 7,033,099 in Kontur against 7,104,954 counted
    "sa": {"SA01": {"SA01|Riyadh and Ad Diriyah": (3_351_352 + 53_918, 3_657_768 + 41_916),
                    "SA01|rest of the region": (1_033_940, 452_854)}},
    "ae": {"Abu Dhabi": _ad_split()},
}
# which OSM admin polygon (data/geo/gulf/admin.gpkg) each sub-unit is; None = the rest
SPLIT_POLY = {"SA01|Riyadh and Ad Diriyah": ("Riyadh governorate", "Ad Diriyah governorate"),
              "SA01|rest of the region": None,
              "Abu Dhabi|Abu Dhabi Region": "Abu Dhabi Region",
              "Abu Dhabi|Al Ain Region": "Al Ain Region",
              "Abu Dhabi|Al Dhafra Region": "Al Dhafra Region"}


def split_units(df, cc):
    """df: rows with unit, node, origin, count. Each parent unit in SPLITS[cc] is replaced by
    its sub-units: citizen rows at the sub-unit's share of the parent's citizens, foreign rows at
    its share of the foreigners (the parent's foreign language mix kept). Totals asserted."""
    out = [df[~df["unit"].isin(SPLITS.get(cc, {}))]]
    for parent, subs in SPLITS.get(cc, {}).items():
        d = df[df["unit"] == parent]
        if d.empty:
            raise SystemExit(f"{cc}: no rows for split unit {parent}")
        cit = d["origin"].map(is_citizen).to_numpy()
        C, F = d.loc[cit, "count"].sum(), d.loc[~cit, "count"].sum()
        tc = sum(c for c, _ in subs.values())
        tf = sum(f for _, f in subs.values())
        if abs(tc - C) > 2 or abs(tf - F) > 2:
            raise SystemExit(f"{cc} {parent}: table {tc:,.0f} citizens / {tf:,.0f} foreign, "
                             f"rows {C:,.0f} / {F:,.0f}")
        for sub, (c, f) in subs.items():
            x = d.copy()
            x["unit"] = sub
            x["count"] = np.where(cit, x["count"] * c / tc, x["count"] * f / tf)
            out.append(x)
    res = pd.concat(out, ignore_index=True)
    if abs(res["count"].sum() - df["count"].sum()) > 1:
        raise SystemExit(f"{cc}: split moved the total")
    return res


def is_citizen(origin):
    o = str(origin)
    return not (o.startswith("non-") or o == "expatriate")


# --------------------------------------------------------------------------------- hex features
def hex_features(cc, geoms, pop):
    """(density per km2, industrial share 0-1) for each placement polygon."""
    import geopandas as gpd
    import shapely
    gs = gpd.GeoSeries(geoms, crs=4326)
    eq = gs.to_crs(6933)
    area = eq.area.to_numpy() / 1e6
    dens = np.where(area > 0, pop / np.maximum(area, 1e-9), 0.0)
    ind = np.zeros(len(gs))
    path = GULF / f"{cc}_industrial.gpkg"
    if not path.exists():
        raise SystemExit(f"{path.name} missing: run python sources/gulf_osm.py")
    ip = gpd.read_file(path).to_crs(6933)
    iu = shapely.union_all(shapely.make_valid(ip.geometry.values))
    parts = shapely.get_parts(iu)
    tree = shapely.STRtree(parts)
    hx = eq.geometry.values
    a, b = tree.query(hx, predicate="intersects")
    if len(a):
        inter = shapely.area(shapely.intersection(hx[a], parts[b]))
        np.add.at(ind, a, inter)
        ind = np.clip(ind / np.maximum(area * 1e6, 1e-9), 0, 1)
    return dens, ind


# ---------------------------------------------------------------------------------- the split
def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


L0 = np.log(1000.0)       # the curve's centre: 1,000 people per km2


def score(dens, ind, beta, gamma, beta2=0.0):
    L = np.log(np.maximum(dens, 1.0))
    return beta * L + beta2 * (L - L0) ** 2 + gamma * ind


def foreign_share(pop, dens, ind, F, beta, gamma, beta2=0.0):
    """Per-hex foreign share s with sum(pop * s) == F (to 1e-6), by bisection on a_u."""
    tot = pop.sum()
    if tot <= 0 or F <= 0:
        return np.zeros(len(pop))
    if F >= tot:
        return np.ones(len(pop))
    z = score(dens, ind, beta, gamma, beta2)
    lo, hi = -60.0 - z.max(), 60.0 - z.min()
    for _ in range(80):
        mid = (lo + hi) / 2
        if (pop * _sig(mid + z)).sum() > F:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-10:
            break
    return _sig((lo + hi) / 2 + z)


class GulfWeighter:
    """scatter.py weighter: citizen and foreign dots of each unit on their own hex weights.
    A (unit, node) row holding both (Gulf Arabic carries other GCC nationals) blends the two
    weights by its citizen share in that unit."""

    def __init__(self, cc, place, cit_frac, unit_cf, beta=None, gamma=None):
        self.beta = BETA if beta is None else beta
        self.gamma = GAMMA if gamma is None else gamma
        self.pop = place["pop"].to_numpy(dtype=float)
        self.unit = place["unit"].astype(str).to_numpy()
        self.cit_frac = cit_frac        # {(unit, node): citizen share of that row}
        self.unit_cf = unit_cf          # {unit: (citizens, foreign)}
        print(f"  gulf placement: industrial share and density per hex "
              f"(BETA {self.beta}, GAMMA {self.gamma})")
        self.dens, self.ind = hex_features(cc, place.geometry.values, self.pop)
        self._w = {}
        self.n = {"split": 0, "pop": 0, "none": 0}

    def _unit_weights(self, idx):
        u = self.unit[idx[0]]
        if u not in self._w:
            p = self.pop[idx]
            c, f = self.unit_cf.get(u, (0.0, 0.0))
            if p.sum() <= 0 or c + f <= 0:
                self._w[u] = None
            else:
                # F as a share of the unit's people, applied to the layer's own population
                F = p.sum() * f / (c + f)
                s = foreign_share(p, self.dens[idx], self.ind[idx], F, self.beta, self.gamma)
                wf, wc = p * s, p * (1 - s)
                self._w[u] = (wc / wc.sum() if wc.sum() > 0 else p / p.sum(),
                              wf / wf.sum() if wf.sum() > 0 else p / p.sum())
        return self._w[u]

    def weights(self, node, idx, count, plain=False):
        w = self._unit_weights(idx)
        if w is None:
            p = self.pop[idx]
            self.n["pop" if p.sum() > 0 else "none"] += 1
            return p if p.sum() > 0 else None
        cf = self.cit_frac.get((self.unit[idx[0]], node), 0.0)
        self.n["split"] += 1
        return cf * w[0] + (1 - cf) * w[1]

    def summary(self):
        return (f"{self.n['split']:,} (unit, language) rows placed by the citizen / foreign split "
                f"(sources/gulf_place.py), {self.n['pop']:,} on population, {self.n['none']:,} "
                f"on equal shares")


# ------------------------------------------------- foreign residents in three pools (2026-10-07)
# sources/gulf_place.md §6. Each nationality's men and women (UN DESA, by sex) are read as
#   family:       min(men, women) of each sex, people living as households;
#   single men:   the men beyond that, who live in shared or labour accommodation;
#   single women: the women beyond that, nearly all live-in domestic workers.
# The pools are then placed apart inside each unit: single men by a hex share fitted on Qatar's
# 2020 census zones by sex (`--fit-sex`), family on the rest of the foreign weight, single women
# with households (citizens and families). In Qatar the zones' own sex counts fix each zone's
# single men and families; elsewhere the fitted hex rule alone. Counts per unit never move.
#
# fitted 2026-10-07 on Qatar's 87 zones within its 8 municipalities (`--fit-sex`; gulf_place.md §6)
DELTA = 10.0     # industrial share of the hex
EPS = 0.5        # ln(people per km2), centred on 1,000


def node_sex_pools(dest, dest_cc, year):
    """{node: (men, women, paired)} of `dest`'s migrants from UN DESA by origin and sex, each
    origin on its language mix as the country's build reads it (gulf_mix.origin_mix with the
    origin's both-sexes count; DESA's `Others` on `other`). paired = min(men, women) per origin."""
    import gulf_mix as g
    origins, others, _ = g.desa(dest, year=year)
    out = {}

    def add(mix, m, w):
        p = min(m, w)
        for n, s in mix.items():
            a = out.setdefault(n, [0.0, 0.0, 0.0])
            a[0] += m * s
            a[1] += w * s
            a[2] += p * s
    for iso, (b, m, w) in origins.items():
        add(g.origin_mix(iso, dest_cc, b), m, w)
    add({g.OTHER: 1.0}, others[1], others[2])
    return {n: tuple(v) for n, v in out.items()}


def resolve_pools(node_mw, known):
    """node_sex_pools keyed by today's home-mix node ids -> keyed by the ids the country's
    normalized CSV holds (`known`). Home mixes elsewhere get re-keyed after a Gulf CSV is written
    (2026-10-07: `afroasiatic.arabic.sudanese_arabic` against qa.csv's
    `afroasiatic.sudanese_arabic`), so an id the CSV lacks is matched on its last component."""
    known = set(known)
    last = {}
    for n in known:
        last.setdefault(n.split(".")[-1], []).append(n)
    out, miss = {}, []
    for k, v in node_mw.items():
        if k in known:
            key = k
        elif len(last.get(k.split(".")[-1], [])) == 1:
            key = last[k.split(".")[-1]][0]
        else:
            miss.append(k)
            continue
        a = out.setdefault(key, [0.0, 0.0, 0.0])
        for j in range(3):
            a[j] += v[j]
    if miss:
        print(f"  gulf pools: {len(miss)} home-mix nodes not in the CSV, left out of the sex "
              f"ratios: {miss[:5]}")
    return {n: tuple(v) for n, v in out.items()}


def row_pools(df, node_mw, unit_mw=None):
    """{(unit, node): (family, single men, single women)} for the foreign rows of df.
    node_mw: node_sex_pools(). unit_mw: {unit: (foreign men, foreign women)} where a census gives
    them (Qatar), else each row splits at its node's national sex ratio. With unit_mw, a unit's
    pairs are min(its men x P/M, its women x P/W) and each sex's family share is scaled to that."""
    M = sum(v[0] for v in node_mw.values())
    W = sum(v[1] for v in node_mw.values())
    P = sum(v[2] for v in node_mw.values())
    out = {}
    d = df[~df["origin"].map(is_citizen)]
    for u, g in d.groupby("unit"):
        rows = g.groupby("node")["count"].sum()
        if unit_mw is not None:
            fm, fw = unit_mw[u]
            pair = min(fm * P / M, fw * P / W)
            km, kw = pair / (fm * P / M), pair / (fw * P / W)
        for node, c in rows.items():
            # a node today's home mixes no longer give (the CSV predates a re-keyed mix) takes
            # the country's overall sex ratios
            m, w, p = node_mw.get(node, (M, W, P))
            if m + w <= 0:
                m, w, p = M, W, P
            if unit_mw is None:
                fam_m = fam_w = p
                mm, ww = m, w
            else:
                mm, ww = fm * m / M, fw * w / W
                fam_m, fam_w = mm * (p / m if m else 0) * km, ww * (p / w if w else 0) * kw
            tot = mm + ww
            out[(u, node)] = (c * (fam_m + fam_w) / tot, c * (mm - fam_m) / tot,
                              c * (ww - fam_w) / tot)
    return out


def _solve_share(f, z, target):
    """t with logit t = a + z, sum(f t) == target, by bisection."""
    tot = f.sum()
    if tot <= 0 or target <= 0:
        return np.zeros(len(f))
    if target >= tot:
        return np.ones(len(f))
    lo, hi = -60.0 - z.max(), 60.0 - z.min()
    for _ in range(80):
        mid = (lo + hi) / 2
        if (f * _sig(mid + z)).sum() > target:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-10:
            break
    return _sig((lo + hi) / 2 + z)


def single_men_score(dens, ind, delta=None, eps=None):
    delta = DELTA if delta is None else delta
    eps = EPS if eps is None else eps
    return delta * ind + eps * (np.log(np.maximum(dens, 1.0)) - L0)


class PoolWeighter(GulfWeighter):
    """GulfWeighter with each row's foreign dots split over three pools (row_pools), each on its
    own hex weight. zone: per-hex zone ids and {zone: (people, men, women)} from a census, with
    {unit: citizen men share}; the zones then fix where single men and families go."""

    def __init__(self, cc, place, cit_frac, unit_cf, pools, zone=None, zone_sex=None,
                 cit_male=None, delta=None, eps=None):
        super().__init__(cc, place, cit_frac, unit_cf)
        self.pools = pools
        self.zone = None if zone is None else np.asarray(zone)
        self.zone_sex = zone_sex
        self.cit_male = cit_male or {}
        self.z = single_men_score(self.dens, self.ind, delta, eps)
        self._pw = {}
        self.n["pooled"] = 0
        print(f"  gulf pools: single men by industrial share and density (DELTA "
              f"{DELTA if delta is None else delta}, EPS {EPS if eps is None else eps})"
              + (", per census zone by sex" if zone is not None else ""))

    def split(self, idx):
        """(citizens, foreigners) per hex of one unit, people of the layer, or None."""
        u = self.unit[idx[0]]
        p = self.pop[idx]
        c_, f_ = self.unit_cf.get(u, (0.0, 0.0))
        if p.sum() <= 0 or c_ + f_ <= 0:
            return None
        F = p.sum() * f_ / (c_ + f_)
        s = foreign_share(p, self.dens[idx], self.ind[idx], F, self.beta, self.gamma)
        return p * (1 - s), p * s

    def zone_single_men(self, idx, cit, fr):
        """{zone: single foreign men} of one unit from the census zones by sex: each zone's men
        and women less its placed citizens (at the unit's citizen sex ratio), and the foreign
        men beyond the foreign women. Capped just under the zone's placed foreigners."""
        u = self.unit[idx[0]]
        p = self.pop[idx]
        zn = self.zone[idx]
        qm = self.cit_male.get(u, 0.5)
        out = {}
        for zz in np.unique(zn):
            k = zn == zz
            N, Mz, _Wz = self.zone_sex.get(int(zz), (0, 0, 0))
            if N <= 0 or fr[k].sum() <= 0:
                continue
            lay = p[k].sum()
            fm = max(0.0, lay * Mz / N - cit[k].sum() * qm)
            fw = max(0.0, lay * (N - Mz) / N - cit[k].sum() * (1 - qm))
            out[zz] = min(fm - min(fm, fw), 0.999 * fr[k].sum())
        return out

    def unit_shapes(self, idx):
        """(citizens, family, single men, single women) hex weights of one unit, or None."""
        u = self.unit[idx[0]]
        if u in self._pw:
            return self._pw[u]
        cs = self.split(idx)
        if cs is None:
            self._pw[u] = None
            return None
        cit, fr = cs
        keys = [k for k in self.pools if k[0] == u]
        sm = sum(self.pools[k][1] for k in keys)
        tot = sum(sum(self.pools[k]) for k in keys)
        z = self.z[idx]
        if self.zone is None or tot <= 0:
            t = _solve_share(fr, z, fr.sum() * sm / tot if tot > 0 else 0)
        else:
            t = np.zeros(len(idx))
            zn = self.zone[idx]
            for zz, single in self.zone_single_men(idx, cit, fr).items():
                k = zn == zz
                t[k] = _solve_share(fr[k], z[k], single)
        w_sm = fr * t
        w_fam = fr * (1 - t)
        w_sw = cit + w_fam
        sh = []
        for w in (w_fam, w_sm, w_sw):
            sh.append(w / w.sum() if w.sum() > 0 else fr / fr.sum())
        self._pw[u] = (cit / cit.sum() if cit.sum() > 0 else p / p.sum(), *sh)
        return self._pw[u]

    def weights(self, node, idx, count, plain=False):
        sh = self.unit_shapes(idx)
        if sh is None:
            return super().weights(node, idx, count, plain)
        u = self.unit[idx[0]]
        cf = self.cit_frac.get((u, node), 0.0)
        pl = self.pools.get((u, node))
        if pl is None or sum(pl) <= 0:
            fw = super()._unit_weights(idx)[1]
        else:
            a = np.array(pl) / sum(pl)
            fw = a[0] * sh[1] + a[1] * sh[2] + a[2] * sh[3]
            self.n["pooled"] += 1
        self.n["split"] += 1
        return cf * sh[0] + (1 - cf) * fw

    def summary(self):
        return super().summary() + f"; {self.n['pooled']:,} foreign rows in family / single pools"


def fit_sex(cc="qa"):
    """DELTA / EPS search on Qatar's 2020 zones: each municipality's single foreign men (its
    zones' foreign men beyond foreign women) placed by the hex rule alone, against each zone's
    own count. Loss: foreigner-weighted squared error of the zone's single-men share."""
    import geopandas as gpd
    sys.path[:0] = [str(ROOT), str(ROOT / "countries"), str(ROOT / "taxonomy")]
    from countries import COUNTRIES
    cfg = COUNTRIES[cc]
    place = gpd.read_file(cfg["place"])
    place["unit"] = cfg["place_unit"](place)
    w = cfg["place_weight"](place)
    units = {u: g.index.to_numpy() for u, g in place.groupby("unit")}
    base = {}
    for u, idx in units.items():
        cit, fr = w.split(idx)
        tg = w.zone_single_men(idx, cit, fr)
        base[u] = (idx, fr, tg)

    def run(delta, eps):
        rows = []
        for u, (idx, fr, tg) in base.items():
            z = single_men_score(w.dens[idx], w.ind[idx], delta, eps)
            t = _solve_share(fr, z, sum(tg.values()))
            zn = w.zone[idx]
            for zz, s in tg.items():
                k = zn == zz
                F = fr[k].sum()
                rows.append((u, zz, F, s / F, (fr[k] * t[k]).sum() / F))
        return pd.DataFrame(rows, columns=["unit", "zone", "foreign", "share", "pred"])

    def err(r):
        return float(np.sqrt((r["foreign"] * (r["pred"] - r["share"]) ** 2).sum() / r["foreign"].sum()))
    grid = []
    for delta in [0, 1, 2, 4, 6, 8, 10, 14]:
        for eps in [-1.0, -0.5, -0.25, 0, 0.25, 0.5, 1.0]:
            grid.append((err(run(delta, eps)), delta, eps))
    grid.sort()
    print(f"even within each municipality (0, 0): rms {err(run(0, 0)):.3f}")
    for e, d, ep in grid[:12]:
        print(f"  DELTA {d:>4} EPS {ep:>5}: rms {e:.3f}")
    r = run(grid[0][1], grid[0][2])
    r0 = run(0, 0)
    print(f"r (best) {np.corrcoef(r['share'], r['pred'])[0, 1]:.2f}; zones by foreigners:")
    r["even"] = r0["pred"]
    for _, x in r.sort_values("foreign", ascending=False).head(15).iterrows():
        print(f"    zone {int(x['zone']):>3} {x['unit'][:14]:<14} foreign {x['foreign']:>9,.0f}  "
              f"single men {x['share']:.1%}  even {x['even']:.1%}  placed {x['pred']:.1%}")
    return grid[0]


def citizen_tables(df):
    """From rows with unit, node, origin, count: ({(unit, node): citizen share},
    {unit: (citizens, foreign)})."""
    d = df.assign(cit=df["origin"].map(is_citizen))
    g = d.groupby(["unit", "node", "cit"])["count"].sum().unstack(fill_value=0)
    for k in (True, False):
        if k not in g.columns:
            g[k] = 0
    tot = g[True] + g[False]
    frac = (g[True] / tot.where(tot > 0, 1)).to_dict()
    u = d.groupby(["unit", "cit"])["count"].sum().unstack(fill_value=0)
    cf = {k: (float(r.get(True, 0)), float(r.get(False, 0))) for k, r in u.iterrows()}
    return frac, cf


# ----------------------------------------------------------------------------------- fitting
def _layer(cc):
    import geopandas as gpd
    path = (ROOT / "data" / "geo" / cc / f"{cc}_hexes.gpkg" if cc == "qa"
            else RD_GEO / cc / f"{cc}_hexes.gpkg")
    g = gpd.read_file(path)
    g["unit"] = g["unit"].astype(str)
    pop = g["pop"].to_numpy(dtype=float)
    dens, ind = hex_features(cc, g.geometry.values, pop)
    g["dens"], g["ind"] = dens, ind
    return g


def _kontur_kw():
    """Kuwait's raw Kontur 400 m hexes keyed to religiondots' 143 units by centroid (its own
    placement layer spreads each area evenly over Kontur's footprint, so it has no density
    inside an area to fit on)."""
    import geopandas as gpd
    k = gpd.read_file(RD / "data" / "raw" / "kw" / "kontur_population_KW_20231101.gpkg").to_crs(4326)
    u = gpd.read_file(RD_GEO / "kw" / "kw_units.gpkg")[["unit", "geometry"]]
    pts = gpd.GeoDataFrame({"pop": k["population"].astype(float)},
                           geometry=k.geometry.representative_point(), crs=4326)
    j = gpd.sjoin(pts, u, how="inner", predicate="within")
    j = j[~j.index.duplicated()]
    h = k.loc[j.index, ["geometry"]].copy()
    h["pop"], h["unit"] = j["pop"].values, j["unit"].astype(str).values
    h = gpd.GeoDataFrame(h, geometry="geometry", crs=4326)
    h["dens"], h["ind"] = hex_features("kw", h.geometry.values, h["pop"].to_numpy())
    return h


def _fit_sets():
    """[(name, hexes DataFrame with columns parent, child, pop, dens, ind; children table
    child -> (C, F); used_in_fit)]"""
    import geopandas as gpd
    sets = []
    # Kuwait: census 2021 areas (religiondots' 143 units) within the 6 governorates
    kw = pd.read_csv(NORM / "kw.csv", dtype={"geo_id": str})
    lut = pd.read_csv(RD_GEO / "kw" / "kw_lookup.csv", dtype=str)
    kw["child"] = kw["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    kw["cit"] = kw["origin"].map(is_citizen)
    gov = kw.groupby("child")["governorate"].agg(lambda s: s.mode().iloc[0])
    ch = kw.groupby(["child", "cit"])["count"].sum().unstack(fill_value=0)
    h = _kontur_kw()
    h["child"] = h["unit"]
    h["parent"] = h["child"].map(gov)
    sets.append(("Kuwait areas in governorates", h, ch, True))
    # Oman: register 2024 wilayat (religiondots' 61 units) within the 11 governorates
    om = pd.read_csv(NORM / "om.csv", dtype={"geo_id": str})
    lut = pd.read_csv(RD_GEO / "om" / "om_lookup.csv", dtype=str)
    om["child"] = om["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    om["cit"] = om["origin"].map(is_citizen)
    gov = om.groupby("child")["governorate"].agg(lambda s: s.mode().iloc[0])
    ch = om.groupby(["child", "cit"])["count"].sum().unstack(fill_value=0)
    h = _layer("om")
    h["child"] = h["unit"]
    h["parent"] = h["child"].map(gov)
    sets.append(("Oman wilayat in governorates", h, ch, True))
    # Saudi Arabia, witness: RCRC's census 2022 Riyadh and Ad Diriyah governorates
    adm = gpd.read_file(GULF / "admin.gpkg")
    rc = json.loads((RD / "data" / "raw" / "sa" / "rcrc_population_citizenship_gender_2022.json")
                    .read_text(encoding="utf-8"))["results"]
    rc = pd.DataFrame(rc)
    rc = rc[rc["admregcen"] == "Ar Riyadh"]
    rc["child"] = rc["gov"].map({"Ar Riyadh": "Riyadh governorate",
                                 "Ad Diriyah": "Ad Diriyah governorate"}).fillna("rest")
    rc["cit"] = rc["ctz"] == "Saudi"
    ch = rc.groupby(["child", "cit"])["n"].sum().unstack(fill_value=0)
    h = _layer("sa")
    h = h[h["unit"] == "SA01"].copy()
    h["parent"] = "Riyadh region"
    h["child"] = _admin_child(h, adm, ["Riyadh governorate", "Ad Diriyah governorate"])
    sets.append(("Riyadh region's governorates (witness)", h, ch, False))
    # UAE, witness: SCAD mid-2016 regions of Abu Dhabi emirate (Statistical Yearbook 2017, 3.1.6)
    ch = pd.DataFrame({True: SCAD_2016_CIT, False: SCAD_2016_NON})
    h = _layer("ae")
    h = h[h["unit"] == "Abu Dhabi"].copy()
    h["parent"] = "Abu Dhabi emirate"
    h["child"] = _admin_child(h, adm, list(SCAD_2016_CIT), nearest=True)
    sets.append(("Abu Dhabi emirate's regions (witness)", h, ch, False))
    return sets


# SCAD, Statistical Yearbook of Abu Dhabi 2017, Table 3.1.6, mid-2016 (thousands; citizens to the
# person from the yearbook's prose; non-citizens = region total less citizens, 3.1.5)
SCAD_2016_CIT = {"Abu Dhabi Region": 293_860, "Al Ain Region": 226_285, "Al Dhafra Region": 31_390}
SCAD_2016_TOT = {"Abu Dhabi Region": 1_807_000, "Al Ain Region": 766_900, "Al Dhafra Region": 334_000}
SCAD_2016_NON = {k: SCAD_2016_TOT[k] - v for k, v in SCAD_2016_CIT.items()}


def _admin_child(h, adm, names, nearest=False):
    import geopandas as gpd
    pts = gpd.GeoDataFrame(geometry=h.geometry.representative_point().values, crs=4326,
                           index=h.index)
    a = adm[adm["name"].isin(names)][["name", "geometry"]]
    j = gpd.sjoin(pts, a, how="left", predicate="within")
    j = j[~j.index.duplicated()]
    out = j["name"]
    if nearest and out.isna().any():
        miss = out.isna()
        jn = gpd.sjoin_nearest(pts[miss].to_crs(6933), a.to_crs(6933), how="left")
        jn = jn[~jn.index.duplicated()]
        out = out.copy()
        out[miss] = jn["name"]
    return out.fillna("rest")


def predict(sets, beta, gamma, beta2=0.0):
    """Each child's foreign share as placed, its parent's F spread over the parent's hexes. Hex
    pop is first scaled to each child's own table total, so the test is the placement inside
    the parent, not Kontur's split between children."""
    rows = []
    for name, h, ch, used in sets:
        h = h[h["child"].isin(ch.index)]
        n_child = (ch[True] + ch[False])
        lay = h.groupby("child")["pop"].transform("sum")
        p_all = (h["pop"] * h["child"].map(n_child) / lay.where(lay > 0, np.nan)).fillna(0).to_numpy()
        for par, g in h.groupby("parent"):
            ii = h.index.get_indexer(g.index)
            p = p_all[ii]
            kids = list(g["child"].unique())
            C = sum(ch.loc[k, True] for k in kids)
            F = sum(ch.loc[k, False] for k in kids)
            s = foreign_share(p, g["dens"].to_numpy(), g["ind"].to_numpy(), F, beta, gamma, beta2)
            pred = pd.DataFrame({"child": g["child"].to_numpy(), "pop": p, "f": p * s}
                                ).groupby("child").sum()
            for k in kids:
                n = ch.loc[k, True] + ch.loc[k, False]
                rows.append(dict(set=name, used=used, parent=par, child=k, people=n,
                                 share=ch.loc[k, False] / n,
                                 pred=pred.loc[k, "f"] / pred.loc[k, "pop"] if pred.loc[k, "pop"] > 0 else np.nan,
                                 even=F / (C + F)))
    return pd.DataFrame(rows)


def wmse(r):
    r = r[r["pred"].notna()]
    return float((r["people"] * (r["pred"] - r["share"]) ** 2).sum() / r["people"].sum())


def loss(r):
    return wmse(r[r["used"]])


def fit():
    sets = _fit_sets()
    grid = []
    for beta in np.arange(0, 3.01, 0.25):
        for beta2 in [0, 0.1, 0.2, 0.3, 0.5]:
            for gamma in [0, 1, 2, 4, 6]:
                r = predict(sets, beta, gamma, beta2)
                per = {s: wmse(r[r["set"] == s]) for s in r["set"].unique()}
                grid.append((beta, beta2, gamma, loss(r), per))
    grid.sort(key=lambda x: x[3])
    names = list(grid[0][4])
    print("beta beta2 gamma  loss(fit)  " + "  ".join(n[:14].ljust(14) for n in names))
    for b, b2, g, l_, per in grid[:25]:
        print(f"{b:4.2f} {b2:4.1f} {g:4.1f}  {l_:.5f}    "
              + "  ".join(f"{per[n]:.5f}".ljust(14) for n in names))
    for b in np.arange(0, 3.01, 0.5):
        x = min((t for t in grid if t[1] == 0 and abs(t[0] - b) < 1e-9), key=lambda t: t[3])
        print(f"  beta2 0, beta {b:.1f}: best gamma {x[2]}, loss {x[3]:.5f}  "
              + "  ".join(f"{x[4][n]:.5f}" for n in names))
    best = grid[0]
    report(sets, best[0], best[2], best[1])
    return best


def report(sets, beta, gamma, beta2=0.0):
    r0 = predict(sets, 0.0, 0.0)
    r1 = predict(sets, beta, gamma, beta2)
    print(f"\nBETA {beta}, BETA2 {beta2}, GAMMA {gamma}: loss {loss(r1):.5f} (even spread {loss(r0):.5f})")
    for s in r1["set"].unique():
        a, b = r0[r0["set"] == s], r1[r1["set"] == s]
        cor = np.corrcoef(b["share"], b["pred"])[0, 1] if len(b) > 2 else np.nan
        print(f"\n{s}: people-weighted rms error {np.sqrt(wmse(a)):.3f} -> {np.sqrt(wmse(b)):.3f}; "
              f"r = {cor:.2f}")
        show = b if len(b) <= 6 else b.sort_values("people", ascending=False).head(12)
        for _, x in show.iterrows():
            print(f"    {str(x['child'])[:26]:<26} {x['people']:>10,.0f}  foreign {x['share']:.1%}  "
                  f"even {x['even']:.1%}  placed {x['pred']:.1%}")


def build_layers():
    """data/geo/{sa,ae}/<cc>_hexes.gpkg: religiondots' hexes with SPLITS' parents re-keyed to
    their sub-units by each hex's representative point in OSM's admin polygons (the nearest
    polygon for the few outside every one; for sa, outside the two governorates is the rest)."""
    import geopandas as gpd
    adm = gpd.read_file(GULF / "admin.gpkg")
    for cc, splits in SPLITS.items():
        g = gpd.read_file(RD_GEO / cc / f"{cc}_hexes.gpkg")
        g["unit"] = g["unit"].astype(str)
        for parent, subs in splits.items():
            m = g["unit"] == parent
            named = {}
            for k in subs:
                for poly in ([SPLIT_POLY[k]] if isinstance(SPLIT_POLY[k], str)
                             else SPLIT_POLY[k] or []):
                    named[poly] = k
            rest = [k for k in subs if SPLIT_POLY[k] is None]
            child = _admin_child(g[m], adm, list(named), nearest=not rest)
            g.loc[m, "unit"] = child.map(named).fillna(rest[0] if rest else "").values
            if (g.loc[m, "unit"] == "").any():
                raise SystemExit(f"{cc}: hexes of {parent} with no sub-unit")
            t = g[m].groupby("unit")["pop"].agg(["size", "sum"])
            for k, (c, f) in subs.items():
                print(f"  {cc} {k:<32} {int(t.loc[k, 'size']):>7,} hexes, Kontur {t.loc[k, 'sum']:>11,.0f}, "
                      f"table {c + f:>11,.0f}, foreign {f / (c + f):.1%}")
        out = ROOT / "data" / "geo" / cc / f"{cc}_hexes.gpkg"
        out.parent.mkdir(parents=True, exist_ok=True)
        g.to_file(out, driver="GPKG")
        print(f"  -> {out.relative_to(ROOT)}: {len(g):,} hexes, {g['unit'].nunique()} units")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--layers", action="store_true")
    ap.add_argument("--report", nargs=3, type=float, metavar=("BETA", "BETA2", "GAMMA"))
    ap.add_argument("--fit-sex", action="store_true")
    a = ap.parse_args()
    if a.fit_sex:
        fit_sex()
    if a.layers:
        build_layers()
    if a.fit:
        fit()
    if a.report:
        report(_fit_sets(), a.report[0], a.report[2], a.report[1])
