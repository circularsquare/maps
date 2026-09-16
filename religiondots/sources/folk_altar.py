"""Chinese folk religion as practice (spec §3.13): the home-altar rate among people who name no
religion, and whether it differs between drawn units.

Anita, 2026-09-15: in China, Taiwan and Hong Kong, `chinesefolk` is drawn for people who NAME folk
religion, and for people who name no religion but keep a religious shrine or altar at home. The
item is ISSP's "For religious reasons, do you have in your home a shrine, an altar, or a religious
object", which CGSS 2010 (`na`) and TSCS 2009, 2014 and 2018 carry. `sources/folk_practice.md` is
the evidence for choosing it. This module holds the two pieces both loaders use:

  `rate_test`  Does a yes/no rate among eligible respondents rank the units the same way in two
               halves of the sampling units, more often than when those sampling units are dealt
               into the units at random? It is `tw.py`'s construction (random halves of the
               sampling units inside each round, and the regrouping null of spec §12, "WHERE THE
               SAMPLING UNITS NEST INSIDE THE DRAWN UNITS, THE NULL REGROUPS THEM"), applied to a
               rate instead of an answer's share. The chi-square across units and
               `stability.CELL_CAP`, nationally and inside the unit with the highest rate, are
               vetoes, as they are there.
  `shrink`     Beta-binomial empirical Bayes of each unit's weighted rate toward the national one.
               The prior strength comes from the between-unit variance by the method of moments,
               so a province with six eligible respondents is not drawn at what six people can show.

Used by `sources/cn_altar.py` (provinces, CGSS 2010) and `sources/tw_altar.py` (counties, TSCS).
"""

import numpy as np
import pandas as pd

import stability as shared

N_SPLITS = 200
N_NULL = 400
SEED = 0
MIN_UNITS = 8
ALPHA = shared.STAB_ALPHA
CELL_CAP = shared.CELL_CAP


def _clusters(hit, unit, cluster, group):
    d = pd.DataFrame({
        "hit": np.asarray(hit, dtype=float),
        "unit": np.asarray(unit, dtype=int),
        "cluster": np.asarray(cluster).astype(str),
        "group": np.zeros(len(hit), dtype=int) if group is None else np.asarray(group),
    })
    d["key"] = d["group"].astype(str) + "|" + d["cluster"]
    per = d.groupby("key")["unit"].nunique()
    if (per > 1).any():
        raise SystemExit(f"sampling units that sit in two drawn units: {list(per[per > 1].index)[:5]}")
    return d.groupby("key").agg(h=("hit", "sum"), n=("hit", "size"),
                                unit=("unit", "first"), group=("group", "first"))


def rate_test(hit, unit, cluster, n_units, group=None):
    """The split-half test for a rate. `unit` is 0..n_units-1; `group` is the round, if several.

    Returns a dict: n, k, clusters, median, q95, p, chi_p, cell, top (unit index), top_cell, verdict.
    """
    c = _clusters(hit, unit, cluster, group)
    h = c["h"].to_numpy(float)
    n = c["n"].to_numpy(float)
    u = c["unit"].to_numpy(int)
    g = c["group"].to_numpy()
    n_c = len(c)
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(g == v) for v in np.unique(g)]

    H = np.zeros((N_SPLITS, n_c))
    for s in range(N_SPLITS):
        for idx in groups:
            pick = rng.permutation(idx)[: len(idx) // 2 + rng.integers(0, 2) * (len(idx) % 2)]
            H[s, pick] = 1.0

    def stat(assign):
        U = np.zeros((n_c, n_units))
        U[np.arange(n_c), assign] = 1.0
        hA, nA = (H * h) @ U, (H * n) @ U
        hB, nB = ((1.0 - H) * h) @ U, ((1.0 - H) * n) @ U
        with np.errstate(invalid="ignore", divide="ignore"):
            rA, rB = hA / nA, hB / nB
        vals = []
        for s in range(N_SPLITS):
            ok = (nA[s] > 0) & (nB[s] > 0)
            if ok.sum() >= MIN_UNITS:
                r = shared.rho(rA[s, ok], rB[s, ok])
                if np.isfinite(r):
                    vals.append(r)
        return float(np.median(vals)) if vals else float("nan")

    obs = stat(u)
    null = np.full(N_NULL, np.nan)
    for i in range(N_NULL):
        a = u.copy()
        for idx in groups:
            a[idx] = u[idx][rng.permutation(len(idx))]
        null[i] = stat(a)
    p, q95 = shared.permutation_p(obs, null, ALPHA)

    hu = np.bincount(u, weights=h, minlength=n_units)
    nu = np.bincount(u, weights=n, minlength=n_units)
    have = nu > 0
    chi_p = shared.chi2_p(hu[have], nu[have])
    cell = float(h.max() / h.sum()) if h.sum() else float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(have, hu / np.where(have, nu, 1.0), -1.0)
    top = int(np.argmax(rate))
    top_cell = float(h[u == top].max() / hu[top]) if hu[top] > 0 else 0.0

    if not np.isfinite(p):
        verdict = "no test possible"
    elif p < ALPHA and chi_p < ALPHA and cell <= CELL_CAP and top_cell <= CELL_CAP:
        verdict = "own geography"
    elif p < ALPHA and cell > CELL_CAP:
        verdict = "REFUSED: one sampling unit holds over half"
    elif p < ALPHA and top_cell > CELL_CAP:
        verdict = "REFUSED: the top unit's reading is one sampling unit"
    elif p < ALPHA:
        verdict = "REFUSED: rank test passes, the units do not differ"
    else:
        verdict = "not distinguishable from chance"
    return dict(n=int(n.sum()), k=int(h.sum()), clusters=n_c, median=obs, q95=q95, p=p,
                chi_p=chi_p, cell=cell, top=top, top_cell=top_cell, verdict=verdict)


def shrink(k, n, rate_w, national_w):
    """(each unit's rate pulled toward `national_w`, prior strength in respondents).

    `k` and `n` are unweighted counts per unit, which is what the variance is estimated from;
    `rate_w` are the weighted rates being shrunk. Under a beta-binomial the between-unit variance is
    tau2 = m(1-m)/(M+1), so M = m(1-m)/tau2 - 1, with tau2 the observed spread less binomial noise.
    No spread left means every unit at the national rate (M infinite).
    """
    k, n, rate_w = (np.asarray(x, dtype=float) for x in (k, n, rate_w))
    ok = n > 0
    N = n[ok].sum()
    m = k[ok].sum() / N
    p = k[ok] / n[ok]
    s2 = float((n[ok] * (p - m) ** 2).sum() / N)
    tau2 = s2 - m * (1.0 - m) * ok.sum() / N
    if tau2 <= 0:
        return np.full(len(n), float(national_w)), float("inf")
    M = max(m * (1.0 - m) / tau2 - 1.0, 0.0)
    out = np.where(ok, (n * np.nan_to_num(rate_w) + M * national_w) / (n + M), national_w)
    return out, float(M)
