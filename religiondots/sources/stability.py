"""The split-half stability test's shared pieces, and a map of every copy of the test in this tree.

**Read this before writing a split-half test into a country module.** A survey category is
drawn below national level only if it passes a stability test: do two halves of the survey rank
the units the same way, more than chance would (spec §14.16, §9bi). That test was written into
about eighteen loaders one country at a time, and the copies are **not one algorithm**. This
module holds the pieces that are genuinely the same across them, and the list at the bottom says
which copy uses which piece and how each of the others differs.

**Moving a caller onto this module must not change a single drawn number.** Every migration on
2026-09-14 was checked by running the loader before and after and requiring every file it writes
and its printed tables byte-identical. A copy whose method differs stays where it is, even when
the difference looks like a bug, because fixing it is a redrawing somebody has to rule on.
`python tools/test_stability.py` checks every piece here against brute force.

## THE PIECES

`halvings(n)`
    Every distinct way to split `n` waves (or rounds) into two halves, each once. Even `n`: a
    combination and its complement are the same halving, so keep the copy holding wave 0. Odd
    `n`: the halves differ in size, nothing repeats, and every combination is its own halving.
    Spec §12, "EVERY DISTINCT HALVING MEANS ALL OF THEM WHEN THE WAVE COUNT IS ODD" (Colombia):
    the `if 0 in a` filter alone keeps 4 of 10 on five waves and 1 of 3 on three.

`median_rho(cube, splits, perm=None, tot=None)` and `wave_null(...)`
    The statistic and its null. `cube` is (wave, unit, category) UNWEIGHTED counts. For each
    halving, each unit's share of each category in either half, then Spearman (average ranks)
    across the units; the statistic is the median over the halvings. Spec §12, "ONE SPLIT-HALF
    IS A DRAW, NOT A STATISTIC" (Sweden: three halvings of four rounds gave three verdicts).
    The null relabels the units **independently in every wave**, because a single relabelling
    applied to the whole cube moves no correlation and returns the observed value as its own
    null. Spec §12, "A PERMUTATION NULL THAT SHUFFLES ABOVE THE RESAMPLING UNIT IS THE
    IDENTITY" (Belgium). `tot` is the per-(wave, unit) respondent total when `cube` holds only
    some of the answers (Bolivia's union run); without it the denominator is the cube's own sum.

`permutation_p(obs, null)`
    `(1 + draws at or above obs) / (1 + finite draws)`, and the null's 95th percentile for
    printing. Fewer than `MIN_NULL` finite draws, or an undefined observed value, is "no test
    possible" and comes back as nan. The percentile is `np.quantile` and never decides anything.

`chi2_p(hits, totals)`
    The spatial chi-square: this category against everyone else, across the units, on
    unweighted counts. A rank pass needs it under alpha too, because a column that is zero in
    most units can pass a rank test on how its ties break. Spec §12, "A RANK TEST CAN BE PASSED
    BY A COLUMN THAT IS MOSTLY ZERO" (Sweden). Undefined tables (a zero margin) are nan.

`CELL_CAP`
    Refuse a pass when one sampling cell holds more than half of the answer's respondents: a
    chi-square assumes independent respondents and a cluster is not. Spec §12, "A CHI-SQUARE
    CANNOT VETO A CLUSTER" (Uzbekistan). Only the constant lives here; what a "cell" is (wave x
    cluster, cluster, municipio, municipio-day, round x oblast) is each survey's own.

`halves(cube, a, b, tot=None)` and `top_both_halves(sa, sb)`
    The standout rule for a category that fails: over the halvings, which unit is the highest
    in BOTH halves, and in what share of them. A halving counts only when both halves have some
    of the category. At `STANDOUT_AGREE` or more, with the chi-square and the cell cap holding,
    that unit keeps its measured share and the rest take the share across the others. Spec §12,
    "A RANK TEST CANNOT SEE ONE UNIT STANDING APART, SO ASK WHICH UNIT TOPS BOTH HALVES"
    (Honduras). Ties go to the first unit.

`residual_multiples(multiple, found_none, tail)`
    The small-category rule: under §9bi's residual every tail category in a unit sits at the
    same multiple of its national share, and the tail goes flat when some category is drawn at
    `SMALL_CATEGORY_MULTIPLE` (2x) or more in a unit where the survey found none of it. Test it
    after the standouts are taken out. Spec §12, "SMALL CATEGORIES GO IN THE RESIDUAL UNLESS IT
    DRAWS ONE AT 2x WHERE THE SURVEY FOUND NONE" (Honduras, Bolivia, Puerto Rico).

`cluster_null(stat, n_clusters, n_perm, rng)`
    The null for a survey whose sampling clusters sit INSIDE the drawn units: deal the clusters
    into the units at random, keeping each unit's cluster count, and recompute the statistic.
    Spec §12, "WHERE THE SAMPLING UNITS NEST INSIDE THE DRAWN UNITS, THE NULL REGROUPS THEM"
    (Puerto Rico), and `lits.py`'s PSU label shuffle. `stat` receives the permutation and
    decides what it relabels; the caller's generator is used, so a caller that draws its splits
    from the same generator keeps its sequence.

The constants are the values every copy uses today. Callers still hold their own copies of them
(`be.STAB_*`, `cab.STAB_*`, `bo.CLUSTER_CAP`, `hn.STANDOUT_AGREE` and so on), except `tz.py`,
which reads `SMALL_CATEGORY_MULTIPLE`. Changing a constant a caller reads is a change to the map.

## KNOWN LIMITS, NOT FIXED BECAUSE FIXING THEM COULD MOVE A COUNTRY

* **A unit missing from a wave.** `median_rho` skips a halving, for that category, when some
  unit has no respondents in one half; and the per-wave permutation moves the absent unit's
  empty row onto other units, so the null skips different halvings from the statistic. Nothing
  errors. Spec §12, "A ROUND THAT SKIPS A UNIT BREAKS THE HALVING, AND BREAKS THE NULL
  DIFFERENTLY" (Ukraine). Test on the units present in every wave: `ua.py::EXPECT_ABSENT` does,
  `bo`, `co` and `cab` refuse an empty (wave, unit) cell, and `be`, `se` and `no._stability` do
  not check. Refusing belongs to the caller, because some pools are built around a known gap.
* **Unweighted counts are assumed and not checked.** Weighted counts would still rank, but the
  chi-square would read weights as respondents.
* **Ties.** A mostly-zero column ties heavily; the chi-square veto is the protection, not the
  null (`spearman_null.py` ## TIES has the measurement for the one-split family).
* **The last bit.** A permutation p counts null draws EQUAL to the observed value, so two
  arithmetics for the same rho that differ in the last bit can move a p. `co` and `bo` used
  `sqrt(Sa*Sb)` where this module takes `sqrt(Sa)*sqrt(Sb)`; nothing moved on their data, and a
  future caller changing arithmetic should rerun byte-identical rather than assume.

## EVERY COPY, BY WHAT IT COMPUTES

Checked by reading each one on 2026-09-14. "Uses" names the pieces a copy calls from here.

**1. Median over every halving of the waves, per-wave unit-label null (§9cy's construction).**
All of these compute through this module now.

* `be.py::_stability`: 7 ESS rounds, 35 halvings, no chi-square. Uses `halvings`,
  `median_rho`, `wave_null`, `permutation_p`.
* `no.py::_stability`: adds the chi-square; imported by `dk`, `lv`, `ua`. Uses those four and
  `chi2_p`.
* `se.py::_stability`: adds the chi-square. Uses the same five. Its old filter kept only
  halvings holding round 0, which was right only because Sweden pools 4 and 6 rounds.
* `cab.py::stability`: adds the chi-square and refuses an empty (wave, unit) cell; used by `uz`,
  `tm`, `tz`. Uses the same five. `cab._rho` stays because `tm.py::exact_rank` calls it.
* `co.py::stability`: adds the chi-square, and prints the chronological halving and the largest
  (wave, cluster) cell without deciding on them. Uses the same five.
* `bo.py::stability`: as `co`, but the cell cap decides, `tot` is passed because the union run's
  cube holds one answer, and it computes standouts with `halves` and `top_both_halves`.
* `tools/ess_split_half.py::run`: report only (gr, fi, fr, de, it), no writes. Uses the same
  five; its cube carries a rest-of-base column that is in the denominator and not tested.

**2. One chronological split of the waves, one Spearman, a bar rather than a null.** Not moved.

* `lapop.py::stability` (gt, sv, ec, pa, cr): early half of the waves against the late half,
  `spearman_null`'s exact 95% bar for `n` units, a 1% size gate (`ELIGIBLE_FLOOR`), no
  chi-square. Spec §12's Sweden entry says the LAPOP and barometer modules split by PSU; these
  three split by wave.
* `arabbarometer.py::stability` (eg, iq, jo): the same, with `assert_not_quota` run first.
* `afrobarometer.py::stability` (ng, lr): the same construction with the OLD `1.96/sqrt(n-1)`
  bar, no quota test, no chi-square. `spearman_null.py` explains why it was left alone.
* `ar.py::stability`: two published waves (2008, 2019) over six regions, one Spearman with an
  exact permutation p over all 720 orderings, the exact bar, and a chi-square rebuilt from
  published percentages under two assumed allocations, deciding only an OVERRIDE.

**3. One split of a one-round survey against the old fixed bar `1.96/sqrt(n-1)`.** Not moved.

* `do.py::stability`: clusters split on id parity; chi-square printed, not deciding.
* `uy.py::stability`: January-June against July-December; chi-square gates `UNDER_BAR` only.

The old bar is the standard deviation of one correlation's null, not its 95th percentile, so it
is a stricter test than it says (`ask/answered/007-cr`); here and in 4 it was not re-measured.

**4. Median over random cluster halves against the old fixed bar.** The statistic is not moved.

* `ht.py::stability`: 400 random halves of all clusters, parity split printed, `OVERRIDE`.
* `hn.py::stability`: 400 random halves of the clusters inside each department, weighted
  shares, the median held to the single-correlation bar, chi-square and largest-cluster vetoes.
  Uses `chi2_p` and `top_both_halves`. Its own standout count took a draw where a category was
  absent from a half as agreement at the first department; no printed value changed.

**5. Median over random PSU halves against a PSU-label permutation null.**

* `lits.py::stability` (kg; `uz`, `tm` and `bo` import `lits` for other things): 400 random
  halves of the PSUs, a unit missing from one half dropped from that split only (at least four
  units required), weighted shares, no chi-square, no largest-PSU cap. Uses `cluster_null` and
  `permutation_p`.

**6. Median over every one-against-two municipio halving inside each region, regrouping null.**

* `pr.py::stability`: 23,328 halvings, the 18 municipios dealt into random threes, chi-square
  over regions, largest municipio and municipio-day vetoes. Uses `cluster_null`, `chi2_p`,
  `permutation_p` (its extra rule, no test when a category is undefined in over half the
  halvings, stays in `pr`) and `top_both_halves` (the same change as `hn`; nothing moved).

**7. Not a rank test.**

* `uz.py::two_unit_test`: Tashkent city against the rest for one small group; the absolute
  difference in weighted share against a PSU shuffle within wave, plus a 2x2 chi-square.

**8. Standouts and the 2x rule outside a `stability()` function.**

* `tz.py::standouts` and `tz.py::compose`: use `halvings`, `halves`, `top_both_halves` and
  `residual_multiples`. Its own standout count asked only that the early half have the category.
* `ua.py::_standouts`: uses `halvings`, but recomputes the halves in pandas on the units present
  in both halves of each split, and breaks a tie between units by Python's set order over their
  names, which varies between runs (it cannot reach a verdict: a tie is under 50%). Not moved.
  `ua.py`'s 2x rule divides the drawn share by the national share instead of taking the unit's
  multiple, so categories tied in one unit are separated by rounding; not moved. `ua._chi`
  uses `chi2_p`.

## MIGRATION RECORD, 2026-09-14

Each caller was run unchanged (and that run matched the files on disk), changed, and run again.
Every file written and every printed line was identical. The runs compared, per change:

    be.py                   be, no, dk, lv, ua, tools/ess_split_half.py gr fi fr de it
    no.py and ua.py         no, dk, lv, ua
    se.py                   se
    cab.py                  uz, tm, tz
    co.py                   co
    bo.py                   bo
    tz.py                   tz
    hn.py                   hn
    pr.py                   pr
    lits.py                 kg, uz, tm
    tools/ess_split_half.py gr fi fr de it
"""

import itertools
import sys

import numpy as np

# The values every copy uses today. See the docstring before changing one.
STAB_ALPHA = 0.05
STAB_PERM = 2000
STAB_SEED = 0
MIN_NULL = 20
STANDOUT_AGREE = 0.95
CELL_CAP = 0.5
SMALL_CATEGORY_MULTIPLE = 2.0


def halvings(n):
    """Every distinct halving of `n` waves as `(a, b)` index tuples, `a` holding floor(n/2).

    In `itertools.combinations` order, which every caller's statistic and null were computed in.
    """
    if n < 2:
        raise ValueError(f"a halving needs at least two waves, got {n}")
    return [(a, tuple(sorted(set(range(n)) - set(a))))
            for a in itertools.combinations(range(n), n // 2)
            if n % 2 or 0 in a]


def rank(a):
    """Average ranks from 0, ties shared. Pure numpy, so no scipy import for a few numbers."""
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), dtype=float)
    r[order] = np.arange(len(a), dtype=float)
    s = np.sort(a)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def rho(x, y):
    """Spearman with average ranks, or nan if either side is constant."""
    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    sx, sy = np.sqrt((rx ** 2).sum()), np.sqrt((ry ** 2).sum())
    if sx <= 0 or sy <= 0:
        return np.nan
    return float((rx * ry).sum() / (sx * sy))


def halves(cube, a, b, tot=None):
    """Each unit's share of each category in waves `a` and in waves `b`, as (unit, category).

    The denominator is the cube's own per-unit sum, or `tot` (wave, unit) when given. A unit with
    no respondents in a half has nan shares in that half.
    """
    ta, tb = cube[list(a)].sum(axis=0), cube[list(b)].sum(axis=0)
    if tot is None:
        da, db = ta.sum(axis=1), tb.sum(axis=1)
    else:
        da, db = tot[list(a)].sum(axis=0), tot[list(b)].sum(axis=0)
    sa = ta / np.where(da > 0, da, np.nan)[:, None]
    sb = tb / np.where(db > 0, db, np.nan)[:, None]
    return sa, sb


def median_rho(cube, splits, perm=None, tot=None):
    """Median over `splits` of the Spearman between the halves' unit shares, per category.

    `perm` is one permutation of the units PER WAVE, shape (wave, unit). A halving is skipped for
    a category when a unit has no respondents in one half, or the correlation is undefined; a
    category with nothing left is nan.
    """
    cube = np.asarray(cube, dtype=float)
    if perm is not None:
        cube = np.take_along_axis(cube, perm[:, :, None].repeat(cube.shape[2], axis=2), axis=1)
        if tot is not None:
            tot = np.take_along_axis(tot, perm, axis=1)
    k = cube.shape[2]
    vals = [[] for _ in range(k)]
    for a, b in splits:
        sa, sb = halves(cube, a, b, tot)
        for j in range(k):
            if np.isnan(sa[:, j]).any() or np.isnan(sb[:, j]).any():
                continue
            r = rho(sa[:, j], sb[:, j])
            if np.isfinite(r):
                vals[j].append(r)
    out = np.full(k, np.nan)
    for j in range(k):
        if vals[j]:
            out[j] = float(np.median(vals[j]))
    return out


def wave_null(cube, splits, n_perm=STAB_PERM, seed=STAB_SEED, tot=None):
    """(n_perm, category) medians with the unit labels shuffled independently in every wave."""
    n_w, n_u, k = np.shape(cube)
    rng = np.random.default_rng(seed)
    null = np.full((n_perm, k), np.nan)
    for i in range(n_perm):
        perm = np.stack([rng.permutation(n_u) for _ in range(n_w)])
        null[i] = median_rho(cube, splits, perm=perm, tot=tot)
    return null


def permutation_p(obs, null, alpha=STAB_ALPHA, min_null=MIN_NULL):
    """`(p, null quantile at 1 - alpha)` for one category, or `(nan, nan)` if there is no test."""
    null = np.asarray(null, dtype=float)
    nc = null[np.isfinite(null)]
    if not np.isfinite(obs) or len(nc) < min_null:
        return float("nan"), float("nan")
    p = (1 + int((nc >= obs).sum())) / (1 + len(nc))
    return p, float(np.quantile(nc, 1 - alpha))


def chi2_p(hits, totals):
    """p of the 2 x unit chi-square of `hits` against `totals - hits`; nan if undefined."""
    from scipy.stats import chi2_contingency
    hits, totals = np.asarray(hits), np.asarray(totals)
    try:
        return float(chi2_contingency(np.array([hits, totals - hits]))[1])
    except ValueError:
        return float("nan")


def top_both_halves(sa, sb):
    """Per category, the unit most often highest in both halves and the share of halvings it is.

    `sa`, `sb` are (halving, unit, category) shares. A halving counts only when both halves'
    highest unit is the same AND both have some of the category. Returns `(unit, share)` arrays;
    `unit` is -1 where no halving agrees. Ties between units go to the first.
    """
    sa, sb = np.asarray(sa), np.asarray(sb)
    n_s, _, k = sa.shape
    ta, tb = sa.argmax(axis=1), sb.argmax(axis=1)
    ok = (ta == tb) & (sa.max(axis=1) > 0) & (sb.max(axis=1) > 0)
    tops = np.where(ok, ta, -1)
    unit = np.full(k, -1, dtype=int)
    share = np.zeros(k)
    for j in range(k):
        t = tops[:, j][tops[:, j] >= 0]
        if len(t):
            v, c = np.unique(t, return_counts=True)
            unit[j] = int(v[np.argmax(c)])
            share[j] = c.max() / n_s
    return unit, share


def residual_multiples(multiple, found_none, tail):
    """The 2x rule's evidence: for each tail category, its worst unit among those that found none.

    `multiple` is a Series, unit -> the unit's tail remainder over the national tail share (every
    tail category in a unit sits at that multiple under the residual). `found_none` is a
    DataFrame, unit x category, True where the survey has zero unweighted respondents. Returns
    `(rows, worst)`: `rows` is `[(category, unit, multiple)]` in `tail` order for the categories
    missing somewhere, and `worst` is the first row with the largest multiple, or None. The tail
    goes flat when `worst[2] >= SMALL_CATEGORY_MULTIPLE`.
    """
    rows, worst = [], None
    for c in tail:
        where = [u for u in multiple.index if found_none.loc[u, c]]
        if not where:
            continue
        u = max(where, key=lambda x: multiple[x])
        row = (c, u, float(multiple[u]))
        rows.append(row)
        if worst is None or multiple[u] > worst[2]:
            worst = row
    return rows, worst


def cluster_null(stat, n_clusters, n_perm, rng):
    """(n_perm, category): `stat(p)` for `n_perm` random permutations `p` of the clusters.

    `rng.permutation(assign)` and `assign[rng.permutation(len(assign))]` are the same draw, so a
    caller that shuffled an assignment vector directly gets the same null through this.
    """
    return np.array([stat(rng.permutation(n_clusters)) for _ in range(n_perm)], dtype=float)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(__doc__)
