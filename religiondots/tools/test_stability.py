"""
Does sources/stability.py compute what spec §12 says it does?

Every piece of the shared split-half test is checked here against a brute-force version written
without the shortcut the module takes: halvings from a bitmask over all subsets rather than
`itertools.combinations` and a filter, ranks by counting, Spearman and medians in plain Python,
the chi-square p from the closed-form tail of an even-df chi-square, standouts and the 2x rule
by looping over every cell.

The halving enumeration is the one that has gone wrong in this tree, twice and silently. The
filter `if 0 in a` is right for an even wave count and quietly drops distinct halvings for an
odd one: Colombia lost six of ten splits on five waves (sources.md §9dk) and Turkmenistan's
three waves broke `cab.stability` (§9do). So the enumeration is checked for every count from 2
to 8, odd and even, and the old filter is shown failing on the odd ones.

Two traps are checked by name as well: a permutation applied to every wave alike is the
identity and returns the observed statistic as its own null (Belgium, spec §12), and a halving
where a category is absent from a half must not count as a standout.

Run: python tools/test_stability.py
"""
import math
import os
import random
import statistics
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "6")

import numpy as np
import pandas as pd

HERE = Path(__file__).parent.parent
sys.path.insert(0, str(HERE / "sources"))

import stability as st  # noqa: E402

TOL = 1e-12


def same(a, b):
    """Equal within TOL, with nan equal to nan."""
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return abs(a - b) <= TOL


# ---------------------------------------------------------------------------------------------
# halvings
# ---------------------------------------------------------------------------------------------

def brute_halvings(n):
    """Every unordered split of range(n) into halves of floor(n/2) and ceil(n/2), by bitmask."""
    out = set()
    everyone = frozenset(range(n))
    for mask in range(1 << n):
        a = frozenset(i for i in range(n) if mask >> i & 1)
        if len(a) == n // 2:
            out.add(frozenset([a, everyone - a]))
    return out


def old_filter(n):
    """The pre-2026-09-14 `cab` / `se` filter: keep a combination only if it holds wave 0."""
    import itertools
    return [a for a in itertools.combinations(range(n), n // 2) if 0 in a]


def check_halvings():
    print("halvings against a bitmask enumeration of every subset:")
    for n in range(2, 9):
        got = st.halvings(n)
        want = brute_halvings(n)
        pairs = [frozenset([frozenset(a), frozenset(b)]) for a, b in got]
        for a, b in got:
            assert len(a) == n // 2 and len(b) == n - n // 2, (n, a, b)
            assert list(a) == sorted(a) and list(b) == sorted(b), (n, a, b)
            assert not set(a) & set(b) and set(a) | set(b) == set(range(n)), (n, a, b)
            if n % 2 == 0:
                assert 0 in a, (n, a)
        assert len(set(pairs)) == len(pairs), f"n={n}: a halving is listed twice"
        assert set(pairs) == want, f"n={n}: {len(set(pairs) & want)} of {len(want)} halvings"
        assert [a for a, _ in got] == sorted(a for a, _ in got), f"n={n}: not in combinations order"
        expect = math.comb(n, n // 2) // (2 if n % 2 == 0 else 1)
        assert len(got) == expect, (n, len(got), expect)
        kept = len(old_filter(n))
        print(f"  {n} waves: {len(got):>2} halvings, all distinct, matches brute force; "
              f"the old `if 0 in a` filter keeps {kept}"
              + ("" if kept == len(got) else f"  <- loses {len(got) - kept}"))
        if n % 2 and n >= 3:
            assert kept < len(got), n
    assert len(old_filter(5)) == 4 and len(st.halvings(5)) == 10, "Colombia's figures, §9dk"
    assert len(old_filter(3)) == 1 and len(st.halvings(3)) == 3, "Turkmenistan's figures, §9do"
    for bad in (0, 1):
        try:
            st.halvings(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"halvings({bad}) should refuse")
    print("  five waves: 10 (Colombia lost 6 to the old filter); three: 3 (Turkmenistan's 1)")


# ---------------------------------------------------------------------------------------------
# ranks, Spearman, the statistic
# ---------------------------------------------------------------------------------------------

def brute_rank(x):
    return [sum(v < xi for v in x) + (sum(v == xi for v in x) - 1) / 2 for xi in x]


def brute_rho(x, y):
    rx, ry = brute_rank(x), brute_rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def check_rank_rho(rng):
    print("ranks and Spearman against counting:")
    n_checked = 0
    for _ in range(3000):
        n = rng.randint(2, 12)
        x = [rng.randint(0, 3) / 7 for _ in range(n)]
        y = [rng.randint(0, 5) / 3 for _ in range(n)]
        assert list(st.rank(x)) == brute_rank(x), (x, list(st.rank(x)))
        assert same(st.rho(np.array(x), np.array(y)), brute_rho(x, y)), (x, y)
        n_checked += 1
    assert math.isnan(st.rho(np.array([0.2, 0.2, 0.2]), np.array([0.1, 0.5, 0.3])))
    print(f"  {n_checked:,} random tied vectors of 2-12 units: ranks exact, rho within {TOL:g}; "
          "a constant side is nan")


def brute_median_rho(cube, splits, perm=None, tot=None):
    """Plain loops. `cube[w][u][k]`; `perm[w][u]` is the unit whose row sits at u in wave w."""
    n_w, n_u, n_k = len(cube), len(cube[0]), len(cube[0][0])
    c = [[cube[w][perm[w][u] if perm is not None else u] for u in range(n_u)] for w in range(n_w)]
    t = None
    if tot is not None:
        t = [[tot[w][perm[w][u] if perm is not None else u] for u in range(n_u)]
             for w in range(n_w)]
    out = []
    for k in range(n_k):
        vals = []
        for a, b in splits:
            sides = []
            for half in (a, b):
                shares = []
                for u in range(n_u):
                    num = sum(c[w][u][k] for w in half)
                    den = (sum(t[w][u] for w in half) if t is not None
                           else sum(c[w][u][j] for w in half for j in range(n_k)))
                    shares.append(num / den if den > 0 else None)
                sides.append(shares)
            if any(s is None for s in sides[0] + sides[1]):
                continue
            r = brute_rho(sides[0], sides[1])
            if not math.isnan(r):
                vals.append(r)
        out.append(statistics.median(vals) if vals else float("nan"))
    return out


def random_cube(rng, n_w, n_u, n_k, empty_cell=False):
    cube = [[[float(rng.choice([0, 0, 1, 2, 3, 5, 8])) for _ in range(n_k)] for _ in range(n_u)]
            for _ in range(n_w)]
    if empty_cell:
        cube[rng.randrange(n_w)][rng.randrange(n_u)] = [0.0] * n_k
    return cube


def check_median_rho(rng):
    print("median_rho against plain loops:")
    cases = 0
    for trial in range(60):
        n_w, n_u, n_k = rng.randint(2, 6), rng.randint(3, 7), rng.randint(1, 4)
        cube = random_cube(rng, n_w, n_u, n_k, empty_cell=trial % 3 == 0)
        splits = st.halvings(n_w)
        arr = np.array(cube)
        got = st.median_rho(arr, splits)
        want = brute_median_rho(cube, splits)
        assert all(same(float(g), w) for g, w in zip(got, want)), (trial, list(got), want)

        perm = [rng.sample(range(n_u), n_u) for _ in range(n_w)]
        got = st.median_rho(arr, splits, perm=np.array(perm))
        want = brute_median_rho(cube, splits, perm=perm)
        assert all(same(float(g), w) for g, w in zip(got, want)), ("perm", trial)

        # `tot` larger than the cube's own sum: a cube holding only some of the answers
        tot = [[sum(cube[w][u]) + rng.randint(1, 4) for u in range(n_u)] for w in range(n_w)]
        got = st.median_rho(arr, splits, perm=np.array(perm), tot=np.array(tot, dtype=float))
        want = brute_median_rho(cube, splits, perm=perm, tot=tot)
        assert all(same(float(g), w) for g, w in zip(got, want)), ("tot", trial)
        cases += 3
    print(f"  {cases} random cubes (2-6 waves, 3-7 units, some with an empty (wave, unit) cell), "
          "with and without a per-wave permutation and a separate total: agree")


def check_wave_null(rng):
    print("the per-wave null:")
    cube = np.array(random_cube(rng, 5, 6, 3))
    splits = st.halvings(5)
    obs = st.median_rho(cube, splits)

    # THE IDENTITY TRAP (Belgium, spec §12): one relabelling applied to every wave moves no
    # correlation, so a null built that way is the observed statistic again.
    g = np.array(rng.sample(range(6), 6))
    same_everywhere = st.median_rho(cube, splits, perm=np.tile(g, (5, 1)))
    assert np.allclose(same_everywhere, obs, atol=TOL, equal_nan=True)

    null = st.wave_null(cube, splits, n_perm=6, seed=11)
    gen = np.random.default_rng(11)
    for i in range(6):
        perm = [list(gen.permutation(6)) for _ in range(5)]
        want = brute_median_rho(cube.tolist(), splits, perm=perm)
        assert all(same(float(a), b) for a, b in zip(null[i], want)), i
    assert not np.allclose(null, obs, equal_nan=True), "the null collapsed onto the observed value"
    print("  one permutation applied to every wave returns the observed medians (the trap); "
          "wave_null's draws match a loop over the same generator and differ from them")

    # A planted geography passes and a shuffled one does not.
    n_w, n_u = 4, 8
    level = np.linspace(0.05, 0.6, n_u)
    planted = np.zeros((n_w, n_u, 2))
    for w in range(n_w):
        planted[w, :, 0] = np.round(level * 200)
        planted[w, :, 1] = 200 - planted[w, :, 0]
    sp = st.halvings(n_w)
    p, q95 = st.permutation_p(st.median_rho(planted, sp)[0],
                              st.wave_null(planted, sp, n_perm=400, seed=0)[:, 0])
    assert p < 0.05, p
    flat = planted.copy()
    for w in range(n_w):
        flat[w] = planted[w][np.random.default_rng(w).permutation(n_u)]
    p2, _ = st.permutation_p(st.median_rho(flat, sp)[0],
                             st.wave_null(flat, sp, n_perm=400, seed=0)[:, 0])
    assert p2 >= 0.05, p2
    print(f"  a planted ordering over 8 units passes (p={p:.4f}, null 95th {q95:+.3f}); the same "
          f"shares shuffled differently in every wave fail (p={p2:.4f})")


def check_permutation_p(rng):
    print("permutation_p against counting:")
    for _ in range(500):
        n = rng.randint(0, 60)
        null = [rng.choice([rng.random(), float("nan")]) for _ in range(n)]
        obs = rng.choice([rng.random(), float("nan"), 0.5])
        finite = [v for v in null if not math.isnan(v)]
        p, q = st.permutation_p(obs, np.array(null, dtype=float))
        if math.isnan(obs) or len(finite) < st.MIN_NULL:
            assert math.isnan(p) and math.isnan(q), (obs, len(finite), p)
        else:
            assert p == (1 + sum(v >= obs for v in finite)) / (1 + len(finite))
            assert min(finite) <= q <= max(finite)
    print(f"  500 random nulls: p is (1 + draws at or above) / (1 + finite draws); under "
          f"{st.MIN_NULL} finite draws or an undefined observed value is no test")


# ---------------------------------------------------------------------------------------------
# the chi-square veto
# ---------------------------------------------------------------------------------------------

def brute_chi2_p(hits, totals):
    """Pearson chi-square on the 2 x U table; tail probability in closed form for even df."""
    rows = [hits, [t - h for h, t in zip(hits, totals)]]
    n = sum(totals)
    rsum = [sum(r) for r in rows]
    if 0 in rsum or 0 in totals:
        return float("nan")
    stat = sum((rows[i][u] - rsum[i] * totals[u] / n) ** 2 / (rsum[i] * totals[u] / n)
               for i in range(2) for u in range(len(totals)))
    k, h = (len(totals) - 1) // 2, stat / 2          # df = 2k
    return math.exp(-h) * sum(h ** i / math.factorial(i) for i in range(k))


def check_chi2(rng):
    print("chi2_p against the closed-form chi-square tail:")
    for _ in range(300):
        n_u = rng.choice([3, 5, 7, 9])                   # even degrees of freedom
        totals = [rng.randint(5, 400) for _ in range(n_u)]
        hits = [rng.randint(0, t) for t in totals]
        want = brute_chi2_p(hits, totals)
        got = st.chi2_p(np.array(hits, dtype=float), np.array(totals, dtype=float))
        if math.isnan(want):
            assert math.isnan(got), (hits, totals, got)
        else:
            assert abs(got - want) <= 1e-9 * max(1.0, want), (hits, totals, got, want)
    assert math.isnan(st.chi2_p(np.zeros(4), np.array([10.0, 20, 30, 40]))), "absent everywhere"
    print("  300 random 2 x 3-9 tables agree to 1e-9; a category nobody gave is nan")


# ---------------------------------------------------------------------------------------------
# standouts and the 2x rule
# ---------------------------------------------------------------------------------------------

def check_top_both_halves(rng):
    print("top_both_halves against a loop over every halving:")
    for trial in range(300):
        n_s, n_u, n_k = rng.randint(1, 30), rng.randint(2, 8), rng.randint(1, 4)
        pick = [0.0, 0.0, 0.1, 0.1, 0.3, 0.7]
        sa = np.array([[[rng.choice(pick) for _ in range(n_k)] for _ in range(n_u)]
                       for _ in range(n_s)])
        sb = np.array([[[rng.choice(pick) for _ in range(n_k)] for _ in range(n_u)]
                       for _ in range(n_s)])
        if trial % 4 == 0:
            sb[:, :, 0] = 0.0                                      # absent from every late half
        unit, share = st.top_both_halves(sa, sb)
        for k in range(n_k):
            agree = []
            for s in range(n_s):
                col_a = [sa[s, u, k] for u in range(n_u)]
                col_b = [sb[s, u, k] for u in range(n_u)]
                ta = col_a.index(max(col_a))                       # first highest
                tb = col_b.index(max(col_b))
                if ta == tb and max(col_a) > 0 and max(col_b) > 0:
                    agree.append(ta)
            if agree:
                best = min(set(agree), key=lambda u: (-agree.count(u), u))
                assert unit[k] == best and same(float(share[k]), agree.count(best) / n_s), trial
            else:
                assert unit[k] == -1 and share[k] == 0.0, trial
        if trial % 4 == 0:
            assert unit[0] == -1, "a halving with the category absent from a half counted"
    print("  300 random (halving, unit, category) arrays with ties and empty halves agree; a "
          "category absent from every late half has no standout")


def check_halves(rng):
    print("halves against plain sums:")
    for _ in range(100):
        n_w, n_u, n_k = rng.randint(2, 6), rng.randint(2, 6), rng.randint(1, 4)
        cube = random_cube(rng, n_w, n_u, n_k, empty_cell=True)
        a, b = rng.choice(st.halvings(n_w))
        sa, sb = st.halves(np.array(cube), a, b)
        for side, half in ((sa, a), (sb, b)):
            for u in range(n_u):
                den = sum(cube[w][u][k] for w in half for k in range(n_k))
                for k in range(n_k):
                    num = sum(cube[w][u][k] for w in half)
                    if den > 0:
                        assert same(float(side[u, k]), num / den)
                    else:
                        assert math.isnan(side[u, k])
    print("  100 random cubes: shares exact, a unit with nobody in a half is nan")


def check_residual_multiples(rng):
    print("residual_multiples against a loop over every (category, unit):")
    for _ in range(300):
        units = [f"u{i}" for i in range(rng.randint(1, 8))]
        tail = [f"c{i}" for i in range(rng.randint(1, 5))]
        mult = pd.Series([rng.choice([0.5, 1.0, 1.5, 2.0, 2.5, rng.random() * 3]) for _ in units],
                         index=units)
        none = pd.DataFrame([[rng.random() < 0.4 for _ in tail] for _ in units],
                            index=units, columns=tail)
        rows, worst = st.residual_multiples(mult, none, tail)
        want_rows = []
        for c in tail:
            cand = [(mult[u], -i, u) for i, u in enumerate(units) if none.loc[u, c]]
            if cand:
                m, _, u = max(cand)                                # highest, then first unit
                want_rows.append((c, u, float(m)))
        assert rows == want_rows, (rows, want_rows)
        if want_rows:
            top = max(r[2] for r in want_rows)
            assert worst == next(r for r in want_rows if r[2] == top)
        else:
            assert worst is None
    print("  300 random tails: each category's worst unit and the overall worst agree "
          f"(the tail goes flat at {st.SMALL_CATEGORY_MULTIPLE:g}x or more)")


def check_cluster_null():
    print("cluster_null: shuffling an assignment vector is the same draw as indexing it:")
    for seed in range(20):
        for dtype in (np.int32, np.int64):
            assign = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 4, 4], dtype=dtype)
            a = np.random.default_rng(seed)
            b = np.random.default_rng(seed)
            direct = [a.permutation(assign) for _ in range(5)]
            via = st.cluster_null(lambda p: assign[p], len(assign), 5, b)
            assert all((d == v).all() for d, v in zip(direct, via)), (seed, dtype)
            assert a.random() == b.random(), "the generators left in different states"
    print("  20 seeds, int32 and int64: `rng.permutation(assign)` == `assign[rng.permutation(n)]`, "
          "and the generator ends in the same state")


def main():
    rng = random.Random(20260914)
    check_halvings()
    check_rank_rho(rng)
    check_halves(rng)
    check_median_rho(rng)
    check_wave_null(rng)
    check_permutation_p(rng)
    check_chi2(rng)
    check_top_both_halves(rng)
    check_residual_multiples(rng)
    check_cluster_null()
    print("\nall checks passed")


if __name__ == "__main__":
    main()
