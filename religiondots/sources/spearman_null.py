"""The exact 95% critical value for ONE Spearman correlation over `n` units.

**Read this before adding a split-half test to any module.** It is one function, and the
reason it exists is a bug that ran in two modules for a day:

    bar = 1.96 / np.sqrt(n_units - 1)

`1/sqrt(n-1)` is the exact standard DEVIATION of Spearman's rho under the null, so that line
looks right and its docstring said it was "what it takes to be distinguishable from zero at
95%". **A standard error is not a critical value.** The null is not normal at the unit counts
this project works at — it is a discrete, short-tailed distribution over `n!` orderings — and
1.96 SDs sits well out in its tail. Enumerated, the old line was a **0.017-level test at 7
units** and a 0.023-level test at 12, stricter than advertised everywhere and worst where there
are fewest units. `ask/answered/007-cr` has the measurement and Anita's ruling, 2026-09-09:
*make it a real 95% test*, by replacing the asymptotic bar with the exact null.

## THE RULE, WHICH IS PRE-REGISTERED AND NOT TO BE TUNED

Work in integers, because rho lives on a lattice and a bar between two attainable values is a
bar nothing can land on. For a permutation `s` of `0..n-1`,

    d2  = sum_i (i - s(i))^2                    an integer
    rho = 1 - 6*d2 / (n*(n^2 - 1))              strictly DECREASING in d2

so `rho >= c` and `d2 <= D` are the same event, and the critical value is

    D   = the largest ATTAINABLE value of d2 with  P_null(d2 <= D) <= alpha
    bar = 1 - 6*D / (n*(n^2 - 1))

`bar` is the smallest ATTAINABLE rho whose one-sided exact p-value is at most `alpha`. The test
`rho >= bar` therefore has true size at most `alpha` and as close to it as the lattice allows.

**`D` must be snapped to a value the statistic can actually take, and that is not decoration.**
`d2 = 2*sum(i^2) - 2*sum(i*s(i))` is always EVEN, so half the integers are unreachable. Dropping
the snap returns, at n=7, a bar of +0.6964 sitting between the attainable +0.7143 and +0.6786 —
a bar no correlation can equal, quoted in output as though one could. It accepts exactly the same
orderings as +0.7143, so no verdict depends on it, but a printed number that names an impossible
value is the kind of thing that gets copied into a `.md` and believed. `_check()` asserts the
snap by walking one lattice step down and demanding that value fail.

**This is deliberately not `np.quantile(null, 0.95)`.** Linear interpolation between lattice
points can return a value that is not attainable and whose own tail probability exceeds alpha,
which is the same class of error in the other direction — and it does here. `ask/007-cr` sized
the problem with `np.quantile` and its "exact 95th percentile" column is a shade too generous
at every unit count: at seven units it gives +0.6786, whose own upper tail is **0.0548**, so
using it would have been a 5.5% test. This module ships +0.7143 (tail 0.0441) instead, which is
stricter than the ask's column everywhere. **The two flips the ruling was made on are the same
either way** — Costa Rica's `Católico` is +0.7857 at p=0.024 and El Salvador's `Protestante
Tradicional` is +0.5165 at p=0.031 — and the whole module was re-run to confirm that rather
than argued.

## ENUMERATE OR SAMPLE, AND THE CUTOVER IS A CONSTANT AND NOT A FEELING

`EXACT_NULL_MAX_N = 10`. Ten units is 3,628,800 orderings, which is three seconds and a proof.
Eleven is 39,916,800 and rising by a factor of `n` each step, so from there it is sampled:
`NULL_SAMPLES = 10,000,000` draws at a FIXED `NULL_SEED`, so a given `n` always returns the same
number and a re-run never quietly moves a country. Every result says which of the two it is, and
`critical_rho` returns that string so the caller can print it.

**Ten million rather than two is what it takes for the SNAP to be stable.** The bar lands on a
lattice point, so sampling error only matters when it pushes the estimate across one — and at
two million draws it did, on one seed in six at n=11 and one at n=23, by two steps each. At ten
million, six seeds agree at every unit count in use, and every seed-0 bar is identical to what
two million gave. `--check` re-runs each sampled `n` under three seeds and asserts they agree,
which takes about a minute and a half and is the point of running it.

## WHEN NOTHING CAN PASS

At `n <= 3` even a perfect ordering has p > alpha (at n=3, `P(rho = +1) = 1/6`). There is then no
attainable critical value and `critical_rho` returns `+inf`, so every category fails loudly
rather than a bar of `nan` letting everything through. `python sources/spearman_null.py --check`
asserts that, and it is the constructed rejection the bar's own guard has to survive.

## TIES

The null above has none: it is `n` distinct ranks against a permutation of themselves. Spearman
over tied shares uses average ranks, whose permutation distribution is a different one, so
`ties_note()` exists to DETECT and report ties rather than let a caller run them against the
wrong null.

**Plenty of categories here do have ties, and it was worth measuring rather than assuming.**
Wherever a category is zero in several units of one wave-half its shares tie: El Salvador's
`Religiones Orientales` ties twelve of fourteen departments in the early half, Egypt's two
answers tie five of twenty-three, Jordan's tie two of twelve. The right null for those is the
CONDITIONAL one — the observed late-half average-rank vector permuted against the observed
early-half one — and on 2026-09-09 it was run for every eligible category in all seven
countries, against 400,000 orderings (5,040, exhaustively, at n=7). **Every verdict is the
same**, and the closest call moves the safe way: El Salvador's `Religiones Orientales` goes
from p=0.054 to p=0.071, further outside rather than in. So the untied null is what ships, the
tie flag is printed so a future country's near-boundary tied category gets checked rather than
assumed, and a category that both ties heavily AND lands within about 0.02 of the bar should
have its conditional null run before it is believed.

## THE RULE WAS PRE-REGISTERED, AND IT WAS AMENDED TWICE

§9cp's lesson, applied to a rule rather than to a country: the statistic, the null, the level,
the critical-value convention, the enumerate-or-sample cutover and the predicted blast radius
were all written down before a single number was computed here. **Writing it down first is only
worth something if the amendments are written down too**, so both are below. Neither changed a
verdict, and each was found by the self-test rather than noticed afterwards.

  1. **The snap.** The rule as first written said "the largest INTEGER `D` with
     `P(d2 <= D) <= alpha`", and also called the answer "the smallest attainable value of rho".
     Those disagree, because `d2` is always even: at n=7 the integer clause gives `D = 17`, a
     bar of +0.6964, and no ordering attains it. Taken: the attainable clause, which is stricter
     and is the only one that can be printed honestly. The two accept exactly the same
     orderings, so nothing decided differently.
  2. **`NULL_SAMPLES`, 2,000,000 -> 10,000,000.** The pre-registered figure was two million and
     the seed was fixed so the bar could not move on a re-run. It moved anyway: one seed in six
     shifted the snapped bar by two lattice steps at n=11 and one did at n=23. Ten million makes
     six seeds agree at every unit count in use, and **every seed-0 bar is identical at two
     million, ten million and twenty-five million** — which is the evidence that this bought
     stability and not a different answer.

## WHERE THIS IS USED, AND THE THREE DIFFERENT BARS IN THIS TREE

There are three split-half constructions here and they are not interchangeable. A future
builder should pick one on purpose:

  1. **This module — one Spearman, exact null.** `sources/lapop.py::stability` (cr, sv, gt, ec,
     pa) and `sources/arabbarometer.py::stability` (eg, jo). One split of the waves, one
     correlation per category, and the bar depends only on `n`.
  2. **`sources/lits.py::stability` — a median over 400 random PSU halves, with its own
     per-category permutation null.** Do NOT change it to match this module. Its statistic is
     not one correlation, so no `n`-only bar is the right comparison for it, and its null is
     built per category because a four-respondent category's null runs far higher than a dense
     one's. §9co (Kyrgyzstan) is the worked case, and its own docstring has the argument.
  3. **The old asymptotic bar, still in place where the ruling did not reach.**
     `sources/afrobarometer.py::stability`, and the hard-coded `STABILITY_BAR` constants in
     `sources/do.py`, `sources/ht.py` and `sources/uy.py`. **These are not fixed.** They were
     left alone deliberately on 2026-09-09: the ruling was made on a measured blast radius of
     two categories across the LAPOP and Arab Barometer countries, and moving the bar under
     already-drawn countries nobody has re-measured is the thing AGENT_BRIEF §3 sends to Anita.
     Anything switching one of them over owes her the same before-and-after list first.

Usage: imported, or `python sources/spearman_null.py --check` to print the table and run the
self-test.
"""

import itertools
import math
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np

# The pre-registered constants. Changing one changes which categories carry a geography in
# seven countries, so a change here is a change to the map and not to a helper.
NULL_ALPHA = 0.05
EXACT_NULL_MAX_N = 10          # 10! = 3,628,800 orderings; enumerated exhaustively
NULL_SAMPLES = 10_000_000      # draws above the cutover; below this the SNAP is seed-sensitive
NULL_SEED = 0                  # fixed, so the bar is deterministic

# n -> (pmf over d2, how it was obtained). Enumeration at n=10 is a few seconds and every
# country in a build asks for the same n, so it is computed once per process.
_CACHE = {}


def _d2_scale(n):
    """`rho = 1 - 6*d2/scale`. Integer, and it is what makes the lattice exact."""
    return n * (n * n - 1)


def _null_d2(n, samples=NULL_SAMPLES, seed=NULL_SEED):
    """The null distribution of `d2` as a probability vector indexed by `d2`.

    Exact by enumeration at `n <= EXACT_NULL_MAX_N`, sampled above. Returns
    `(pmf, how)` where `how` is the sentence a caller should print.
    """
    key = (n, samples, seed)
    if key in _CACHE:
        return _CACHE[key]
    idx = np.arange(n)
    max_d2 = (n ** 3 - n) // 3          # the reversal, and the top of the lattice
    counts = np.zeros(max_d2 + 1, dtype=np.int64)

    if n <= EXACT_NULL_MAX_N:
        total = math.factorial(n)
        it = itertools.permutations(range(n))
        while True:
            chunk = list(itertools.islice(it, 200_000))
            if not chunk:
                break
            a = np.array(chunk, dtype=np.int16)
            counts += np.bincount(((a - idx) ** 2).sum(axis=1), minlength=max_d2 + 1)
        assert counts.sum() == total, (counts.sum(), total)
        how = f"exact, all {total:,} orderings"
    else:
        rng = np.random.default_rng(seed)
        left = samples
        while left:
            b = min(left, 100_000)
            perms = rng.random((b, n)).argsort(axis=1)
            counts += np.bincount(((perms - idx) ** 2).sum(axis=1), minlength=max_d2 + 1)
            left -= b
        how = f"sampled, {samples:,} draws, seed {seed}"

    pmf = counts / counts.sum()
    _CACHE[key] = (pmf, how)
    return _CACHE[key]


def critical_rho(n, alpha=NULL_ALPHA, samples=NULL_SAMPLES, seed=NULL_SEED):
    """The smallest attainable Spearman rho on `n` units with one-sided exact p <= `alpha`.

    Returns `(bar, how)`. `bar` is `+inf` where no ordering at all reaches `alpha`, which is
    every `n <= 3`: the honest answer there is that the test cannot pass, not a low bar.
    """
    if n < 2:
        raise ValueError(f"a rank correlation needs at least two units, got {n}")
    pmf, how = _null_d2(n, samples, seed)
    cdf = np.cumsum(pmf)
    # `pmf > 0` is the snap to the attainable lattice. Under enumeration it is exact; under
    # sampling, every d2 anywhere near the 5% tail is drawn thousands of times out of two
    # million, so an attainable value cannot be missed where it matters.
    ok = np.flatnonzero((cdf <= alpha + 1e-12) & (pmf > 0))
    if ok.size == 0:
        return float("inf"), f"{how} — NOTHING reaches p<={alpha:g} on {n} units"
    return 1.0 - 6.0 * int(ok[-1]) / _d2_scale(n), how


def exact_p(rho, n, samples=NULL_SAMPLES, seed=NULL_SEED):
    """One-sided `P(null rho >= rho)`. For printing beside a verdict; it never decides one."""
    if not np.isfinite(rho):
        return float("nan")
    pmf, _ = _null_d2(n, samples, seed)
    # The tolerance has to absorb floating-point error in `rho` without ever crossing a real
    # lattice gap, which is 2 in d2. 1e-9 was too tight to survive a rho quoted to four
    # decimals: +0.6786 at n=7 is d2=18 and floored to 17, reporting p=0.044 for a value whose
    # p is 0.055. Display only — `stability` decides on `rho >= bar` — but a wrong p in a
    # printed table is a number somebody copies.
    d2 = int(math.floor((1.0 - rho) * _d2_scale(n) / 6.0 + 1e-6))
    if d2 < 0:
        return 0.0
    return float(pmf[:min(d2, len(pmf) - 1) + 1].sum())


def ties_note(a, b):
    """`""` when neither ordering has ties, else a sentence naming how many. See ## TIES."""
    ta = len(a) - len(np.unique(np.asarray(a, dtype=float)))
    tb = len(b) - len(np.unique(np.asarray(b, dtype=float)))
    if not (ta or tb):
        return ""
    return (f" — TIED SHARES ({ta} in the early half, {tb} in the late), so the untied null "
            "above is not exactly this category's null")


def _check():
    """The table, plus the guards. Run as `python sources/spearman_null.py --check`."""
    used = {7: "cr", 10: "pa", 11: "jo leave-one-out", 12: "jo", 14: "sv",
            20: "ec (units in both halves)", 22: "gt", 23: "eg"}
    print(f"{'n':>4}{'fixed 1.96/sqrt(n-1)':>22}{'exact 95% bar':>15}"
          f"{'true size of fixed':>20}{'true size of new':>18}   how")
    for n in sorted(used):
        fixed = 1.96 / math.sqrt(n - 1)
        bar, how = critical_rho(n)
        print(f"{n:>4}{fixed:>22.4f}{bar:>15.4f}{exact_p(fixed, n):>20.4f}"
              f"{exact_p(bar, n):>18.4f}   {how}  [{used[n]}]")

    print("\n  guards:")
    # 1. The bar must be BOTH a valid 95% cut and the smallest one. Its own tail probability
    #    has to be at or under alpha, and the next ATTAINABLE value below it — one lattice step
    #    looser, so a lower rho and a larger d2 — has to be over alpha. Failing the first means
    #    the test is bigger than it claims; failing the second means it is needlessly strict,
    #    which is the bug this whole change is fixing.
    for n in sorted(used):
        bar, _ = critical_rho(n)
        pmf, _ = _null_d2(n)
        d_bar = int(round((1.0 - bar) * _d2_scale(n) / 6.0))
        nxt = int(np.flatnonzero(pmf[d_bar + 1:] > 0)[0]) + d_bar + 1   # next attainable d2
        looser = 1.0 - 6.0 * nxt / _d2_scale(n)
        assert pmf[d_bar] > 0, (n, d_bar, "the bar is not an attainable value of rho")
        assert exact_p(bar, n) <= NULL_ALPHA + 1e-12, (n, exact_p(bar, n))
        assert exact_p(looser, n) > NULL_ALPHA, (n, looser, exact_p(looser, n))
    print(f"    every bar from n={min(used)} to n={max(used)} is attainable, has "
          f"p<={NULL_ALPHA}, and the next attainable value below it does not")

    # 2. THE CONSTRUCTED REJECTION. n=3 has six orderings and a perfect one has p=1/6, so
    #    nothing can pass; the bar must be unattainable rather than a number a category clears.
    bar3, _ = critical_rho(3)
    assert bar3 == float("inf"), bar3
    assert not (1.0 >= bar3), "a perfect ordering on three units must NOT pass"
    print(f"    n=3: a PERFECT ordering scores rho=+1.000 at p={exact_p(1.0, 3):.4f} and the "
          f"bar is {bar3} — rejected, as it must be")
    bar2, how2 = critical_rho(2)
    assert bar2 == float("inf"), bar2
    print(f"    n=2: same, bar {bar2}   [{how2}]")

    # 3. The new bar is looser than the old one everywhere in use, which is the whole finding.
    #    If it is ever STRICTER, a category could flip pass -> fail, and that is a redrawing
    #    nobody ruled on.
    for n in sorted(used):
        bar, _ = critical_rho(n)
        assert bar < 1.96 / math.sqrt(n - 1), n
    print("    the new bar is below the old one at every n in use, so nothing can flip "
          "pass -> fail")

    # 4. Sampling noise must not move the lattice point the bar snaps to. This is the guard
    #    that made NULL_SAMPLES 10,000,000 rather than 2,000,000: at 2M it failed here.
    print("\n  the sampled bars under three seeds (a minute or so):")
    for n in sorted(n for n in used if n > EXACT_NULL_MAX_N):
        bars = [critical_rho(n, seed=s)[0] for s in (0, 1, 2)]
        print(f"    n={n:<3} " + "  ".join(f"{b:+.4f}" for b in bars)
              + ("   identical" if len(set(bars)) == 1 else "   *** DIFFER ***"))
        assert len(set(bars)) == 1, (n, bars, "the sampled bar depends on the seed — raise "
                                              "NULL_SAMPLES, do not ship a bar that moves")
    print("\n  all guards passed")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _check()
    else:
        print(__doc__)
