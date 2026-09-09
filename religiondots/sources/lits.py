"""The EBRD Life in Transition Survey — the shared machinery for every country drawn from it.

`sources/kg.py` is the first. `sources.md` §11ag opened the source, §11aj closed Uzbekistan on
it and priced Tajikistan and the Kyrgyz Republic, and §9co is the Kyrgyz build. Read §11aj
before extending this to another country: **the binding constraint is regional coverage and
the weights do not show the hole.**

## What the file is

LiTS III, fielded late 2015 into early 2016 across 32 transition economies plus Germany and
Italy. 75 primary sampling units per country, twenty interviews each, about 1,500 achieved.
`data/raw/lits/lits_iii.dta`, 170,487,994 bytes, 51,206 rows, 1,290 columns. **No auth, no
cookie, no terms gate** (§11ag), and it carries `region_name`, `PSU_number`, `urban`,
`district_l1`/`district_l2` and three weight columns.

## THE CARD HAS EIGHT SUBSTANTIVE CODES, NOT SIX

§11ag read the printed questionnaire (p41) and recorded six: Muslim, Orthodox Christian,
Other Christian including Protestant, Jewish, Atheistic-agnostic-none, Other. **The delivered
`q922` value labels carry eight**, with `BUDDHIST` and `CATHOLIC` as well, plus `Refusal` at
-99. Both extra codes are used in the Kyrgyz sample. Corrected here rather than in the older
note, because this is the file talking.

    -99 Refusal                                 4 ORTHODOX CHRISTIAN
      1 ATHEISTIC / AGNOSTIC / NONE             5 CATHOLIC
      2 BUDDHIST                                6 OTHER CHRISTIAN, INCLUDING PROTESTANT
      3 JEWISH                                  7 MUSLIM
                                                8 OTHER

## LiTS IV DOES NOT HAVE THE QUESTION, SO THERE IS NOTHING TO POOL WITH

Established by §11aj against the delivered file: LiTS IV's 1,319 columns contain no religion
variable, and its only religious content is a country-specific identity battery that asks
which identities matter to someone rather than what their religion is. **LiTS III is the only
round with a religion question**, so every country here is one wave of about 1,500 people and
the sample cannot be grown. That is the fact the split-half has to be read against.

## THE SPLIT-HALF IS ON PSUs AND IS TESTED AGAINST A PERMUTATION NULL

`sources/lapop.py` splits its waves in half and compares each category's ordering across the
two halves against an `n`-only bar. **Neither half of that construction transfers.**

  * There is one wave, so the split has to be on something else. It is on PSUs, which is also
    the right unit: two respondents in the same PSU are the same twenty-household cluster and
    splitting on rows would split a cluster in half and count it twice.
  * A single random split of 75 PSUs is noisy enough that the answer changes run to run, so
    the statistic is the **median over 400 random PSU halves**. Any bar for ONE Spearman
    correlation is the wrong comparison for a median-of-400, which has far less draw-to-draw
    variance. That was true of `1.96/sqrt(n-1)`, which is what lapop used when this module was
    written and which at nine units sits at +0.693 against a permutation null's own 95th
    percentile of +0.45; it is **still true of the exact null that replaced it on 2026-09-09**
    (`sources/spearman_null.py`, +0.600 at nine units, enumerated over all 362,880 orderings,
    and still well clear of this module's own +0.45). Correcting the arithmetic of the
    single-correlation bar did not make it the right instrument for this statistic, and this
    module was deliberately left alone in that change.

So the null is built rather than assumed: **the PSU-to-region labels are shuffled** and the
whole median-of-400 statistic recomputed, 400 times. A category carries its own geography when
its observed median beats 95% of that null. `[[reference_check_needs_power]]` — measure the
null before believing the bar, and do not move a bar to make something pass.

**That is not one new bar. It is one per category, and it is not uniformly looser.** In
Kyrgyzstan the five denser categories' nulls sit between +0.450 and +0.550 against the fixed
+0.693, while `OTHER`'s sits at +0.750 and `JEWISH`'s at +1.000, both STRICTER than the bar
they replaced. Report those side by side for the next country too: a null that is looser where
a median-of-400 is stable and stricter where it is not is a re-calibration, and one that is
looser everywhere is a loosening. `alpha` stays at `stability`'s 0.05 default in either case,
so what changes is the instrument and never the level.

Usage: imported. `sources/kg.py` is the worked example.
"""

import itertools
import math
import os
import sys
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "lits")
DTA = os.path.join(RAW, "lits_iii.dta")

# The columns any country here needs. `district_l1`/`district_l2` and the PSU coordinates are
# in the file and are NOT read: 75 PSUs over a country give about twenty respondents per
# district, which is below anything this map can draw.
COLS = ["country", "PSU_number", "region_name", "urban",
        "weight_population", "weight_sample", "q922"]

REFUSAL = "Refusal"

# How many orderings of the units may be enumerated exhaustively rather than sampled. Nine
# units is 362,880, which numpy does in about a second, and the resulting statement is a proof
# rather than a sample. sources/arabbarometer.py has the argument for why the observed
# ordering must be excluded from the null.
EXACT_PERM_MAX = 1_000_000


def load(country_match):
    """The LiTS III rows for one country, with `region`, `code` and `w` columns.

    `country_match` is matched case-insensitively against the `country` label, which for the
    Kyrgyz Republic is the string `Kyrgyz Rep.` and not `Kyrgyzstan`.
    """
    if not os.path.exists(DTA):
        raise SystemExit(f"{DTA} missing — §11ag has the URL; it is an open download")
    df = pd.read_stata(DTA, columns=COLS, convert_categoricals=True)
    hit = df[df["country"].astype(str).str.contains(country_match, case=False, na=False)]
    if hit.empty:
        raise SystemExit(f"no LiTS III rows match country {country_match!r}; the file has "
                         f"{sorted(set(df['country'].astype(str)))}")
    hit = hit.copy()
    hit["region"] = hit["region_name"].astype(str).str.strip()
    hit["code"] = hit["q922"].astype(str)
    hit["w"] = hit["weight_population"].astype(float)
    missing = int(hit["q922"].isna().sum())
    hit = hit[hit["q922"].notna()]
    print(f"LiTS III {country_match}: {len(hit):,} respondents with a religion answer "
          f"({missing} without), {hit['PSU_number'].nunique()} PSUs, "
          f"{hit['region'].nunique()} regions")
    return hit


def coverage(df, pop, unit_col="geo_id"):
    """WHICH UNITS THE SURVEY NEVER REACHED, which is the first thing to ask of this file.

    §11aj's finding: LiTS III's 75 PSUs miss four of Uzbekistan's fourteen regions, 37.8% of
    the country, and `weight_population` sums to a plausible-looking total anyway because the
    unsampled regions simply contribute nothing to it. A weight total is not a coverage check.
    """
    seen = set(df[unit_col].unique())
    missing = sorted(set(pop.index) - seen)
    reached = pop.drop(index=missing).sum() if missing else pop.sum()
    print(f"  coverage: {len(seen)} of {len(pop)} units sampled, "
          f"{reached / pop.sum():.2%} of the population")
    if missing:
        for u in missing:
            print(f"    NOT SAMPLED  {u}  {int(pop[u]):,} people")
    return missing


def national(df):
    """Weighted national shares, indexed by the q922 label."""
    return df.groupby("code")["w"].sum() / df["w"].sum()


def held_out(df, pop, country, unit_col="geo_id", pop_source="the population table", seed=0,
             n_perm=20000, names=None):
    """Test the region decode without touching the religion column.

    The survey's weighted share of respondents per unit against the population table's share,
    ranked against every other way those units could have been paired. `sources/lapop.py` has
    the long argument and `sources/arabbarometer.py` the small-country fix this copies: the
    observed ordering is excluded from the null **by value**, and where the orderings are few
    enough all of them are checked rather than sampled.

    A survey stratified by region does well here by construction. That is fine — what this
    asks is whether the NAMES were joined to the right polygons.

    ## AN ORDERING THAT ONLY SWAPS TWO UNITS THE SURVEY CANNOT TELL APART IS NOT A WRONG ANSWER

    Kyrgyzstan hit this on the first run. The observed r is +0.9866 and **exactly one** of the
    362,879 other orderings beats it, at +0.9870: the one that swaps Issyk-Kul and Batken.
    Those two are 7.49% and 8.17% of the population. The quantity being correlated is each
    unit's share of the WHOLE sample, a multinomial proportion over all 1,500 respondents, so
    `se` below is computed on 1,500 and not on either unit's own interviews: near 7.5% that is
    ±1.33pp at 95%. **The two units are 0.69pp apart on it, half the criterion.** No
    correlation computed on that quantity can distinguish them, and calling the swap a wrong
    answer that the check caught would be reporting sampling noise as a finding.

    Do not attach that error to a unit's own sample size. Issyk-Kul and Batken are n=100 and
    n=140, and an error computed on about 120 interviews would be roughly ±4.9pp, four times
    wider; the whole-sample denominator is both the right one and the conservative one, since
    clustering and weighting inflate the true error and a narrower `se` forgives less.

    So a beating ordering is forgiven **only** when every unit it moves is moved to a unit
    whose population share is within 1.96 standard errors of its own, on the survey's own
    sample size. That is a computed criterion rather than a widened bar: it does not move
    where the check fails, it says which pairs the check has no power over, and it names them
    on every run. `[[reference_check_needs_power]]`. Anything else that beats the observed r
    still stops the build.

    ## THE WITNESS HAS TO BE ON THE JOIN THIS CHECK IS TESTING, WHICH IS THE LABEL DECODE

    The forgiveness is not a substitute for a witness, and Kyrgyzstan's record got the witness
    wrong for a day. What this check runs on is the SURVEY'S OWN REGION LABEL against the
    population table's unit. `sources/kg_geo.py`'s two named witnesses, the SOATE-to-pcode
    identity and the Russian-name match, both pin THE OFFICE to COD and say nothing about which
    pcode `И-КУЛЬСКАЯ` goes to; and swapping that label with `БАТКЕНСКАЯ` makes this check pass
    MORE cleanly than the truth, +0.9870 with none of the 362,879 orderings reaching it against
    +0.9866 with one (`sources/kg.md` §9.5). What closes it is `kg_geo.lits_decode_witness`,
    which requires each survey label to abbreviate exactly one of COD's Russian names token for
    token, and which no permutation of the decode passes.

    **Any country added to this file needs the same thing**: a witness on the label decode
    itself, not on some other join that happens to be nearby. `[[reference_name_join_wrong_neighbour]]`.
    """
    print("\n  held-out check (nothing here touches the religion column):")
    share_s = df.groupby(unit_col)["w"].sum() / df["w"].sum()
    share_p = pop / pop.sum()
    j = pd.concat([share_s.rename("survey"), share_p.rename("pop")], axis=1).dropna()
    if len(j) != len(share_s):
        raise SystemExit(f"{len(share_s) - len(j)} sampled units have no population")
    r = float(np.corrcoef(j["survey"], j["pop"])[0, 1])
    if not np.isfinite(r):
        raise SystemExit(f"the survey/{pop_source} correlation over {country}'s {len(j)} "
                         "units is undefined, so this check cannot say anything")
    ratio = (j["survey"] / j["pop"]).sort_values()
    print(f"    unit share of respondents vs {pop_source}:  r = {r:+.3f} over {len(j)} units")
    print(f"      thinnest sampled {ratio.index[0]} at {ratio.iloc[0]:.2f}x its population "
          f"share, fullest {ratio.index[-1]} at {ratio.iloc[-1]:.2f}x")

    a, b = j["survey"].to_numpy(), j["pop"].to_numpy()
    n = len(j)
    total = math.factorial(n)
    exact = total <= EXACT_PERM_MAX
    if exact:
        P = np.array(list(itertools.permutations(b)))
        how = f"all {total - 1:,} other orderings of the same units"
    else:
        rng = np.random.default_rng(seed)
        P = np.array([rng.permutation(b) for _ in range(n_perm)])
        how = f"{n_perm:,} random pairings of the same units"
    # correlate `a` against every row of P at once
    az = (a - a.mean()) / a.std()
    Pz = (P - P.mean(axis=1, keepdims=True)) / P.std(axis=1, keepdims=True)
    perm = (Pz @ az) / n
    same = (P == b).all(axis=1)
    hit = (perm >= r - 1e-12) & ~same
    print(f"      against {how}: best r = {perm[~same].max():+.4f}, and {int(hit.sum())} "
          f"reach the observed r = {r:+.4f}")
    if total <= 1_000_000:
        print(f"      {n} units allow {total:,} orderings, so the strongest this check can "
              f"say is 1 in {total - 1:,}"
              + ("" if n >= 7 else " — too weak to carry the join on its own"))

    # The survey's own standard error on each unit's share of respondents. Two units whose
    # population shares are inside this of each other cannot be told apart by any correlation
    # computed on it — see the docstring.
    se = 1.96 * np.sqrt(b * (1 - b) / len(df))
    lab = names if names is not None else list(j.index)
    real = 0
    for row in P[hit]:
        moved = [i for i in range(n) if row[i] != b[i]]
        if not all(abs(b[i] - row[i]) <= se[i] for i in moved):
            real += 1
            continue
        pairs = ", ".join(
            f"{lab[i]} ({b[i] * 100:.2f}%) <-> {lab[int(np.argmin(np.abs(b - row[i])))]} "
            f"({row[i] * 100:.2f}%), {abs(b[i] - row[i]) * 100:.2f}pp apart against "
            f"+/-{se[i] * 100:.2f}pp"
            for i in moved[: max(1, len(moved) // 2)])
        print(f"      FORGIVEN, no power to separate: {pairs}")
    if real:
        raise SystemExit(
            f"{real} of {how} beat the observed r={r:+.4f} for {country} by moving units "
            "the survey CAN tell apart. The population check does not pin this decode, so the "
            "join needs a witness that does before anything is drawn.")
    return r


def _matrices(df, units, cats, unit_col="geo_id"):
    """Per-PSU weight by category, plus each PSU's unit index. The split-half runs on these."""
    psus = sorted(df["PSU_number"].unique())
    pi = {p: i for i, p in enumerate(psus)}
    ci = {c: i for i, c in enumerate(cats)}
    ui = {u: i for i, u in enumerate(units)}
    W = np.zeros((len(psus), len(cats)))
    home = {}
    for psu, g in df.groupby("PSU_number"):
        u = g[unit_col].unique()
        if len(u) != 1:
            raise SystemExit(f"PSU {psu} spans {len(u)} units — the split cannot be on PSUs")
        home[psu] = ui[u[0]]
        for c, w in g.groupby("code")["w"].sum().items():
            W[pi[psu], ci[c]] += w
    return W, np.array([home[p] for p in psus]), psus


def _median_rho(W, assign, splits, n_units):
    """Median Spearman across `splits` PSU halves, per category. NaN where undefined.

    Vectorised over splits and categories together, because this runs `n_perm + 1` times and
    a Python loop over 400 x 400 x 9 correlations takes minutes. `rankdata(..., axis=)`
    handles the ties, which are real here: a category with no respondent in three oblasts has
    three tied zeros and `argsort().argsort()` would rank them arbitrarily.
    """
    n_psu, n_cat = W.shape
    Wtot = W.sum(axis=1)
    M = np.zeros((n_units, n_psu))
    M[assign, np.arange(n_psu)] = 1.0
    S = splits.astype(float)                               # (K, P)
    # (K, R) denominators and (K, R, C) numerators, one matmul each
    den = S @ (M * Wtot).T                                 # (K, R)
    num = np.einsum("kp,rp,pc->krc", S, M, W)
    denb = 1.0 - S
    den_b = denb @ (M * Wtot).T
    num_b = np.einsum("kp,rp,pc->krc", denb, M, W)
    with np.errstate(invalid="ignore", divide="ignore"):
        A = num / np.where(den > 0, den, np.nan)[:, :, None]
        B = num_b / np.where(den_b > 0, den_b, np.nan)[:, :, None]
    # A unit missing from one half of a split is dropped from THAT split rather than the split
    # being thrown away: the small units here have three or four PSUs each, so an all-units
    # rule would discard a third of the splits and condition the statistic on the split. The
    # mask depends only on the denominators, so it is the same for every category.
    present = (den > 0) & (den_b > 0)                      # (K, R)
    A = np.where(present[:, :, None], A, np.nan)
    B = np.where(present[:, :, None], B, np.nan)
    ra = rankdata(A, axis=1, nan_policy="omit")
    rb = rankdata(B, axis=1, nan_policy="omit")
    m = np.isfinite(ra) & np.isfinite(rb)
    n = m.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        za = np.where(m, ra, np.nan)
        zb = np.where(m, rb, np.nan)
        za = za - np.nanmean(za, axis=1, keepdims=True)
        zb = zb - np.nanmean(zb, axis=1, keepdims=True)
        sa = np.sqrt(np.nansum(za ** 2, axis=1))
        sb = np.sqrt(np.nansum(zb ** 2, axis=1))
        out = np.nansum(za * zb, axis=1) / (sa * sb)       # 0/0 -> nan for a constant column
    out = np.where((n >= 4) & (sa > 0) & (sb > 0), out, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmedian(out, axis=0), np.isnan(out).mean(axis=0)


def stability(df, nat, units, unit_col="geo_id", n_split=400, n_perm=400, seed=0, alpha=0.05):
    """WHICH CATEGORIES CARRY THEIR OWN GEOGRAPHY — a permutation null, not a fixed bar.

    Read the module docstring for why a bar for ONE Spearman correlation — `1.96/sqrt(n-1)`,
    or the exact null in `sources/spearman_null.py` that replaced it — is not the right
    comparison here, and why this module was not switched to the second one when lapop and
    arabbarometer were. Being wrong about the null's shape and being wrong about which
    statistic you are testing are two different errors, and only the first one got fixed.
    The statistic is the median Spearman across `n_split` random halves of the PSUs; the null is
    the same statistic with the PSU-to-region labels shuffled, `n_perm` times. A category
    passes when fewer than `alpha` of the null medians reach its observed one.

    A failure is a FAILURE TO DEMONSTRATE SIGNAL and not a demonstration of noise (§14.16).
    What it earns a category is the national rate inside each unit's residual and a sentence
    in `note_public`, never deletion: the people are still drawn and only the claim to know
    where they are is withdrawn.
    """
    cats = list(nat.sort_values(ascending=False).index)
    W, assign, psus = _matrices(df, units, cats, unit_col)
    rng = np.random.default_rng(seed)
    splits = np.zeros((n_split, len(psus)), dtype=bool)
    for k in range(n_split):
        splits[k, rng.permutation(len(psus))[: len(psus) // 2]] = True

    obs, undef = _median_rho(W, assign, splits, len(units))
    null = np.full((n_perm, len(cats)), np.nan)
    for jx in range(n_perm):
        null[jx] = _median_rho(W, rng.permutation(assign), splits, len(units))[0]

    print(f"\n  split-half stability on {len(psus)} PSUs, median of {n_split} halves, against "
          f"a {n_perm}-draw permutation null (§14.16, and see the lits.py docstring):")
    print(f"    {'category':<42}{'national':>9}{'median rho':>12}{'null 95th':>11}{'p':>8}  "
          "verdict")
    carries = []
    for i, c in enumerate(cats):
        nc = null[:, i][np.isfinite(null[:, i])]
        if not np.isfinite(obs[i]) or len(nc) < 20:
            print(f"    {c[:40]:<42}{nat[c] * 100:8.3f}%{'':>12}{'':>11}{'':>8}  "
                  f"no test possible, undefined in {undef[i]:.0%} of halves")
            continue
        p = (1 + int((nc >= obs[i]).sum())) / (1 + len(nc))
        passed = p < alpha
        if passed:
            carries.append(c)
        print(f"    {c[:40]:<42}{nat[c] * 100:8.3f}%{obs[i]:+12.3f}"
              f"{np.quantile(nc, 1 - alpha):+11.3f}{p:8.4f}  "
              + ("own geography" if passed else "NOT distinguishable from chance"))
    return carries


def build(df, nat, large, small, pop, units, unit_col="geo_id", unit_noun="unit"):
    """Shares x population -> counts, as a closed partition of every unit.

    Copied from `sources/lapop.py::build` and unchanged in behaviour: a unit's share of a
    category that cleared the split-half passes through untouched, and what is left of that
    unit is divided among the rest at their NATIONAL relative proportions, so the tail's
    geography is the residual of the stable measurements rather than a flat national rate.
    """
    by_unit = df.groupby([unit_col, "code"])["w"].sum().unstack(fill_value=0.0)
    for c in nat.index:
        if c not in by_unit.columns:
            by_unit[c] = 0.0
    unit_share = by_unit.div(by_unit.sum(axis=1), axis=0)

    small_total = float(sum(nat[c] for c in small))
    residual = 1.0 - unit_share[large].sum(axis=1)
    if (residual <= 0).any():
        raise SystemExit(f"units with no room for the tail: "
                         f"{sorted(residual[residual <= 0].index)}")
    print(f"    the tail is {residual.min():.1%} of {residual.idxmin()} and "
          f"{residual.max():.1%} of {residual.idxmax()}, against {small_total:.1%} nationally")

    rows = []
    for unit in units:
        p = int(pop[unit])
        for c in large:
            rows.append((unit, c, unit_share.loc[unit, c] * p, f"{unit_noun} share"))
        for c in small:
            rows.append((unit, c, residual[unit] * (nat[c] / small_total) * p,
                         f"national share within the {unit_noun}'s residual"))

    out = pd.DataFrame(rows, columns=["geo_id", "source_category", "count", "basis_note"])
    out["count"] = out["count"].round().astype("int64")
    drift = int(sum(int(pop[u]) for u in units)) - int(out["count"].sum())
    if abs(drift) > len(out):
        raise SystemExit(f"rounding drift {drift} is larger than one person per row")
    if drift:
        out.loc[out["count"].idxmax(), "count"] += drift
    print(f"  rounding drift {drift:+d} people, absorbed into the largest cell")
    return out


def lean(df, out, excluded, units, unit_col="geo_id"):
    """§3.5's lean check, WITH LEAVE-ONE-OUT, because nine units is where one of them decides.

    Correlate each excluded residual's per-unit share against every drawn category's per-unit
    share, then drop each unit in turn and report the range. Serbia's +0.60 was computed over
    168 municipalities; the same number over nine can be one city, and the leave-one-out range
    is what says which it is.
    """
    if not excluded:
        print("\n  §3.5 lean: nothing is excluded from the partition, so nothing to lean.")
        return
    print("\n  §3.5 lean check on the excluded residual, with leave-one-out over the units:")
    tot = df.groupby(unit_col)["w"].sum()
    for ex in excluded:
        share_ex = (df[df["code"] == ex].groupby(unit_col)["w"].sum()
                    .reindex(units, fill_value=0.0) / tot.reindex(units))
        n_ex = int((df["code"] == ex).sum())
        if share_ex.gt(0).sum() < 3:
            print(f"    {ex!r}: {n_ex} respondents in "
                  f"{int(share_ex.gt(0).sum())} of {len(units)} units — too few non-zero "
                  "units for a correlation to mean anything, so no lean is measurable")
            continue
        drawn = out.pivot_table(index="geo_id", columns="source_category", values="count",
                                aggfunc="sum")
        drawn = drawn.div(drawn.sum(axis=1), axis=0).reindex(units)
        for c in drawn.columns:
            full = float(np.corrcoef(share_ex, drawn[c])[0, 1])
            loo = [float(np.corrcoef(share_ex.drop(u), drawn[c].drop(u))[0, 1])
                   for u in units]
            print(f"    {ex[:20]:<22} vs {c[:28]:<30} r = {full:+.3f}  "
                  f"leave-one-out {min(loo):+.3f} to {max(loo):+.3f}")
