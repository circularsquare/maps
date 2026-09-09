"""LAPOP AmericasBarometer — the parts every country drawn from it shares.

`sources.md` §11ad is the assessment of the source; `sources/gt.md` is the first country built
on it and carries the worked argument. This module exists because **§11ad found nine countries
this survey could serve**, and nine hand-copied versions of the construction below would drift
apart in exactly the places that matter — which waves are pooled, what counts as a missing
answer, and which categories are allowed to carry a geography.

What is here is only what is genuinely the same in every country. What is NOT here, and must
stay in each `sources/<cc>.py`:

  * the decode from `prov` to a polygon. **It is different in every country and it is the
    single most dangerous step.** Guatemala's `prov` is 200 plus the official department
    number and COD's pcode is the same number, so it joins on the code; El Salvador's is 300
    plus the official number while COD's pcodes are ALPHABETICAL, so a code join mispairs
    twelve of fourteen and it must join on the name. See `sources/sv_geo.py`.
  * `CARRIES`, the list of categories drawn on their own unit shares, asserted per country so
    a change in the data is a failure here rather than a silent redrawing.
  * `OVERRIDE`, where a country draws a category that FAILED the split-half. One named
    category, a written reason, and a person's decision — see `stability()`. Guatemala has
    one (`Ninguna (creyente)`, +0.21, Anita 2026-09-08); El Salvador has none.
  * every word of the docstring a reader will actually consult.

`held_out()` and `stability()` ARE here, and both were per-country until El Salvador showed
why they should not be: the age check they used to share turned out to have no power in
either country, and finding that once fixed it everywhere. Their thresholds are arguments
rather than per-country constants for the same reason.

## The construction, in one paragraph

Pool the waves that asked, weight by `weight1500`, cut by `prov`. That gives a share per unit.
The magnitude comes from OCHA COD-PS, joined by the country's own module. **No magnitude is
invented: every person drawn is a person COD-PS counts in that unit, and the survey only
decides the column** (spec §14.4 rule 1, the construction `sources/kz.py` uses). Every row is
`modelled` in §7's sense, because nobody counted religion in any of these countries.
"""

import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LAPOP_DIR = os.path.join(ROOT, "data", "raw", "lapop")
DTA = os.path.join(LAPOP_DIR, "Grand_Merge_2004-2023_LAPOP_AmericasBarometer_v1.0_FREE.dta")
SLIM = os.path.join(LAPOP_DIR, "lapop_slim.feather")

# The ten columns any country here needs out of 1,408. `q2` and `ur` are for the held-out
# checks and never build a count.
USECOLS = ["pais", "year", "wave", "q3c", "q3cn", "prov", "weight1500", "wt", "q2", "ur"]

# LAPOP's `a`/`b`/`c`/`z` are don't-know / no-answer / not-applicable / not-asked-here, and
# they arrive as strings in the same column as the numeric codes.
MISSING = {"a", "b", "c", "z", "", "."}

# The waves that carry the religion question AT ALL. It does not appear before 2010, and the
# 2021 round carries neither religion nor geography — it is the COVID telephone round. A
# country may have fewer than these; it can never have more.
RELIGION_WAVES = [2010, 2012, 2014, 2016, 2018, 2023]

# q3c / q3cn value labels, verbatim from the .dta's `q3c_es` label set (spec §2.4: the
# source's own words travel with the row). The four-digit codes are country-specific options.
CATEGORY = {
    1: "Católico",
    2: "Protestante, Protestante Tradicional o Protestante no Evangélico",
    3: "Religiones Orientales no Cristianas",
    4: "Ninguna (Cree en un Ser Superior pero no pertenece a ninguna religión)",
    5: "Evangélica y Pentecostal",
    6: "Iglesia de los Santos de los Últimos Días (Mormones)",
    7: "Religiones Tradicionales",
    10: "Judío (Ortodoxo, Conservador o Reformado)",
    11: "Agnóstico o ateo (no cree en Dios)",
    12: "Testigos de Jehová",
    77: "Otro",
    1501: "Espírita Kardecista",
    2701: "Musulmán",
    2702: "Hindú",
    4113: "Musulmán",
    4114: "Ortodoxo griego / Ortodoxo oriental",
}

# A category is ELIGIBLE for its own geography if it is at least this much of the country.
# §11ad measured this instrument's sub-national cut failing below about 1%, against two
# censuses, with Jehovah's Witnesses at 0.5% coming out ANTI-correlated in Peru. Eligibility
# is not permission — `stability()` is the stricter test and it decides.
ELIGIBLE_FLOOR = 0.01


def valid(s):
    """LAPOP columns mix numeric codes with letter missing-codes; this is the real answers."""
    return s.notna() & ~s.astype(str).str.strip().str.lower().isin(MISSING)


def fetch():
    """Slim the 1.1 GB Stata file to the ten columns any country needs.

    The .dta itself is NOT downloaded here. LAPOP's Grand Merge is behind a click-through on
    `lapopsurveys.org` (free, no institutional affiliation, a name and an email address); the
    file to ask for is `Grand_Merge_2004-2023_LAPOP_AmericasBarometer_v1.0_FREE.dta.zip`,
    62 MB, and it unzips to the path above. `sources/gt.md` has the walk-through.
    """
    import pyreadstat
    if not os.path.exists(DTA):
        raise SystemExit(f"{DTA} missing — see sources/gt.md for the download")
    print(f"reading {os.path.getsize(DTA) / 1e9:.2f} GB, {len(USECOLS)} of 1,408 columns…")
    df, _ = pyreadstat.read_dta(DTA, usecols=USECOLS, apply_value_formats=False)
    df.to_feather(SLIM)
    print(f"wrote {SLIM} ({os.path.getsize(SLIM):,} bytes, {len(df):,} rows)")


def load(pais, expect_waves):
    """One country's respondents who answered the religion question and have a `prov`.

    Returns a frame with `code` (the religion), `prov_code`, `w` (the weight) and `wave`.
    Mapping `prov_code` to a polygon is the caller's job and is not the same twice.
    """
    if not os.path.exists(SLIM):
        raise SystemExit(f"{SLIM} missing — run the country's module with --fetch")
    df = pd.read_feather(SLIM)
    df = df[df["pais"] == pais].copy()

    # q3c is the question; q3cn is the same question under a second variable name in the
    # waves that carry it. They never both answer for one respondent, and this asserts it.
    both = (valid(df["q3c"]) & valid(df["q3cn"])).sum()
    if both:
        raise SystemExit(f"{both} respondents in pais={pais} answer both q3c and q3cn — the "
                         "two instruments are not disjoint and pooling them would double count")
    df["rel"] = df["q3c"].where(valid(df["q3c"]), df["q3cn"].where(valid(df["q3cn"])))

    df = df[valid(df["rel"]) & valid(df["prov"])].copy()
    df["code"] = df["rel"].astype(float).astype(int)
    df["prov_code"] = df["prov"].astype(float).astype(int)
    df["w"] = df["weight1500"].fillna(1.0)

    unknown = sorted(set(df["code"]) - set(CATEGORY))
    if unknown:
        raise SystemExit(f"religion codes with no label: {unknown} — the card has changed")

    waves = sorted(int(w) for w in df["wave"].unique())
    if waves != list(expect_waves):
        raise SystemExit(f"waves changed for pais={pais}: {waves}, expected {expect_waves}")
    if not set(waves).issubset(RELIGION_WAVES):
        raise SystemExit(f"waves outside the religion set: {sorted(set(waves) - set(RELIGION_WAVES))}")
    return df


def national(df):
    """Weighted national share per religion code, pooled over the waves in `df`."""
    return df.groupby("code")["w"].sum() / df["w"].sum()


def _age_bands(pop):
    """COD-PS's `T_*` age columns -> {column: midpoint}, read off whatever the file has.

    The bands are NOT the same in every country: Guatemala's top band is `T_70Plus` and
    El Salvador's is `T_80Plus`, so a hard-coded list silently drops the oldest people in
    half the countries here. Derived from the columns present instead.
    """
    import re
    mids = {}
    for c in pop.columns:
        m = re.fullmatch(r"T_(\d+)_(\d+)", str(c))
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            # LAPOP interviews adults, so the comparison has to be against adults. The child
            # bands are dropped entirely and 15-19 is carried at ~40% by `held_out`. Leaving
            # them in silently turns "mean adult age" into mean age of the whole population,
            # which still correlates and is measuring something else.
            if hi < 15:
                continue
            mids[c] = (lo + hi) / 2
            continue
        m = re.fullmatch(r"T_(\d+)Plus", str(c))
        if m:
            # an open top band; +5 is the usual convention and the test is a correlation,
            # so the exact value moves every unit the same way and cannot create a signal
            mids[c] = int(m.group(1)) + 5
    if not mids:
        raise SystemExit(f"no T_<age> columns in COD-PS: {list(pop.columns)[:12]}")
    return mids


def held_out(df, pop, country, unit_col="geo_id", n_perm=20000, seed=0, pop_source="COD-PS"):
    """Test the `prov` decode without touching the religion column.

    §11ad validated this instrument's sub-national cut against censuses in Mexico, Peru and
    Suriname. **That is evidence about LAPOP, not about a particular country's join**, so
    every country here needs a local test that a permuted decode would fail.

    ## THE TEST IS THE PERMUTATION, NOT THE CORRELATION

    `pop_source` names the population table in the printed output and nothing else. It is a
    parameter because **Ecuador is not drawn on COD-PS**: it has a 2022 census, COD-PS's
    2020 projection misses it by 3.4% unevenly, and a line of output that says `COD-PS` when
    the magnitude came from INEC would be a false provenance on screen. See `sources/ec.py`.

    The check is LAPOP's weighted unit distribution against the population distribution,
    and what makes it evidence is **how it compares with the wrong answers**: the unit labels
    are shuffled `n_perm` times and the observed r is ranked against those. A correlation
    that any random pairing could produce says nothing; one that beats every random pairing
    pins the decode. `sources/kz.py` used the same construction (`r=0.897 over 183 wards,
    which none of 2,000 random pairings comes near`) and it is the right shape here too.

    ## AND IT IS THE WRONG SHAPE BELOW ABOUT TEN UNITS — UNCHANGED HERE, DELIBERATELY

    "Fail if ANY sampled pairing reaches the observed r" assumes the possible orderings vastly
    outnumber the draws, and stops being true on a small country. `n` units give `n!`
    orderings, one of which is the CORRECT one, so 20,000 draws return the right answer back
    with probability about `20000/n!` and the check then hard-fails a perfect decode on
    nothing but unit count: at 8 units that is roughly every other run. `sources/arabbarometer.py`
    hit it (Lebanon has about eight governorates), and `held_out` there now excludes the
    observed ordering from the null and enumerates every ordering exhaustively when there are
    few enough of them. Read that one before copying this one.

    **Not ported, because nothing this module draws is anywhere near it.** El Salvador is the
    smallest at 14 units, and 14! = 8.7e10 against 20,000 draws is a probability of 2e-7;
    Guatemala has 22 and Ecuador 23. Changing a file three built countries import, to fix a
    failure none of them can have, is the worse trade. **A future LAPOP country with under
    about ten first-level units — Costa Rica has 7 provinces, Panama 10 — will hit it on its
    first run**, and the fix is arabbarometer's, sitting there written.

    ## AND THE AGE CHECK IS REPORTED, NEVER ASSERTED, BECAUSE IT HAS NO POWER

    An earlier version of this ran a SECOND test — mean adult age per unit against COD-PS's
    own age bands — and asserted on it. **It was measured afterwards and it cannot
    discriminate anything.** For a between-unit correlation to mean something, the units'
    true means have to differ by more than the sampling noise on them, and they do not:

        Guatemala     between-unit variance 0.887  vs  mean sampling variance 1.013   F=0.88
        El Salvador                         0.218                            0.609   F=0.36

    F below 1 means LAPOP's departmental mean ages are, as a set, indistinguishable from
    fourteen or twenty-two draws from the same distribution. **Guatemala's r=+0.53 was
    therefore luck and was briefly written up as a passing witness; it is not one.** El
    Salvador's r=-0.11 is the same non-result with the sign the other way, and failing a
    country on it would have been a false alarm.

    So the age comparison is still printed, with its own F beside it, and it never decides
    anything. A number on screen that cannot fail is a diagnostic; a number that decides
    without power is a coin toss wearing a lab coat.
    """
    print("\n  held-out checks (nothing here touches the religion column):")

    share_lapop = df.groupby(unit_col)["w"].sum() / df["w"].sum()
    share_pop = pop["pop"] / pop["pop"].sum()
    j = pd.concat([share_lapop.rename("lapop"), share_pop.rename("pop")], axis=1).dropna()
    if len(j) != len(share_pop):
        raise SystemExit(f"{len(share_pop) - len(j)} units have population but no LAPOP rows")
    r = np.corrcoef(j["lapop"], j["pop"])[0, 1]
    ratio = (j["lapop"] / j["pop"]).sort_values()
    print(f"    unit population share, LAPOP vs {pop_source}:  r = {r:+.3f} over {len(j)}")
    print(f"      thinnest sampled {ratio.index[0]} at {ratio.iloc[0]:.2f}x its population "
          f"share, fullest {ratio.index[-1]} at {ratio.iloc[-1]:.2f}x")

    rng = np.random.default_rng(seed)
    a, b = j["lapop"].to_numpy(), j["pop"].to_numpy()
    perm = np.array([np.corrcoef(a, rng.permutation(b))[0, 1] for _ in range(n_perm)])
    beaten = int((perm >= r).sum())
    print(f"      against {n_perm:,} random pairings of the same units: best random "
          f"r = {perm.max():+.3f}, and {beaten} reach the observed one")
    if beaten:
        raise SystemExit(
            f"{beaten} of {n_perm} random pairings of {country}'s units match or beat the "
            f"observed r={r:+.3f}. The population check does not pin this decode, so the "
            "join needs a witness that does before anything is drawn.")

    mids = _age_bands(pop)
    wts = {c: (0.4 if c.startswith("T_15_") else 1.0) for c in mids}   # 15-19 is ~40% adult
    age_pop = ((sum(pop[c] * wts[c] * mids[c] for c in mids)
                / sum(pop[c] * wts[c] for c in mids)).rename("pop"))
    ages = df.assign(age=pd.to_numeric(df["q2"], errors="coerce")).dropna(subset=["age"])
    grp = ages.groupby(unit_col)["age"]
    means, sds, ns = grp.mean(), grp.std(), grp.size()
    F = means.var(ddof=1) / ((sds / np.sqrt(ns)) ** 2).mean()
    j2 = pd.concat([means.rename("lapop"), age_pop.rename("pop")], axis=1).dropna()
    r2 = np.corrcoef(j2["lapop"], j2["pop"])[0, 1]
    print(f"    mean adult age, LAPOP vs {pop_source} bands:   r = {r2:+.3f} over {len(j2)} "
          f"({len(mids)} bands) — REPORTED, NOT ASSERTED")
    print(f"      LAPOP spans {j2['lapop'].min():.1f}-{j2['lapop'].max():.1f} years, "
          f"{pop_source} {j2['pop'].min():.1f}-{j2['pop'].max():.1f}")
    print(f"      between-unit variance / sampling variance = {F:.2f}, so this comparison "
          f"has {'real' if F >= 2 else 'NO'} power and {'is' if F >= 2 else 'is not'} "
          "evidence either way")


def stability(df, nat, n_units, unit_col="geo_id", override=None):
    """WHICH CATEGORIES CARRY THEIR OWN GEOGRAPHY — the split-half decides it.

    §14.16's test, and the reason China's Buddhism is drawn plainly while its Protestantism
    is drawn with a warning. Each category is ranked across the units in the earlier half of
    the waves and again in the later half, and the two orderings are compared. A category
    whose ranking does not replicate across halves has not demonstrated that it HAS a
    geography, whatever its pooled spread looks like.

    **THIS REPLACED A SIZE THRESHOLD, AND THE TWO DISAGREE.** Guatemala's first build cut at
    4% of the country, on the reasoning that §11ad measured this instrument failing below
    about 1% of a province. That cut would have drawn `Protestante Tradicional` (5.4%
    national) on its own department shares, and its split-half correlation is **-0.04**:
    twenty-two departments, four hundred and eighty respondents, and an ordering that does
    not survive being asked twice. Size is eligibility; stability is evidence, and only the
    second one is about whether a map should be drawn.

    The bar is not a taste. A Spearman correlation over `n` units has a standard error of
    about 1/sqrt(n-1), so the bar is what it takes to be distinguishable from zero at 95%.

    A low value is a FAILURE TO DEMONSTRATE SIGNAL rather than a demonstration of noise
    (§14.16, in those words). What it earns a category is the national rate inside each
    unit's own residual and a sentence in `note_public`, not deletion: the people are still
    drawn, and only the claim to know where they are is withdrawn.

    ## `override` — DRAWING A CATEGORY THAT FAILED, WHICH IS ANITA'S CALL AND NOT AN AGENT'S

    `{code: "the reason"}`. A category listed here is drawn on its own unit shares even though
    its split-half is under the bar, and the reason is printed on every run so it cannot
    become invisible.

    **This exists because the split-half answers a narrower question than the decision needs.**
    It asks whether the ORDERING replicates. It does not ask whether the units differ at all,
    and a category can fail the first while passing the second overwhelmingly — that is
    spec §14.16's China exactly, where Protestantism had a rank stability of +0.17 and a
    spatial chi-square of p=1.3e-84, and Anita's call was to draw it with the weakness named.

    **So the bar is never moved to make something pass.** Moving it would be fitting the test
    to the answer, and it would silently change every other category too. An override is one
    named category, with a written reason, decided by a person. Before proposing one, run the
    chi-square: if the units do NOT differ significantly, there is nothing to draw and the
    override is not available.
    """
    override = override or {}
    bar = 1.96 / np.sqrt(n_units - 1)
    waves = sorted(df["wave"].unique())
    cut = waves[len(waves) // 2]
    early, late = df[df["wave"] < cut], df[df["wave"] >= cut]
    print(f"\n  split-half stability across waves (§14.16), bar = +{bar:.2f} at 95% on "
          f"{n_units} units:")
    print(f"    {waves[0]}-{waves[len(waves) // 2 - 1]} n={len(early):,}   "
          f"{cut}-{waves[-1]} n={len(late):,}")
    print(f"    {'category':<44}{'national':>9}{'spearman':>10}{'pearson':>9}  verdict")

    carries = []
    for c in sorted(nat.index, key=lambda k: -nat[k]):
        if nat[c] < ELIGIBLE_FLOOR:
            print(f"    {CATEGORY[c][:42]:<44}{nat[c] * 100:8.2f}%{'':>10}{'':>9}  "
                  f"too small to place (§11ad)")
            continue

        def share(d):
            return d.groupby(unit_col).apply(
                lambda x: x.loc[x["code"] == c, "w"].sum() / x["w"].sum(),
                include_groups=False)

        j = pd.concat([share(early).rename("e"), share(late).rename("l")], axis=1).dropna()
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sp = j["e"].corr(j["l"], method="spearman")
            pe = j["e"].corr(j["l"])
        # A category that is zero in every unit of one half has no defined correlation,
        # which is the strongest possible failure of the test rather than a missing value.
        passed = bool(np.isfinite(sp)) and sp >= bar
        forced = c in override
        if passed or forced:
            carries.append(c)
        shown = f"{sp:+10.2f}{pe:+9.2f}" if np.isfinite(sp) else f"{'undefined':>10}{'':>9}"
        if passed:
            verdict = "own geography"
        elif forced:
            verdict = "own geography — UNDER THE BAR, drawn on Anita's call"
        else:
            verdict = "NOT distinguishable from zero"
        print(f"    {CATEGORY[c][:42]:<44}{nat[c] * 100:8.2f}%{shown}  {verdict}")
        if forced and not passed:
            print(f"        reason: {override[c]}")
    stale = sorted(set(override) - set(nat.index))
    if stale:
        raise SystemExit(f"override names categories this country does not have: {stale}")
    return carries


def build(df, nat, large, small, pop, units, unit_col="geo_id", unit_noun="unit"):
    """Shares x population -> counts, as a closed partition of every unit.

    THE MEASURED SHARES PASS THROUGH UNTOUCHED. A unit's share of a category that cleared
    the split-half is applied as it stands; what is left of that unit is divided among the
    rest at their NATIONAL relative proportions. So the tail's geography is the residual of
    the stable measurements rather than a flat national rate, which is a real difference the
    survey measured and should not throw away.

    Guatemala's first build renormalised the large categories to leave a FIXED national tail
    in every unit. That forces their combined share to be identical everywhere, which the
    data contradicts, so it is gone.

    `pop` is a Series indexed like `units`; `units` is the ordered list of unit ids.
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
            rows.append((unit, CATEGORY[c], unit_share.loc[unit, c] * p,
                         f"{unit_noun} share"))
        for c in small:
            rows.append((unit, CATEGORY[c], residual[unit] * (nat[c] / small_total) * p,
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
