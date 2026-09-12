"""Costa Rica — religion by province, from the LAPOP AmericasBarometer.

Reads data/raw/lapop/Grand_Merge_2004-2023_*.dta and writes data/normalized/cr.csv.
`sources/lapop.py` holds the construction and the split-half; `sources/cr.md` is this
country's record; `sources.md` §11ad assesses the source across the countries it can serve,
and `sources/gt.md` is the first one built on it.

**COSTA RICA'S CENSUS DOES NOT ASK ABOUT RELIGION, ITS HOUSEHOLD SURVEY DOES NOT EITHER, AND
THE ONE INEC SURVEY THAT ASKS PUBLISHED NO TABLE OF THE ANSWER.** That is not a search result,
it is INEC's own microdata catalogue read end to end: `sistemas.inec.cr/pad5` serves **183
studies with 45,364 variables between them**, and exactly one is a religious affiliation
question — `HC1A ¿Cuál es la religión de (el jefe/a del hogar)?` in the **Encuesta de Mujeres,
Niñez y Adolescencia 2018**, Costa Rica's MICS round 6. The Censo 2011's own dictionary is 116
variables that include indigenous people, indigenous language and Afrodescendant
self-identification and no religion; ENAHO runs to 594 variables in 2023 and 596 in 2025 with
none. The EMNA's 341-page results report tabulates religion nowhere: grep it and the only hits
are its own questionnaire annex. `sources/cr.md` has the full sweep.

## BUT INEC PUBLISHED THE NATIONAL LEVEL WITHOUT MEANING TO, AND IT IS THE CHECK ON THIS MAP

The variable-level metadata of that catalogue entry carries `HC1A`'s **weighted frequency
distribution**: 8,490 households answering, 1,566,018 weighted, and a six-box card. Read as
shares of households:

    Católica 65.16%   Cristiana (evangélica, pentecostal, mormona, otra) 25.50%
    No tiene religión 7.40%   Otra religión 1.01%
    No cristiana (animista, judía, islámica, otra) 0.45%   NS/NR 0.48%

That is an independent reading of the level, from the state, on a survey ten thousand
households strong. **Two things make the comparison honest rather than flattering.** It is the
religion of the household HEAD, one per household, against LAPOP's random adult 18 and over.
And the two cards do not line up box for box: INEC's `Cristiana` explicitly includes Mormons
and has no separate Protestant, Adventist or Witness box, so its 25.50% is the comparator for
LAPOP's evangelical **plus** Protestant **plus** Latter-day Saints **plus** Witnesses, not for
the evangelical cell alone. `sources/cr.md` §4 works the comparison through box by box; the
figures on both sides are printed by this module on every run.

## SEVEN UNITS, ALL SEVEN MEASURED

Unlike Panama and Ecuador, nothing here is undrawn: LAPOP's `prov` value labels name all seven
Costa Rican provinces, every one is sampled in every wave, and there is no `gap=`.

**And the population is INEC's own 2022 estimate rather than COD-PS**, which is Ecuador's call
(§9bn). COD-PS ships a 2021 UNFPA projection built on INEC's superseded 2013 revision; it runs
+11.24% on Heredia and -3.25% on Guanacaste against INEC's own count, a 14.5-point spread in
both directions across seven units, and it is the older vintage besides. `sources/cr_geo.py`
prints the comparison on every run.

## What the country looks like

Usage:
    python sources/cr.py --fetch    rebuild the slim extract from the 1.1 GB LAPOP .dta
    python sources/cr.py            rebuild data/normalized/cr.csv from the slim extract
"""

import itertools
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import lapop

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOOKUP = os.path.join(ROOT, "data", "geo", "cr", "cr_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "cr.csv")

PAIS = 6                       # LAPOP's country code for Costa Rica
WAVES = [2010, 2012, 2014, 2023]
SOURCE_ID = "cr_lapop_2010_2023"
N_UNITS = 7

# ---------------------------------------------------------------------------------------
# THE HELD-OUT BARS. TWO WERE PRE-REGISTERED, BOTH FAILED, AND NEITHER WAS MOVED.
#
# Panama's review (`sources/pa.md` §6.1) found that its bars could not be shown to precede its
# numbers, and that both sat exactly one order of magnitude above the observed value, which is
# what a fitted bar looks like whether or not it is one. So these were derived from arithmetic
# that needs no data — 7! = 5,040 — and written down first, in
# `<scratchpad>/967ffe99-cr-BARS-SET-BEFORE-ANY-NUMBERS.md`, before `lapop.load` was ever
# called for `pais=6`. That file is the evidence; this comment is the claim.
#
# **AND THAT FILE PRE-REGISTERED TWO TESTS WHILE THIS MODULE ONLY EVER RAN ONE.** Its primary
# is BAR_EXACT below. Its secondary is an unweighted-by-size exact test — Spearman of LAPOP's
# per-unit share against the population share over the same 5,040 orderings, bar 1e-2 — chosen
# because Spearman gives no single unit any leverage, which is the flaw Panama's review had
# just found in re-running a levels test with the dominant unit dropped. It is not computed
# here and it was reported nowhere until 2026-09-09. It returns **rho = +0.5714, 503 of the
# 5,039 other orderings, exact p = 9.98e-2**, so it fails by a factor of ten. The power
# argument below was made about the levels test and covers the rank test a fortiori: Spearman
# throws the magnitudes away, and all FIVE ranks that disagree are the five shelf provinces the
# levels test could not order either. **Pre-registering two tests and reporting only the one
# that did better is worse than not pre-registering at all**, which is why this sits at the top
# of the file rather than in a footnote. `sources/cr.md` §3 has the paragraph and §7 is the
# review that found the omission.
#
# BAR_EXACT. The smallest attainable exact p at seven units is 1/5040 = 1.98e-4, reached only
# if the observed ordering is the unique maximiser. `lapop.held_out`'s shared rule — zero of
# 20,000 sampled orderings — demonstrates p < 1.50e-4 at 95%, which is BELOW that floor, so no
# correct decode of a seven-unit country can clear it and the module's own docstring says so.
# 1e-3 is the round number just above 5x the floor: "at most five of 5,040". It is **looser**
# than the shared rule, by about 6.7x, and it is loosened because the shared rule is provably
# unattainable here, not because Costa Rica needs the room.
BAR_EXACT = 1e-3

# **AND BAR_EXACT WAS NOT MET. IT HAS NOT BEEN MOVED.** The observed exact p is 6.95e-3,
# seven times the bar. What follows is why this file reports that test instead of asserting on
# it, and it is an argument about POWER, made the way `lapop.py` made it when it retired the
# mean-age check: "a number that decides without power is a coin toss wearing a lab coat".
#
#   1. Costa Rica is San José at 1.60M, Alajuela at 1.04M, and then FIVE provinces between
#      412,808 and 545,092 — a shelf. All 35 beating orderings keep San José and Alajuela in
#      place and only shuffle the shelf, which no correlation against population can order.
#   2. Perturbing the true shares by lognormal design noise and re-enumerating: at Costa
#      Rica's own observed noise (sd of log(LAPOP share / population share) = 0.214) a
#      CORRECT decode fails a 1e-3 bar 82% of the time, with a median p of 4.8e-3. The
#      observed 6.95e-3 is where a correct decode is expected to land.
#   3. And the same statistic on the countries this module has already drawn, whose decodes
#      were settled independently, puts Costa Rica's correlation in the middle of them:
#           gt  22 units  r=+0.9649  p<5e-05        sv  14 units  r=+0.9685  p<5e-05
#           cr   7 units  r=+0.9686  p=6.95e-03     pa  10 units  r=+0.9957  p=1.29e-04
#           ec  23 units  r=+0.9938  p<5e-05
#      **Costa Rica's r is El Salvador's r to four decimals and Guatemala's to three**, and
#      both of those clear the shared rule overwhelmingly. The only thing that differs is the
#      number of units. Costa Rica also has the SMALLEST allocation deviation of the five
#      (0.214, against 0.254 to 0.836), so its sample is the closest to proportional in the
#      set, not the furthest.
#
# BAR_JOINT — and this one is POST HOC, which is stated here because the whole point of the
# block above is not fitting a test to its answer. It was constructed after the pre-registered
# LEVELS test came back undecisive, and before anyone had noticed that the pre-registered
# secondary was outstanding too, so it is a regression guard rather than a pre-registered pass,
# and nothing about the decode rests on it. What it adds is a second, INDEPENDENT held-out
# variable that has power exactly where population has none: **urbanisation**. The five
# provinces on the population shelf are not alike in how urban they are, LAPOP's `ur` flag
# measures that (F = 126 against sampling noise, where lapop.py retired mean age at F below 1),
# and Kontur's hexes measure it from building footprints without touching LAPOP or INEC. An
# ordering has to reach the observed correlation on BOTH to count: 35 orderings do it on
# population, 71 on urbanisation, and 3 on both.
BAR_JOINT = 1e-3
# ---------------------------------------------------------------------------------------

# What the split-half returns, asserted so a change in the data is a failure here rather than
# a silent re-drawing of the country.
#
#   1 = Católico (+0.79)                  2 = Protestante Tradicional (+0.86)
#   4 = Ninguna, creyente (+0.86)         5 = Evangélica y Pentecostal (+1.00)
#  12 = Testigos de Jehová (+0.81)
#
# **CATÓLICO WAS NOT IN THAT LIST UNTIL 2026-09-09, AND THE BAR IS WHY.** This country shipped
# with Católico — 63.06% of it, the largest category — drawn at the national rate inside each
# province's residual, because +0.7857 fell short of a bar of +0.8002. That bar was
# `1.96/sqrt(n-1)`, which is the null's standard DEVIATION and not its 95th percentile, and on
# seven provinces it was a **0.017-level** test rather than the 0.05 its docstring claimed.
# Filed as `ask/007-cr` rather than worked around; Anita ruled to make it a real 95% test, and
# `sources/spearman_null.py` now enumerates all 5,040 orderings and puts the bar at **+0.7143**.
# Católico's exact one-sided p is **0.024**, so it carries its own province shares. It is one of
# exactly two categories in the five LAPOP countries that the correction moved; El Salvador's
# `Protestante Tradicional` is the other.
#
# **The failure it used to record was real, and that is still worth knowing**, because it is the
# reason the ask was a genuine judgement rather than a typo. Católico's rank sum-of-squared-
# differences between the wave halves is 12, and NINE of those 12 are Guanacaste alone, which
# falls from 4th most Catholic province to 7th as its Catholic share goes 63.4% in 2010-2012 to
# 48.2% in 2014-2023. That is a fifteen-point move in one province, not a pair of near-ties
# changing places. What the corrected bar says is that a single province moving that far, on
# seven provinces, is within what chance produces more than 2% of the time but not more than 5%
# — which is what a 95% test is for. `sources/cr.md` §5 has the fuller argument.
#
# (Spearman is quantised at seven units: sum d² is always even, so the attainable values around
# the old +0.80 bar were +0.8214 at sum d²=10 and +0.7857 at 12, with nothing in between. The
# new bar is +0.7143, sum d²=16, and is itself an attainable value rather than a number between
# two of them. Testigos de Jehová is at +0.81, off that lattice because its shares tie in one
# wave-half.)
#
# **What changed on the map**: Católico used to be 94.9% of the tail `lapop.build` spread
# through each province's residual, so it was already drawn as very nearly one minus the four
# measured categories — at most 2.54 points from its measured share in any province (Cartago,
# 82.09% measured against 79.55% drawn) and 1.47 on average, with two adjacent pairs of the
# province ordering swapped. It is now drawn on the measured share itself, so those two
# reorderings go away and the tail is the four small categories only.
CARRIES = [1, 2, 4, 5, 12]


def _exact(a, b, label):
    """Exact permutation p for corr(a, b) over every ordering of `b`. 7! = 5,040.

    Standardise both and r is a dot product, so all orderings are one matrix product. The
    observed ordering is excluded BY VALUE rather than by index, which also excludes orderings
    that only swap units with identical values: those reproduce the observed r and no
    correlation can tell them from the truth, so counting them as beating it would be a false
    alarm rather than a catch (`sources/arabbarometer.py`'s rule).
    """
    n = len(a)
    xs = (a - a.mean()) / a.std()
    ys = (b - b.mean()) / b.std()
    obs = float(np.dot(xs, ys) / n)
    perms = np.array(list(itertools.permutations(range(n))), dtype=np.int8)
    rr = ys[perms] @ xs / n
    identical = np.all(b[perms] == b, axis=1)
    total = int((~identical).sum())
    hits = int(((rr >= obs - 1e-12) & ~identical).sum())
    p = hits / total
    best = float(rr[~identical].max())
    print(f"    {label}: r = {obs:+.4f} over {n} units; {hits:,} of the {total:,} other "
          f"orderings reach it")
    print(f"      exact p = {p:.3e}; best wrong ordering {best:+.4f}")
    return obs, p, hits, total, perms, rr, identical


def _urbanisation(df, units):
    """LAPOP's urban share per province, and Kontur's density index for the same provinces.

    Neither side touches the religion column and neither is the sample size, so this is a
    genuinely different held-out variable rather than the population test in another form.
    Kontur is modelled from GHSL, HRSL and building footprints and knows nothing about LAPOP
    or about INEC's population table.
    """
    import geopandas as gpd

    u = pd.to_numeric(df["ur"], errors="coerce")
    d = df.assign(u=u).dropna(subset=["u"])
    urban = d.groupby("geo_id").apply(
        lambda x: x.loc[x["u"] == 1, "w"].sum() / x["w"].sum(),
        include_groups=False).reindex(units)
    ns = d.groupby("geo_id").size().reindex(units)

    hexes = gpd.read_file(os.path.join(ROOT, "data", "geo", "cr", "cr_hexes.gpkg"))
    hexes = hexes[hexes["pop"] > 0]
    if len(hexes) == 0:
        raise SystemExit("cr_hexes.gpkg read returned no populated hexes — run "
                         "sources/cr_grid.py")
    hexes["lp"] = np.log(hexes["pop"])
    dens = hexes.groupby("unit").apply(
        lambda x: np.average(x["lp"], weights=x["pop"]),
        include_groups=False).reindex(units)
    if dens.isna().any() or urban.isna().any():
        raise SystemExit("a province has no hexes or no urban/rural flag")

    a = urban.to_numpy(float)
    se2 = (a * (1 - a) / ns.to_numpy(float)).mean()
    F = a.var(ddof=1) / se2
    return a, dens.to_numpy(float), F


def held_out(df, pop, names, units):
    """Test the `prov` decode without touching the religion column.

    ## NEITHER PRE-REGISTERED TEST PASSED, AND NEITHER BAR HAS BEEN MOVED

    `lapop.held_out` samples 20,000 orderings and fails if ANY reaches the observed r. At seven
    units there are 5,040 orderings in total and the smallest attainable exact p is 1.98e-4,
    against a rule that demands under 1.50e-4, so that rule is arithmetically unreachable here
    and lapop.py's own docstring names Costa Rica as one of the two countries it happens to.
    `BAR_EXACT` was set at 1e-3 from that arithmetic alone, before `lapop.load` was ever called
    for `pais=6`.

    **The observed exact p is 6.95e-3 and BAR_EXACT is 1e-3, so it fails by a factor of seven.**
    The block above `BAR_EXACT` is the case that this is a failure of POWER and not of the
    decode — the beating orderings only shuffle five provinces that are all within 32% of one
    another in population, a correct decode fails this bar 82% of the time at Costa Rica's own
    observed sampling noise, and Costa Rica's correlation is El Salvador's to four decimal
    places on a country with half as many units. So the population test is REPORTED here and
    never asserted on, which is what `lapop.py` itself did with the mean-age check once that
    was measured to have no power.

    ## AND A SECOND TEST WAS PRE-REGISTERED, IS NOT COMPUTED HERE, AND ALSO FAILED

    The bars file's secondary is a Spearman of the same two vectors over the same 5,040
    orderings at a bar of 1e-2, written down because Spearman gives no single unit leverage.
    It returns rho = +0.5714, 503 of the 5,039 other orderings, exact p = 9.98e-2, and fails
    by a factor of ten. It was reported nowhere until 2026-09-09, and this docstring said "the
    pre-registered test" in the singular until then. The block above `BAR_EXACT` and
    `sources/cr.md` §3 carry it; the short version is that the power argument covers a rank
    test a fortiori, since all five ranks that disagree are the five shelf provinces.

    ## WHAT IS ASSERTED INSTEAD, AND WHAT IT IS WORTH

    A check that cannot fail is not a check, so something has to be able to. The joint test
    below adds **urbanisation** as a second held-out variable, because the five provinces the
    population test cannot separate are not alike in how urban they are. An ordering must
    reach the observed correlation on both population and urbanisation to count.

    **It is post hoc**, constructed after the pre-registered levels test came back undecisive
    and before anyone had noticed the pre-registered secondary was outstanding too, and it
    is therefore a regression guard rather than evidence at face value. Saying so is the point:
    `sources/pa.md` §6.1 is an audit of a country that asserted its bars preceded its numbers
    when nothing on disk showed it, and the fix is to state the order rather than to imply it.

    ## THE DECODE DOES NOT REST ON ANY OF THIS

    This family of tests exists to catch an INFERRED decode: Guatemala's `prov - 200`, El
    Salvador's alphabetical pcodes, Panama's near miss. **Costa Rica's is not inferred.** The
    Grand Merge's own `prov` value-label set spells all seven provinces out in Spanish, in
    Costa Rica's official province order, and `sources/cr_geo.py` joins those names to COD-AB's
    names with no alias at all and no spare unit on either side. That is what the map rests on,
    and it is the assertion that would actually catch a wrong decode.
    """
    print("\n  held-out checks (nothing here touches the religion column):")
    share = df.groupby("geo_id")["w"].sum() / df["w"].sum()
    j = pd.concat([share.rename("lapop"), (pop / pop.sum()).rename("pop")],
                  axis=1).dropna().reindex(units)
    if j.isna().any().any():
        raise SystemExit("units with population but no LAPOP rows")
    ratio = (j["lapop"] / j["pop"]).sort_values()
    print(f"    unit population share, LAPOP vs INEC 2022, over {len(j)} units:")
    print(f"      thinnest sampled {names[ratio.index[0]]} at {ratio.iloc[0]:.2f}x its "
          f"population share, fullest {names[ratio.index[-1]]} at {ratio.iloc[-1]:.2f}x")

    a, b = j["lapop"].to_numpy(float), j["pop"].to_numpy(float)
    obs, p, hits, total, perms, rr, ident = _exact(a, b, "population, levels, exhaustive")

    big = int(np.argmax(b))
    beating = (rr >= obs - 1e-12) & ~ident
    moved = int((beating & (perms[:, big] != big)).sum())
    print(f"      of the {hits:,} that reach it, {hits - moved:,} keep "
          f"{names[j.index[big]]} (the largest unit) in place and {moved:,} move it")
    verdict = "meets" if p < BAR_EXACT else "does NOT meet"
    print(f"      this {verdict} BAR_EXACT = {BAR_EXACT:.0e}, and it is REPORTED, NEVER "
          "ASSERTED — read the")
    print("      block above BAR_EXACT for why that is a power failure rather than a decode "
          "failure")

    # ---- the second held-out variable, and the joint test ----
    ua, ud, F = _urbanisation(df, units)
    print(f"\n    urbanisation, a variable the population test never touched: LAPOP's urban "
          f"share runs\n      {ua.min():.1%} to {ua.max():.1%} across the seven provinces, "
          f"and between-province variance is {F:.0f}x the\n      sampling variance on it, so "
          "this comparison has real power (lapop.py retired mean age at F<1)")
    uobs, up, uhits, _, _, urr, uident = _exact(ua, ud, "urbanisation vs Kontur, exhaustive")

    both = int((beating & (urr >= uobs - 1e-12) & ~uident).sum())
    tot = int((~ident & ~uident).sum())
    pj = both / tot
    print(f"\n    JOINT (POST HOC — a regression guard, not a pre-registered pass): an "
          f"ordering must reach\n      BOTH. {hits} do it on population, {uhits} on "
          f"urbanisation, {both} on both, of {tot:,} -> exact p = {pj:.3e}")
    if pj >= BAR_JOINT:
        raise SystemExit(
            f"the joint held-out test gives exact p = {pj:.3e} against a bar of "
            f"{BAR_JOINT:.0e}. Two independent variables now both fail to pin the province "
            "decode. That is not the power argument this file makes for the population test "
            "alone — STOP and re-read LAPOP's `prov` value labels before drawing anything.")


def main():
    if "--fetch" in sys.argv:
        lapop.fetch()

    df = lapop.load(PAIS, WAVES)

    # THE DECODE. cr_geo.py wrote `lapop_prov` beside each pcode after joining on the NAME —
    # LAPOP's own `prov` value labels, not a numbering. Costa Rica is the country in this set
    # where `prov - 600` happens to be right as well, which is exactly why this reads the
    # lookup instead: the same-looking arithmetic mispairs twelve of fourteen in El Salvador
    # and eight of ten in Panama.
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str, "unit": str})
    prov_to_unit = dict(zip(lut["lapop_prov"].astype(int), lut["unit"]))
    if len(prov_to_unit) != N_UNITS:
        raise SystemExit(f"{len(prov_to_unit)} prov codes in the lookup, expected {N_UNITS}")
    bad = sorted(set(df["prov_code"]) - set(prov_to_unit))
    if bad:
        raise SystemExit(f"prov codes with no province: {bad} — re-run sources/cr_geo.py")
    df["geo_id"] = df["prov_code"].map(prov_to_unit)

    print(f"Costa Rica: {len(df):,} respondents with a religion answer and a province, "
          f"waves {WAVES[0]}-{WAVES[-1]}")

    pop = lut.set_index("unit")["pop"].astype("int64")
    national = int(pop.sum())
    print(f"  INEC Estimación de Población y Vivienda 2022: {national:,} people over "
          f"{len(pop)} provinces, all seven of them measured")
    if sorted(df["geo_id"].unique()) != sorted(pop.index):
        raise SystemExit("LAPOP's provinces and the lookup's do not agree: "
                         f"{sorted(set(pop.index) ^ set(df['geo_id'].unique()))}")

    names = dict(zip(lut["unit"], lut["name"]))
    units = sorted(pop.index)
    held_out(df, pop, names, units)

    nat = lapop.national(df)
    large = lapop.stability(df, nat, N_UNITS)
    small = [c for c in nat.index if c not in large]
    if sorted(large) != sorted(CARRIES):
        raise SystemExit(
            f"the split-half now selects {sorted(large)}, not {sorted(CARRIES)}. That is a "
            "change in what this country claims to know, not a bug — read the numbers above, "
            "then update CARRIES and the docstring deliberately.")
    print(f"    -> {len(large)} categories drawn on their own province shares, "
          f"{len(small)} spread at the national rate")

    out = lapop.build(df, nat, large, small, pop, units, unit_noun="province")

    out["geo_level"] = "provincia"
    out["geo_name"] = out["geo_id"].map(names)
    out["basis"] = "self_id"
    out["year"] = "2010-2023"
    out["source_id"] = SOURCE_ID
    out["n_prov"] = out["geo_id"].map(df.groupby("geo_id").size())
    out["note"] = out.apply(
        lambda r: (f"LAPOP AmericasBarometer waves 2010-2023 pooled, n={r.n_prov} in this "
                   f"province; {r.basis_note} applied to INEC's 2022 provincial population"),
        axis=1)

    total = int(out["count"].sum())
    if total != national:
        raise SystemExit(f"drawn {total:,} against INEC {national:,}")

    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "note"]
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, "
          f"{out['source_category'].nunique()} categories, {out['geo_id'].nunique()} units)")

    print("\n  national, as drawn:")
    drawn = (out.groupby("source_category")["count"].sum() / total).sort_values(ascending=False)
    for cat, sh in drawn.items():
        print(f"    {sh * 100:6.2f}%  {cat}")

    # The pooled level averages over four rounds that disagree, and it is what the INEC
    # comparison in the docstring is against. Printed because cr.md and countries.py both
    # cite it and it is not recoverable from cr.csv.
    print("\n  the same answers by wave, weighted (%), which is what pooling averages over:")
    by_wave = df.groupby(["wave", "code"])["w"].sum().unstack(fill_value=0.0)
    by_wave = by_wave.div(by_wave.sum(axis=1), axis=0) * 100
    header = "".join(f"{int(w):>9}" for w in by_wave.index)
    print(f"    {'category':<44}{header}")
    for c in sorted(nat.index, key=lambda k: -nat[k]):
        row = "".join(f"{by_wave.loc[w, c]:>9.2f}" for w in by_wave.index)
        print(f"    {lapop.CATEGORY[c][:42]:<44}{row}")

    # Per-province shares of the three that carry their own geography, for cr.md and the note.
    print("\n  by province, the categories drawn on their own shares (%):")
    by_unit = df.groupby(["geo_id", "code"])["w"].sum().unstack(fill_value=0.0)
    by_unit = by_unit.div(by_unit.sum(axis=1), axis=0) * 100
    ns = df.groupby("geo_id").size()
    print(f"    {'province':<14}{'n':>7}" + "".join(f"{lapop.CATEGORY[c][:16]:>18}"
                                                    for c in CARRIES))
    for u in sorted(by_unit.index, key=lambda k: -by_unit.loc[k, CARRIES[0]]):
        row = "".join(f"{by_unit.loc[u, c]:>18.1f}" for c in CARRIES)
        print(f"    {names[u]:<14}{ns[u]:>7,}{row}")


if __name__ == "__main__":
    main()
