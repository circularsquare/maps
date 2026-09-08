"""Guatemala — religion by departamento, from the LAPOP AmericasBarometer.

Reads data/raw/lapop/Grand_Merge_2004-2023_*.dta and writes data/normalized/gt.csv.
sources/gt.md has the acquisition route, the terms, and the checks in prose.
sources.md §11ad is the assessment of the source as a whole, across all nine countries it
could serve; read that before extending this file to another one.

**GUATEMALA'S CENSUS HAS NEVER ASKED ABOUT RELIGION AND THIS IS THE ONLY ROUTE.** sources.md
§11x closed the country on two witnesses — the UNSD oracle reports 1964 as the only Guatemalan
religion tabulation ever forwarded, and IPUMS's `RELIGION` variable says the same — after
finding INE's own site behind a Radware captcha and `censopoblacion.gt` a parked domain. None
of that has changed and this file does not reopen it. It is a survey standing where a census
would be, and every row it writes is `modelled` in §7's sense.

## What is done to the survey, in one paragraph

Pooled over the six waves that asked (2010, 2012, 2014, 2016, 2018, 2023 — the question does
not appear before 2010 and the 2021 phone round carries neither religion nor geography),
weighted by `weight1500`, cut by `prov`, which is the department. That gives a share per
department. The magnitude comes from OCHA COD-PS 2024, which is INE's own projection, joined
on the pcode by `sources/gt_geo.py`. **No magnitude is invented: every person drawn is a
person COD-PS counts in that department, and the survey only decides the column** — §14.4
rule 1, the same construction `sources/kz.py` uses.

## The four largest categories carry their own geography, the other seven do not

    51.97%  Católico                                 -> department share
    34.52%  Evangélica y Pentecostal                 -> department share
     5.43%  Protestante / Protestante Tradicional    -> department share
     4.85%  Ninguna (cree en un Ser Superior)        -> department share
    ------------------------------------------------------------------------
     1.14%  Otro                                     -> national share, spread by population
     0.64%  Agnóstico o ateo                         -> national share
     0.49%  Testigos de Jehová                       -> national share
     0.42%  Mormones                                 -> national share
     0.31%  Religiones Orientales no Cristianas      -> national share
     0.22%  Religiones Tradicionales                 -> national share
     0.01%  Judío                                    -> national share

**THE CUT IS NOT A JUDGEMENT CALL, BECAUSE GUATEMALA'S OWN LIST HAS A HOLE IN IT.** There is
nothing between 1.14% and 4.85%, so any threshold in that range picks the same four. And the
reason for cutting at all is measured rather than assumed: §11ad validated this instrument's
provincial cut against three censuses already on this map, and it holds for anything that is
a real share of a province (Mexico Catholic r=+0.92, Suriname Hindu r=+0.98) and **fails below
about 1%** — Jehovah's Witnesses come out at r=+0.26 in Mexico and **r=−0.21 in Peru**, which
is worse than no information. Drawing the small categories on their own department shares
would put a confident geography on 43 respondents.

So they are drawn at the national rate everywhere. That is a known-wrong geography, stated as
such, and it is the honest one: the survey knows how many there are and not where they are.
The alternative was a 3.2% hole in the country, and spec §6.12 is about how badly a blank
reads on a dot map.

## What the survey cannot see at all, and it is not a small thing

`Religiones Tradicionales` is 0.22% here, in a country the 2018 census found to be 43.6%
indigenous. §11ad measured what this instrument does to folk practice, in the one place a
census could check it: in Suriname LAPOP's traditional-religion share is **0.21× the census's**
and the missing people come back as Christians (that cell reads 1.50× census). The card has no
Maya-spirituality option and `costumbre` is not a word on it. **So 0.22% is a floor and the
country note says so.** It is not corrected here, because §14.4 forbids inventing a magnitude
and nothing published says what the right one is.

Usage:
    python sources/gt.py --fetch    rebuild the slim extract from the 1.1 GB LAPOP .dta
    python sources/gt.py            rebuild data/normalized/gt.csv from the slim extract
"""

import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LAPOP_DIR = os.path.join(ROOT, "data", "raw", "lapop")
DTA = os.path.join(LAPOP_DIR, "Grand_Merge_2004-2023_LAPOP_AmericasBarometer_v1.0_FREE.dta")
SLIM = os.path.join(LAPOP_DIR, "lapop_slim.feather")
POP = os.path.join(ROOT, "data", "raw", "gt", "gtm_admpop_adm1_2024.csv")
LOOKUP = os.path.join(ROOT, "data", "geo", "gt", "gt_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "gt.csv")

PAIS = 2                       # LAPOP's country code for Guatemala
WAVES = [2010, 2012, 2014, 2016, 2018, 2023]
SOURCE_ID = "gt_lapop_2010_2023"

# The columns worth carrying out of a 1,408-variable file. `q2` and `ur` are here for the
# held-out checks below and are not used to build a single count.
USECOLS = ["pais", "year", "wave", "q3c", "q3cn", "prov", "weight1500", "wt", "q2", "ur"]

# LAPOP's `a`/`b`/`c`/`z` are don't-know / no-answer / not-applicable / not-asked-here, and
# they arrive as strings in the same column as the numeric codes.
MISSING = {"a", "b", "c", "z", "", "."}

# q3c / q3cn value labels, verbatim from the .dta's `q3c_es` label set (spec §2.4: the
# source's own words travel with the row).
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
}

# A category is ELIGIBLE for its own department geography if it is at least this much of the
# country — §11ad measured this instrument's provincial cut failing below about 1%, in two
# censuses, and Jehovah's Witnesses at 0.5% came out ANTI-correlated in Peru. Eligibility is
# not permission: `stability()` decides that, and it is the stricter of the two.
ELIGIBLE_FLOOR = 0.01

# What the split-half currently returns, asserted so a change in the data is a failure here
# rather than a silent re-drawing of the country.
CARRIES = [1, 5]


def valid(s):
    """LAPOP columns mix numeric codes with letter missing-codes; this is the real answers."""
    return s.notna() & ~s.astype(str).str.strip().str.lower().isin(MISSING)


def fetch():
    """Slim the 1.1 GB Stata file to the ten columns this country needs.

    The .dta itself is NOT downloaded here. LAPOP's Grand Merge is behind a click-through on
    `lapopsurveys.org` (free, no institutional affiliation, a name and an email); the file to
    ask for is `Grand_Merge_2004-2023_LAPOP_AmericasBarometer_v1.0_FREE.dta.zip`, 62 MB, and
    it unzips to the path above. sources/gt.md has the walk-through.
    """
    import pyreadstat
    if not os.path.exists(DTA):
        raise SystemExit(f"{DTA} missing — see sources/gt.md for the download")
    print(f"reading {os.path.getsize(DTA) / 1e9:.2f} GB, {len(USECOLS)} of 1,408 columns…")
    df, _ = pyreadstat.read_dta(DTA, usecols=USECOLS, apply_value_formats=False)
    df.to_feather(SLIM)
    print(f"wrote {SLIM} ({os.path.getsize(SLIM):,} bytes, {len(df):,} rows)")


def load():
    if not os.path.exists(SLIM):
        raise SystemExit(f"{SLIM} missing — run with --fetch")
    df = pd.read_feather(SLIM)
    df = df[df["pais"] == PAIS].copy()

    # q3c is the question; q3cn is the same question under a second variable name in the
    # waves that carry it. They never both answer for one respondent, and this asserts it.
    both = (valid(df["q3c"]) & valid(df["q3cn"])).sum()
    if both:
        raise SystemExit(f"{both} Guatemalan respondents answer both q3c and q3cn — the two "
                         "instruments are not disjoint and pooling them would double count")
    df["rel"] = df["q3c"].where(valid(df["q3c"]), df["q3cn"].where(valid(df["q3cn"])))

    df = df[valid(df["rel"]) & valid(df["prov"])].copy()
    df["code"] = df["rel"].astype(float).astype(int)
    df["prov_code"] = df["prov"].astype(float).astype(int)
    df["w"] = df["weight1500"].fillna(1.0)

    unknown = sorted(set(df["code"]) - set(CATEGORY))
    if unknown:
        raise SystemExit(f"religion codes with no label: {unknown} — the card has changed")

    # LAPOP's prov for Guatemala is 200 + the official department number. gt_geo.py asserts
    # the other half of this identity against COD.
    df["geo_id"] = (df["prov_code"] - 200).astype(str).str.zfill(2)
    if not set(df["prov_code"]).issubset(set(range(201, 223))):
        raise SystemExit(f"prov codes outside 201..222: "
                         f"{sorted(set(df['prov_code']) - set(range(201, 223)))}")

    waves = sorted(df["wave"].unique())
    if waves != WAVES:
        raise SystemExit(f"waves changed: {waves}, expected {WAVES}")
    return df


def held_out_checks(df, pop):
    """Two tests of the department decode that use neither religion nor the pcode.

    §11ad validated this instrument's provincial cut against censuses in Mexico, Peru and
    Suriname. That is evidence about LAPOP, not about THIS join, so these are the local
    version: if `prov - 200` were not the official department number, both would collapse.
    """
    print("\n  held-out checks (nothing here touches the religion column):")

    # 1. The weighted department distribution against COD-PS's population distribution.
    share_lapop = df.groupby("geo_id")["w"].sum() / df["w"].sum()
    share_pop = pop["pop"] / pop["pop"].sum()
    j = pd.concat([share_lapop.rename("lapop"), share_pop.rename("pop")], axis=1).dropna()
    r = np.corrcoef(j["lapop"], j["pop"])[0, 1]
    print(f"    department population share, LAPOP vs COD-PS:  r = {r:+.3f} over {len(j)}")
    worst = (j["lapop"] / j["pop"]).sort_values()
    print(f"      thinnest sampled {worst.index[0]} at {worst.iloc[0]:.2f}x its population "
          f"share, fullest {worst.index[-1]} at {worst.iloc[-1]:.2f}x")
    if r < 0.85:
        raise SystemExit("LAPOP's weighted department distribution does not track "
                         "Guatemala's population. The prov decode is probably permuted.")

    # 2. Mean adult age, LAPOP against COD-PS's own age bands. Guatemala's departments differ
    #    a lot on this — the western highlands are much younger than the capital — so a
    #    permutation shows up here too, and it is independent of test 1's magnitudes.
    mids = {"T_15_19": 18.5, "T_20_24": 22, "T_25_29": 27, "T_30_34": 32, "T_35_39": 37,
            "T_40_44": 42, "T_45_49": 47, "T_50_54": 52, "T_55_59": 57, "T_60_64": 62,
            "T_65_69": 67, "T_70Plus": 75}
    # the 15-19 band is ~40% adult; the rest are wholly adult
    wts = {c: (0.4 if c == "T_15_19" else 1.0) for c in mids}
    num = sum(pop[c] * wts[c] * mids[c] for c in mids)
    den = sum(pop[c] * wts[c] for c in mids)
    age_pop = (num / den).rename("pop")
    age_lapop = df.assign(age=pd.to_numeric(df["q2"], errors="coerce")).dropna(subset=["age"])
    age_lapop = (age_lapop.groupby("geo_id")
                 .apply(lambda g: np.average(g["age"], weights=g["w"]), include_groups=False)
                 .rename("lapop"))
    a = pd.concat([age_lapop, age_pop], axis=1).dropna()
    r2 = np.corrcoef(a["lapop"], a["pop"])[0, 1]
    print(f"    mean adult age, LAPOP vs COD-PS bands:         r = {r2:+.3f} over {len(a)}")
    print(f"      LAPOP spans {a['lapop'].min():.1f}-{a['lapop'].max():.1f} years, "
          f"COD-PS {a['pop'].min():.1f}-{a['pop'].max():.1f}")
    if r2 < 0.4:
        raise SystemExit("LAPOP's department age structure does not track COD-PS's. Two "
                         "independent decodes of `prov` now disagree with the pcode. STOP.")


def stability(df, nat, n_units):
    """WHICH CATEGORIES CARRY THEIR OWN DEPARTMENT GEOGRAPHY — the split-half decides it.

    §14.16's test, and the reason China's Buddhism is drawn plainly while its Protestantism
    is drawn with a warning. Each category is ranked across the 22 departments in the
    2010-2014 waves and again in 2016-2023, and the two orderings are compared. A category
    whose ranking does not replicate across halves has not demonstrated that it HAS a
    geography, whatever its pooled spread looks like.

    **THIS REPLACED A SIZE THRESHOLD, AND THE TWO DISAGREE.** The first version of this file
    cut at 4% of the country, on the reasoning that §11ad measured this instrument failing
    below about 1% of a province. That cut would have drawn `Protestante Tradicional`
    (5.4% national) on its own department shares, and its split-half correlation is **-0.04**:
    twenty-two departments, four hundred and eighty respondents, and an ordering that does not
    survive being asked twice. Size is eligibility; stability is evidence, and only the second
    one is about whether a map should be drawn.

    The bar is not a taste. A Spearman correlation over `n` units has a standard error of
    about 1/sqrt(n-1), so the bar is what it takes to be distinguishable from zero at 95%.
    On Guatemala's 22 departments that is +0.43.

    A low value is a FAILURE TO DEMONSTRATE SIGNAL rather than a demonstration of noise
    (§14.16, in those words). What it earns a category is the national spread and a sentence
    in `note_public`, not deletion: the people are still drawn, and only the claim that we
    know where they are is withdrawn.
    """
    bar = 1.96 / np.sqrt(n_units - 1)
    early, late = df[df["wave"] <= 2014], df[df["wave"] >= 2016]
    print(f"\n  split-half stability across waves (§14.16), bar = +{bar:.2f} at 95% on "
          f"{n_units} units:")
    print(f"    2010-2014 n={len(early):,}   2016-2023 n={len(late):,}")
    print(f"    {'category':<44}{'national':>9}{'spearman':>10}{'pearson':>9}  verdict")

    carries = []
    for c in sorted(nat.index, key=lambda k: -nat[k]):
        if nat[c] < ELIGIBLE_FLOOR:
            print(f"    {CATEGORY[c][:42]:<44}{nat[c] * 100:8.2f}%{'':>10}{'':>9}  "
                  f"too small to place (§11ad)")
            continue

        def share(d):
            return d.groupby("geo_id").apply(
                lambda x: x.loc[x["code"] == c, "w"].sum() / x["w"].sum(),
                include_groups=False)

        j = pd.concat([share(early).rename("e"), share(late).rename("l")], axis=1).dropna()
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sp = j["e"].corr(j["l"], method="spearman")
            pe = j["e"].corr(j["l"])
        # A category that is zero in every department of one half has no defined correlation,
        # which is the strongest possible failure of the test rather than a missing value.
        ok = bool(np.isfinite(sp)) and sp >= bar
        if ok:
            carries.append(c)
        print(f"    {CATEGORY[c][:42]:<44}{nat[c] * 100:8.2f}%{sp:+10.2f}{pe:+9.2f}  "
              f"{'own geography' if ok else 'NOT distinguishable from zero'}")
    return carries


def main():
    if "--fetch" in sys.argv:
        fetch()

    df = load()
    print(f"Guatemala: {len(df):,} respondents with a religion answer and a department, "
          f"waves {WAVES[0]}-{WAVES[-1]}")

    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    pop = pd.read_csv(POP, encoding="utf-8-sig")
    pop["geo_id"] = pop["ADM1_PCODE"].astype(str).str.replace("^GT", "", regex=True).str.zfill(2)
    pop = pop.set_index("geo_id").rename(columns={"T_TL": "pop"})
    missing = sorted(set(lut["geo_id"]) - set(pop.index))
    if missing:
        raise SystemExit(f"no COD-PS population for {missing}")
    if sorted(df["geo_id"].unique()) != sorted(lut["geo_id"]):
        raise SystemExit("LAPOP's departments and the lookup's do not agree: "
                         f"{sorted(set(lut['geo_id']) ^ set(df['geo_id'].unique()))}")

    held_out_checks(df, pop)

    # ---- national shares, and the split-half that decides who carries a geography ----
    nat = df.groupby("code")["w"].sum() / df["w"].sum()
    large = stability(df, nat, len(lut))
    small = [c for c in nat.index if c not in large]
    if sorted(large) != sorted(CARRIES):
        raise SystemExit(
            f"the split-half now selects {sorted(large)}, not {sorted(CARRIES)}. That is a "
            "change in what this country claims to know, not a bug — read the numbers above, "
            "then update CARRIES and the docstring deliberately.")
    print(f"    -> {len(large)} categories drawn on their own department shares, "
          f"{len(small)} spread at the national rate")

    small_total = float(sum(nat[c] for c in small))

    # ---- the construction, and it passes the two measured shares through untouched ----
    #
    # A department's Catholic and Evangelical shares are what the survey measured and what
    # the split-half says it may claim, so they are applied as they stand. What is left of
    # that department is then divided among the other nine at their NATIONAL relative
    # proportions. The tail's geography is therefore the residual of two stable measurements
    # rather than a flat national rate: a department where the two big answers take 93% has
    # a smaller tail than one where they take 88%, which is a real difference this file
    # measured and should not throw away.
    #
    # An earlier version renormalised the big two to leave a FIXED national tail in every
    # department. That forced the Catholic-plus-Evangelical total to be identical everywhere,
    # which is a claim the data contradicts, so it is gone.
    by_dep = df.groupby(["geo_id", "code"])["w"].sum().unstack(fill_value=0.0)
    for c in CATEGORY:
        if c not in by_dep.columns:
            by_dep[c] = 0.0
    dep_share = by_dep.div(by_dep.sum(axis=1), axis=0)

    residual = 1.0 - dep_share[large].sum(axis=1)
    if (residual <= 0).any():
        raise SystemExit(f"departments with no room for the tail: "
                         f"{sorted(residual[residual <= 0].index)}")
    print(f"    the tail is {residual.min():.1%} of {residual.idxmin()} and "
          f"{residual.max():.1%} of {residual.idxmax()}, against {small_total:.1%} nationally")

    rows = []
    for geo_id in sorted(lut["geo_id"]):
        p = int(pop.loc[geo_id, "pop"])
        for c in large:
            rows.append((geo_id, CATEGORY[c], dep_share.loc[geo_id, c] * p,
                         "department share"))
        for c in small:
            rows.append((geo_id, CATEGORY[c],
                         residual[geo_id] * (nat[c] / small_total) * p,
                         "national share within the department's residual"))

    out = pd.DataFrame(rows, columns=["geo_id", "source_category", "count", "basis_note"])
    # Integers, with the fractional remainder carried along the department order so the
    # national total is exact rather than approximately right (spec §4.1a's habit).
    out["count"] = out["count"].round().astype("int64")
    drift = int(pop["pop"].sum()) - int(out["count"].sum())
    if abs(drift) > len(out):
        raise SystemExit(f"rounding drift {drift} is larger than one person per row")
    if drift:
        # put it on the largest cell of the largest department, which is Catholic Guatemala
        idx = out["count"].idxmax()
        out.loc[idx, "count"] += drift
    print(f"  rounding drift {drift:+d} people, absorbed into the largest cell")

    out["geo_level"] = "departamento"
    out["geo_name"] = out["geo_id"].map(dict(zip(lut["geo_id"], lut["name"])))
    out["basis"] = "self_id"
    out["year"] = "2010-2023"
    out["source_id"] = SOURCE_ID
    out["n_dept"] = out["geo_id"].map(df.groupby("geo_id").size())
    out["note"] = out.apply(
        lambda r: (f"LAPOP AmericasBarometer waves 2010-2023 pooled, n={r.n_dept} in this "
                   f"department; {r.basis_note} applied to the COD-PS 2024 population"),
        axis=1)

    total = int(out["count"].sum())
    if total != int(pop["pop"].sum()):
        raise SystemExit(f"drawn {total:,} against COD-PS {int(pop['pop'].sum()):,}")

    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "note"]
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, "
          f"{out['source_category'].nunique()} categories, {out['geo_id'].nunique()} units)")

    # ---- what the country looks like, and the weakest cell named ----
    print("\n  national, as drawn:")
    nat_drawn = (out.groupby("source_category")["count"].sum() / total).sort_values(ascending=False)
    for cat, sh in nat_drawn.items():
        print(f"    {sh * 100:6.2f}%  {cat}")

    print(f"\n  the {len(large)} categories drawn on their own department shares, with the "
          "sample behind each:")
    n_by = df.groupby("geo_id").size()
    show = (out.pivot_table(index="geo_id", columns="source_category", values="count",
                            aggfunc="sum"))
    show = show.div(show.sum(axis=1), axis=0)
    cath = CATEGORY[1]
    ev = CATEGORY[5]
    order = show[ev].sort_values(ascending=False).index
    print(f"    {'dept':<16}{'n':>6}{'Catholic':>10}{'Evang':>8}{'  95% CI on Catholic'}")
    for geo_id in order:
        n = int(n_by[geo_id])
        p = show.loc[geo_id, cath]
        ci = 1.96 * np.sqrt(max(p * (1 - p), 1e-9) / n)
        name = lut.set_index("geo_id").loc[geo_id, "name"]
        print(f"    {name:<16}{n:>6}{p * 100:9.1f}%{show.loc[geo_id, ev] * 100:7.1f}%"
              f"   +/-{ci * 100:4.1f}pp")
    thin = n_by.idxmin()
    print(f"\n    the thinnest department is {lut.set_index('geo_id').loc[thin, 'name']} at "
          f"n={int(n_by[thin])}, which countries.py names in note_public")


if __name__ == "__main__":
    main()
