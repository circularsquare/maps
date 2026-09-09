"""Dominican Republic — religion by province, from ONE's own ENHOGAR-MICS6 2019.

Reads data/raw/do/mics6_2019_{hogares,miembros}.csv and writes data/normalized/do.csv.
`sources/do.md` is this country's record and `sources.md` §9cf is the write-up.

## THE QUEUE PRICED THIS COUNTRY FROM LAPOP AND REFUSED TO BUILD IT. ONE ASKED THE QUESTION

`queue.md` §11ad offered the Dominican Republic as **8,904 AmericasBarometer respondents
pooled over six waves, 29 of 32 provinces**, and §9bn refused it: three provinces were never
sampled and seven more appear in one wave-half only, several on n under 20, so a third of the
country would have been drawn at the national rate. Anita, 2026-09-08: *"probably cant do DR
without additional data."*

**The additional data is ONE's own.** The *Encuesta Nacional de Hogares de Propósitos
Múltiples* is the Dominican household survey series, run since 2005, and the 2019 round was
fielded as **MICS6** with UNICEF. MICS6's household questionnaire carries `HC1A`, *"¿A cuál
religión pertenece el jefe o la jefa del hogar?"*, and ONE published the microdata openly.

    LAPOP AmericasBarometer 2010-2023 pooled     8,904 respondents,  29 of 32 provinces
    ENHOGAR-MICS6 2019                          31,488 households,   32 of 32 provinces
                                                96,968 people in them

Eleven times the sample, every province, six answers instead of a card whose Adventist box
does not exist and whose Witness box was withdrawn after 2016 (§9bn). The thinnest province
is **San José de Ocoa at 1,759 people**, which is a fifth of LAPOP's whole national pool, and
it is one of the three LAPOP never sampled at all.

## THE CENSUS HALF IS CLOSED, AND IT WAS CHECKED TWICE RATHER THAN ASSUMED

The Dominican Republic is absent from the UNSD oracle, which proves only that no tabulation
was forwarded (§12). Two harder checks:

  * **The 2010 census.** CELADE hosts ONE's own REDATAM instance at
    `prod.redatam.org/bindom/`, and `CPV2010`'s published dictionary
    (`cpv2010_repdom_pub.dic`, downloadable from the portal) has **no religion variable** —
    the only hits on `religi` in 121 KB of it are an institutional-dwelling type and a
    community-form question about which building is the emergency shelter.
  * **The 2022 census.** ONE publishes the X CNPV person microdata openly, and its codebook
    runs `P25_ORDEN` through `P67_ANO` with **no religion question**. `P64` is the
    self-identification by skin colour and culture, which is not the same thing and must not
    be used as if it were.

So this is a survey standing where no census has stood, and every row is `modelled`.

## IT IS THE HOUSEHOLD HEAD'S RELIGION, AND THAT IS THE REAL LIMITATION

`HC1A` is asked once per household, about its head. Every member of that household is drawn
in the head's column, which is what MICS's own tabulations do and what the map needs, and it
is a modelling step rather than a measurement: a household with a Catholic head and an
evangelical daughter draws both as Catholic. The direction of the error is not knowable from
this file. **`note_public` says so plainly**, because a reader looking at 52.8% Catholic is
entitled to know it is 52.8% of people living in Catholic-headed households.

    52.78%  Católica            1.94%  Adventista
    22.34%  Evangélica          1.38%  Otra religión
    20.57%  Ninguna religión    0.99%  Testigo de Jehová

Those are the SURVEY's own weighted shares. As drawn they move slightly, because each
province carries its 2022 census population rather than its 2019 survey weight; anything
quoted to a reader has to come off `data/normalized/do.csv`.

## THE COUNTRY IS SPLIT NORTH AND SOUTH-EAST AND IT IS NOT SUBTLE

**Catholic identification runs from 25.3% of La Romana to 83.2% of Hermanas Mirabal**, a
fifty-seven point spread over one small country. The Catholic end is the Cibao, the northern
agricultural interior, six of whose provinces are above 76%: Hermanas Mirabal, La Vega,
Sánchez Ramírez, Espaillat, Monseñor Nouel and Duarte. The other end is the eastern sugar belt
and the Haitian border in
the south-west, and the two are unlike each other. **La Romana is 49.4% evangelical**, the
highest in the country and twice the national rate, in the province of the old sugar mills
and the Anglophone Afro-Caribbean *cocolo* migration; **Pedernales and Baoruco are both 43.0%
no religion**, the highest in the country, in the two poorest and emptiest provinces on the
Haitian frontier. Those are two different ways of not being Catholic and they are 200 km
apart.

## FIVE OF THE SIX CATEGORIES CARRY THEIR OWN GEOGRAPHY, AND THE WITNESSES DO NOT

§14.16's split-half needs the country ranked twice. A single-round survey has no waves, so
the split is by **cluster parity**: ENHOGAR's 1,747 primary sampling units alternate into two
half-samples of about 48,500 people each. Splitting on the sampling unit rather than on the
person is the point, because the clustering is what a naive standard error gets wrong.

    Católica                  52.78%   +0.94
    Evangélica                22.34%   +0.94
    Ninguna religión          20.57%   +0.91
    Adventista                 1.94%   +0.55
    Otra religión              1.38%   +0.54
    -------------------------------------------- bar +0.35 on 32 provinces
    Testigo de Jehová          0.99%   -0.02   NOT distinguishable

**`Testigo de Jehová` is drawn at the national 0.99% in every province**, and there is no
`UNDER_BAR` exception in this file. Its chi-square is p=4e-27, which looks like a licence and
is not one: the chi-square is computed on unweighted persons inside 1,747 clusters and takes
no account of the design, so for a category at 1% it is measuring the household clustering as
much as the geography. The rank test respects the cluster split and says the two halves do
not agree at all. Uruguay's `Judio` exception (§9ce) rested on one province holding nine
tenths of the cell on 80,196 respondents; there is no such argument here, so the rule is
applied as written.

**A country with exactly ONE failing category cannot use `sources/lapop.py`'s residual
construction**, and this is the first here to find that out. `lapop.build()` gives each unit's
residual to the failing categories at their national relative proportions, which is right for
several and a no-op for one: with a single failing category the residual *is* its own measured
share, so the construction would draw the Witnesses exactly where the survey put them. It also
divides by zero on Hato Mayor, whose measured Witness share is 0.00%. `main()` sets the tail
to its national share directly instead and rescales the five stable shares to fill the rest,
which keeps their measured proportions to one another.

## THE CROSS-CHECK IS THE ONE URUGUAY INVENTED, AND IT COMES OUT WELL

LAPOP measured the same 29 of 32 provinces with a different instrument and a different
sponsor, and **its `prov` is 2100 plus ONE's official province number**, the same numbering
ENHOGAR's `HH7A` uses, so the comparison costs one subtraction. It is reported and never
asserted. See `cross_check()` for what comes back.

Usage:
    python sources/do.py --fetch    two CSVs and a .sav, about 46 MB
    python sources/do.py            rebuild data/normalized/do.csv
"""

import os
import sys
import urllib.request

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "do")
HOGARES = os.path.join(RAW, "mics6_2019_hogares.csv")
MIEMBROS = os.path.join(RAW, "mics6_2019_miembros.csv")
SAV = os.path.join(RAW, "mics6_2019_hogares.sav")
LOOKUP = os.path.join(ROOT, "data", "geo", "do", "do_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "do.csv")

SOURCE_ID = "do_one_enhogar_mics6_2019"
N_PROVINCES = 32

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

# www.one.gob.do answers 403 to curl AND to WebFetch behind a Cloudflare challenge, so these
# go through the Wayback Machine's byte-identical copies. The canonical URLs are the tails.
# The .sav is TRUNCATED at exactly 1,048,576 bytes by the archive's capture cap and is used
# for its label dictionary only, which SPSS keeps in the header — see check_labels().
DOWNLOADS = {
    "mics6_2019_hogares.csv":
        "https://web.archive.org/web/20211114145824id_/https://www.one.gob.do/catalogo-datos/"
        "ENHOGAR/ENHOGAR-MICS6-2019/ENHOGAR-MICS6-2019-PUB-HOGARES.csv",
    "mics6_2019_miembros.csv":
        "https://web.archive.org/web/20211114155315id_/https://www.one.gob.do/catalogo-datos/"
        "ENHOGAR/ENHOGAR-MICS6-2019/ENHOGAR-MICS6-2019-PUB-MIEMBROS-HOGARES.csv",
    "mics6_2019_hogares.sav":
        "https://web.archive.org/web/20211022030705id_/https://www.one.gob.do/catalogo-datos/"
        "ENHOGAR/ENHOGAR-MICS6-2019/ENHOGAR-MICS6-2019-PUB-HOGARES.sav",
}

# HC1A's value labels, verbatim from the .sav's own label set, which prints them in capitals.
# Spec §2.4 keeps the source's own words; these strings are what reaches `source_category`
# and what `taxonomy/do2019.py` keys on. Code 5 is not on the card.
CATEGORY = {
    1: "CATÓLICA",
    2: "EVANGÉLICA",
    3: "ADVENTISTA",
    4: "TESTIGO DE JEHOVÁ",
    6: "OTRA RELIGIÓN (Especifique)",
    7: "NINGUNA RELIGIÓN",
}

# The household-questionnaire result code for a completed interview. Every other code is a
# household with no religion answer, and none of them has members in the person file.
COMPLETED = 1

# The split-half bar, 1.96/sqrt(n-1) on 32 provinces (§14.16).
STABILITY_BAR = 1.96 / np.sqrt(N_PROVINCES - 1)

# Categories drawn on their own province shares. Asserted against `stability()` on every run,
# so a change in the data stops the build rather than quietly redrawing the country.
CARRIES = [1, 2, 7, 3, 6]

# There is no UNDER_BAR in this file. `TESTIGO DE JEHOVÁ` fails the split-half at -0.02 and
# is drawn at the national rate inside each province's residual; the module docstring says
# why its chi-square is not a licence to override.


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for name, url in DOWNLOADS.items():
        dst = os.path.join(RAW, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 100_000:
            print(f"  have {name} ({os.path.getsize(dst):,} bytes)")
            continue
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=900) as r, open(dst + ".part", "wb") as f:
            f.write(r.read())
        os.replace(dst + ".part", dst)
        print(f"  got  {name} ({os.path.getsize(dst):,} bytes)")


def check_labels():
    """Assert CATEGORY against the .sav's own label set.

    The CSV carries codes and no labels, so without this the six strings in `CATEGORY` would
    be a claim about a file rather than a reading of it. The .sav reaches us truncated at
    1,048,576 bytes — the Wayback Machine's capture cap and not a damaged file — and SPSS
    writes the whole label dictionary into the header, so `metadataonly=True` reads it in
    full while the cases stay unreachable. That is why the DATA comes from the CSV.
    """
    if not os.path.exists(SAV):
        raise SystemExit(f"{SAV} missing — run with --fetch")
    import pyreadstat

    _, meta = pyreadstat.read_sav(SAV, metadataonly=True)
    got = {int(k): v for k, v in meta.variable_value_labels.get("HC1A", {}).items()}
    if got != CATEGORY:
        raise SystemExit(f"HC1A's value labels have changed: {got} against {CATEGORY} — the "
                         "answer card is not the one this file was written against")
    q = meta.column_names_to_labels.get("HC1A", "")
    print(f"  HC1A: {q}")
    print(f"    {len(got)} answers, matching the .sav's label set exactly")


def load():
    """The person file joined to its household, with the universe asserted."""
    for p in (HOGARES, MIEMBROS):
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing — run with --fetch")

    hh = pd.read_csv(HOGARES, usecols=["HH1", "HH2", "HH7", "HH7A", "HC1A", "HH46",
                                       "hhweight"], low_memory=False)
    hl = pd.read_csv(MIEMBROS, usecols=["HH1", "HH2", "HH6", "HH7", "HL4", "religion",
                                        "hhweight"], low_memory=False)
    hh["HC1A"] = pd.to_numeric(hh["HC1A"], errors="coerce")
    print(f"Dominican Republic: {len(hh):,} households in the ENHOGAR-MICS6 2019 sample, "
          f"{len(hl):,} people in the person file")

    done = hh["HH46"] == COMPLETED
    if int(done.sum()) != int(hh["HC1A"].notna().sum()):
        raise SystemExit(f"{int(done.sum()):,} completed household interviews but "
                         f"{int(hh['HC1A'].notna().sum()):,} religion answers — the two are "
                         "supposed to be the same households")
    print(f"  {int(done.sum()):,} completed interviews, and exactly those carry HC1A")

    unknown = sorted(set(hh.loc[hh["HC1A"].notna(), "HC1A"].astype(int)) - set(CATEGORY))
    if unknown:
        raise SystemExit(f"HC1A codes with no label: {unknown} — the card has changed")

    m = hl.merge(hh[["HH1", "HH2", "HH7A", "HC1A"]], on=["HH1", "HH2"], how="left",
                 validate="many_to_one")
    if m["HH7A"].isna().any():
        raise SystemExit(f"{int(m['HH7A'].isna().sum()):,} people are in the person file "
                         "with no household row — the (HH1, HH2) key is not what it looks like")
    if m["HC1A"].isna().any():
        raise SystemExit(f"{int(m['HC1A'].isna().sum()):,} people live in a household with no "
                         "religion answer; this file assumes the person file holds only "
                         "completed interviews")

    # THE RECODED VARIABLE AS A THIRD WITNESS ON THE JOIN. The person file carries MICS's own
    # `religion`, a four-way collapse of HC1A that folds Adventists, Witnesses and Other
    # together. It is not used to build anything; it is here because a wrong (HH1, HH2) join
    # would break the block structure below while leaving every total intact.
    xtab = pd.crosstab(m["HC1A"].astype(int), m["religion"].astype(int))
    expect = {1: {1}, 2: {2}, 3: {3}, 4: {3}, 6: {3}, 7: {4}}
    for code, row in xtab.iterrows():
        got = set(row[row > 0].index)
        if got != expect[code]:
            raise SystemExit(f"HC1A={code} spreads across recoded religion {got}, expected "
                             f"{expect[code]} — the household join is wrong")
    print("  the person file's own recoded `religion` is block-diagonal against HC1A, "
          "so the (HH1, HH2) join holds")

    m["code"] = m["HC1A"].astype(int)
    m["w"] = m["hhweight"].astype(float)
    m["prov"] = m["HH7A"].astype(int)
    bad = sorted(set(m["prov"]) - set(range(1, N_PROVINCES + 1)))
    if bad:
        raise SystemExit(f"province codes outside 1-{N_PROVINCES}: {bad}")
    if m["prov"].nunique() != N_PROVINCES:
        raise SystemExit(f"{m['prov'].nunique()} provinces have respondents, expected "
                         f"{N_PROVINCES}")
    print(f"  {len(m):,} people in {int(done.sum()):,} households, all {N_PROVINCES} "
          f"provinces, weighting to {m['w'].sum():,.0f}")
    return m


def national(df):
    return df.groupby("code")["w"].sum() / df["w"].sum()


def held_out(df, pop, unit_col="geo_id", n_perm=20000, seed=0):
    """Test the province decode without touching the religion column.

    The survey's own weighted share of people per province against the 2022 census, ranked
    against 20,000 shuffles of the unit labels. 32! is 2.6e35 against 20,000 draws, so
    `sources/lapop.py`'s small-country failure mode is nowhere near.
    """
    print("\n  held-out check (nothing here touches the religion column):")
    share_survey = df.groupby(unit_col)["w"].sum() / df["w"].sum()
    share_pop = pop / pop.sum()
    j = pd.concat([share_survey.rename("enhogar"), share_pop.rename("census")],
                  axis=1).dropna()
    if len(j) != len(share_pop):
        raise SystemExit(f"{len(share_pop) - len(j)} provinces have population but no "
                         "ENHOGAR respondents")
    r = np.corrcoef(j["enhogar"], j["census"])[0, 1]
    ratio = (j["enhogar"] / j["census"]).sort_values()
    print(f"    province share of people, ENHOGAR 2019 vs X CNPV 2022:  r = {r:+.4f} "
          f"over {len(j)}")
    print(f"      thinnest {ratio.index[0]} at {ratio.iloc[0]:.3f}x its census share, "
          f"fullest {ratio.index[-1]} at {ratio.iloc[-1]:.3f}x")

    rng = np.random.default_rng(seed)
    a, b = j["enhogar"].to_numpy(), j["census"].to_numpy()
    perm = np.array([np.corrcoef(a, rng.permutation(b))[0, 1] for _ in range(n_perm)])
    beaten = int((perm >= r).sum())
    print(f"      against {n_perm:,} random pairings: best random r = {perm.max():+.3f}, "
          f"and {beaten} reach the observed one")
    if beaten:
        raise SystemExit(f"{beaten} of {n_perm} random pairings match or beat r={r:+.3f}; "
                         "the population check does not pin this join")


def stability(df, nat, unit_col="geo_id"):
    """WHICH CATEGORIES CARRY THEIR OWN GEOGRAPHY — §14.16, split on the sampling unit.

    A one-round survey has no waves, so the country is ranked twice by splitting ENHOGAR's
    1,747 clusters on the parity of their id. **The split is on the CLUSTER and not on the
    person** because the cluster is the sampling unit: two halves drawn person-by-person out
    of the same 1,747 neighbourhoods would agree with each other far better than two real
    samples of the Dominican Republic would, and the test would pass everything.

    The chi-square beside each is printed for the same reason it is in `sources/uy.py` — it
    asks whether the provinces differ at all — but here it is NOT a licence, because it is
    computed on unweighted persons and ignores the clustering. See the module docstring.
    """
    from scipy import stats

    part = df["HH1"].astype(int) % 2
    early, late = df[part == 0], df[part == 1]
    print(f"\n  split-half on cluster parity (§14.16), bar = +{STABILITY_BAR:.2f} on "
          f"{N_PROVINCES} provinces:")
    print(f"    even clusters n={len(early):,}   odd clusters n={len(late):,}   "
          f"({df['HH1'].nunique():,} clusters)")
    print(f"    {'category':<30}{'national':>10}{'rank':>7}{'shares':>8}"
          f"{'chi-sq p':>11}  verdict")

    n_unit = df.groupby(unit_col).size()
    carries = []
    for c in sorted(nat.index, key=lambda k: -nat[k]):
        def share(d):
            return d.groupby(unit_col).apply(
                lambda x: x.loc[x["code"] == c, "w"].sum() / x["w"].sum(),
                include_groups=False)

        j = pd.concat([share(early).rename("e"), share(late).rename("l")], axis=1).dropna()
        sp = j["e"].corr(j["l"], method="spearman")
        pe = j["e"].corr(j["l"])
        hit = df[df["code"] == c].groupby(unit_col).size().reindex(n_unit.index).fillna(0)
        p = stats.chi2_contingency(np.vstack([hit, n_unit - hit]))[1]
        passed = bool(np.isfinite(sp)) and sp >= STABILITY_BAR
        verdict = ("own geography" if passed else
                   "national rate inside the province's residual")
        print(f"    {CATEGORY[c]:<30}{nat[c] * 100:9.2f}%{sp:+7.2f}{pe:+8.2f}"
              f"{p:11.1e}  {verdict}")
        if passed:
            carries.append(c)
    if carries != CARRIES:
        raise SystemExit(
            f"the split-half now says {[CATEGORY[c] for c in carries]} carry their own "
            f"geography, against CARRIES={[CATEGORY[c] for c in CARRIES]}. Which categories "
            "this country claims to place is what just moved; read the numbers above, then "
            "edit CARRIES deliberately. Do not move the bar.")
    return carries


def cross_check(df, names, n_perm=20000, seed=0):
    """The AmericasBarometer as an outside witness on 29 of the same 32 provinces.

    §11ad's pool is the source this country was going to be built from and was refused. It is
    a good outside witness even so, and it is nearly free: **LAPOP's `prov` for the Dominican
    Republic is 2100 plus ONE's official province number**, which is exactly what ENHOGAR's
    `HH7A` holds, so the decode is a subtraction. The three codes it never issues — 2110,
    2116 and 2131 — are Independencia, Pedernales and San José de Ocoa, which is §9bn's
    "three never sampled" identified by name for the first time.

    Reported, never asserted. LAPOP's thinnest cells here are Dajabón at n=7 and Hato Mayor
    at n=12, so a fair amount of the scatter below is its sampling error rather than any
    disagreement about the country.
    """
    import lapop

    if not os.path.exists(lapop.SLIM):
        print("\n  cross-check against the AmericasBarometer SKIPPED — "
              f"{lapop.SLIM} is not on disk")
        return
    lp = lapop.load(21, [2010, 2012, 2014, 2016, 2018, 2023])
    lp["prov"] = lp["prov_code"] - 2100
    missing = sorted(set(range(1, N_PROVINCES + 1)) - set(lp["prov"]))
    print(f"\n  cross-check: LAPOP 2010-2023 (n={len(lp):,}) on {lp['prov'].nunique()} of "
          f"{N_PROVINCES} provinces, an independent instrument")
    print(f"    never sampled by LAPOP: "
          f"{', '.join(names[p] for p in missing)}")

    ltot = lp.groupby("prov")["w"].sum()
    etot = df.groupby("prov")["w"].sum()
    pairs = [
        ("Catholic", [1], [1]),
        ("non-Catholic Christian", [2, 3, 4], [2, 5, 12]),
        ("no religion", [7], [4, 11]),
    ]
    rng = np.random.default_rng(seed)
    for label, ec, lc in pairs:
        a = (df[df["code"].isin(ec)].groupby("prov")["w"].sum()
             .reindex(etot.index).fillna(0) / etot)
        b = (lp[lp["code"].isin(lc)].groupby("prov")["w"].sum()
             .reindex(ltot.index).fillna(0) / ltot)
        j = pd.concat([a.rename("one"), b.rename("lapop")], axis=1).dropna()
        r = np.corrcoef(j["one"], j["lapop"])[0, 1]
        x, y = j["one"].to_numpy(), j["lapop"].to_numpy()
        perm = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(n_perm)])
        print(f"    {label:<24} r = {r:+.2f}   {int((perm >= r).sum()):>5} of {n_perm:,} "
              f"random pairings reach it")
        print(f"      ENHOGAR national {a.mul(etot).sum() / etot.sum() * 100:5.1f}%   "
              f"LAPOP national {b.mul(ltot).sum() / ltot.sum() * 100:5.1f}%")


def main():
    if "--fetch" in sys.argv:
        fetch()

    check_labels()
    df = load()

    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str, "unit": str})
    if len(lut) != N_PROVINCES:
        raise SystemExit(f"{len(lut)} provinces in the lookup, expected {N_PROVINCES} — "
                         "re-run sources/do_geo.py")
    prov_to_unit = dict(zip(lut["enhogar_hh7a"].astype(int), lut["unit"]))
    names_by_prov = dict(zip(lut["enhogar_hh7a"].astype(int), lut["name"]))
    names = dict(zip(lut["geo_id"], lut["name"]))
    pop = pd.Series(dict(zip(lut["geo_id"], lut["pop_2022"].astype(int))))
    df["geo_id"] = df["prov"].map(prov_to_unit)
    if df["geo_id"].isna().any():
        raise SystemExit("a province code came out of the lookup with no pcode")

    held_out(df, pop)
    nat = national(df)
    carries = stability(df, nat)
    cross_check(df, names_by_prov)

    # ---- shares x population ----
    small = [c for c in nat.index if c not in carries]
    by_unit = df.groupby(["geo_id", "code"])["w"].sum().unstack(fill_value=0.0)
    for c in CATEGORY:
        if c not in by_unit.columns:
            raise SystemExit(f"{CATEGORY[c]} is absent from every province")
    unit_share = by_unit.div(by_unit.sum(axis=1), axis=0)

    # THE TAIL IS SET TO ITS NATIONAL SHARE, NOT TO THE PROVINCE'S RESIDUAL, and the
    # difference matters here in a way it does not in `sources/lapop.py`'s `build()`.
    # That function divides each unit's residual among the failing categories at their
    # national relative proportions, which is right when there are several of them. **With
    # exactly one it is a no-op**: the residual IS that category's own measured share, so
    # "national rate inside the residual" would draw the Witnesses exactly where the survey
    # put them, which is the thing the split-half just declined to license. It would also
    # have failed outright, because Hato Mayor's measured Witness share is 0.00% and its
    # residual is therefore zero.
    #
    # So: every province gets the national 0.99% of Witnesses, and its five stable shares
    # are held in the proportions the survey measured and scaled to fill the remaining
    # 99.01%. Nobody is deleted and only the claim to know where the Witnesses are is
    # withdrawn.
    small_total = float(sum(nat[c] for c in small))
    carried = unit_share[carries].sum(axis=1)
    if (carried <= 0).any():
        raise SystemExit(f"provinces whose stable categories sum to nothing: "
                         f"{sorted(carried[carried <= 0].index)}")
    measured_small = 1.0 - carried
    print(f"\n  the tail is set to its national {small_total:.2%} in every province; as "
          f"measured it ran {measured_small.min():.2%} in {names[measured_small.idxmin()]} "
          f"to {measured_small.max():.2%} in {names[measured_small.idxmax()]}")

    units = sorted(lut["geo_id"])
    rows = []
    for unit in units:
        p = int(pop[unit])
        for c in sorted(CATEGORY):
            if c in carries:
                share = unit_share.loc[unit, c] / carried[unit] * (1.0 - small_total)
            else:
                share = nat[c]
            rows.append((unit, CATEGORY[c], share * p))
    out = pd.DataFrame(rows, columns=["geo_id", "source_category", "count"])
    out["count"] = out["count"].round().astype("int64")

    target = int(pop.sum())
    drift = target - int(out["count"].sum())
    if abs(drift) > len(out):
        raise SystemExit(f"rounding drift {drift} is larger than one person per row")
    if drift:
        out.loc[out["count"].idxmax(), "count"] += drift
    print(f"  rounding drift {drift:+d} people, absorbed into the largest cell")

    n_by = df.groupby("geo_id").size()
    hh_by = df.groupby("geo_id")["HH1"].nunique()
    out["geo_level"] = "provincia"
    out["geo_name"] = out["geo_id"].map(names)
    out["basis"] = "self_id"
    out["year"] = "2019"
    out["source_id"] = SOURCE_ID
    small_names = {CATEGORY[c] for c in small}
    out["note"] = [
        (f"ONE ENHOGAR-MICS6 2019, religion of the household head, n={int(n_by[g]):,} people "
         f"in {int(hh_by[g]):,} clusters in this province; "
         + ("national share, this category having failed the split-half"
            if cat in small_names else "province share")
         + ", applied to the province's X CNPV 2022 census population")
        for g, cat in zip(out["geo_id"], out["source_category"])]

    total = int(out["count"].sum())
    if total != target:
        raise SystemExit(f"drawn {total:,} against a target of {target:,}")
    if out["geo_id"].nunique() != N_PROVINCES:
        raise SystemExit(f"{out['geo_id'].nunique()} provinces drawn")

    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, "
          f"{out['source_category'].nunique()} categories, {N_PROVINCES} provinces)")

    print("\n  national, as drawn:")
    drawn = (out.groupby("source_category")["count"].sum() / total).sort_values(
        ascending=False)
    for cat, sh in drawn.items():
        print(f"    {sh * 100:6.2f}%  {cat}")

    print("\n  by province, sorted by the Catholic share:")
    show = out.pivot_table(index="geo_id", columns="source_category", values="count",
                           aggfunc="sum")
    show = show.div(show.sum(axis=1), axis=0) * 100
    cath = show[CATEGORY[1]]
    print(f"    {'province':<24}{'n':>7}{'Cath':>7}{'Evang':>7}{'None':>7}"
          f"{'Advent':>8}{'Other':>7}{'JW':>6}")
    for g in cath.sort_values().index:
        print(f"    {names[g]:<24}{int(n_by[g]):>7,}{show.loc[g, CATEGORY[1]]:7.1f}"
              f"{show.loc[g, CATEGORY[2]]:7.1f}{show.loc[g, CATEGORY[7]]:7.1f}"
              f"{show.loc[g, CATEGORY[3]]:8.1f}{show.loc[g, CATEGORY[6]]:7.1f}"
              f"{show.loc[g, CATEGORY[4]]:6.1f}")
    thin = n_by.idxmin()
    hw = 1.96 * np.sqrt(0.53 * 0.47 / int(n_by[thin]))
    print(f"\n    thinnest sample {names[thin]} at n={int(n_by[thin]):,} people "
          f"(±{hw * 100:.1f} points on a share near 53%, before the design effect), "
          f"fullest {names[n_by.idxmax()]} at n={int(n_by.max()):,}")


if __name__ == "__main__":
    main()
