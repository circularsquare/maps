"""Ecuador — religion by provincia, from the LAPOP AmericasBarometer.

Reads data/raw/lapop/Grand_Merge_2004-2023_*.dta and writes data/normalized/ec.csv.
`sources/lapop.py` holds the construction and the split-half; `sources/ec.md` is this
country's record; `sources.md` §11ad assesses the source across the nine countries it could
serve, and `sources/gt.md` is the first one built on it.

**ECUADOR'S CENSUS DOES NOT ASK ABOUT RELIGION.** sources.md §11x closed the country; this
build corroborates that from INEC's own metadata rather than by reputation, the way §11ac
closed Venezuela and Colombia. The 2022 census person file `CPV_Población_2022_Nacional_Sector
Editado` publishes **88 variables** in INEC's ANDA catalogue and not one is religion — the
list runs from `P01` household relationship through `P11R` self-identification by culture and
customs, `P12` indigenous nationality, `P1001I` indigenous language, to `P35` fertility.
Ecuador asks what people are, what they speak and whom they descend from, and never what they
believe. So this is a survey standing where a census would be, and every row is `modelled`.

**THE POPULATION IS NOT COD-PS, AND THAT IS THE FIRST TIME IN THIS SET.** Guatemala and El
Salvador are drawn on OCHA projections because neither has counted recently. Ecuador counted
16,938,986 people in November 2022 and INEC publishes them by province. COD-PS's 2020
projection says 17,510,643 — **3.4% high, and unevenly**: Loja -6.9%, Galápagos -13.5%,
Pichincha -4.3%, against Manabí +2.0%. `sources/ec_geo.py` has the argument and the third
thing the census fixes, which is COD-PS's 25th row holding 41,907 people in a *zona no
delimitada* that COD-AB has no polygon for.

## THE THINNEST-SAMPLED COUNTRY IN THE SET, AND FOUR WAVES RATHER THAN SIX

7,387 respondents over 23 provinces is **321 apiece**, against El Salvador's 647 and
Guatemala's 405. Ecuador is in the 2010, 2012, 2016 and 2023 rounds and not the 2014 or 2018
ones, so the split-half runs on 4,351 against 3,036.

## THREE PROVINCES ARE ASSUMED, ONE IS LEFT BLANK, AND THE LINE BETWEEN THEM IS EVIDENCE

Four of the twenty-four provinces cannot be drawn on their own measured shares, and **they do
not get the same treatment, because they are not the same case.** The line is whether anything
measured the place at all.

**Assumed at the national rate — Carchi (n=20), Pastaza (n=32), Orellana (n=55).** Each was
sampled in the 2010 wave and no other. The split-half needs a province in both halves to rank
it twice, so these three drop out of the test that licenses the geography the other twenty are
drawn on: §14.16's rule about categories, applied to units. **But one wave did measure them**,
and that is enough to anchor an assumption on. 466,909 people, 2.76% of Ecuador. Anita's call,
2026-09-08: *"i feel like carchi is fine to assume and we can just do it."*

**NOT DRAWN AT ALL — Galápagos.** LAPOP has **no code 920**. Not unsampled: not offered. There
is no such value in the `prov_es` label set, so no Ecuadorian respondent could ever have been
placed there, and there is no reading of any kind to anchor an assumption on. Its **28,583
people, 0.17% of Ecuador**, are in `countries.py`'s `gap=` and are not drawn as a §3.5
undercount. Anita, 2026-09-08: *"galapagos maybe we just leave empty for now. no data."* It
keeps its polygon and its Kontur hexes; only the religion is absent. **It is also the reason
the distinction is worth making**: Galápagos is globally famous, so a confident-looking
national average there is a claim a reader will check and this map cannot support.

**Drawing the three on their own shares was the alternative and it is worse in three ways at
once.** Their sampling half-widths are ±19, ±15 and ±11 points on a share near 75%, so none
of the three is distinguishable from the national rate to begin with. They would carry a 2010
LEVEL while every other province carries a four-wave average, and 2010 is **4.16 points more
Catholic** than the pool — a bias, not noise, and all three the same direction. And **Carchi's
twenty interviews contain three of the eleven answers**: eighteen Catholics, one `Ninguna`,
one agnostic, and nothing else at all. Drawn on its own shares it would be a province of
172,828 people with **zero Evangelicals in a country that is 11% Evangelical**, and hard zeros
on eight of the eleven categories. A hard zero is a strong claim and twenty interviews cannot
make it.

**A neighbour-average fallback was tested instead of the national rate and it LOST.**
Leave-one-out over the twenty measured provinces, predicting each from its bordering provinces
rather than from the country: Catholic 8.08 pp mean error against 7.60, Evangelical 6.14
against 5.29, `Ninguna` 3.39 against 3.41. Ecuador's religion does not vary smoothly across
space — it jumps at province lines, because sierra, coast and Amazon interleave, and
Chimborazo is 85.9% Catholic beside neighbours averaging 67%. Galápagos has no land neighbours
at all, so it could not have been reached that way in any case.

Guatemala drew Zacapa on n=40 and said so in `note_public`, which is the opposite call. The
difference is that Zacapa is in every wave and therefore inside the split-half; these three
are not, and that is the line this file draws.

## WHAT THE COUNTRY LOOKS LIKE

    75.54%  Católico
    10.95%  Evangélica y Pentecostal
     5.97%  Ninguna (cree en un Ser Superior pero no pertenece a ninguna religión)
     2.60%  Protestante / Protestante Tradicional
     2.05%  Otro
     1.43%  Testigos de Jehová
     0.67%  Agnóstico o ateo
     0.36%  Mormones
     0.35%  Religiones Orientales no Cristianas
     0.06%  Religiones Tradicionales
     0.02%  Judío

**The most Catholic country the AmericasBarometer has drawn here**, by thirty points over
either Central American one. And `taxonomy/ec2023.py` carries the warning that goes with two
of those rows: LAPOP's answer card CHANGED after 2016, `Otro` did not exist before it, and
the Witnesses, Mormons and Jews lost their boxes after it.

## THREE CATEGORIES CARRY THEIR OWN GEOGRAPHY

The split-half on a bar of **+0.45** (20 provinces, against El Salvador's +0.54 on 14 and
Guatemala's +0.43 on 22):

    Católico                       75.54%   +0.71   own geography
    Evangélica y Pentecostal       10.95%   +0.48   own geography — clears by 0.03
    Ninguna (creyente)              5.97%   +0.63   own geography
    ------------------------------------------------------------------
    Protestante Tradicional         2.60%   +0.16   national rate
    Otro                            2.05%   undef.  national rate — SEE BELOW
    Testigos de Jehová              1.43%   +0.34   national rate
    the five under 1%                               national rate (§11ad)

**`Otro`'s "undefined" is not a failure to demonstrate geography and the printed verdict is
misleading.** `lapop.stability` calls an all-zero half "the strongest possible failure of the
test", and for every other category that is right. Here the early half is zero because **the
answer did not exist in 2010 or 2012** — code 77 is exactly zero in those waves in all 28
countries. The verdict is the same and the reason is not; it goes at the national rate
because a category whose definition moved mid-pool has no stable geography to claim.

Usage:
    python sources/ec.py --fetch    rebuild the slim extract from the 1.1 GB LAPOP .dta
    python sources/ec.py            rebuild data/normalized/ec.csv from the slim extract
"""

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
POP = os.path.join(ROOT, "data", "geo", "ec", "ec_pop_2022.csv")
LOOKUP = os.path.join(ROOT, "data", "geo", "ec", "ec_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "ec.csv")

PAIS = 9                       # LAPOP's country code for Ecuador
WAVES = [2010, 2012, 2016, 2023]
SOURCE_ID = "ec_lapop_2010_2023"
N_PROVINCES = 24               # what the census and COD-AB both have
N_SAMPLED = 23                 # what LAPOP offers: everything but Galápagos
CENSUS_TOTAL = 16_938_986

# What the split-half returns, asserted so a change in the data is a failure here rather
# than a silent re-drawing of the country.
#   1 = Católico (+0.71), 5 = Evangélica y Pentecostal (+0.48), 4 = Ninguna, creyente (+0.63)
CARRIES = [1, 4, 5]

# The provinces the split-half cannot rank, drawn at the national rate. Each was sampled
# once, in 2010, so there IS a reading to anchor the assumption on — it is the second reading
# that is missing. Asserted rather than derived silently, because the set changing means
# LAPOP's sample design changed and somebody should look.
NATIONAL_RATE = {
    "EC04": "Carchi: sampled in the 2010 wave only, so the split-half cannot rank it twice",
    "EC16": "Pastaza: sampled in the 2010 wave only, so the split-half cannot rank it twice",
    "EC22": "Orellana: sampled in the 2010 wave only, so the split-half cannot rank it twice",
}

# The province that is NOT DRAWN AT ALL. Anita, 2026-09-08: *"galapagos maybe we just leave
# empty for now. no data."* LAPOP has no code 920 — Galápagos was never offered as an answer,
# so there is no Ecuadorian respondent anywhere who could have been placed there and nothing
# to anchor an assumption on. The other three are assumed because one wave measured them; this
# one is left blank because nothing did. Its people are in `gap=` and not drawn as an
# undercount (§3.5). It keeps its polygon and its hexes; only the religion is absent.
NOT_DRAWN = {"EC20": "Galápagos"}


def main():
    if "--fetch" in sys.argv:
        lapop.fetch()

    df = lapop.load(PAIS, WAVES)

    # THE DECODE. ec_geo.py wrote `lapop_prov` beside each pcode after joining on the NAME
    # and proving the code join agrees on all 23 — Guatemala's case, not El Salvador's. This
    # reads that table rather than doing arithmetic on `prov`, so that if the two keys ever
    # stop agreeing the failure is in ec_geo.py where it can be seen.
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str, "unit": str})
    if len(lut) != N_PROVINCES:
        raise SystemExit(f"{len(lut)} provinces in the lookup, expected {N_PROVINCES}")
    lut["lapop_prov"] = pd.to_numeric(lut["lapop_prov"], errors="coerce")
    sampled = lut.dropna(subset=["lapop_prov"])
    if len(sampled) != N_SAMPLED:
        raise SystemExit(f"{len(sampled)} provinces carry a LAPOP prov code, expected "
                         f"{N_SAMPLED} — re-run sources/ec_geo.py")
    prov_to_unit = dict(zip(sampled["lapop_prov"].astype(int), sampled["unit"]))
    bad = sorted(set(df["prov_code"]) - set(prov_to_unit))
    if bad:
        raise SystemExit(f"prov codes with no province: {bad} — re-run sources/ec_geo.py")
    df["geo_id"] = df["prov_code"].map(prov_to_unit)

    names = dict(zip(lut["geo_id"], lut["name"]))
    print(f"Ecuador: {len(df):,} respondents with a religion answer and a province, "
          f"waves {WAVES[0]}-{WAVES[-1]}, {df['geo_id'].nunique()} of {N_PROVINCES} provinces")

    pop = pd.read_csv(POP, encoding="utf-8-sig")
    pop["geo_id"] = pop["geo_id"].astype(str).str.strip()
    pop = pop.set_index("geo_id")
    if sorted(pop.index) != sorted(lut["geo_id"]):
        raise SystemExit("the census population and the lookup cover different provinces")
    if int(pop["pop"].sum()) != CENSUS_TOTAL:
        raise SystemExit(f"the census file sums to {int(pop['pop'].sum()):,}, not "
                         f"{CENSUS_TOTAL:,} — re-run sources/ec_geo.py")
    if sorted(df["geo_id"].unique()) != sorted(sampled["geo_id"]):
        raise SystemExit("LAPOP's provinces and the lookup's do not agree: "
                         f"{sorted(set(sampled['geo_id']) ^ set(df['geo_id'].unique()))}")

    # The decode is tested on every province LAPOP offers, which is the strongest form of
    # the check — 23 points, not the 20 that go on to carry their own shares.
    lapop.held_out(df, pop.loc[sorted(sampled["geo_id"])], "Ecuador",
                   pop_source="INEC 2022")

    # ---- which provinces may carry a geography at all ----
    waves = sorted(df["wave"].unique())
    cut = waves[len(waves) // 2]
    in_both = sorted(set(df.loc[df["wave"] < cut, "geo_id"])
                     & set(df.loc[df["wave"] >= cut, "geo_id"]))
    at_national = sorted(set(lut["geo_id"]) - set(in_both) - set(NOT_DRAWN))
    if set(at_national) != set(NATIONAL_RATE):
        raise SystemExit(
            f"the provinces outside the split-half are now {at_national}, not "
            f"{sorted(NATIONAL_RATE)}. LAPOP's sample design has changed — read the wave "
            "table before touching NATIONAL_RATE, because which provinces this country "
            "claims to know is what just moved.")
    # Galápagos must be absent from the survey entirely, not merely thin. If LAPOP ever
    # starts offering code 920 this stops, because then there IS something to draw.
    seen = sorted(set(df["geo_id"]) & set(NOT_DRAWN))
    if seen:
        raise SystemExit(f"{seen} now has LAPOP respondents but is in NOT_DRAWN — the "
                         "province card has changed and it should be drawn")
    print(f"\n  {len(in_both)} provinces appear in BOTH halves and may carry their own "
          f"shares; {len(at_national)} are assumed at the national rate:")
    for gid in at_national:
        n = int((df["geo_id"] == gid).sum())
        print(f"    {gid}  n={n:<4} {NATIONAL_RATE[gid]}")
    share_nat = pop.loc[at_national, "pop"].sum() / pop["pop"].sum()
    print(f"    together {int(pop.loc[at_national, 'pop'].sum()):,} people, "
          f"{share_nat:.2%} of Ecuador")
    gap = int(pop.loc[sorted(NOT_DRAWN), "pop"].sum())
    print(f"  and {len(NOT_DRAWN)} is NOT DRAWN: "
          f"{', '.join(f'{k} {v}' for k, v in NOT_DRAWN.items())}, {gap:,} people "
          f"({gap / pop['pop'].sum():.2%}), left blank because nothing measured it")

    # The national level uses EVERY respondent, including the three provinces that do not
    # carry a geography — they are part of the national probability sample and dropping
    # them would bias the level, not just the placement.
    nat = lapop.national(df)
    large = lapop.stability(df, nat, len(in_both))
    small = [c for c in nat.index if c not in large]
    if sorted(large) != sorted(CARRIES):
        raise SystemExit(
            f"the split-half now selects {sorted(large)}, not {sorted(CARRIES)}. That is a "
            "change in what this country claims to know, not a bug — read the numbers above, "
            "then update CARRIES and the docstring deliberately.")
    print(f"    -> {len(large)} categories drawn on their own province shares, "
          f"{len(small)} spread at the national rate")

    # ---- the 20 provinces the survey may place, on their own shares ----
    out = lapop.build(df[df["geo_id"].isin(in_both)], nat, large, small,
                      pop["pop"], in_both, unit_noun="province")

    # ---- and the three it may only assume, at the national rate on their own population ----
    rows = []
    for gid in at_national:
        p = int(pop.loc[gid, "pop"])
        for c in nat.index:
            rows.append((gid, lapop.CATEGORY[c], nat[c] * p,
                         "national share; " + NATIONAL_RATE[gid].split(": ", 1)[1]))
    rest = pd.DataFrame(rows, columns=["geo_id", "source_category", "count", "basis_note"])
    rest["count"] = rest["count"].round().astype("int64")
    for gid in at_national:
        m = rest["geo_id"] == gid
        drift = int(pop.loc[gid, "pop"]) - int(rest.loc[m, "count"].sum())
        if abs(drift) > int(m.sum()):
            raise SystemExit(f"{gid}: rounding drift {drift} exceeds one person per row")
        rest.loc[rest.loc[m, "count"].idxmax(), "count"] += drift
    out = pd.concat([out, rest], ignore_index=True)

    out["geo_level"] = "provincia"
    out["geo_name"] = out["geo_id"].map(names)
    out["basis"] = "self_id"
    out["year"] = "2010-2023"
    out["source_id"] = SOURCE_ID
    n_by = df.groupby("geo_id").size()
    out["n_prov"] = out["geo_id"].map(n_by).fillna(0).astype(int)
    out["note"] = out.apply(
        lambda r: (f"LAPOP AmericasBarometer waves 2010-2023 pooled, n={r.n_prov} in this "
                   f"province; {r.basis_note} applied to the INEC 2022 census population"),
        axis=1)

    total = int(out["count"].sum())
    drawn_pop = int(pop["pop"].sum()) - gap
    if total != drawn_pop:
        raise SystemExit(f"drawn {total:,} against the {drawn_pop:,} the census counts "
                         f"outside {sorted(NOT_DRAWN)}")
    if out["geo_id"].nunique() != N_PROVINCES - len(NOT_DRAWN):
        raise SystemExit(f"{out['geo_id'].nunique()} provinces drawn, expected "
                         f"{N_PROVINCES - len(NOT_DRAWN)}")
    if set(out["geo_id"]) & set(NOT_DRAWN):
        raise SystemExit(f"{sorted(set(out['geo_id']) & set(NOT_DRAWN))} is in the output "
                         "and must not be")

    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "note"]
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, "
          f"{out['source_category'].nunique()} categories, {out['geo_id'].nunique()} units)")
    print(f"  {gap:,} people in {', '.join(NOT_DRAWN.values())} are NOT in this file and "
          f"belong in countries.py's `gap=` ({gap / int(pop['pop'].sum()):.2%} of Ecuador)")

    print("\n  national, as drawn:")
    drawn = (out.groupby("source_category")["count"].sum() / total).sort_values(ascending=False)
    for cat, sh in drawn.items():
        print(f"    {sh * 100:6.2f}%  {cat}")

    print(f"\n  the {len(large)} categories drawn on their own province shares, with the "
          "sample behind each:")
    show = out[out["geo_id"].isin(in_both)].pivot_table(
        index="geo_id", columns="source_category", values="count", aggfunc="sum")
    show = show.div(show.sum(axis=1), axis=0)
    cols_show = [lapop.CATEGORY[c] for c in large]
    head = "".join(f"{c.split(',')[0].split('(')[0][:13]:>14}" for c in cols_show)
    print(f"    {'province':<32}{'n':>6}{head}")
    for geo_id in show[cols_show[0]].sort_values().index:
        vals = "".join(f"{show.loc[geo_id, c] * 100:13.1f}%" for c in cols_show)
        print(f"    {names[geo_id]:<32}{int(n_by[geo_id]):>6}{vals}")
    thin = n_by[in_both].idxmin()
    hw = 1.96 * np.sqrt(0.75 * 0.25 / int(n_by[thin]))
    print(f"\n    of these the thinnest is {names[thin]} at n={int(n_by[thin])} "
          f"(±{hw * 100:.1f} points on a share near 75%), the fullest "
          f"{names[n_by[in_both].idxmax()]} at n={int(n_by[in_both].max())}")


if __name__ == "__main__":
    main()
