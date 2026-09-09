"""Panama — religion by province and comarca, from the LAPOP AmericasBarometer.

Reads data/raw/lapop/Grand_Merge_2004-2023_*.dta and writes data/normalized/pa.csv.
`sources/lapop.py` holds the construction and the split-half; `sources/pa.md` is this
country's record; `sources.md` §11ad assesses the source across the nine countries it could
serve, and `sources/gt.md` is the first one built on it.

**PANAMA'S CENSUSES DO NOT ASK ABOUT RELIGION AND ITS OFFICE HAS NEVER PUBLISHED THE ANSWER
IT HOLDS.** sources.md §11x read the person-variable dictionary of all five censuses INEC
serves on its own REDATAM — 1980, 1990, 2000, 2010 and 2023, 252 variables between them — and
not one is religion; the 2023 census asks indigenous group and Afrodescendant group and stops.
That negative was re-confirmed for this build and it is not the whole story: **INEC has asked
the question twice on household surveys and published neither answer.** The MICS 2013
household listing asks `HC1.A ¿Qué religión profesa (nombre)?` of every member of 11,100
households, with a card of ten-plus options, and the 230-page report tabulates none of it; the
Encuesta de Propósitos Múltiples of April 2022 asked 30,554 people in 16,324 households, of
which 992 in the comarcas, and its results reached the public only as a press conference on
28 September 2022. `sources/pa.md` has what was searched. So this is a survey standing where a
census would be, and every row is `modelled` in §7.

## TEN UNITS OF THIRTEEN, AND THE TWO THAT ARE MISSING ARE THE INTERESTING ONES

LAPOP's own `prov` value labels name exactly ten Panamanian units. There is no code for
**Comarca Guna Yala** or **Comarca Emberá-Wounaan** in any wave, so neither is drawn: 44,374
people, 1.09% of the country, into `gap=`. That is Ecuador's Galapagos rule applied (§9bn) —
the line is whether ANYTHING measured the place — and it costs more here than it did there,
because these are two of the three indigenous comarcas and the third, Ngäbe Buglé, is the
unit where this survey's picture is least like the rest of the country.

**Panamá Oeste is a different problem and is not a gap.** It became a province in 2014 out of
the western districts of Panamá; LAPOP never adopted it, and the 2023 round still codes all
789 of its Panamá-area respondents to `708`. So the two are drawn as one unit of 2,092,950
people, 51.5% of the country. `sources/pa_geo.py` dissolves the polygons to match.

## What the country looks like

    64.44%  Católico
    21.06%  Evangélica y Pentecostal
     6.71%  Ninguna (Cree en un Ser Superior pero no pertenece a ninguna religión)
     3.30%  Protestante, Protestante Tradicional o Protestante no Evangélico
     1.56%  Otro
     1.20%  Religiones Orientales no Cristianas
     0.60%  Testigos de Jehová
     0.49%  Agnóstico o ateo (no cree en Dios)
     0.43%  Religiones Tradicionales
     0.18%  Iglesia de los Santos de los Últimos Días (Mormones)
     0.02%  Judío

**INEC's own April 2022 survey read 65% Catholic and 22% Evangelical**, against 64.4% and
21.1% here. That is the only independent reading of the level that exists for Panama. It is a
national check and says nothing about the provinces. Its third figure, 8% professing no
religion, is against 7.2% here once the two no-religion cells are added.

**Catholic and no-religion are like-for-like; the Evangelical pair is not.** INEC's card is
Católica, Evangélica, Adventista, Testigo de Jehová, Otras, Ninguna, with **no Protestante
box**, so its `Evangélica` is the comparator for LAPOP's `Evangélica y Pentecostal` PLUS
`Protestante Tradicional`, and its `Adventista` has no LAPOP box of its own either. Compared
that way it is 24.4% here against INEC's 24%; the 21.1-against-22 pair reads better and puts
different denominators beside each other.

## THREE CATEGORIES CARRY THEIR OWN GEOGRAPHY

The split-half, on a bar of +0.65 (ten units, against El Salvador's +0.54 on fourteen and
Guatemala's +0.43 on twenty-two):

    Católico                              64.44%   +0.85   own geography
    Evangélica y Pentecostal              21.06%   +0.87   own geography
    Ninguna (creyente)                     6.71%   +0.82   own geography
    ------------------------------------------------------------------------
    Protestante Tradicional                3.30%   +0.46   national rate
    Otro                                   1.56%   undef   national rate
    Religiones Orientales no Cristianas    1.20%   +0.40   national rate
    the five under 1%                                      national rate (§11ad)

**Ten units is the fewest of any country in this set and the bar is correspondingly the
highest**, and all three still clear it comfortably. The two that fail are not close.
`Otro` returns an undefined correlation because it is zero in every unit of one wave-half,
which §14.16 treats as the strongest possible failure rather than a missing value.

Usage:
    python sources/pa.py --fetch    rebuild the slim extract from the 1.1 GB LAPOP .dta
    python sources/pa.py            rebuild data/normalized/pa.csv from the slim extract
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
POP = os.path.join(ROOT, "data", "raw", "pa", "pan_admpop_adm1_2023.csv")
LOOKUP = os.path.join(ROOT, "data", "geo", "pa", "pa_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "pa.csv")

PAIS = 7                       # LAPOP's country code for Panama
WAVES = [2010, 2012, 2014, 2023]
SOURCE_ID = "pa_lapop_2010_2023"
N_UNITS = 10                   # units LAPOP measures, of the 12 in pa_provincias.gpkg

# Panama Oeste is inside LAPOP's `708`; the polygons are already dissolved, and the COD-PS
# rows have to be added the same way or the population would be short by 653,664.
MERGE_POP = {"PA11": "PA12"}

# The two comarcas LAPOP has no code for, in any wave. They keep their polygons and their
# hexes and draw no religion. See the docstring and `gap=`.
NOT_SAMPLED = ["PA06", "PA08"]

# What the split-half returns, asserted so a change in the data is a failure here rather than
# a silent re-drawing of the country.
#   1 = Católico (+0.85), 5 = Evangélica y Pentecostal (+0.87), 4 = Ninguna, creyente (+0.82)
CARRIES = [1, 4, 5]


def _exact_p(a, b, label, chunk=200_000):
    """Exact permutation p for the correlation of `a` with `b`, over every ordering of `b`.

    Standardise both and r is a dot product, so all n! orderings are one matrix product done
    in chunks. The observed ordering is excluded BY VALUE rather than by index, which also
    excludes orderings that only swap units of identical population: those reproduce the
    observed r and no correlation can tell them from the truth, so counting them as beating it
    would be a false alarm rather than a catch (`sources/arabbarometer.py`'s rule).
    """
    n = len(a)
    xs = (a - a.mean()) / a.std()
    ys = (b - b.mean()) / b.std()
    obs = float(np.dot(xs, ys) / n)
    perms = np.array(list(itertools.permutations(range(n))), dtype=np.int8)
    hits = same = 0
    best = -2.0
    for i in range(0, len(perms), chunk):
        blk = perms[i:i + chunk]
        rr = ys[blk] @ xs / n
        identical = np.all(b[blk] == b, axis=1)
        same += int(identical.sum())
        hits += int(((rr >= obs - 1e-12) & ~identical).sum())
        best = max(best, float(rr[~identical].max()))
    total = len(perms) - same
    p = hits / total
    print(f"    {label}: r = {obs:+.4f} over {n} units; {hits:,} of the {total:,} other "
          f"orderings reach it")
    print(f"      exact p = {p:.2e}, one in {total / max(hits, 1):,.0f}; "
          f"best wrong ordering {best:+.4f}")
    return obs, p, hits, total, perms


def held_out_exact(df, pop, names, bar=1e-3, bar_dropped=1e-2):
    """Test the `prov` decode without touching the religion column — EXACTLY, not sampled.

    ## WHY THIS IS NOT `lapop.held_out`, AND WHY THE BAR IS DELIBERATELY LOOSER

    `lapop.held_out` samples 20,000 orderings and fails if ANY of them reaches the observed r.
    That rule is right where the orderings vastly outnumber the draws — its own countries have
    14, 22 and 23 units — and Panama has ten, so 3,628,800. **Run here it fails, and it fails
    for a correct decode: 3 of 20,000 reached r=+0.996 on the first run.**

    That is NOT the small-country artefact lapop.py's docstring warns about and it is not the
    one `sources/arabbarometer.py` fixes. Only ONE of those draws was the observed ordering
    coming back; excluding it by value, which is arabbarometer's fix, leaves the failure
    standing. Enumerating all 3,628,800 orderings says why:

        468 of the 3,628,799 other orderings reach r = +0.9957, and ALL 468 OF THEM
        KEEP PANAMA WITH PANAMA OESTE — 51% of the country — IN PLACE.

    **It is leverage, not unit count.** One unit is half the population and half the sample,
    so once that point is paired correctly the correlation is above +0.99 however the other
    nine are shuffled, and 468 of those 362,880 shufflings land above the observed value. The
    expected number of hits in 20,000 draws is 20,000 x 1.29e-4 = 2.6, so a correct decode
    clears "zero of 20,000" only when the sampler happens to draw none of them, which is
    (1 - 1.29e-4) ** 20000 = **7.6% of seeds**, about one in thirteen. A test a right answer
    fails twelve times in thirteen is not a test.

    So the statistic is the same and the null is the same; what changes is that the p-value is
    **computed exactly instead of being required to round to zero at 20,000 draws**. The bar
    is 1e-3, and it is a LOOSER bar than the old rule rather than a stricter one: "0 of 20,000"
    demonstrates p < 1.5e-4 at 95% confidence on a sample, and 1e-3 is about 6.7x larger. The
    loosening is deliberate and is what the paragraph above justifies — the old bar is
    unattainable for this country whatever the decode — and 1e-3 still requires the asserted
    pairing to sit inside the top tenth of a percent of all 3,628,799 wrong orderings. Panama
    returns 1.29e-4.

    **Neither bar was written down before the numbers were computed, and a reader should weigh
    them accordingly.** 1e-3 and 1e-2 are the round numbers one order of magnitude above the
    observed 1.29e-4 and 1.29e-3, this module is untracked so it has no history, and nothing on
    disk shows either of them being fixed before the run; `sources/pa.md` §6.1 is the audit that
    says so. They are argued for here rather than pre-registered.

    And because the whole objection is leverage, the same test is run again **with the
    dominant unit dropped**, on a bar of 1e-2. Nine units, r = +0.921, exact p = 1.29e-3.
    **That is a consistency check and not a second witness.** The 468 nine-unit orderings that
    beat are the ten-unit test's same 468 restricted to the nine, so p9/p10 = 10.0000 exactly;
    once every beating ordering fixes the dominant unit, which is what the first test
    established, the re-run cannot fail when the first passes. It says nothing about the other
    nine. Testing those would need a statistic the first test did not use — rank instead of
    level, equal unit weights, or a null drawn without inheriting the ten-unit pairing.

    ## THE DECODE DOES NOT REST ON THIS CHECK ANYWAY

    This test exists to catch an INFERRED decode: Guatemala's `prov - 200`, El Salvador's
    alphabetical pcodes. **Panama's is not inferred.** The Grand Merge's own `prov` value-label
    set spells the ten units out in Spanish, and `sources/pa_geo.py` joins those names to
    COD-AB's names with one alias. The population check is a second witness, not the first.

    The mean-age check `lapop.held_out` also prints is not reproduced. It was measured across
    this module's countries and has no power (F below 1 in both), and lapop.py reports it
    without asserting for that reason; a diagnostic with no power is not worth a third of this
    function.
    """
    print("\n  held-out check (nothing here touches the religion column):")
    share = df.groupby("geo_id")["w"].sum() / df["w"].sum()
    j = pd.concat([share.rename("lapop"), (pop / pop.sum()).rename("pop")], axis=1).dropna()
    if len(j) != len(pop):
        raise SystemExit(f"{len(pop) - len(j)} units have population but no LAPOP rows")
    ratio = (j["lapop"] / j["pop"]).sort_values()
    print(f"    unit population share, LAPOP vs COD-PS 2023, over {len(j)} units:")
    print(f"      thinnest sampled {names[ratio.index[0]]} at {ratio.iloc[0]:.2f}x its "
          f"population share, fullest {names[ratio.index[-1]]} at {ratio.iloc[-1]:.2f}x")

    a, b = j["lapop"].to_numpy(float), j["pop"].to_numpy(float)
    obs, p, hits, total, perms = _exact_p(a, b, "all units, exhaustive")
    big = int(np.argmax(b))
    rr = ((b[perms] - b.mean()) / b.std()) @ ((a - a.mean()) / a.std()) / len(a)
    beating = (rr >= obs - 1e-12) & ~np.all(b[perms] == b, axis=1)
    moved = int((beating & (perms[:, big] != big)).sum())
    print(f"      of the {hits:,} that reach it, {hits - moved:,} keep "
          f"{names[j.index[big]]} in place and {moved:,} move it")
    if p >= bar:
        raise SystemExit(f"exact p = {p:.2e} against a bar of {bar:.0e}: the population "
                         "check does not pin this decode.")

    k = j.drop(index=j.index[big])
    print(f"    and again with {names[j.index[big]]} dropped — a CONSISTENCY CHECK, not a "
          "second witness, because\n      the beating orderings are the same set restricted "
          "to nine units and p9 = 10 x p10 exactly:")
    _, p2, _, _, _ = _exact_p(k["lapop"].to_numpy(float), k["pop"].to_numpy(float),
                              "nine units, exhaustive")
    if p2 >= bar_dropped:
        raise SystemExit(f"with the dominant unit dropped the exact p is {p2:.2e} against a "
                         f"bar of {bar_dropped:.0e}. This cannot happen while the first test "
                         "passes and every beating ordering fixes the dominant unit, so the "
                         "data or the statistic has changed shape — read the two p-values.")


def main():
    if "--fetch" in sys.argv:
        lapop.fetch()

    df = lapop.load(PAIS, WAVES)

    # THE DECODE. pa_geo.py wrote `lapop_prov` beside each pcode after joining on the NAME —
    # LAPOP's own `prov` value labels, not a numbering — and proving that the code join
    # mispairs eight of ten, `708` Panamá onto `PA08` Kuna Yala among them. This reads that
    # table rather than doing arithmetic on `prov`, which is the whole point.
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str, "unit": str},
                      keep_default_na=False, na_values=[])
    lut_drawn = lut[lut["lapop_prov"] != ""]
    prov_to_unit = dict(zip(lut_drawn["lapop_prov"].astype(int), lut_drawn["unit"]))
    if len(prov_to_unit) != N_UNITS:
        raise SystemExit(f"{len(prov_to_unit)} prov codes in the lookup, expected {N_UNITS}")
    if sorted(set(lut["geo_id"]) - set(lut_drawn["geo_id"])) != sorted(NOT_SAMPLED):
        raise SystemExit("the units with no LAPOP code are not the two comarcas — re-run "
                         "sources/pa_geo.py")
    bad = sorted(set(df["prov_code"]) - set(prov_to_unit))
    if bad:
        raise SystemExit(f"prov codes with no province: {bad} — re-run sources/pa_geo.py")
    df["geo_id"] = df["prov_code"].map(prov_to_unit)

    print(f"Panama: {len(df):,} respondents with a religion answer and a province, "
          f"waves {WAVES[0]}-{WAVES[-1]}")

    pop = pd.read_csv(POP, encoding="utf-8-sig")
    pop["geo_id"] = pop["ADM1_PCODE"].astype(str).str.strip()
    for src, dst in MERGE_POP.items():
        num = pop.select_dtypes("number").columns
        pop.loc[pop["geo_id"] == dst, num] = (
            pop.loc[pop["geo_id"] == dst, num].to_numpy()
            + pop.loc[pop["geo_id"] == src, num].to_numpy())
        pop = pop[pop["geo_id"] != src]
    national = int(pop["T_TL"].sum())
    pop = pop.set_index("geo_id").rename(columns={"T_TL": "pop"})
    undrawn = int(pop.loc[NOT_SAMPLED, "pop"].sum())
    print(f"  COD-PS 2023: {national:,} people over {len(pop)} units; "
          f"{undrawn:,} of them ({undrawn / national:.2%}) are in Emberá and Kuna Yala, "
          "which LAPOP never sampled and this file does not draw")
    pop = pop.drop(index=NOT_SAMPLED)

    missing = sorted(set(lut_drawn["geo_id"]) - set(pop.index))
    if missing:
        raise SystemExit(f"no COD-PS population for {missing}")
    if sorted(df["geo_id"].unique()) != sorted(lut_drawn["geo_id"]):
        raise SystemExit("LAPOP's provinces and the lookup's do not agree: "
                         f"{sorted(set(lut_drawn['geo_id']) ^ set(df['geo_id'].unique()))}")

    held_out_exact(df, pop["pop"], names=dict(zip(lut["geo_id"], lut["name"])))

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

    units = sorted(lut_drawn["geo_id"])
    out = lapop.build(df, nat, large, small, pop["pop"], units, unit_noun="province")

    out["geo_level"] = "provincia o comarca"
    out["geo_name"] = out["geo_id"].map(dict(zip(lut["geo_id"], lut["name"])))
    out["basis"] = "self_id"
    out["year"] = "2010-2023"
    out["source_id"] = SOURCE_ID
    out["n_prov"] = out["geo_id"].map(df.groupby("geo_id").size())
    out["note"] = out.apply(
        lambda r: (f"LAPOP AmericasBarometer waves 2010-2023 pooled, n={r.n_prov} in this "
                   f"province; {r.basis_note} applied to the COD-PS 2023 population"),
        axis=1)

    total = int(out["count"].sum())
    if total != int(pop["pop"].sum()):
        raise SystemExit(f"drawn {total:,} against COD-PS {int(pop['pop'].sum()):,}")
    if set(out["geo_id"]) & set(NOT_SAMPLED):
        raise SystemExit("a comarca LAPOP never sampled has rows — it must not be drawn")

    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "note"]
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, "
          f"{out['source_category'].nunique()} categories, {out['geo_id'].nunique()} units)")

    print("\n  national, as drawn:")
    drawn = (out.groupby("source_category")["count"].sum() / total).sort_values(ascending=False)
    for cat, sh in drawn.items():
        print(f"    {sh * 100:6.2f}%  {cat}")

    # The pooled level is an average over four rounds that disagree, and two of the eleven
    # boxes only exist in part of the pool. Printed because `note_public`, `pa.md` and
    # `taxonomy/pa2023.py` all cite it and it is not otherwise recoverable from pa.csv.
    print("\n  the same answers by wave, weighted (%), which is what pooling averages over:")
    by_wave = df.groupby(["wave", "code"])["w"].sum().unstack(fill_value=0.0)
    by_wave = by_wave.div(by_wave.sum(axis=1), axis=0) * 100
    header = "".join(f"{int(w):>9}" for w in by_wave.index)
    print(f"    {'category':<44}{header}")
    for c in sorted(nat.index, key=lambda k: -nat[k]):
        row = "".join(f"{by_wave.loc[w, c]:9.2f}" for w in by_wave.index)
        print(f"    {lapop.CATEGORY[c][:42]:<44}{row}")

    print(f"\n  the {len(large)} categories drawn on their own province shares, with the "
          "sample behind each:")
    n_by = df.groupby("geo_id").size()
    show = out.pivot_table(index="geo_id", columns="source_category", values="count",
                           aggfunc="sum")
    show = show.div(show.sum(axis=1), axis=0)
    names = dict(zip(lut["geo_id"], lut["name"]))
    cols_show = [lapop.CATEGORY[c] for c in large]
    head = "".join(f"{c.split(',')[0][:14]:>15}" for c in cols_show)
    print(f"    {'province':<24}{'n':>6}{head}")
    for geo_id in show[cols_show[0]].sort_values().index:
        n = int(n_by[geo_id])
        vals = "".join(f"{show.loc[geo_id, c] * 100:14.1f}%" for c in cols_show)
        print(f"    {names[geo_id]:<24}{n:>6}{vals}")
    thin = n_by.idxmin()
    print(f"\n    the thinnest province is {names[thin]} at n={int(n_by[thin])}, "
          f"the fullest {names[n_by.idxmax()]} at n={int(n_by.max())}")


if __name__ == "__main__":
    main()
