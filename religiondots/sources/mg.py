"""Madagascar — religion by region, the three large churches included, from four pooled
Afrobarometer rounds on the region counts of the 2018 census.

Reads data/raw/afrobarometer/*.sav, data/geo/mg/mg_lookup.csv and mg_districts.csv; writes
data/normalized/mg.csv. `sources/mg.md` is the record, `sources/tz.py` the construction this
copies, and `sources.md` §11aq the scout's reading of the census forms.

## THE CENSUS DOES NOT ASK

Neither the 1993 (RGPH-2) nor the 2018 (RGPH-3) household form has a religion item, and INSTAT's
retired document store holds no religion table (§11aq). So this is the Nigeria construction
(ask/answered/010-ng): each unit at its own survey mix and at its census population, the national
level computed and never fitted, and every row `modelled`.

    row margin      region populations     RGPH-3 2018, Tableau 6          EXACT
    the composition each unit's own mix    Afrobarometer R5, R6, R7, R9    measured
    the national level                     neither                         computed

## THE CHURCHES ARE KEPT APART, AS IN CAMEROON

In most countries the survey's denominations are folded back into Christian, because the share answering
`Christian only` is set by the fieldwork (Liberia 23% to 72% by round, `afrobarometer.py`). In
Madagascar it is 0.3% to 2.3% of respondents in the drawn rounds, and the three large churches
hold their level round to round: Roman Catholic 35.1-40.5%, the Calvinist box (FJKM, the Church of
Jesus Christ in Madagascar) 19.7-23.8%, Lutheran 12.7-14.6%. `CONLY_SWING_MAX` asserts the first.

The playbook asks for an outside witness to the LEVEL before a denomination is drawn. Two exist,
both Demographic and Health Surveys run by INSTAT, whose respondent tables print Catholic beside
the three mainline Protestant churches together: EDSMD-V 2021 Tableau 3.1 (`FJKM/FLM/Anglikana`)
and EDSMD-IV 2008-09 Tableau 3.2 (`Protestante/FLM`). Their Catholic-to-Protestant ratio runs
0.94 to 1.01; `witness_eds()` asserts the drawn ratio sits near that range. Nothing outside the
Afrobarometer splits the FJKM from the Lutherans, and that is said wherever it is drawn.

## NONE AND TRADITIONAL ARE ONE BOUNDARY THE FIELDWORK MOVES, SO THEY ARE POOLED AS ONE BOX

The card offers `Traditional/ethnic religion`, `None`, `Atheist` and `Agnostic` separately in every
drawn round (`report_card()`), and nationally the first runs 8.5, 4.5, 1.5, 1.3% in rounds 5, 6, 7
and 9 while `None` runs 8.2, 4.0, 13.0, 12.7%. Region by region it is the same people changing box:
Melaky 45% traditional and 0% none in rounds 5-6, 0% and 12% in rounds 7 and 9; Atsimo Atsinanana
32% and 1%, then 0% and 17% (`swap_table()` prints every unit). That is `Christian only`'s trap in
another place, and the answer is the same one: pool to a level the probing cannot move. So the two
boxes (with atheist and agnostic) are tested, placed and levelled as ONE category, `None or
traditional`, and only then split into its two answers at one national ratio.

The ratio is rounds 7 and 9's, on census weights: 90.5% `None`. Both DHS surveys ask the two
separately too (`Traditionnelle/Animiste`, `Sans religion/Aucune`) and put `None` at 89.6-95.8% of
the pair; `NONE_FRACTION` is asserted inside that range. The pooled rounds' own ratio (72%) is the
one the swap corrupts and is not used.

`None` stays `unaffiliated`. That is step 2 of the "no religion" procedure (WORKFLOW_PLAN.md; Anita's
Madagascar ruling, 2026-09-14 night): the box was offered beside a traditional one. Pooling here
decides the geography, not the node. `sources/mg.md` §5 has the alternative that was built first and
dropped.

## ROUND 4 IS LEFT OUT

Round 4 (2008) is cut by the six old provinces. Its `DISTRICT` column would place its respondents
in 21 of the 22 regions, but no Betsiboka district was sampled, which leaves an empty (round, unit)
cell that `cab.stability` refuses. Tanzania's round 5 is the precedent: the round goes, not the
unit. Madagascar is not in round 8.

## ROUND 9 RE-CUTS TWO REGIONS AND RE-NUMBERS THE CODES

Round 9 (2022) names Vatovavy and Fitovinany apart and Haute Matsiatra as `Matsiatra Ambony`, and
its REGION codes no longer mean what rounds 5-7's did (433 is Haute Matsiatra in rounds 5-7 and
Ihorombe in round 9). Units are decoded by LABEL only, and Vatovavy and Fitovinany are drawn as the
census's one region. Rounds 6 and 7 carry a district in `LOCATION.LEVEL.1` and round 9 the old
province; `check_locations()` asserts both against the label decode.

Usage:
    python sources/mg.py        rebuild data/normalized/mg.csv (the .sav files are shared; fetch
                                them with `python sources/afrobarometer.py --fetch` if absent)
"""

import os
import re
import sys
import unicodedata

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import afrobarometer as ab
import cab
import stability
from tz import key, round_within_rows, standouts

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOOKUP = os.path.join(ROOT, "data", "geo", "mg", "mg_lookup.csv")
DISTRICTS = os.path.join(ROOT, "data", "geo", "mg", "mg_districts.csv")
OUT = os.path.join(ROOT, "data", "normalized", "mg.csv")

COUNTRY = "Madagascar"
ROUNDS_READ = [4, 5, 6, 7, 9]
ROUNDS = [5, 6, 7, 9]
EARLY = [5, 6]
RECENT = [7, 9]
SOURCE_ID = "mg_afrobarometer_2013_2022_rgph2018"
YEARS = "2013-2022"

N_UNITS = 22
CENSUS_2018 = 25_674_196

PAIR = "None or traditional"
NONE = "None"
TRAD = "Traditional/ethnic religion"
# What is tested, placed and levelled.
CATEGORIES = [
    "Roman Catholic", "Calvinist (FJKM)", "Lutheran", "Anglican", "Seventh Day Adventist",
    "Pentecostal", "Jehovah's Witness", "Other Christian", "Muslim", PAIR, "Other",
]
# What is written: the pair split back into the card's two answers.
OUT_CATEGORIES = [c for c in CATEGORIES if c != PAIR]
OUT_CATEGORIES[OUT_CATEGORIES.index("Other"):OUT_CATEGORIES.index("Other")] = [TRAD, NONE]

# Every answer Madagascar's respondents give in rounds 4-9 -> the category it is drawn as, keyed
# through `tz.key()` (which cuts at the first bracket, so R8-R9's long catch-all labels and R4's
# `Calviniste (FJKM)` key the same as the short ones). Named one by one, because an answer that
# falls through a default is silently dropped.
#   * `Calviniste (FJKM)` (R4, code 420) and `Calvinist` (R5-R9, code 31) are one box: R4's own
#     label names the church, and the level runs 23.5, 19.7, 23.8, 20.7, 23.8% across them.
#   * `Other Christian` is the rest of the card's Christian answers, `Christian only` included:
#     `Independent`, `Evangelical`, `Baptist`, `Orthodox`, `Coptic`, `Methodist`, `Presbyterian`,
#     `Dutch Reformed`, `Apostolic`, `Church of Christ`, `Mennonite`, `Zionist Christian Church`,
#     and four Madagascar-only boxes: `Rhema`, `Vahao ny Oloko` and `Toby Betela` (R5) and
#     `Fifohazana` (R9), the revival movement inside the FJKM, Lutheran, Anglican and Catholic
#     churches. Together about 4% of the pool; none has a node this sample could level.
#   * `Muslim only`, `Sunni only` and `Ismaeli` are Muslim.
#   * `None`, `Atheist`, `Agnostic` and `Traditional/ethnic religion` are the pair (docstring).
#   * `Jewish` (1 respondent, R7) goes to `Other`.
GROUP = {
    "roman catholic": "Roman Catholic",
    "calvinist": "Calvinist (FJKM)", "calviniste": "Calvinist (FJKM)",
    "lutheran": "Lutheran", "anglican": "Anglican",
    "seventh day adventist": "Seventh Day Adventist",
    "pentecostal": "Pentecostal",
    "jehovah's witness": "Jehovah's Witness",
    "christian only": "Other Christian", "independent": "Other Christian",
    "evangelical": "Other Christian", "baptist": "Other Christian", "orthodox": "Other Christian",
    "coptic": "Other Christian", "methodist": "Other Christian",
    "presbyterian": "Other Christian", "dutch reformed": "Other Christian",
    "apostolic": "Other Christian", "church of christ": "Other Christian",
    "mennonite": "Other Christian", "zionist christian church": "Other Christian",
    "rhema": "Other Christian", "vahao ny oloko": "Other Christian",
    "toby betela": "Other Christian", "fifohazana": "Other Christian",
    "muslim only": "Muslim", "sunni only": "Muslim", "ismaeli": "Muslim",
    "traditional/ethnic religion": PAIR, "none": PAIR, "atheist": PAIR, "agnostic": PAIR,
    "other": "Other", "jewish": "Other",
}
# Inside the pair, which of the card's two answers each box is.
BOX = {"traditional/ethnic religion": TRAD, "none": NONE, "atheist": NONE, "agnostic": NONE}

# Every `REGION` label rounds 5-9 use -> the census region name in mg_lookup.csv, keyed through
# `ckey()`. `Vatovavy Fitonany` (R6) is a typing slip; R9's `Vatovavy` and `Fitovinany` are the
# 2021 split, drawn as the census's one region.
NORM = {
    "alaotra mangoro": "Alaotra Mangoro", "amoron'i mania": "Amoron'i Mania",
    "analamanga": "Analamanga", "analanjirofo": "Analanjirofo", "androy": "Androy",
    "anosy": "Anosy", "atsimo andrefana": "Atsimo Andrefana",
    "atsimo atsinanana": "Atsimo Atsinanana", "atsinanana": "Atsinanana",
    "betsiboka": "Betsiboka", "boeny": "Boeny", "bongolava": "Bongolava", "diana": "Diana",
    "haute matsiatra": "Haute Matsiatra", "matsiatra ambony": "Haute Matsiatra",
    "ihorombe": "Ihorombe", "itasy": "Itasy", "melaky": "Melaky", "menabe": "Menabe",
    "sava": "Sava", "sofia": "Sofia", "vakinankaratra": "Vakinankaratra",
    "vatovavy fitovinany": "Vatovavy Fitovinany", "vatovavy fitonany": "Vatovavy Fitovinany",
    "vatovavy": "Vatovavy Fitovinany", "fitovinany": "Vatovavy Fitovinany",
}
# Two labels may name one unit inside a round only here.
SPLIT_LABELS = {"Vatovavy Fitovinany": {"vatovavy", "fitovinany"}}

# The share answering `Christian only` across the drawn rounds must stay under this for the
# churches to be kept apart (docstring). Measured 2026-09-14: 0.3% to 2.3%.
CONLY_SWING_MAX = 0.05

# Set from the split-half's output, 2026-09-14; asserted, so a change in the data is a failure
# here rather than a quiet redrawing. Median rho over the 3 halvings of 4 rounds, null 95th about
# +0.30, every one with chi-square p under 1e-9: Catholic +0.548, FJKM +0.826, Lutheran +0.797,
# Adventist +0.548, Other Christian +0.385, Muslim +0.574, None or traditional +0.682 (tested
# apart, None alone was +0.857 and traditional alone +0.246). `Jehovah's Witness` clears the rank
# test (+0.333, p 0.044) at 0.54% of the pool, under ab.ELIGIBLE_FLOOR, so it is not placed.
# Anglican, Pentecostal and Other fail. `sources/mg.md` §4 has the table.
CARRIES = ["Roman Catholic", "Calvinist (FJKM)", "Lutheran", "Seventh Day Adventist",
           "Other Christian", "Muslim", PAIR]
# No failing category tops both halves in 95% of the halvings (Anglican's best, Atsinanana, in 1
# of 3).
STANDOUTS = {}
# Spec §12's small-category rule keeps the tail as the RESIDUAL: its worst multiple is 1.68x
# (Anglican and Jehovah's Witness in Bongolava, where none of 104 pooled respondents gave either),
# under the 2x bar. With None and traditional tested apart, the same rule sent the tail flat
# (Anglican 3.01x in Melaky); pooling them is what changed it.
TAIL_FLAT = False
# Spec §12 (Norway): the pooled level against the recent rounds, both recomposed on the census.
LEVEL_GAP_MAX = 0.035

# `None`'s share of the pair, rounds 7 and 9 on census weights; asserted to 0.005 and inside the
# DHS range widened by DHS_FRACTION_SLACK.
NONE_FRACTION = 0.905
DHS_FRACTION_SLACK = 0.02

# The DHS witness. Weighted percentages of respondents aged 15-49 (INSTAT and ICF), transcribed
# from EDSMD-V 2021 Tableau 3.1 (FR376, PDF page 82) and EDSMD-IV 2008-09 Tableau 3.2 (FR236, PDF
# page 63). `protestant` is `FJKM/FLM/Anglikana` in 2021 and `Protestante/FLM` in 2008-09; 2008-09
# has no `Autre chrétien` row, so its `other` holds the other Christians and the `Secte` code.
EDS = {
    ("EDSMD-V 2021", "women"): dict(catholic=32.5, protestant=34.6, muslim=1.3, traditional=2.0,
                                    none=20.6, other_christian=8.5, other=0.5),
    ("EDSMD-V 2021", "men"): dict(catholic=31.8, protestant=32.5, muslim=1.8, traditional=1.1,
                                  none=25.0, other_christian=7.1, other=0.8),
    ("EDSMD-IV 2008-09", "women"): dict(catholic=35.7, protestant=35.6, muslim=0.7,
                                        traditional=2.3, none=19.9, other=5.7),
    ("EDSMD-IV 2008-09", "men"): dict(catholic=34.1, protestant=33.6, muslim=0.9,
                                      traditional=2.1, none=24.7, other=4.4),
}
EDS_RATIO_SLACK = 0.10          # the drawn Catholic/Protestant ratio may sit 10% outside EDS's range


def ckey(s):
    return " ".join(str(s).replace("’", "'").split()).strip().casefold()


def letters(s):
    return re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", str(s)).casefold())


def report_card():
    """Which boxes each drawn round's showcard offered at all (value labels, not responses)."""
    import pyreadstat

    watch = ["christian only", "roman catholic", "calvinist", "lutheran", "none",
             "traditional/ethnic religion", "atheist", "agnostic", "other"]
    print("\n  boxes on each round's showcard (value labels, not responses):")
    lacking = []
    for rnd, name, _url, relname, _wt in ab.ROUNDS:
        if rnd not in ROUNDS:
            continue
        _d, meta = pyreadstat.read_sav(os.path.join(ab.AB_DIR, name), metadataonly=True)
        col = next(c for c in meta.column_names if c.upper() == relname.upper())
        have = {key(v) for v in meta.variable_value_labels.get(col, {}).values()}
        print(f"    R{rnd}: " + ", ".join(f"{w} {'yes' if w in have else 'NO'}" for w in watch))
        for w in ("none", "traditional/ethnic religion", "calvinist", "lutheran"):
            if w not in have:
                lacking.append((rnd, w))
    if lacking:
        raise SystemExit(f"boxes this build relies on are missing from the card: {lacking}")


def district_unit(label, cod):
    """The unit a district label is in, by COD-AB's district names, or None (tz.py's rule)."""
    lab = str(label).strip()
    if not lab or lab.lower() in ("nan", "none"):
        return None
    word = letters(lab.replace("'", " ").split()[0])
    if not word:
        return None
    hits = set()
    for d, u in cod:
        first = letters(str(d).split()[0])
        if letters(d).startswith(word) or (len(first) >= 5 and word.startswith(first)):
            hits.add(u)
    return next(iter(hits)) if len(hits) == 1 else None


def check_locations(df, cod, province, nm):
    """R6 and R7 carry a district, R9 the old province (`LOCATION.LEVEL.1`); both must agree
    with the region label."""
    print("\n  region label against LOCATION.LEVEL.1:")
    for rnd in (6, 7):
        d = df[df["round"] == rnd]
        du = d["LOCATION.LEVEL.1"].map(lambda s: district_unit(s, cod))
        matched = du.notna()
        disagree = matched & (du != d["geo_id"])
        print(f"    R{rnd} (district): {int(matched.sum()):,} of {len(d):,} respondents' districts "
              f"match a COD-AB name; {int(disagree.sum())} disagree with the region label")
        if disagree.any():
            pairs = (d[disagree].assign(du=du[disagree])
                     .groupby(["geo_id", "LOCATION.LEVEL.1", "du"]).size())
            for (g, loc, u), n in pairs.items():
                print(f"      {nm[g]} / {loc} -> {nm[u]}  ({n})")
        if matched.sum() < 0.5 * len(d):
            raise SystemExit(f"R{rnd}: fewer than half the district labels match COD-AB; the "
                             "check has no power, read the labels")
        if disagree.sum() > 0.01 * matched.sum():
            raise SystemExit(f"R{rnd}: more than 1% of district-matched respondents sit in a "
                             "different unit from their region label")
    d = df[df["round"] == 9]
    got = d["LOCATION.LEVEL.1"].map(letters)
    want = d["geo_id"].map(province).map(letters)
    bad = got != want
    print(f"    R9 (old province): {int((~bad).sum()):,} of {len(d):,} agree with the region "
          "label's province")
    if bad.any():
        print(d[bad].groupby(["LOCATION.LEVEL.1", "geo_raw"]).size().to_string())
        raise SystemExit("R9: a region label sits in a different old province from its location")


def unit_shares(df, units, rounds, col="category", cats=None):
    cats = cats or CATEGORIES
    d = df[df["round"].isin(rounds)]
    t = d.groupby(["geo_id", col])["w"].sum().unstack(fill_value=0.0)
    t = t.reindex(index=units, columns=cats, fill_value=0.0)
    return t.div(t.sum(axis=1), axis=0)


def swap_table(df, units, nm):
    """The evidence for the pair: each unit's traditional and none shares, early against late."""
    cats = [TRAD, NONE, "rest"]
    d = df.assign(box=df["box"].fillna("rest"))
    e = unit_shares(d, units, EARLY, "box", cats)
    l = unit_shares(d, units, RECENT, "box", cats)
    print(f"\n  the pair, per unit, rounds {EARLY} -> {RECENT} (% of respondents):")
    print(f"    {'':<22}{'trad':>11}{'none':>11}{'both':>11}")
    both_e, both_l = e[TRAD] + e[NONE], l[TRAD] + l[NONE]
    for u in both_l.sort_values(ascending=False).index:
        print(f"    {nm[u]:<22}" + "".join(f"{100 * a:5.0f}->{100 * b:<4.0f}" for a, b in (
            (e.loc[u, TRAD], l.loc[u, TRAD]), (e.loc[u, NONE], l.loc[u, NONE]),
            (both_e[u], both_l[u]))))
    rho = lambda a, b: float(pd.Series(a).rank().corr(pd.Series(b).rank()))
    print(f"    rank agreement early/late across units: traditional {rho(e[TRAD], l[TRAD]):+.2f}, "
          f"none {rho(e[NONE], l[NONE]):+.2f}, the pair {rho(both_e, both_l):+.2f}")


def compose(df, nat, units, carried, stand):
    """`tz.py::compose` on this module's CATEGORIES: carried, standouts, then the 2x rule."""
    by = df.groupby(["geo_id", "category"])["w"].sum().unstack(fill_value=0.0)
    by = by.reindex(index=units, columns=CATEGORIES, fill_value=0.0)
    own = by.div(by.sum(axis=1), axis=0)
    nraw = df.groupby(["geo_id", "category"]).size().unstack(fill_value=0)
    nraw = nraw.reindex(index=units, columns=CATEGORIES, fill_value=0)

    fixed = pd.DataFrame(0.0, index=units, columns=CATEGORIES)
    for c in carried:
        fixed[c] = own[c]
    for c, u in stand.items():
        rest = [x for x in units if x != u]
        fixed[c] = float(by.loc[rest, c].sum() / by.loc[rest].sum().sum())
        fixed.loc[u, c] = own.loc[u, c]
    tail = [c for c in CATEGORIES if c not in carried and c not in stand]
    tail_nat = float(sum(nat[c] for c in tail))

    frame = fixed.copy()
    remainder = 1.0 - fixed[carried + list(stand)].sum(axis=1)
    for c in tail:
        frame[c] = remainder * nat[c] / tail_nat
    mult = remainder / tail_nat
    print("\n  spec §12 small-category rule: the residual's multiple of national share, in units "
          "where the survey found none of that category:")
    rows, worst = stability.residual_multiples(mult, nraw == 0, tail)
    for c, u, m in rows:
        print(f"    {c:<30} worst {m:.2f}x in {u} (found none there; national "
              f"{100 * nat[c]:.2f}%)")
    flat = worst is not None and worst[2] >= stability.SMALL_CATEGORY_MULTIPLE
    if flat:
        print(f"    {worst[0]} at {worst[2]:.2f}x in {worst[1]}: 2x or more, so the tail goes FLAT")
        scale = (1.0 - tail_nat) / fixed[carried + list(stand)].sum(axis=1)
        frame = fixed.mul(scale, axis=0)
        for c in tail:
            frame[c] = nat[c]
    else:
        print("    under 2x everywhere, so the tail stays the residual")
    if (frame.sum(axis=1) - 1.0).abs().max() > 1e-9:
        raise SystemExit("a unit's shares do not sum to 1")
    return frame, own, nraw, flat


def none_fraction(df, units, w):
    """`None`'s share of the pair in rounds 7 and 9 on census weights, against both DHS surveys."""
    late = unit_shares(df.assign(box=df["box"].fillna("rest")), units, RECENT, "box",
                       [TRAD, NONE, "rest"]).mul(w, axis=0).sum()
    frac = float(late[NONE] / (late[NONE] + late[TRAD]))
    dhs = {k: v["none"] / (v["none"] + v["traditional"]) for k, v in EDS.items()}
    print(f"\n  None's share of the pair: rounds {RECENT} on census weights {frac:.3f} ("
          f"None {100 * late[NONE]:.2f}%, traditional {100 * late[TRAD]:.2f}%); DHS "
          + ", ".join(f"{s} {x} {f:.3f}" for (s, x), f in dhs.items()))
    lo, hi = min(dhs.values()) - DHS_FRACTION_SLACK, max(dhs.values()) + DHS_FRACTION_SLACK
    if not lo <= frac <= hi:
        raise SystemExit(f"None's share of the pair {frac:.3f} is outside the DHS range widened "
                         f"to {lo:.3f}-{hi:.3f}")
    if abs(frac - NONE_FRACTION) > 0.005:
        raise SystemExit(f"None's share of the pair is now {frac:.3f}, not {NONE_FRACTION}; "
                         "edit NONE_FRACTION and the docstring deliberately")
    return frac


def witness_eds(drawn):
    """The two DHS surveys against the drawn national shares. Asserts the Catholic/Protestant
    ratio; prints the rest, which is where the two instruments disagree."""
    prot = drawn["Calvinist (FJKM)"] + drawn["Lutheran"] + drawn["Anglican"]
    ratio = drawn["Roman Catholic"] / prot
    print("\n  DHS witness (respondents 15-49; the Afrobarometer is adults 18 and over):")
    print(f"    {'':<28}{'Catholic':>9}{'FJKM+FLM+Angl':>14}{'ratio':>7}{'Muslim':>8}"
          f"{'Trad.':>7}{'None':>7}")
    ratios = []
    for (svy, sex), v in EDS.items():
        r = v["catholic"] / v["protestant"]
        ratios.append(r)
        print(f"    {svy + ', ' + sex:<28}{v['catholic']:8.1f}%{v['protestant']:13.1f}%{r:7.2f}"
              f"{v['muslim']:7.1f}%{v['traditional']:6.1f}%{v['none']:6.1f}%")
    print(f"    {'this map, as drawn':<28}{100 * drawn['Roman Catholic']:8.1f}%{100 * prot:13.1f}%"
          f"{ratio:7.2f}{100 * drawn['Muslim']:7.1f}%"
          f"{100 * drawn[TRAD]:6.1f}%{100 * drawn[NONE]:6.1f}%")
    lo, hi = min(ratios) * (1 - EDS_RATIO_SLACK), max(ratios) * (1 + EDS_RATIO_SLACK)
    if not lo <= ratio <= hi:
        raise SystemExit(f"the drawn Catholic/Protestant ratio {ratio:.2f} is outside the DHS "
                         f"range widened to {lo:.2f}-{hi:.2f}; the churches' level has no witness")
    print(f"    the drawn ratio {ratio:.2f} is inside the DHS range {min(ratios):.2f}-"
          f"{max(ratios):.2f} widened by {EDS_RATIO_SLACK:.0%}")


def main():
    print("=== Afrobarometer, Madagascar ===")
    raw = ab.load(COUNTRY, expect_rounds=ROUNDS_READ, regroup=True, extra=["LOCATION.LEVEL.1"])
    print(f"\n  pooled: {len(raw):,} respondents with a religion answer over five rounds")

    ct = pd.crosstab(raw["category"], raw["round"])
    ct["all"] = ct.sum(axis=1)
    print("\n  every answer as it arrives (this is what GROUP collapses):")
    print(ct.sort_values("all", ascending=False).to_string(max_colwidth=44))

    conly = raw[raw["category"].map(key) == "christian only"]
    by_round = (conly.groupby("round")["w"].sum() / raw.groupby("round")["w"].sum())
    by_round = by_round.reindex(ROUNDS_READ).fillna(0.0)
    print("\n  share answering `Christian only` rather than naming a church, by round:")
    for r, v in by_round.items():
        print(f"    R{r}  {v:6.1%}")
    used = by_round.reindex(ROUNDS)
    if used.max() - used.min() > CONLY_SWING_MAX:
        raise SystemExit(f"the `Christian only` share now swings {100 * (used.max() - used.min()):.1f} "
                         "points across the drawn rounds, so the churches cannot be kept apart; "
                         "re-read the docstring")
    report_card()

    # ---- group ----
    k = raw["category"].map(key)
    unmapped = sorted(set(k) - set(GROUP))
    if unmapped:
        raise SystemExit(f"answers with no category: {unmapped}; add them to GROUP deliberately")
    df = raw.copy()
    df["raw_category"] = raw["category"]
    df["category"] = k.map(GROUP)
    df["box"] = k.map(BOX)
    ab.assert_one_wording(df, COUNTRY)

    rn = df.assign(show=df["box"].fillna(df["category"])).groupby(["round", "show"])["w"].sum()
    rn = rn.unstack(fill_value=0)
    print("\n  grouped, by round (survey weighting, %), the pair shown as its two answers, round 4 "
          "included for the record:")
    print((100 * rn.div(rn.sum(axis=1), axis=0)).round(1).reindex(columns=OUT_CATEGORIES)
          .T.to_string())

    # ---- units ----
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str}, keep_default_na=False)
    if len(lut) != N_UNITS:
        raise SystemExit(f"{LOOKUP} has {len(lut)} units, expected {N_UNITS}; re-run mg_geo.py")
    nm = dict(zip(lut["geo_id"], lut["name"]))
    name_to_id = dict(zip(lut["name"], lut["geo_id"]))
    province = dict(zip(lut["geo_id"], lut["province"]))
    pop = lut.set_index("geo_id")["pop"].astype(float)
    if int(pop.sum()) != CENSUS_2018:
        raise SystemExit(f"mg_lookup.csv sums to {int(pop.sum()):,}, not {CENSUS_2018:,}")
    units = sorted(lut["geo_id"])
    dl = pd.read_csv(DISTRICTS, dtype=str, keep_default_na=False)
    cod = list(zip(dl["district"], dl["unit"]))

    n4 = int((df["round"] == 4).sum())
    df = df[df["round"].isin(ROUNDS)].copy()
    print(f"\n  round 4 left out ({n4:,} respondents): six old provinces, and its districts reach "
          "21 of the 22 regions (no Betsiboka); see the docstring")

    df["unit_name"] = df["geo_raw"].map(ckey).map(NORM)
    bad = sorted(df.loc[df["unit_name"].isna(), "geo_raw"].astype(str).unique())
    if bad:
        raise SystemExit(f"REGION labels with no unit: {bad}; add them to NORM deliberately")
    for (rnd, u), labs in df.groupby(["round", "unit_name"])["geo_raw"].unique().items():
        folded = {ckey(x) for x in labs}
        if len(folded) > 1 and folded != SPLIT_LABELS.get(u):
            raise SystemExit(f"R{rnd}: {sorted(labs)} all decode to {u}")
    df["geo_id"] = df["unit_name"].map(name_to_id)
    if df["geo_id"].isna().any():
        raise SystemExit(f"unit names not in the lookup: {sorted(df.loc[df['geo_id'].isna(), 'unit_name'].unique())}")
    moved = (df.groupby("geo_code")["geo_id"].nunique() > 1)
    print(f"\n  REGION codes naming different units in different rounds: "
          f"{sorted(int(c) for c in moved[moved].index)} (decoded by label, never by code)")
    check_locations(df, cod, province, nm)

    per_round = df.groupby("round")["geo_id"].nunique()
    print("    units present per round: " + ", ".join(f"R{r} {n}" for r, n in per_round.items()))
    if (per_round != N_UNITS).any():
        raise SystemExit("a drawn round does not sample all 22 units")

    # ---- the decode, per round, without touching the religion column ----
    for rnd in ROUNDS:
        print(f"\n  R{rnd}:", end="")
        ab.held_out(df[df["round"] == rnd], pop, f"{COUNTRY} R{rnd}", pop_source="RGPH-3 2018")
    ab.held_out(df, pop, COUNTRY, pop_source="RGPH-3 2018")

    swap_table(df, units, nm)

    nat = ab.national(df).reindex(CATEGORIES).fillna(0.0)
    print(f"\n  the survey's national shares, pooled over R{', R'.join(map(str, ROUNDS))} "
          f"(n={len(df):,}):")
    for c in CATEGORIES:
        print(f"    {c:<30}{nat[c]:8.3%}")

    # ---- quota, then the split-half ----
    dfw = df.rename(columns={"round": "wave", "category": "code"})[["wave", "geo_id", "code", "w"]]
    cab.assert_not_quota(dfw, COUNTRY, ROUNDS, unit_col="geo_id", cat_col="code")
    carries, table = cab.stability(dfw, CATEGORIES, units, f"{N_UNITS} regions")
    carries = [c for c in CATEGORIES if c in carries and nat[c] >= ab.ELIGIBLE_FLOOR]
    if sorted(carries) != sorted(CARRIES):
        raise SystemExit(f"the split-half now selects {carries}, not {CARRIES}. "
                         "Read the table above, then edit CARRIES and the docstring deliberately.")
    failing = [c for c in CATEGORIES if c not in carries and nat[c] >= ab.ELIGIBLE_FLOOR]
    stand = standouts(dfw, CATEGORIES, units, failing, table)
    if stand != STANDOUTS:
        raise SystemExit(f"standouts are now {stand}, against STANDOUTS={STANDOUTS}")

    # ---- compose ----
    frame, own, nraw, flat = compose(df, nat, units, carries, stand)
    if flat != TAIL_FLAT:
        raise SystemExit(f"the small-category rule now gives flat={flat}, against TAIL_FLAT="
                         f"{TAIL_FLAT}; read the multiples above and decide deliberately")

    # ---- level: pooled against the recent rounds, both recomposed on the census ----
    w = pop.reindex(units) / pop.sum()
    pooled = frame.mul(w, axis=0).sum()
    recent = unit_shares(df, units, RECENT).mul(w, axis=0).sum()
    print(f"\n  national level: the survey pool, as composed on census weights, and rounds "
          f"{' and '.join(map(str, RECENT))} alone on census weights:")
    print(f"    {'category':<30}{'survey':>9}{'pooled':>9}{'recent':>9}{'pooled-recent':>15}")
    for c in CATEGORIES:
        print(f"    {c:<30}{100 * nat[c]:8.2f}%{100 * pooled[c]:8.2f}%{100 * recent[c]:8.2f}%"
              f"{100 * (pooled[c] - recent[c]):+14.2f}")
    stale = [c for c in CATEGORIES if abs(pooled[c] - recent[c]) > LEVEL_GAP_MAX]
    if stale:
        raise SystemExit(f"the pooled level differs from rounds {RECENT} by more than "
                         f"{100 * LEVEL_GAP_MAX:.1f} points for {stale}; spec §12 (Norway) says "
                         "consider §3.4 before drawing")

    # ---- split the pair into the card's two answers ----
    frac = none_fraction(df, units, w)
    zero_pair = sorted(nm[u] for u in units if frame.loc[u, PAIR] == 0)
    out_frame = frame.drop(columns=PAIR)
    out_frame[NONE] = frame[PAIR] * frac
    out_frame[TRAD] = frame[PAIR] * (1.0 - frac)
    out_frame = out_frame[OUT_CATEGORIES]

    counts = round_within_rows(out_frame.mul(pop.reindex(units), axis=0))
    if not (counts.sum(axis=1) == pop.reindex(units).round().astype("int64")).all():
        raise SystemExit("a unit's drawn total is not its census population")
    drawn = counts.sum(axis=0) / counts.sum().sum()
    witness_eds(drawn)

    # ---- write ----
    n_by = df.groupby("geo_id").size()

    def note_for(c):
        if c in (NONE, TRAD):
            part = frac if c == NONE else 1.0 - frac
            base = ("the unit's own measured share of none or traditional" if PAIR in carries
                    else "none or traditional at the national share")
            return (f"{base}, of which {100 * part:.1f}% is drawn as {c.lower()} "
                    f"(rounds 7 and 9, nationally)")
        if c in carries:
            return "the unit's own measured share"
        if c in stand:
            return f"the unit's own share in {nm[stand[c]]}, the rest of the country's elsewhere"
        return "the national share" if flat else "the national proportion within the unit's remainder"

    basis_note = {c: note_for(c) for c in OUT_CATEGORIES}
    out = counts.stack().rename("count").reset_index()
    out.columns = ["geo_id", "source_category", "count"]
    out["geo_level"] = "region"
    out["geo_name"] = out["geo_id"].map(nm)
    out["basis"] = "self_id"
    out["year"] = YEARS
    out["source_id"] = SOURCE_ID
    out["note"] = out.apply(
        lambda r: ("RGPH-3 2018 census population composed with the unit's own mix from "
                   f"Afrobarometer rounds 5, 6, 7 and 9 pooled (n={int(n_by[r.geo_id])} here); "
                   f"{basis_note[r.source_category]}"), axis=1)
    total = int(out["count"].sum())
    if total != CENSUS_2018:
        raise SystemExit(f"drawn {total:,} against the census {CENSUS_2018:,}")
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, "
          f"{out['source_category'].nunique()} categories, {out['geo_id'].nunique()} units)")

    print("\n  national, as drawn:")
    for c, n in counts.sum(axis=0).sort_values(ascending=False).items():
        print(f"    {100 * n / total:6.2f}%  {c}  ({n:,})")
    share = counts.div(counts.sum(axis=1), axis=0)
    short = ["Cath", "FJKM", "Luth", "Angl", "SDA", "Pent", "JW", "OthC", "Musl", "Trad", "None",
             "Oth"]
    print("\n  as drawn, by unit, most Catholic first (pooled n on the right):")
    print(f"    {'':<22}" + "".join(f"{s:>6}" for s in short))
    for u in share.sort_values("Roman Catholic", ascending=False).index:
        s = share.loc[u]
        print(f"    {nm[u]:<22}" + "".join(f"{100 * s[c]:6.1f}" for c in OUT_CATEGORIES)
              + f"{int(pop[u]):>11,}  n={int(n_by[u])}")
    zero = [(nm[u], c) for u in units for c in carries if c != PAIR and frame.loc[u, c] == 0]
    print(f"  drawn at zero in a carried category (no pooled respondent gave it): {zero or 'none'}")
    print(f"  drawn with no one of no religion or traditional religion: {zero_pair or 'none'}")
    print(f"  thinnest unit {nm[n_by.idxmin()]} n={int(n_by.min())}; median n={int(n_by.median())}; "
          f"total n={int(n_by.sum()):,}")


if __name__ == "__main__":
    main()
