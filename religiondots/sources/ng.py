"""Nigeria — the state pattern of Christianity and Islam, from six pooled Afrobarometer rounds.

Reads data/raw/afrobarometer/*.sav and data/geo/ng/ng_lookup.csv, writes
data/normalized/ng.csv. `sources/ng.md` has the acquisition route, the terms and the full
scouting record; `sources/afrobarometer.py` holds the construction shared with every other
country drawn from that survey; `sources.md` §11ai assesses the source across Africa.

## NIGERIA DOES NOT ASK, AND THE REASON IT DOES NOT ASK IS THIS MAP'S SUBJECT

**The last Nigerian census to publish religion was 1963, and it is not the last one to ask.**
The 1973 census asked and was annulled in 1975 with nothing released, amid allegations that
the returns had been falsified; the 1991 and 2006 censuses did not ask at all, and the
National Population Commission has said the postponed 2023 census will not either. The reason
given is consistent and is this map's subject: a Christian/Muslim count is explosive in a
federation whose revenue and whose offices are shared out by population. So the country with
the largest population of Muslims and the largest population of Christians in Africa has
published no count of either since independence, and `sources.md` §11p's flat *"Nigeria does
not ask"* is still true.

**This map draws it anyway, from a survey, and says so.** The alternative is a hole the size of
Nigeria in the middle of the continent, which tells a reader nothing at all, and §12's rule is
that modelled is the floor rather than the plan. What is on the map is what 11,909 people told
the Afrobarometer between May 2008 and April 2022, applied to a population table somebody else
projected. Every dot is `modelled` and draws desaturated.

## WHAT IS DRAWN, AND WHERE EACH NUMBER COMES FROM

    row margin      state populations       COD-PS 2022 (NPC projection)   EXACT
    the composition each state's own mix    Afrobarometer R4-R9 pooled     measured
    the national level                      neither, and see below         computed

**Nothing here fits a column margin, and that is the whole difference from `sources/lr.py`.**
Liberia had a census religion total to fit to, so its Afrobarometer supplied only the
interaction. Nigeria has no enumeration of any kind, so there is nothing with that property to
fit to and the alternatives are all somebody else's estimate: fitting to Pew would set a
`self_id` magnitude from a compiler's synthesis, which spec §3.1 does not allow and which
would in any case launder this survey through a figure partly built on it, and fitting to the
NDHS would mean scaling a 15-to-49 distribution up to a whole population, which is what §3.4
refused for Brazil.

**So the national level is COMPUTED rather than measured or fitted, and it is not the survey's
own published one.** Each state is drawn at its own measured mix and at its COD-PS population,
so the country's total is the state pattern reweighted by NPC's projected state populations.
That is a different number from the Afrobarometer's own pooled national share, because the
pooled sample's state mix is not COD-PS's: `held_out` reports the far-northern states sampled
below their population weight and the far-southern ones above it, and round 6 has no Adamawa,
Borno or Yobe at all. Both figures are printed on every build and `note_public` gives the
reader both, along with the two published estimates that disagree with each.

## THE DENOMINATIONS ARE NOT DRAWN, AND NEITHER IS THE SUNNI/SHIA SPLIT

Afrobarometer's card offers about twenty Christian denominations and, on the Muslim side,
Sunni, Shia, Ismaili and three Sufi brotherhoods plus Izala. Nigerians fill nearly all of them:
Roman Catholic takes 8.6% of the pooled respondents, Pentecostal 4.8%, Anglican 3.0%, the
Tijaniyya 1.0% and Izala 0.7%. None of it is drawn, for §11ai's reason, which bites here as
hard as it did in Liberia. The share answering `Christian only` rather than naming a
denomination, by round:

    R4  32.3%    R5  19.4%    R6  28.0%    R7  44.9%    R8  47.3%    R9  41.1%

A 27.9-point range with no trend, over fourteen years in which nothing like that happened to
Nigerian denominational life. A pooled Catholic share is therefore a measurement of which
rounds are in the pool. `GROUP` below collapses the card to five categories that the probing
cannot move, because a Catholic and a `Christian only` are both Christian in every round.

**The Muslim side has a second reason on top of that one**, and it is §14. `Shia` and `Shia
only` take 58 of 11,909 respondents across six rounds, in a country where the Islamic Movement
in Nigeria has been proscribed since 2019 and its members killed in numbers. A national
0.55% built from 58 respondents cannot say where Nigerian Shi'ism is, and §14.4's rule 2 says
that for a persecuted group this map draws nothing finer than the state publishes, which here
is nothing at all. Izala is not drawn either, on the arithmetic alone: all 85 of its
respondents are in rounds 4, 5 and 7, so its pooled share measures which rounds are in the
pool. `report_card()` prints, per round, whether an answer was on that round's showcard at
all, because a zero in the crosstab is a fact about the respondents and not about the card,
and this file used to state the stronger claim from the weaker evidence.

## THE STATE LABELS ARE `[[reference_pooled_survey_labels]]`

`REGION` brings its own label set every round: rounds 8 and 9 upper-case the whole set,
`Akwa Ibom` and `Cross River` gain and lose their hyphens, Nasarawa is spelled `Nassarawa` in
round 4, and the Federal Capital Territory is `FCT`, `Fct Abuja` and `FCT ABUJA` in three
different rounds. Pooled on the raw string Nigeria has **78 units instead of 37**, and the
split-half is then run on halves that share almost nothing. `NORM` is the harmonisation and
every label is asserted to be in it.

**Round 6 has no Adamawa, Borno or Yobe**, which is the Boko Haram insurgency: that round was
in the field in December 2014 and January 2015, when Borno and Yobe were substantially outside
federal control. All three states are in R4 and R5, so all 37 appear in both halves of the
split-half and the test runs on the units its bar was computed for. What it does mean is that
the north-east is measured on fewer rounds than the rest of the country.

Usage:
    python sources/ng.py --fetch    download the six merged Afrobarometer rounds (~280 MB)
    python sources/ng.py            rebuild data/normalized/ng.csv
"""

import os
import re
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import afrobarometer as ab

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOOKUP = os.path.join(ROOT, "data", "geo", "ng", "ng_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "ng.csv")

COUNTRY = "Nigeria"
ROUNDS = [4, 5, 6, 7, 8, 9]
SOURCE_ID = "ng_afrobarometer_2008_2022_codps2022"
YEARS = "2008-2022"

N_UNITS = 37
CODPS_2022 = 216_798_930

# The five categories drawn. Three of them are the Afrobarometer card's own boxes, verbatim;
# `Christian` and `Muslim` are that card's own umbrella answers (`Christian only`, `Muslim
# only`) with their denominational children folded back in, for the reason in the docstring.
CATEGORIES = ["Christian", "Muslim", "Traditional/ethnic religion", "None", "Other"]

# Every answer Nigerians give across the six rounds -> the category it is drawn as, keyed
# through `key()`. 35 answers, and every one of them is named here rather than falling through
# a default, because an answer that falls through is silently dropped.
#
# Placements worth stating rather than assuming:
#   * `Independent` is Afrobarometer's box for an African independent church. In Nigeria that
#     is the Aladura churches (Cherubim and Seraphim, Christ Apostolic, the Celestial Church of
#     Christ) and it is Christian; the survey gives it 15 respondents, which is a count of who
#     recognised the word rather than of the Aladura.
#   * `Zionist Christian Church` and `Dutch Reformed` are southern African boxes on a
#     continental card. 22 Nigerians between them, all Christian.
#   * `Izala`, `Tijaniya`, `Qadiriya` and `Mouridiya` are Muslim, and the brotherhoods are not
#     an alternative to being Muslim.
#   * `Atheist` and `Agnostic` go to `None` with the survey's own no-religion box. Two
#     respondents in fourteen years hold them between them, which cannot support a node of
#     their own; ke2019.py, mw2018.py and lr2022.py make the same call for the same reason.
GROUP = {
    # Christian
    "christian only": "Christian",
    "roman catholic": "Christian",
    "orthodox": "Christian",
    "coptic": "Christian",
    "anglican": "Christian",
    "lutheran": "Christian",
    "methodist": "Christian",
    "presbyterian": "Christian",
    "baptist": "Christian",
    "quaker/friends": "Christian",
    "mennonite": "Christian",
    "evangelical": "Christian",
    "pentecostal": "Christian",
    "independent": "Christian",
    "jehovah's witness": "Christian",
    "seventh day adventist": "Christian",
    "mormon": "Christian",
    "church of christ": "Christian",
    "zionist christian church": "Christian",
    "dutch reformed": "Christian",
    "calvinist": "Christian",
    # Muslim
    "muslim only": "Muslim",
    "sunni only": "Muslim",
    "shia": "Muslim",
    "shia only": "Muslim",
    "ismaeli": "Muslim",
    "izala": "Muslim",
    "mouridiya brotherhood": "Muslim",
    "tijaniya brotherhood": "Muslim",
    "qadiriya brotherhood": "Muslim",
    # the rest
    "traditional/ethnic religion": "Traditional/ethnic religion",
    "other": "Other",
    "none": "None",
    "atheist": "None",
    "agnostic": "None",
}

# Every `REGION` label the six rounds use -> COD-AB's state name, keyed through `ckey()`.
# Rounds 8 and 9 upper-case the lot; the hyphens, the double-s Nassarawa and the three
# spellings of the capital territory are the rest of it.
NORM = {
    "abia": "Abia",
    "adamawa": "Adamawa",
    "akwa ibom": "Akwa Ibom", "akwa-ibom": "Akwa Ibom",
    "anambra": "Anambra",
    "bauchi": "Bauchi",
    "bayelsa": "Bayelsa",
    "benue": "Benue",
    "borno": "Borno",
    "cross river": "Cross River", "cross-river": "Cross River",
    "delta": "Delta",
    "ebonyi": "Ebonyi",
    "edo": "Edo",
    "ekiti": "Ekiti",
    "enugu": "Enugu",
    "fct": "Federal Capital Territory", "fct abuja": "Federal Capital Territory",
    "gombe": "Gombe",
    "imo": "Imo",
    "jigawa": "Jigawa",
    "kaduna": "Kaduna",
    "kano": "Kano",
    "katsina": "Katsina",
    "kebbi": "Kebbi",
    "kogi": "Kogi",
    "kwara": "Kwara",
    "lagos": "Lagos",
    "nasarawa": "Nasarawa", "nassarawa": "Nasarawa",
    "niger": "Niger",
    "ogun": "Ogun",
    "ondo": "Ondo",
    "osun": "Osun",
    "oyo": "Oyo",
    "plateau": "Plateau",
    "rivers": "Rivers",
    "sokoto": "Sokoto",
    "taraba": "Taraba",
    "yobe": "Yobe",
    "zamfara": "Zamfara",
}

# The other two published national estimates, for `check_level()`. Neither is drawn from.
#
# NDHS 2018 (NPC and ICF), *Nigeria Demographic and Health Survey 2018*, report FR359, Table
# 3.1 "Background characteristics of respondents", page 51. 41,821 women and 11,867 men aged
# 15-49, weighted percents, printed separately by sex and never combined by the report — which
# is why both rows are carried here and neither is averaged. `Christian` is the report's
# `Catholic` plus its `Other Christian`. The survey's own table has NO no-religion category at
# all, which is a fact about the questionnaire rather than about Nigeria.
#
# THE UNIVERSE IS 15-49 AND THE MAP'S IS EVERYONE, at both ends. The NDHS sees neither the
# under-15s, who are about 43% of Nigeria on COD-PS's own age columns, nor anyone over 49.
# §11ah priced exactly this gap for Haiti against a census that crossed religion with age;
# Nigeria has no such census, so here it CANNOT BE PRICED and is stated instead. That is also
# why this file does not fit to the NDHS: scaling a 15-to-49 distribution up to a whole
# population is what §3.4 refused for Brazil.
NDHS_2018_WOMEN = {"Catholic": 0.104, "Other Christian": 0.356, "Christian": 0.460,
                   "Islam": 0.535, "Traditionalist": 0.003, "Other": 0.002}
NDHS_2018_MEN = {"Catholic": 0.113, "Other Christian": 0.345, "Christian": 0.458,
                 "Islam": 0.535, "Traditionalist": 0.006, "Other": 0.001}

# Pew Research Center, "5 facts about religion in Nigeria", 11 November 2025, carrying the
# 2020 estimates from *The World's Religious Groups: How Their Sizes Changed from 2010 to
# 2020* (June 2025): Muslims 56.1% and about 120 million, Christians 43.4% and about 93
# million, everything else 0.6% and about 1.25 million. A synthesis of the available surveys
# rather than a count — Pew has no Nigerian enumeration to work from either, and the same
# short-read says the last census to measure religion at all was 1973, whose results were never
# published amid falsification allegations, leaving 1963 as the last public figure.
PEW_2020 = {"Christian": 0.434, "Muslim": 0.561, "Other": 0.006}

# The six geopolitical zones, which is how Nigeria's own federal-character arithmetic groups
# the states and how every Nigerian discussion of this subject is framed. Used ONLY to make the
# printout readable and to state the north-east coverage gap; nothing is drawn on them and no
# number is computed from them that is not also computed per state.
ZONES = {
    "North West": ["Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Sokoto", "Zamfara"],
    "North East": ["Adamawa", "Bauchi", "Borno", "Gombe", "Taraba", "Yobe"],
    "North Central": ["Benue", "Federal Capital Territory", "Kogi", "Kwara", "Nasarawa",
                      "Niger", "Plateau"],
    "South West": ["Ekiti", "Lagos", "Ogun", "Ondo", "Osun", "Oyo"],
    "South East": ["Abia", "Anambra", "Ebonyi", "Enugu", "Imo"],
    "South South": ["Akwa Ibom", "Bayelsa", "Cross River", "Delta", "Edo", "Rivers"],
}

# What is drawn on its own state shares, asserted so a change in the data is a failure here
# rather than a silent re-drawing of the country. Set from the split-half below.
CARRIES = ["Christian", "Muslim"]

# THE ONLY OUTSIDE EVIDENCE THIS COUNTRY HAS, and it is a legal fact rather than a statistic.
# What it is worth is set out honestly at `check_sharia()`: one leg is real evidence about the
# geography and the other is a decode smoke test, and a third and a fourth form of it were
# written first and were too fine for this data.
# Between October 1999 and 2001 twelve northern states extended Sharia from personal-status
# matters to criminal law, each by an act of its own House of Assembly, and no other state did.
# That is a decision a Muslim-majority state took; it is not a decision every Muslim-majority
# state took, so the implication runs ONE WAY and `check_sharia()` tests it that way.
#
# THE SECOND LEG IS A BULK SEPARATION, and it is deliberately coarse. Two sharper forms were
# written first and both were claims the data cannot carry:
#
#   * "Kaduna is the least Muslim of the twelve", because its law applied the code only in the
#     Muslim-majority local government areas. The survey puts Gombe 66.3% and Kaduna 67.4%, and
#     Gombe's Tangale and Waja south is a real Christian district, so this was an assertion
#     about a one-point gap.
#   * "the twelve are separated from every non-adopting state with no overlap". Adamawa, which
#     did not adopt, comes out at 68.0% on 200 pooled respondents, above Gombe's 66.3%.
#
# Neither was tuned to pass. What replaced them is a median-to-median gap with a bar far below
# the observed value, which is what a check for a BROKEN DECODE should look like: it cannot be
# moved by one thinly-sampled state, and a join that had sent the north's respondents to the
# south would fail it by a mile.
SHARIA_STATES = ["Bauchi", "Borno", "Gombe", "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi",
                 "Niger", "Sokoto", "Yobe", "Zamfara"]
SHARIA_MEDIAN_GAP = 0.40


def key(s):
    """An Afrobarometer religion label reduced to what identifies the answer.

    Verbatim from `sources/lr.py`, deliberately: it is a label normaliser and not a grouping,
    and the four things it absorbs are the four this survey does to a label between rounds.
    It is copied rather than lifted into `sources/afrobarometer.py` because a country module
    owning its own key is what lets one country decide that two spellings are one answer while
    another does not.
    """
    s = unicodedata.normalize("NFKC", str(s)).replace("’", "'").replace("‘", "'")
    s = s.split("(")[0]
    s = re.sub(r"\s*/\s*", "/", s)
    return " ".join(s.split()).strip().casefold()


def ckey(s):
    return " ".join(str(s).split()).strip().casefold()


def report_zones(share, pop, n_by, nm):
    """The state table, grouped into the six zones. Reporting only; nothing is drawn on it."""
    print(f"\n  as drawn, by state within the six geopolitical zones "
          f"(the pooled sample behind each in brackets):")
    print(f"    {'state':<28}{'Christian':>10}{'Muslim':>9}{'Trad':>7}{'None':>7}"
          f"{'Other':>7}{'people':>14}{'n':>7}")
    by_name = {nm[g]: g for g in share.index}
    for zone, members in ZONES.items():
        zp = sum(float(pop[by_name[m]]) for m in members)
        zc = sum(share.loc[by_name[m], "Christian"] * float(pop[by_name[m]]) for m in members)
        zm = sum(share.loc[by_name[m], "Muslim"] * float(pop[by_name[m]]) for m in members)
        print(f"    -- {zone}: {zp / 1e6:.1f}M people, {zc / zp:.1%} Christian, "
              f"{zm / zp:.1%} Muslim")
        for m in sorted(members, key=lambda m: -share.loc[by_name[m], "Muslim"]):
            g = by_name[m]
            s = share.loc[g]
            print(f"    {m:<28}{s['Christian'] * 100:9.1f}%{s['Muslim'] * 100:8.1f}%"
                  f"{s['Traditional/ethnic religion'] * 100:6.1f}%{s['None'] * 100:6.1f}%"
                  f"{s['Other'] * 100:6.1f}%{int(pop[g]):>14,}{int(n_by[g]):>7}")


def round_within_rows(m):
    """Largest-remainder rounding inside each row, so every state total stays exact."""
    out = np.zeros(m.shape, dtype="int64")
    for i in range(m.shape[0]):
        row = m.iloc[i].to_numpy(dtype=float)
        target = int(round(row.sum()))
        base = np.floor(row).astype("int64")
        short = target - int(base.sum())
        if short:
            base[np.argsort(-(row - base))[:short]] += 1
        out[i] = base
    return pd.DataFrame(out, index=m.index, columns=m.columns)


def report_card(answers):
    """Which answers each round's SHOWCARD offered, as against which ones anybody chose.

    A zero in the crosstab is a fact about the respondents; an answer that is not in the
    round's value-label set is a fact about the questionnaire, and the two support very
    different statements. This reads the label sets only — `metadataonly=True`, so it does not
    re-read the 280 MB of respondents `ab.load` has already been through — and prints, for
    every answer this country's people give, the rounds whose card carried it.

    `answers` is the set of `key()`ed answers actually chosen, so the card's dozens of
    never-chosen boxes do not fill the screen.
    """
    import pyreadstat

    on_card = {}
    for rnd, name, _url, relname, _wt in ab.ROUNDS:
        p = os.path.join(ab.AB_DIR, name)
        try:
            _d, meta = pyreadstat.read_sav(p, metadataonly=True)
        except Exception as exc:                       # noqa: BLE001 - reported, not raised
            print(f"  (could not read R{rnd} metadata: {exc})")
            return
        col = next((c for c in meta.column_names if c.upper() == relname.upper()), None)
        labels = meta.variable_value_labels.get(col, {}) if col else {}
        for lab in labels.values():
            on_card.setdefault(key(lab), set()).add(rnd)

    print("\n  which rounds' showcard carried each answer Nigerians chose (the value-label "
          "set, not the responses):")
    for a in sorted(answers):
        rounds = sorted(on_card.get(a, ()))
        missing = [r for r in ROUNDS if r not in rounds]
        if missing:
            print(f"    {a[:44]:<46}on the card in R{', R'.join(map(str, rounds)) or 'none'} "
                  f"— NOT offered in R{', R'.join(map(str, missing))}")
    unlisted = sorted(a for a in answers if a not in on_card)
    if unlisted:
        print(f"    answers chosen that appear on no round's card: {unlisted} — that should "
              "be impossible, so the label read above is wrong")


def check_level(nat, drawn):
    """The national level, against the other published estimates. Reports, decides nothing.

    THE LEVEL IS THE DISPUTED NUMBER IN THIS COUNTRY. The figures are set out here on every
    build so nobody has to take the docstring's word for the spread, and `note_public` gives
    the reader the same ones. It is called AFTER the build and takes the drawn shares as an
    argument, because a version of this that printed before it labelled the survey's own pooled
    figure `drawn here` and that was not what got drawn.
    """
    chr_ = float(drawn.get("Christian", 0.0))
    mus = float(drawn.get("Muslim", 0.0))
    print("\n  the national CHRISTIAN/MUSLIM balance, which is the number Nigeria argues "
          "about, against the other published estimates:")
    print(f"    {'source':<44}{'Christian':>11}{'Muslim':>9}")
    print(f"    {'THIS MAP, as drawn':<44}{chr_:10.1%}{mus:9.1%}")
    print(f"    {'Afrobarometer R4-R9, its own pooled weighting':<44}"
          f"{float(nat.get('Christian', 0.0)):10.1%}{float(nat.get('Muslim', 0.0)):9.1%}")
    print(f"    {'NDHS 2018 FR359 Table 3.1, women 15-49':<44}"
          f"{NDHS_2018_WOMEN['Christian']:10.1%}{NDHS_2018_WOMEN['Islam']:9.1%}")
    print(f"    {'NDHS 2018 FR359 Table 3.1, men 15-49':<44}"
          f"{NDHS_2018_MEN['Christian']:10.1%}{NDHS_2018_MEN['Islam']:9.1%}")
    print(f"    {'Pew Research Center, 2020 estimate':<44}{PEW_2020['Christian']:10.1%}"
          f"{PEW_2020['Muslim']:9.1%}")
    others = [NDHS_2018_WOMEN["Islam"], NDHS_2018_MEN["Islam"], PEW_2020["Muslim"]]
    lo, hi = min([mus] + others), max([mus] + others)
    print(f"    a {100 * (hi - lo):.1f}-point spread on the Muslim share, which over "
          f"{CODPS_2022 / 1e6:.0f} million people is {(hi - lo) * CODPS_2022 / 1e6:.0f} "
          "million of them")
    print(f"    of the {sum(1 for v in [mus] + others if v > 0.5)} of these {1 + len(others)} "
          "that are Muslim-majority, this map is "
          + ("one" if mus > 0.5 else "not one"))


def check_sharia(share, nm):
    """The twelve Sharia states, which is the only outside evidence this country has.

    Nothing about religion counts in Nigeria is published, so there is no table to validate
    against. What there is instead is a legal record: twelve states whose legislatures extended
    the Sharia penal code between 1999 and 2001 and twenty-five that did not.

    THE TWO LEGS ARE WORTH DIFFERENT AMOUNTS AND THIS SAYS SO. The first — every one of the
    twelve comes out Muslim-majority — is genuine evidence about the geography: the survey was
    never told which twelve they are, and a random assignment of the fifteen Muslim-majority
    labels it produces would cover a named twelve about once in four million. The second is a
    median-to-median gap and it is **a smoke test on the decode, not corroboration of a fine
    geography**: a join that had sent the north's respondents south would fail it by a mile,
    and nothing subtler would.

    THE IMPLICATION RUNS ONE WAY and the first leg is written that way. Adopting the Sharia
    penal code takes a Muslim majority; having a Muslim majority does not make a state adopt
    it, and the states that show it here are the well-documented mixed ones, Adamawa on the
    Cameroon border and Kwara and Oyo where Yoruba Muslims are the larger community.
    """
    muslim = share["Muslim"].rename(index=nm)
    christian = share["Christian"].rename(index=nm)
    majority = sorted(muslim.index[muslim > christian])

    print("\n  the only outside evidence — the twelve states that extended the Sharia penal "
          "code, 1999 to 2001:")
    print(f"    {', '.join(SHARIA_STATES)}")
    print(f"    the survey's Muslim-majority states ({len(majority)}): {', '.join(majority)}")

    missing = [s for s in SHARIA_STATES if s not in majority]
    if missing:
        raise SystemExit(
            f"{missing} extended the Sharia penal code and the survey does not make them "
            "Muslim-majority. That is the only outside check this country has, and it failing "
            "means the decode or the pool has changed; nothing should be drawn until someone "
            "reads why.")
    extra = [s for s in majority if s not in SHARIA_STATES]
    print(f"    Muslim-majority without having adopted it: {extra or 'none'} — expected, and "
          "not asserted; see the docstring")

    within = muslim.reindex(SHARIA_STATES).sort_values()
    print(f"    the twelve, least Muslim first: "
          + ", ".join(f"{s} {v:.0%}" for s, v in within.items()))
    rest = muslim.drop(index=SHARIA_STATES)
    gap = float(within.median()) - float(rest.median())
    print(f"    median Muslim share: {within.median():.1%} across the twelve against "
          f"{rest.median():.1%} across the other {len(rest)} — a {100 * gap:.0f}-point gap "
          f"against a {100 * SHARIA_MEDIAN_GAP:.0f}-point bar. This leg is a smoke test on "
          "the decode and not corroboration of the fine geography; see the docstring.")
    if gap < SHARIA_MEDIAN_GAP:
        raise SystemExit(
            f"the twelve states that adopted the Sharia penal code are only {100 * gap:.0f} "
            f"points more Muslim at the median than the other {len(rest)}, against a bar of "
            f"{100 * SHARIA_MEDIAN_GAP:.0f}. That separation is the second half of the only "
            "outside check this country has; read why before drawing.")


def main():
    if "--fetch" in sys.argv:
        ab.fetch()

    print("=== Afrobarometer, Nigeria ===")
    # regroup=True: the answers are grouped to the five below, so the one-answer-two-spellings
    # guard runs on the GROUPED column instead of the raw one.
    raw = ab.load(COUNTRY, expect_rounds=ROUNDS, regroup=True)
    print(f"\n  pooled: {len(raw):,} respondents with a religion answer, "
          f"{raw['category'].nunique()} distinct labels over six rounds")

    # ---- the raw answers, printed before anything is grouped ----
    ct = pd.crosstab(raw["category"], raw["round"])
    ct["all"] = ct.sum(axis=1)
    print("\n  every answer as it arrives (this is what GROUP collapses):")
    print(ct.sort_values("all", ascending=False).to_string(max_colwidth=44))

    # ---- how much of `Christian only` is probing, which is why the grouping happens ----
    conly = raw[raw["category"].map(key) == "christian only"]
    by_round = (conly.groupby("round")["w"].sum() / raw.groupby("round")["w"].sum())
    print("\n  share answering `Christian only` rather than naming a denomination, by round:")
    for r, v in by_round.items():
        print(f"    R{r}  {v:6.1%}")
    print(f"    range {by_round.min():.1%} to {by_round.max():.1%} — the fieldwork, not the "
          "country. See the docstring.")
    report_card(set(raw["category"].map(key)))
    if by_round.max() - by_round.min() < 0.15:
        raise SystemExit(
            "the `Christian only` share no longer swings between rounds, so the argument in "
            "the docstring for collapsing the denominations no longer holds. Re-read it and "
            "decide deliberately rather than leaving this file as it is.")

    # ---- group ----
    k = raw["category"].map(key)
    unmapped = sorted(set(k) - set(GROUP))
    if unmapped:
        raise SystemExit(f"answers with no category: {unmapped}. An answer that falls through "
                         "here is silently dropped — add it to GROUP deliberately, with the "
                         "reason.")
    stale = sorted(set(GROUP) - set(k))
    if stale:
        print(f"  GROUP covers answers this pool no longer has: {stale}")
    df = raw.copy()
    df["raw_category"] = raw["category"]
    df["category"] = k.map(GROUP)
    ab.assert_one_wording(df, COUNTRY)
    if sorted(set(df["category"])) != sorted(CATEGORIES):
        raise SystemExit(f"grouped to {sorted(set(df['category']))}, which is not the five in "
                         f"CATEGORIES: {sorted(CATEGORIES)}")

    # ---- harmonise the state labels BEFORE any test is run on them ----
    ck = df["geo_raw"].map(ckey)
    df["state"] = ck.map(NORM)
    print(f"\n  {df['geo_raw'].nunique()} distinct REGION labels over the six rounds -> "
          f"{df['state'].nunique()} states")
    bad = sorted(df.loc[df["state"].isna(), "geo_raw"].astype(str).unique())
    if bad:
        raise SystemExit(f"REGION labels with no state: {bad}. A label that falls through here "
                         "is silently dropped and the split-half is then run on fewer units "
                         "than its bar assumes — add it to NORM deliberately.")
    dup = df.groupby(["round", "state"])["geo_raw"].nunique()
    if (dup > 1).any():
        raise SystemExit(f"one round uses two labels for the same state: "
                         f"{dup[dup > 1].to_dict()}")
    per_round = df.groupby("round")["state"].nunique()
    print("    states present per round: "
          + ", ".join(f"R{r} {n}" for r, n in per_round.items()))
    absent = {r: sorted(set(NORM.values()) - set(g["state"]))
              for r, g in df.groupby("round") if g["state"].nunique() != N_UNITS}
    if absent:
        print(f"    not sampled in a round: {absent} — the north-east during the insurgency; "
              "see the docstring")

    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    if len(lut) != N_UNITS:
        raise SystemExit(f"{LOOKUP} has {len(lut)} states, expected {N_UNITS} — re-run "
                         "sources/ng_geo.py")
    names = dict(zip(lut["name"], lut["geo_id"]))
    if sorted(set(df["state"])) != sorted(lut["name"]):
        raise SystemExit("the harmonised states are not the 37 in ng_lookup.csv: "
                         f"{sorted(set(df['state']) ^ set(lut['name']))}")
    df["geo_id"] = df["state"].map(names)
    units = sorted(lut["geo_id"])
    pop = lut.set_index("geo_id")["pop"].astype(float)
    if int(pop.sum()) != CODPS_2022:
        raise SystemExit(f"ng_lookup.csv sums to {int(pop.sum()):,}, not {CODPS_2022:,}")

    # ---- the decode, tested without touching the religion column ----
    ab.held_out(df, pop, COUNTRY, pop_source="COD-PS 2022")

    # ---- which categories carry their own geography ----
    nat = ab.national(df)
    print("\n  the survey's national shares, pooled over six rounds:")
    for c in sorted(CATEGORIES, key=lambda c: -float(nat.get(c, 0.0))):
        print(f"    {c:<30}{float(nat.get(c, 0.0)):8.3%}")
    large = ab.stability(df, nat, N_UNITS)
    small = [c for c in nat.index if c not in large]
    if sorted(large) != sorted(CARRIES):
        raise SystemExit(
            f"the split-half now selects {sorted(large)}, not {sorted(CARRIES)}. That is a "
            "change in what this country claims to know, not a bug — read the numbers above, "
            "then update CARRIES and the docstring deliberately.")

    # ---- the Sharia witness, on the survey's own shares before any population enters ----
    by_unit = df.groupby(["geo_id", "category"])["w"].sum().unstack(fill_value=0.0)
    for c in CATEGORIES:
        if c not in by_unit.columns:
            by_unit[c] = 0.0
    nm = dict(zip(lut["geo_id"], lut["name"]))
    check_sharia(by_unit.div(by_unit.sum(axis=1), axis=0), nm)

    # ---- seed, then fit to both margins ----
    #
    # NOT `ab.build`, and the reason is the same one `sources/lr.py` gives. That function
    # divides a state's RESIDUAL among the tail categories, and seven states here have no
    # residual at all: Akwa Ibom, Gombe, Kano, Kebbi, Kogi, Kwara and Nasarawa, in every one
    # of which all the pooled respondents answered Christian or Muslim and there is therefore
    # no room to put the traditionalists in. What they share is the lopsided answer and nothing
    # else — Kano's 736 respondents are the second largest sample in the country.
    #
    # SO THE TAIL IS PUT IN ADDITIVELY RATHER THAN OUT OF A RESIDUAL, and the two categories
    # that earned a geography keep their measured RATIO to each other in every state. Nothing
    # is fitted, so the national level falls out of the state pattern and the state
    # populations, which is the block below.
    #
    # An earlier version ran an IPF against the survey's own pooled national shares, on
    # `sources/lr.py`'s pattern, and it was wrong in a way worth recording: with COD-PS as the
    # row margin, forcing the columns back to the survey's own national total UNDOES the
    # population reweighting the row margin just did. The pool's state mix is not COD-PS's, so
    # those are two different weightings of the same states and the fit bends every state's
    # measured share to reconcile them.
    print("\n  each state's measured mix x its COD-PS population:")
    unit_share = by_unit.div(by_unit.sum(axis=1), axis=0)
    tail_total = float(sum(nat.get(c, 0.0) for c in small))
    ratio = unit_share[large].div(unit_share[large].sum(axis=1), axis=0)
    if ratio.isna().any().any():
        raise SystemExit("a state has no Christian and no Muslim respondents, so there is no "
                         "ratio to draw it on")
    frame = pd.DataFrame(index=units, columns=CATEGORIES, dtype=float)
    basis = {}
    for c in CATEGORIES:
        if c in large:
            frame[c] = (ratio[c] * (1.0 - tail_total)).reindex(units).to_numpy()
            note = "the state's own measured share"
        else:
            frame[c] = float(nat.get(c, 0.0))
            note = "no state geography of its own, so the national share"
        for u in units:
            basis[(u, c)] = note
    if (frame.sum(axis=1) - 1.0).abs().max() > 1e-12:
        raise SystemExit("a state's five shares do not sum to 1")
    zero = [(nm[u], c) for u in units for c in CATEGORIES if frame.loc[u, c] <= 0]
    if zero:
        # §3.5 drops rather than invents: a state where nobody answered Muslim is drawn with
        # none, and `note_public` says to read that as the survey having found none.
        print(f"    drawn at zero, because no pooled respondent there gave that answer "
              f"(§3.5 drops rather than invents): {zero}")
    counts = round_within_rows(frame.mul(pop.reindex(units), axis=0))

    if not (counts.sum(axis=1) == pop.reindex(units).round().astype("int64")).all():
        raise SystemExit("a state's drawn total is not its COD-PS population")

    # ---- THE RECOMPOSITION, WHICH IS THE ONE PLACE THIS COUNTRY'S NATIONAL NUMBER MOVES ----
    drawn_nat = counts.sum(axis=0) / counts.sum().sum()
    print("\n  the survey's own pooled national shares against the same state shares "
          "reweighted by COD-PS, which is what is drawn:")
    print(f"    {'category':<30}{'survey':>10}{'drawn':>10}{'shift':>9}")
    for c in CATEGORIES:
        s, d = float(nat.get(c, 0.0)), float(drawn_nat[c])
        print(f"    {c:<30}{s:9.2%}{d:10.2%}{(d - s) * 100:+8.2f}")
    print(f"    the two differ because the pool's state mix is not COD-PS's: see "
          f"`held_out` above, and R6's missing north-east")
    check_level(nat, drawn_nat)

    # ---- write ----
    n_by = df.groupby("geo_id").size()
    out = counts.stack().rename("count").reset_index()
    out.columns = ["geo_id", "source_category", "count"]
    out["geo_level"] = "state"
    out["geo_name"] = out["geo_id"].map(nm)
    out["basis"] = "self_id"
    out["year"] = YEARS
    out["source_id"] = SOURCE_ID
    out["note"] = out.apply(
        lambda r: ("COD-PS 2022 state population composed with the state's own mix from "
                   f"Afrobarometer rounds 4-9 pooled (n={int(n_by[r.geo_id])} here); "
                   f"{basis[(r.geo_id, r.source_category)]}"),
        axis=1)

    total = int(out["count"].sum())
    if total != CODPS_2022:
        raise SystemExit(f"drawn {total:,} against COD-PS {CODPS_2022:,}")

    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, "
          f"{out['source_category'].nunique()} categories, {out['geo_id'].nunique()} states)")

    # ---- what the file now says, which is what note_public has to reproduce ----
    drawn = out.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print("\n  national, as drawn:")
    for cat, n in drawn.items():
        print(f"    {n / total * 100:6.2f}%  {cat}  ({n:,})")
    share = counts.div(counts.sum(axis=1), axis=0)
    report_zones(share, pop, n_by, nm)
    m = counts["Muslim"]
    c = counts["Christian"]
    print(f"\n  most Muslims: {nm[m.idxmax()]} at {int(m.max()):,}; "
          f"highest Muslim share: {nm[share['Muslim'].idxmax()]} at "
          f"{share['Muslim'].max():.1%}")
    print(f"  most Christians: {nm[c.idxmax()]} at {int(c.max()):,}; "
          f"highest Christian share: {nm[share['Christian'].idxmax()]} at "
          f"{share['Christian'].max():.1%}")
    print(f"  thinnest state {nm[n_by.idxmin()]} at n={int(n_by.min())}; "
          f"median n={int(n_by.median())}, total n={int(n_by.sum()):,}")


if __name__ == "__main__":
    main()
