"""Iraq — Sunni, Shia and undifferentiated Muslims by governorate, from the pooled Arab Barometer.

Reads data/raw/arabbarometer/*.sav and writes data/normalized/iq.csv.
`sources/iq.md` has the acquisition route, the state route that was closed first, the terms and
the checks in prose. `sources/arabbarometer.py` holds the construction. `sources.md` §11af
assesses the source across the Arab world, §9bz is Egypt and §9cq is Jordan, and §11al is
Lebanon, which is the country this one has to be told apart from.

**THIS IS THE FIRST SUNNI/SHIA GEOGRAPHY ON THIS MAP.** Two other countries carry the two
nodes at all — Russia, from Sreda's Arena, and Bulgaria, whose census prints the split — and
neither is a country where the division organises anything. Iraq's does.

## THE 2024 CENSUS COUNTED IRAQ AND DID NOT ASK THIS, AND THE OFFICE HAS PUBLISHED NOTHING

§11r closed Iraq on the state: the census enumerated 20-21 November 2024 and returned
46,118,793 people, the first full count since 1987, and its published tables are governorate
crossed with urban/rural, sex and age and nothing else. The reporting on whether religion or
sect was on the form conflicts and does not need resolving, because nothing was published
either way. Iraq is also absent from the UNSD Demographic Yearbook's religion table. The last
Iraqi census to publish religion at all was 1987.

## WHAT IS DRAWN, AND THE UNDIFFERENTIATED QUARTER IS DRAWN AS ITSELF

Five answers reach the map: **Shia**, **Sunni**, **Just a Muslim**, **Christian** and the two
residuals. `Just a Muslim` is `Q1012A`'s own wording for a Muslim who declines to place
himself in a branch, it is **22.8% of the country weighted**, and it is drawn on the bare
`islam` node rather than being shared out between Sunni and Shia. Splitting it would be
inventing a magnitude the source does not publish (§14.4 rule 1) and would do it on the one
quantity in this file that is most obviously a fact about the interview: the undifferentiated
share runs **17.0% in wave V, 37.8% in VI-3, 25.2% in VII and 19.7% in VIII**, a factor of
2.2 across four fieldworks in six years. Russia is the precedent and it is exact —
`branches.py`'s note on `islam.shia` records that 4.66% of Russia answered *"I profess Islam,
but am neither Sunni nor Shia"* and that those people stay on the parent.

## THE SECT ITEM IS REFUSED FOR THE MAGHREB AND IS THE WHOLE POINT HERE

§11af tested `Q1012A` across Morocco, Algeria, Tunisia, Libya and Sudan and rejected it:
*"just a Muslim"* is the modal answer in all five (44.9% in Morocco, 81.8% in Sudan) and
wholly-Maliki Morocco returns 16.2% Maliki, so there it records which label a person
volunteers rather than anything about the country. It left open whether the same variable
behaves differently *"in Iraq and Lebanon, where sect is a salient public identity rather than
an unmarked default"*. **In Iraq it does**, and three measurements say so rather than one
argument:

  * the undifferentiated share is **22.8%**, against 44.9-81.8% across North Africa;
  * of the Iraqis who do name a branch, **60.9% say Shia**, which is the figure every
    published estimate of Iraq gives (the CIA World Factbook's band is 61-64% Shia and 29-34%
    Sunni of the whole population, and Iraq's own state publishes none);
  * the geography reproduces the one everybody already knows, without being told it — see
    `SHIA_SOUTH` and `SUNNI_NORTHWEST` below, which are asserted rather than admired.

## AND LEBANON IS WHY THE SAME PAGE CANNOT BE READ TWICE THE SAME WAY

Arab Barometer's LEBANESE sample is a fixed sect-by-governorate quota and §14.16's split-half
passes it at its strongest (§11al, `sources/lb.md`). Iraq is not: `ab.assert_not_quota` runs
before `ab.stability` on every category in this file, and Iraq's worst wave pair has **no
exact agreement outside the degenerate cells at all**. That is not a reassurance to be
repeated from the queue row, it is a check this build runs.

## THE FOUR WAVES ARE THE FOUR THAT ASKED, AND WAVES II AND III ARE LEFT OUT ON THE CARD

The files offer Iraq a religion answer in six waves. **Only V, VI-3, VII and VIII carry the
sect follow-up**: wave II has no `q1012a` column at all and wave III has one that is entirely
empty for Iraq. Pooling them would put 2,449 Iraqi Muslims from 2011 and 2013 into the
undifferentiated bucket **because of which questionnaire they were handed**, which is the
module docstring's card rule in its purest form, and it would do it to the single most
instrument-sensitive quantity in the file. They are named in `omit=` rather than dropped in
silence.

Usage:
    python sources/iq.py --fetch    download the Arab Barometer waves (~46 MB of zips)
    python sources/iq.py            rebuild data/normalized/iq.csv
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
from scipy import stats

import arabbarometer as ab

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOOKUP = os.path.join(ROOT, "data", "geo", "iq", "iq_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "iq.csv")

COUNTRY = "Iraq"

# The pool: the four waves whose card carries the sect follow-up. See the docstring.
WAVES = ["V", "VI-3", "VII", "VIII"]
OMIT = {
    "II": ("2011, 1,234 Iraqis, and the file has no `q1012a` column at all — the sect "
           "follow-up was not asked. Pooling it would put every one of its Muslims into the "
           "undifferentiated bucket because of the questionnaire they were handed."),
    "III": ("2013, 1,215 Iraqis. The file HAS a `q1012a` and it is empty for every Iraqi in "
            "it, so the item was fielded elsewhere and not here. Same consequence as wave "
            "II."),
}
SOURCE_ID = "iq_arabbarometer_2018_2024"
# Arab Barometer's own wave dates: V 2018-2019, VI part 3 March to April 2021, VII October
# 2021 to July 2022, VIII September 2023 to July 2024. The .sav files carry no interview date
# for Iraq, so this is the wave range and not a field-date range.
YEARS = "2018-2024"

# The units the split-half is run on. All eighteen governorates appear in both wave halves —
# wave V misses Muthanna and Dohuk and wave VIII misses Dohuk, but VI-3 covers the early half
# and VII the late one — so the overlap is the full eighteen and `ab.stability` asserts it
# before believing any correlation.
N_UNITS_BOTH_HALVES = 18

# The sanity band on the Shia share OF THE MUSLIMS WHO NAME A BRANCH, recorded before the
# build ran. It is the one quantity here with a published external comparison: the CIA World
# Factbook puts Iraq at 61-64% Shia and 29-34% Sunni, which is 64-68% Shia of the named, and
# the range of serious estimates runs from about 55% to about 70%. The band is deliberately
# wider than that, because it is a guard against a re-release re-levelling the pool rather
# than a test of Iraq's composition.
SHIA_OF_NAMED_BAND = (0.50, 0.78)

# What is drawn on its own governorate shares, asserted so a change in the data is a failure
# here rather than a silent re-drawing of the country. Set from the build's own output on
# 2026-09-09 and read against the printed table before being written down.
CARRIES = ["Just a Muslim", "Shia", "Sunni"]

# ---------------------------------------------------------------------------------------
# THE CARD, AND THE THREE READINGS OF IT THIS FILE MAKES
# ---------------------------------------------------------------------------------------
# `Q1012A` (waves V and VI-3) and `Q1012A_MUSLIM` (VII and VIII) are the same question,
# *"What is your religious denomination?"*, asked of everyone who said Muslim. Its card is not
# the same card four times, which matters twice below.
#
# 1. THE MADHHAB ANSWERS ARE FOLDED INTO THEIR BRANCH AND ARE NOT DRAWN AS SCHOOLS.
#
# The card offers Sunni, Shia and Just a Muslim beside Hanbali, Shafi'i, Maliki, Ja'fari,
# Druze, Ibadi and Ahmadiyya. A respondent who answers `Shafi'i` has answered `Sunni` more
# precisely and a respondent who answers `Ja'fari` has answered `Shia` more precisely; there
# is no reading on which a Shafi'i is not a Sunni. They are folded here rather than mapped
# separately in `taxonomy/iq2024.py`, and the reason is not tidiness:
#
#   * `taxonomy/branches.py` HAS the four Sunni schools and the Ja'fari, added with Türkiye,
#     so the nodes exist and the fold is not forced by the tree;
#   * **this instrument does not measure a madhhab.** Shafi'i runs 0.1%, 0.2%, 1.7%, 2.6%
#     across the four waves and Ja'fari 1.6%, 2.8%, 1.9%, 4.2% — a factor of 15 and of 2.6
#     between fieldworks. §11af rejected exactly this column as a madhhab layer for the
#     Maghreb; Türkiye's schools come from a state survey that asks the madhhab outright and
#     does not put Sunni and Shia on the same card, which is a different instrument;
#   * and left unfolded they are the wrong shape for the machinery. Iraq's Shafi'is are
#     overwhelmingly Kurds, five of them are in the early wave half, so `ab.stability` would
#     rank them on noise, fail them, and `ab.build` would then spread Kurdish Shafi'is across
#     Basra at the national rate. Folded, they are Sunnis where they were counted.
#
# 2. `Alawi` IS THE ONE ENTRY HERE THAT IS A JUDGEMENT AND IT IS TWO PEOPLE.
#
# Two respondents in 8,345, 0.019% weighted. Iraq has no Alawite community of any size; in an
# Iraqi Shia setting `علوي` is ordinarily a claim of descent from Ali rather than a claim to
# the Syrian sect. It goes to Shia. The alternative was to leave it as its own answer, where
# it falls under the 1% eligibility floor and gets spread over the country as roughly 8,600
# Alawites who are not there. Named here so that a reader who disagrees can find it.
SECT_FOLD = {
    "Shafi'i": "Sunni",       # the school of the Kurds and of most of Nineveh
    "Hanbali": "Sunni",
    "Maliki": "Sunni",
    "Ja'fari": "Shia",        # the Twelver school; naming it is naming Twelver Shiism
    "Alawi": "Shia",          # two respondents, see above
}

# Two spellings of one answer, and one of them is a typographical slip in the questionnaire.
# Applied to the RAW label before anything is folded or composed; `ab.assert_one_wording` is
# run again afterwards on the composed category.
#
#   * the apostrophe. Waves V, VII and VIII print `Ja’fari` and `Shafi’i` with U+2019 and wave
#     VI-3 prints `Ja'fari` with the ASCII apostrophe. `ab.fold` normalises NFKC, which does
#     not touch the curly quote, so without this the pool carries each answer twice with every
#     total still adding up.
#   * `Shafi'i'` is wave VI-3's own trailing-apostrophe typo for the same box, three people.
#   * `Malki` is the survey's spelling of Maliki, in waves VI-3 and VIII.
#   * `refused` (wave V) and `Refused to answer` (VI-3, VII, VIII) are one interviewer-coded
#     box, which is Jordan's finding (§9cq) on the same survey.
RECODE = {
    "Ja’fari": "Ja'fari",
    "Shafi’i": "Shafi'i",
    "Shafi'i'": "Shafi'i",
    "Malki": "Maliki",
    "refused": "Refused to answer",
}

# Answers that leave the universe rather than being drawn.
#
# **THE SECT REFUSALS DO NOT LEAVE IT, AND THAT IS THE DIFFERENCE FROM JORDAN.** Jordan's
# eleven refusals were refusals of the RELIGION question, so nothing was known about them.
# Here the refusal is on the follow-up: the respondent has already said Muslim and then
# declined to name a branch, which is 83 people who are Muslims of unstated denomination and
# not people of unknown religion. They are composed into `Muslim, denomination not given`
# below and drawn on `islam`, alongside `Just a Muslim`.
DROPPED = {
    "Atheist": ("2 respondents, both in wave V, which is the only one of the four whose card "
                "offers the box at all. A share pooled over four waves for an option that was "
                "on one of the four cards measures which questionnaire was used; Egypt "
                "(§9bz) dropped its two for the same reason."),
    "Refused to answer": ("6 respondents in wave VI-3 who refused the RELIGION question "
                          "itself, so nothing is known about them. A refusal is not a "
                          "religion."),
}

# The composed `category`, where the two columns collide or where neither wording will do.
# `Other` is a box on BOTH questionnaires — a religion that is neither Islam nor Christianity,
# and a Muslim denomination that is none of the ones offered — and they are different answers
# from different people. Left as they are, the two merge into one category with the totals
# still adding up, which is `assert_one_wording`'s failure one column across and which nothing
# would print a warning about, because the two spellings are identical.
COMPOSED = {
    "sect_other": "Muslim, other denomination",
    "sect_none": "Muslim, denomination not given",
    "religion_other": "Other religion",
}

# Every `Q1` label the four Iraqi waves use -> the governorate p-code. Keys are the label
# folded by `key()`: ordinal prefix stripped, lower-cased, Arabic orthography normalised,
# everything but letters removed.
#
# Three of these are the mechanism rather than untidiness:
#   * `Diwaniyah` is AL-QADISIYYAH, named for its capital. Wave VI-3 alone uses the city and
#     the other three use the governorate. A transliteration join misses it entirely and there
#     is no spelling of Qadisiyah that reaches it.
#   * `Dhi War` is wave V's typo for Dhi Qar, 163 respondents; `Muthana`, `Missan`, `Babel`,
#     `Ninewa` and `Sulaymaniya` are ordinary romanisation drift. All are transcribed, never
#     repaired.
#   * `Salahaddin` is Salah al-Din with the article run into the name, as `Azurqa` was in
#     Jordan.
NORM = {
    "baghdad": "IQG08",
    "salahaddin": "IQG16", "salahaldin": "IQG16", "salahalden": "IQG16",
    "diyala": "IQG10", "diala": "IQG10",
    "wasit": "IQG18", "wassit": "IQG18",
    "maysan": "IQG14", "missan": "IQG14",
    "basra": "IQG02", "basrah": "IQG02", "albasrah": "IQG02",
    "dhiqar": "IQG17", "thiqar": "IQG17", "dhiwar": "IQG17",
    "muthanna": "IQG03", "muthana": "IQG03", "almuthanna": "IQG03",
    "qadisiyah": "IQG05", "qadisiya": "IQG05", "diwaniyah": "IQG05",
    "babylon": "IQG07", "babel": "IQG07", "babil": "IQG07",
    "karbala": "IQG12", "kerbela": "IQG12",
    "najaf": "IQG04", "alnajaf": "IQG04",
    "anbar": "IQG01", "alanbar": "IQG01",
    "nineveh": "IQG15", "ninewa": "IQG15", "ninevah": "IQG15",
    "dohuk": "IQG09", "duhok": "IQG09", "dohouk": "IQG09",
    "erbil": "IQG11", "irbil": "IQG11",
    "kirkuk": "IQG13",
    "sulaymaniyah": "IQG06", "sulaymaniya": "IQG06", "sulaimaniya": "IQG06",
}

# THE `Q1` CODE MEANS THE SAME THING IN ALL FOUR WAVES, WHICH IS NOT TRUE OF THIS SURVEY
# ANYWHERE ELSE. Jordan's code means three different things across nine waves and is therefore
# only a witness (§9cq); Iraq's is `70000 + n` in waves V, VII and VIII and `7000 + n` in wave
# VI-3, with the same `n` throughout. It is still not used as the pooling key — the label is,
# because that is what the module does everywhere — and it is used as the check on the label
# harmonisation instead. A permutation of `NORM` would preserve every total and break this on
# thousands of rows at once, which is exactly what
# `[[reference_name_join_wrong_neighbour]]` asks for.
AB_ORDER = ["IQG08", "IQG16", "IQG10", "IQG18", "IQG14", "IQG02", "IQG17", "IQG03", "IQG05",
            "IQG07", "IQG12", "IQG04", "IQG01", "IQG15", "IQG09", "IQG11", "IQG13", "IQG06"]
CODE_SYSTEMS = {
    "70000+n": (["V", "VII", "VIII"], {70000 + i + 1: g for i, g in enumerate(AB_ORDER)}),
    "7000+n": (["VI-3"], {7000 + i + 1: g for i, g in enumerate(AB_ORDER)}),
}

# THE EXTERNAL CHECK ON THE GEOGRAPHY, and it is the strongest one available for a country
# whose state publishes nothing. Iraq's sectarian map is not in dispute and has not been for a
# century: the mid-Euphrates and the south are Shia, the western desert and the Kurdish north
# are Sunni, and Baghdad, Diyala and Kirkuk are the mixed middle. The survey is never told
# this. What is asserted is that it reproduces it — every southern governorate majority Shia
# among the branch-namers, every north-western one majority Sunni — which a survey measuring
# something other than sect could not do.
SHIA_SOUTH = ["IQG02", "IQG17", "IQG14", "IQG03", "IQG05", "IQG04", "IQG12", "IQG07", "IQG18"]
SUNNI_NORTHWEST = ["IQG01", "IQG16", "IQG15", "IQG09", "IQG11", "IQG06"]
MIXED_MIDDLE = ["IQG08", "IQG10", "IQG13"]

_ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ـ": ""})
_MARKS = re.compile(r"[ً-ْٰ]")
_ORDINAL = re.compile(r"^\s*\d+\s*[.)]\s*")
_KEEP = re.compile(r"[^a-zء-ي]")


def key(s):
    """Fold a `Q1` label to something two waves' spellings of it share.

    Latin and Arabic in one function, and Arabic only because it costs nothing: all four Iraqi
    waves print their governorates in English, but wave VII prints Jordan's in Arabic on the
    same variable, so a re-release could. Arabic is folded for the alef forms, final ya, ta
    marbuta, the tatweel and the diacritics. Spaces and hyphens go too, which is what makes
    `Salah al-Din`, `Salahaddin` and `Al-Basrah` reachable from one key.
    """
    s = unicodedata.normalize("NFKC", str(s)).strip()
    s = _ORDINAL.sub("", s).lower()
    s = _MARKS.sub("", s).translate(_ALEF)
    return _KEEP.sub("", s)


def compose(df):
    """One `category` per respondent, from the religion answer and the sect follow-up.

    The rule, in order: a non-Muslim keeps the religion answer; a Muslim takes the sect
    answer; a Muslim whose sect answer is a non-answer or is missing becomes `Muslim,
    denomination not given`.

    Two guards, because both failures would be invisible:

      * **a Muslim must not carry a Christian denomination.** Waves V and VI-3 ask `Q1012A` of
        everybody and put Maronite, Orthodox, Catholic and Armenian on the same card as Sunni
        and Shia, so the column holds four Christians' denominations in wave V. They are
        removed by the Muslim restriction and this asserts that the restriction worked.
      * **every Muslim must end up somewhere.** A silently dropped Muslim would shrink the
        denominator and re-level every share in the country.
    """
    rel = df["category"].astype(str).str.strip()
    sect = df["sect"].astype(str).str.strip().where(df["sect"].notna())
    muslim = rel.eq("Muslim")

    christian_denoms = {"Maronite", "Orthodox", "Catholic", "Armenian", "Just a Christian",
                        "Sabean Mandean", "Druze"}
    stray = sorted(set(sect[muslim].dropna()) & christian_denoms)
    if stray:
        raise SystemExit(f"respondents who said Muslim carry {stray} on the sect item, which "
                         "means the Muslim restriction is not doing what it says")

    nonanswer = {"Refused to answer", "don't know", "Don't know", "The respondent does not "
                 "wish to continue"}
    out = rel.copy()
    out[muslim] = sect[muslim]
    out[muslim & sect.isna()] = COMPOSED["sect_none"]
    out[muslim & sect.isin(nonanswer)] = COMPOSED["sect_none"]
    out[muslim & sect.eq("Other")] = COMPOSED["sect_other"]
    out[~muslim & rel.isin(["Other", "Something else: SPECIFY_______"])] = \
        COMPOSED["religion_other"]

    if out[muslim].isna().any():
        raise SystemExit(f"{int(out[muslim].isna().sum())} Muslims came out of the "
                         "composition with no category")
    print("\n  composed category, by wave:")
    print(pd.crosstab(out, df["wave"]).to_string())
    return out


def code_witness(g):
    """The harmonised names against the `Q1` CODES, which the pool never keys on."""
    print("\n  the names against the codes, in all four waves (the code is not the key):")
    for label, (waves, table) in CODE_SYSTEMS.items():
        sub = g[g["wave"].isin(waves)]
        if sub.empty:
            raise SystemExit(f"CODE_SYSTEMS names waves {waves}, none of which is in the pool")
        codes = pd.to_numeric(sub["geo_code"], errors="coerce")
        expect = codes.map(table)
        unknown = sorted(set(codes[expect.isna()].dropna()))
        if unknown:
            raise SystemExit(f"{label}: codes outside the system: {unknown}")
        bad = sub[expect != sub["geo_id"]]
        print(f"    {label:<10} waves {','.join(waves):<14} {len(sub):>5} respondents, "
              f"{len(bad)} disagreements")
        if len(bad):
            raise SystemExit(
                f"{len(bad)} respondents whose {label} code says one governorate and whose "
                f"label says another: {bad[['wave', 'geo_code', 'geo_raw', 'geo_id']].head(12)}")


def sect_geography(g, names, pop):
    """Does the survey reproduce Iraq's known sectarian map without being told it?"""
    named = g[g["category"].isin(["Shia", "Sunni"])]
    tab = named.assign(shia=named["category"].eq("Shia")).groupby("geo_id")["shia"].agg(
        n="size", shia="sum")
    tab["pct"] = tab["shia"] / tab["n"] * 100
    chi2, p, dof, _ = stats.chi2_contingency(
        np.array([tab["shia"].values, (tab["n"] - tab["shia"]).values]))
    print(f"\n  Shia share of the branch-namers, by governorate (chi-square across the "
          f"{len(tab)}: chi2={chi2:.0f} dof={dof} p={p:.3g}):")
    band = {**{u: "south" for u in SHIA_SOUTH},
            **{u: "north/west" for u in SUNNI_NORTHWEST},
            **{u: "middle" for u in MIXED_MIDDLE}}
    if sorted(band) != sorted(tab.index):
        raise SystemExit(f"the three bands do not partition the eighteen: "
                         f"{sorted(set(band) ^ set(tab.index))}")
    for gid, r in tab.sort_values("pct", ascending=False).iterrows():
        print(f"    {names[gid]:<14}{r['pct']:6.1f}% Shia   n={int(r['n']):>5}   "
              f"{band[gid]}")

    wrong_south = [names[u] for u in SHIA_SOUTH if tab.loc[u, "pct"] <= 50]
    wrong_north = [names[u] for u in SUNNI_NORTHWEST if tab.loc[u, "pct"] >= 50]
    south = tab.loc[SHIA_SOUTH, "shia"].sum() / tab.loc[SHIA_SOUTH, "n"].sum()
    north = tab.loc[SUNNI_NORTHWEST, "shia"].sum() / tab.loc[SUNNI_NORTHWEST, "n"].sum()
    middle = tab.loc[MIXED_MIDDLE, "shia"].sum() / tab.loc[MIXED_MIDDLE, "n"].sum()
    p_band = stats.chi2_contingency(
        pd.crosstab(named["geo_id"].map(band), named["category"]).values)[1]
    print(f"  the south {south * 100:.1f}% Shia, the north and west {north * 100:.1f}%, the "
          f"mixed middle {middle * 100:.1f}%  (p={p_band:.3g})")
    if wrong_south or wrong_north:
        raise SystemExit(
            f"the survey does not reproduce Iraq's sectarian geography: {wrong_south} come "
            f"out Sunni-majority in the south and {wrong_north} Shia-majority in the north "
            "and west. That is the check this country's whole claim rests on, so STOP and "
            "read the table above before changing the bands to fit it.")


def leave_one_out(g, nat, large, names):
    """What the split-half verdict costs if one governorate is removed, for each in turn."""
    import spearman_null

    print("\n  leave-one-out on the split-half, for the answers that carry:")
    waves = sorted(g["wave_no"].unique())
    cut = waves[len(waves) // 2]
    bar, _how = spearman_null.critical_rho(N_UNITS_BOTH_HALVES - 1)
    for c in large:
        rows = []
        for drop in sorted(g["geo_id"].unique()):
            sub = g[g["geo_id"] != drop]
            e, l = sub[sub["wave_no"] < cut], sub[sub["wave_no"] >= cut]

            def share(d):
                return d.groupby("geo_id").apply(
                    lambda x: x.loc[x["category"] == c, "w"].sum() / x["w"].sum(),
                    include_groups=False)

            j = pd.concat([share(e).rename("e"), share(l).rename("l")], axis=1).dropna()
            rows.append((drop, j["e"].corr(j["l"], method="spearman")))
        rows.sort(key=lambda t: t[1])
        worst, lo = rows[0]
        print(f"    {c:<26} {lo:+.3f} without {names[worst]}, {rows[-1][1]:+.3f} without "
              f"{names[rows[-1][0]]}, bar +{bar:.4f} at 17 units")
        if lo < bar:
            print(f"      NOTE: removing {names[worst]} alone takes this under the bar it "
                  "would be judged against at 17 units.")


def main():
    if "--fetch" in sys.argv:
        ab.fetch()
    ab.unzip()

    print("=== Arab Barometer, Iraq ===")
    df = ab.load(COUNTRY, expect_waves=WAVES, waves=WAVES, omit=OMIT,
                 extra={"sect": ("q1012a_muslim", "q1012a")})
    print(f"\n  pooled: {len(df):,} respondents with a religion answer")
    print("  Q1012 by wave:")
    print(pd.crosstab(df["category"], df["wave"]).to_string())
    print("\n  the sect item's RAW answers by wave (before any recode):")
    print(pd.crosstab(df["sect"], df["wave"]).to_string())

    # ---- one spelling per answer, then one branch per madhhab ----
    for raw, replacement in RECODE.items():
        n = int((df["sect"] == raw).sum())
        if not n:
            raise SystemExit(f"RECODE names {raw!r}, which this pool's sect column does not "
                             "contain. Either a re-release has changed the wording or the "
                             "entry was never right; do not delete it without reading what "
                             "it was for.")
        print(f"  recoded {n} sect answer(s) from {raw!r} to {replacement!r}")
        df.loc[df["sect"] == raw, "sect"] = replacement
    for raw, branch in SECT_FOLD.items():
        n = int((df["sect"] == raw).sum())
        if not n:
            raise SystemExit(f"SECT_FOLD names {raw!r}, which this pool does not contain")
        print(f"  folded {n} {raw!r} into {branch!r}")
        df.loc[df["sect"] == raw, "sect"] = branch

    df["category"] = compose(df)
    ab.assert_one_wording(df, COUNTRY)

    # ---- the external cross-check, on the pool as it arrives ----
    named = df["category"].isin(["Shia", "Sunni"])
    shia = df["category"].eq("Shia")
    shia_of_named = df.loc[shia, "w"].sum() / df.loc[named, "w"].sum()
    undiff = df["category"].isin([COMPOSED["sect_none"], COMPOSED["sect_other"],
                                  "Just a Muslim"])
    print(f"\n  of the Iraqis who name a branch, {shia_of_named * 100:.1f}% say Shia "
          f"(n={int(named.sum()):,} weighted)")
    print(f"    against the CIA World Factbook's 61-64% Shia and 29-34% Sunni of the whole "
          "population,\n    which is 64-68% of the named. Iraq's own state publishes "
          "nothing.")
    print(f"  and {df.loc[undiff, 'w'].sum() / df['w'].sum() * 100:.1f}% of the country names "
          "no branch at all, against 44.9-81.8%\n    across the five North African countries "
          "§11af tested this item on and rejected it for.")
    for wv, sub in df.groupby("wave"):
        u = sub["category"].isin([COMPOSED["sect_none"], COMPOSED["sect_other"],
                                  "Just a Muslim"])
        print(f"      wave {wv:<5} {sub.loc[u, 'w'].sum() / sub['w'].sum() * 100:5.1f}% name "
              f"no branch")
    if not SHIA_OF_NAMED_BAND[0] <= shia_of_named <= SHIA_OF_NAMED_BAND[1]:
        raise SystemExit(
            f"the Shia share of the branch-namers is {shia_of_named:.1%}, outside "
            f"{SHIA_OF_NAMED_BAND[0]:.0%}-{SHIA_OF_NAMED_BAND[1]:.0%}. Every published "
            "estimate of Iraq sits inside that; a reading outside it means the instrument or "
            "the pool has changed and the country should not be drawn from it without "
            "re-reading sources/iq.md.")

    # ---- drop the answers that leave the universe ----
    for cat, why in DROPPED.items():
        n = int((df["category"] == cat).sum())
        if not n:
            raise SystemExit(f"DROPPED names {cat!r}, which this pool no longer contains")
        print(f"\n  dropping {n} respondent(s) who answered {cat!r}: {why}")
        df = df[df["category"] != cat]

    # ---- harmonise the governorate labels BEFORE any test is run on them ----
    g = df[df["geo_raw"].notna()].copy()
    g["raw"] = g["geo_raw"].astype(str).str.strip()
    k = g["raw"].map(key)
    g["geo_id"] = k.map(NORM)
    print(f"\n  {g['raw'].nunique()} distinct Q1 labels over the four waves")
    unmapped = sorted(g.loc[g["geo_id"].isna(), "raw"].unique())
    if unmapped:
        raise SystemExit(f"Q1 labels with no governorate: {unmapped} "
                         f"(folded: {sorted({key(u) for u in unmapped})}). A label that falls "
                         "through here is silently dropped and the split-half is then run on "
                         "fewer units than its bar assumes — add it to NORM deliberately.")
    dup = g.groupby(["wave", "geo_id"])["raw"].nunique()
    if (dup > 1).any():
        raise SystemExit(f"one wave uses two labels for the same governorate: "
                         f"{dup[dup > 1].to_dict()}")
    print(f"  usable: {len(g):,} respondents over {g['geo_id'].nunique()} governorates")

    code_witness(g)

    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    if sorted(set(g["geo_id"])) != sorted(lut["geo_id"]):
        raise SystemExit("the harmonised governorates are not the 18 in iq_lookup.csv: "
                         f"{sorted(set(g['geo_id']) ^ set(lut['geo_id']))}")
    names = dict(zip(lut["geo_id"], lut["name"]))
    pop = lut.set_index("geo_id")["pop"]
    units = sorted(lut["geo_id"])
    print(f"  all {len(units)} governorates sampled; the 2024 census counted "
          f"{int(pop.sum()):,} people")

    # ---- the decode, tested without touching the religion column ----
    ab.held_out(g, pop[units], COUNTRY, pop_source="the 2024 census")

    # ---- does the survey reproduce the geography everybody already knows ----
    sect_geography(g, names, pop)

    nat = ab.national(g)
    large = ab.stability(g, nat, N_UNITS_BOTH_HALVES)
    leave_one_out(g, nat, large, names)
    small = [c for c in nat.index if c not in large]
    if sorted(large) != sorted(CARRIES):
        raise SystemExit(
            f"the split-half selects {sorted(large)}, not {sorted(CARRIES)}. That is a change "
            "in what this country claims to know, not a bug — read the numbers above, then "
            "update CARRIES and the docstring deliberately.")

    # ---- build ----
    out = ab.build(g, nat, large, small, pop, units, unit_noun="governorate")
    n_by = g.groupby("geo_id").size()
    out["geo_level"] = "governorate"
    out["geo_name"] = out["geo_id"].map(names)
    out["basis"] = "self_id"
    out["year"] = YEARS
    out["source_id"] = SOURCE_ID
    out["n_gov"] = out["geo_id"].map(n_by)
    out["note"] = out.apply(
        lambda r: (f"Arab Barometer waves V, VI-3, VII and VIII pooled, n={r.n_gov} in this "
                   f"governorate; {r.basis_note} applied to the 2024 census's governorate "
                   "population"),
        axis=1)

    total = int(out["count"].sum())
    if total != int(pop.sum()):
        raise SystemExit(f"drawn {total:,} against the census's {int(pop.sum()):,}")

    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people)")
    wide = out.pivot_table(index="geo_name", columns="source_category", values="count",
                           aggfunc="sum").fillna(0).astype("int64")
    print((wide.div(wide.sum(axis=1), axis=0) * 100).round(1).to_string())


if __name__ == "__main__":
    main()
