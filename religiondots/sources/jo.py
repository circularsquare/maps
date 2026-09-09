"""Jordan — religion by governorate, from the pooled Arab Barometer.

Reads data/raw/arabbarometer/*.sav and writes data/normalized/jo.csv.
`sources/jo.md` has the acquisition route, the state route that was tested first, the terms
and the checks in prose. `sources/arabbarometer.py` holds the construction, which is
`sources/lapop.py`'s pointed at a different survey. `sources.md` §11af assesses the source
across the whole Arab world and §9bz is Egypt, the first country built on it.

**JORDAN HAS ASKED RELIGION IN BOTH OF ITS LAST TWO CENSUSES AND PUBLISHED IT IN NEITHER.**
The Department of Statistics prints its own questionnaires: the 2015 form's person block runs
201-216 and item 208 is `Religion`, `1.mustim 2.chistian 3.other` (DOS's spelling), and the
2004 private household register asks `الديانة` — `1. الإسلام 2. المسيحية 3. أخرى` — in the same
position. Neither census's published table set contains a religion table: 2015 has 133 tables
in ten sections and 2004 has 165 in eight, and religion is in none of them. Jordan is also
absent from the UNSD Demographic Yearbook's religion table. So the answers exist and nobody
outside DOS has seen them.

## What is drawn

Two answers, Muslim and Christian, which is the whole of Jordan's `Q1012` card in every wave
it has been fielded in. See `CARRIES` and the module's own output for which of them carries a
governorate geography; that is decided by the split-half on the day, not asserted here.

## THE UNIVERSE IS NOT THE SAME UNIVERSE IN EVERY WAVE, AND IT IS MEASURED RATHER THAN GUESSED

DOS's 11,937,000 is the **whole resident population** and roughly three in ten of them are not
Jordanian citizens: the 2015 census counted 9,531,712 people of whom 2,918,125 were
non-Jordanian. The Arab Barometer's Jordanian sample is mostly citizens — but **not always**,
and the file says so itself. Wave IV's `q1020jo` records **303 Syrians among its 1,500
respondents**, 20.2%, which is about the non-citizen share of the country; every other wave
returns Jordanian or Palestinian origin for all but a handful.

That wave is therefore the only in-data measurement of whether Jordan's non-citizen quarter
differs, and it does: **0 of 303 Syrian-origin respondents are Christian against 28 of 1,197
Jordanian- and Palestinian-origin ones in the same wave** (2.34%, chi-square p=0.014). It is
re-run on every build below rather than quoted.

**Nothing is corrected for it**, because a correction would be an invented magnitude (§14.4
rule 1) and nothing published gives the religion of Jordan's non-citizens at any geography.
What is done instead is what §3.5 asks: the shares are applied to the whole resident
population, and `note_public` says that the survey behind them is largely a survey of
citizens and that Jordan's non-citizens are measurably less Christian, so the Christian share
drawn here is a ceiling rather than a point estimate.

## WAVE I ASKS THE QUESTION AND CANNOT BE USED, WHICH ARE TWO DIFFERENT FACTS

`ab.load` raises on a wave that has the country's rows and no `Q1012`, and wave I is that wave
for every country including Jordan (1,143 respondents). It is not a re-release that renamed
something and it is not a wave that skipped the question: **wave I is numbered the old way and
its religion item is `q711`**, `1 muslim, 2 christian, 3 sunni muslim (lebanon & bahrain),
4 shiite muslim (lebanon & bahrain), 5 druze (lebanon)`, answered by 1,142 of the 1,143.

It is still left out, and the reason is the geography rather than the decode: **wave I has 181
columns and not one of them is subnational.** There is no `q1`, no governorate, no region, no
district — `country` is the finest unit in the file. A wave with no geography cannot enter a
pool that is cut by governorate however cleanly its religion column decodes. It is worth one
line as an external check and it gets one: unweighted, wave I reads **1.57% Christian**, which
is inside the band the other nine waves occupy.

## WAVE II WAS INVISIBLE TO THIS MODULE'S PREDECESSOR AND IS 1,188 OF THESE RESPONDENTS

Wave II spells its country labels `8. Jordan`, `5. Egypt`, `17. Saudi Arabia` — the numeric
code repeated inside the label — so the exact-match country filter `ab.load` used until
2026-09-08 found no rows and skipped the wave in silence, for every Arab Barometer country
ever built here. `ab.country_key`'s docstring is the write-up. Jordan takes the wave:
`q1012 Religion`, `q1 Province/Governorate/State` over all twelve, a weight, 39 Christians in
1,188. **Egypt does not**, because Egypt is already drawn and moving it is not this file's
call; `sources/eg.py`'s `WAVES` says so at the point where it matters.

## THE ANSWER CODES AND THE ANSWER WORDINGS BOTH MOVE, AND SO DOES THE GOVERNORATE NUMBERING

`Q1012`'s codes are re-used between waves (§9bz), which `ab.load` handles by decoding through
each wave's own labels. Two things then remain for this file:

  * **three spellings of one refusal.** `refused` (wave V), `Refused to answer` (waves VI-2 and
    VII) and wave II's `99999. declined to answer` are the same interviewer-coded box, and
    `RECODE` merges them so `DROPPED` can take them out as one answer. They are eleven people
    and they leave the universe rather than being spread, because a refusal is not a religion.
  * **the governorate CODE means three different things.** Waves II and III number Jordan's
    governorates `3501..3512` in the official order; waves V, VII and VIII use `800` followed
    by Jordan's own governorate number, so `80011` is Amman and `80034` is Aqaba; waves IV,
    VI-1, VI-2 and VI-3 use an arbitrary `8001..8012` in an order that is not the official one
    and is not even the same between IV and VI. **So the code can never be the pooling key** —
    and in the six waves where it does mean something it is a witness on the names instead,
    which is what `CODE_SYSTEMS` below is for. `[[reference_name_join_wrong_neighbour]]` asks
    for exactly this: test the join in code order rather than eyeballing the names.

Usage:
    python sources/jo.py --fetch    download the Arab Barometer waves (~46 MB of zips)
    python sources/jo.py            rebuild data/normalized/jo.csv
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
import spearman_null

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOOKUP = os.path.join(ROOT, "data", "geo", "jo", "jo_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "jo.csv")

COUNTRY = "Jordan"

# The pool. Wave I fields Jordan and is left out on purpose — see the module docstring; it has
# no subnational variable, so `waves=` states the choice instead of letting `ab.load` fail on
# the missing `Q1012` and read as a broken file.
WAVES = ["II", "III", "IV", "V", "VI-1", "VI-2", "VI-3", "VII", "VIII"]
SOURCE_ID = "jo_arabbarometer_2010_2024"
# Arab Barometer's own wave dates: II 2010-2011, III 2012-2014, IV 2016-2017, V 2018-2019,
# VI July 2020 to April 2021, VII October 2021 to July 2022, VIII September 2023 to July 2024.
# The .sav files carry no interview date for Jordan, so this is the wave range and not a
# field-date range, and the vintage is stated at that grain everywhere it appears.
YEARS = "2010-2024"

# The units the split-half is run on. All twelve governorates are sampled in all nine waves,
# so the overlap is the full twelve and the bar is 1.96/sqrt(12-1) = +0.591. `ab.stability`
# asserts the overlap really is twelve before believing any correlation.
N_UNITS_BOTH_HALVES = 12

# The sanity band on the pooled weighted Christian share, RECORDED BEFORE THE BUILD RAN and
# recorded honestly: it was set knowing the per-wave shares from a scratch probe (they run
# 0.34% to 2.12% weighted), so it is a guard against a future re-release rather than a test
# this build passed blind. The published external figures it brackets are the two that exist:
# **2.7% of Jordanians** (Eissa Masarweh, 197,000 resident Jordanian Christians at mid-2021,
# carried forward from the 2004 census's ratio) and, since citizens are about seven in ten
# residents, **about 1.8% of the resident population** that DOS's denominator counts.
CHRISTIAN_BAND = (0.008, 0.030)

# What is drawn on its own governorate shares, asserted so a change in the data is a failure
# here rather than a silent re-drawing of the country.
#
# **BOTH, AND THE TWO ARE THE SAME NUMBER FOR A REASON WORTH KNOWING.** Jordan's card has
# exactly two answers, so a governorate's Muslim share is one minus its Christian share, the
# two rankings are exact reverses, and their Spearman correlations are identical by
# construction. They pass or fail together and it is not a coincidence to be read as
# corroboration. Egypt (§9bz) is the same shape.
#
# **AND THE PASS IS THIN: +0.617 against a bar of +0.504.** `leave_one_out` below prints what
# it costs — without Balqa, the most Christian governorate, it is +0.509. The bar is §14.16's
# and is applied as written; moving it, in either direction, would silently change every other
# country and is not this file's call (AGENT_BRIEF §3).
#
# **THIN AGAINST THE FIXED BAR IS NOT THE SAME AS MARGINAL, AND `ask/007-cr` IS WHY THAT WAS
# WORTH SAYING HERE.** That ask, filed the day before this build, showed `1.96/sqrt(n-1)` is
# not the 0.05-level test its docstring claimed: it is stricter than the exact permutation null
# everywhere, and worst at small n. This build's own run appended the evidence at Jordan's unit
# counts — a **0.023-level** test on twelve units — and Anita ruled on 2026-09-09 to replace
# it. The bar is now `sources/spearman_null.py`'s exact null, **+0.5035 on twelve units**
# against the +0.5910 this file was built under. **Jordan did not move**: +0.617 cleared the
# fixed bar, the exact 95th and the exact 97.5th (+0.5804) alike, and the pass is more
# comfortable on the honest null than on the one originally applied.
# The fragility that remains is entirely Balqa's leverage, which no choice of bar addresses —
# and on the corrected bar `leave_one_out` reads slightly differently, because +0.509 without
# Balqa now clears the twelve-unit bar while still failing the eleven-unit one it should be
# judged against. The warning it prints is unchanged.
#
# What the fragility buys is a sentence
# in `note_public` rather than a different verdict, and three things sit behind the verdict
# that the split-half does not see: the twelve governorates differ at p=2.6e-18, Balqa, Madaba
# and Ajloun together read 3.50% against 1.41% elsewhere at p=4.1e-10, and the ordering the
# survey produces is the one Jordan's own historic Christian geography would predict.
#
# The alternative is not "draw it more carefully". With two answers and neither carrying,
# `ab.build` gives every governorate the national rate, so the map would say 1.41% Christian
# in Balqa and 1.41% in Tafilah — a claim the data contradicts at p=2.6e-18.
CARRIES = ["Christian", "Muslim"]

# Three spellings of one interviewer-coded refusal -> one. This is a reading of the
# questionnaires and `ab.fold` deliberately will not do it for us: see `ab.assert_one_wording`.
# All three sit on code 99/99999 in their own wave's label set, all three are the box the
# interviewer ticks when the respondent will not answer, and none of them is a religion.
# It matches on the RAW label, so wave II's ordinal prefixes are spelled out here in full;
# `ab.fold` strips them when it looks for a collision, but nothing rewrites the data.
RECODE = {
    "refused": "Refused to answer",                     # wave V
    "99999. declined to answer": "Refused to answer",   # wave II
    "1. muslim": "Muslim",                              # wave II
    "2. christian": "Christian",                        # wave II
}

# Answers that leave the universe rather than being drawn.
DROPPED = {
    "Refused to answer": ("11 respondents over three waves. A refusal is not a religion, and "
                          "spreading it at the national rate would draw ~9,600 Jordanians on "
                          "the strength of eleven people who declined to say."),
}

# Every `Q1` label the nine Jordanian waves use -> the governorate. Keys are the label folded
# by `key()` below: ordinal prefix stripped, lower-cased, Arabic orthography normalised.
#
# Two of these are the mechanism rather than untidiness, and both are §9bz's `The Lake` again:
#   * `The capital` and `العاصمة` are AMMAN. The governorate's name is Muhafazat al-Asima,
#     "the Capital Governorate"; DOS itself prints العاصمة in Arabic and `Amman` in English.
#     A transliteration join misses it and a translation join finds it.
#   * `Azurqa` is az-Zarqa with the article run into the name, and `Ajioun` is Ajloun with the
#     l read as an i. Both are the survey's own spelling and are transcribed, never repaired.
NORM = {
    "the capital": "JO11", "capital": "JO11", "amman": "JO11", "العاصمه": "JO11",
    "balqa": "JO12", "al balqa": "JO12", "albalqa": "JO12", "البلقاء": "JO12",
    "zarqa": "JO13", "azzarqa": "JO13", "az zarqa": "JO13", "azurqa": "JO13",
    "الزرقاء": "JO13",
    "madaba": "JO14", "مادبا": "JO14",
    "irbid": "JO21", "اربد": "JO21",
    "mafraq": "JO22", "al mafraq": "JO22", "almafraq": "JO22", "المفرق": "JO22",
    "jerash": "JO23", "jarash": "JO23", "جرش": "JO23",
    "ajloun": "JO24", "ajlun": "JO24", "ajioun": "JO24", "عجلون": "JO24",
    "karak": "JO31", "al karak": "JO31", "alkarak": "JO31", "الكرك": "JO31",
    "tafilah": "JO32", "tafila": "JO32", "tafiela": "JO32", "الطفيله": "JO32",
    "maan": "JO33", "ma an": "JO33", "معان": "JO33",
    "aqaba": "JO34", "al aqaba": "JO34", "العقبه": "JO34",
}

# The two wave-groups whose `Q1` CODE means something, and what it means. Everything else in
# the file uses an arbitrary 1..12 order and is checked only by its label.
#
#   * waves II and III: 3500 + the governorate's RANK in Jordan's official order;
#   * waves V, VII and VIII: 800 + Jordan's own governorate number, which is `geo_id`'s digits.
#
# Both are asserted against the harmonised names on every respondent. A permutation of `NORM`
# would preserve every total and break this on hundreds of rows at once.
OFFICIAL_ORDER = ["JO11", "JO12", "JO13", "JO14", "JO21", "JO22",
                  "JO23", "JO24", "JO31", "JO32", "JO33", "JO34"]
CODE_SYSTEMS = {
    "rank+3500": (["II", "III"],
                  {3500 + i + 1: g for i, g in enumerate(OFFICIAL_ORDER)}),
    "800+governorate number": (["V", "VII", "VIII"],
                               {80000 + int(g[2:]): g for g in OFFICIAL_ORDER}),
}

# Where Jordan's Christians are said to be, used ONLY to name a contrast in the output and
# never to weight anything. Fuheis and Husn are the two best-known Christian towns and sit in
# Balqa and Irbid; Madaba and Ajloun carry the other historic communities.
CHRISTIAN_HEARTLAND = ["JO12", "JO14", "JO24"]

_ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ـ": ""})
_MARKS = re.compile(r"[ً-ْٰ]")
_ORDINAL = re.compile(r"^\s*\d+\s*[.)]\s*")
_KEEP = re.compile(r"[^a-zء-ي ]")


def key(s):
    """Fold a `Q1` label to something two waves' spellings of it share.

    Latin and Arabic in one function because wave VII prints Jordan's governorates in Arabic
    and the other eight print them in English. Arabic is folded for the alef forms, final ya,
    ta marbuta, the tatweel and the diacritics — the marks that differ between two spellings
    of one name and never between two names — which is `sources/eg_geo.py`'s `ar()`.
    """
    s = unicodedata.normalize("NFKC", str(s)).strip()
    s = _ORDINAL.sub("", s).lower()
    s = _MARKS.sub("", s).translate(_ALEF)
    return re.sub(r"\s+", " ", _KEEP.sub(" ", s)).strip()


def main():
    if "--fetch" in sys.argv:
        ab.fetch()
    ab.unzip()

    print("=== Arab Barometer, Jordan ===")
    df = ab.load(COUNTRY, expect_waves=WAVES, waves=WAVES, recode=RECODE)
    print(f"\n  pooled: {len(df):,} respondents with a religion answer")
    print("  answers, by wave (the CARD is not the same card twice):")
    print(pd.crosstab(df["category"], df["wave"]).to_string())

    # ---- the external cross-check, on the pool as it arrives ----
    # Run before anything is dropped or joined, because that is the state a reader can
    # reproduce from the .sav files alone.
    chr_all = df["category"].astype(str).str.strip().str.lower().eq("christian")
    w_share = df.loc[chr_all, "w"].sum() / df["w"].sum()
    print(f"\n  pooled Christian share: {chr_all.mean() * 100:.2f}% unweighted, "
          f"{w_share * 100:.2f}% weighted")
    print("    against the only published external figures: 2.7% of JORDANIANS (Masarweh, "
          "197,000\n    at mid-2021, on the 2004 census's ratio), which is about 1.8% of the "
          "RESIDENT\n    population DOS counts. Jordan's state has published neither.")
    if not CHRISTIAN_BAND[0] <= w_share <= CHRISTIAN_BAND[1]:
        raise SystemExit(
            f"the weighted national Christian share is {w_share:.2%}, outside "
            f"{CHRISTIAN_BAND[0]:.1%}-{CHRISTIAN_BAND[1]:.1%}. That band brackets both "
            "published universes; a reading outside it means the instrument or the pool has "
            "changed and the country should not be drawn from it without re-reading "
            "sources/jo.md.")

    # ---- the figures in note_public that are about the SURVEY and not about the CSV ----
    note_public_figures(df)

    # ---- the universe check: wave IV is the only wave that sampled non-citizens ----
    origins(df)

    # ---- drop the answers that leave the universe ----
    for cat, why in DROPPED.items():
        n = int((df["category"] == cat).sum())
        if not n:
            raise SystemExit(f"DROPPED names {cat!r}, which this pool no longer contains")
        print(f"\n  dropping {n} respondent(s) who answered {cat!r}: {why}")
        refusal_lean(df, cat)
        df = df[df["category"] != cat]

    # ---- harmonise the governorate labels BEFORE any test is run on them ----
    g = df[df["geo_raw"].notna()].copy()
    g["raw"] = g["geo_raw"].astype(str).str.strip()
    k = g["raw"].map(key)
    g["geo_id"] = k.map(NORM)
    print(f"\n  {g['raw'].nunique()} distinct Q1 labels over the nine waves")
    unmapped = sorted(g.loc[g["geo_id"].isna(), "raw"].unique())
    if unmapped:
        raise SystemExit(f"Q1 labels with no governorate: {unmapped} "
                         f"(folded: {sorted({key(u) for u in unmapped})}). A label that falls "
                         "through here is silently dropped and the split-half is then run on "
                         "fewer units than its bar assumes — add it to NORM deliberately.")

    # Two raw labels in the SAME wave collapsing to one governorate would mean the label set is
    # not what it looks like, and the sum would be right while the units were wrong.
    dup = g.groupby(["wave", "geo_id"])["raw"].nunique()
    if (dup > 1).any():
        raise SystemExit(f"one wave uses two labels for the same governorate: "
                         f"{dup[dup > 1].to_dict()}")
    print(f"  usable: {len(g):,} respondents over {g['geo_id'].nunique()} governorates")

    # ---- the names against the CODES, in the six waves whose codes mean something ----
    code_witness(g)

    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    if sorted(set(g["geo_id"])) != sorted(lut["geo_id"]):
        raise SystemExit("the harmonised governorates are not the 12 in jo_lookup.csv: "
                         f"{sorted(set(g['geo_id']) ^ set(lut['geo_id']))}")
    names = dict(zip(lut["geo_id"], lut["name"]))
    pop = lut.set_index("geo_id")["pop"]
    units = sorted(lut["geo_id"])
    print(f"  all {len(units)} governorates sampled; DOS end-2025: {int(pop.sum()):,} people")

    # ---- the decode, tested without touching the religion column ----
    ab.held_out(g, pop[units], COUNTRY, pop_source="DOS end-2025")

    # ---- do the governorates differ at all, and which categories carry a geography ----
    tab = g.assign(chr=g["category"].eq("Christian")).groupby("geo_id")["chr"].agg(
        n="size", chr="sum")
    tab["pct"] = tab["chr"] / tab["n"] * 100
    chi2, p, dof, _ = stats.chi2_contingency(
        np.array([tab["chr"].values, (tab["n"] - tab["chr"]).values]))
    print(f"\n  Christian share by governorate, pooled (chi-square across the {len(tab)}: "
          f"chi2={chi2:.1f} dof={dof} p={p:.3g}):")
    for gid, r in tab.sort_values("pct", ascending=False).iterrows():
        print(f"    {names[gid]:<10}{r['pct']:6.2f}%   n={int(r['n']):>5}  "
              f"{int(r['chr']):>3} Christians")

    heart = g["geo_id"].isin(CHRISTIAN_HEARTLAND)
    t2 = g.assign(chr=g["category"].eq("Christian")).groupby(heart)["chr"].agg(["mean", "size"])
    p_h = stats.chi2_contingency(
        pd.crosstab(heart, g["category"].eq("Christian")).values)[1]
    print(f"  Balqa, Madaba and Ajloun {t2.loc[True, 'mean'] * 100:.2f}% Christian on n="
          f"{int(t2.loc[True, 'size']):,}, the rest {t2.loc[False, 'mean'] * 100:.2f}% on n="
          f"{int(t2.loc[False, 'size']):,}  (p={p_h:.3g})")

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
        lambda r: (f"Arab Barometer waves II to VIII pooled, n={r.n_gov} in this governorate; "
                   f"{r.basis_note} applied to the Department of Statistics' end-2025 "
                   "governorate population estimate"),
        axis=1)

    total = int(out["count"].sum())
    if total != int(pop.sum()):
        raise SystemExit(f"drawn {total:,} against DOS {int(pop.sum()):,}")

    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, "
          f"{out['source_category'].nunique()} categories, {out['geo_id'].nunique()} units)")

    # ---- what the file now says, which is what note_public has to reproduce ----
    drawn = out.groupby("source_category")["count"].sum()
    print("\n  national, as drawn (population-weighted, so not the survey's own weighting):")
    for cat, n in drawn.sort_values(ascending=False).items():
        print(f"    {n / total * 100:6.2f}%  {cat}  ({n:,})")
    ch = out[out["source_category"] == "Christian"].set_index("geo_id")["count"]
    tot = out.groupby("geo_id")["count"].sum()
    share = (ch / tot * 100).sort_values(ascending=False)
    print("\n  Christian share as drawn, with the sample behind each:")
    for gid, s in share.items():
        n = int(n_by[gid])
        phat = s / 100
        ci = 1.96 * np.sqrt(max(phat * (1 - phat), 1e-9) / n)
        print(f"    {names[gid]:<10}{s:6.2f}%  n={n:>5}  +/-{ci * 100:4.2f}pp   "
              f"{int(ch[gid]):>9,} people")
    thin = n_by.idxmin()
    print(f"\n  thinnest governorate {names[thin]} at n={int(n_by[thin])}; "
          f"median n={int(n_by.median())}")
    print(f"  largest Christian count: {names[ch.idxmax()]} at {int(ch.max()):,}; "
          f"largest share: {names[share.idxmax()]} at {share.max():.2f}%")


# The waves whose `Q1012` card offers a box for having no religion, and the count of Jordanian
# respondents they hold. `note_public` says not one of them ticked it, which is a claim about
# the questionnaire as much as about the country, so both halves are asserted below.
NO_RELIGION_WAVES = ["VI-1", "VI-2", "VI-3", "VII", "VIII"]
NOTE_POOLED = 14_917          # respondents with any Q1012 answer, refusals included
NOTE_NAMED = 14_906           # of them, the ones who named a religion; NOTE_POOLED - DROPPED
NOTE_CHRISTIANS = 245         # of them, Christian
NOTE_NO_RELIGION_N = 8_034    # respondents in the five waves that offered a no-religion box


def note_public_figures(df):
    """Assert the five figures in `note_public` that data/normalized/jo.csv cannot carry.

    The standing rule is that every figure in a `note_public` reproduces from the country's
    normalized CSV. Five here cannot, because they are about the SURVEY rather than about the
    people drawn: the size of the pool, the part of it that named a religion, the number of
    Christian interviews behind the whole ordering, and the two halves of the claim that
    nobody chose the no-religion box. The CSV holds counts of Jordanians, not counts of
    respondents.

    So they are asserted here instead, on every build, which gives the same guarantee by a
    different route: a figure in the note cannot go stale without this failing. `origins` does
    the same for the wave IV universe figures.

    `NOTE_NAMED` was the one that was not asserted until 2026-09-09, and it is the one with a
    moving part: it is `NOTE_POOLED` minus the refusals, so it goes stale the moment `DROPPED`
    changes size, which is exactly the change nobody would think to re-read the note for.
    """
    pooled = len(df)
    named = pooled - int(df["category"].isin(DROPPED).sum())
    christians = int(df["category"].eq("Christian").sum())
    five = df[df["wave"].isin(NO_RELIGION_WAVES)]
    print(f"\n  note_public's survey figures: pooled {pooled:,}, of them {named:,} who named "
          f"a religion,\n    Christian interviews {christians}, and {len(five):,} respondents "
          f"in the {len(NO_RELIGION_WAVES)} waves whose card offers a no-religion box")
    for got, want, what in ((pooled, NOTE_POOLED, "the pooled respondent count"),
                            (named, NOTE_NAMED, "the count who named a religion"),
                            (christians, NOTE_CHRISTIANS, "the Christian interview count"),
                            (len(five), NOTE_NO_RELIGION_N,
                             "the sample of the waves offering a no-religion box")):
        if got != want:
            raise SystemExit(f"{what} is {got:,} and note_public says {want:,}. Fix the note "
                             "and this constant together; a figure on screen that the build "
                             "cannot reproduce is the thing this check exists to prevent.")
    stray = sorted(set(five["category"]) - {"Muslim", "Christian", "Refused to answer"})
    if stray:
        raise SystemExit(f"note_public says nobody in those waves chose anything but Muslim, "
                         f"Christian or a refusal, and {stray} is now in the pool")
    print("    all four reproduce, and no Jordanian in those waves chose any other answer")


def origins(df):
    """Wave IV sampled Jordan's non-citizens; every other wave effectively did not.

    Re-read out of the .sav rather than quoted, because it is the only measurement anywhere of
    whether the quarter of Jordan that DOS counts and the survey mostly does not is different.
    """
    import pyreadstat

    p = os.path.join(ab.AB_DIR, "ABIV_English_Updated.sav")
    raw, meta = pyreadstat.read_sav(p)
    c = ab._col(raw, "country")
    cname = raw[c].map(meta.variable_value_labels.get(c, {})).astype(str).str.strip()
    sub = raw[cname.map(ab.country_key) == ab.country_key(COUNTRY)].copy()
    ocol, rcol = ab._col(raw, "q1020jo"), ab._col(raw, "q1012")
    if ocol is None or rcol is None:
        raise SystemExit("wave IV no longer carries q1020jo and q1012 — the universe check "
                         "in sources/jo.md cannot be re-run and must not be quoted")
    sub["origin"] = sub[ocol].map(meta.variable_value_labels.get(ocol, {}))
    sub["chr"] = sub[rcol].map(meta.variable_value_labels.get(rcol, {})).eq("Christian")
    sub = sub[sub["origin"].notna()]
    syr = sub["origin"].astype(str).str.lower().eq("syrian")
    if not syr.sum():
        raise SystemExit("wave IV no longer records any Syrian-origin Jordanian respondents")
    pval = stats.chi2_contingency(pd.crosstab(syr, sub["chr"]).values)[1]
    print(f"\n  the universe, measured on wave IV, the one wave that sampled non-citizens:")
    for lab, m in [("Syrian origin", syr), ("Jordanian or Palestinian origin", ~syr)]:
        s = sub[m]
        print(f"    {lab:<32} n={len(s):>5}   {int(s['chr'].sum()):>3} Christian  "
              f"{s['chr'].mean() * 100:5.2f}%")
    print(f"    chi-square p={pval:.4g}. DOS's denominator counts everybody; this survey "
          "mostly does not,\n    so the Christian share drawn here is a ceiling. Nothing "
          "corrects for it (§14.4 rule 1).")


def refusal_lean(df, cat):
    """§3.5's lean check: which way does dropping this residual tilt the country?

    Correlate the refused share per governorate against the Christian share per governorate.
    A positive correlation means the refusals are concentrated where Christians are, so
    dropping them makes the map less Christian than Jordan is, and vice versa.

    WITH LEAVE-ONE-OUT, because eleven refusals over twelve units is not enough to carry a
    correlation and the honest output is the range rather than the point. A single governorate
    that can move the sign is the whole finding.
    """
    ref = df.assign(r=df["category"].eq(cat), c=df["category"].eq("Christian"),
                    gid=df["geo_raw"].map(lambda s: NORM.get(key(s))))
    per = ref[ref["gid"].notna()].groupby("gid").agg(
        n=("r", "size"), r=("r", "sum"), c=("c", "sum"))
    per = per[per["n"] > 0]
    x, y = per["r"] / per["n"], per["c"] / per["n"]
    if x.std() == 0 or y.std() == 0:
        print("    §3.5 lean: undefined, one of the two shares does not vary")
        return
    r_all = float(np.corrcoef(x, y)[0, 1])
    loo = []
    for i in range(len(per)):
        m = np.ones(len(per), dtype=bool)
        m[i] = False
        if x[m].std() == 0 or y[m].std() == 0:
            continue
        loo.append(float(np.corrcoef(x[m], y[m])[0, 1]))
    print(f"    §3.5 lean: refused share vs Christian share over {len(per)} governorates, "
          f"r = {r_all:+.3f};\n      leave-one-out range {min(loo):+.3f} to {max(loo):+.3f}, "
          f"and the sign {'FLIPS' if min(loo) * max(loo) < 0 else 'holds'}. "
          f"{int(per['r'].sum())} refusals is too few to carry this either way; it is reported "
          "so that\n      nobody reads the drop as neutral, not because it measures anything.")


def leave_one_out(g, nat, large, names):
    """Re-run the split-half twelve times, dropping one governorate each time.

    The split-half over twelve units has a bar of +0.504 and twelve points, so one leveraged
    governorate can carry it or sink it on its own. §9cm's Panama is the precedent in the other
    direction: 468 of 3.6 million orderings reached its held-out r and every one of them kept
    the same 51% unit in place, which was leverage rather than evidence. This prints what the
    verdict would have been without each unit, which is the cheapest possible guard against
    the same thing.

    **It names every unit on the minimum and not the first of them**, which is a correction
    made on 2026-09-09 rather than the original behaviour; `sources/jo.md` §9.4 has the
    finding. Jordan's minimum is a tie and the difference matters to how the fragility reads.
    """
    # The exact null, the same bar `ab.stability` uses. It was `1.96/sqrt(n-1)` until
    # 2026-09-09 (+0.591 at 12 units, +0.620 at 11); `sources/spearman_null.py` has why that
    # was a 0.023-level test rather than a 0.05 one. Jordan's verdict is unchanged either way,
    # and this function decides nothing in any case — it prints.
    bar12, _ = spearman_null.critical_rho(N_UNITS_BOTH_HALVES)
    bar11, _ = spearman_null.critical_rho(N_UNITS_BOTH_HALVES - 1)
    waves = sorted(g["wave_no"].unique())
    cut = waves[len(waves) // 2]
    early, late = g[g["wave_no"] < cut], g[g["wave_no"] >= cut]
    print(f"\n  leave-one-out on the split-half. Two bars are printed on purpose: dropping a "
          f"unit\n  lowers the correlation AND raises the bar (+{bar12:.3f} at 12 units, "
          f"+{bar11:.3f} at 11), so\n  counting against the 11-unit bar is the harsher of the "
          "two readings and both are given:")
    for c in sorted(nat.index, key=lambda k: -nat[k]):
        if nat[c] < ab.ELIGIBLE_FLOOR:
            continue

        def share(d):
            return d.groupby("geo_id").apply(
                lambda x: x.loc[x["category"] == c, "w"].sum() / x["w"].sum(),
                include_groups=False)

        j = pd.concat([share(early).rename("e"), share(late).rename("l")], axis=1).dropna()
        rs = {}
        for gid in j.index:
            k = j.drop(index=gid)
            rs[gid] = k["e"].corr(k["l"], method="spearman")
        lo, hi = min(rs.values()), max(rs.values())
        # EVERY UNIT ON THE MINIMUM, NOT THE FIRST OF THEM. `min(rs, key=rs.get)` returns
        # whichever tied unit `j` happens to be ordered by, and Jordan is a tie: dropping
        # Balqa and dropping Ajloun both give +0.5093. Reported as one unit, that reads as
        # "one governorate carries this"; reported as two, it is the worse and truer statement
        # that the top two of the ordering are EACH individually load-bearing. The tolerance
        # is the precision this line prints at, so anything shown as the same number is named
        # as the same number.
        worst = sorted(gid for gid, v in rs.items() if abs(v - lo) < 5e-4)
        print(f"    {str(c)[:26]:<28} r ranges {lo:+.3f} to {hi:+.3f} over the 12 drops; "
              f"{sum(1 for v in rs.values() if v >= bar12)}/12 clear +{bar12:.3f}, "
              f"{sum(1 for v in rs.values() if v >= bar11)}/12 clear +{bar11:.3f}")
        shown = ", ".join(names.get(gid, gid) for gid in worst)
        if len(worst) == 1:
            head = f"      the unit that matters most is {shown}: without it, {lo:+.3f}"
            tail = "  <-- THE VERDICT DEPENDS ON ONE GOVERNORATE"
        else:
            head = (f"      {len(worst)} units matter equally and each is individually "
                    f"load-bearing: {shown};\n      dropping any one of them alone gives "
                    f"{lo:+.3f}")
            tail = (f"  <-- THE VERDICT DEPENDS ON ANY ONE OF "
                    f"{len(worst)} GOVERNORATES")
        print(head + ("" if lo >= bar11 else tail))


def code_witness(g):
    """The harmonised names against the governorate CODE, where the code means something.

    Six of the nine waves number `Q1` in a way that can be decoded independently of any name
    (see `CODE_SYSTEMS`). Every respondent in those waves must land on the same governorate
    both ways. This is the check `[[reference_name_join_wrong_neighbour]]` asks for: a
    permutation of `NORM` preserves every total and is invisible to arithmetic, and it breaks
    this on hundreds of rows at once.
    """
    print("\n  the names against the codes (a witness, not the key):")
    for label, (waves, table) in CODE_SYSTEMS.items():
        sub = g[g["wave"].isin(waves)]
        if not len(sub):
            raise SystemExit(f"CODE_SYSTEMS names waves {waves}, none of which is in the pool")
        by_code = sub["geo_code"].astype("Int64").map(table)
        bad = sub[by_code.isna() | (by_code.to_numpy() != sub["geo_id"].to_numpy())]
        print(f"    {label:<24} waves {','.join(waves):<12} n={len(sub):>6}  "
              f"{len(sub) - len(bad):>6} agree, {len(bad)} do not")
        if len(bad):
            print(bad.groupby(["wave", "geo_code", "geo_raw", "geo_id"]).size()
                  .head(20).to_string())
            raise SystemExit(f"the {label} code and the harmonised name disagree on "
                             f"{len(bad)} respondents. One of the two is wrong and the totals "
                             "cannot tell you which — STOP.")


if __name__ == "__main__":
    main()
