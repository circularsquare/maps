"""Afrobarometer — the parts any country drawn from it share.

The third member of the barometer family here, after `sources/lapop.py` (§11ad) and
`sources/arabbarometer.py` (§11af), and deliberately the same shape as both: pool the rounds
that asked, weight, cut by the survey's own subnational unit, apply the shares to a population
table somebody else counted. `sources.md` §11ai is the assessment of the source;
`sources/lr.md` is the first country built on it.

What is here is only what is genuinely the same in every country. What is NOT here, and must
stay in each `sources/<cc>.py`:

  * the decode from the survey's `REGION` label to a polygon, which is a name join in this
    survey and never a code join — see below;
  * how the answers are grouped, if they are grouped at all;
  * `CARRIES`, the categories drawn on their own unit shares, asserted per country;
  * every word of the docstring a reader will actually consult.

## THE RELIGION QUESTION MOVES VARIABLE NAME EVERY OTHER ROUND

    R4   Q90    "Q90. Religion of respondent"
    R5   Q98A   "Q98a. Religion of respondent"
    R6   Q98A
    R7   Q98    "Q98. Religion of respondent"
    R8   Q98A
    R9   Q95    "Q95. Religion of respondent"

`Q98A` in round 7 exists and is something else entirely, so a module that looks for one name
and falls back to another would read the wrong column without failing. `ROUNDS` below names
the column for each round explicitly and `load()` asserts the variable LABEL says religion.

## THE WEIGHT CHANGES NAME AND MEANING AT ROUND 8

Rounds 1-7 carry `withinwt`, one within-country weight. Round 8 replaced it with two,
`withinwt_ea` and `withinwt_hh`, because the sampling frame moved from enumeration areas to
households; `withinwt_hh` is the one Afrobarometer's own codebook uses for within-country
estimates from R8 on. `load()` asserts that whichever weight it takes sums to the country's
respondent count, which is the property every Afrobarometer within-country weight has and the
cross-country ones (`Combinwt`) do not.

## THE ANSWER CARD IS THE SAME CARD AND THE PROBING IS NOT

This is the trap that matters, and it is `sources/lapop.py`'s "the ANSWER SET rather than the
sample is what limits it" arriving in a new form. Afrobarometer's religion card offers
`Christian only` alongside about twenty named denominations, and **how many respondents end up
in `Christian only` is set by how hard that round's fieldwork probed**. Liberia, the same
question, the same country, six rounds:

    R4  28.0%    R5  44.4%    R6  23.1%    R7  67.4%    R8  72.0%    R9  61.6%

Nothing about Liberia moved like that, and the swing is not monotone in time. So **a pooled denominational share is a measurement of
the fieldwork**, and a country that wants the denominations has to argue for them rather than
read them off. Grouping the answers up to a level the probing cannot move — which for Liberia
is the five categories its own census uses — is the construction `sources/lr.py` chose, with
the reasoning written down there.

**AND IT IS NOT A LIBERIAN PROBLEM.** Measured 2026-09-08 over all six rounds and every
country in them, the `Christian only` share swings by more than fifteen points in **ten** of
the twenty-eight countries with three or more rounds, and the median country swings 10.2
points:

    Botswana  5.0 25.6 28.9 56.9 54.6 70.7   65.7p     Zambia   12.7 11.7 15.3 23.1  3.7  7.7   19.5p
    Liberia  28.0 44.4 23.1 67.4 72.0 61.6   48.9p     Ghana    21.8 18.2  8.9 20.3 20.3 28.0   19.1p
    Nigeria  32.3 19.4 28.0 44.9 47.3 41.1   27.9p     S Africa 34.6 30.8 33.6 48.4 49.0 38.5   18.1p
    Kenya     9.5 11.4 16.7 24.8 30.7 37.2   27.7p     Namibia  10.4 13.3  8.8 22.1 16.2 26.8   18.0p
    Gabon        -    - 20.9 24.9 45.1 40.4   24.2p    Tanzania  8.4  3.6  6.4 15.3 14.1 21.3   17.7p

**The obvious rescue does not work.** If this were one questionnaire change it would be a round
effect, correctable by a per-round factor — but it moves by country AND round: between R7 and
R8 Zambia falls 23.1% to 3.7% while Kenya rises 24.8% to 30.7% and Botswana barely moves. It
is the fieldwork team, not the release. So there is nothing to divide out, and the rule is the
one above: group up to a level the probing cannot move, and make a country that wants the
denominations bring an outside witness to their LEVEL rather than their ranking.

The one country-specific thing to check is the number, not the rule: Mali swings 1.3 points
and Madagascar 2.0, so a country that hardly uses the catch-all at all can have its
denominations argued for. Print the by-round share before deciding, the way `sources/lr.py`
does, and assert it afterwards so a re-release re-opens the question.

## THE UNIT LABELS ARE `[[reference_pooled_survey_labels]]`

`REGION` is a per-round label set: Liberia's fifteen counties arrive as `Rivercess` in one
round and `River Cess` in the next, `Bassa` for `Grand Bassa`, and the whole set upper-cased in
round 9. Pooling on the raw string splits units; pooling on the numeric code is worse, because
the codes are a shared cross-country range that is re-cut between rounds. Every country module
must supply its own label -> unit table and `load()` returns the raw label so it can.

## AND SO IS `COUNTRY`, WHICH IS WORSE, BECAUSE THE FAILURE IS SILENT

Found on review, 2026-09-08. `load()` selects one country by name, and three countries are not
one name across the six rounds:

    Cote d'Ivoire     R5 `Cote d’Ivoire`  R6 `Cote d'Ivoire`  R7-R9 `Côte d'Ivoire`
    Eswatini          R5-R6 `Swaziland`   R7 `eSwatini`       R8-R9 `Eswatini`
    Cabo Verde        R4-R6 `Cape Verde`  R7-R9 `Cabo Verde`

That is a curly apostrophe, an accent, a lower-case first letter and two renames. Asking for
`Côte d'Ivoire` gets three of the five rounds it is in and asking for `Cabo Verde` gets half
the respondents, with **no error and no missing unit** — the pool is simply smaller and its
round mix is wrong, which given the section above is the one property that matters most. All
three are on the queue §11ai opened.

`COUNTRY_ALIASES` names every spelling of the countries that have more than one, `load()`
matches through `fold()` rather than a bare `casefold`, and any round in which the country
matched nothing is now reported by name; a round that matched nothing while holding a label
that differs only by an accent is an error rather than a note.

## Access

`afrobarometer.org/data/merged-data/` links every merged round as a plain file on the same
host, no form and no account; the URLs are in `ROUNDS`. The data usage policy asks for a
citation — *"Afrobarometer Data, [Country(ies)], [Round(s)], [Year(s)], available at
http://www.afrobarometer.org"* — and gates only early access and the geocoded extracts, which
are not used here. Read before anything was downloaded, because of Nişanyan (§11ac).
"""

import math
import os
import ssl
import sys
import unicodedata
import urllib.request
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AB_DIR = os.path.join(ROOT, "data", "raw", "afrobarometer")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")

# (round, filename, url, religion column, weight column). The religion column is named per
# round on purpose — see the module docstring; `Q98A` exists in round 7 and is not religion.
ROUNDS = [
    (4, "merged_r4_data.sav",
     "https://www.afrobarometer.org/wp-content/uploads/2022/02/merged_r4_data.sav",
     "Q90", "Withinwt"),
    (5, "merged-round-5-data-34-countries-2011-2013-last-update-july-2015_0.sav",
     "https://www.afrobarometer.org/wp-content/uploads/2022/02/"
     "merged-round-5-data-34-countries-2011-2013-last-update-july-2015_0.sav",
     "Q98A", "withinwt"),
    (6, "merged_r6_data_2016_36countries2.sav",
     "https://www.afrobarometer.org/wp-content/uploads/2022/02/"
     "merged_r6_data_2016_36countries2.sav",
     "Q98A", "withinwt"),
    (7, "r7_merged_data_34ctry.release.sav",
     "https://www.afrobarometer.org/wp-content/uploads/2022/02/"
     "r7_merged_data_34ctry.release.sav",
     "Q98", "withinwt"),
    (8, "afrobarometer_release-dataset_merge-34ctry_r8_en_2023-03-01.sav",
     "https://www.afrobarometer.org/wp-content/uploads/2023/03/"
     "afrobarometer_release-dataset_merge-34ctry_r8_en_2023-03-01.sav",
     "Q98A", "withinwt_hh"),
    (9, "R9.Merge_39ctry.20Nov23.final_.release_Updated.4Jun25-3.sav",
     "https://www.afrobarometer.org/wp-content/uploads/2025/06/"
     "R9.Merge_39ctry.20Nov23.final_.release_Updated.4Jun25-3.sav",
     "Q95", "withinwt_hh"),
]

# The COUNTRY column is a per-round label set exactly as REGION is, and `load()` selects on
# it, so a country spelled two ways loses whole rounds without failing. These are every case
# in R4-R9, measured 2026-09-08; a country named here is matched on ALL of its spellings, and
# the key is what a country module passes to `load()`. See the docstring.
COUNTRY_ALIASES = {
    "cote d'ivoire": ["Cote d'Ivoire", "Cote d’Ivoire", "Côte d'Ivoire"],
    "eswatini": ["Swaziland", "eSwatini", "Eswatini"],
    "cabo verde": ["Cape Verde", "Cabo Verde"],
}

# Answers that are not a religion. Kept here rather than per country because they are the
# survey's own non-response codes and mean the same thing in every one of them.
NON_ANSWERS = {"don't know", "dont know", "refused", "missing", "refused to answer",
               "don't know/haven't heard enough to say", "not asked in country"}

# Below this many possible orderings of a country's units, `held_out` checks every one of them
# instead of sampling. 8! = 40,320 is under it and 9! = 362,880 is over.
EXACT_PERM_MAX = 50_000

# A category is ELIGIBLE for its own geography if it is at least this much of the country. The
# floor is `sources/lapop.py`'s and the reasoning is §11ad's: that instrument's sub-national
# cut was measured failing below about 1% against two censuses. Eligibility is not permission —
# `stability()` is the stricter test and it decides.
ELIGIBLE_FLOOR = 0.01


def fetch(rounds=None):
    """Download the merged rounds named, or all six. ~280 MB of .sav."""
    os.makedirs(AB_DIR, exist_ok=True)
    want = set(rounds) if rounds else {r[0] for r in ROUNDS}
    ctx = ssl.create_default_context()
    for rnd, name, url, _rel, _wt in ROUNDS:
        if rnd not in want:
            continue
        dst = os.path.join(AB_DIR, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
            print(f"  have R{rnd} {name} ({os.path.getsize(dst):,} bytes)")
            continue
        req = urllib.request.Request(
            url, headers={"User-Agent": UA,
                          "Referer": "https://www.afrobarometer.org/data/merged-data/"})
        with urllib.request.urlopen(req, timeout=1800, context=ctx) as r:
            data = r.read()
        # §5a: a 200 is not a download. SPSS .sav files start "$FL2" or "$FL3".
        if data[:2] != b"$F":
            raise SystemExit(f"R{rnd} is not a .sav — starts {data[:16]!r}")
        with open(dst + ".part", "wb") as f:
            f.write(data)
        os.replace(dst + ".part", dst)
        print(f"  got  R{rnd} {name} ({os.path.getsize(dst):,} bytes)")


def _read(path):
    """Round 6's .sav is not valid UTF-8 and the default read raises on it."""
    import pyreadstat

    try:
        return pyreadstat.read_sav(path)
    except pyreadstat._readstat_parser.ReadstatError:
        return pyreadstat.read_sav(path, encoding="LATIN1")


def _col(df, name):
    up = {c.upper(): c for c in df.columns}
    return up.get(name.upper())


def _deaccent(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def country_spellings(country):
    """Every label `country` arrives under, folded. One name is the normal case.

    The alias lookup is itself accent-insensitive, because `Côte d'Ivoire` and `Cote
    d'Ivoire` are two of the spellings it has to find the table from.
    """
    aliases = {}
    for k, group in COUNTRY_ALIASES.items():
        # Symmetric: any spelling in a group finds the whole group, so a module that asks for
        # `Swaziland` gets the same six rounds as one that asks for `Eswatini`.
        for n in list(group) + [k]:
            aliases[_deaccent(fold(n))] = group
    names = list(aliases.get(_deaccent(fold(country)), [country]))
    if fold(country) not in {fold(n) for n in names}:
        names.append(country)
    return {fold(n) for n in names}


def fold(label):
    """The differences between two wordings that carry no meaning: case, spacing, edge marks.

    Deliberately narrow, and the narrowness is the point — the same function, for the same
    reason, as `sources/arabbarometer.py`'s. It exists to DETECT a collision, not to resolve
    one; see `assert_one_wording`.
    """
    s = unicodedata.normalize("NFKC", str(label))
    s = s.replace("’", "'")
    return " ".join(s.split()).strip(" .,:;!?_-").casefold()


def assert_one_wording(df, country, cat_col="category"):
    """ONE ANSWER, ONE SPELLING, or this refuses to pool and hands the call back.

    The pooling key from `load()` onward is the label STRING, and nothing about `pd.concat`
    notices that round 8 spells an answer differently from round 6. What that costs is
    `sources/arabbarometer.py`'s list and it is the same list here: two rows in `national()`,
    each ranked by `stability()` on half the respondents, each measured against
    `ELIGIBLE_FLOOR` separately, and a taxonomy `MAP` that needs both spellings or silently
    resolves one to `None`. **Every one of those preserves the totals.**

    This does not fold the data. Whether two spellings are one box on one showcard is a
    reading of the questionnaires and belongs in the country module next to the reason.
    """
    seen = {}
    for (lab, rnd), n in df.groupby([cat_col, "round"], sort=False).size().items():
        seen.setdefault(fold(lab), {}).setdefault(lab, []).append((rnd, int(n)))
    clashes = {f: v for f, v in seen.items() if len(v) > 1}
    if not clashes:
        return
    lines = []
    for f, spellings in sorted(clashes.items()):
        lines.append(f"    {f!r} arrives as {len(spellings)} categories:")
        for lab, rounds in sorted(spellings.items()):
            where = ", ".join(f"R{r} n={n}" for r, n in sorted(rounds))
            lines.append(f"      {lab!r}  {sum(n for _r, n in rounds)} respondents  ({where})")
    raise SystemExit(
        f"{country}'s pooled rounds spell one answer more than one way, so the pool would "
        "carry it as two categories with the totals still adding up:\n"
        + "\n".join(lines)
        + "\n  Nothing is folded here on purpose. Decide it in the country module, re-word "
        "the `category` column there with the reason written down, and call "
        "`ab.assert_one_wording` again on the result.")


def report_wordings(df, cat_col="category"):
    """Print the one-answer-two-spellings clashes instead of raising on them.

    For a country module that REGROUPS the answers before using them — which is the normal
    case in this survey, because the denominational card cannot be pooled as it stands — the
    clash is about to be resolved and stopping the load would make the documented escape hatch
    unreachable. It is still printed, because a spelling that appears here and not in the
    country module's grouping table is a silent drop.
    """
    seen = {}
    for lab in df[cat_col].unique():
        seen.setdefault(fold(lab), []).append(lab)
    clashes = {f: v for f, v in seen.items() if len(v) > 1}
    if clashes:
        print(f"  {len(clashes)} answer(s) arrive under more than one spelling, to be "
              "resolved by the country module's grouping:")
        for f, spellings in sorted(clashes.items()):
            print(f"    {f!r}: {spellings}")


def load(country, expect_rounds=None, regroup=False):
    """One country's respondents, decoded round by round through that round's own labels.

    Returns [`round`, `round_no`, `category`, `geo_raw`, `geo_code`, `w`], one row per
    respondent who gave a religion answer. `category` is the ANSWER'S OWN WORDING (spec §2.4)
    and never a numeric code; the codes are a shared cross-country range that moves between
    rounds, so pooling on them merges different answers while every total still adds up.

    Non-answers (`Don't know`, `Refused`, `Missing`) are dropped here and reported, because
    they are the survey's own non-response codes rather than a religion anybody holds.

    `expect_rounds` is the list of rounds the country is known to appear in, asserted so a
    re-release that adds or drops one is a failure here rather than a quiet re-levelling.

    `regroup=True` says the country module is about to group these answers into something
    else, so the one-answer-two-spellings clash is REPORTED rather than raised on: the module
    is expected to call `assert_one_wording` on the grouped column instead. Leave it False for
    a country that pools the raw answers, which is where the silent split would happen.
    """
    frames = []
    misses = []
    want = country_spellings(country)
    for rnd, name, _url, relname, wtname in ROUNDS:
        p = os.path.join(AB_DIR, name)
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing — run the country module with --fetch")
        df, meta = _read(p)
        ccol = _col(df, "COUNTRY")
        if ccol is None:
            raise SystemExit(f"R{rnd} has no COUNTRY column")
        clab = meta.variable_value_labels.get(ccol, {})
        cname = df[ccol].map(clab).fillna(df[ccol].astype(str)).astype(str).str.strip()
        # The COUNTRY label is a per-round label set — see the docstring. `fold` absorbs the
        # case and apostrophe variants and `COUNTRY_ALIASES` the renames; an accent is
        # neither, so a round that misses on the accent alone is caught below.
        sub = df[cname.map(fold).isin(want)]
        if not len(sub):
            here = sorted(set(cname))
            near = [c for c in here
                    if {_deaccent(f) for f in want} & {_deaccent(fold(c))}]
            if near:
                raise SystemExit(
                    f"R{rnd} has no rows for {country!r} but does have {near} — that is the "
                    "same country under a different spelling and the round would be dropped "
                    "silently. Add it to COUNTRY_ALIASES.")
            misses.append(rnd)
            continue

        rel = _col(df, relname)
        if rel is None:
            raise SystemExit(f"R{rnd} has {country} rows but no {relname}")
        # The variable NAME moves between rounds and one of the names is re-used for a
        # different question, so the LABEL is what is asserted.
        vlabel = str(meta.column_names_to_labels.get(rel, ""))
        if "religion of respondent" not in vlabel.casefold():
            raise SystemExit(f"R{rnd} {relname} is labelled {vlabel!r}, which is not the "
                             "religion question — the release has renumbered it")
        geo = _col(df, "REGION")
        if geo is None:
            raise SystemExit(f"R{rnd} has no REGION column")
        wt = _col(df, wtname)
        if wt is None:
            raise SystemExit(f"R{rnd} has no {wtname} column")
        w = pd.to_numeric(sub[wt], errors="coerce")
        # Every Afrobarometer WITHIN-country weight averages 1 over the country; the
        # cross-country ones do not, and taking one by mistake would silently re-level the
        # unit shares by country size.
        if not 0.98 <= w.sum() / len(sub) <= 1.02:
            raise SystemExit(f"R{rnd} {wtname} sums to {w.sum():.1f} over {len(sub)} "
                             f"{country} respondents — that is not a within-country weight")

        rlab = meta.variable_value_labels.get(rel, {})
        glab = meta.variable_value_labels.get(geo, {})
        undecoded = sorted(set(sub[rel].dropna()) - set(rlab))
        if undecoded:
            raise SystemExit(f"R{rnd}: {relname} codes with no label: {undecoded}")
        frames.append(pd.DataFrame({
            "round": rnd,
            "round_no": rnd,
            "category": sub[rel].map(rlab),
            "geo_code": sub[geo].to_numpy(),
            "geo_raw": sub[geo].map(glab).to_numpy(),
            "w": w.to_numpy(),
        }))
        # The fieldwork window, because a pooled survey's date range is a thing the country's
        # note_public says out loud and nothing else in the build would check it.
        dcol = _col(df, "DATEINTR")
        span = ""
        if dcol is not None:
            d = pd.to_datetime(sub[dcol], errors="coerce")
            if d.notna().any():
                span = f"  {d.min():%Y-%m} to {d.max():%Y-%m}"
        print(f"  R{rnd}  n={len(sub):>6}  religion answered by "
              f"{int(sub[rel].notna().sum()):>6}  "
              f"{sub[geo].map(glab).nunique()} REGION labels  weight {wtname}{span}")

    if misses:
        # Most countries genuinely are not in every round, so this is a note. It is printed
        # because the alternative is a builder setting `expect_rounds` from a pool that had
        # already lost a round to a spelling nobody looked for.
        print(f"  no {country} rows in R{', R'.join(str(r) for r in misses)} — check that "
              "the country is really absent and not spelled differently there "
              "(COUNTRY_ALIASES)")
    if not frames:
        raise SystemExit(f"no {country} rows in any round")
    out = pd.concat(frames, ignore_index=True)
    out = out[out["category"].notna()].copy()

    junk = out["category"].map(lambda c: fold(c) in NON_ANSWERS)
    if junk.any():
        counts = out.loc[junk, "category"].value_counts().to_dict()
        print(f"  dropping {int(junk.sum())} non-answers: {counts}")
        out = out[~junk].copy()

    rounds = sorted(set(out["round"]))
    if expect_rounds is not None and rounds != list(expect_rounds):
        raise SystemExit(f"{country} now appears in rounds {rounds}, expected "
                         f"{list(expect_rounds)} — a re-release has changed the pool")
    if regroup:
        report_wordings(out)
    else:
        assert_one_wording(out, country)
    return out


def national(df, cat_col="category"):
    """Weighted national share per answer, pooled over the rounds in `df`."""
    return df.groupby(cat_col)["w"].sum() / df["w"].sum()


def held_out(df, pop, country, unit_col="geo_id", n_perm=20000, seed=0,
             pop_source="the population table"):
    """Test the unit decode without touching the religion column.

    THE TEST IS THE PERMUTATION, NOT THE CORRELATION. The survey's weighted share of
    respondents per unit is compared with the population table's share, and what makes it
    evidence is how that compares with the wrong answers: the unit labels are shuffled and the
    observed r is ranked against those. A correlation any random pairing could produce says
    nothing; one that beats every random pairing pins the decode. `sources/lapop.py` has the
    longer argument and `sources/kz.py` the original.

    A survey that samples proportional to population will do very well here and that is fine —
    the question this answers is whether the NAMES were joined to the right polygons, which a
    permutation is exactly the test for.

    Below `EXACT_PERM_MAX` orderings every one of them is checked rather than sampled, and the
    observed ordering is excluded by VALUE rather than by index so that orderings which only
    swap equal-population units are not counted as beating it.
    """
    print("\n  held-out check (nothing here touches the religion column):")
    share_s = df.groupby(unit_col)["w"].sum() / df["w"].sum()
    share_p = pop / pop.sum()
    j = pd.concat([share_s.rename("survey"), share_p.rename("pop")], axis=1).dropna()
    if len(j) != len(share_s):
        raise SystemExit(f"{len(share_s) - len(j)} sampled units have no population")
    r = np.corrcoef(j["survey"], j["pop"])[0, 1]
    if not np.isfinite(r):
        raise SystemExit(f"the survey/{pop_source} correlation over {country}'s {len(j)} "
                         "units is undefined, so this check cannot say anything")
    ratio = (j["survey"] / j["pop"]).sort_values()
    print(f"    unit share of respondents vs {pop_source}:  r = {r:+.3f} over {len(j)} units")
    print(f"      thinnest sampled {ratio.index[0]} at {ratio.iloc[0]:.2f}x its population "
          f"share, fullest {ratio.index[-1]} at {ratio.iloc[-1]:.2f}x")

    rng = np.random.default_rng(seed)
    a, b = j["survey"].to_numpy(), j["pop"].to_numpy()
    n = len(j)
    total = math.factorial(n)
    exact = total <= EXACT_PERM_MAX
    if exact:
        import itertools
        pairings = [b[list(p)] for p in itertools.permutations(range(n))]
        how = f"all {total - 1:,} other orderings of the same units"
    else:
        pairings = [rng.permutation(b) for _ in range(n_perm)]
        how = f"{n_perm:,} random pairings of the same units"
    perm = np.array([np.corrcoef(a, p)[0, 1] for p in pairings])
    same = np.array([np.array_equal(p, b) for p in pairings])
    beaten = int(((perm >= r) & ~same).sum())
    print(f"      against {how}: best r = {perm[~same].max():+.3f}, and {beaten} reach the "
          f"observed one")
    if total <= 1_000_000:
        print(f"      {n} units allow {total:,} orderings, so the strongest this check can "
              f"say is 1 in {total - 1:,}"
              + ("" if n >= 7 else " — too weak to carry the join on its own"))
    if beaten:
        raise SystemExit(
            f"{beaten} of {how} match or beat the observed r={r:+.3f} for {country}. The "
            "population check does not pin this decode, so the join needs a witness that "
            "does before anything is drawn.")
    return r


def stability(df, nat, n_units, unit_col="geo_id", cat_col="category", override=None):
    """WHICH CATEGORIES CARRY THEIR OWN GEOGRAPHY — the split-half decides it.

    Spec §14.16's test. Each category is ranked across the units in the earlier half of the
    rounds and again in the later half, and the two orderings are compared. A category whose
    ranking does not replicate has not demonstrated that it HAS a geography, whatever its
    pooled spread looks like. The bar is 1.96/sqrt(n-1), which is what a Spearman correlation
    over `n` units needs to be distinguishable from zero at 95%; it is never moved to make
    something pass, because moving it would silently change every other category too.

    **HARMONISE THE UNITS BEFORE CALLING THIS.** §11af's Egypt run is the cautionary case: a
    stability test run on unharmonised labels reports noise as a negative, and a negative is
    what this project treats as evidence.

    In THIS survey the split-half carries a second job, and it is the one the module docstring
    is about. The probing depth behind `Christian only` changes between rounds, so a
    denominational category is measured on a different fraction of respondents in each half.
    A category that cannot survive that is one whose level depends on the fieldwork, and the
    test failing it is the right answer rather than a false negative.

    `override` is `{category: "the reason"}` and is a person's decision, not an agent's.
    """
    override = override or {}
    bar = 1.96 / np.sqrt(n_units - 1)
    rounds = sorted(df["round_no"].unique())
    cut = rounds[len(rounds) // 2]
    early, late = df[df["round_no"] < cut], df[df["round_no"] >= cut]
    print(f"\n  split-half stability across rounds (§14.16), bar = +{bar:.3f} at 95% on "
          f"{n_units} units:")
    print(f"    rounds {sorted(set(early['round']))} n={len(early):,}   "
          f"rounds {sorted(set(late['round']))} n={len(late):,}")
    print(f"    {'answer':<30}{'national':>10}{'spearman':>10}{'pearson':>9}  verdict")

    carries = []
    for c in sorted(nat.index, key=lambda k: -nat[k]):
        if nat[c] < ELIGIBLE_FLOOR:
            print(f"    {str(c)[:28]:<30}{nat[c] * 100:9.2f}%{'':>10}{'':>9}  "
                  f"too small to place (§11ad)")
            continue

        def share(d):
            return d.groupby(unit_col).apply(
                lambda x: x.loc[x[cat_col] == c, "w"].sum() / x["w"].sum(),
                include_groups=False)

        j = pd.concat([share(early).rename("e"), share(late).rename("l")], axis=1).dropna()
        if len(j) != n_units:
            raise SystemExit(f"{len(j)} units appear in both round halves, not the {n_units} "
                             "the bar was computed for — re-read the pool before trusting "
                             "any correlation below")
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sp = j["e"].corr(j["l"], method="spearman")
            pe = j["e"].corr(j["l"])
        passed = bool(np.isfinite(sp)) and sp >= bar
        forced = c in override
        if passed or forced:
            carries.append(c)
        shown = f"{sp:+10.3f}{pe:+9.3f}" if np.isfinite(sp) else f"{'undefined':>10}{'':>9}"
        if passed:
            verdict = f"own geography  ({len(j)} units in both halves)"
        elif forced:
            verdict = "own geography — UNDER THE BAR, drawn on Anita's call"
        else:
            verdict = "NOT distinguishable from zero"
        print(f"    {str(c)[:28]:<30}{nat[c] * 100:9.2f}%{shown}  {verdict}")
        if forced and not passed:
            print(f"        reason: {override[c]}")
    stale = sorted(set(override) - set(nat.index))
    if stale:
        raise SystemExit(f"override names categories this country does not have: {stale}")
    return carries


def build(df, nat, large, small, pop, units, unit_col="geo_id", cat_col="category",
          unit_noun="unit"):
    """Shares x population -> counts, as a closed partition of every unit.

    THE MEASURED SHARES PASS THROUGH UNTOUCHED. A unit's share of a category that cleared the
    split-half is applied as it stands; what is left of that unit is divided among the rest at
    their NATIONAL relative proportions, so the tail's geography is the residual of the stable
    measurements rather than a flat national rate. `sources/lapop.py` has the argument.

    Returns float counts, not integers: a country that fits this table to a second margin
    afterwards must round once, at the end, and not twice.
    """
    by_unit = df.groupby([unit_col, cat_col])["w"].sum().unstack(fill_value=0.0)
    for c in nat.index:
        if c not in by_unit.columns:
            by_unit[c] = 0.0
    unit_share = by_unit.div(by_unit.sum(axis=1), axis=0)

    residual = 1.0 - unit_share[large].sum(axis=1)
    small_total = float(sum(nat[c] for c in small))
    if small:
        if (residual <= 0).any():
            raise SystemExit(f"units with no room for the tail: "
                             f"{sorted(residual[residual <= 0].index)}")
        print(f"    the tail is {residual.min():.1%} of {residual.idxmin()} and "
              f"{residual.max():.1%} of {residual.idxmax()}, against {small_total:.1%} "
              "nationally")
    else:
        if residual.abs().max() > 1e-9:
            raise SystemExit("no tail categories, but the large ones do not sum to 1 in "
                             f"every unit (worst {residual.abs().max():.3g})")
        print(f"    no tail: the {len(large)} drawn answers are a closed partition of every "
              f"{unit_noun}")

    rows = []
    for unit in units:
        p = float(pop[unit])
        for c in large:
            rows.append((unit, c, unit_share.loc[unit, c] * p, f"{unit_noun} share"))
        for c in small:
            rows.append((unit, c, residual[unit] * (nat[c] / small_total) * p,
                         f"national share within the {unit_noun}'s residual"))
    return pd.DataFrame(rows, columns=["geo_id", "source_category", "count", "basis_note"])


def main():
    if "--fetch" in sys.argv:
        fetch()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
