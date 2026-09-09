"""Liberia — the 2022 census's five religions, given a county geography by the Afrobarometer.

Reads data/raw/afrobarometer/*.sav and data/geo/lr/lr_lookup.csv, writes
data/normalized/lr.csv. `sources/lr.md` has the acquisition route, the terms and the full
scouting record; `sources/afrobarometer.py` holds the construction shared with every other
country drawn from that survey; `sources.md` §11ai assesses the source across Africa.

## LIBERIA ASKS THE QUESTION AND PUBLISHES THE ANSWER ONLY FOR THE WHOLE COUNTRY

LISGIS has asked religion in every modern census and has never published it below the national
line. The 2022 census (5,250,187 people, census night 10/11 November 2022) prints it as
**Table A13, five categories, national, split by sex and nothing else**, in a report whose
other eighteen appendix tables are all by county. The 2008 census did the same: national in the
Final Report, national plus urban/rural in the 2012 analytical monographs, and one county-level
religion MAP of the urban population in the 2011 Census Atlas. UNSD's table 28 holds the 2008
national row and nothing finer. So the country's own geography of religion has never been
published, and `sources/lr.md` lists what was searched.

## WHAT IS DRAWN, AND WHERE EACH NUMBER COMES FROM

    row margin      county populations          2022 census, Table A4      EXACT
    column margin   the five religion totals    2022 census, Table A13     EXACT
    the interaction the county pattern          Afrobarometer R4-R9        measured

Both census margins sum to 5,250,187 to the person, so the table is fitted to them by
iterative proportional fitting and **no magnitude is invented** (spec §14.4 rule 1): every
person drawn is a person the census counted, in a county the census counted them in, under a
religion label the census printed a total for. The survey decides only how a county's people
are divided between the five columns.

This is not §6's rejected IPF, which was a 2000 composition forced onto 2010 totals. Both
margins here are the same census, taken the same night, from the same report.

## THE DENOMINATIONS ARE NOT DRAWN, AND THE REASON IS THE PROBING RATHER THAN THE SAMPLE

Afrobarometer's card offers `Christian only` beside twenty-odd named denominations, and
Liberia's pooled 7,163 respondents give Pentecostal 9.6%, Methodist 7.0%, Lutheran 5.8%,
Baptist 5.3% and Roman Catholic 4.3% — a split no Liberian census has ever published and the
obvious prize here. It is not taken, because **the share that lands in `Christian only`
is set by how hard each round's fieldwork probed**:

    R4  28.0%    R5  44.4%    R6  23.1%    R7  67.4%    R8  72.0%    R9  61.6%

Liberia did not move like that; the range is 48.9 points and it is not monotone in time. Pooled denominational shares would therefore be a measurement
of which rounds are in the pool, drawn at 1:1,000 on a map whose subject is whether to trust
it. Grouping up to the five categories the census itself uses removes the whole effect: a
Methodist and a `Christian only` are both Christian in every round, so the collapsed shares do
not move with the probing. `CENSUS_CATEGORY` below is that grouping, one line per answer, and
the raw crosstab is printed on every build so the decision can be re-read rather than trusted.

## THE ONE EXTERNAL WITNESS TO THE GEOGRAPHY, AND IT IS LISGIS'S OWN

There is no county religion table to validate against — that is why this file exists — but
there is one published county-level statement. The **2011 Census Atlas**, Fig 9-58, on the 2008
census: *"with the exception of Cape Mount County which was predominantly Muslims, all of the
other 14 counties revealed a predominance of Christians."* That is a testable claim about which
county is the Muslim one, made by the statistics office from the census, thirteen years before
the last Afrobarometer round. `check_atlas()` asserts the survey reproduces it.

## THE COUNTY LABELS ARE `[[reference_pooled_survey_labels]]`

`REGION` brings its own label set every round: `Rivercess` and `River Cess`, `Bassa` for Grand
Bassa, `Cape Mount` for Grand Cape Mount, and the whole set upper-cased in round 9. Pooled on
the raw string, Liberia has 33 units instead of 15 and the split-half is run on halves that
share almost nothing. `NORM` is the harmonisation and every label is asserted to be in it.

Usage:
    python sources/lr.py --fetch    download the six merged Afrobarometer rounds (~280 MB)
    python sources/lr.py            rebuild data/normalized/lr.csv
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
LOOKUP = os.path.join(ROOT, "data", "geo", "lr", "lr_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "lr.csv")

COUNTRY = "Liberia"
ROUNDS = [4, 5, 6, 7, 8, 9]
SOURCE_ID = "lr_census2022_afrobarometer_2008_2024"
YEARS = "2022"

N_UNITS = 15

# 2022 Liberia Population and Housing Census, FINAL RESULTS, Table A13 "Distribution of the
# Population by Religious Affiliation and Sex", page 78. The five categories are the census's
# own wording and its own totals; they sum to 5,250,187, which is Table A1's national count and
# Table A4's county total.
CENSUS_RELIGION = {
    "Christian": 4_458_286,
    "Muslim": 628_859,
    "Traditional African Religion": 25_445,
    "Other religion": 3_431,
    "No religion": 134_166,
}
CENSUS_TOTAL = 5_250_187

# Afrobarometer's answer -> the census category it belongs to. Keyed through `key()`, which
# strips the parenthetical gloss round 8 added, normalises the apostrophe and the spacing round
# the slash, and casefolds; that is what turns 41 raw labels into these 30 answers. See the
# module docstring for why the grouping happens at all.
#
# Two placements worth stating rather than assuming:
#   * `Independent` is Afrobarometer's box for an African independent church, which in Liberia
#     is a Christian body and is what the census counts under Christian.
#   * `Atheist` and `Agnostic` go to `No religion`, which is the census's own catch-all for
#     people with no religious affiliation; the census offers no separate box for either.
CENSUS_CATEGORY = {
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
    "mouridiya brotherhood": "Muslim",
    "tijaniya brotherhood": "Muslim",
    # the rest
    "traditional/ethnic religion": "Traditional African Religion",
    "bahai": "Other religion",
    "other": "Other religion",
    "none": "No religion",
    "atheist": "No religion",
    "agnostic": "No religion",
}

# Every `REGION` label the six rounds use -> the county, keyed through `ckey()`. Three counties
# lose their `Grand` in round 8 and Rivercess is spelled two ways; round 9 upper-cases the lot.
NORM = {
    "bomi": "Bomi",
    "bong": "Bong",
    "gbarpolu": "Gbarpolu",
    "grand bassa": "Grand Bassa", "bassa": "Grand Bassa",
    "grand cape mount": "Grand Cape Mount", "cape mount": "Grand Cape Mount",
    "grand gedeh": "Grand Gedeh",
    "grand kru": "Grand Kru",
    "lofa": "Lofa",
    "margibi": "Margibi",
    "maryland": "Maryland",
    "montserrado": "Montserrado",
    "nimba": "Nimba",
    "river cess": "River Cess", "rivercess": "River Cess",
    "river gee": "River Gee",
    "sinoe": "Sinoe",
}

# What is drawn on its own county shares, asserted so a change in the data is a failure here
# rather than a silent re-drawing of the country. Set from the split-half below.
CARRIES = ["Christian", "Muslim"]

# The 2011 Census Atlas, Fig 9-58, on the 2008 census: Grand Cape Mount is the one county with
# a Muslim majority and the other fourteen are Christian-majority. Asserted, not quoted.
ATLAS_MUSLIM_COUNTY = "Grand Cape Mount"


def key(s):
    """An Afrobarometer religion label reduced to what identifies the answer."""
    s = unicodedata.normalize("NFKC", str(s)).replace("’", "'").replace("‘", "'")
    s = s.split("(")[0]
    s = re.sub(r"\s*/\s*", "/", s)
    return " ".join(s.split()).strip().casefold()


def ckey(s):
    return " ".join(str(s).split()).strip().casefold()


def ipf(seed, row_target, col_target, tol=1e-9, max_iter=500):
    """Fit `seed` to exact row and column margins.

    Both margins here are census totals over the same 5,250,187 people, so the fit exists and
    the iteration converges; a seed cell of zero stays zero, which is why the tail categories
    are seeded at their national rate rather than at the survey's zero counts.
    """
    m = seed.to_numpy(dtype=float).copy()
    r = row_target.reindex(seed.index).to_numpy(dtype=float)
    c = col_target.reindex(seed.columns).to_numpy(dtype=float)
    if abs(r.sum() - c.sum()) > 0.5:
        raise SystemExit(f"the two margins disagree: rows {r.sum():,.0f}, cols {c.sum():,.0f}")
    for i in range(max_iter):
        m *= (c / np.where(m.sum(0) == 0, 1, m.sum(0)))[None, :]
        m *= (r / np.where(m.sum(1) == 0, 1, m.sum(1)))[:, None]
        err = max(np.abs(m.sum(1) - r).max(), np.abs(m.sum(0) - c).max())
        if err < tol * r.sum():
            break
    else:
        raise SystemExit(f"IPF did not converge in {max_iter} passes (worst {err:,.3f})")
    print(f"    IPF converged in {i + 1} passes; worst margin error {err:.3g} people")
    return pd.DataFrame(m, index=seed.index, columns=seed.columns)


def round_within_rows(m):
    """Largest-remainder rounding inside each row, so every county total stays exact."""
    out = np.zeros(m.shape, dtype="int64")
    for i in range(m.shape[0]):
        row = m.iloc[i].to_numpy(dtype=float)
        target = int(round(row.sum()))
        base = np.floor(row).astype("int64")
        short = target - int(base.sum())
        if short:
            order = np.argsort(-(row - base))
            base[order[:short]] += 1
        out[i] = base
    return pd.DataFrame(out, index=m.index, columns=m.columns)


def check_atlas(share, lut):
    """The 2011 Census Atlas said Grand Cape Mount, and only it, is Muslim-majority."""
    nm = dict(zip(lut["geo_id"], lut["name"]))
    muslim = share["Muslim"].rename(index=nm)
    christian = share["Christian"].rename(index=nm)
    majority_muslim = sorted(muslim.index[muslim > christian])
    print("\n  the one published county-level witness — 2011 Census Atlas Fig 9-58, on the "
          "2008 census:")
    print('    "with the exception of Cape Mount County which was predominantly Muslims, all '
          'of the')
    print('     other 14 counties revealed a predominance of Christians."')
    print(f"    the survey's Muslim-majority counties: {majority_muslim or 'none'}")
    if majority_muslim != [ATLAS_MUSLIM_COUNTY]:
        raise SystemExit(
            f"the survey makes {majority_muslim} Muslim-majority, and the only county-level "
            f"religion statement LISGIS has ever published says {ATLAS_MUSLIM_COUNTY!r} and "
            "no other. That is the single external check this country has; it failing means "
            "the decode or the pool has changed and nothing should be drawn until someone "
            "reads why.")
    print(f"    {ATLAS_MUSLIM_COUNTY}: {muslim[ATLAS_MUSLIM_COUNTY]:.1%} Muslim against "
          f"{christian[ATLAS_MUSLIM_COUNTY]:.1%} Christian — the atlas reproduced")


def main():
    if "--fetch" in sys.argv:
        ab.fetch()

    print("=== Afrobarometer, Liberia ===")
    # regroup=True: the answers are grouped to the census's five below, so the
    # one-answer-two-spellings guard runs on the GROUPED column instead of the raw one.
    raw = ab.load(COUNTRY, expect_rounds=ROUNDS, regroup=True)
    print(f"\n  pooled: {len(raw):,} respondents with a religion answer, "
          f"{raw['category'].nunique()} distinct labels over six rounds")

    # ---- the raw answers, printed before anything is grouped ----
    ct = pd.crosstab(raw["category"], raw["round"])
    ct["all"] = ct.sum(axis=1)
    print("\n  every answer as it arrives (this is what CENSUS_CATEGORY groups):")
    print(ct.sort_values("all", ascending=False).to_string(max_colwidth=52))

    # ---- how much of `Christian only` is probing, which is why the grouping happens ----
    conly = raw[raw["category"].map(key) == "christian only"]
    by_round = (conly.groupby("round")["w"].sum() / raw.groupby("round")["w"].sum())
    print("\n  share answering `Christian only` rather than naming a denomination, by round:")
    for r, v in by_round.items():
        print(f"    R{r}  {v:6.1%}")
    print(f"    range {by_round.min():.1%} to {by_round.max():.1%} — the fieldwork, not the "
          "country. See the docstring.")
    if by_round.max() - by_round.min() < 0.15:
        raise SystemExit(
            "the `Christian only` share no longer swings between rounds, so the argument in "
            "the docstring for collapsing to the census's five categories no longer holds. "
            "Re-read it and decide deliberately rather than leaving this file as it is.")

    # ---- group to the census's own five categories ----
    k = raw["category"].map(key)
    unmapped = sorted(set(k) - set(CENSUS_CATEGORY))
    if unmapped:
        raise SystemExit(f"answers with no census category: {unmapped}. An answer that falls "
                         "through here is silently dropped — add it to CENSUS_CATEGORY "
                         "deliberately, with the reason.")
    stale = sorted(set(CENSUS_CATEGORY) - set(k))
    if stale:
        print(f"  CENSUS_CATEGORY covers answers this pool no longer has: {stale}")
    df = raw.copy()
    df["raw_category"] = raw["category"]
    df["category"] = k.map(CENSUS_CATEGORY)
    ab.assert_one_wording(df, COUNTRY)
    if sorted(set(df["category"])) != sorted(CENSUS_RELIGION):
        raise SystemExit(f"grouped to {sorted(set(df['category']))}, which is not the "
                         f"census's five: {sorted(CENSUS_RELIGION)}")

    # ---- harmonise the county labels BEFORE any test is run on them ----
    ck = df["geo_raw"].map(ckey)
    df["county"] = ck.map(NORM)
    print(f"\n  {df['geo_raw'].nunique()} distinct REGION labels over the six rounds -> "
          f"{df['county'].nunique()} counties")
    bad = sorted(df.loc[df["county"].isna(), "geo_raw"].astype(str).unique())
    if bad:
        raise SystemExit(f"REGION labels with no county: {bad}. A label that falls through "
                         "here is silently dropped and the split-half is then run on fewer "
                         "units than its bar assumes — add it to NORM deliberately.")
    dup = df.groupby(["round", "county"])["geo_raw"].nunique()
    if (dup > 1).any():
        raise SystemExit(f"one round uses two labels for the same county: "
                         f"{dup[dup > 1].to_dict()}")

    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    if len(lut) != N_UNITS:
        raise SystemExit(f"{LOOKUP} has {len(lut)} counties, expected {N_UNITS}")
    names = dict(zip(lut["name"], lut["geo_id"]))
    if sorted(set(df["county"])) != sorted(lut["name"]):
        raise SystemExit("the harmonised counties are not the 15 in lr_lookup.csv: "
                         f"{sorted(set(df['county']) ^ set(lut['name']))}")
    df["geo_id"] = df["county"].map(names)
    units = sorted(lut["geo_id"])
    pop = lut.set_index("geo_id")["pop"]
    if int(pop.sum()) != CENSUS_TOTAL:
        raise SystemExit(f"lr_lookup.csv sums to {int(pop.sum()):,}, not {CENSUS_TOTAL:,}")
    if sum(CENSUS_RELIGION.values()) != CENSUS_TOTAL:
        raise SystemExit("Table A13 transcription does not sum to the census total")

    # ---- the decode, tested without touching the religion column ----
    ab.held_out(df, pop, COUNTRY, pop_source="the 2022 census")

    # ---- the survey against the census's own national margins ----
    nat = ab.national(df)
    print("\n  the survey's national shares against the 2022 census, which is what the fit "
          "below corrects:")
    print(f"    {'category':<30}{'survey':>9}{'census':>9}{'ratio':>8}")
    for c in sorted(CENSUS_RELIGION, key=lambda c: -CENSUS_RELIGION[c]):
        s = float(nat.get(c, 0.0))
        cs = CENSUS_RELIGION[c] / CENSUS_TOTAL
        print(f"    {c:<30}{s:8.2%}{cs:9.2%}{s / cs:8.2f}")

    # ---- which categories carry their own geography ----
    large = ab.stability(df, nat, N_UNITS)
    small = [c for c in nat.index if c not in large]
    if sorted(large) != sorted(CARRIES):
        raise SystemExit(
            f"the split-half now selects {sorted(large)}, not {sorted(CARRIES)}. That is a "
            "change in what this country claims to know, not a bug — read the numbers above, "
            "then update CARRIES and the docstring deliberately.")

    # ---- the atlas check, on the survey's own shares ----
    by_unit = df.groupby(["geo_id", "category"])["w"].sum().unstack(fill_value=0.0)
    check_atlas(by_unit.div(by_unit.sum(axis=1), axis=0), lut)

    # ---- seed, then fit to both census margins ----
    #
    # NOT `ab.build`, which is the shared path for a country with no census margin to fit to.
    # It divides a unit's residual among the tail, and Grand Cape Mount's residual is exactly
    # zero: all 191 of its pooled respondents answered Christian or Muslim, so there is no
    # room to put its traditionalists in. Here the tail does not have to come out of a
    # residual, because the column margins are supplied by the census — so the seed carries
    # only the PATTERN (measured for the two categories that earned one, flat for the three
    # that did not) and the IPF sets every level.
    print("\n  fitting the county x religion table to two exact census margins:")
    unit_share = by_unit.div(by_unit.sum(axis=1), axis=0)
    seed = pd.DataFrame(index=units, columns=sorted(CENSUS_RELIGION), dtype=float)
    basis = {}
    for c in seed.columns:
        if c in large:
            seed[c] = unit_share[c].reindex(units).to_numpy()
            note = "county share, fitted to the census total for this religion"
        else:
            seed[c] = float(nat.get(c, 0.0))
            note = ("no county geography of its own, so the national share, fitted to the "
                    "census total for this religion")
        for u in units:
            basis[(u, c)] = note
    zero = [(u, c) for u in units for c in seed.columns if seed.loc[u, c] <= 0]
    if zero:
        # A zero seed cell stays zero through the IPF, which is a hard claim about a county.
        print(f"    seed cells at zero (they stay zero, §3.5 drops rather than invents): "
              f"{[(dict(zip(lut['geo_id'], lut['name']))[u], c) for u, c in zero]}")
    seed = seed.mul(pop.reindex(units), axis=0)
    fitted = ipf(seed, pop, pd.Series(CENSUS_RELIGION))
    counts = round_within_rows(fitted)

    if not (counts.sum(axis=1) == pop.reindex(units)).all():
        raise SystemExit("a county's drawn total is not its census population")
    drift = counts.sum(axis=0) - pd.Series(CENSUS_RELIGION).reindex(counts.columns)
    print("    rounding drift against Table A13, per category: "
          + ", ".join(f"{c} {int(d):+d}" for c, d in drift.items()))
    if drift.abs().max() > N_UNITS:
        raise SystemExit(f"rounding drift {drift.to_dict()} exceeds one person per county")

    # ---- write ----
    n_by = df.groupby("geo_id").size()
    out = counts.stack().rename("count").reset_index()
    out.columns = ["geo_id", "source_category", "count"]
    out["geo_level"] = "county"
    out["geo_name"] = out["geo_id"].map(dict(zip(lut["geo_id"], lut["name"])))
    out["basis"] = "self_id"
    out["year"] = YEARS
    out["source_id"] = SOURCE_ID
    out["note"] = out.apply(
        lambda r: ("2022 census county population and national religion totals, fitted with "
                   f"the county pattern of Afrobarometer rounds 4-9 pooled (n="
                   f"{int(n_by[r.geo_id])} in this county); "
                   f"{basis[(r.geo_id, r.source_category)]}"),
        axis=1)

    total = int(out["count"].sum())
    if total != CENSUS_TOTAL:
        raise SystemExit(f"drawn {total:,} against the census {CENSUS_TOTAL:,}")

    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, "
          f"{out['source_category'].nunique()} categories, {out['geo_id'].nunique()} counties)")

    # ---- what the file now says, which is what note_public has to reproduce ----
    drawn = out.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print("\n  national, as drawn:")
    for cat, n in drawn.items():
        print(f"    {n / total * 100:6.2f}%  {cat}  ({n:,})")
    share = (counts.div(counts.sum(axis=1), axis=0) * 100)
    nm = dict(zip(lut["geo_id"], lut["name"]))
    print("\n  as drawn, by county (the sample behind each in brackets):")
    print(f"    {'county':<18}{'Christian':>10}{'Muslim':>9}{'Trad':>7}{'None':>7}"
          f"{'Other':>7}{'n':>7}")
    for gid in share.sort_values("Muslim", ascending=False).index:
        s = share.loc[gid]
        print(f"    {nm[gid]:<18}{s['Christian']:9.1f}%{s['Muslim']:8.1f}%"
              f"{s['Traditional African Religion']:6.1f}%{s['No religion']:6.1f}%"
              f"{s['Other religion']:6.1f}%{int(n_by[gid]):7d}")
    m = counts["Muslim"]
    print(f"\n  most Muslims: {nm[m.idxmax()]} at {int(m.max()):,}; "
          f"highest Muslim share: {nm[share['Muslim'].idxmax()]} at "
          f"{share['Muslim'].max():.1f}%")
    print(f"  thinnest county {nm[n_by.idxmin()]} at n={int(n_by.min())}; "
          f"median n={int(n_by.median())}")


if __name__ == "__main__":
    main()
