# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _se_counts():
    """Sweden at NUTS 3: 21 lan, from two halves of one census table.

    Sweden records no religion for anybody. The population register was the Church of
    Sweden's until 1991 and the state has not carried the field since; SCB publishes no
    religion table of its own and has no maatwerk-style shelf of commissioned ones. What it
    does do is produce the Church of Sweden's membership figures FOR the church, per parish,
    kommun and lan, and the church publishes them: 5,627,932 members at 31/12/2021, 53.94%
    of the country. That is one denomination, so it cannot be the source; sources/se.py
    fetches it anyway and prints the comparison, which is the whole subject of the country.

      * **Swedish citizens, 9.57M.** ESS, `ctzcntr = Yes`. Rounds 5 to 8 are NUTS 3, 6,449
        people over 21 lan; rounds 9 and 11 are NUTS 2 and Sweden is absent from round 10.
        Each category is drawn at the lan where it passes the stability test there, and at
        its riksomrade (all six rounds) where only that level passes, which is Italy's
        split (§9as). se.csv has one row per lan either way.
      * **Foreign residents, 856k.** Eurostat's 2021 census table `cens_21ctz_r3`, 200 named
        citizenships at NUTS 3, crossed with Pew's composition for each origin country. This
        is 8.2% of Sweden and it carries more than half of the country's Muslims and two
        fifths of its Oriental Orthodox.

    **Both come out of the same census table**, which publishes `NAT` and `FOR` next to the
    named citizenships, so the halves partition the country by construction. 99.55% is drawn.

    **Three of the ten citizen categories carry their own lan shares and three their
    riksomrade's**; the other four are drawn at the national rate inside each lan's residual,
    per §9bi. taxonomy/se2024.py's REVIEW says what each loses.
    """
    from se2024 import resolve

    cit = pd.read_csv(HERE / "data" / "normalized" / "se.csv", dtype={"geo_id": str},
                      keep_default_na=False, na_values=[""])
    cit = cit[cit["geo_level"] == "nuts3"].copy()
    cit["node"] = cit["source_category"].map(resolve)
    unmapped = sorted(set(cit.loc[cit["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"se.csv has unmapped source categories: {unmapped}")

    ext = pd.read_csv(HERE / "data" / "normalized" / "se_foreign.csv",
                      dtype={"geo_id": str})
    ext = ext[ext["geo_level"] == "nuts3"]

    df = pd.concat([cit[["geo_id", "node", "count"]], ext[["geo_id", "node", "count"]]],
                   ignore_index=True)
    df["congregations"] = 0
    # A survey is not a count of anybody, a nationality model is not either, and the Orthodox
    # communion split is an imputation from a national register, so nothing here is
    # `measured` and §7 desaturates all of it.
    df["tier"] = "modelled"
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "tier"]]


def _se_place_weight(place):
    """Sweden's 290 kommuner, weighted by municipal population.

    The same weighter Greece and Finland use. The counting units are already 495,000 people
    each, so this matters less than it does in Greece or France, but it is what stops
    Norrbotten being one flat blob the size of Ireland with its dots spread over the
    mountains.

    **It is a population weight and not a religion one, and in Sweden that costs two
    nameable things.** Stockholms lan is drawn as a single composition over 2.4M people, so
    its 6.3% Muslim share spreads in proportion to where anyone lives and the real
    geography, the north-western suburbs against the inner city, is invisible. And
    Sodertalje, which is in Stockholms lan and holds most of Sweden's Syriac Orthodox
    population, gets its Oriental Orthodox dots at the county rate like everywhere else.
    SCB's population by country of birth per kommun is open and would let Sweden use the
    Italy weighter for the foreign half; that is a named improvement rather than a thing
    this build does. See sources/se.md §6.
    """
    if "pop" not in place.columns:
        print("  !! se_lau.gpkg has no `pop` column — run sources/se_geo.py")
        return None
    return _GrLauWeighter(place)


ENTRY = {
    "se": dict(
        name="Sweden",
        name_in="Sweden",
        source="ESS rounds 5-9 and 11 (citizens) + Eurostat census 2021 x Pew 2020 (residents)",
        basis="self-identification, sample survey (citizens); nationality-derived (residents)",
        note_public=(
            "**Sweden counts its largest church exactly and this map draws a different "
            "number.** Statistics Sweden produces the Church of Sweden's membership figures "
            "for it, parish by parish, and at the end of 2021 they came to 5,627,932 people, "
            "**53.94%** of everybody. The European Social Survey asks people whether they "
            "consider themselves as belonging to any particular religion, and the ones who "
            "answer Church of Sweden are **21.47%**. Neither is a mistake. Membership here is "
            "something you are born into by baptism and leave by filling in a form, and about "
            "three and a half million Swedes have not filled it in and also do not describe "
            "themselves as belonging to the church. Finland has the same split and it is "
            "seventeen points wide; Sweden's is thirty-two. "
            "**The two also disagree about where the Lutherans are, and that is the more "
            "interesting half.** Ranked by membership, the top of the country is Norrbotten "
            "at 67.6% and the north generally; ranked by what people say, the top is "
            "Kronoberg at **35.6%** and the south-west generally, with Kalmar, Halland and "
            "Jonkoping behind it. Nominal membership is highest in the north and stated "
            "belonging is highest in the Smaland free-church belt, which are two real "
            "geographies of the same church rather than a contradiction. Stockholm is last "
            "on both, at 45.0% of members and **13.3%** of answers. "
            "**63.5% of Sweden is drawn as belonging to no religion**, which is the largest "
            "such share on this map. It runs from **76.4%** in Gavleborg to 50.8% in "
            "Kronoberg. The survey offers no atheist or agnostic option, so everyone who "
            "answers no lands in one category and nothing here reaches the secular node; and "
            "this figure is a residual rather than a direct reading, because it is what is "
            "left of each county after the answers that earned their own geography. "
            "**Islam is 5.31% and more than half of it is counted rather than surveyed.** "
            "Sweden has 856,212 foreign citizens, 8.2% of the country, and the census counts "
            "them by nationality at county level; that half of the build is where the Syrian, "
            "Iraqi, Somali and Afghan arrivals of the last fifteen years are, and it is also "
            "why Islam is higher here than the citizen survey alone would say. Stockholm is "
            "the highest county at 7.3% and Jamtland the lowest at 2.3%. "
            "**The Orthodox answer is split three ways and Sweden is the country where that "
            "matters.** Asked which church, the survey offers one Orthodox box. MUCF, which "
            "pays the state grant to faith communities and counts their members to do it, "
            "shows that box covers 69,153 Eastern Orthodox, 72,133 Oriental Orthodox and "
            "9,542 in the Assyrian Church of the East, so the non-Chalcedonian half is the "
            "larger one, mostly the Syriac churches of Sodertalje. Every other country here "
            "with an undifferentiated Orthodox answer puts it on Eastern Orthodox, and doing "
            "that in Sweden would file about half the people in a communion they left in 451. "
            "**The free-church belt survives 130 respondents, which is the surprise here.** "
            "Sweden's frikyrkor are one box on the survey card and are reached by 130 people "
            "in four rounds, and the pattern is still strong enough to draw: **7.7%** of "
            "Orebro, 6.3% of Jonkoping and 4.2% of Vasterbotten against 3.1% nationally, and "
            "0.7% of Kalmar at the bottom. That is the Orebromissionen, the Svenska "
            "Alliansmissionen and the EFS coast, three revivals a century apart showing up in "
            "the right three places. **What it cannot do is name any of them**: "
            "Equmeniakyrkan, Pingst, the Evangeliska Frikyrkan, EFS and Svenska "
            "Alliansmissionen are a quarter of a million people between them and the survey "
            "offers no way to tell them apart. EFS is hidden twice over, because it works "
            "inside the Church of Sweden and its members answer the same way as everyone "
            "else in it. "
            "**What this cannot do.** The counting is 6,449 Swedish citizens interviewed "
            "between 2010 and 2016, a median of 203 per county, and only the Church of "
            "Sweden, Islam and the free churches show a county pattern that repeats across "
            "survey rounds. Catholics, the Orthodox and Jews only show one across Sweden's "
            "eight larger regions, counting 2,657 more interviews from 2018 and 2023, so "
            "every county gets its region's rate. Everything else is placed at the national "
            "rate inside each county's residual, and the map is not making a claim about "
            "where those people are. The Jewish figure rests on ten people, seven of them in "
            "Stockholm and none in the Gothenburg region, so Gothenburg's community is "
            "almost missing here."),
        how="survey, 9,106 people; foreign residents by nationality",
        grain="counties, 495,000 people on average",
        gap_share=0.004481,
        gap=("0.45% of Sweden: the 24,612 people, 0.24%, whose citizenship the 2021 census "
             "did not establish and who are therefore in neither half; and the 0.21% of "
             "Swedish citizens who were asked about religion and declined"),
        counts=_se_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "se" / "se_lau.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_se_place_weight,
        note="THE QUEUE PRICED SWEDEN AT NUTS 2 AND FOUR OF ITS SIX USABLE ROUNDS ARE NUTS 3. "
             "§11ai's block listed Sweden as *'NUTS-2 (8) to verify'*; `regunit` says rounds "
             "5, 6, 7 and 8 are NUTS level 3, the 21 lan, and rounds 9 and 11 are NUTS level "
             "2. Sweden is ABSENT FROM ROUND 10 entirely. Run the same stability test at "
             "both levels: the coarse one brings back the Catholics (+0.833) and the "
             "Orthodox (+0.888) and takes `Annan protestantisk forsamling` from +0.333 to "
             "**-0.119**, because NUTS 2 puts Jonkoping at 6.25% in the same unit as Kalmar "
             "at 0.69%. **So it is drawn at both, split by category, which is Italy's "
             "construction (§9as); rebuilt so on 2026-09-14 with Anita's approval.** Svenska "
             "kyrkan, Islam and the free churches take their own lan's share (rounds 5-8); "
             "Catholics, Orthodox and Jews take their riksomrade's share inside each lan "
             "(all six rounds). No religion passes at the riksomrade too and stays the "
             "residual, because fixing it there leaves the tail negative in 6 lan. "
             "sources/se.md §2 has the table. "
             "**THE SPLIT-HALF USES §9cy's NULL, NOT `spearman_null`'s BAR, AND SWEDEN IS "
             "WHY THAT MATTERS.** Belgium brought the test to ESS the same day and rebuilt "
             "its null: resample by ROUND, take the MEDIAN Spearman over every split, "
             "permute the unit labels per round. Sweden shows what one halving is worth. "
             "Four rounds admit three distinct halvings, and on `Svenska kyrkan` at 21 lan "
             "they give +0.125, +0.434 and +0.458 against a fixed bar of +0.3701, so the "
             "country's largest religious category is drawn or flattened depending on which "
             "two rounds you happened to put together. **And Sweden adds one requirement to "
             "§9cy's test: a spatial chi-square at 0.05, which can only make a category "
             "fail.** `Annan icke-kristen religion` (21 respondents) and `Osterlandsk "
             "religion` (23) both CLEAR the permutation test at p = 0.018 and 0.022, with "
             "chi-squares of 0.32 and 0.40 — the lan are not distinguishable for either. A "
             "rank correlation over a column that is zero in most units is decided by how "
             "the ties break, so a small category does not merely lose power against a rank "
             "test, it can be passed by one. Three of ten carry. Whether Greece, Finland, "
             "France, Germany and Italy should be re-run the same way is `ask/012-be`, filed "
             "by Belgium the same day; Sweden's evidence is appended to it rather than asked "
             "again. "
             "**§11k's 'Sweden. SCB carries nothing' is true of SCB's catalogue and false of "
             "what SCB produces.** §9cu's instruction was to check the office's "
             "custom-tabulation shelf before building any ESS country, and Sweden has no "
             "shelf: SCB publishes no browsable archive of past commissioned work, ordering "
             "is bespoke and paid, and a site search for `trossamfund` returns civil-society "
             "accounts and occupational medians. But the Church of Sweden's own "
             "`Medlemsutveckling` PDF says in its header that the population and membership "
             "figures in it are produced by SCB on the church's commission, for every parish, "
             "all 290 kommuner and all 21 lan. The commissioned religion tabulation exists "
             "and lives on the customer's website. "
             "**The variable is `rlgdnase`; `rlgdnse` does not exist** and raises "
             "E201VariableNotFound, which is Finland's `rlgdnafi` trap again. Its labels come "
             "back in SWEDISH even with `metadataLanguage:\"en\"`, because ESS translates the "
             "harmonised variables and leaves the country-specific ones in the field "
             "language. "
             "**The free-church result is the one to look at.** `Annan protestantisk "
             "forsamling` is 130 respondents in four rounds and it passes both tests, coming "
             "out at 7.71% of Orebro, 6.25% of Jonkoping and 4.22% of Vasterbotten against "
             "3.05% nationally, with Kalmar last at 0.69%. Orebromissionen, Svenska "
             "Alliansmissionen and the EFS coast, in the right three places, recovered from "
             "a sample nobody would have bet on. It is the strongest evidence here that the "
             "instrument is working, and it is also the reason the two noise passes had to be "
             "refused rather than the whole tier being distrusted. "
             "**The boundaries cost nothing**: the GISCO LAU 2021 bundle on disk since Poland "
             "carries all 290 kommuner with their NUTS 3 code and population, 290 of 290 in "
             "both directions. Finland and Portugal were the others.",
    ),
}
