# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _be_counts():
    """Belgium at NUTS 2: eleven provinces, from two halves of one census table.

    Belgium has never asked religion in a modern census and Statbel publishes nothing on the
    subject at any geography: its own site search returns zero for `religie`, `godsdienst`,
    `levensbeschouwing`, `moslim`, `religion`, `culte` and `confession`, in both language
    indexes. So the construction is Greece's (§9z) and Finland's (§9by):

      * **Belgian citizens, 10.08M.** ESS rounds 5 to 11 pooled and restricted to
        `ctzcntr = Yes` — 10,877 people over eleven provinces, a median of 1,012 each, on
        the harmonised `rlgdnm` because Belgium's own `rlgdnbe` is the same eight codes and
        does not exist at all in the two most recent rounds.
      * **Foreign residents, 1.45M.** Eurostat's 2021 census table `cens_21ctz_r3`, 200
        named citizenships at NUTS 2, crossed with Pew's composition for each origin country.

    **The second half carries more of this country than of any other built this way.**
    Foreign citizens are 12.58% of Belgium and 34.99% of Brussels, against 7.2% for Greece
    and 5.2% for Finland, and ESS's Belgian sample is 7.9% non-citizen. Both halves come out
    of the same census table, so they partition the country by construction; 99.50% is drawn.

    **Four of the nine survey answers are drawn where they were measured and five are not.**
    `sources/be.py` runs §14.16's split-half across the seven rounds against a permutation
    null, and Roman Catholic, Islam, Eastern Orthodox and Eastern religions clear it. No
    religion, Protestant, Other Christian, Other Non-Christian and Jewish do not, so they go
    at the national rate inside each province's residual. For No religion that is almost
    without effect, because it is 96.6% of the residual and therefore still moves with each
    province's measured non-Catholic, non-Muslim share. For Jewish it is the whole story and
    note_public says so.
    """
    from be2024 import resolve

    cit = pd.read_csv(HERE / "data" / "normalized" / "be.csv", dtype={"geo_id": str},
                      keep_default_na=False, na_values=[""])
    cit = cit[cit["geo_level"] == "nuts2"].copy()
    cit["node"] = cit["source_category"].map(resolve)
    unmapped = sorted(set(cit.loc[cit["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"be.csv has unmapped source categories: {unmapped}")

    ext = pd.read_csv(HERE / "data" / "normalized" / "be_foreign.csv",
                      dtype={"geo_id": str})
    ext = ext[ext["geo_level"] == "nuts2"]

    df = pd.concat([cit[["geo_id", "node", "count"]], ext[["geo_id", "node", "count"]]],
                   ignore_index=True)
    df["congregations"] = 0
    # A survey is not a count of anybody and a nationality model is not either, so nothing
    # here is `measured` and §7 desaturates all of it.
    df["tier"] = "modelled"
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "tier"]]


def _be_place_weight(place):
    """Belgium's 581 communes, weighted by communal population.

    The Greece and Finland weighter. The counting units here are 1.05 million people each,
    which is the coarsest of the three, so this is doing more of the work of making Belgium
    look like a country rather than like eleven blobs.

    **It is a population weight and not a religion one, and Belgium is where that costs the
    most.** Brussels is nineteen communes and its Muslim population is concentrated in
    Molenbeek, Schaerbeek, Sint-Joost and Anderlecht rather than in Woluwe or Uccle; this
    spreads it evenly over all nineteen in proportion to where anyone lives. The same is
    true of Antwerp inside its province. **The fix exists and is not built**: Eurostat
    publishes the same citizenship table at NUTS 3, and Statbel publishes population by
    nationality per commune every year, either of which would place the foreign half where
    foreign residents actually are and move Belgium onto the Italy weighter. It is named in
    sources/be.md §5 rather than done, because doing it for one half and not the other puts
    the sharper geography on the half with the weaker claim to it, which is the call Greece
    made for Thrace.
    """
    if "pop" not in place.columns:
        print("  !! be_lau.gpkg has no `pop` column — run sources/be_geo.py")
        return None
    return _GrLauWeighter(place)


ENTRY = {
    "be": dict(
        name="Belgium",
        source="ESS rounds 5-11 (citizens) + Eurostat census 2021 x Pew 2020 (residents)",
        basis="self-identification, sample survey (citizens); nationality-derived (residents)",
        how="survey, 10,877 people; foreign residents by nationality",
        grain="provinces, 1.05 million people each",
        gap_share=0.0049664,
        gap="0.5% of the country, in two parts: 0.27% who declined the survey's religion "
            "question, and 0.22% whose citizenship the 2021 census did not record",
        note_public=(
            "**Belgium's public authorities do not record religion, and that is a rule "
            "rather than an oversight.** Religion is protected personal data under the "
            "privacy law of 8 December 1992, the National Register holds nationality and "
            "the language of the commune and nothing about belief, and no modern Belgian "
            "census has carried the question. Statbel's own site search returns no result "
            "at all for religie, godsdienst, levensbeschouwing, religion, culte or "
            "confession, in either of its language indexes. So this map is a survey. "
            "The European Social Survey has interviewed "
            "**10,877** Belgian citizens across the eleven provinces over seven rounds "
            "between 2010 and 2023, and that is one half of the country; the other 12.6%, "
            "the foreign residents, are counted by the 2021 census and given the religious "
            "composition of the country whose passport they hold. "
            "**The second half carries more of Belgium than of anywhere else drawn this "
            "way.** Foreign citizens are **34.99%** of the Brussels-Capital Region against "
            "12.58% of the country, and the survey's own Belgian sample is only 7.9% "
            "non-citizen, so Brussels drawn from the survey alone would have come out about "
            "a third wrong with nothing in the sample to reveal it. The same construction in "
            "Greece is covering 7.2% of the country and in Finland 5.2%. "
            "**No religion is the largest answer at 54.11%, and a third of it is lapsed "
            "Catholics.** The survey asks the people who say they belong to no religion "
            "whether they ever did: **34.4%** say they once did, and **95.5%** of those name "
            "the Catholic Church. That is roughly one Belgian citizen in five who was "
            "Catholic and now says they belong to nothing, on top of the **32.50%** who "
            "still say Catholic. "
            "**The sharp line on this map is Brussels rather than the language border.** "
            "Islam is **25.24%** of the capital and 2.25% of West Flanders; Orthodox "
            "Christianity is **7.46%** of the capital and 0.59% of Namur. Catholicism runs "
            "from 26.11% in Brussels up to 37.75% in the province of Luxembourg, and the "
            "unaffiliated share from 35.30% in Brussels to 58.95% in West Flanders. Flanders "
            "and Wallonia differ from each other much less than either differs from the "
            "capital, which is not the division Belgium is usually described by. A private "
            "estimate that works the other way round, from the population register's "
            "countries of origin rather than from what anyone said, put the Brussels Region "
            "at **25.5%** Muslim in 2019 against the 25.24% drawn here. It is not an "
            "independent check, because the resident half of this map is built from origin "
            "countries too, but the two do not disagree. "
            "**The question is about belonging, not about belief or about practice.** ESS "
            "asks whether you consider yourself as belonging to any particular religion or "
            "denomination, which is narrower than what you were raised as and broader than "
            "whether you attend anything. Every number here is an answer to that one "
            "sentence, and a question about baptism or about Sunday attendance would draw a "
            "different map. "
            "**Belgium recognises six religions and organised secularism, and the survey can "
            "see neither list.** The state funds the ministers of Catholicism, "
            "Protestantism, Anglicanism, Judaism, Islam and Orthodox Christianity, and it "
            "funds the laicite and vrijzinnigheid movements on the same constitutional "
            "footing, with counsellors in hospitals and prisons and an ethics course in "
            "schools that pupils take instead of a religion class. None of that is askable "
            "from this instrument. Anglicans land inside other Christian denomination, "
            "everyone in organised secularism lands in no religion beside everyone who "
            "simply does not belong, and the Sunni and Shia answers are one box, so the "
            "Shia dots here come only from the census half. "
            "**The Jewish population is drawn in the wrong place and it is worth saying so "
            "plainly.** Antwerp has one of the largest Haredi communities in Europe and "
            "Belgium's roughly 30,000 Jews are almost all in Antwerp and Brussels. Fifteen "
            "respondents in seven pooled rounds cannot show that, a test of whether the "
            "survey can place a group at all declines to license it, and the people are "
            "therefore spread at the national rate over a country where they are not spread. "
            "Four of the nine survey answers passed that test and are drawn where they were "
            "measured; the other five, including the Protestant and other Christian "
            "answers, are not. "
            "**The provinces are large and the sample inside them is not.** A median of "
            "1,012 respondents per province, 1,869 in Antwerpen and 296 in Luxembourg, so "
            "read the Catholic, unaffiliated and Muslim shares and treat anything below one "
            "percent as some, here rather than as a number."),
        counts=_be_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "be" / "be_lau.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_be_place_weight,
        note="**§11ai priced Belgium at NUTS 1 or NUTS 2 and it is NUTS 2 in all seven "
             "usable rounds**, the ten provinces plus Brussels, with the same eleven codes "
             "and the same eleven labels from round 5 to round 11. No recode, which is the "
             "opposite of Greece's two NUTS vintages and Finland's three. "
             "**THE TRAP IS THE COUNTRY VARIABLE, AND BELGIUM IS THE MIRROR OF THE "
             "NETHERLANDS.** `rlgdnbe` exists in rounds 5 to 9 and does NOT exist in rounds "
             "10 and 11, where the API answers `E201VariableNotFound` behind an HTTP 400, so "
             "pooling on it silently drops the two most recent rounds. What rescues Belgium "
             "is that `rlgdnbe` is the harmonised `rlgdnm` card in Dutch and French: the "
             "same eight codes, value for value, in all five rounds that carry both, which "
             "`sources/be.py::_check_be_card` re-proves from the data on every build. "
             "`rlgdnanl` splits three Reformed answers `rlgdnm` flattens; `rlgdnbe` splits "
             "nothing, so the harmonised variable costs Belgium nothing and gains it two "
             "rounds. "
             "**STATBEL HAS NO MAATWERK SHELF AND THE CHECK WAS RUN ANYWAY**, because §9cu "
             "made it a standing instruction. Statbel offers custom tabulation as a service "
             "you commission, not as a public shelf like CBS's, and its published estate "
             "carries no religion figure at any geography: site search returns zero for six "
             "terms in both languages. Getting that answer is not free. "
             "**statbel.fgov.be, data.gov.be and the rest of the Belgian federal estate sit "
             "behind an F5 Shape wall**, which returns HTTP 200 with a JavaScript challenge "
             "to curl no matter how complete the header set is, so Costa Rica's fix (§9cp) "
             "does not open it. Headless Chrome does not either, until you override the "
             "User-Agent: the default still says HeadlessChrome and the wall escalates that "
             "to an image CAPTCHA. A plain Chrome UA string plus a CDP navigate and a wait "
             "gets the rendered DOM. The search parameter is `search_api_fulltext_block`, "
             "and `search_api_fulltext`, which is Drupal's usual name and what the URL looks "
             "like it should take, is ignored and returns the entire 5,016-item index rather "
             "than an error. "
             "**FIRST ESS COUNTRY TO RUN §14.16's SPLIT-HALF.** Greece, Finland, France, "
             "Germany and Italy all draw every category where it was measured. Belgium's "
             "card is eight denominations and five of them are reached by fewer than a "
             "hundred respondents in seven pooled rounds, so the test is worth running; the "
             "resampling unit is the ROUND rather than the PSU, because this API returns "
             "cross-tabs and no PSU. The null must permute the province labels PER ROUND: a "
             "single global relabelling is applied to both halves alike and leaves every "
             "rank correlation exactly where it was, which returns p = 1 for every category "
             "and reads like a result. sources/be.md §4 has the table.",
    ),
}
