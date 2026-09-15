# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _jo_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Jordan is Egypt's problem at a tenth of the scale. Ma'an, Mafraq and Aqaba are 66,288 km2
    between them, 74.7% of the country's land, and hold 9.5% of its people; the badia east and
    south of the highlands is empty. Drawn flat, the difference between Madaba and Mafraq gets
    painted onto basalt desert instead of onto the towns (sources/jo_grid.py).
    """
    return _kontur_place_weight(place, "jo_hexes.gpkg", "sources/jo_grid.py")


def _jo_counts():
    """Arab Barometer waves II to VIII pooled, at governorate: 2 categories, all 12 units.

    THE COUNTRY IS A SURVEY ON DOS'S OWN ESTIMATE AND EVERY ROW IS `modelled` (§7b). `Q1012`
    gives a governorate share; the Department of Statistics' Table 2.2 gives the number of
    people it applies to, at end-2025. No magnitude is invented: every person drawn is a
    person DOS counts in that governorate, and the survey only decides the column (§14.4 rule
    1, the construction sources/kz.py uses).

    JORDAN ASKED THE QUESTION TWICE AND PUBLISHED NEITHER ANSWER. DOS prints its own
    questionnaires and both carry a religion item: the 2015 form's person block runs 201-216
    with 208 `Religion`, `1.mustim 2.chistian 3.other` in DOS's own spelling, and the 2004
    private household register asks `al-diyana` with Islam, Christianity and other in the same
    position. The 2015 census's published tables number 133 across ten sections and the 2004
    census's 165 across eight; religion is in neither set, the jorinfo.dos.gov.jo databank's
    520 tables carry no religion variable, and Jordan is absent from the UNSD Demographic
    Yearbook's religion table. So the answers exist and nobody outside DOS has seen them.

    DRAWN AT GOVERNORATE, following Anita's Egypt ruling rather than re-asking it. §14.4 rule
    2 is the same question here as it was in ask/answered/001-eg -- a religious minority
    mapped finer than its own state publishes -- and her answer there was to draw at
    governorate because governorates are big units. Jordan's are bigger in every sense that
    matters: twelve of them for 11.9 million people, the smallest holding 120,300, against
    Egypt's twenty-seven. And Jordan's Christians are not situated as Egypt's Copts are; the
    constitution reserves parliamentary seats for them and the community is publicly counted
    by its own churches. Nothing finer exists anyway, because the survey cuts by governorate
    and carries nothing below it.

    ALL TWELVE GOVERNORATES ARE SAMPLED IN ALL NINE WAVES, so there is no gap: every one of
    DOS's 11,937,000 people is drawn. Eleven respondents refused the question and leave the
    survey's universe rather than the country's, which is why there is no `gap_share`.

    BOTH ANSWERS CARRY THEIR OWN GEOGRAPHY AND THERE IS NO TAIL, as in Egypt, and for the same
    reason: the card has two boxes, both clear the 1% eligibility floor and both clear the
    split-half. THE PASS IS THIN AND SAYING SO IS PART OF THE ENTRY. Spearman +0.617 against a
    bar of +0.591 on twelve units, and dropping Balqa alone takes it to +0.509. It is §14.16's
    test applied as written, and three things the test cannot see stand behind it: the twelve
    governorates differ at p=2.6e-18, Balqa with Madaba and Ajloun read 3.50% Christian
    against 1.41% elsewhere at p=4.1e-10, and the ordering the survey returns is the one
    Jordan's own Christian geography would predict. The alternative is not a more careful map:
    with neither answer carrying, every governorate would be drawn at the national rate, which
    the data contradicts at p=2.6e-18.

    THE SURVEY IS MOSTLY A SURVEY OF CITIZENS AND THE DENOMINATOR COUNTS EVERYBODY. Roughly
    three residents in ten are not Jordanian nationals; the 2015 census counted 9,531,712
    people of whom 2,918,125 were non-Jordanian. Wave IV is the only wave that sampled them
    (303 Syrians in 1,500 respondents) and it measures them as different: 0 of 303 Christian
    against 28 of 1,197 among Jordanian- and Palestinian-origin respondents in the same wave,
    p=0.014. So applying the pooled share to the whole resident population makes the Christian
    figure a ceiling. Nothing corrects for it, because no published source gives the religion
    of Jordan's non-citizens at any geography and §14.4 rule 1 forbids inventing one.

    THE UNIVERSE IS ADULTS AND THE DOTS ARE EVERYBODY. The Arab Barometer interviews people
    aged 18 and over in every wave; the shares are applied to the whole population, which
    assumes Jordan's children are distributed like its adults. Drawing only the adults would
    leave a third of a young country blank, and §6.12 is about how badly a blank reads on a
    dot map.
    """
    from jo2024 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "jo.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])

    lut = pd.read_csv(HERE / "data" / "geo" / "jo" / "jo_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"jo.csv governorates with no polygon: {missing} -- re-run "
                         "sources/jo_geo.py, the lookup is stale")
    if df["unit"].nunique() != 12:
        raise SystemExit(f"{df['unit'].nunique()} governorates, expected 12")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"jo.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    # EVERY row, without exception -- there is no measured tier in this country (§7).
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "jo": dict(
        name="Jordan",
        source="Arab Barometer, nine waves 2010 to 2024 (Arab Barometer, Princeton "
               "University), against the Department of Statistics' own governorate population "
               "estimates for end-2025",
        basis="self-identification, adults 18 and over",
        note_public=(
            "**Jordan has asked about religion in its last two censuses and published the "
            "answer from neither.** The Department of Statistics prints its own forms, and "
            "both carry the question: item 208 on the 2015 census schedule is `Religion`, "
            "with boxes for Muslim, Christian and other, and the 2004 household register asks "
            "the same thing in the same place. Neither census's published tables include it, "
            "the department's online databank has no religion table among its 520, and Jordan "
            "has never reported one to the UN. So this map is a survey standing where a "
            "census would be: **14,906 people** who named a religion across nine rounds of "
            "the Arab Barometer between 2010 and 2024, pooled, with each governorate's "
            "answers applied "
            "to the department's own end-2025 estimate of that governorate's population. "
            "Nobody counted this, and the dots are drawn desaturated to say so. "
            "**Christians are 1.40% of the country as drawn, and Balqa is where they are "
            "densest.** Balqa reads **4.33%**, then Ajloun at 3.67% and Karak at 2.32%, "
            "against 0.66% in Zarqa and 0.25% in Ma'an. Balqa contains Fuheis, Mahis and "
            "Salt, three of the best known Christian towns in the country, although the "
            "survey asks nobody which town they live in and the 4.33% is the governorate's. "
            "Amman is "
            "only 1.57% and still holds **78,590** of the 166,950 Christians on this map, "
            "which is nearly half of them, so the highest share and the largest community are "
            "not in the same place. The whole ordering rests on **245 Christian interviews** "
            "spread across twelve governorates and fourteen years, so read the top three as a "
            "group rather than as a ranking. "
            "**Two governorates are drawn with no Christians at all, and that is the sample "
            "rather than the country.** None of Jerash's 480 respondents and none of "
            "Tafilah's 386 answered Christian, which on those samples is consistent with "
            "anything up to about 0.6% and 0.8%. Jerash has a Christian community and this "
            "survey did not reach it. Madaba is the other one to read carefully: it is drawn "
            "at **0.51%**, the lowest of the settled governorates, and Madaba town is one of "
            "the oldest Christian centres in the country, its mosaics included. 455 "
            "interviews cannot tell 0.5% from 2%. "
            "**The survey is mostly a survey of citizens and the population it is applied to "
            "is everybody.** About three residents in ten are not Jordanian nationals, mostly "
            "Syrians, and only one of the nine rounds sampled them: in that round, fielded in "
            "2016, none of the **303 Syrians** interviewed was Christian, against 2.34% of "
            "the 1,197 Jordanians and Palestinians beside them. Applying one set of shares to "
            "the whole resident population therefore makes 1.40% a ceiling rather than a "
            "middle estimate. Nothing here corrects for it, because no published source gives "
            "the religion of Jordan's non-citizens at any geography. "
            "**The card has two boxes and that is the limit of what this map can say.** One "
            "Muslim box and one Christian box, in all nine rounds, so nothing here separates "
            "Sunni from anything else, and nothing separates the Greek Orthodox from the "
            "Melkite Greek Catholics, the Latins, the Syriacs and the Protestants, even "
            "though that split is a real feature of Jordanian Christianity. Every one of the "
            "**14,917** people interviewed answered Muslim or Christian or declined to "
            "answer: the five most recent rounds offered a box for having no religion, and "
            "not one of their 8,034 respondents ticked it."),
        how="survey, nine rounds 2010 to 2024 pooled",
        grain="governorates, 995,000 people on average",
        counts=_jo_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "jo" / "jo_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_jo_place_weight,
        note="THE STATE ROUTE WAS TESTED FIRST AND IS CLOSED, NOT MISSING. Jordan's 2015 and "
             "2004 censuses both ask religion -- DOS publishes both questionnaires and the "
             "item is there, three boxes, in the person block -- and neither census's table "
             "set contains it (2015: 133 tables in ten sections; 2004: 165 in eight). The "
             "jorinfo.dos.gov.jo PxWeb databank was enumerated in full, 520 tables, and the "
             "thirteen that mention a religion word are all ISIC division 94, activities of "
             "religious organisations, in economic tables. Jordan is absent from the UNSD "
             "Demographic Yearbook's religion table. There is no HDX COD for Jordan at all, "
             "so the boundaries are geoBoundaries and the populations are DOS's own, which is "
             "better than COD-PS would have been anyway. "
             "DRAWN AT GOVERNORATE FOLLOWING ANITA'S EGYPT RULING (ask/answered/001-eg) "
             "rather than re-asking it. queue.md's §11ag row records her unblocking this "
             "group with `build them, and decide what to show once there is something to "
             "show`; Lebanon is the one she asked to have raised with her and Jordan is not. "
             "Twelve governorates for 11.9 million people is coarser than Egypt's "
             "twenty-seven, and Jordan reserves parliamentary seats for Christians, so rule "
             "2's concern is weaker here in both directions. "
             "WAVE II WAS INVISIBLE TO THE ARAB BAROMETER MODULE UNTIL THIS COUNTRY. It "
             "spells its country labels `8. Jordan`, `5. Egypt`, `17. Saudi Arabia`, so "
             "ab.load's exact-match filter found no rows and skipped the wave in silence for "
             "every country ever built from this survey. It is 1,188 of Jordan's respondents "
             "and 39 of its 245 Christians. Egypt is NOT rebuilt on it: that would move an "
             "already-drawn country's numbers, which is Anita's call and not a side effect of "
             "a shared-module fix, so sources/eg.py now pins its four waves with the reason. "
             "Wave II reads 5.83% Christian for Egypt against the drawn 5.93%, so nothing "
             "about it looks like a correction waiting to happen. "
             "WAVE I IS LEFT OUT FOR THE GEOGRAPHY AND NOT THE DECODE. It asks religion as "
             "`q711` rather than `Q1012` because it predates the questionnaire renumbering, "
             "and 1,142 of its 1,143 Jordanians answered -- but the file has 181 columns and "
             "not one of them is subnational, so it cannot enter a pool cut by governorate. "
             "Unweighted it reads 1.57% Christian, inside the band the other nine occupy. "
             "THE GOVERNORATE CODE MEANS THREE DIFFERENT THINGS AND IS THEREFORE A WITNESS. "
             "Waves II and III number Jordan's governorates 3501 to 3512 in the official "
             "order; waves V, VII and VIII use 800 followed by Jordan's own governorate "
             "number, so 80011 is Amman and 80034 is Aqaba; waves IV and VI use an arbitrary "
             "1 to 12 order that is not even shared between them. So the code can never be "
             "the pooling key, and in the six waves where it decodes it checks the harmonised "
             "names instead: 10,175 respondents, all agreeing, which no permutation of the "
             "name table could survive. "
             "THE SPLIT-HALF PASSES THINLY AND THE ENTRY SAYS SO. Spearman +0.617 against "
             "§14.16's bar of +0.591 on twelve units; leave-one-out ranges +0.509 to +0.717 "
             "and dropping Balqa alone takes it under. The bar is applied as written and was "
             "not moved. What stands behind the verdict is that the governorates differ at "
             "p=2.6e-18 and that Balqa, Madaba and Ajloun together read 3.50% against 1.41% "
             "elsewhere at p=4.1e-10. "
             "THE HELD-OUT DECODE IS PINNED at r = +1.000 between the survey's governorate "
             "shares of respondents and DOS's governorate populations, which none of 20,000 "
             "random pairings reaches. The Arab Barometer quota-samples by governorate, so a "
             "high correlation is expected; what it tests is that the names were joined to "
             "the right polygons.",
    ),
}
