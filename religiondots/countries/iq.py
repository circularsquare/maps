# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _iq_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Anbar, Muthanna and Najaf are 217,674 km2 between them, 50.1% of Iraq's land, and hold
    10.8% of its people. Drawn flat, the Sunni/Shia line that is the whole content of this
    country would be painted across the western badiya and the southern desert instead of
    along the Tigris and the Euphrates, where everybody lives (sources/iq_grid.py).
    """
    return _kontur_place_weight(place, "iq_hexes.gpkg", "sources/iq_grid.py")


def _iq_counts():
    """Arab Barometer waves V, VI-3, VII and VIII pooled, at governorate: 7 categories, 18 units.

    THE COUNTRY IS A SURVEY ON THE 2024 CENSUS AND EVERY ROW IS `modelled` (§7b). `Q1012`
    crossed with `Q1012A` gives a governorate share; the census enumerated 20-21 November 2024
    gives the number of people it applies to. No magnitude is invented: every person drawn is
    a person the census counts in that governorate, and the survey only decides the column
    (§14.4 rule 1).

    IRAQ COUNTED ITSELF AND PUBLISHED NOTHING ABOUT RELIGION. The 2024 census is the first
    full count since 1987 and returned 46,118,793 people; its published tables are governorate
    crossed with urban/rural, sex and age, and nothing else. The reporting on whether religion
    or sect survived onto the form conflicts and does not need resolving, because nothing was
    published either way. Iraq is absent from the UNSD Demographic Yearbook's religion table
    and its last census to publish religion at all was 1987. §11r closed the state route on
    the abstract itself rather than on a search.

    DRAWN AT GOVERNORATE, following Anita's Egypt ruling rather than re-asking it.
    ask/answered/001-eg ruled that a religious minority may be drawn at governorate because
    governorates are big, and sources.md §11af recorded before either was built that Jordan,
    Lebanon, Iraq and Yemen inherit that ruling. Iraq's eighteen average 2.56 million people,
    which is more than twice Egypt's twenty-seven. Nothing finer exists anyway: the survey
    cuts by governorate and carries nothing below it.

    AND §14.4 RULE 2's ACUTE CASE IS ANSWERED BY THE INSTRUMENT RATHER THAN BY A CHOICE. The
    groups the rule is about here are the Yazidis, the Christians and the Sabean-Mandaeans,
    and none of them is placed anywhere: 25 Christians and 27 others in 8,335 respondents are
    both far under §11ad's 1% eligibility floor, so they fail the test that decides whether a
    category carries its own geography and are spread at the national rate over all eighteen
    governorates. The map says these communities exist and says nothing about where they live.
    What IS drawn with a geography is the Sunni/Shia distribution, which is among the most
    published facts about Iraq and is §14.2's reflect case rather than its reveal case.

    NEARLY A QUARTER OF THE COUNTRY IS DRAWN ON THE BARE `islam` NODE AND IT IS NOT SHARED
    OUT. 23.7% of Iraqis in this pool are Muslims who declined a branch, on three answers:
    `Just a Muslim`, which is the card's own wording, plus the sect card's `Other` and the
    people who refused the follow-up after answering Muslim. Russia is the precedent, where
    4.66% answered "I profess Islam, but am neither Sunni nor Shia" and stay on the parent.
    Splitting them at each governorate's observed Sunni:Shia ratio would invent a magnitude no
    source publishes and would do it on the most instrument-sensitive quantity in the file:
    the undifferentiated share runs 17.7%, 42.7%, 27.1% and 21.2% across the four waves.

    THOSE PEOPLE ARE NOT DISTRIBUTED AT RANDOM, WHICH IS THE THING TO KNOW ABOUT THEM. The
    answer passed the split-half in its own right (+0.766) and runs 45.9% in Diyala, 40.1% in
    Salah al-Din, 39.6% in Nineveh, 33.2% in Kirkuk and 30.5% in Anbar against 2.6% in Erbil
    and Duhok and 7.0% in Najaf. Declining to name a sect is a Baghdad and mixed-belt
    behaviour, so the drawn Sunni share in Baghdad, Diyala, Kirkuk and Nineveh is a floor.

    ALL EIGHTEEN GOVERNORATES ARE SAMPLED, so there is no gap: every one of the census's
    46,118,793 people is drawn. Two respondents answered Atheist on the one card of four that
    offered the box and six refused the religion question itself; those eight leave the
    survey's universe rather than the country's, which is why there is no `gap_share`.

    THE THREE LARGE ANSWERS CARRY THEIR OWN GEOGRAPHY AND THE PASS IS NOT THIN. Spearman
    +0.917 for Shia, +0.965 for Sunni and +0.766 for Just a Muslim, against §14.16's bar of
    +0.4014 on eighteen units, and no single governorate's removal takes any of them near it.
    The survey also reproduces Iraq's known sectarian map without being told it, which
    sources/iq.py asserts rather than admires: all nine southern and mid-Euphrates
    governorates come out majority Shia among those who name a branch and all six western and
    Kurdish ones majority Sunni, 98.2% against 1.5%.

    THE UNIVERSE IS ADULTS AND THE DOTS ARE EVERYBODY. The Arab Barometer interviews people
    aged 18 and over; the shares are applied to the whole population, which assumes Iraq's
    children are distributed like its adults. 35.9% of Iraqis are under 15, so drawing only
    the adults would leave a third of the country blank, and §6.12 is about how badly a blank
    reads on a dot map.
    """
    from iq2024 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "iq.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])

    lut = pd.read_csv(HERE / "data" / "geo" / "iq" / "iq_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"iq.csv governorates with no polygon: {missing} -- re-run "
                         "sources/iq_geo.py, the lookup is stale")
    if df["unit"].nunique() != 18:
        raise SystemExit(f"{df['unit'].nunique()} governorates, expected 18")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"iq.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    # EVERY row, without exception -- there is no measured tier in this country (§7).
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "iq": dict(
        name="Iraq",
        source="Arab Barometer, four waves 2018 to 2024 (Arab Barometer, Princeton "
               "University), against the 2024 Population and Housing Census (Central "
               "Statistical Organisation)",
        basis="self-identification, adults 18 and over",
        note_public=(
            "**Iraq counted itself in November 2024 and did not publish a word about "
            "religion.** The census was the country's first full count since 1987 and "
            "returned **46,118,793** people; what the Central Statistical Organisation "
            "printed from it is population by governorate crossed with urban or rural, sex "
            "and age, and nothing else. Accounts of whether sect and religion survived onto "
            "the form disagree, and it makes no difference, because neither was published. "
            "The last Iraqi census to publish religion was 1987 and Iraq has never reported "
            "one to the UN. So this map is a survey standing where a census would be: "
            "**8,335 people** interviewed across four rounds of the Arab Barometer between "
            "2018 and 2024, with each governorate's answers applied to that governorate's "
            "census population. Nobody counted this, and the dots are drawn desaturated to "
            "say so. "
            "**Shia are 45.2% of the country as drawn and Sunnis 30.6%, and the two sit "
            "where everyone who knows Iraq would put them.** Najaf reads **92.8%** Shia, "
            "then Karbala at 89.2% and Dhi Qar at 88.5%, and the whole mid-Euphrates and "
            "south runs above two thirds. Erbil is **97.0%** Sunni, Sulaymaniyah 93.1% and "
            "Duhok 92.5%, which is Iraqi Kurdistan, and Anbar is 62.6%. Baghdad is the one "
            "place where the two are drawn together in quantity: 58.9% Shia against 14.2% "
            "Sunni, on **5.76 million** and **1.39 million** people. Nobody told the survey "
            "any of this; it is what 6,224 Iraqis said their branch was. "
            "**Nearly a quarter of the country is drawn as Muslim with no branch at all, and "
            "that is what people answered.** The questionnaire offers Sunni and Shia beside a "
            "box reading `just a Muslim`, **23.7%** of Iraqis here take that box or decline "
            "the question after saying Muslim, and none of them is shared out between the two "
            "branches. They are **10.9 million** people on the plain Islam colour. "
            "**Where they are is the interesting part.** They are 45.9% of Diyala, 40.1% of "
            "Salah al-Din, 39.6% of Nineveh, 33.2% of Kirkuk and 30.5% of Anbar, against 2.6% "
            "in Erbil and Duhok and 7.0% in Najaf. Not naming a sect is a Baghdad and "
            "mixed-belt answer, and it is rare in the places where one branch holds almost "
            "everybody. So the Sunni share drawn in Baghdad, Diyala, Kirkuk and Nineveh is a "
            "floor rather than an estimate. The share also moves with the round, from **17.7%** "
            "in 2018 and 2019 to **42.7%** in the spring of 2021, which is a fact about "
            "fieldwork rather than about Iraq and is the reason nothing here is built on its "
            "level. "
            "**Christians are 0.31% of the map, about 144,000 people, and they are drawn "
            "everywhere instead of where they are.** That figure is close to the 150,000 "
            "usually given for what remains of a community estimated at 1.5 million in 2003, "
            "but it rests on **25 interviews** in 8,335, which is far too few to say which "
            "governorate anyone is in. They are therefore spread at the national rate rather "
            "than drawn in Nineveh, the Nineveh Plains and Baghdad, where the churches "
            "actually are. The same is true of the **135,000** on Other religion, which is "
            "where Iraq's Yazidis, Sabean-Mandaeans and Kaka'i sit: the survey returned 27 of "
            "them, and the Yazidis alone are usually put at 400,000 to 500,000, so that "
            "number is a floor of roughly a third. A household survey does not reach a "
            "population that has been in camps since 2014."),
        how="survey, four rounds 2018 to 2024 pooled",
        grain="governorates, 2.56 million people on average",
        counts=_iq_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "iq" / "iq_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_iq_place_weight,
        note="THE SECT ITEM IS THE POINT AND IT IS THE SAME COLUMN §11af REJECTED FOR THE "
             "MAGHREB. `Q1012A` asks a Muslim's denomination, and across Morocco, Algeria, "
             "Tunisia, Libya and Sudan 44.9 to 81.8% answer `just a Muslim` and wholly-Maliki "
             "Morocco returns 16.2% Maliki, so there it records a label rather than a school. "
             "§11af left open whether it behaves differently in Iraq and Lebanon, where sect "
             "is a salient public identity. In Iraq it does, on three measurements: the "
             "undifferentiated share is 24.5% rather than 45 to 82%; 60.9% of those who name "
             "a branch say Shia, against every published estimate's 61 to 64% of the whole "
             "population; and the geography reproduces Iraq's known sectarian map, which "
             "sources/iq.py asserts rather than admires. "
             "TWO WAVES ARE LEFT OUT ON THE CARD RATHER THAN ON THE ANSWERS. The files offer "
             "Iraq a religion answer in six waves; waves II (2011) and III (2013) have no "
             "usable sect follow-up, II having no `q1012a` column at all and III's being "
             "empty for every Iraqi in it. Pooling them would put 2,449 Iraqi Muslims into "
             "the undifferentiated bucket because of the questionnaire they were handed, and "
             "that bucket is the quantity most sensitive to the instrument. They are named in "
             "`omit=` with the reason. "
             "THE MADHHAB ANSWERS ARE FOLDED UP INTO THEIR BRANCH AND branches.py HAS THE "
             "NODES FOR THEM. 105 Shafi'i, 19 Hanbali and 2 Maliki go to Sunni; 221 Ja'fari "
             "and 2 Alawi go to Shia. The tree gained islam.sunni.shafii and "
             "islam.shia.jaafari with Turkiye, so this is a choice: Shafi'i runs 0.1%, 0.2%, "
             "1.7% and 2.6% across the four waves and Ja'fari 1.6% to 4.2%, which is the "
             "fieldwork rather than the country, and Turkiye's schools come from a state "
             "survey that asks the madhhab outright without Sunni and Shia on the same card. "
             "Left unfolded, Iraq's Shafi'is would fail the split-half on five early-wave "
             "respondents and be spread across Basra. "
             "THE GOVERNORATE CODE MEANS THE SAME THING IN ALL FOUR WAVES, WHICH IS UNIQUE IN "
             "THIS SURVEY. Jordan's means three different things across nine waves; Iraq's is "
             "70000 plus a governorate number in V, VII and VIII and 7000 plus the same "
             "number in VI-3. It is still not the pooling key, and it checks the harmonised "
             "names instead: 8,335 respondents, zero disagreements. Three labels need reading "
             "rather than transliterating, and `Diwaniyah` is the one that matters, because "
             "it is Al-Qadisiyyah under its capital's name and no spelling of Qadisiyah "
             "reaches it. "
             "THE HELD-OUT DECODE IS PINNED at r = +0.985 between the survey's governorate "
             "shares of respondents and the census's governorate populations, which none of "
             "20,000 random pairings reaches. LEBANON'S QUOTA CHECK WAS RUN HERE and Iraq is "
             "clean: no wave pair returns an identical composition in any free cell, "
             "Bonferroni p = 1 against a bar of 0.001, where Lebanon comes in at 1.4e-4. "
             "THE BOUNDARIES ARE COD-AB AND NOT geoBoundaries, WHICH IS UNUSUAL HERE AND "
             "DELIBERATE. gbOpen's IRQ/ADM1 draws Baghdad at 912 km2 against Iraq's own "
             "published 4,555 and COD-AB's 5,100, handing the rest of the governorate to "
             "Babil, Diyala and Salah al-Din. Baghdad is 21.2% of the country's population, "
             "so that would have crammed a fifth of Iraq's dots into a fifth of the right "
             "polygon and drawn the overflow in three neighbours. "
             "HALABJA IS IRAQ'S NINETEENTH GOVERNORATE SINCE 2014 AND IS NOT DRAWN "
             "SEPARATELY, because none of the three inputs carries it: COD-AB's ADM1 is "
             "eighteen features, the 2024 census tabulates eighteen, and the survey's `Q1` "
             "offers eighteen. Its people are inside Sulaymaniyah's 2,401,724.",
    ),
}
